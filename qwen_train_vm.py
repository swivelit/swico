#!/usr/bin/env python3
"""Advanced CPU-only LoRA SFT pipeline for Qwen/Qwen3-0.6B.

The E5 retriever and Qwen generator are intentionally trained separately.
This trainer consumes the conversational CSV format:

    conversation_id,turn_index,role,content,language

Key properties:
- conversation-level train/validation/test split (no turn leakage)
- strict role/order validation and exact-conversation deduplication
- Qwen chat-template rendering with thinking disabled for normal chat SFT
- assistant-only causal-LM labels with a robust chat-template-mask fallback
- LoRA/PEFT training instead of full-weight fine-tuning
- CPU BF16 when supported, gradient accumulation/checkpointing controls
- validation-driven early stopping and best-checkpoint restoration
- timestamped resumable runs and compatibility fingerprints
- memory/disk preflight checks plus a runtime memory guard
- final adapter export, optional merged model export, held-out loss/perplexity
- deterministic sample generation report from held-out conversations

Examples:
    ./run_qwen_training.sh --print-config
    SWICO_QWEN_PROFILE=smoke ./run_qwen_training.sh
    nohup ./run_qwen_training.sh > qwen-launcher.log 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from training_config import (
    ConfigError,
    env_bool,
    env_float,
    env_int,
    env_optional_float,
    env_optional_int,
    env_optional_text,
    env_text,
    initialize_environment,
)

LOADED_ENV_FILE = initialize_environment()

# Configure CPU threading before torch is imported.
_DEFAULT_THREADS = max(1, min(8, os.cpu_count() or 1))
_REQUESTED_THREADS = env_int("SWICO_QWEN_CPU_THREADS", env_int("SWICO_CPU_THREADS", _DEFAULT_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(_REQUESTED_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_REQUESTED_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import pandas as pd
import psutil
import torch
try:
    from datasets import Dataset
except ImportError:  # Allows config/CSV audits to run before the full VM install.
    Dataset = None  # type: ignore[assignment,misc]
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # Training-only dependency; print-config and prepare-only remain usable.
    LoraConfig = PeftModel = get_peft_model = None  # type: ignore[assignment]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


SCRIPT_VERSION = "1.1.0-qwen3-cpu-lora-hardening"
BASE_MODEL = "Qwen/Qwen3-0.6B"
REQUIRED_COLUMNS = ("conversation_id", "turn_index", "role", "content", "language")
VALID_ROLES = {"system", "user", "assistant"}
ALL_CONVERSATIONS = -1
CANONICAL_LANGUAGES = ("en", "ta", "tanglish", "ta-en")
LANGUAGE_BUCKETS = ("english", "tamil", "tanglish", "tamil-english-mixed")
LANGUAGE_DEFINITIONS = {
    "en": "English",
    "ta": "Tamil written primarily in Tamil script",
    "tanglish": "Tamil expressed using English/Latin letters, with natural English mixing where appropriate",
    "ta-en": "Natural Tamil-English mixed text using Tamil script plus English where appropriate, matching the user's language style",
}
CANONICAL_SYSTEM_PROMPTS = {
    "en": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer in English.",
    "ta": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer in Tamil.",
    "tanglish": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer in tanglish.",
    "ta-en": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer using a natural mix of Tamil script and English that matches the user's language style.",
}
DEFAULT_DATA_QUALITY_REPETITION_THRESHOLD = 0.30
DEFAULT_GENERATION_HEALTH_THRESHOLDS = {
    "min_termination_rate": 0.95,
    "max_max_token_hit_rate": 0.05,
    "max_repeated_4gram_ratio": 0.20,
    "require_script_adherence": True,
}


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    max_conversations: int | None
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    eval_generation_samples: int
    eval_steps: int
    save_steps: int


PROFILES: dict[str, Profile] = {
    "smoke": Profile("smoke", 64, 1.0, 1, 4, 512, 4, 1, 1),
    "vm": Profile("vm", 12_000, 2.0, 1, 16, 512, 40, 250, 250),
    "full": Profile("full", None, 3.0, 1, 16, 768, 40, 250, 250),
}


class ResourceStopRequested(RuntimeError):
    """Raised after Trainer has been asked to save/stop due to memory pressure."""


class QwenResourceGuardCallback(TrainerCallback):
    def __init__(
        self,
        output_path: Path,
        logger: logging.Logger,
        interval_steps: int,
        emergency_available_memory_gib: float,
        max_process_rss_gib: float | None,
    ) -> None:
        self.output_path = output_path
        self.logger = logger
        self.interval_steps = max(1, interval_steps)
        self.emergency_available_memory_gib = emergency_available_memory_gib
        self.max_process_rss_gib = max_process_rss_gib
        self.process = psutil.Process(os.getpid())
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if state.global_step <= 0 or state.global_step % self.interval_steps:
            return control
        vm = psutil.virtual_memory()
        rss_gib = self.process.memory_info().rss / (1024**3)
        available_gib = vm.available / (1024**3)
        payload = {
            "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "global_step": int(state.global_step),
            "available_memory_gib": round(available_gib, 3),
            "process_rss_gib": round(rss_gib, 3),
            "emergency_available_memory_gib": self.emergency_available_memory_gib,
            "max_process_rss_gib": self.max_process_rss_gib,
        }
        atomic_write_json(self.output_path, payload)
        too_low = available_gib < self.emergency_available_memory_gib
        too_large = self.max_process_rss_gib is not None and rss_gib > self.max_process_rss_gib
        if too_low or too_large:
            self.triggered = True
            reason = "available RAM below emergency threshold" if too_low else "process RSS above configured limit"
            self.logger.error("Memory guard requested checkpoint + stop: %s", reason)
            control.should_save = True
            control.should_training_stop = True
        return control


class CausalLMCollator:
    """Pad variable-length input_ids/labels while preserving -100 label masks."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for feature in features:
            ids = list(feature["input_ids"])
            labs = list(feature["labels"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def stable_hash(text: str, seed: int = 0) -> str:
    return hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sanitize_label(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned[:48] or None


def _path_from_env(name: str, default: str | None) -> Path | None:
    value = env_optional_text(name)
    if value is None:
        return Path(default) if default is not None else None
    return Path(value)


def _split_csv_text(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated item")
    return items


def parse_conversation_cap(value: str | int | None) -> int | None:
    """Parse ``auto``, a positive cap, or the explicit ``all`` sentinel."""
    if value is None:
        return None
    if isinstance(value, int):
        if value == ALL_CONVERSATIONS or value > 0:
            return value
        raise argparse.ArgumentTypeError("conversation cap must be a positive integer, auto, or all")
    text = str(value).strip().lower()
    if text in {"", "auto", "default", "none", "null"}:
        return None
    if text == "all":
        return ALL_CONVERSATIONS
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "conversation cap must be a positive integer, auto, or all"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("conversation cap must be a positive integer, auto, or all")
    return parsed


def conversation_cap_source(value: int | None) -> str:
    if value == ALL_CONVERSATIONS:
        return "explicit all"
    return "explicit integer" if value is not None else "profile default"


def conversation_cap_request(value: int | None) -> str | int:
    return "all" if value == ALL_CONVERSATIONS else (value if value is not None else "auto")


def quality_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metric": "word-level repeated 4-gram ratio",
        "severe_repetition_threshold": args.data_quality_repetition_threshold,
        "action": "exclude conversation when any assistant response exceeds threshold",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only LoRA SFT for Qwen/Qwen3-0.6B conversational training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=LOADED_ENV_FILE or Path("qwen_training.env"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default=env_text("SWICO_QWEN_PROFILE", "vm"))
    parser.add_argument("--data", type=Path, default=_path_from_env("SWICO_QWEN_DATA_PATH", "data/qwen_dataset - qwen_dataset.csv"))
    parser.add_argument("--base-model", default=env_text("SWICO_QWEN_BASE_MODEL", BASE_MODEL))
    parser.add_argument("--output-root", type=Path, default=Path(env_text("SWICO_QWEN_OUTPUT_ROOT", "training_artifacts/qwen3-0.6b-swico")))
    parser.add_argument("--output", type=Path, default=_path_from_env("SWICO_QWEN_OUTPUT_DIR", None))
    parser.add_argument("--run-mode", choices=("auto", "new", "resume-latest"), default=env_text("SWICO_QWEN_RUN_MODE", "auto"))
    parser.add_argument("--run-label", default=env_optional_text("SWICO_QWEN_RUN_LABEL"))
    parser.add_argument("--create-latest-links", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_CREATE_LATEST_LINKS", True))

    parser.add_argument("--threads", type=int, default=env_int("SWICO_QWEN_CPU_THREADS", _REQUESTED_THREADS))
    parser.add_argument("--seed", type=int, default=env_int("SWICO_QWEN_SEED", 42))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None if env_optional_text("SWICO_QWEN_BF16") is None else env_bool("SWICO_QWEN_BF16", False))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_OFFLINE", False))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_RESUME", True))
    parser.add_argument("--prepare-only", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_PREPARE_ONLY", False))

    try:
        env_conversation_cap = parse_conversation_cap(os.environ.get("SWICO_QWEN_MAX_CONVERSATIONS"))
    except argparse.ArgumentTypeError as exc:
        raise ConfigError(f"SWICO_QWEN_MAX_CONVERSATIONS: {exc}") from exc
    parser.add_argument(
        "--max-conversations",
        type=parse_conversation_cap,
        default=env_conversation_cap,
        help="auto/profile cap, a positive integer, or all",
    )
    parser.add_argument("--train-split-fraction", type=float, default=env_float("SWICO_QWEN_TRAIN_SPLIT_FRACTION", 0.80))
    parser.add_argument("--validation-split-fraction", type=float, default=env_float("SWICO_QWEN_VALIDATION_SPLIT_FRACTION", 0.10))
    parser.add_argument("--test-split-fraction", type=float, default=env_float("SWICO_QWEN_TEST_SPLIT_FRACTION", 0.10))
    parser.add_argument("--max-seq-length", type=int, default=env_optional_int("SWICO_QWEN_MAX_SEQ_LENGTH"))

    parser.add_argument("--epochs", type=float, default=env_optional_float("SWICO_QWEN_EPOCHS"))
    parser.add_argument("--batch-size", type=int, default=env_optional_int("SWICO_QWEN_BATCH_SIZE"))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("SWICO_QWEN_EVAL_BATCH_SIZE", 1))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=env_optional_int("SWICO_QWEN_GRADIENT_ACCUMULATION_STEPS"))
    parser.add_argument("--learning-rate", type=float, default=env_float("SWICO_QWEN_LEARNING_RATE", 5.0e-5))
    parser.add_argument("--weight-decay", type=float, default=env_float("SWICO_QWEN_WEIGHT_DECAY", 0.01))
    parser.add_argument("--warmup-ratio", type=float, default=env_float("SWICO_QWEN_WARMUP_RATIO", 0.05))
    parser.add_argument("--lr-scheduler-type", default=env_text("SWICO_QWEN_LR_SCHEDULER_TYPE", "cosine"))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("SWICO_QWEN_MAX_GRAD_NORM", 1.0))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_GRADIENT_CHECKPOINTING", True))
    parser.add_argument("--logging-steps", type=int, default=env_int("SWICO_QWEN_LOGGING_STEPS", 5))
    parser.add_argument("--save-total-limit", type=int, default=env_int("SWICO_QWEN_SAVE_TOTAL_LIMIT", 2))
    parser.add_argument("--eval-steps", type=int, default=env_optional_int("SWICO_QWEN_EVAL_STEPS"))
    parser.add_argument("--save-steps", type=int, default=env_optional_int("SWICO_QWEN_SAVE_STEPS"))
    parser.add_argument("--dataloader-num-workers", type=int, default=env_int("SWICO_QWEN_DATALOADER_NUM_WORKERS", 0))

    parser.add_argument("--lora-r", type=int, default=env_int("SWICO_QWEN_LORA_R", 8))
    parser.add_argument("--lora-alpha", type=int, default=env_int("SWICO_QWEN_LORA_ALPHA", 16))
    parser.add_argument("--lora-dropout", type=float, default=env_float("SWICO_QWEN_LORA_DROPOUT", 0.05))
    parser.add_argument(
        "--lora-target-modules",
        type=_split_csv_text,
        default=_split_csv_text(env_text("SWICO_QWEN_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")),
    )

    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_EARLY_STOPPING", True))
    parser.add_argument("--early-stopping-patience", type=int, default=env_int("SWICO_QWEN_EARLY_STOPPING_PATIENCE", 3))
    parser.add_argument("--early-stopping-threshold", type=float, default=env_float("SWICO_QWEN_EARLY_STOPPING_THRESHOLD", 0.001))
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_LOAD_BEST_MODEL_AT_END", True))
    parser.add_argument("--merge-adapter", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_MERGE_ADAPTER", False))
    parser.add_argument("--eval-generation-samples", type=int, default=env_optional_int("SWICO_QWEN_EVAL_GENERATION_SAMPLES"))
    parser.add_argument("--generation-max-new-tokens", type=int, default=env_int("SWICO_QWEN_GENERATION_MAX_NEW_TOKENS", 256))
    parser.add_argument("--data-quality-repetition-threshold", type=float, default=env_float("SWICO_QWEN_DATA_QUALITY_REPETITION_THRESHOLD", DEFAULT_DATA_QUALITY_REPETITION_THRESHOLD))
    parser.add_argument("--audit-tokenization", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_AUDIT_TOKENIZATION", False))
    parser.add_argument("--min-language-termination-rate", type=float, default=env_float("SWICO_QWEN_HEALTH_MIN_TERMINATION_RATE", DEFAULT_GENERATION_HEALTH_THRESHOLDS["min_termination_rate"]))
    parser.add_argument("--max-language-max-token-hit-rate", type=float, default=env_float("SWICO_QWEN_HEALTH_MAX_TOKEN_HIT_RATE", DEFAULT_GENERATION_HEALTH_THRESHOLDS["max_max_token_hit_rate"]))
    parser.add_argument("--max-language-repeated-4gram-ratio", type=float, default=env_float("SWICO_QWEN_HEALTH_MAX_REPEATED_4GRAM_RATIO", DEFAULT_GENERATION_HEALTH_THRESHOLDS["max_repeated_4gram_ratio"]))
    parser.add_argument("--require-language-script-adherence", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_HEALTH_REQUIRE_SCRIPT_ADHERENCE", True))

    parser.add_argument("--memory-guard", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_MEMORY_GUARD", True))
    parser.add_argument("--memory-guard-interval-steps", type=int, default=env_int("SWICO_QWEN_MEMORY_GUARD_INTERVAL_STEPS", 5))
    parser.add_argument("--emergency-available-memory-gib", type=float, default=env_float("SWICO_QWEN_EMERGENCY_AVAILABLE_MEMORY_GIB", 4.0))
    parser.add_argument("--max-process-rss-gib", type=float, default=env_optional_float("SWICO_QWEN_MAX_PROCESS_RSS_GIB"))
    parser.add_argument("--min-available-memory-gib", type=float, default=env_float("SWICO_QWEN_MIN_AVAILABLE_MEMORY_GIB", 8.0))
    parser.add_argument("--min-free-disk-gib", type=float, default=env_float("SWICO_QWEN_MIN_FREE_DISK_GIB", 4.0))
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def resolve_profile(args: argparse.Namespace) -> Profile:
    base = PROFILES[args.profile]
    return dataclasses.replace(
        base,
        max_conversations=(
            None if args.max_conversations == ALL_CONVERSATIONS
            else args.max_conversations if args.max_conversations is not None else base.max_conversations
        ),
        epochs=args.epochs if args.epochs is not None else base.epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else base.gradient_accumulation_steps
        ),
        max_seq_length=args.max_seq_length if args.max_seq_length is not None else base.max_seq_length,
        eval_generation_samples=(
            args.eval_generation_samples
            if args.eval_generation_samples is not None
            else base.eval_generation_samples
        ),
        eval_steps=args.eval_steps if args.eval_steps is not None else base.eval_steps,
        save_steps=args.save_steps if args.save_steps is not None else base.save_steps,
    )


def validate_config(args: argparse.Namespace, profile: Profile) -> None:
    if args.data is None:
        raise ConfigError("SWICO_QWEN_DATA_PATH or --data is required")
    if profile.epochs <= 0 or profile.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ConfigError("epochs and batch sizes must be positive")
    if profile.gradient_accumulation_steps <= 0:
        raise ConfigError("gradient accumulation steps must be positive")
    if profile.eval_steps <= 0 or profile.save_steps <= 0:
        raise ConfigError("evaluation and save steps must be positive")
    if args.load_best_model_at_end and profile.save_steps % profile.eval_steps:
        raise ConfigError("save steps must be a multiple of eval steps when loading the best model")
    if profile.max_seq_length < 64:
        raise ConfigError("max sequence length must be at least 64")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ConfigError("learning rate and max grad norm must be positive")
    if args.generation_max_new_tokens <= 0:
        raise ConfigError("generation max new tokens must be positive")
    if args.data_quality_repetition_threshold <= 0 or args.data_quality_repetition_threshold > 1:
        raise ConfigError("data quality repetition threshold must be in (0, 1]")
    if not 0 <= args.min_language_termination_rate <= 1:
        raise ConfigError("minimum language termination rate must be in [0, 1]")
    if not 0 <= args.max_language_max_token_hit_rate <= 1:
        raise ConfigError("maximum language max-token-hit rate must be in [0, 1]")
    if not 0 <= args.max_language_repeated_4gram_ratio <= 1:
        raise ConfigError("maximum language repeated-4gram ratio must be in [0, 1]")
    if not 0 <= args.lora_dropout < 1:
        raise ConfigError("LoRA dropout must be in [0, 1)")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ConfigError("LoRA rank and alpha must be positive")
    fractions = (args.train_split_fraction, args.validation_split_fraction, args.test_split_fraction)
    if any(value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ConfigError("train/validation/test fractions must be positive and sum to 1.0")
    if args.early_stopping_patience < 1 or args.early_stopping_threshold < 0:
        raise ConfigError("invalid early stopping configuration")
    if args.min_available_memory_gib <= 0 or args.min_free_disk_gib <= 0:
        raise ConfigError("preflight memory/disk limits must be positive")


def qwen_effective_config(args: argparse.Namespace, profile: Profile) -> dict[str, Any]:
    profile_default = PROFILES[args.profile]
    return {
        "script_version": SCRIPT_VERSION,
        "profile": dataclasses.asdict(profile),
        "data": str(args.data),
        "base_model": args.base_model,
        "output_root": str(args.output_root),
        "output": str(args.output) if args.output else None,
        "run_mode": args.run_mode,
        "run_label": args.run_label,
        "threads": args.threads,
        "seed": args.seed,
        "bf16": args.bf16,
        "offline": args.offline,
        "splits": [args.train_split_fraction, args.validation_split_fraction, args.test_split_fraction],
        "max_conversations": {
            "requested_override": conversation_cap_request(args.max_conversations),
            "profile_default": profile_default.max_conversations,
            "effective": profile.max_conversations,
            "source": conversation_cap_source(args.max_conversations),
        },
        "max_seq_length": profile.max_seq_length,
        "epochs": profile.epochs,
        "batch_size": profile.batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "eval_batch_size": args.eval_batch_size,
        "eval_steps": profile.eval_steps,
        "save_steps": profile.save_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(args.lora_target_modules),
        },
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "load_best_model_at_end": args.load_best_model_at_end,
        "merge_adapter": args.merge_adapter,
        "generation_sample_count": profile.eval_generation_samples,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "generation_health_thresholds": {
            "min_language_termination_rate": args.min_language_termination_rate,
            "max_language_max_token_hit_rate": args.max_language_max_token_hit_rate,
            "max_language_repeated_4gram_ratio": args.max_language_repeated_4gram_ratio,
            "require_language_script_adherence": args.require_language_script_adherence,
        },
        "data_quality": {
            "repetition_metric": "word-level repeated 4-gram ratio",
            "severe_repetition_threshold": args.data_quality_repetition_threshold,
        },
        "language_validation": {
            "canonical_languages": list(CANONICAL_LANGUAGES),
            "definitions": LANGUAGE_DEFINITIONS,
            "system_prompt_normalization": "derive from coherent non-system language and replace system content deterministically",
        },
        "audit_tokenization": args.audit_tokenization,
        "memory_guard": args.memory_guard,
        "emergency_available_memory_gib": args.emergency_available_memory_gib,
        "max_process_rss_gib": args.max_process_rss_gib,
    }


def training_fingerprint(config: dict[str, Any], data_hash: str | None = None) -> str:
    material = dict(config)
    material.pop("output", None)
    material.pop("run_mode", None)
    material.pop("run_label", None)
    # Generation/reporting controls are post-training configuration.  They
    # must not make a compatible interrupted training checkpoint unresumable.
    material.pop("generation_max_new_tokens", None)
    material.pop("generation_sample_count", None)
    material.pop("generation_health_thresholds", None)
    material.pop("merge_adapter", None)
    material.pop("audit_tokenization", None)
    if isinstance(material.get("profile"), dict):
        material["profile"] = dict(material["profile"])
        material["profile"].pop("eval_generation_samples", None)
    if data_hash:
        material["data_sha256"] = data_hash
    return stable_hash(json.dumps(material, sort_keys=True, ensure_ascii=False))


def is_completed_run(path: Path) -> bool:
    state = path / "run_state.json"
    if not state.exists():
        return False
    try:
        return bool(json.loads(state.read_text(encoding="utf-8")).get("final_evaluation_complete"))
    except Exception:
        return False


def newest_compatible_incomplete_run(root: Path, fingerprint: str) -> Path | None:
    runs = root / "runs"
    if not runs.exists():
        return None
    candidates: list[Path] = []
    for path in runs.iterdir():
        if not path.is_dir() or is_completed_run(path):
            continue
        manifest = path / "run_manifest.json"
        if not manifest.exists():
            continue
        with contextlib.suppress(Exception):
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("invocation_fingerprint") == fingerprint:
                candidates.append(path)
    return max(candidates, key=lambda p: p.name) if candidates else None


def allocate_run(args: argparse.Namespace, profile: Profile, fingerprint: str) -> tuple[Path, bool]:
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        return output, bool(args.resume and get_last_checkpoint(str(output / "trainer")))

    root = args.output_root.resolve()
    (root / "runs").mkdir(parents=True, exist_ok=True)
    if args.run_mode in {"auto", "resume-latest"} and args.resume:
        compatible = newest_compatible_incomplete_run(root, fingerprint)
        if compatible is not None:
            return compatible, True
        if args.run_mode == "resume-latest":
            raise RuntimeError("No compatible incomplete Qwen run exists to resume")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = sanitize_label(args.run_label)
    stem = f"{stamp}_{profile.name}" + (f"_{label}" if label else "")
    output = root / "runs" / stem
    counter = 1
    while output.exists():
        output = root / "runs" / f"{stem}_{counter:02d}"
        counter += 1
    output.mkdir(parents=True)
    return output, False


def replace_pointer(root: Path, name: str, target: Path) -> None:
    pointer = root / name
    with contextlib.suppress(FileNotFoundError):
        if pointer.is_symlink() or pointer.is_file():
            pointer.unlink()
    with contextlib.suppress(OSError):
        if pointer.is_dir() and not pointer.is_symlink():
            shutil.rmtree(pointer)
    try:
        pointer.symlink_to(target, target_is_directory=True)
    except OSError:
        (root / f"{name}.txt").write_text(str(target) + "\n", encoding="utf-8")


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("swico.qwen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def preflight(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(args.output_root.resolve().parent)
    available_gib = vm.available / (1024**3)
    free_disk_gib = disk.free / (1024**3)
    if available_gib < args.min_available_memory_gib:
        raise RuntimeError(
            f"Only {available_gib:.2f} GiB RAM available; require at least {args.min_available_memory_gib:.2f} GiB"
        )
    if free_disk_gib < args.min_free_disk_gib:
        raise RuntimeError(
            f"Only {free_disk_gib:.2f} GiB disk free; require at least {args.min_free_disk_gib:.2f} GiB"
        )
    torch.set_num_threads(args.threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    bf16_supported = False
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16)
        _ = a @ a
        bf16_supported = True
    except Exception:
        pass
    use_bf16 = bf16_supported if args.bf16 is None else bool(args.bf16 and bf16_supported)
    payload = {
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "available_memory_gib": round(available_gib, 3),
        "total_memory_gib": round(vm.total / (1024**3), 3),
        "free_disk_gib": round(free_disk_gib, 3),
        "cpu_bf16_supported": bf16_supported,
        "bf16_enabled": use_bf16,
        "torch_version": torch.__version__,
    }
    logger.info("CPU runtime configured: threads=%d logical_cpus=%s", args.threads, os.cpu_count())
    logger.info("CPU BF16 autocast: %s", "enabled" if use_bf16 else "disabled")
    return payload


def normalize_content(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def canonical_language(value: Any) -> str:
    """Return the one canonical non-system language code used by Qwen."""
    aliases = {
        "en": "en",
        "english": "en",
        "ta": "ta",
        "tamil": "ta",
        "tanglish": "tanglish",
        "ta-en": "ta-en",
        "ta_en": "ta-en",
        "tamil-english-mixed": "ta-en",
    }
    normalized = normalize_content(value).lower()
    canonical = aliases.get(normalized)
    if canonical not in CANONICAL_LANGUAGES:
        raise ValueError(
            f"Unsupported Qwen language {value!r}; expected one of {', '.join(CANONICAL_LANGUAGES)}"
        )
    return canonical


def has_tamil_script(text: str) -> bool:
    return any("\u0b80" <= character <= "\u0bff" for character in text)


def has_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def validate_language_style(language: str, messages: list[dict[str, str]], *, context: str = "conversation") -> None:
    """Validate broad script/style expectations without rejecting acronyms or punctuation."""
    text = "\n".join(message["content"] for message in messages if message["role"] != "system")
    tamil = has_tamil_script(text)
    latin = has_latin_letters(text)
    valid = {
        "en": not tamil,
        "ta": tamil,
        "tanglish": not tamil,
        "ta-en": tamil and latin,
    }[language]
    if not valid:
        raise ValueError(
            f"{context} language/style mismatch for {language}: "
            f"tamil_script={tamil}, latin_letters={latin}"
        )


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.lower(), flags=re.UNICODE)


def repeated_4gram_ratio(text: str) -> float:
    words = word_tokens(text)
    grams = [tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def validate_and_group_conversations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Qwen dataset is missing required columns: {sorted(missing)}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["conversation_id"] = frame["conversation_id"].map(normalize_content)
    frame["role"] = frame["role"].map(lambda value: normalize_content(value).lower())
    # Content is carried through verbatim.  Only the emptiness check below
    # trims a temporary view; user/assistant answer text is never rewritten.
    frame["content"] = frame["content"].map(lambda value: str(value))
    frame["language"] = frame["language"].map(normalize_content)
    try:
        frame["turn_index"] = frame["turn_index"].astype(int)
    except Exception as exc:
        raise ValueError("turn_index must contain integers") from exc
    frame = frame[(frame["conversation_id"] != "") & (frame["content"].map(lambda value: value.strip()) != "")].copy()
    if frame.empty:
        raise ValueError("Qwen dataset contains no non-empty messages")
    invalid_roles = sorted(set(frame["role"]) - VALID_ROLES)
    if invalid_roles:
        raise ValueError(f"Unsupported roles: {invalid_roles}")

    conversations: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for conversation_id, group in frame.groupby("conversation_id", sort=False):
        group = group.sort_values("turn_index", kind="stable")
        indices = group["turn_index"].tolist()
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate turn_index values in conversation {conversation_id}")
        roles = group["role"].tolist()
        if "assistant" not in roles or "user" not in roles:
            raise ValueError(f"Conversation {conversation_id} must contain at least one user and assistant message")
        if any(role == "system" for role in roles[1:]):
            raise ValueError(f"Conversation {conversation_id}: system message is only allowed first")
        non_system_languages = {
            canonical_language(row.language)
            for row in group.itertuples(index=False)
            if str(row.role) != "system"
        }
        if len(non_system_languages) != 1:
            raise ValueError(
                f"Conversation {conversation_id} must have one coherent non-system language; "
                f"found {sorted(non_system_languages)}"
            )
        language = next(iter(non_system_languages))
        messages = [
            {"role": str(row.role), "content": str(row.content)}
            for row in group.itertuples(index=False)
        ]
        validate_language_style(language, messages, context=f"conversation {conversation_id}")
        normalized_system_prompt_count = 0
        for message in messages:
            if message["role"] == "system":
                canonical_prompt = CANONICAL_SYSTEM_PROMPTS[language]
                if message["content"] != canonical_prompt:
                    normalized_system_prompt_count += 1
                message["content"] = canonical_prompt
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        digest = stable_hash(serialized)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        conversations.append(
            {
                "conversation_id": str(conversation_id),
                "messages": messages,
                "language": language,
                "digest": digest,
                "system_prompt_normalized_count": normalized_system_prompt_count,
            }
        )
    if len(conversations) < 3:
        raise ValueError("At least 3 unique conversations are required for train/validation/test splitting")
    return conversations


def audit_conversation_quality(
    conversations: list[dict[str, Any]], threshold: float = DEFAULT_DATA_QUALITY_REPETITION_THRESHOLD
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exclude only conversations with clearly pathological assistant repetition."""
    if not 0 < threshold <= 1:
        raise ValueError("repetition threshold must be in (0, 1]")
    by_language = {
        language: {"conversations_seen": 0, "conversations_excluded": 0, "assistant_messages_seen": 0, "assistant_messages_excluded": 0}
        for language in CANONICAL_LANGUAGES
    }
    suspicious_messages: list[dict[str, Any]] = []
    excluded_ids: set[str] = set()
    for conversation in conversations:
        language = conversation["language"]
        by_language[language]["conversations_seen"] += 1
        for index, message in enumerate(conversation["messages"]):
            if message["role"] != "assistant":
                continue
            by_language[language]["assistant_messages_seen"] += 1
            ratio = repeated_4gram_ratio(message["content"])
            if ratio > threshold:
                excluded_ids.add(conversation["conversation_id"])
                by_language[language]["assistant_messages_excluded"] += 1
                suspicious_messages.append(
                    {
                        "conversation_id": conversation["conversation_id"],
                        "assistant_message_index": index,
                        "language": language,
                        "repeated_4gram_ratio": round(ratio, 6),
                        "reason": "word-level repeated-4gram ratio above severe threshold",
                    }
                )
    filtered = []
    for conversation in conversations:
        if conversation["conversation_id"] in excluded_ids:
            by_language[conversation["language"]]["conversations_excluded"] += 1
        else:
            filtered.append(conversation)
    report = {
        "metric": "word-level repeated 4-gram ratio",
        "severe_repetition_threshold": threshold,
        "conversations_seen": len(conversations),
        "conversations_excluded": len(excluded_ids),
        "assistant_messages_flagged": len(suspicious_messages),
        "excluded_conversation_ids": sorted(excluded_ids),
        "by_language": by_language,
        "suspicious_assistant_messages": sorted(
            suspicious_messages,
            key=lambda item: (-item["repeated_4gram_ratio"], item["conversation_id"], item["assistant_message_index"]),
        ),
        "content_action": "pathological conversations excluded; user and assistant content was not rewritten",
    }
    return filtered, report


def split_conversations(
    conversations: list[dict[str, Any]],
    seed: int,
    fractions: tuple[float, float, float],
    max_conversations: int | None,
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(conversations, key=lambda item: stable_hash(item["digest"], seed))
    if max_conversations is not None:
        ordered = ordered[:max_conversations]
    n = len(ordered)
    if n < 3:
        raise ValueError("Profile cap leaves fewer than 3 conversations")
    train_n = max(1, int(round(n * fractions[0])))
    val_n = max(1, int(round(n * fractions[1])))
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1
    return {
        "train": ordered[:train_n],
        "validation": ordered[train_n : train_n + val_n],
        "test": ordered[train_n + val_n :],
    }


def prepare_dataset(
    data_path: Path,
    output: Path,
    profile: Profile,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Qwen dataset does not exist: {data_path}")
    frame = pd.read_csv(data_path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)
    conversations_before_quality = validate_and_group_conversations(frame)
    conversations, quality_report = audit_conversation_quality(
        conversations_before_quality,
        threshold=args.data_quality_repetition_threshold,
    )
    splits = split_conversations(
        conversations,
        seed=args.seed,
        fractions=(args.train_split_fraction, args.validation_split_fraction, args.test_split_fraction),
        max_conversations=profile.max_conversations,
    )
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    atomic_write_json(prepared / "data_quality.json", quality_report)
    for split_name, values in splits.items():
        with (prepared / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in values:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    ids = {name: {item["conversation_id"] for item in values} for name, values in splits.items()}
    overlaps = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_test": len(ids["train"] & ids["test"]),
        "validation_test": len(ids["validation"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Conversation leakage detected: {overlaps}")
    meta = {
        "source": str(data_path),
        "source_sha256": file_sha256(data_path),
        "raw_message_rows": int(len(frame)),
        "unique_conversations_after_dedup": len(conversations_before_quality),
        "conversations_after_quality_filter": len(conversations),
        "data_quality_excluded_conversations": quality_report["conversations_excluded"],
        "data_quality_excluded_assistant_messages": quality_report["assistant_messages_flagged"],
        "system_prompts_normalized": sum(
            int(item.get("system_prompt_normalized_count", 0)) for item in conversations_before_quality
        ),
        "system_prompt_normalization": {
            "version": SCRIPT_VERSION,
            "canonical_languages": list(CANONICAL_LANGUAGES),
            "normalized_system_prompt_rows": sum(
                int(item.get("system_prompt_normalized_count", 0)) for item in conversations_before_quality
            ),
            "normalized_conversations": sum(
                int(item.get("system_prompt_normalized_count", 0)) > 0 for item in conversations_before_quality
            ),
        },
        "max_conversations_cap_requested": conversation_cap_request(args.max_conversations),
        "max_conversations_cap_profile_default": PROFILES[args.profile].max_conversations,
        "max_conversations_cap_effective": profile.max_conversations,
        "max_conversations_cap_source": conversation_cap_source(args.max_conversations),
        "selected_conversations_before_split": sum(len(values) for values in splits.values()),
        "conversations_excluded_by_cap": max(
            0, len(conversations) - sum(len(values) for values in splits.values())
        ),
        "split_conversations": {name: len(values) for name, values in splits.items()},
        "conversation_id_overlap": overlaps,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(prepared / "metadata.json", meta)
    logger.info(
        "Prepared conversations: train=%d validation=%d test=%d",
        len(splits["train"]), len(splits["validation"]), len(splits["test"]),
    )
    if profile.max_conversations is not None and len(conversations) > profile.max_conversations:
        logger.info(
            "Conversation cap is active: dataset has %d quality-filtered conversations; profile selects %d",
            len(conversations),
            profile.max_conversations,
        )
    return splits, meta


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def apply_template(tokenizer, messages: list[dict[str, str]], *, tokenize: bool, **kwargs):
    """Apply Qwen's template with normal-chat thinking disabled when supported."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=tokenize, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)


def assistant_only_tokens(
    tokenizer,
    messages: list[dict[str, str]],
    max_seq_length: int,
    *,
    return_diagnostics: bool = False,
    allow_prompt_only: bool = False,
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], dict[str, object]]:
    """Label assistant content plus the Qwen end-of-message token.

    Current Transformers/Qwen templates expose native assistant generation
    spans.  Those spans preserve the active template's role headers and
    thinking wrapper without duplicating its formatting here.  Qwen's
    ``<|im_end|>`` is outside the generation span, so the tokenizer's EOS
    token is added after every assistant span.  An offset-based marker
    fallback is retained for older Transformers versions with no native mask;
    it never searches for a raw response string.
    """
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    assistant_count = sum(1 for message in messages if message["role"] == "assistant")
    if assistant_count == 0:
        raise ValueError("A conversation contains no assistant messages")

    def flatten(value) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            value = value[0]
        return [int(item) for item in value]

    termination_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple, set)):
        termination_ids.update(int(item) for item in eos_id if item is not None)
    elif eos_id is not None:
        termination_ids.add(int(eos_id))
    def label_termination(ids: list[int], labels: list[int], mask: list[int]) -> None:
        index = 0
        while index < len(ids):
            if not mask[index]:
                index += 1
                continue
            end = index
            while end + 1 < len(ids) and mask[end + 1]:
                end += 1
            lookahead = end + 1
            while lookahead < len(ids) and lookahead <= end + 8 and not mask[lookahead]:
                if ids[lookahead] in termination_ids:
                    labels[lookahead] = ids[lookahead]
                    break
                lookahead += 1
            index = end + 1

    native_ids: list[int] | None = None
    native_mask: list[int] | None = None
    try:
        native = apply_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        if hasattr(native, "keys") and "input_ids" in native:
            candidate_mask = native.get("assistant_masks")
            if candidate_mask is None:
                candidate_mask = native.get("assistant_tokens_mask")
            if candidate_mask is not None:
                candidate_mask_values = flatten(candidate_mask)
                if any(candidate_mask_values):
                    native_ids = flatten(native["input_ids"])
                    native_mask = candidate_mask_values
        elif isinstance(native, dict) and "input_ids" in native:
            candidate_mask = native.get("assistant_masks")
            if candidate_mask is None:
                candidate_mask = native.get("assistant_tokens_mask")
            if candidate_mask is not None:
                candidate_mask_values = flatten(candidate_mask)
                if any(candidate_mask_values):
                    native_ids = flatten(native["input_ids"])
                    native_mask = candidate_mask_values
    except (TypeError, ValueError, KeyError, AssertionError):
        native_ids = None
        native_mask = None

    if native_ids is not None and native_mask is not None:
        if len(native_ids) != len(native_mask):
            raise ValueError("Qwen tokenizer returned mismatched input_ids and assistant mask lengths")
        labels = [token_id if mask else -100 for token_id, mask in zip(native_ids, native_mask)]
        label_termination(native_ids, labels, native_mask)
        ids = native_ids
    else:
        marker_nonce = "SWICO_QWEN_ASSISTANT_BOUNDARY_7F3A9C"
        marked_messages: list[dict[str, str]] = []
        assistant_index = 0
        for message in messages:
            cloned = dict(message)
            if message["role"] == "assistant":
                start_marker = f"<{marker_nonce}_START_{assistant_index}>"
                end_marker = f"<{marker_nonce}_END_{assistant_index}>"
                content = str(message["content"])
                if start_marker in content or end_marker in content:
                    raise ValueError("Qwen assistant-boundary marker unexpectedly appears in training data")
                cloned["content"] = f"{start_marker}{content}{end_marker}"
                assistant_index += 1
            marked_messages.append(cloned)
        marked_rendered = str(apply_template(tokenizer, marked_messages, tokenize=False, add_generation_prompt=False))

        clean_parts: list[str] = []
        assistant_spans: list[tuple[int, int]] = []
        source_pos = 0
        clean_pos = 0
        for assistant_index in range(assistant_count):
            start_marker = f"<{marker_nonce}_START_{assistant_index}>"
            end_marker = f"<{marker_nonce}_END_{assistant_index}>"
            start_at = marked_rendered.find(start_marker, source_pos)
            if start_at < 0:
                raise ValueError(f"Qwen chat template did not preserve assistant start marker {assistant_index}")
            before = marked_rendered[source_pos:start_at]
            clean_parts.append(before)
            clean_pos += len(before)
            content_start = start_at + len(start_marker)
            end_at = marked_rendered.find(end_marker, content_start)
            if end_at < 0:
                raise ValueError(f"Qwen chat template did not preserve assistant end marker {assistant_index}")
            assistant_text = marked_rendered[content_start:end_at]
            span_start = clean_pos
            clean_parts.append(assistant_text)
            clean_pos += len(assistant_text)
            assistant_spans.append((span_start, clean_pos))
            source_pos = end_at + len(end_marker)
        clean_parts.append(marked_rendered[source_pos:])
        rendered = "".join(clean_parts)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = list(encoded["offset_mapping"])
        if len(ids) != len(offsets):
            raise ValueError("Tokenizer returned mismatched input_ids and offset_mapping lengths")
        fallback_mask: list[int] = []
        labels = []
        for token_id, offset in zip(ids, offsets):
            token_start, token_end = int(offset[0]), int(offset[1])
            is_assistant = token_end > token_start and any(
                token_start < span_end and token_end > span_start
                for span_start, span_end in assistant_spans
            )
            fallback_mask.append(1 if is_assistant else 0)
            labels.append(token_id if is_assistant else -100)
        label_termination(ids, labels, fallback_mask)

    pre_truncation_token_count = len(ids)
    pre_truncation_supervised_tokens = sum(label != -100 for label in labels)
    ids = ids[-max_seq_length:]
    labels = labels[-max_seq_length:]
    post_truncation_supervised_tokens = sum(label != -100 for label in labels)
    if not any(label != -100 for label in labels) and not allow_prompt_only:
        raise ValueError(
            "A conversation has no assistant content or termination tokens after tokenization/truncation; "
            "increase SWICO_QWEN_MAX_SEQ_LENGTH"
        )
    if return_diagnostics:
        return ids, labels, {
            "pre_truncation_token_count": pre_truncation_token_count,
            "post_truncation_token_count": len(ids),
            "truncated": pre_truncation_token_count > max_seq_length,
            "assistant_supervised_tokens_before_truncation": pre_truncation_supervised_tokens,
            "assistant_supervised_tokens_retained": post_truncation_supervised_tokens,
            "supervised_token_retention_ratio": (
                post_truncation_supervised_tokens / pre_truncation_supervised_tokens
                if pre_truncation_supervised_tokens
                else 0.0
            ),
        }
    return ids, labels


def _tokenization_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "conversation_count": 0,
            "pre_truncation_token_count": {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None},
            "post_truncation_token_count": {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None},
            "conversations_truncated": 0,
            "truncation_rate": None,
            "assistant_supervised_tokens_before_truncation": 0,
            "assistant_supervised_tokens_retained": 0,
            "supervised_token_retention_ratio": None,
        }

    def percentile(values: list[int], fraction: float) -> float:
        values = sorted(values)
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(values[lower])
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    def distribution(key: str) -> dict[str, float]:
        values = [int(item[key]) for item in records]
        return {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    before = sum(int(item["assistant_supervised_tokens_before_truncation"]) for item in records)
    retained = sum(int(item["assistant_supervised_tokens_retained"]) for item in records)
    return {
        "conversation_count": len(records),
        "pre_truncation_token_count": distribution("pre_truncation_token_count"),
        "post_truncation_token_count": distribution("post_truncation_token_count"),
        "conversations_truncated": sum(bool(item["truncated"]) for item in records),
        "truncation_rate": sum(bool(item["truncated"]) for item in records) / len(records),
        "assistant_supervised_tokens_before_truncation": before,
        "assistant_supervised_tokens_retained": retained,
        "supervised_token_retention_ratio": retained / before if before else None,
    }


def tokenize_split(
    tokenizer,
    rows: list[dict[str, Any]],
    max_seq_length: int,
    diagnostics: dict[str, list[dict[str, Any]]] | None = None,
    allow_prompt_only: bool = False,
) -> Dataset:
    if Dataset is None:
        raise RuntimeError("datasets is required for Qwen tokenization; install requirements-cpu.txt")
    payload = {"input_ids": [], "labels": []}
    for row in rows:
        result = assistant_only_tokens(
            tokenizer,
            row["messages"],
            max_seq_length,
            return_diagnostics=diagnostics is not None,
            allow_prompt_only=allow_prompt_only,
        )
        if diagnostics is None:
            ids, labels = result  # type: ignore[misc]
        else:
            ids, labels, stats = result  # type: ignore[misc]
            stats = dict(stats)
            stats["language"] = row["language"]
            diagnostics.setdefault(row["language"], []).append(stats)
        payload["input_ids"].append(ids)
        payload["labels"].append(labels)
    return Dataset.from_dict(payload)


def build_tokenization_audit(
    diagnostics_by_language: dict[str, list[dict[str, Any]]],
    *,
    max_seq_length: int,
    split_name: str,
) -> dict[str, Any]:
    return {
        "split": split_name,
        "max_seq_length": max_seq_length,
        "by_language": {
            language: _tokenization_summary(diagnostics_by_language.get(language, []))
            for language in CANONICAL_LANGUAGES
        },
        "overall": _tokenization_summary(
            [item for values in diagnostics_by_language.values() for item in values]
        ),
    }


def bf16_dtype(enabled: bool) -> torch.dtype:
    return torch.bfloat16 if enabled else torch.float32


def load_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=args.offline,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_lora_model(args: argparse.Namespace, use_bf16: bool):
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for Qwen training; install requirements-cpu.txt")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.offline,
        trust_remote_code=False,
        torch_dtype=bf16_dtype(use_bf16),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.lora_target_modules),
    )
    model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    return model


def make_training_args(
    output_dir: Path,
    profile: Profile,
    args: argparse.Namespace,
    use_bf16: bool,
) -> TrainingArguments:
    supported = inspect.signature(TrainingArguments.__init__).parameters
    candidate: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": profile.epochs,
        "per_device_train_batch_size": profile.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "save_steps": profile.save_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": profile.eval_steps,
        "load_best_model_at_end": bool(args.load_best_model_at_end),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": bool(use_bf16),
        "fp16": False,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "use_cpu": True,
        "no_cuda": True,
        "optim": "adamw_torch",
    }
    # eval_strategy replaced evaluation_strategy in newer Transformers; never pass both.
    if "eval_strategy" in supported:
        candidate.pop("evaluation_strategy", None)
    else:
        candidate.pop("eval_strategy", None)
    kwargs = {key: value for key, value in candidate.items() if key in supported}
    return TrainingArguments(**kwargs)


def count_trainable_parameters(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": round(100.0 * trainable / max(total, 1), 6),
    }


def qwen_language_bucket(row: dict[str, Any]) -> str:
    language = str(row.get("language", "en")).split(",")[0].strip().lower()
    if language in {"ta-en", "ta_en", "tamil-english-mixed"}:
        return "tamil-english-mixed"
    if language == "tanglish":
        return "tanglish"
    if language == "ta":
        return "tamil"
    return "english"


def stratified_generation_rows(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Select deterministic held-out conversations across the four language buckets."""
    if count <= 0 or not rows:
        return []
    buckets = ("english", "tamil", "tanglish", "tamil-english-mixed")
    grouped = {bucket: [] for bucket in buckets}
    for row in rows:
        grouped[qwen_language_bucket(row)].append(row)
    for bucket in buckets:
        grouped[bucket].sort(key=lambda row: stable_hash(str(row["conversation_id"]), seed))

    selected: list[dict[str, Any]] = []
    base, remainder = divmod(min(count, len(rows)), len(buckets))
    targets = {bucket: base + (index < remainder) for index, bucket in enumerate(buckets)}
    for bucket in buckets:
        selected.extend(grouped[bucket][: targets[bucket]])

    # If a bucket is unavailable, fill its quota deterministically from the
    # remaining test set rather than reducing the requested evaluation size.
    selected_ids = {row["conversation_id"] for row in selected}
    remaining = [row for row in rows if row["conversation_id"] not in selected_ids]
    remaining.sort(key=lambda row: stable_hash(str(row["conversation_id"]), seed + 1))
    selected.extend(remaining[: max(0, min(count, len(rows)) - len(selected))])
    selected.sort(key=lambda row: stable_hash(str(row["conversation_id"]), seed + 2))
    return selected[:count]


def generate_samples(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    count: int,
    max_new_tokens: int,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    model.eval()
    samples: list[dict[str, Any]] = []
    termination_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple, set)):
        termination_ids.update(int(item) for item in eos_id if item is not None)
    elif eos_id is not None:
        termination_ids.add(int(eos_id))
    for row in rows[:count]:
        messages = row["messages"]
        # Prompt on everything before the final assistant turn.
        assistant_positions = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if not assistant_positions:
            continue
        final_assistant = assistant_positions[-1]
        prompt_messages = messages[:final_assistant]
        expected = messages[final_assistant]["content"]
        rendered = apply_template(
            tokenizer,
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # ``apply_chat_template(..., return_dict=True)`` returns a BatchEncoding
        # in modern Transformers.  BatchEncoding is mapping-like but is not
        # guaranteed to be a built-in ``dict``; treating the whole object as
        # ``input_ids`` makes ``generate()`` fail when it accesses ``.shape``.
        if hasattr(rendered, "keys") and "input_ids" in rendered:
            input_ids = rendered["input_ids"]
            attention_mask = rendered.get("attention_mask")
        elif torch.is_tensor(rendered):
            input_ids = rendered
            attention_mask = None
        else:
            input_ids = torch.as_tensor(rendered, dtype=torch.long)
            attention_mask = None

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        started_at = __import__("time").perf_counter()
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generation_seconds = max(__import__("time").perf_counter() - started_at, 1e-9)
        completion = generated[0, input_ids.shape[-1] :]
        completion_ids = [int(token_id) for token_id in completion.detach().cpu().tolist()]
        produced_eos = any(token_id in termination_ids for token_id in completion_ids)
        output_token_count = len(completion_ids)
        text = tokenizer.decode(completion, skip_special_tokens=True).strip()
        words = __import__("re").findall(r"[^\W_]+(?:['’][^\W_]+)?", text.lower(), flags=__import__("re").UNICODE)
        four_grams = [tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3))]
        repeated_four_gram_ratio = (
            (len(four_grams) - len(set(four_grams))) / len(four_grams) if four_grams else 0.0
        )
        generated_has_tamil = any("\u0b80" <= character <= "\u0bff" for character in text)
        generated_has_latin = bool(__import__("re").search(r"[A-Za-z]", text))
        generated_bucket = qwen_language_bucket(row)
        script_style_adherent = {
            "english": not generated_has_tamil,
            "tamil": generated_has_tamil,
            "tanglish": not generated_has_tamil,
            "tamil-english-mixed": generated_has_tamil and generated_has_latin,
        }.get(generated_bucket, False)
        samples.append(
            {
                "conversation_id": row["conversation_id"],
                "language": qwen_language_bucket(row),
                "prompt_last_message": prompt_messages[-1]["content"] if prompt_messages else "",
                "expected": expected,
                "generated": text,
                "output_token_count": output_token_count,
                "eos_or_end_of_message_produced": produced_eos,
                "max_new_tokens_reached": output_token_count >= max_new_tokens,
                "max_token_hit": output_token_count >= max_new_tokens and not produced_eos,
                "generation_time_seconds": generation_seconds,
                "tokens_per_second": output_token_count / generation_seconds,
                "repeated_4gram_ratio": repeated_four_gram_ratio,
                "script_style_adherent": script_style_adherent,
            }
        )
    return samples


def summarize_generation_samples(
    samples: list[dict[str, Any]], thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    thresholds = dict(
        {
            "min_termination_rate": 0.95,
            "max_max_token_hit_rate": 0.05,
            "max_repeated_4gram_ratio": 0.20,
            "require_script_adherence": True,
        }
        if thresholds is None
        else thresholds
    )
    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(values)
        if not count:
            return {
                "sample_count": 0,
                "termination_rate": None,
                "max_new_tokens_reached_rate": None,
                "max_token_hit_rate": None,
                "mean_output_token_count": None,
                "mean_generation_time_seconds": None,
                "mean_tokens_per_second": None,
                "mean_repeated_4gram_ratio": None,
                "script_adherence_rate": None,
            }
        mean = lambda key: sum(float(item[key]) for item in values) / count
        return {
            "sample_count": count,
            "termination_rate": sum(bool(item["eos_or_end_of_message_produced"]) for item in values) / count,
            "max_new_tokens_reached_rate": sum(bool(item["max_new_tokens_reached"]) for item in values) / count,
            "max_token_hit_rate": sum(bool(item["max_token_hit"]) for item in values) / count,
            "mean_output_token_count": mean("output_token_count"),
            "mean_generation_time_seconds": mean("generation_time_seconds"),
            "mean_tokens_per_second": mean("tokens_per_second"),
            "mean_repeated_4gram_ratio": mean("repeated_4gram_ratio"),
            "script_adherence_rate": sum(bool(item.get("script_style_adherent", False)) for item in values) / count,
        }

    by_language = {
        language: summarize([sample for sample in samples if sample.get("language") == language])
        for language in ("english", "tamil", "tanglish", "tamil-english-mixed")
    }
    overall = summarize(samples)
    unhealthy_reasons = []
    min_termination = float(thresholds["min_termination_rate"])
    max_token_hit = float(thresholds["max_max_token_hit_rate"])
    max_repetition = float(thresholds["max_repeated_4gram_ratio"])
    if overall["termination_rate"] is not None and overall["termination_rate"] < min_termination:
        unhealthy_reasons.append(f"termination rate below {min_termination:.0%}")
    if overall["max_token_hit_rate"] is not None and overall["max_token_hit_rate"] > max_token_hit:
        unhealthy_reasons.append(f"max-token-hit rate above {max_token_hit:.0%}")
    if overall["mean_repeated_4gram_ratio"] is not None and overall["mean_repeated_4gram_ratio"] > max_repetition:
        unhealthy_reasons.append("overall repeated-4gram ratio above threshold")
    if thresholds.get("require_script_adherence", True) and overall["script_adherence_rate"] is not None and overall["script_adherence_rate"] < 1.0:
        unhealthy_reasons.append("overall script adherence below threshold")
    for language in ("english", "tamil", "tanglish", "tamil-english-mixed"):
        summary = by_language[language]
        if not summary["sample_count"]:
            unhealthy_reasons.append(f"missing language bucket: {language}")
            continue
        if summary["termination_rate"] < min_termination:
            unhealthy_reasons.append(f"{language} termination rate below threshold")
        if summary["max_token_hit_rate"] > max_token_hit:
            unhealthy_reasons.append(f"{language} max-token-hit rate above threshold")
        if summary["mean_repeated_4gram_ratio"] > max_repetition:
            unhealthy_reasons.append(f"{language} repeated-4gram ratio above threshold")
        if thresholds.get("require_script_adherence", True) and summary["script_adherence_rate"] < 1.0:
            unhealthy_reasons.append(f"{language} script adherence below threshold")
    return {
        "overall": overall,
        "by_language": by_language,
        "thresholds": thresholds,
        "health": {
            "healthy": not unhealthy_reasons,
            "unhealthy_reasons": unhealthy_reasons,
            "manual_quality_review_required": True,
        },
    }


def save_markdown_report(path: Path, report: dict[str, Any]) -> None:
    metrics = report.get("test_metrics", {})
    generation = report.get("generation_evaluation", {})
    generation_health = generation.get("health", {})
    health_line = "HEALTHY"
    if not generation_health.get("healthy", True):
        health_line = "UNHEALTHY CANDIDATE: " + "; ".join(generation_health.get("unhealthy_reasons", []))
    lines = [
        "# Swico Qwen3-0.6B LoRA training report",
        "",
        f"- Status: **{report.get('status')}**",
        f"- Base model: `{report.get('base_model')}`",
        f"- Profile: `{report.get('profile')}`",
        f"- Train conversations: {report.get('split_conversations', {}).get('train')}",
        f"- Validation conversations: {report.get('split_conversations', {}).get('validation')}",
        f"- Test conversations: {report.get('split_conversations', {}).get('test')}",
        f"- Test loss: {metrics.get('eval_loss', metrics.get('test_loss', 'n/a'))}",
        f"- Test perplexity: {report.get('test_perplexity', 'n/a')}",
        f"- Adapter path: `{report.get('adapter_path')}`",
        f"- Evaluated model: **{report.get('evaluated_model_status', 'unknown')}**",
        f"- Generation health: **{health_line}**",
        f"- Generation evaluation: samples={report.get('effective_training_config', {}).get('generation_sample_count')}, max_new_tokens={report.get('effective_training_config', {}).get('generation_max_new_tokens')}",
        f"- Manual quality review required: **{report.get('candidate_status', {}).get('manual_quality_review_required', True)}**",
        f"- Promotion eligible: **{report.get('candidate_status', {}).get('promotion_eligible', False)}**",
        "",
        "## Effective training configuration",
        "",
        f"- Dataset SHA-256: `{report.get('effective_training_config', {}).get('dataset_sha256')}`",
        f"- Conversations: raw rows={report.get('dataset', {}).get('raw_message_rows')}, deduplicated={report.get('dataset', {}).get('unique_conversations_after_dedup')}, selected={report.get('dataset', {}).get('selected_conversations_before_split')}",
        f"- Conversation cap: {report.get('effective_training_config', {}).get('max_conversations_cap_effective')} ({report.get('effective_training_config', {}).get('max_conversations_cap_source')})",
        f"- Sequence length: {report.get('effective_training_config', {}).get('max_seq_length')}",
        f"- Learning rate: {report.get('effective_training_config', {}).get('learning_rate')}",
        f"- Epochs: requested={report.get('effective_training_config', {}).get('requested_epochs')}, actual={report.get('effective_training_config', {}).get('actual_epochs')}",
        f"- Global steps: {report.get('effective_training_config', {}).get('global_steps')}",
        f"- Batch: physical={report.get('effective_training_config', {}).get('batch_size')}, gradient accumulation={report.get('effective_training_config', {}).get('gradient_accumulation_steps')}",
        f"- Best checkpoint: `{report.get('effective_training_config', {}).get('best_checkpoint')}`",
        f"- Best validation loss: {report.get('effective_training_config', {}).get('best_validation_loss')}",
        f"- BF16: supported={report.get('effective_training_config', {}).get('bf16_supported')}, enabled={report.get('effective_training_config', {}).get('bf16_enabled')}",
        f"- LoRA: {report.get('effective_training_config', {}).get('lora')}",
        f"- Export status: {report.get('effective_training_config', {}).get('merged_status')}",
        f"- Data-quality exclusions: conversations={report.get('data_quality', {}).get('excluded_conversations')}, assistant messages={report.get('data_quality', {}).get('excluded_assistant_messages')}",
        f"- Tokenization audit: `{report.get('effective_training_config', {}).get('tokenization_audit_path')}`",
        "",
        "## LoRA parameters",
        "",
        f"- Trainable parameters: {report.get('parameters', {}).get('trainable_parameters')}",
        f"- Trainable percentage: {report.get('parameters', {}).get('trainable_percent')}%",
        "",
        "## Sample generations",
        "",
        f"Overall generation summary: {generation.get('overall', {})}",
        "",
        "Per-language generation summary:",
        "",
    ]
    for language, summary in generation.get("by_language", {}).items():
        lines.append(f"- {language}: {summary}")
    lines.append("")
    for sample in report.get("generation_samples", []):
        lines.extend(
            [
                f"### {sample['conversation_id']}",
                f"Prompt: {sample['prompt_last_message']}",
                "",
                f"Expected: {sample['expected']}",
                "",
                f"Generated: {sample['generated']}",
                f"Metrics: tokens={sample['output_token_count']}, terminated={sample['eos_or_end_of_message_produced']}, max-token-hit={sample['max_token_hit']}, time={sample['generation_time_seconds']:.3f}s, tokens/sec={sample['tokens_per_second']:.2f}, repeated-4gram-ratio={sample['repeated_4gram_ratio']:.3f}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = resolve_profile(args)
    validate_config(args, profile)
    config = qwen_effective_config(args, profile)
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    data_path = args.data.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data_hash = file_sha256(data_path)
    fingerprint = training_fingerprint(config, data_hash=data_hash)
    config["dataset_sha256"] = data_hash
    output, resumed = allocate_run(args, profile, fingerprint)
    logger = setup_logger(output)
    if args.create_latest_links:
        replace_pointer(args.output_root.resolve(), "latest", output)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "invocation_fingerprint": fingerprint,
        "data_sha256": data_hash,
        "resumed": resumed,
        "output": str(output),
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(output / "run_config.json", config)
    system = preflight(args, logger)
    atomic_write_json(output / "system.json", system)
    use_bf16 = bool(system["bf16_enabled"])

    splits, dataset_meta = prepare_dataset(data_path, output, profile, args, logger)
    config["dataset_preparation"] = dataset_meta
    atomic_write_json(output / "run_config.json", config)
    if args.prepare_only and not args.audit_tokenization:
        atomic_write_json(output / "run_state.json", {"prepared": True, "final_evaluation_complete": False})
        logger.info("Prepare-only requested; stopping before model download/training")
        return 0

    tokenizer = load_tokenizer(args)
    tokenized: dict[str, Dataset] = {}
    tokenization_audit_by_split: dict[str, Any] = {}
    all_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for name, rows in splits.items():
        diagnostics: dict[str, list[dict[str, Any]]] = {}
        tokenized[name] = tokenize_split(
            tokenizer,
            rows,
            profile.max_seq_length,
            diagnostics=diagnostics,
            allow_prompt_only=args.audit_tokenization,
        )
        tokenization_audit_by_split[name] = build_tokenization_audit(
            diagnostics, max_seq_length=profile.max_seq_length, split_name=name
        )
        for language, values in diagnostics.items():
            all_diagnostics.setdefault(language, []).extend(values)
    tokenization_audit = {
        "version": SCRIPT_VERSION,
        "max_seq_length": profile.max_seq_length,
        "by_split": tokenization_audit_by_split,
        "overall_by_language": {
            language: _tokenization_summary(all_diagnostics.get(language, []))
            for language in CANONICAL_LANGUAGES
        },
        "overall": _tokenization_summary(
            [item for values in all_diagnostics.values() for item in values]
        ),
    }
    atomic_write_json(output / "prepared" / "tokenization_audit.json", tokenization_audit)
    config["tokenization_audit"] = tokenization_audit
    atomic_write_json(output / "run_config.json", config)
    if args.audit_tokenization:
        atomic_write_json(output / "run_state.json", {"prepared": True, "tokenization_audit_complete": True, "final_evaluation_complete": False})
        logger.info("Tokenization audit requested; stopping before model download/training")
        return 0
    logger.info(
        "Tokenized conversations: train=%d validation=%d test=%d max_seq_length=%d",
        len(tokenized["train"]), len(tokenized["validation"]), len(tokenized["test"]), profile.max_seq_length,
    )

    model = build_lora_model(args, use_bf16)
    parameters = count_trainable_parameters(model)
    logger.info(
        "LoRA trainable parameters: %d / %d (%.4f%%)",
        parameters["trainable_parameters"], parameters["total_parameters"], parameters["trainable_percent"],
    )
    trainer_dir = output / "trainer"
    training_args = make_training_args(trainer_dir, profile, args, use_bf16)
    callbacks: list[TrainerCallback] = []
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    guard: QwenResourceGuardCallback | None = None
    if args.memory_guard:
        guard = QwenResourceGuardCallback(
            output_path=output / "memory_guard.json",
            logger=logger,
            interval_steps=args.memory_guard_interval_steps,
            emergency_available_memory_gib=args.emergency_available_memory_gib,
            max_process_rss_gib=args.max_process_rss_gib,
        )
        callbacks.append(guard)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CausalLMCollator(tokenizer),
        callbacks=callbacks,
    )
    checkpoint = get_last_checkpoint(str(trainer_dir)) if args.resume and trainer_dir.exists() else None
    logger.info("Training Qwen LoRA%s", f" from {checkpoint}" if checkpoint else "")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()

    # Early stopping is a training-only callback.  Held-out evaluation below
    # uses the ``test_`` metric prefix, so leaving this callback attached makes
    # Transformers look for ``eval_loss`` during test evaluation and emit a
    # misleading warning that early stopping was disabled.
    if args.early_stopping:
        trainer.remove_callback(EarlyStoppingCallback)

    if guard is not None and guard.triggered:
        atomic_write_json(
            output / "run_state.json",
            {"resource_stop_requested": True, "final_evaluation_complete": False, "global_step": trainer.state.global_step},
        )
        raise ResourceStopRequested("Memory guard paused training after requesting a checkpoint; rerun the same command to resume")

    adapter_dir = output / "models" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    test_loss = float(test_metrics.get("test_loss", test_metrics.get("eval_loss", float("nan"))))
    perplexity = math.exp(min(test_loss, 20.0)) if math.isfinite(test_loss) else None
    generation_rows = stratified_generation_rows(splits["test"], profile.eval_generation_samples, args.seed)
    generation_samples = generate_samples(
        trainer.model,
        tokenizer,
        generation_rows,
        len(generation_rows),
        args.generation_max_new_tokens,
        use_bf16,
    )
    generation_evaluation = summarize_generation_samples(
        generation_samples,
        thresholds={
            "min_termination_rate": args.min_language_termination_rate,
            "max_max_token_hit_rate": args.max_language_max_token_hit_rate,
            "max_repeated_4gram_ratio": args.max_language_repeated_4gram_ratio,
            "require_script_adherence": args.require_language_script_adherence,
        },
    )
    logger.info(
        "Held-out generation: %d samples, termination=%.1f%%, max-token-hit=%.1f%%",
        len(generation_samples),
        100.0 * (generation_evaluation["overall"]["termination_rate"] or 0.0),
        100.0 * (generation_evaluation["overall"]["max_token_hit_rate"] or 0.0),
    )

    merged_path: str | None = None
    if args.merge_adapter:
        logger.info("Merging LoRA adapter into a standalone model; this temporarily uses more RAM")
        merged_dir = output / "models" / "merged"
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        merged_path = str(merged_dir)

    report = {
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_model": args.base_model,
        "profile": profile.name,
        "dataset": dataset_meta,
        "split_conversations": dataset_meta["split_conversations"],
        "parameters": parameters,
        "train_metrics": train_result.metrics,
        "test_metrics": test_metrics,
        "test_perplexity": perplexity,
        "adapter_path": str(adapter_dir),
        "merged_model_path": merged_path,
        "evaluated_model_status": "adapter-based",
        "generation_samples": generation_samples,
        "generation_evaluation": generation_evaluation,
        "tokenization_audit": tokenization_audit,
        "data_quality": {
            "report_path": str(output / "prepared" / "data_quality.json"),
            "excluded_conversations": dataset_meta["data_quality_excluded_conversations"],
            "excluded_assistant_messages": dataset_meta["data_quality_excluded_assistant_messages"],
        },
        "candidate_status": {
            "training_completed": True,
            "generation_healthy": generation_evaluation["health"]["healthy"],
            "beats_base": None,
            "does_not_regress_champion": None,
            "manual_quality_review_required": True,
            "automatic_promotion_gate_passed": False,
            "promotion_eligible": False,
            "reason": "Training completion and generation health do not establish factual quality or safe deployment; run base/champion comparison and manual review.",
        },
        "effective_training_config": {
            "dataset_sha256": data_hash,
            "conversation_counts": dataset_meta["split_conversations"],
            "raw_message_rows": dataset_meta["raw_message_rows"],
            "unique_conversations_after_dedup": dataset_meta["unique_conversations_after_dedup"],
            "selected_conversations": dataset_meta["selected_conversations_before_split"],
            "max_conversations_cap_requested": dataset_meta["max_conversations_cap_requested"],
            "max_conversations_cap_effective": dataset_meta["max_conversations_cap_effective"],
            "max_conversations_cap_source": dataset_meta["max_conversations_cap_source"],
            "max_seq_length": profile.max_seq_length,
            "learning_rate": args.learning_rate,
            "requested_epochs": profile.epochs,
            "actual_epochs": trainer.state.epoch,
            "global_steps": trainer.state.global_step,
            "batch_size": profile.batch_size,
            "gradient_accumulation_steps": profile.gradient_accumulation_steps,
            "eval_steps": profile.eval_steps,
            "save_steps": profile.save_steps,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": list(args.lora_target_modules),
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "bf16_supported": system["cpu_bf16_supported"],
            "bf16_enabled": system["bf16_enabled"],
            "merged_status": "merged export created" if merged_path else "adapter-only export; merged export disabled",
            "generation_sample_count": len(generation_samples),
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "generation_health_thresholds": generation_evaluation["thresholds"],
            "language_validation": {
                "canonical_languages": list(CANONICAL_LANGUAGES),
                "definitions": LANGUAGE_DEFINITIONS,
                "system_prompts_normalized": dataset_meta["system_prompts_normalized"],
            },
            "data_quality": quality_config_from_args(args),
            "tokenization_audit_path": str(output / "prepared" / "tokenization_audit.json"),
        },
    }
    reports_dir = output / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports_dir / "final_report.json", report)
    save_markdown_report(reports_dir / "final_report.md", report)
    atomic_write_json(
        output / "run_state.json",
        {
            "final_evaluation_complete": True,
            "completed_at": report["completed_at"],
            "global_step": trainer.state.global_step,
        },
    )
    if args.create_latest_links:
        replace_pointer(args.output_root.resolve(), "latest-completed", output)
    logger.info("Qwen training complete. Adapter: %s", adapter_dir)
    logger.info("Final report: %s", reports_dir / "final_report.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, ResourceStopRequested, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

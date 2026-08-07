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
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


SCRIPT_VERSION = "1.0.0-qwen3-cpu-lora"
BASE_MODEL = "Qwen/Qwen3-0.6B"
REQUIRED_COLUMNS = ("conversation_id", "turn_index", "role", "content", "language")
VALID_ROLES = {"system", "user", "assistant"}


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    max_conversations: int | None
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    eval_generation_samples: int


PROFILES: dict[str, Profile] = {
    "smoke": Profile("smoke", 64, 1.0, 1, 4, 512, 1),
    "vm": Profile("vm", 12_000, 3.0, 1, 16, 512, 3),
    "full": Profile("full", None, 3.0, 1, 16, 768, 5),
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

    parser.add_argument("--max-conversations", type=int, default=env_optional_int("SWICO_QWEN_MAX_CONVERSATIONS"))
    parser.add_argument("--train-split-fraction", type=float, default=env_float("SWICO_QWEN_TRAIN_SPLIT_FRACTION", 0.80))
    parser.add_argument("--validation-split-fraction", type=float, default=env_float("SWICO_QWEN_VALIDATION_SPLIT_FRACTION", 0.10))
    parser.add_argument("--test-split-fraction", type=float, default=env_float("SWICO_QWEN_TEST_SPLIT_FRACTION", 0.10))
    parser.add_argument("--max-seq-length", type=int, default=env_optional_int("SWICO_QWEN_MAX_SEQ_LENGTH"))

    parser.add_argument("--epochs", type=float, default=env_optional_float("SWICO_QWEN_EPOCHS"))
    parser.add_argument("--batch-size", type=int, default=env_optional_int("SWICO_QWEN_BATCH_SIZE"))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("SWICO_QWEN_EVAL_BATCH_SIZE", 1))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=env_optional_int("SWICO_QWEN_GRADIENT_ACCUMULATION_STEPS"))
    parser.add_argument("--learning-rate", type=float, default=env_float("SWICO_QWEN_LEARNING_RATE", 1.0e-4))
    parser.add_argument("--weight-decay", type=float, default=env_float("SWICO_QWEN_WEIGHT_DECAY", 0.01))
    parser.add_argument("--warmup-ratio", type=float, default=env_float("SWICO_QWEN_WARMUP_RATIO", 0.05))
    parser.add_argument("--lr-scheduler-type", default=env_text("SWICO_QWEN_LR_SCHEDULER_TYPE", "cosine"))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("SWICO_QWEN_MAX_GRAD_NORM", 1.0))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_GRADIENT_CHECKPOINTING", True))
    parser.add_argument("--logging-steps", type=int, default=env_int("SWICO_QWEN_LOGGING_STEPS", 5))
    parser.add_argument("--save-total-limit", type=int, default=env_int("SWICO_QWEN_SAVE_TOTAL_LIMIT", 2))
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
    parser.add_argument("--early-stopping-patience", type=int, default=env_int("SWICO_QWEN_EARLY_STOPPING_PATIENCE", 2))
    parser.add_argument("--early-stopping-threshold", type=float, default=env_float("SWICO_QWEN_EARLY_STOPPING_THRESHOLD", 0.001))
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_LOAD_BEST_MODEL_AT_END", True))
    parser.add_argument("--merge-adapter", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_QWEN_MERGE_ADAPTER", False))
    parser.add_argument("--eval-generation-samples", type=int, default=env_optional_int("SWICO_QWEN_EVAL_GENERATION_SAMPLES"))
    parser.add_argument("--generation-max-new-tokens", type=int, default=env_int("SWICO_QWEN_GENERATION_MAX_NEW_TOKENS", 128))

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
        max_conversations=args.max_conversations if args.max_conversations is not None else base.max_conversations,
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
    )


def validate_config(args: argparse.Namespace, profile: Profile) -> None:
    if args.data is None:
        raise ConfigError("SWICO_QWEN_DATA_PATH or --data is required")
    if profile.epochs <= 0 or profile.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ConfigError("epochs and batch sizes must be positive")
    if profile.gradient_accumulation_steps <= 0:
        raise ConfigError("gradient accumulation steps must be positive")
    if profile.max_seq_length < 64:
        raise ConfigError("max sequence length must be at least 64")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ConfigError("learning rate and max grad norm must be positive")
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
        "eval_batch_size": args.eval_batch_size,
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
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "memory_guard": args.memory_guard,
        "emergency_available_memory_gib": args.emergency_available_memory_gib,
        "max_process_rss_gib": args.max_process_rss_gib,
    }


def training_fingerprint(config: dict[str, Any], data_hash: str | None = None) -> str:
    material = dict(config)
    material.pop("output", None)
    material.pop("run_mode", None)
    material.pop("run_label", None)
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


def validate_and_group_conversations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Qwen dataset is missing required columns: {sorted(missing)}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["conversation_id"] = frame["conversation_id"].map(normalize_content)
    frame["role"] = frame["role"].map(lambda value: normalize_content(value).lower())
    frame["content"] = frame["content"].map(lambda value: str(value).strip())
    frame["language"] = frame["language"].map(normalize_content)
    try:
        frame["turn_index"] = frame["turn_index"].astype(int)
    except Exception as exc:
        raise ValueError("turn_index must contain integers") from exc
    frame = frame[(frame["conversation_id"] != "") & (frame["content"] != "")].copy()
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
        messages = [
            {"role": str(row.role), "content": str(row.content)}
            for row in group.itertuples(index=False)
        ]
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        digest = stable_hash(serialized)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        conversations.append(
            {
                "conversation_id": str(conversation_id),
                "messages": messages,
                "language": ",".join(sorted(set(group["language"].astype(str)) - {""})),
                "digest": digest,
            }
        )
    if len(conversations) < 3:
        raise ValueError("At least 3 unique conversations are required for train/validation/test splitting")
    return conversations


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
    conversations = validate_and_group_conversations(frame)
    splits = split_conversations(
        conversations,
        seed=args.seed,
        fractions=(args.train_split_fraction, args.validation_split_fraction, args.test_split_fraction),
        max_conversations=profile.max_conversations,
    )
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
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
        "unique_conversations_after_dedup": len(conversations),
        "split_conversations": {name: len(values) for name, values in splits.items()},
        "conversation_id_overlap": overlaps,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(prepared / "metadata.json", meta)
    logger.info(
        "Prepared conversations: train=%d validation=%d test=%d",
        len(splits["train"]), len(splits["validation"]), len(splits["test"]),
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


def assistant_only_tokens(tokenizer, messages: list[dict[str, str]], max_seq_length: int) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and train only on assistant content tokens.

    Qwen3's distributed chat template does not expose Transformers'
    ``{% generation %}`` spans, so ``return_assistant_tokens_mask=True`` is
    not reliable.  Instead we render the exact chat text once, derive the
    character ranges occupied by assistant message content, and map those
    ranges to token offsets from the fast tokenizer.  This avoids false
    "conversation truncated" failures caused by template boundary rewrites.
    """
    rendered = str(
        apply_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    assistant_spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        # Rendering the preceding turns with add_generation_prompt=True gives
        # the exact text immediately before this assistant's response.
        prefix = str(
            apply_template(
                tokenizer,
                messages[:index],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        content = str(message["content"])
        start = len(prefix)

        # Normal Qwen3 path: assistant content follows the generation prompt
        # verbatim.  Keep a guarded search fallback for compatible templates
        # that insert harmless whitespace around the response.
        if not rendered.startswith(content, start):
            located = rendered.find(content, max(0, start - 16))
            if located < 0:
                raise ValueError(
                    f"Could not locate assistant response {index} in rendered Qwen chat template"
                )
            start = located
        assistant_spans.append((start, start + len(content)))

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    labels: list[int] = []
    for token_id, offset in zip(ids, offsets, strict=False):
        token_start, token_end = int(offset[0]), int(offset[1])
        is_assistant = token_end > token_start and any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in assistant_spans
        )
        labels.append(int(token_id) if is_assistant else -100)

    ids = ids[-max_seq_length:]
    labels = labels[-max_seq_length:]
    if not any(label != -100 for label in labels):
        raise ValueError(
            "A conversation has no assistant response tokens after tokenization/truncation; "
            "verify the chat template or increase SWICO_QWEN_MAX_SEQ_LENGTH"
        )
    return ids, labels


def tokenize_split(tokenizer, rows: list[dict[str, Any]], max_seq_length: int) -> Dataset:
    payload = {"input_ids": [], "labels": []}
    for row in rows:
        ids, labels = assistant_only_tokens(tokenizer, row["messages"], max_seq_length)
        payload["input_ids"].append(ids)
        payload["labels"].append(labels)
    return Dataset.from_dict(payload)


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
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
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
        )
        if isinstance(rendered, dict):
            input_ids = rendered["input_ids"]
            attention_mask = rendered.get("attention_mask")
        else:
            input_ids = rendered
            attention_mask = None
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = generated[0, input_ids.shape[-1] :]
        text = tokenizer.decode(completion, skip_special_tokens=True).strip()
        samples.append(
            {
                "conversation_id": row["conversation_id"],
                "prompt_last_message": prompt_messages[-1]["content"] if prompt_messages else "",
                "expected": expected,
                "generated": text,
            }
        )
    return samples


def save_markdown_report(path: Path, report: dict[str, Any]) -> None:
    metrics = report.get("test_metrics", {})
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
        "",
        "## LoRA parameters",
        "",
        f"- Trainable parameters: {report.get('parameters', {}).get('trainable_parameters')}",
        f"- Trainable percentage: {report.get('parameters', {}).get('trainable_percent')}%",
        "",
        "## Sample generations",
        "",
    ]
    for sample in report.get("generation_samples", []):
        lines.extend(
            [
                f"### {sample['conversation_id']}",
                f"Prompt: {sample['prompt_last_message']}",
                "",
                f"Expected: {sample['expected']}",
                "",
                f"Generated: {sample['generated']}",
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
    if args.prepare_only:
        atomic_write_json(output / "run_state.json", {"prepared": True, "final_evaluation_complete": False})
        logger.info("Prepare-only requested; stopping before model download/training")
        return 0

    tokenizer = load_tokenizer(args)
    tokenized = {
        name: tokenize_split(tokenizer, rows, profile.max_seq_length)
        for name, rows in splits.items()
    }
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
    generation_samples = generate_samples(
        trainer.model,
        tokenizer,
        splits["test"],
        profile.eval_generation_samples,
        args.generation_max_new_tokens,
        use_bf16,
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
        "generation_samples": generation_samples,
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

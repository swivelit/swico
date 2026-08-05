#!/usr/bin/env python3
"""
Swico CPU retrieval-model fine-tuning pipeline.

Designed for a 4-physical-core / 8-logical-CPU VM with about 29 GiB RAM,
no GPU, and constrained compute. It fine-tunes intfloat/multilingual-e5-small
for query-to-passage retrieval; it does not train a general-purpose LLM.

Version 4 capabilities:
- typed training.env configuration with strict validation and CLI overrides
- leakage-resistant connected-component data splitting
- repeated-boilerplate removal and exact-pair deduplication
- configurable partial encoder fine-tuning
- stage 1 in-batch-negative retrieval training
- stage 2 guarded hard-negative curriculum
- no-duplicate batches and configurable Matryoshka loss
- validation-driven early stopping with callback-state resume
- automatic restoration of the best validation checkpoint
- stage-1 versus stage-2 validation selection
- bounded IR evaluation, confidence calibration and latency reporting
- timestamped run directories with automatic compatible resume
- resource-aware safe autotuning, memory guard and baseline quality protection
- resumable, configuration-safe checkpoints and disk cleanup

Examples:
  ./run_vm_training.sh --print-config
  SWICO_PROFILE=smoke ./run_vm_training.sh
  ./run_vm_training.sh
  ./run_vm_training.sh --env-file experiments/run-02.env
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import inspect
import json
from importlib.metadata import PackageNotFoundError, version as package_version
import logging
import math
import os
import random
import re
import resource
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import Version

from training_config import (
    ConfigError,
    env_bool,
    env_float,
    env_float_tuple,
    env_int,
    env_optional_bool,
    env_optional_float,
    env_optional_int,
    env_optional_int_tuple,
    env_optional_text,
    env_text,
    initialize_environment,
)

# Load the typed env file before importing torch/numpy so CPU library thread
# settings are effective from process startup. Existing shell variables take
# precedence over values in the file.
try:
    LOADED_ENV_FILE = initialize_environment(sys.argv[1:])
except ConfigError as exc:
    print(f"Configuration error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
_DEFAULT_THREADS = max(1, min(env_int("SWICO_CPU_THREADS", 8), os.cpu_count() or 4))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import Dataset

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

try:
    # Sentence Transformers 5.6+ namespaced paths. Import these first to avoid
    # deprecation warnings while keeping a fallback for older compatible releases.
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.sentence_transformer.losses import (
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
    )
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


BASE_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
SCRIPT_VERSION = "4.0.0-cpu-adaptive-runs"


class ResourceStopRequested(RuntimeError):
    """Raised after a safe checkpoint when the memory guard stops a stage."""


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    max_train_rows: int | None
    stage1_epochs: float
    stage2_epochs: float
    batch_size: int
    max_seq_length: int
    trainable_layers: int
    eval_queries: int
    eval_corpus: int
    mining_top_k: int
    hard_negatives: bool
    matryoshka_dims: tuple[int, ...]


PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        name="smoke",
        max_train_rows=2_000,
        stage1_epochs=1.0,
        stage2_epochs=0.0,
        batch_size=16,
        max_seq_length=160,
        trainable_layers=2,
        eval_queries=200,
        eval_corpus=1_000,
        mining_top_k=8,
        hard_negatives=False,
        matryoshka_dims=(384, 256),
    ),
    "vm": Profile(
        name="vm",
        max_train_rows=45_000,
        stage1_epochs=5.0,
        stage2_epochs=4.0,
        batch_size=64,
        max_seq_length=192,
        trainable_layers=4,
        eval_queries=1_000,
        eval_corpus=5_000,
        mining_top_k=16,
        hard_negatives=True,
        matryoshka_dims=(384, 256, 128),
    ),
    "full": Profile(
        name="full",
        max_train_rows=None,
        stage1_epochs=5.0,
        stage2_epochs=4.0,
        batch_size=64,
        max_seq_length=192,
        trainable_layers=4,
        eval_queries=2_000,
        eval_corpus=10_000,
        mining_top_k=24,
        hard_negatives=True,
        matryoshka_dims=(384, 256, 128),
    ),
}


@dataclasses.dataclass
class EvalBundle:
    name: str
    queries: dict[str, str]
    corpus: dict[str, str]
    relevant_docs: dict[str, set[str]]
    query_order: list[str]
    corpus_order: list[str]
    evaluator: InformationRetrievalEvaluator


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot == rroot:
            return
        if self.size[lroot] < self.size[rroot]:
            lroot, rroot = rroot, lroot
        self.parent[rroot] = lroot
        self.size[lroot] += self.size[rroot]


def _path_from_env(name: str, default: str | None) -> Path | None:
    value = env_optional_text(name)
    if value is None:
        return Path(default) if default is not None else None
    return Path(value)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one number is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only fine-tuning for the Swico multilingual E5 retrieval model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=LOADED_ENV_FILE or Path("training.env"),
        help="Typed env file loaded before command-line parsing",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default=env_text("SWICO_PROFILE", "vm"))
    parser.add_argument("--data", type=Path, default=_path_from_env("SWICO_DATA_PATH", None), help="CSV file or ZIP containing a CSV")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(env_text("SWICO_OUTPUT_ROOT", "training_artifacts/e5-small-swico")),
        help="Root that contains timestamped runs/ directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_path_from_env("SWICO_OUTPUT_DIR", None),
        help="Exact run directory override; normally leave SWICO_OUTPUT_DIR=auto",
    )
    parser.add_argument(
        "--run-mode",
        choices=("auto", "new", "resume-latest"),
        default=env_text("SWICO_RUN_MODE", "auto"),
        help="auto resumes a compatible incomplete run, otherwise creates a timestamped run",
    )
    parser.add_argument("--run-id", default=env_optional_text("SWICO_RUN_ID"))
    parser.add_argument("--run-label", default=env_optional_text("SWICO_RUN_LABEL"))
    parser.add_argument(
        "--create-latest-links",
        action=argparse.BooleanOptionalAction,
        default=env_bool("SWICO_CREATE_LATEST_LINKS", True),
    )
    parser.add_argument("--base-model", default=env_text("SWICO_BASE_MODEL", BASE_MODEL))
    parser.add_argument("--threads", type=int, default=env_int("SWICO_CPU_THREADS", _DEFAULT_THREADS))
    parser.add_argument("--seed", type=int, default=env_int("SWICO_SEED", 42))

    parser.add_argument("--boilerplate-threshold", type=int, default=env_int("SWICO_BOILERPLATE_THRESHOLD", 100))
    parser.add_argument("--max-train-rows", type=int, default=env_optional_int("SWICO_MAX_TRAIN_ROWS"))
    parser.add_argument("--train-split-fraction", type=float, default=env_float("SWICO_TRAIN_SPLIT_FRACTION", 0.80))
    parser.add_argument("--validation-split-fraction", type=float, default=env_float("SWICO_VALIDATION_SPLIT_FRACTION", 0.10))
    parser.add_argument("--test-split-fraction", type=float, default=env_float("SWICO_TEST_SPLIT_FRACTION", 0.10))

    parser.add_argument("--batch-size", type=int, default=env_optional_int("SWICO_BATCH_SIZE"))
    parser.add_argument("--max-seq-length", type=int, default=env_optional_int("SWICO_MAX_SEQ_LENGTH"))
    parser.add_argument("--trainable-layers", type=int, default=env_optional_int("SWICO_TRAINABLE_LAYERS"))
    parser.add_argument("--stage1-epochs", type=float, default=env_optional_float("SWICO_STAGE1_EPOCHS"))
    parser.add_argument("--stage2-epochs", type=float, default=env_optional_float("SWICO_STAGE2_EPOCHS"))
    parser.add_argument("--stage1-lr", type=float, default=env_float("SWICO_STAGE1_LR", 2.0e-5))
    parser.add_argument("--stage2-lr", type=float, default=env_float("SWICO_STAGE2_LR", 8.0e-6))
    parser.add_argument("--weight-decay", type=float, default=env_float("SWICO_WEIGHT_DECAY", 0.01))
    parser.add_argument("--warmup-ratio", type=float, default=env_float("SWICO_WARMUP_RATIO", 0.06))
    parser.add_argument("--lr-scheduler-type", default=env_text("SWICO_LR_SCHEDULER_TYPE", "cosine"))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=env_int("SWICO_GRADIENT_ACCUMULATION_STEPS", 1))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("SWICO_MAX_GRAD_NORM", 1.0))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_GRADIENT_CHECKPOINTING", False))
    parser.add_argument("--dataloader-num-workers", type=int, default=env_int("SWICO_DATALOADER_NUM_WORKERS", 0))
    parser.add_argument("--dataloader-drop-last", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_DATALOADER_DROP_LAST", True))
    parser.add_argument("--logging-steps", type=int, default=env_int("SWICO_LOGGING_STEPS", 25))
    parser.add_argument("--save-total-limit", type=int, default=env_int("SWICO_SAVE_TOTAL_LIMIT", 2))

    parser.add_argument("--eval-batch-size", type=int, default=env_int("SWICO_EVAL_BATCH_SIZE", 64))
    parser.add_argument("--eval-loss-rows", type=int, default=env_int("SWICO_EVAL_LOSS_ROWS", 512))
    parser.add_argument("--eval-corpus-chunk-size", type=int, default=env_int("SWICO_EVAL_CORPUS_CHUNK_SIZE", 5000))
    parser.add_argument("--eval-queries", type=int, default=env_optional_int("SWICO_EVAL_QUERIES"))
    parser.add_argument("--eval-corpus", type=int, default=env_optional_int("SWICO_EVAL_CORPUS"))

    parser.add_argument("--hard-negatives", action=argparse.BooleanOptionalAction, default=env_optional_bool("SWICO_HARD_NEGATIVES"))
    parser.add_argument("--mining-top-k", type=int, default=env_optional_int("SWICO_MINING_TOP_K"))
    parser.add_argument("--min-hard-negative-margin", type=float, default=env_float("SWICO_MIN_HARD_NEGATIVE_MARGIN", 0.02))
    parser.add_argument("--mining-batch-size", type=int, default=env_int("SWICO_MINING_BATCH_SIZE", 64))
    parser.add_argument("--mining-chunk-size", type=int, default=env_int("SWICO_MINING_CHUNK_SIZE", 512))
    parser.add_argument("--hnsw-m", type=int, default=env_int("SWICO_HNSW_M", 32))
    parser.add_argument("--hnsw-ef-construction", type=int, default=env_int("SWICO_HNSW_EF_CONSTRUCTION", 80))
    parser.add_argument("--hnsw-ef-search-factor", type=int, default=env_int("SWICO_HNSW_EF_SEARCH_FACTOR", 4))
    parser.add_argument("--negative-max-token-jaccard", type=float, default=env_float("SWICO_NEGATIVE_MAX_TOKEN_JACCARD", 0.88))

    parser.add_argument("--loss-scale", type=float, default=env_float("SWICO_LOSS_SCALE", 20.0))
    parser.add_argument("--matryoshka-dims", type=_parse_int_tuple, default=env_optional_int_tuple("SWICO_MATRYOSHKA_DIMS"))
    parser.add_argument("--matryoshka-weights", type=_parse_float_tuple, default=env_float_tuple("SWICO_MATRYOSHKA_WEIGHTS", (1.0, 0.35, 0.35)))
    parser.add_argument("--matryoshka-dims-per-step", type=int, default=env_int("SWICO_MATRYOSHKA_DIMS_PER_STEP", -1))

    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_EARLY_STOPPING", True))
    parser.add_argument("--early-stopping-patience", type=int, default=env_int("SWICO_EARLY_STOPPING_PATIENCE", 2))
    parser.add_argument("--early-stopping-threshold", type=float, default=env_float("SWICO_EARLY_STOPPING_THRESHOLD", 0.001))
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_LOAD_BEST_MODEL_AT_END", True))

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=env_optional_bool("SWICO_BF16"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_RESUME", True))
    parser.add_argument("--prepare-only", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_PREPARE_ONLY", False))
    parser.add_argument("--skip-base-eval", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_SKIP_BASE_EVAL", False))
    parser.add_argument("--keep-intermediate", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_KEEP_INTERMEDIATE", False))
    parser.add_argument("--keep-checkpoints", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_KEEP_CHECKPOINTS", False))
    parser.add_argument("--overwrite-output", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_OVERWRITE_OUTPUT", False))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_OFFLINE", False), help="Use only locally cached model files")

    parser.add_argument("--print-config", action="store_true", help="Validate and print the effective configuration without reading data or training")
    parser.add_argument("--min-available-memory-gib", type=float, default=env_float("SWICO_MIN_AVAILABLE_MEMORY_GIB", 8.0))
    parser.add_argument("--min-free-disk-gib", type=float, default=env_float("SWICO_MIN_FREE_DISK_GIB", 2.0))
    parser.add_argument("--warn-free-disk-gib", type=float, default=env_float("SWICO_WARN_FREE_DISK_GIB", 4.0))

    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_AUTOTUNE", True))
    parser.add_argument(
        "--autotune-mode",
        choices=("safe", "aggressive"),
        default=env_text("SWICO_AUTOTUNE_MODE", "safe"),
        help="safe changes only throughput/resource controls; aggressive may also raise the physical training batch",
    )
    parser.add_argument("--autotune-max-inference-batch", type=int, default=env_int("SWICO_AUTOTUNE_MAX_INFERENCE_BATCH", 128))
    parser.add_argument("--autotune-max-train-batch", type=int, default=env_int("SWICO_AUTOTUNE_MAX_TRAIN_BATCH", 96))
    parser.add_argument("--autotune-memory-reserve-gib", type=float, default=env_float("SWICO_AUTOTUNE_MEMORY_RESERVE_GIB", 8.0))
    parser.add_argument("--autotune-memory-utilization", type=float, default=env_float("SWICO_AUTOTUNE_MEMORY_UTILIZATION", 0.70))
    parser.add_argument("--autotune-train-batch", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_AUTOTUNE_TRAIN_BATCH", False))

    parser.add_argument("--memory-guard", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_MEMORY_GUARD", True))
    parser.add_argument("--memory-guard-interval-steps", type=int, default=env_int("SWICO_MEMORY_GUARD_INTERVAL_STEPS", 5))
    parser.add_argument("--emergency-available-memory-gib", type=float, default=env_float("SWICO_EMERGENCY_AVAILABLE_MEMORY_GIB", 4.0))
    parser.add_argument("--max-process-rss-gib", type=float, default=env_optional_float("SWICO_MAX_PROCESS_RSS_GIB"))

    parser.add_argument("--baseline-protection", action=argparse.BooleanOptionalAction, default=env_bool("SWICO_BASELINE_PROTECTION", True))
    parser.add_argument("--min-validation-gain", type=float, default=env_float("SWICO_MIN_VALIDATION_GAIN", 0.0))
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def effective_configuration_preview(args: argparse.Namespace, profile: Profile) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "env_file": str(LOADED_ENV_FILE) if LOADED_ENV_FILE else None,
        "profile": dataclasses.asdict(profile),
        "arguments": {
            key: _json_safe(value)
            for key, value in vars(args).items()
            if key not in {"print_config"}
        },
        "precedence": ["command_line", "shell_environment", "env_file", "profile_defaults"],
    }


def configure_logging(output: Path) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("swico_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@dataclasses.dataclass(frozen=True)
class RunContext:
    root: Path
    output: Path
    run_id: str
    resumed: bool
    exact_output_override: bool
    invocation_fingerprint: str


def _safe_run_token(value: str, fallback: str = "run") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (token or fallback)[:80]


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def invocation_configuration(args: argparse.Namespace, profile: Profile) -> dict[str, Any]:
    excluded = {
        "output",
        "output_root",
        "run_mode",
        "run_id",
        "run_label",
        "create_latest_links",
        "overwrite_output",
        "print_config",
    }
    return {
        "script_version": SCRIPT_VERSION,
        "profile": dataclasses.asdict(profile),
        "arguments": {
            key: _json_safe(value)
            for key, value in vars(args).items()
            if key not in excluded
        },
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_is_complete(run_dir: Path) -> bool:
    state = _read_json(run_dir / "run_state.json") or {}
    return bool(state.get("final_evaluation_complete") and state.get("completed_at"))


def _latest_compatible_incomplete_run(runs_dir: Path, invocation_fingerprint: str) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        if _run_is_complete(candidate):
            continue
        manifest = _read_json(candidate / "run_manifest.json") or {}
        if manifest.get("invocation_fingerprint") == invocation_fingerprint:
            return candidate
    return None


def _allocate_unique_timestamped_run(runs_dir: Path, profile_name: str, label: str | None) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [timestamp, _safe_run_token(profile_name, "profile")]
    if label:
        parts.append(_safe_run_token(label, "experiment"))
    base = "_".join(parts)
    for suffix in range(1000):
        name = base if suffix == 0 else f"{base}-{suffix:02d}"
        candidate = runs_dir / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a unique timestamped run directory")


def resolve_run_context(args: argparse.Namespace, profile: Profile) -> RunContext:
    invocation = invocation_configuration(args, profile)
    fingerprint = _payload_fingerprint(invocation)

    if args.output is not None:
        output = args.output.expanduser().resolve()
        return RunContext(
            root=output.parent,
            output=output,
            run_id=output.name,
            resumed=output.exists() and not _run_is_complete(output),
            exact_output_override=True,
            invocation_fingerprint=fingerprint,
        )

    root = args.output_root.expanduser().resolve()
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        run_id = _safe_run_token(args.run_id)
        output = runs_dir / run_id
        resumed = output.exists()
        output.mkdir(parents=True, exist_ok=True)
        return RunContext(root, output, run_id, resumed, False, fingerprint)

    if args.resume and args.run_mode in {"auto", "resume-latest"}:
        compatible = _latest_compatible_incomplete_run(runs_dir, fingerprint)
        if compatible is not None:
            return RunContext(root, compatible, compatible.name, True, False, fingerprint)
        if args.run_mode == "resume-latest":
            raise RuntimeError(
                "No compatible incomplete run exists under "
                f"{runs_dir}. Use SWICO_RUN_MODE=new to start a new run."
            )

    output = _allocate_unique_timestamped_run(runs_dir, profile.name, args.run_label)
    return RunContext(root, output, output.name, False, False, fingerprint)


def _replace_relative_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.parent / f".{link.name}.tmp-{os.getpid()}"
    with contextlib.suppress(FileNotFoundError):
        temporary.unlink()
    relative_target = os.path.relpath(target, start=link.parent)
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link)


def update_run_links(context: RunContext, completed: bool = False) -> None:
    if context.exact_output_override:
        return
    context.root.mkdir(parents=True, exist_ok=True)
    (context.root / "LATEST_RUN.txt").write_text(
        str(context.output) + "\n", encoding="utf-8"
    )
    with contextlib.suppress(OSError):
        _replace_relative_symlink(context.root / "latest", context.output)
    if completed:
        (context.root / "LATEST_COMPLETED_RUN.txt").write_text(
            str(context.output) + "\n", encoding="utf-8"
        )
        with contextlib.suppress(OSError):
            _replace_relative_symlink(context.root / "latest-completed", context.output)
        incomplete_pointer = context.root / "LATEST_INCOMPLETE_RUN.txt"
        with contextlib.suppress(OSError):
            if incomplete_pointer.read_text(encoding="utf-8").strip() == str(context.output):
                incomplete_pointer.unlink()
        incomplete_link = context.root / "latest-incomplete"
        with contextlib.suppress(OSError):
            if incomplete_link.is_symlink() and incomplete_link.resolve() == context.output.resolve():
                incomplete_link.unlink()
    else:
        (context.root / "LATEST_INCOMPLETE_RUN.txt").write_text(
            str(context.output) + "\n", encoding="utf-8"
        )
        with contextlib.suppress(OSError):
            _replace_relative_symlink(context.root / "latest-incomplete", context.output)


def stable_hash(text: str, seed: int = 0) -> str:
    return hashlib.sha1(f"{seed}\0{text}".encode("utf-8", errors="ignore")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def normalize_visible_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def split_sentences(value: str) -> list[str]:
    text = normalize_visible_text(value)
    if not text:
        return []
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()]


def discover_data_file(explicit: Path | None, work_dir: Path, logger: logging.Logger) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("data.csv"),
            Path("combined_deduplicated_dataset.csv"),
            Path("train_augmented.csv"),
            Path("train_augmented .csv"),
            Path("dataset.zip"),
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() != ".zip":
            return candidate.resolve()
        extract_dir = work_dir / "extracted_dataset"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(candidate) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV file exists inside {candidate}")
            member = max(members, key=lambda name: archive.getinfo(name).file_size)
            target = extract_dir / "data.csv"
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            logger.info("Extracted dataset member %r to %s", member, target)
            return target.resolve()

    raise FileNotFoundError(
        "No dataset found. Expected data.csv, train_augmented.csv, or dataset.zip in the repository."
    )


def strip_repeated_boilerplate(
    positives: Sequence[str], threshold: int
) -> tuple[list[str], dict[str, Any]]:
    sentence_frequency: Counter[str] = Counter()
    sentence_lists: list[list[str]] = []
    for positive in positives:
        sentences = split_sentences(positive)
        sentence_lists.append(sentences)
        sentence_frequency.update({normalize_text(sentence) for sentence in sentences})

    cleaned: list[str] = []
    changed = 0
    removed_sentences = 0
    fallback_original = 0
    for original, sentences in zip(positives, sentence_lists, strict=True):
        retained = [
            sentence
            for sentence in sentences
            if sentence_frequency[normalize_text(sentence)] < threshold
        ]
        if not retained:
            fallback_original += 1
            retained = sentences
        value = " ".join(retained).strip() or normalize_visible_text(original)
        if value != normalize_visible_text(original):
            changed += 1
            removed_sentences += max(0, len(sentences) - len(retained))
        cleaned.append(value)

    top_removed = [
        {"sentence": sentence, "frequency": count}
        for sentence, count in sentence_frequency.most_common(25)
        if count >= threshold
    ]
    return cleaned, {
        "threshold": threshold,
        "rows_changed": changed,
        "sentences_removed": removed_sentences,
        "rows_falling_back_to_original": fallback_original,
        "top_repeated_sentences": top_removed,
    }


def assign_components(frame: pd.DataFrame) -> pd.DataFrame:
    union_find = UnionFind()
    for query_norm, positive_norm in frame[["query_norm", "positive_norm"]].itertuples(index=False):
        union_find.union(f"q:{query_norm}", f"p:{positive_norm}")
    output = frame.copy()
    output["component"] = [union_find.find(f"q:{value}") for value in output["query_norm"]]
    output["component"] = output["component"].map(lambda value: stable_hash(value)[:20])
    return output


def component_split(
    frame: pd.DataFrame,
    seed: int,
    fractions: Mapping[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    fractions = fractions or {"train": 0.80, "validation": 0.10, "test": 0.10}
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.0")

    groups = [(str(component), indices.tolist()) for component, indices in frame.groupby("component").groups.items()]
    groups.sort(key=lambda item: stable_hash(item[0], seed))
    total_rows = len(frame)
    targets = {name: total_rows * fraction for name, fraction in fractions.items()}
    assigned: dict[str, list[int]] = {name: [] for name in fractions}
    counts = {name: 0 for name in fractions}

    for _, indices in groups:
        deficits = {
            name: (targets[name] - counts[name]) / max(targets[name], 1.0)
            for name in fractions
        }
        destination = max(deficits, key=deficits.get)
        assigned[destination].extend(indices)
        counts[destination] += len(indices)

    return {
        name: frame.loc[indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for name, indices in assigned.items()
    }


def sample_complete_components(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    groups = [(str(component), group) for component, group in frame.groupby("component", sort=False)]
    groups.sort(key=lambda item: stable_hash(item[0], seed))
    selected: list[pd.DataFrame] = []
    row_count = 0
    for _, group in groups:
        if row_count >= max_rows:
            break
        selected.append(group)
        row_count += len(group)
    return pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def verify_split_isolation(splits: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    names = list(splits)
    overlaps: dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for column in ("component", "query_norm", "positive_norm"):
                key = f"{left}_{right}_{column}_overlap"
                overlaps[key] = len(set(splits[left][column]) & set(splits[right][column]))
    return overlaps


def prepare_dataset(
    data_path: Path,
    prepared_dir: Path,
    boilerplate_threshold: int,
    seed: int,
    max_train_rows: int | None,
    split_fractions: Mapping[str, float],
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(data_path)
    cache_key = {
        "script_version": SCRIPT_VERSION,
        "source_sha256": source_hash,
        "boilerplate_threshold": boilerplate_threshold,
        "seed": seed,
        "max_train_rows": max_train_rows,
        "split_fractions": dict(split_fractions),
    }
    cache_meta_path = prepared_dir / "metadata.json"
    split_paths = {
        "train": prepared_dir / "train.csv.gz",
        "validation": prepared_dir / "validation.csv.gz",
        "test": prepared_dir / "test.csv.gz",
    }

    if cache_meta_path.exists() and all(path.exists() for path in split_paths.values()):
        cached_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if cached_meta.get("cache_key") == cache_key:
            logger.info("Using cached prepared dataset from %s", prepared_dir)
            splits = {name: pd.read_csv(path, compression="gzip") for name, path in split_paths.items()}
            return splits, cached_meta

    logger.info("Reading dataset: %s", data_path)
    raw = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    required = {"query", "positive"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    raw = raw[["query", "positive"]].copy()
    raw["query"] = raw["query"].map(normalize_visible_text)
    raw["positive"] = raw["positive"].map(normalize_visible_text)
    original_rows = len(raw)
    raw = raw[(raw["query"] != "") & (raw["positive"] != "")].reset_index(drop=True)
    nonempty_rows = len(raw)

    cleaned_positives, boilerplate_report = strip_repeated_boilerplate(
        raw["positive"].tolist(), threshold=boilerplate_threshold
    )
    raw["positive_original"] = raw["positive"]
    raw["positive"] = cleaned_positives
    raw["query_norm"] = raw["query"].map(normalize_text)
    raw["positive_norm"] = raw["positive"].map(normalize_text)
    before_dedup = len(raw)
    raw = raw.drop_duplicates(subset=["query_norm", "positive_norm"]).reset_index(drop=True)
    after_dedup = len(raw)
    raw["row_uid"] = [
        stable_hash(f"{query_norm}\0{positive_norm}")
        for query_norm, positive_norm in raw[["query_norm", "positive_norm"]].itertuples(index=False)
    ]
    raw = assign_components(raw)

    full_splits = component_split(raw, seed=seed, fractions=split_fractions)
    full_train_rows = len(full_splits["train"])
    full_splits["train"] = sample_complete_components(full_splits["train"], max_train_rows, seed)
    overlaps = verify_split_isolation(full_splits)
    if any(value != 0 for value in overlaps.values()):
        raise RuntimeError(f"Leakage-resistant split failed: {overlaps}")

    for name, frame in full_splits.items():
        frame.to_csv(split_paths[name], index=False, compression="gzip", quoting=csv.QUOTE_MINIMAL)

    metadata: dict[str, Any] = {
        "cache_key": cache_key,
        "source": str(data_path),
        "original_rows": original_rows,
        "nonempty_rows": nonempty_rows,
        "rows_before_exact_pair_dedup": before_dedup,
        "rows_after_exact_pair_dedup": after_dedup,
        "exact_pairs_removed": before_dedup - after_dedup,
        "full_train_rows_before_profile_cap": full_train_rows,
        "split_rows": {name: len(frame) for name, frame in full_splits.items()},
        "unique_queries": int(raw["query_norm"].nunique()),
        "unique_passages": int(raw["positive_norm"].nunique()),
        "components": int(raw["component"].nunique()),
        "boilerplate": boilerplate_report,
        "split_isolation": overlaps,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(cache_meta_path, metadata)
    logger.info("Prepared rows: train=%d validation=%d test=%d", *(len(full_splits[k]) for k in ("train", "validation", "test")))
    return full_splits, metadata


def profile_with_overrides(args: argparse.Namespace) -> Profile:
    base = PROFILES[args.profile]
    return dataclasses.replace(
        base,
        max_train_rows=args.max_train_rows if args.max_train_rows is not None else base.max_train_rows,
        stage1_epochs=args.stage1_epochs if args.stage1_epochs is not None else base.stage1_epochs,
        stage2_epochs=args.stage2_epochs if args.stage2_epochs is not None else base.stage2_epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        max_seq_length=args.max_seq_length if args.max_seq_length is not None else base.max_seq_length,
        trainable_layers=args.trainable_layers if args.trainable_layers is not None else base.trainable_layers,
        eval_queries=args.eval_queries if args.eval_queries is not None else base.eval_queries,
        eval_corpus=args.eval_corpus if args.eval_corpus is not None else base.eval_corpus,
        mining_top_k=args.mining_top_k if args.mining_top_k is not None else base.mining_top_k,
        hard_negatives=args.hard_negatives if args.hard_negatives is not None else base.hard_negatives,
        matryoshka_dims=args.matryoshka_dims if args.matryoshka_dims is not None else base.matryoshka_dims,
    )


def validate_profile(profile: Profile, args: argparse.Namespace) -> None:
    if profile.max_train_rows is not None and profile.max_train_rows < 100:
        raise ValueError("max_train_rows must be at least 100 when a cap is used")
    if profile.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for in-batch-negative training")
    if not 32 <= profile.max_seq_length <= 512:
        raise ValueError("max_seq_length must be between 32 and 512")
    if profile.trainable_layers < 1:
        raise ValueError("trainable_layers must be at least 1")
    if profile.stage1_epochs <= 0 or profile.stage2_epochs < 0:
        raise ValueError("stage1_epochs must be positive and stage2_epochs cannot be negative")
    if args.stage1_lr <= 0 or args.stage2_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers cannot be negative")
    if args.logging_steps < 1 or args.save_total_limit < 1:
        raise ValueError("logging_steps and save_total_limit must be at least 1")
    if args.eval_batch_size < 1 or args.eval_loss_rows < 1 or args.eval_corpus_chunk_size < 1:
        raise ValueError("evaluation batch, loss-row, and corpus-chunk settings must be at least 1")
    if profile.eval_queries < 1 or profile.eval_corpus < 2:
        raise ValueError("eval_queries must be positive and eval_corpus must be at least 2")
    if profile.mining_top_k < 2:
        raise ValueError("mining_top_k must be at least 2")
    if args.min_hard_negative_margin < 0:
        raise ValueError("min_hard_negative_margin cannot be negative")
    if args.mining_batch_size < 1 or args.mining_chunk_size < 1:
        raise ValueError("mining batch and chunk sizes must be positive")
    if args.hnsw_m < 2 or args.hnsw_ef_construction < 2 or args.hnsw_ef_search_factor < 1:
        raise ValueError("HNSW settings are outside their safe ranges")
    if not 0.0 <= args.negative_max_token_jaccard <= 1.0:
        raise ValueError("negative_max_token_jaccard must be in [0, 1]")
    if args.loss_scale <= 0:
        raise ValueError("loss_scale must be positive")
    if not profile.matryoshka_dims or any(value <= 0 for value in profile.matryoshka_dims):
        raise ValueError("matryoshka_dims must contain positive dimensions")
    if len(set(profile.matryoshka_dims)) != len(profile.matryoshka_dims):
        raise ValueError("matryoshka_dims cannot contain duplicates")
    if tuple(sorted(profile.matryoshka_dims, reverse=True)) != profile.matryoshka_dims:
        raise ValueError("matryoshka_dims must be ordered from largest to smallest")
    if len(args.matryoshka_weights) < len(profile.matryoshka_dims):
        raise ValueError("matryoshka_weights must provide at least one weight per configured dimension")
    if any(weight <= 0 for weight in args.matryoshka_weights[: len(profile.matryoshka_dims)]):
        raise ValueError("matryoshka_weights must be positive")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    if args.early_stopping_threshold < 0:
        raise ValueError("early_stopping_threshold cannot be negative")
    if args.early_stopping and not args.load_best_model_at_end:
        raise ValueError("early stopping requires load_best_model_at_end=true")
    fractions = (
        args.train_split_fraction,
        args.validation_split_fraction,
        args.test_split_fraction,
    )
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("all split fractions must be between 0 and 1")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("train, validation, and test split fractions must sum to 1.0")
    if args.min_available_memory_gib <= 0 or args.min_free_disk_gib <= 0:
        raise ValueError("preflight memory and disk limits must be positive")
    if args.warn_free_disk_gib < args.min_free_disk_gib:
        raise ValueError("warn_free_disk_gib cannot be lower than min_free_disk_gib")
    if args.run_id and args.run_mode == "resume-latest":
        raise ValueError("run_id cannot be combined with run_mode=resume-latest")
    if args.autotune_max_inference_batch < 1 or args.autotune_max_train_batch < 2:
        raise ValueError("autotune batch limits are outside their safe ranges")
    if args.autotune_memory_reserve_gib <= 0:
        raise ValueError("autotune_memory_reserve_gib must be positive")
    if not 0.1 <= args.autotune_memory_utilization <= 0.95:
        raise ValueError("autotune_memory_utilization must be between 0.1 and 0.95")
    if args.autotune_train_batch and args.autotune_mode != "aggressive":
        raise ValueError("autotune_train_batch=true requires autotune_mode=aggressive")
    if args.memory_guard_interval_steps < 1:
        raise ValueError("memory_guard_interval_steps must be at least 1")
    if args.emergency_available_memory_gib <= 0:
        raise ValueError("emergency_available_memory_gib must be positive")
    if args.max_process_rss_gib is not None and args.max_process_rss_gib <= 0:
        raise ValueError("max_process_rss_gib must be positive or auto")
    if args.emergency_available_memory_gib >= args.min_available_memory_gib:
        raise ValueError(
            "emergency_available_memory_gib must be lower than min_available_memory_gib"
        )
    if args.baseline_protection and args.skip_base_eval:
        raise ValueError("baseline_protection requires skip_base_eval=false")
    if args.min_validation_gain < 0:
        raise ValueError("min_validation_gain cannot be negative")


def configure_runtime(threads: int, seed: int, logger: logging.Logger) -> int:
    threads = max(1, min(threads, os.cpu_count() or threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = True
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("high")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(
        "CPU runtime configured: torch_threads=%d interop_threads=%d logical_cpus=%d",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        os.cpu_count() or 0,
    )
    if faiss is not None:
        with contextlib.suppress(Exception):
            faiss.omp_set_num_threads(threads)
    return threads


def system_report(output: Path, threads: int) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(output.resolve())
    cpu_flags = ""
    with contextlib.suppress(OSError, StopIteration):
        cpu_flags = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.lower().startswith("flags")
        )
    soft_open, hard_open = resource.getrlimit(resource.RLIMIT_NOFILE)
    report = {
        "script_version": SCRIPT_VERSION,
        "python": sys.version,
        "torch": torch.__version__,
        "logical_cpus": os.cpu_count(),
        "configured_threads": threads,
        "memory_total_gib": round(memory.total / 2**30, 2),
        "memory_available_gib": round(memory.available / 2**30, 2),
        "disk_total_gib": round(disk.total / 2**30, 2),
        "disk_free_gib": round(disk.free / 2**30, 2),
        "open_files_soft_limit": soft_open,
        "open_files_hard_limit": hard_open,
        "cpu_has_avx2": "avx2" in cpu_flags.split(),
        "cpu_has_avx512": "avx512f" in cpu_flags.split(),
        "cpu_has_bf16": any(flag in cpu_flags.split() for flag in ("avx512_bf16", "amx_bf16")),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(output / "system.json", report)
    return report


def preflight(
    report: Mapping[str, Any],
    min_available_memory_gib: float,
    min_free_disk_gib: float,
    warn_free_disk_gib: float,
    logger: logging.Logger,
) -> None:
    available_memory = float(report["memory_available_gib"])
    if available_memory < min_available_memory_gib:
        raise RuntimeError(
            f"Only {available_memory:.2f} GiB RAM is available; "
            f"the configured minimum is {min_available_memory_gib:.2f} GiB."
        )
    free_disk = float(report["disk_free_gib"])
    if free_disk < min_free_disk_gib:
        raise RuntimeError(
            f"Only {free_disk:.2f} GiB disk is free; "
            f"the configured minimum is {min_free_disk_gib:.2f} GiB."
        )
    if free_disk < warn_free_disk_gib:
        logger.warning(
            "Only %.2f GiB disk is free. Keep-checkpoints and extra exports should remain disabled.",
            free_disk,
        )
    if not bool(report["cpu_has_avx2"]):
        logger.warning("AVX2 was not detected; CPU training will be substantially less efficient.")


def apply_adaptive_resource_plan(
    profile: Profile,
    args: argparse.Namespace,
    report: Mapping[str, Any],
    output: Path,
    logger: logging.Logger,
) -> tuple[Profile, dict[str, Any]]:
    """Increase only resource/throughput settings that are safe by default.

    Safe mode never changes epochs, learning rates, sequence length, trainable
    layers, loss settings, mining semantics, or the physical training batch.
    Aggressive mode may raise the physical training batch only when explicitly
    enabled, which can change optimization behavior and therefore is opt-in.
    """

    persisted_path = output / "autotune.json"
    if persisted_path.exists():
        persisted = _read_json(persisted_path)
        if persisted and isinstance(persisted.get("resolved"), dict):
            resolved = persisted["resolved"]
            args.eval_batch_size = int(resolved["eval_batch_size"])
            args.mining_batch_size = int(resolved["mining_batch_size"])
            args.mining_chunk_size = int(resolved["mining_chunk_size"])
            args.eval_corpus_chunk_size = int(resolved["eval_corpus_chunk_size"])
            profile = dataclasses.replace(profile, batch_size=int(resolved["train_batch_size"]))
            logger.info("Reusing persisted adaptive resource plan from %s", persisted_path)
            return profile, persisted

    before = {
        "train_batch_size": profile.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "mining_batch_size": args.mining_batch_size,
        "mining_chunk_size": args.mining_chunk_size,
        "eval_corpus_chunk_size": args.eval_corpus_chunk_size,
        "threads": args.threads,
        "dataloader_num_workers": args.dataloader_num_workers,
    }
    selected = dict(before)
    reasons: list[str] = []
    quality_sensitive_changes: list[str] = []

    if args.autotune:
        available = float(report["memory_available_gib"])
        reserve = max(
            float(args.autotune_memory_reserve_gib),
            float(args.min_available_memory_gib),
        )
        usable_budget = max(0.0, available - reserve) * float(args.autotune_memory_utilization)

        if usable_budget >= 12.0:
            inference_cap = 128
        elif usable_budget >= 8.0:
            inference_cap = 96
        elif usable_budget >= 4.0:
            inference_cap = 64
        elif usable_budget >= 2.0:
            inference_cap = 32
        else:
            inference_cap = 16
        inference_cap = max(1, min(inference_cap, args.autotune_max_inference_batch))

        selected["eval_batch_size"] = min(
            args.autotune_max_inference_batch,
            inference_cap,
        )
        selected["mining_batch_size"] = min(
            args.autotune_max_inference_batch,
            inference_cap,
        )
        selected["mining_chunk_size"] = min(
            4096,
            max(256, selected["mining_batch_size"] * 8),
        )
        selected["eval_corpus_chunk_size"] = max(
            before["eval_corpus_chunk_size"],
            min(max(profile.eval_corpus, before["eval_corpus_chunk_size"]), 20_000),
        )
        reasons.append(
            f"safe inference/mining scaling used {usable_budget:.2f} GiB adaptive budget "
            f"while reserving at least {reserve:.2f} GiB"
        )

        if args.autotune_mode == "aggressive" and args.autotune_train_batch:
            if usable_budget >= 12.0:
                train_cap = 96
            elif usable_budget >= 8.0:
                train_cap = 80
            elif usable_budget >= 5.0:
                train_cap = 64
            elif usable_budget >= 3.0:
                train_cap = 48
            else:
                train_cap = 32
            train_cap = max(2, min(train_cap, args.autotune_max_train_batch))
            if train_cap > profile.batch_size:
                selected["train_batch_size"] = train_cap
                quality_sensitive_changes.append("train_batch_size")
                reasons.append(
                    "aggressive mode raised the physical training batch; validation gates and "
                    "baseline protection remain active, but identical accuracy cannot be guaranteed"
                )

    args.eval_batch_size = int(selected["eval_batch_size"])
    args.mining_batch_size = int(selected["mining_batch_size"])
    args.mining_chunk_size = int(selected["mining_chunk_size"])
    args.eval_corpus_chunk_size = int(selected["eval_corpus_chunk_size"])
    profile = dataclasses.replace(profile, batch_size=int(selected["train_batch_size"]))

    payload = {
        "enabled": bool(args.autotune),
        "mode": args.autotune_mode,
        "before": before,
        "resolved": selected,
        "quality_sensitive_changes": quality_sensitive_changes,
        "quality_preserving_safe_mode": not quality_sensitive_changes,
        "reasons": reasons,
        "memory": {
            "available_gib": float(report["memory_available_gib"]),
            "reserve_gib": float(args.autotune_memory_reserve_gib),
            "utilization": float(args.autotune_memory_utilization),
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(persisted_path, payload)
    logger.info(
        "Adaptive plan: train_batch=%d eval_batch=%d mining_batch=%d mining_chunk=%d mode=%s",
        profile.batch_size,
        args.eval_batch_size,
        args.mining_batch_size,
        args.mining_chunk_size,
        args.autotune_mode if args.autotune else "disabled",
    )
    return profile, payload


class ResourceGuardCallback(TrainerCallback):
    """Request a checkpoint and graceful stop before system memory is exhausted."""

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
        self.last_snapshot: dict[str, Any] | None = None

    def _check(self, state: Any, control: Any, event: str) -> Any:
        if self.triggered:
            control.should_save = True
            control.should_training_stop = True
            return control
        global_step = int(getattr(state, "global_step", 0) or 0)
        if event == "step" and global_step % self.interval_steps != 0:
            return control

        memory = psutil.virtual_memory()
        rss_gib = self.process.memory_info().rss / 2**30
        available_gib = memory.available / 2**30
        snapshot = {
            "event": event,
            "global_step": global_step,
            "available_memory_gib": round(available_gib, 3),
            "process_rss_gib": round(rss_gib, 3),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.last_snapshot = snapshot
        low_available = available_gib < self.emergency_available_memory_gib
        rss_exceeded = (
            self.max_process_rss_gib is not None and rss_gib > self.max_process_rss_gib
        )
        if low_available or rss_exceeded:
            self.triggered = True
            snapshot["reason"] = (
                "low_available_memory" if low_available else "process_rss_limit_exceeded"
            )
            atomic_write_json(self.output_path, snapshot)
            self.logger.error(
                "Memory guard requested a safe checkpoint/stop: available=%.2f GiB rss=%.2f GiB reason=%s",
                available_gib,
                rss_gib,
                snapshot["reason"],
            )
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._check(state, control, "step")

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._check(state, control, "evaluate")


def transformer_module(model: SentenceTransformer) -> Any:
    for module in model._modules.values():
        if hasattr(module, "auto_model"):
            return module
    raise RuntimeError("Could not locate the transformer module inside SentenceTransformer")


def encoder_layers(auto_model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = [
        ("encoder", "layer"),
        ("transformer", "layer"),
        ("bert", "encoder", "layer"),
        ("roberta", "encoder", "layer"),
        ("xlm_roberta", "encoder", "layer"),
    ]
    for path in candidates:
        current: Any = auto_model
        valid = True
        for part in path:
            if not hasattr(current, part):
                valid = False
                break
            current = getattr(current, part)
        if valid and isinstance(current, (torch.nn.ModuleList, list, tuple)):
            return current
    raise RuntimeError("Unsupported transformer layout: encoder layers could not be found")


def freeze_for_partial_training(model: SentenceTransformer, trainable_layers: int, logger: logging.Logger) -> dict[str, Any]:
    module = transformer_module(model)
    auto_model = module.auto_model
    layers = encoder_layers(auto_model)
    if not 1 <= trainable_layers <= len(layers):
        raise ValueError(f"trainable_layers must be between 1 and {len(layers)}")

    for parameter in auto_model.parameters():
        parameter.requires_grad = False
    for layer in layers[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    # Preserve trainability of any task-specific modules after the base transformer.
    modules = list(model._modules.values())
    transformer_index = modules.index(module)
    for extra_module in modules[transformer_index + 1 :]:
        for parameter in extra_module.parameters():
            parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    summary = {
        "encoder_layers": len(layers),
        "trainable_encoder_layers": trainable_layers,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round(100.0 * trainable / total, 4),
    }
    logger.info(
        "Partial fine-tuning: last %d/%d layers; trainable=%s/%s (%.3f%%)",
        trainable_layers,
        len(layers),
        f"{trainable:,}",
        f"{total:,}",
        summary["trainable_percent"],
    )
    return summary


def load_model(
    source: str | Path,
    max_seq_length: int,
    trainable_layers: int,
    offline: bool,
    logger: logging.Logger,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    source_value = str(source)
    model = SentenceTransformer(
        source_value,
        device="cpu",
        local_files_only=offline,
    )
    model.max_seq_length = max_seq_length
    model.to("cpu")
    summary = freeze_for_partial_training(model, trainable_layers, logger)
    return model, summary


def formatted_dataset(frame: pd.DataFrame, include_negative: bool = False) -> Dataset:
    payload: dict[str, list[str]] = {
        "anchor": [QUERY_PREFIX + value for value in frame["query"].tolist()],
        "positive": [PASSAGE_PREFIX + value for value in frame["positive"].tolist()],
    }
    if include_negative:
        if "negative" not in frame.columns:
            raise ValueError("negative column is required for hard-negative training")
        payload["negative"] = [PASSAGE_PREFIX + value for value in frame["negative"].tolist()]
    return Dataset.from_dict(payload)


def build_eval_bundle(
    name: str,
    eval_frame: pd.DataFrame,
    document_pool: pd.DataFrame,
    max_queries: int,
    max_corpus: int,
    seed: int,
    batch_size: int,
    corpus_chunk_size: int,
    output_path: Path,
) -> EvalBundle:
    query_groups = [(query_norm, group) for query_norm, group in eval_frame.groupby("query_norm", sort=False)]
    query_groups.sort(key=lambda item: stable_hash(item[0], seed))
    query_groups = query_groups[:max_queries]

    selected_queries: dict[str, str] = {}
    selected_relevant_norms: dict[str, set[str]] = {}
    required_passages: dict[str, str] = {}
    for index, (query_norm, group) in enumerate(query_groups):
        query_id = f"q{index:06d}"
        selected_queries[query_id] = QUERY_PREFIX + str(group.iloc[0]["query"])
        relevant_norms = set(group["positive_norm"])
        selected_relevant_norms[query_id] = relevant_norms
        for row in group.itertuples(index=False):
            required_passages[str(row.positive_norm)] = str(row.positive)

    unique_pool = (
        document_pool[["positive_norm", "positive"]]
        .drop_duplicates(subset=["positive_norm"])
        .sort_values("positive_norm")
    )
    distractors: list[tuple[str, str]] = []
    required_norms = set(required_passages)
    for row in unique_pool.itertuples(index=False):
        if row.positive_norm not in required_norms:
            distractors.append((str(row.positive_norm), str(row.positive)))
    distractors.sort(key=lambda item: stable_hash(item[0], seed + 7))

    corpus_items: list[tuple[str, str]] = list(required_passages.items())
    remaining = max(0, max_corpus - len(corpus_items))
    corpus_items.extend(distractors[:remaining])
    corpus_items = corpus_items[:max_corpus]

    norm_to_doc_id: dict[str, str] = {}
    corpus: dict[str, str] = {}
    corpus_order: list[str] = []
    for index, (positive_norm, positive) in enumerate(corpus_items):
        doc_id = f"d{index:06d}"
        norm_to_doc_id[positive_norm] = doc_id
        corpus[doc_id] = PASSAGE_PREFIX + positive
        corpus_order.append(doc_id)

    relevant_docs: dict[str, set[str]] = {}
    filtered_queries: dict[str, str] = {}
    query_order: list[str] = []
    for query_id, query in selected_queries.items():
        docs = {norm_to_doc_id[value] for value in selected_relevant_norms[query_id] if value in norm_to_doc_id}
        if docs:
            filtered_queries[query_id] = query
            relevant_docs[query_id] = docs
            query_order.append(query_id)

    evaluator = InformationRetrievalEvaluator(
        queries=filtered_queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        corpus_chunk_size=min(corpus_chunk_size, max(1, len(corpus))),
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[100],
        show_progress_bar=False,
        batch_size=batch_size,
        name=name,
        write_csv=True,
        main_score_function="cosine",
    )
    output_path.mkdir(parents=True, exist_ok=True)
    return EvalBundle(
        name=name,
        queries=filtered_queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        query_order=query_order,
        corpus_order=corpus_order,
        evaluator=evaluator,
    )


def evaluate_bundle(
    model: SentenceTransformer,
    bundle: EvalBundle,
    output_path: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    started = time.perf_counter()
    metrics = bundle.evaluator(model, output_path=str(output_path))
    duration = time.perf_counter() - started
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics["evaluation_seconds"] = duration
    logger.info(
        "%s evaluation complete: primary=%s value=%.6f queries=%d corpus=%d seconds=%.2f",
        bundle.name,
        bundle.evaluator.primary_metric,
        metrics.get(bundle.evaluator.primary_metric, float("nan")),
        len(bundle.queries),
        len(bundle.corpus),
        duration,
    )
    return metrics


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.lower()))
    right_tokens = set(re.findall(r"\w+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def random_negative_indices(
    components: Sequence[str],
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    all_indices = list(range(len(components)))
    output: list[int] = []
    for index, component in enumerate(components):
        candidate = index
        for _ in range(100):
            candidate = rng.choice(all_indices)
            if candidate != index and components[candidate] != component:
                break
        output.append(candidate)
    return output


def mine_hard_negatives(
    model: SentenceTransformer,
    train_frame: pd.DataFrame,
    cache_dir: Path,
    base_model: str,
    top_k: int,
    min_margin: float,
    mining_batch_size: int,
    mining_chunk_size: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search_factor: int,
    negative_max_token_jaccard: float,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested_backend = "faiss_hnsw_inner_product" if faiss is not None else "numpy_exact_chunked"
    fingerprint_payload = {
        "script_version": SCRIPT_VERSION,
        "base_model": base_model,
        "row_uids": sorted(train_frame["row_uid"].tolist()),
        "top_k": top_k,
        "min_margin": min_margin,
        "mining_batch_size": mining_batch_size,
        "mining_chunk_size": mining_chunk_size,
        "hnsw_m": hnsw_m,
        "hnsw_ef_construction": hnsw_ef_construction,
        "hnsw_ef_search_factor": hnsw_ef_search_factor,
        "negative_max_token_jaccard": negative_max_token_jaccard,
        "seed": seed,
        "backend": requested_backend,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    csv_path = cache_dir / "hard_negatives.csv.gz"
    meta_path = cache_dir / "hard_negatives.json"
    if csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            cached = pd.read_csv(csv_path, compression="gzip")
            merged = train_frame.merge(cached[["row_uid", "negative"]], on="row_uid", how="left", validate="one_to_one")
            if merged["negative"].notna().all():
                logger.info("Using cached hard negatives from %s", csv_path)
                return merged, meta

    corpus_frame = (
        train_frame[["positive_norm", "positive", "component"]]
        .drop_duplicates(subset=["positive_norm"])
        .reset_index(drop=True)
    )
    corpus_texts = [PASSAGE_PREFIX + value for value in corpus_frame["positive"].tolist()]
    logger.info("Encoding %d unique passages for hard-negative mining", len(corpus_texts))
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=mining_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32", copy=False)
    positive_index = {value: index for index, value in enumerate(corpus_frame["positive_norm"])}
    corpus_components = corpus_frame["component"].astype(str).tolist()
    corpus_norms = corpus_frame["positive_norm"].astype(str).tolist()
    corpus_values = corpus_frame["positive"].astype(str).tolist()

    if faiss is not None:
        index = faiss.IndexHNSWFlat(
            corpus_embeddings.shape[1],
            hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = hnsw_ef_construction
        index.hnsw.efSearch = max(top_k, top_k * hnsw_ef_search_factor)
        index.add(corpus_embeddings)
        mining_backend = requested_backend
    else:
        index = None
        mining_backend = requested_backend
        logger.warning("faiss-cpu is unavailable; using slower chunked NumPy hard-negative search")

    negatives: list[str] = []
    mining_modes: Counter[str] = Counter()
    margins: list[float] = []
    chunk_size = mining_chunk_size
    fallback_indices = random_negative_indices(train_frame["component"].astype(str).tolist(), seed)

    for start in range(0, len(train_frame), chunk_size):
        stop = min(len(train_frame), start + chunk_size)
        chunk = train_frame.iloc[start:stop]
        query_embeddings = model.encode(
            [QUERY_PREFIX + value for value in chunk["query"].tolist()],
            batch_size=mining_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32", copy=False)

        if index is not None:
            scores, indices = index.search(query_embeddings, min(top_k, len(corpus_frame)))
        else:
            scores, indices = search_top_k(
                query_embeddings,
                corpus_embeddings,
                k=min(top_k, len(corpus_frame)),
            )

        for local_index, row in enumerate(chunk.itertuples(index=False)):
            global_index = start + local_index
            pos_idx = positive_index[str(row.positive_norm)]
            positive_score = float(np.dot(query_embeddings[local_index], corpus_embeddings[pos_idx]))
            selected: int | None = None
            selected_score: float | None = None

            for candidate_score, candidate_index in zip(scores[local_index], indices[local_index], strict=True):
                candidate_index = int(candidate_index)
                if candidate_index < 0 or candidate_index == pos_idx:
                    continue
                if corpus_components[candidate_index] == str(row.component):
                    continue
                if token_jaccard(str(row.positive_norm), corpus_norms[candidate_index]) >= negative_max_token_jaccard:
                    continue
                if float(candidate_score) <= positive_score - min_margin:
                    selected = candidate_index
                    selected_score = float(candidate_score)
                    mining_modes["semi_hard"] += 1
                    break

            if selected is None:
                # Map the deterministic row-level fallback to a corpus passage with a different component.
                fallback_row = fallback_indices[global_index]
                candidate_norm = str(train_frame.iloc[fallback_row]["positive_norm"])
                selected = positive_index[candidate_norm]
                if selected == pos_idx or corpus_components[selected] == str(row.component):
                    selected = None
                    for offset in range(1, len(corpus_frame) + 1):
                        candidate_index = (pos_idx + offset) % len(corpus_frame)
                        if corpus_components[candidate_index] != str(row.component):
                            selected = candidate_index
                            break
                    if selected is None:
                        raise RuntimeError("Hard-negative mining requires at least two disconnected components")
                selected_score = float(np.dot(query_embeddings[local_index], corpus_embeddings[selected]))
                mining_modes["random_fallback"] += 1

            negatives.append(corpus_values[selected])
            margins.append(positive_score - float(selected_score))

        logger.info("Hard-negative mining progress: %d/%d rows", stop, len(train_frame))

    mined = train_frame.copy()
    mined["negative"] = negatives
    cache_frame = mined[["row_uid", "negative"]]
    cache_frame.to_csv(csv_path, index=False, compression="gzip")
    metadata = {
        "fingerprint": fingerprint,
        "rows": len(mined),
        "unique_corpus_passages": len(corpus_frame),
        "mining_backend": mining_backend,
        "mining_modes": dict(mining_modes),
        "mean_positive_minus_negative_margin": float(np.mean(margins)) if margins else None,
        "median_positive_minus_negative_margin": float(np.median(margins)) if margins else None,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(meta_path, metadata)
    del corpus_embeddings
    gc.collect()
    return mined, metadata


def make_loss(
    model: SentenceTransformer,
    dims: Sequence[int],
    loss_scale: float,
    matryoshka_weights: Sequence[float],
    matryoshka_dims_per_step: int,
) -> torch.nn.Module:
    base_loss = MultipleNegativesRankingLoss(model=model, scale=loss_scale)
    embedding_dimension = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if embedding_dimension is None:
        raise RuntimeError("The model did not report an embedding dimension")
    valid_dims = [dimension for dimension in dims if dimension <= embedding_dimension]
    if len(valid_dims) <= 1:
        return base_loss
    valid_weights = list(matryoshka_weights[: len(valid_dims)])
    return MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=valid_dims,
        matryoshka_weights=valid_weights,
        n_dims_per_step=matryoshka_dims_per_step,
    )


def _installed_transformers_major() -> int:
    try:
        return Version(package_version("transformers")).major
    except (PackageNotFoundError, ValueError):
        return 0


def make_training_args(
    output_dir: Path,
    epochs: float,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    dataloader_num_workers: int,
    dataloader_drop_last: bool,
    logging_steps: int,
    save_total_limit: int,
    primary_metric: str,
    load_best_model_at_end: bool,
    seed: int,
    use_bf16: bool,
) -> SentenceTransformerTrainingArguments:
    """Build safe arguments across supported Transformers 4.x and 5.x APIs."""

    supported = set(inspect.signature(SentenceTransformerTrainingArguments.__init__).parameters)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "fp16": False,
        "bf16": use_bf16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": False,
        "dataloader_drop_last": dataloader_drop_last,
        "save_strategy": "epoch",
        "save_total_limit": max(2 if load_best_model_at_end else 1, save_total_limit),
        "load_best_model_at_end": load_best_model_at_end,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "report_to": "none",
        "optim": "adamw_torch",
        "seed": seed,
        "data_seed": seed,
        "remove_unused_columns": True,
    }

    if load_best_model_at_end:
        kwargs["metric_for_best_model"] = primary_metric
        kwargs["greater_is_better"] = True

    if "restore_callback_states_from_checkpoint" in supported:
        kwargs["restore_callback_states_from_checkpoint"] = True
    if "save_only_model" in supported:
        kwargs["save_only_model"] = False

    if "use_cpu" in supported:
        kwargs["use_cpu"] = True
    elif "no_cuda" in supported:
        kwargs["no_cuda"] = True
    else:
        raise RuntimeError(
            "The installed Transformers TrainingArguments has neither use_cpu nor no_cuda; "
            "this release is not supported by the CPU trainer."
        )

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "The installed Transformers TrainingArguments has no evaluation-strategy field."
        )

    transformers_major = _installed_transformers_major()
    if transformers_major >= 5 and "warmup_steps" in supported:
        # Transformers 5 accepts a float in [0, 1) here as a ratio and warns
        # when the legacy warmup_ratio field is used.
        kwargs["warmup_steps"] = warmup_ratio
    elif "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in supported:
        kwargs["warmup_steps"] = warmup_ratio
    else:
        raise RuntimeError(
            "The installed Transformers TrainingArguments has no warmup_ratio or warmup_steps field."
        )

    if "save_safetensors" in supported:
        kwargs["save_safetensors"] = True

    unsupported = sorted(key for key in kwargs if key not in supported)
    if unsupported:
        try:
            import sentence_transformers as sentence_transformers_package
            import transformers as transformers_package

            versions = (
                f"sentence-transformers={getattr(sentence_transformers_package, '__version__', 'unknown')} "
                f"transformers={getattr(transformers_package, '__version__', 'unknown')}"
            )
        except Exception:
            versions = "installed package versions unavailable"
        raise RuntimeError(
            "Unsupported training arguments for the installed libraries: "
            f"{', '.join(unsupported)} ({versions})."
        )

    return SentenceTransformerTrainingArguments(**kwargs)


def train_stage(
    stage_name: str,
    model: SentenceTransformer,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    evaluator: InformationRetrievalEvaluator,
    output: Path,
    epochs: float,
    batch_size: int,
    eval_batch_size: int,
    eval_loss_rows: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    dataloader_num_workers: int,
    dataloader_drop_last: bool,
    logging_steps: int,
    save_total_limit: int,
    dims: Sequence[int],
    matryoshka_weights: Sequence[float],
    matryoshka_dims_per_step: int,
    loss_scale: float,
    include_negative: bool,
    resume: bool,
    seed: int,
    use_bf16: bool,
    early_stopping: bool,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    load_best_model_at_end: bool,
    memory_guard: bool,
    memory_guard_interval_steps: int,
    emergency_available_memory_gib: float,
    max_process_rss_gib: float | None,
    logger: logging.Logger,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    checkpoint_dir = output / "checkpoints" / stage_name
    model_dir = output / "models" / stage_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = formatted_dataset(train_frame, include_negative=include_negative)
    eval_loss_frame = eval_frame.head(min(eval_loss_rows, len(eval_frame))).copy()
    if include_negative and "negative" not in eval_loss_frame.columns:
        # Evaluation loss can safely use pairs while the stage trains on triplets.
        eval_dataset = formatted_dataset(eval_loss_frame, include_negative=False)
    else:
        eval_dataset = formatted_dataset(eval_loss_frame, include_negative=include_negative)
    loss = make_loss(
        model,
        dims=dims,
        loss_scale=loss_scale,
        matryoshka_weights=matryoshka_weights,
        matryoshka_dims_per_step=matryoshka_dims_per_step,
    )
    primary_metric = str(evaluator.primary_metric)
    training_args = make_training_args(
        output_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=dataloader_drop_last,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        primary_metric=primary_metric,
        load_best_model_at_end=load_best_model_at_end,
        seed=seed,
        use_bf16=use_bf16,
    )

    callbacks: list[Any] = []
    resource_guard: ResourceGuardCallback | None = None
    if memory_guard:
        resource_guard = ResourceGuardCallback(
            output_path=output / "reports" / f"{stage_name}_memory_guard.json",
            logger=logger,
            interval_steps=memory_guard_interval_steps,
            emergency_available_memory_gib=emergency_available_memory_gib,
            max_process_rss_gib=max_process_rss_gib,
        )
        callbacks.append(resource_guard)
    early_stopping_active = bool(early_stopping and epochs > 1.0)
    if early_stopping_active:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )
    elif early_stopping:
        logger.info(
            "%s early stopping is configured but inactive because max epochs is %.2f",
            stage_name,
            epochs,
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    resume_from = last_checkpoint if resume and last_checkpoint else None
    logger.info(
        "Starting %s: rows=%d max_epochs=%.2f batch=%d lr=%g early_stop=%s patience=%d threshold=%g metric=%s resume=%s",
        stage_name,
        len(train_frame),
        epochs,
        batch_size,
        learning_rate,
        early_stopping_active,
        early_stopping_patience,
        early_stopping_threshold,
        primary_metric,
        resume_from or "none",
    )
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume_from)
    duration = time.perf_counter() - started

    # When load_best_model_at_end is enabled, Trainer has already restored the
    # best validation checkpoint here. Saving now promotes that model, not the
    # final (possibly overfit) epoch.
    trainer.save_model(str(model_dir))
    trainer.save_state()
    completed_epochs = float(trainer.state.epoch or 0.0)
    resource_stopped = bool(resource_guard and resource_guard.triggered)
    stopped_early = bool(
        early_stopping_active
        and not resource_stopped
        and completed_epochs + 1e-6 < epochs
    )
    summary = {
        "stage": stage_name,
        "rows": len(train_frame),
        "max_epochs": epochs,
        "completed_epochs": completed_epochs,
        "stopped_early": stopped_early,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "bf16_cpu_amp": use_bf16,
        "early_stopping": {
            "enabled": early_stopping_active,
            "patience_evaluations": early_stopping_patience,
            "minimum_improvement": early_stopping_threshold,
            "metric": primary_metric,
            "greater_is_better": True,
        },
        "best_metric": (
            float(trainer.state.best_metric)
            if trainer.state.best_metric is not None
            else None
        ),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "load_best_model_at_end": load_best_model_at_end,
        "memory_guard": {
            "enabled": memory_guard,
            "triggered": bool(resource_guard and resource_guard.triggered),
            "last_snapshot": resource_guard.last_snapshot if resource_guard else None,
            "emergency_available_memory_gib": emergency_available_memory_gib,
            "max_process_rss_gib": max_process_rss_gib,
        },
        "duration_seconds": duration,
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "metrics": {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, (int, float))
        },
        "model_dir": str(model_dir),
    }
    atomic_write_json(output / "reports" / f"{stage_name}.json", summary)
    logger.info(
        "%s complete: step=%d epochs=%.3f stopped_early=%s best_%s=%s loss=%.6f seconds=%.2f",
        stage_name,
        summary["global_step"],
        completed_epochs,
        stopped_early,
        primary_metric,
        f"{summary['best_metric']:.6f}" if summary["best_metric"] is not None else "n/a",
        summary["training_loss"],
        duration,
    )
    if resource_stopped:
        raise ResourceStopRequested(
            f"{stage_name} was safely paused by the memory guard after saving state. "
            "Free memory or lower the relevant batch sizes, then rerun the same command to resume."
        )
    return model, summary


def search_top_k(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(corpus_embeddings))
    if faiss is not None:
        index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        index.add(corpus_embeddings.astype("float32", copy=False))
        return index.search(query_embeddings.astype("float32", copy=False), k)
    scores = query_embeddings @ corpus_embeddings.T
    indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(len(scores))[:, None]
    selected_scores = scores[row_indices, indices]
    order = np.argsort(-selected_scores, axis=1)
    return selected_scores[row_indices, order], indices[row_indices, order]


def detailed_retrieval_metrics(
    model: SentenceTransformer,
    bundle: EvalBundle,
    dimensions: Sequence[int],
    batch_size: int,
    logger: logging.Logger,
    include_confidence_calibration: bool = False,
) -> dict[str, Any]:
    corpus_texts = [bundle.corpus[doc_id] for doc_id in bundle.corpus_order]
    query_texts = [bundle.queries[query_id] for query_id in bundle.query_order]
    doc_index = {doc_id: index for index, doc_id in enumerate(bundle.corpus_order)}
    relevant_indices = [
        {doc_index[doc_id] for doc_id in bundle.relevant_docs[query_id]}
        for query_id in bundle.query_order
    ]

    started = time.perf_counter()
    corpus_embeddings_full = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32", copy=False)
    query_embeddings_full = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32", copy=False)
    encoding_seconds = time.perf_counter() - started

    results: dict[str, Any] = {
        "queries": len(query_texts),
        "corpus": len(corpus_texts),
        "encoding_seconds": encoding_seconds,
        "dimensions": {},
    }
    full_dim = corpus_embeddings_full.shape[1]
    for dimension in dimensions:
        if dimension > full_dim:
            continue
        corpus_embeddings = corpus_embeddings_full[:, :dimension].copy()
        query_embeddings = query_embeddings_full[:, :dimension].copy()
        corpus_embeddings /= np.maximum(np.linalg.norm(corpus_embeddings, axis=1, keepdims=True), 1e-12)
        query_embeddings /= np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
        search_started = time.perf_counter()
        scores, indices = search_top_k(query_embeddings, corpus_embeddings, k=10)
        search_seconds = time.perf_counter() - search_started

        ranks: list[int | None] = []
        top1_correct: list[bool] = []
        top1_scores: list[float] = []
        recalls = {1: [], 5: [], 10: []}
        ndcgs: list[float] = []
        for query_index, relevant in enumerate(relevant_indices):
            retrieved = [int(value) for value in indices[query_index]]
            first_rank: int | None = None
            dcg = 0.0
            for rank, candidate in enumerate(retrieved, start=1):
                if candidate in relevant:
                    if first_rank is None:
                        first_rank = rank
                    dcg += 1.0 / math.log2(rank + 1)
            ideal_hits = min(len(relevant), 10)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)) or 1.0
            ndcgs.append(dcg / idcg)
            ranks.append(first_rank)
            top1_correct.append(first_rank == 1)
            top1_scores.append(float(scores[query_index, 0]))
            for k in recalls:
                hits = sum(candidate in relevant for candidate in retrieved[:k])
                recalls[k].append(hits / max(1, len(relevant)))

        reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]
        target_thresholds: dict[str, Any] | None = None
        best_f1: dict[str, Any] | None = None
        if include_confidence_calibration:
            threshold_rows: list[dict[str, Any]] = []
            score_array = np.asarray(top1_scores)
            correct_array = np.asarray(top1_correct, dtype=bool)
            candidates = np.unique(np.quantile(score_array, np.linspace(0.0, 1.0, 201)))
            for threshold in candidates:
                accepted = score_array >= threshold
                accepted_count = int(accepted.sum())
                if accepted_count == 0:
                    continue
                precision = float(correct_array[accepted].mean())
                coverage = float(accepted.mean())
                f1 = 0.0 if precision + coverage == 0 else 2 * precision * coverage / (precision + coverage)
                threshold_rows.append(
                    {
                        "threshold": float(threshold),
                        "precision": precision,
                        "coverage": coverage,
                        "accepted": accepted_count,
                        "f1_precision_coverage": f1,
                    }
                )

            target_thresholds = {}
            for target in (0.90, 0.95, 0.98):
                eligible = [row for row in threshold_rows if row["precision"] >= target]
                best = max(eligible, key=lambda row: (row["coverage"], row["precision"])) if eligible else None
                target_thresholds[f"precision_{int(target * 100)}"] = best
            best_f1 = max(threshold_rows, key=lambda row: row["f1_precision_coverage"]) if threshold_rows else None

        dimension_metrics = {
            "accuracy_at_1": float(np.mean(top1_correct)),
            "accuracy_at_5": float(np.mean([rank is not None and rank <= 5 for rank in ranks])),
            "accuracy_at_10": float(np.mean([rank is not None and rank <= 10 for rank in ranks])),
            "recall_at_1": float(np.mean(recalls[1])),
            "recall_at_5": float(np.mean(recalls[5])),
            "recall_at_10": float(np.mean(recalls[10])),
            "mrr_at_10": float(np.mean(reciprocal_ranks)),
            "ndcg_at_10": float(np.mean(ndcgs)),
            "search_seconds": search_seconds,
            "search_ms_per_query": 1000.0 * search_seconds / max(1, len(query_texts)),
            "confidence_thresholds": target_thresholds,
            "best_f1_threshold": best_f1,
        }
        results["dimensions"][str(dimension)] = dimension_metrics
        logger.info(
            "%s dim=%d acc@1=%.4f mrr@10=%.4f ndcg@10=%.4f search_ms/query=%.3f",
            bundle.name,
            dimension,
            dimension_metrics["accuracy_at_1"],
            dimension_metrics["mrr_at_10"],
            dimension_metrics["ndcg_at_10"],
            dimension_metrics["search_ms_per_query"],
        )

    del corpus_embeddings_full, query_embeddings_full
    gc.collect()
    return results


def report_markdown(payload: Mapping[str, Any]) -> str:
    dataset = payload["dataset"]
    profile = payload["profile"]
    system = payload["system"]
    run_config = payload.get("run_config", {})
    run_info = run_config.get("run", {})
    autotune = run_config.get("autotune", {})
    final_test = payload.get("final_test_detailed", {})
    dim_metrics = final_test.get("dimensions", {})
    rows = []
    for dimension, metrics in dim_metrics.items():
        rows.append(
            f"| {dimension} | {metrics['accuracy_at_1']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['mrr_at_10']:.4f} | {metrics['ndcg_at_10']:.4f} | {metrics['search_ms_per_query']:.3f} |"
        )
    metrics_table = "\n".join(rows) if rows else "| n/a | n/a | n/a | n/a | n/a | n/a |"
    return f"""# Swico CPU Retrieval Training Report

## Run

- Script version: `{SCRIPT_VERSION}`
- Run ID: `{run_info.get('run_id', 'unknown')}`
- Run directory: `{run_info.get('run_directory', 'unknown')}`
- Base model: `{payload['base_model']}`
- Profile: `{profile['name']}`
- Train rows: `{dataset['split_rows']['train']}`
- Validation rows: `{dataset['split_rows']['validation']}`
- Test rows: `{dataset['split_rows']['test']}`
- Trainable encoder layers: `{profile['trainable_layers']}`
- Maximum sequence length: `{profile['max_seq_length']}`
- Physical training batch size: `{profile['batch_size']}`
- Stage-1 maximum epochs: `{profile['stage1_epochs']}`
- Stage-2 maximum epochs: `{profile['stage2_epochs']}`
- Adaptive mode: `{autotune.get('mode', 'disabled')}`
- Adaptive quality-sensitive changes: `{autotune.get('quality_sensitive_changes', [])}`
- Selected validation stage: `{payload.get('selected_stage', 'unknown')}`
- CPU BF16 autocast: `{payload.get('bf16_cpu_amp', False)}`

## VM

- Logical CPUs: `{system['logical_cpus']}`
- Configured PyTorch threads: `{system['configured_threads']}`
- RAM total: `{system['memory_total_gib']} GiB`
- Disk free at start: `{system['disk_free_gib']} GiB`
- AVX2: `{system['cpu_has_avx2']}`
- AVX-512: `{system['cpu_has_avx512']}`
- BF16 instructions: `{system['cpu_has_bf16']}`

## Data quality

- Original rows: `{dataset['original_rows']}`
- Rows after cleaning/deduplication: `{dataset['rows_after_exact_pair_dedup']}`
- Exact pairs removed: `{dataset['exact_pairs_removed']}`
- Rows with repeated boilerplate removed: `{dataset['boilerplate']['rows_changed']}`
- Repeated sentences removed: `{dataset['boilerplate']['sentences_removed']}`
- Connected components: `{dataset['components']}`
- Cross-split component/query/passage overlaps: all zero

## Final test retrieval metrics

| Dimension | Accuracy@1 | Recall@5 | MRR@10 | NDCG@10 | Search ms/query |
|---:|---:|---:|---:|---:|---:|
{metrics_table}

## Artifacts

- Final model: `models/final/`
- Machine-readable report: `reports/final_report.json`
- Training log: `training.log`
- Prepared leakage-resistant splits: `prepared/`
- Hard-negative cache: `mining/`

Confidence thresholds are calibrated only from `final_validation_detailed`; the held-out test split is not used to choose them. Revalidate these thresholds against real Swico production queries before enforcing them.
"""


def clone_model_tree(source: Path, destination: Path, logger: logging.Logger) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    try:
        shutil.copytree(source, destination, copy_function=os.link)
        logger.info("Created disk-efficient final-model hard links from %s", source)
    except OSError:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        logger.info("Copied final model from %s", source)


def clean_completed_artifacts(output: Path, keep_intermediate: bool, keep_checkpoints: bool, logger: logging.Logger) -> None:
    if not keep_checkpoints:
        checkpoint_root = output / "checkpoints"
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root, ignore_errors=True)
            logger.info("Removed completed training checkpoints to save disk")
    if not keep_intermediate:
        stage1_model = output / "models" / "stage1"
        if stage1_model.exists():
            shutil.rmtree(stage1_model, ignore_errors=True)
            logger.info("Removed intermediate stage-1 model to save disk")


def promote_validation_winner(
    output: Path,
    candidate_metrics: Mapping[str, Mapping[str, float]],
    primary_metric: str,
    baseline_protection: bool,
    minimum_validation_gain: float,
    base_model_source: str,
    max_seq_length: int,
    offline: bool,
    logger: logging.Logger,
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for name, metrics in candidate_metrics.items():
        if primary_metric not in metrics:
            continue
        value = float(metrics[primary_metric])
        if not math.isfinite(value):
            raise RuntimeError(f"Validation metric {primary_metric!r} for {name} is not finite")
        scores[name] = value

    trained_scores = {name: score for name, score in scores.items() if name != "base"}
    if not trained_scores:
        raise RuntimeError("No trained candidate has a usable validation score")
    selected = max(trained_scores, key=trained_scores.get)

    if baseline_protection:
        if "base" not in scores:
            raise RuntimeError("Baseline protection is enabled but base validation metrics are missing")
        required = scores["base"] + minimum_validation_gain
        if trained_scores[selected] < required:
            logger.warning(
                "Quality gate rejected trained candidates: best=%s %.6f required>=%.6f; promoting base model",
                selected,
                trained_scores[selected],
                required,
            )
            selected = "base"

    final_dir = output / "models" / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    if selected == "base":
        base_model = SentenceTransformer(
            base_model_source,
            device="cpu",
            local_files_only=offline,
        )
        base_model.max_seq_length = max_seq_length
        base_model.save(str(final_dir))
        del base_model
        gc.collect()
    else:
        clone_model_tree(output / "models" / selected, final_dir, logger)

    logger.info(
        "Validation selection: metric=%s scores=%s selected=%s baseline_protection=%s",
        primary_metric,
        {key: round(value, 6) for key, value in scores.items()},
        selected,
        baseline_protection,
    )
    return selected, scores


def main() -> int:
    args = parse_args()
    profile = profile_with_overrides(args)
    validate_profile(profile, args)
    if args.print_config:
        preview = effective_configuration_preview(args, profile)
        preview["run_directory_policy"] = {
            "output_root": str(args.output_root),
            "exact_output_override": str(args.output) if args.output else None,
            "run_mode": args.run_mode,
            "timestamp_format": "YYYYMMDDTHHMMSSZ_profile[_label]",
            "completed_runs_are_never_reused": True,
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    configured_invocation = invocation_configuration(args, profile)
    context = resolve_run_context(args, profile)
    output = context.output
    if args.overwrite_output and output.exists():
        protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output in protected:
            raise RuntimeError(f"Refusing to delete protected output path: {output}")
        shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        context = dataclasses.replace(context, resumed=False)

    logger = configure_logging(output)
    logger.info("Configuration file: %s", LOADED_ENV_FILE or "none (built-in defaults and shell/CLI only)")
    logger.info("Selected profile: %s", profile.name)
    logger.info("Run directory: %s (%s)", output, "resumed" if context.resumed else "new")
    if args.create_latest_links:
        update_run_links(context, completed=False)
    atomic_write_json(
        output / "run_manifest.json",
        {
            "run_id": context.run_id,
            "run_directory": str(output),
            "output_root": str(context.root),
            "resumed": context.resumed,
            "exact_output_override": context.exact_output_override,
            "invocation_fingerprint": context.invocation_fingerprint,
            "configured_invocation": configured_invocation,
            "status": "initializing",
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )

    configured_threads = configure_runtime(args.threads, args.seed, logger)
    system = system_report(output, configured_threads)
    preflight(
        system,
        min_available_memory_gib=args.min_available_memory_gib,
        min_free_disk_gib=args.min_free_disk_gib,
        warn_free_disk_gib=args.warn_free_disk_gib,
        logger=logger,
    )
    profile, autotune_report = apply_adaptive_resource_plan(
        profile=profile,
        args=args,
        report=system,
        output=output,
        logger=logger,
    )
    validate_profile(profile, args)
    use_bf16 = bool(system["cpu_has_bf16"]) if args.bf16 is None else bool(args.bf16)
    logger.info("CPU BF16 autocast: %s", "enabled" if use_bf16 else "disabled")

    atomic_write_json(
        output / "run_manifest.json",
        {
            "run_id": context.run_id,
            "run_directory": str(output),
            "output_root": str(context.root),
            "resumed": context.resumed,
            "exact_output_override": context.exact_output_override,
            "invocation_fingerprint": context.invocation_fingerprint,
            "configured_invocation": configured_invocation,
            "resolved_invocation": invocation_configuration(args, profile),
            "autotune": autotune_report,
            "status": "running",
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )

    data_path = discover_data_file(args.data, output / "working", logger)
    splits, dataset_meta = prepare_dataset(
        data_path=data_path,
        prepared_dir=output / "prepared",
        boilerplate_threshold=args.boilerplate_threshold,
        seed=args.seed,
        max_train_rows=profile.max_train_rows,
        split_fractions={
            "train": args.train_split_fraction,
            "validation": args.validation_split_fraction,
            "test": args.test_split_fraction,
        },
        logger=logger,
    )
    run_config = {
        "script_version": SCRIPT_VERSION,
        "run": {
            "run_id": context.run_id,
            "run_directory": str(output),
            "output_root": str(context.root),
        },
        "base_model": args.base_model,
        "profile": dataclasses.asdict(profile),
        "autotune": {
            "enabled": bool(args.autotune),
            "mode": args.autotune_mode,
            "resolved": autotune_report.get("resolved", {}),
            "quality_sensitive_changes": autotune_report.get("quality_sensitive_changes", []),
        },
        "dataset": {
            "source_sha256": dataset_meta["cache_key"]["source_sha256"],
            "boilerplate_threshold": args.boilerplate_threshold,
            "split_fractions": {
                "train": args.train_split_fraction,
                "validation": args.validation_split_fraction,
                "test": args.test_split_fraction,
            },
        },
        "runtime": {
            "threads": configured_threads,
            "bf16_cpu_amp": use_bf16,
            "seed": args.seed,
        },
        "optimizer": {
            "stage1_lr": args.stage1_lr,
            "stage2_lr": args.stage2_lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": args.lr_scheduler_type,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_drop_last": args.dataloader_drop_last,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
        },
        "evaluation": {
            "batch_size": args.eval_batch_size,
            "loss_rows": args.eval_loss_rows,
            "corpus_chunk_size": args.eval_corpus_chunk_size,
            "early_stopping": args.early_stopping,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "load_best_model_at_end": args.load_best_model_at_end,
            "baseline_protection": args.baseline_protection,
            "minimum_validation_gain": args.min_validation_gain,
        },
        "hard_negative_mining": {
            "minimum_margin": args.min_hard_negative_margin,
            "batch_size": args.mining_batch_size,
            "chunk_size": args.mining_chunk_size,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "hnsw_ef_search_factor": args.hnsw_ef_search_factor,
            "maximum_token_jaccard": args.negative_max_token_jaccard,
        },
        "preflight": {
            "min_available_memory_gib": args.min_available_memory_gib,
            "min_free_disk_gib": args.min_free_disk_gib,
            "warn_free_disk_gib": args.warn_free_disk_gib,
        },
        "memory_guard": {
            "enabled": args.memory_guard,
            "interval_steps": args.memory_guard_interval_steps,
            "emergency_available_memory_gib": args.emergency_available_memory_gib,
            "max_process_rss_gib": args.max_process_rss_gib,
        },
        "loss": {
            "scale": args.loss_scale,
            "matryoshka_weights": list(args.matryoshka_weights),
            "matryoshka_dims_per_step": args.matryoshka_dims_per_step,
        },
    }
    resume_config = {
        key: value
        for key, value in run_config.items()
        if key not in {"preflight", "run"}
    }
    atomic_write_json(
        output / "run_config.json",
        {
            "configuration_source": str(LOADED_ENV_FILE) if LOADED_ENV_FILE else None,
            "effective_training_config": run_config,
            "resume_compatibility_config": resume_config,
        },
    )
    if args.prepare_only:
        logger.info("Dataset preparation complete; --prepare-only requested, so training is skipped")
        return 0

    state_path = output / "run_state.json"
    state: dict[str, Any] = {}
    if state_path.exists() and args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_config") != resume_config:
            raise RuntimeError(
                "Existing output was created with a different configuration. Use a new --output or --overwrite-output."
            )
    elif state_path.exists() and not args.resume:
        raise RuntimeError("Output already contains training state. Use --resume or --overwrite-output.")
    else:
        state = {
            "run_config": resume_config,
            "stage1_complete": False,
            "stage2_complete": False,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_write_json(state_path, state)

    final_model_dir = output / "models" / "final"
    stage1_model_dir = output / "models" / "stage1"
    if state.get("stage2_complete") and not final_model_dir.exists():
        selected_stage = str(state.get("selected_stage", ""))
        selected_model_dir = output / "models" / selected_stage
        if selected_stage in {"stage1", "stage2"} and selected_model_dir.exists():
            logger.warning(
                "Final model directory is missing; restoring it from the recorded selected stage: %s",
                selected_stage,
            )
            clone_model_tree(selected_model_dir, final_model_dir, logger)
        elif selected_stage == "base":
            logger.warning("Final model directory is missing; restoring the recorded base-model winner")
            base_model = SentenceTransformer(
                args.base_model,
                device="cpu",
                local_files_only=args.offline,
            )
            base_model.max_seq_length = profile.max_seq_length
            base_model.save(str(final_model_dir))
            del base_model
            gc.collect()
        else:
            logger.warning(
                "Run state marks training complete, but neither the final model nor the selected intermediate model exists; "
                "resetting the incomplete stage state so --resume can rebuild it safely"
            )
            state["stage2_complete"] = False
            state["final_evaluation_complete"] = False
            if not stage1_model_dir.exists():
                state["stage1_complete"] = False
            atomic_write_json(state_path, state)
    elif state.get("stage1_complete") and not stage1_model_dir.exists() and not final_model_dir.exists():
        logger.warning(
            "Run state marks stage 1 complete, but its model directory is missing; resetting stage 1 for a safe resume"
        )
        state["stage1_complete"] = False
        state["stage2_complete"] = False
        state["final_evaluation_complete"] = False
        atomic_write_json(state_path, state)

    document_pool = pd.concat([splits["train"], splits["validation"], splits["test"]], ignore_index=True)
    validation_bundle = build_eval_bundle(
        name="validation",
        eval_frame=splits["validation"],
        document_pool=document_pool,
        max_queries=profile.eval_queries,
        max_corpus=profile.eval_corpus,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        corpus_chunk_size=args.eval_corpus_chunk_size,
        output_path=output / "evaluation" / "validation",
    )
    test_bundle = build_eval_bundle(
        name="test",
        eval_frame=splits["test"],
        document_pool=document_pool,
        max_queries=profile.eval_queries,
        max_corpus=profile.eval_corpus,
        seed=args.seed + 1,
        batch_size=args.eval_batch_size,
        corpus_chunk_size=args.eval_corpus_chunk_size,
        output_path=output / "evaluation" / "test",
    )

    stage_summaries: dict[str, Any] = {}
    stage_validation_metrics: dict[str, dict[str, float]] = {}
    base_validation_metrics = _read_json(output / "reports" / "base_validation.json") or {}
    for completed_stage in ("stage1", "stage2"):
        stage_report = output / "reports" / f"{completed_stage}.json"
        if stage_report.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                stage_summaries[completed_stage] = json.loads(stage_report.read_text(encoding="utf-8"))
        validation_report = output / "reports" / f"{completed_stage}_validation.json"
        if validation_report.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                stage_validation_metrics[completed_stage] = json.loads(
                    validation_report.read_text(encoding="utf-8")
                )
    parameter_summary: dict[str, Any]

    if state.get("stage2_complete") and (output / "models" / "final").exists():
        logger.info("Final model already exists; skipping completed training stages")
        model, parameter_summary = load_model(
            output / "models" / "final",
            profile.max_seq_length,
            profile.trainable_layers,
            offline=True,
            logger=logger,
        )
    else:
        if state.get("stage1_complete") and (output / "models" / "stage1").exists():
            model, parameter_summary = load_model(
                output / "models" / "stage1",
                profile.max_seq_length,
                profile.trainable_layers,
                offline=True,
                logger=logger,
            )
        else:
            model, parameter_summary = load_model(
                args.base_model,
                profile.max_seq_length,
                profile.trainable_layers,
                offline=args.offline,
                logger=logger,
            )
            if not args.skip_base_eval:
                base_validation_metrics = evaluate_bundle(
                    model,
                    validation_bundle,
                    output / "evaluation" / "base",
                    logger,
                )
                atomic_write_json(
                    output / "reports" / "base_validation.json",
                    base_validation_metrics,
                )

            model, stage1_summary = train_stage(
                stage_name="stage1",
                model=model,
                train_frame=splits["train"],
                eval_frame=splits["validation"],
                evaluator=validation_bundle.evaluator,
                output=output,
                epochs=profile.stage1_epochs,
                batch_size=profile.batch_size,
                eval_batch_size=args.eval_batch_size,
                eval_loss_rows=args.eval_loss_rows,
                learning_rate=args.stage1_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                lr_scheduler_type=args.lr_scheduler_type,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                gradient_checkpointing=args.gradient_checkpointing,
                dataloader_num_workers=args.dataloader_num_workers,
                dataloader_drop_last=args.dataloader_drop_last,
                logging_steps=args.logging_steps,
                save_total_limit=args.save_total_limit,
                dims=profile.matryoshka_dims,
                matryoshka_weights=args.matryoshka_weights,
                matryoshka_dims_per_step=args.matryoshka_dims_per_step,
                loss_scale=args.loss_scale,
                include_negative=False,
                resume=args.resume,
                seed=args.seed,
                use_bf16=use_bf16,
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
                load_best_model_at_end=args.load_best_model_at_end,
                memory_guard=args.memory_guard,
                memory_guard_interval_steps=args.memory_guard_interval_steps,
                emergency_available_memory_gib=args.emergency_available_memory_gib,
                max_process_rss_gib=args.max_process_rss_gib,
                logger=logger,
            )
            stage_summaries["stage1"] = stage1_summary
            state["stage1_complete"] = True
            state["stage1_completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_write_json(state_path, state)
            if not args.keep_checkpoints:
                shutil.rmtree(output / "checkpoints" / "stage1", ignore_errors=True)

        if "stage1" not in stage_validation_metrics:
            stage1_validation = evaluate_bundle(
                model,
                validation_bundle,
                output / "evaluation" / "stage1_validation",
                logger,
            )
            stage_validation_metrics["stage1"] = stage1_validation
            atomic_write_json(output / "reports" / "stage1_validation.json", stage1_validation)

        if profile.hard_negatives and profile.stage2_epochs > 0 and not state.get("stage2_complete"):
            mined_train, mining_summary = mine_hard_negatives(
                model=model,
                train_frame=splits["train"],
                cache_dir=output / "mining",
                base_model=args.base_model,
                top_k=profile.mining_top_k,
                min_margin=args.min_hard_negative_margin,
                mining_batch_size=args.mining_batch_size,
                mining_chunk_size=args.mining_chunk_size,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                hnsw_ef_search_factor=args.hnsw_ef_search_factor,
                negative_max_token_jaccard=args.negative_max_token_jaccard,
                seed=args.seed,
                logger=logger,
            )
            atomic_write_json(output / "reports" / "hard_negative_mining.json", mining_summary)
            model, stage2_summary = train_stage(
                stage_name="stage2",
                model=model,
                train_frame=mined_train,
                eval_frame=splits["validation"],
                evaluator=validation_bundle.evaluator,
                output=output,
                epochs=profile.stage2_epochs,
                batch_size=profile.batch_size,
                eval_batch_size=args.eval_batch_size,
                eval_loss_rows=args.eval_loss_rows,
                learning_rate=args.stage2_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                lr_scheduler_type=args.lr_scheduler_type,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                gradient_checkpointing=args.gradient_checkpointing,
                dataloader_num_workers=args.dataloader_num_workers,
                dataloader_drop_last=args.dataloader_drop_last,
                logging_steps=args.logging_steps,
                save_total_limit=args.save_total_limit,
                dims=profile.matryoshka_dims,
                matryoshka_weights=args.matryoshka_weights,
                matryoshka_dims_per_step=args.matryoshka_dims_per_step,
                loss_scale=args.loss_scale,
                include_negative=True,
                resume=args.resume,
                seed=args.seed + 1,
                use_bf16=use_bf16,
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
                load_best_model_at_end=args.load_best_model_at_end,
                memory_guard=args.memory_guard,
                memory_guard_interval_steps=args.memory_guard_interval_steps,
                emergency_available_memory_gib=args.emergency_available_memory_gib,
                max_process_rss_gib=args.max_process_rss_gib,
                logger=logger,
            )
            stage_summaries["stage2"] = stage2_summary
            stage2_validation = evaluate_bundle(
                model,
                validation_bundle,
                output / "evaluation" / "stage2_validation",
                logger,
            )
            stage_validation_metrics["stage2"] = stage2_validation
            atomic_write_json(output / "reports" / "stage2_validation.json", stage2_validation)

            primary_metric = str(validation_bundle.evaluator.primary_metric)
            candidate_metrics: dict[str, Mapping[str, float]] = {
                "stage1": stage_validation_metrics["stage1"],
                "stage2": stage2_validation,
            }
            if base_validation_metrics:
                candidate_metrics["base"] = base_validation_metrics
            selected_stage, selection_scores = promote_validation_winner(
                output=output,
                candidate_metrics=candidate_metrics,
                primary_metric=primary_metric,
                baseline_protection=args.baseline_protection,
                minimum_validation_gain=args.min_validation_gain,
                base_model_source=args.base_model,
                max_seq_length=profile.max_seq_length,
                offline=args.offline,
                logger=logger,
            )
            if not args.keep_intermediate:
                shutil.rmtree(output / "models" / "stage2", ignore_errors=True)
            state["stage2_complete"] = True
            state["stage2_completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            state["selected_stage"] = selected_stage
            state["selection_metric"] = primary_metric
            state["selection_scores"] = selection_scores
            state["baseline_protection"] = args.baseline_protection
            atomic_write_json(state_path, state)
        elif not state.get("stage2_complete"):
            primary_metric = str(validation_bundle.evaluator.primary_metric)
            candidate_metrics = {"stage1": stage_validation_metrics["stage1"]}
            if base_validation_metrics:
                candidate_metrics["base"] = base_validation_metrics
            selected_stage, selection_scores = promote_validation_winner(
                output=output,
                candidate_metrics=candidate_metrics,
                primary_metric=primary_metric,
                baseline_protection=args.baseline_protection,
                minimum_validation_gain=args.min_validation_gain,
                base_model_source=args.base_model,
                max_seq_length=profile.max_seq_length,
                offline=args.offline,
                logger=logger,
            )
            state["stage2_complete"] = True
            state["stage2_skipped"] = True
            state["selected_stage"] = selected_stage
            state["selection_metric"] = primary_metric
            state["selection_scores"] = selection_scores
            state["baseline_protection"] = args.baseline_protection
            atomic_write_json(state_path, state)

    # Reload the standalone final model before final evaluation.
    del model
    gc.collect()
    final_model = SentenceTransformer(str(output / "models" / "final"), device="cpu")
    final_model.max_seq_length = profile.max_seq_length
    final_validation = evaluate_bundle(
        final_model,
        validation_bundle,
        output / "evaluation" / "final_validation",
        logger,
    )
    final_test = evaluate_bundle(
        final_model,
        test_bundle,
        output / "evaluation" / "final_test",
        logger,
    )
    detailed_validation = detailed_retrieval_metrics(
        final_model,
        validation_bundle,
        dimensions=profile.matryoshka_dims,
        batch_size=args.eval_batch_size,
        logger=logger,
        include_confidence_calibration=True,
    )
    detailed_test = detailed_retrieval_metrics(
        final_model,
        test_bundle,
        dimensions=profile.matryoshka_dims,
        batch_size=args.eval_batch_size,
        logger=logger,
        include_confidence_calibration=False,
    )

    final_payload: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "base_model": args.base_model,
        "profile": dataclasses.asdict(profile),
        "configuration_source": str(LOADED_ENV_FILE) if LOADED_ENV_FILE else None,
        "run_config": run_config,
        "bf16_cpu_amp": use_bf16,
        "system": system,
        "dataset": dataset_meta,
        "parameters": parameter_summary,
        "stages": stage_summaries,
        "stage_validation": stage_validation_metrics,
        "selected_stage": state.get("selected_stage", "unknown"),
        "final_validation": final_validation,
        "final_test": final_test,
        "final_validation_detailed": detailed_validation,
        "final_test_detailed": detailed_test,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(output / "reports" / "final_report.json", final_payload)
    (output / "reports" / "final_report.md").write_text(report_markdown(final_payload), encoding="utf-8")
    state["final_evaluation_complete"] = True
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_write_json(state_path, state)
    clean_completed_artifacts(output, args.keep_intermediate, args.keep_checkpoints, logger)
    manifest = _read_json(output / "run_manifest.json") or {}
    manifest.update(
        {
            "status": "completed",
            "completed_at": state["completed_at"],
            "selected_stage": state.get("selected_stage"),
            "final_model": str(output / "models" / "final"),
            "final_report": str(output / "reports" / "final_report.md"),
        }
    )
    atomic_write_json(output / "run_manifest.json", manifest)
    if args.create_latest_links:
        update_run_links(context, completed=True)
    logger.info("Training pipeline complete. Run: %s", output)
    logger.info("Final model: %s", output / "models" / "final")
    logger.info("Report: %s", output / "reports" / "final_report.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourceStopRequested as exc:
        print(f"Resource guard: {exc}", file=sys.stderr)
        raise SystemExit(75) from exc
    except KeyboardInterrupt:
        print("\nTraining interrupted. Re-run with --resume to continue from the latest saved checkpoint.", file=sys.stderr)
        raise SystemExit(130)

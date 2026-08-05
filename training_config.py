"""Typed environment-file configuration for the Swico CPU trainer.

Configuration precedence is intentionally explicit:

1. Command-line arguments
2. Existing process environment variables
3. Values loaded from the selected env file
4. Built-in profile defaults

The loader only handles configuration. It never imports torch or any model package,
so it can run before CPU-library thread variables are finalized.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values, load_dotenv


class ConfigError(ValueError):
    """Raised when an environment-file value is missing, malformed, or unknown."""


SUPPORTED_ENV_KEYS = frozenset(
    {
        "SWICO_ENV_FILE",
        "SWICO_CONFIG_STRICT",
        "SWICO_PROFILE",
        "SWICO_DATA_PATH",
        "SWICO_OUTPUT_ROOT",
        "SWICO_OUTPUT_DIR",
        "SWICO_RUN_MODE",
        "SWICO_RUN_ID",
        "SWICO_RUN_LABEL",
        "SWICO_CREATE_LATEST_LINKS",
        "SWICO_BASE_MODEL",
        "SWICO_CPU_THREADS",
        "SWICO_SEED",
        "SWICO_BF16",
        "SWICO_RESUME",
        "SWICO_PREPARE_ONLY",
        "SWICO_SKIP_BASE_EVAL",
        "SWICO_KEEP_INTERMEDIATE",
        "SWICO_KEEP_CHECKPOINTS",
        "SWICO_OVERWRITE_OUTPUT",
        "SWICO_OFFLINE",
        "SWICO_BOILERPLATE_THRESHOLD",
        "SWICO_MAX_TRAIN_ROWS",
        "SWICO_TRAIN_SPLIT_FRACTION",
        "SWICO_VALIDATION_SPLIT_FRACTION",
        "SWICO_TEST_SPLIT_FRACTION",
        "SWICO_BATCH_SIZE",
        "SWICO_MAX_SEQ_LENGTH",
        "SWICO_TRAINABLE_LAYERS",
        "SWICO_STAGE1_EPOCHS",
        "SWICO_STAGE2_EPOCHS",
        "SWICO_STAGE1_LR",
        "SWICO_STAGE2_LR",
        "SWICO_WEIGHT_DECAY",
        "SWICO_WARMUP_RATIO",
        "SWICO_LR_SCHEDULER_TYPE",
        "SWICO_GRADIENT_ACCUMULATION_STEPS",
        "SWICO_MAX_GRAD_NORM",
        "SWICO_GRADIENT_CHECKPOINTING",
        "SWICO_DATALOADER_NUM_WORKERS",
        "SWICO_DATALOADER_DROP_LAST",
        "SWICO_LOGGING_STEPS",
        "SWICO_SAVE_TOTAL_LIMIT",
        "SWICO_EVAL_BATCH_SIZE",
        "SWICO_EVAL_LOSS_ROWS",
        "SWICO_EVAL_CORPUS_CHUNK_SIZE",
        "SWICO_EVAL_QUERIES",
        "SWICO_EVAL_CORPUS",
        "SWICO_HARD_NEGATIVES",
        "SWICO_MINING_TOP_K",
        "SWICO_MIN_HARD_NEGATIVE_MARGIN",
        "SWICO_MINING_BATCH_SIZE",
        "SWICO_MINING_CHUNK_SIZE",
        "SWICO_HNSW_M",
        "SWICO_HNSW_EF_CONSTRUCTION",
        "SWICO_HNSW_EF_SEARCH_FACTOR",
        "SWICO_NEGATIVE_MAX_TOKEN_JACCARD",
        "SWICO_LOSS_SCALE",
        "SWICO_MATRYOSHKA_DIMS",
        "SWICO_MATRYOSHKA_WEIGHTS",
        "SWICO_MATRYOSHKA_DIMS_PER_STEP",
        "SWICO_EARLY_STOPPING",
        "SWICO_EARLY_STOPPING_PATIENCE",
        "SWICO_EARLY_STOPPING_THRESHOLD",
        "SWICO_LOAD_BEST_MODEL_AT_END",
        "SWICO_MIN_AVAILABLE_MEMORY_GIB",
        "SWICO_MIN_FREE_DISK_GIB",
        "SWICO_WARN_FREE_DISK_GIB",
        "SWICO_AUTOTUNE",
        "SWICO_AUTOTUNE_MODE",
        "SWICO_AUTOTUNE_MAX_INFERENCE_BATCH",
        "SWICO_AUTOTUNE_MAX_TRAIN_BATCH",
        "SWICO_AUTOTUNE_MEMORY_RESERVE_GIB",
        "SWICO_AUTOTUNE_MEMORY_UTILIZATION",
        "SWICO_AUTOTUNE_TRAIN_BATCH",
        "SWICO_MEMORY_GUARD",
        "SWICO_MEMORY_GUARD_INTERVAL_STEPS",
        "SWICO_EMERGENCY_AVAILABLE_MEMORY_GIB",
        "SWICO_MAX_PROCESS_RSS_GIB",
        "SWICO_BASELINE_PROTECTION",
        "SWICO_MIN_VALIDATION_GAIN",
    }
)

_AUTO_VALUES = {"", "auto", "default", "none", "null"}
_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}


def _extract_env_file(argv: Iterable[str]) -> tuple[Path, bool]:
    values = list(argv)
    explicit = False
    selected: str | None = None
    for index, value in enumerate(values):
        if value == "--env-file":
            explicit = True
            if index + 1 >= len(values):
                raise ConfigError("--env-file requires a path")
            selected = values[index + 1]
            break
        if value.startswith("--env-file="):
            explicit = True
            selected = value.split("=", 1)[1]
            break
    if selected is None:
        selected = os.environ.get("SWICO_ENV_FILE")
        explicit = selected is not None
    if not selected:
        selected = "training.env"
    return Path(selected).expanduser(), explicit


def _parse_bool_text(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0; got {value!r}"
    )


def initialize_environment(argv: Iterable[str] | None = None) -> Path | None:
    """Load the requested env file without overriding shell environment values."""

    selected, explicit = _extract_env_file(sys.argv[1:] if argv is None else argv)
    if not selected.is_absolute():
        selected = (Path.cwd() / selected).resolve()
    if not selected.exists():
        if explicit:
            raise ConfigError(f"Configured env file does not exist: {selected}")
        return None
    if not selected.is_file():
        raise ConfigError(f"Configured env path is not a regular file: {selected}")

    raw_values = dotenv_values(selected, interpolate=False)
    malformed = sorted(key for key, value in raw_values.items() if key is None or value is None)
    if malformed:
        raise ConfigError(f"Malformed entries exist in {selected}: {malformed}")

    strict_raw = os.environ.get(
        "SWICO_CONFIG_STRICT",
        str(raw_values.get("SWICO_CONFIG_STRICT", "true")),
    )
    strict = _parse_bool_text("SWICO_CONFIG_STRICT", strict_raw)
    unknown = sorted(
        key
        for key in raw_values
        if key and key.startswith("SWICO_") and key not in SUPPORTED_ENV_KEYS
    )
    if strict and unknown:
        raise ConfigError(
            "Unknown SWICO_ configuration keys in "
            f"{selected}: {', '.join(unknown)}. Fix the typo or set SWICO_CONFIG_STRICT=false."
        )

    load_dotenv(selected, override=False, interpolate=False)
    os.environ.setdefault("SWICO_ENV_FILE", str(selected))
    unknown_runtime = sorted(
        key
        for key in os.environ
        if key.startswith("SWICO_") and key not in SUPPORTED_ENV_KEYS
    )
    if strict and unknown_runtime:
        raise ConfigError(
            "Unknown SWICO_ configuration keys in the process environment: "
            f"{', '.join(unknown_runtime)}. Fix the typo or set SWICO_CONFIG_STRICT=false."
        )
    return selected


def env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def env_optional_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in _AUTO_VALUES:
        return None
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer; got {value!r}") from exc


def env_optional_int(name: str) -> int | None:
    value = env_optional_text(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer or auto; got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number; got {value!r}") from exc


def env_optional_float(name: str) -> float | None:
    value = env_optional_text(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number or auto; got {value!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return _parse_bool_text(name, value)


def env_optional_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in _AUTO_VALUES:
        return None
    return _parse_bool_text(name, value)


def _split_csv(name: str, value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ConfigError(f"{name} must be a comma-separated list without empty items")
    return items


def env_optional_int_tuple(name: str) -> tuple[int, ...] | None:
    value = env_optional_text(name)
    if value is None:
        return None
    try:
        return tuple(int(item) for item in _split_csv(name, value))
    except ValueError as exc:
        raise ConfigError(f"{name} must contain comma-separated integers; got {value!r}") from exc


def env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in _AUTO_VALUES:
        return default
    try:
        return tuple(float(item) for item in _split_csv(name, value))
    except ValueError as exc:
        raise ConfigError(f"{name} must contain comma-separated numbers; got {value!r}") from exc

#!/usr/bin/env python3
"""
Swico CPU retrieval-model fine-tuning pipeline.

Designed for a 4-physical-core / 8-logical-CPU VM with about 29 GiB RAM,
no GPU, and constrained disk. It fine-tunes intfloat/multilingual-e5-small
for query-to-passage retrieval; it does not train a general-purpose LLM.

Default VM profile:
- leakage-resistant connected-component data split
- repeated-boilerplate removal and exact-pair deduplication
- last-four-layer partial fine-tuning (lower layers and large embeddings frozen)
- stage 1: in-batch-negative retrieval training
- stage 2: guarded hard-negative curriculum
- no-duplicate batches
- Matryoshka dimensions 384 / 256 / 128
- bounded IR evaluation, confidence calibration, latency report
- one resumable checkpoint at a time

Examples:
  python train_vm.py --profile smoke
  python train_vm.py --profile vm --resume
  python train_vm.py --profile full --resume
  python train_vm.py --profile vm --prepare-only
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

# Configure CPU libraries before importing torch/numpy.
_DEFAULT_THREADS = max(1, min(8, os.cpu_count() or 4))
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

from transformers.trainer_utils import get_last_checkpoint


BASE_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
SCRIPT_VERSION = "2.1.1-cpu"


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
        stage1_epochs=1.0,
        stage2_epochs=1.0,
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
        stage1_epochs=1.0,
        stage2_epochs=1.0,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only fine-tuning for the Swico multilingual E5 retrieval model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="vm")
    parser.add_argument("--data", type=Path, default=None, help="CSV file or ZIP containing a CSV")
    parser.add_argument("--output", type=Path, default=Path("training_artifacts/e5-small-swico"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--threads", type=int, default=_DEFAULT_THREADS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boilerplate-threshold", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--trainable-layers", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=float, default=None)
    parser.add_argument("--stage2-epochs", type=float, default=None)
    parser.add_argument("--stage1-lr", type=float, default=2.0e-5)
    parser.add_argument("--stage2-lr", type=float, default=8.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--min-hard-negative-margin", type=float, default=0.02)
    parser.add_argument("--hard-negatives", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only locally cached model files")
    return parser.parse_args()


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

    full_splits = component_split(raw, seed=seed)
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
        hard_negatives=args.hard_negatives if args.hard_negatives is not None else base.hard_negatives,
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
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.min_hard_negative_margin < 0:
        raise ValueError("min_hard_negative_margin cannot be negative")


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


def preflight(report: Mapping[str, Any], logger: logging.Logger) -> None:
    if float(report["memory_available_gib"]) < 8.0:
        raise RuntimeError("At least 8 GiB available RAM is required for this CPU profile.")
    free_disk = float(report["disk_free_gib"])
    if free_disk < 2.0:
        raise RuntimeError("Less than 2 GiB disk is free. Free disk space before training.")
    if free_disk < 4.0:
        logger.warning("Only %.2f GiB disk is free. Keep-checkpoints and extra exports should remain disabled.", free_disk)
    if not bool(report["cpu_has_avx2"]):
        logger.warning("AVX2 was not detected; CPU training will be substantially less efficient.")


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
        corpus_chunk_size=min(5_000, max(1, len(corpus))),
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
        batch_size=64,
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
            32,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = max(64, top_k * 4)
        index.add(corpus_embeddings)
        mining_backend = requested_backend
    else:
        index = None
        mining_backend = requested_backend
        logger.warning("faiss-cpu is unavailable; using slower chunked NumPy hard-negative search")

    negatives: list[str] = []
    mining_modes: Counter[str] = Counter()
    margins: list[float] = []
    chunk_size = 512
    fallback_indices = random_negative_indices(train_frame["component"].astype(str).tolist(), seed)

    for start in range(0, len(train_frame), chunk_size):
        stop = min(len(train_frame), start + chunk_size)
        chunk = train_frame.iloc[start:stop]
        query_embeddings = model.encode(
            [QUERY_PREFIX + value for value in chunk["query"].tolist()],
            batch_size=64,
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
                if token_jaccard(str(row.positive_norm), corpus_norms[candidate_index]) >= 0.88:
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


def make_loss(model: SentenceTransformer, dims: Sequence[int]) -> torch.nn.Module:
    base_loss = MultipleNegativesRankingLoss(model=model, scale=20.0)
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
    weights = [1.0] + [0.35] * (len(valid_dims) - 1)
    return MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=valid_dims,
        matryoshka_weights=weights,
        n_dims_per_step=-1,
    )


def make_training_args(
    output_dir: Path,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    train_rows: int,
    seed: int,
    use_bf16: bool,
) -> SentenceTransformerTrainingArguments:
    """Build arguments across the supported Transformers 4.x and 5.x APIs.

    Transformers 5 removed ``save_safetensors`` and replaced ``warmup_ratio``
    with a float-capable ``warmup_steps`` field. Sentence Transformers 5.6.1
    supports both major Transformers lines, so detect the installed signature
    instead of assuming one version.
    """

    steps_per_epoch = max(1, math.ceil(train_rows / batch_size))
    save_steps = max(100, min(250, max(1, steps_per_epoch // 4)))
    supported = set(inspect.signature(SentenceTransformerTrainingArguments.__init__).parameters)

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(16, min(64, batch_size * 2)),
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,
        "fp16": False,
        "bf16": use_bf16,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "dataloader_drop_last": True,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 1,
        "load_best_model_at_end": False,
        "logging_strategy": "steps",
        "logging_steps": 25,
        "logging_first_step": True,
        "report_to": "none",
        "optim": "adamw_torch",
        "seed": seed,
        "data_seed": seed,
        "remove_unused_columns": True,
    }

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

    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in supported:
        # Transformers 5 accepts a float in [0, 1) as a ratio.
        kwargs["warmup_steps"] = warmup_ratio
    else:
        raise RuntimeError(
            "The installed Transformers TrainingArguments has no warmup_ratio or warmup_steps field."
        )

    if "save_safetensors" in supported:
        # Transformers 4 exposes this option. Transformers 5 always saves safely
        # and removed the argument entirely.
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
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    dims: Sequence[int],
    include_negative: bool,
    resume: bool,
    seed: int,
    use_bf16: bool,
    logger: logging.Logger,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    checkpoint_dir = output / "checkpoints" / stage_name
    model_dir = output / "models" / stage_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = formatted_dataset(train_frame, include_negative=include_negative)
    eval_loss_frame = eval_frame.head(min(512, len(eval_frame))).copy()
    if include_negative and "negative" not in eval_loss_frame.columns:
        # Evaluation loss can safely use pairs while the stage trains on triplets.
        eval_dataset = formatted_dataset(eval_loss_frame, include_negative=False)
    else:
        eval_dataset = formatted_dataset(eval_loss_frame, include_negative=include_negative)
    loss = make_loss(model, dims)
    training_args = make_training_args(
        output_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        train_rows=len(train_frame),
        seed=seed,
        use_bf16=use_bf16,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    resume_from = last_checkpoint if resume and last_checkpoint else None
    logger.info(
        "Starting %s: rows=%d epochs=%.2f batch=%d lr=%g resume=%s",
        stage_name,
        len(train_frame),
        epochs,
        batch_size,
        learning_rate,
        resume_from or "none",
    )
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume_from)
    duration = time.perf_counter() - started
    trainer.save_model(str(model_dir))
    trainer.save_state()
    summary = {
        "stage": stage_name,
        "rows": len(train_frame),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "bf16_cpu_amp": use_bf16,
        "duration_seconds": duration,
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "metrics": {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))},
        "model_dir": str(model_dir),
    }
    atomic_write_json(output / "reports" / f"{stage_name}.json", summary)
    logger.info("%s complete: step=%d loss=%.6f seconds=%.2f", stage_name, summary["global_step"], summary["training_loss"], duration)
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
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32", copy=False)
    query_embeddings_full = model.encode(
        query_texts,
        batch_size=64,
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
- Base model: `{payload['base_model']}`
- Profile: `{profile['name']}`
- Train rows: `{dataset['split_rows']['train']}`
- Validation rows: `{dataset['split_rows']['validation']}`
- Test rows: `{dataset['split_rows']['test']}`
- Trainable encoder layers: `{profile['trainable_layers']}`
- Maximum sequence length: `{profile['max_seq_length']}`
- Batch size: `{profile['batch_size']}`
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


def main() -> int:
    args = parse_args()
    profile = profile_with_overrides(args)
    validate_profile(profile, args)
    output = args.output.resolve()
    if args.overwrite_output and output.exists():
        protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output in protected:
            raise RuntimeError(f"Refusing to delete protected output path: {output}")
        shutil.rmtree(output)
    logger = configure_logging(output)
    configured_threads = configure_runtime(args.threads, args.seed, logger)
    system = system_report(output, configured_threads)
    preflight(system, logger)
    use_bf16 = bool(system["cpu_has_bf16"]) if args.bf16 is None else bool(args.bf16)
    logger.info("CPU BF16 autocast: %s", "enabled" if use_bf16 else "disabled")

    data_path = discover_data_file(args.data, output / "working", logger)
    splits, dataset_meta = prepare_dataset(
        data_path=data_path,
        prepared_dir=output / "prepared",
        boilerplate_threshold=args.boilerplate_threshold,
        seed=args.seed,
        max_train_rows=profile.max_train_rows,
        logger=logger,
    )
    run_config = {
        "script_version": SCRIPT_VERSION,
        "base_model": args.base_model,
        "profile": dataclasses.asdict(profile),
        "stage1_lr": args.stage1_lr,
        "stage2_lr": args.stage2_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "min_hard_negative_margin": args.min_hard_negative_margin,
        "boilerplate_threshold": args.boilerplate_threshold,
        "bf16_cpu_amp": use_bf16,
        "seed": args.seed,
        "data_source_sha256": dataset_meta["cache_key"]["source_sha256"],
    }
    atomic_write_json(output / "run_config.json", run_config)
    if args.prepare_only:
        logger.info("Dataset preparation complete; --prepare-only requested, so training is skipped")
        return 0

    state_path = output / "run_state.json"
    state: dict[str, Any] = {}
    if state_path.exists() and args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_config") != run_config:
            raise RuntimeError(
                "Existing output was created with a different configuration. Use a new --output or --overwrite-output."
            )
    elif state_path.exists() and not args.resume:
        raise RuntimeError("Output already contains training state. Use --resume or --overwrite-output.")
    else:
        state = {
            "run_config": run_config,
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
        batch_size=64,
        output_path=output / "evaluation" / "validation",
    )
    test_bundle = build_eval_bundle(
        name="test",
        eval_frame=splits["test"],
        document_pool=document_pool,
        max_queries=profile.eval_queries,
        max_corpus=profile.eval_corpus,
        seed=args.seed + 1,
        batch_size=64,
        output_path=output / "evaluation" / "test",
    )

    stage_summaries: dict[str, Any] = {}
    stage_validation_metrics: dict[str, dict[str, float]] = {}
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
                base_metrics = evaluate_bundle(
                    model,
                    validation_bundle,
                    output / "evaluation" / "base",
                    logger,
                )
                atomic_write_json(output / "reports" / "base_validation.json", base_metrics)

            model, stage1_summary = train_stage(
                stage_name="stage1",
                model=model,
                train_frame=splits["train"],
                eval_frame=splits["validation"],
                evaluator=validation_bundle.evaluator,
                output=output,
                epochs=profile.stage1_epochs,
                batch_size=profile.batch_size,
                learning_rate=args.stage1_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                dims=profile.matryoshka_dims,
                include_negative=False,
                resume=args.resume,
                seed=args.seed,
                use_bf16=use_bf16,
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
                learning_rate=args.stage2_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                dims=profile.matryoshka_dims,
                include_negative=True,
                resume=args.resume,
                seed=args.seed + 1,
                use_bf16=use_bf16,
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
            missing_metric_stages = [
                stage_name
                for stage_name, metrics in (
                    ("stage1", stage_validation_metrics["stage1"]),
                    ("stage2", stage2_validation),
                )
                if primary_metric not in metrics
            ]
            if missing_metric_stages:
                available = {
                    "stage1": sorted(stage_validation_metrics["stage1"].keys()),
                    "stage2": sorted(stage2_validation.keys()),
                }
                raise RuntimeError(
                    f"Validation metric {primary_metric!r} is missing for {missing_metric_stages}. "
                    f"Available metrics: {available}"
                )
            stage1_score = float(stage_validation_metrics["stage1"][primary_metric])
            stage2_score = float(stage2_validation[primary_metric])
            if not math.isfinite(stage1_score) or not math.isfinite(stage2_score):
                raise RuntimeError(
                    f"Validation metric {primary_metric!r} must be finite: "
                    f"stage1={stage1_score}, stage2={stage2_score}"
                )
            selected_stage = "stage2" if stage2_score >= stage1_score else "stage1"
            selected_source = output / "models" / selected_stage
            final_dir = output / "models" / "final"
            clone_model_tree(selected_source, final_dir, logger)
            logger.info(
                "Validation selection: metric=%s stage1=%.6f stage2=%.6f selected=%s",
                primary_metric,
                stage1_score,
                stage2_score,
                selected_stage,
            )
            if not args.keep_intermediate:
                shutil.rmtree(output / "models" / "stage2", ignore_errors=True)
            state["stage2_complete"] = True
            state["stage2_completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            state["selected_stage"] = selected_stage
            state["selection_metric"] = primary_metric
            state["selection_scores"] = {"stage1": stage1_score, "stage2": stage2_score}
            atomic_write_json(state_path, state)
        elif not state.get("stage2_complete"):
            final_dir = output / "models" / "final"
            clone_model_tree(output / "models" / "stage1", final_dir, logger)
            state["stage2_complete"] = True
            state["stage2_skipped"] = True
            state["selected_stage"] = "stage1"
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
        logger=logger,
        include_confidence_calibration=True,
    )
    detailed_test = detailed_retrieval_metrics(
        final_model,
        test_bundle,
        dimensions=profile.matryoshka_dims,
        logger=logger,
        include_confidence_calibration=False,
    )

    final_payload: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "base_model": args.base_model,
        "profile": dataclasses.asdict(profile),
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
    logger.info("Training pipeline complete. Final model: %s", output / "models" / "final")
    logger.info("Report: %s", output / "reports" / "final_report.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted. Re-run with --resume to continue from the latest saved checkpoint.", file=sys.stderr)
        raise SystemExit(130)
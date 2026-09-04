#!/usr/bin/env python3
"""Model-free Qwen prepared-data and frozen-benchmark utility."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from qwen_train_vm import ids_manifest_sha256, qwen_language_bucket, resolve_prepared_language


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ids_from_rows(path: Path) -> list[str]:
    rows = read_rows(path)
    ids = [str(row.get("conversation_id", "")).strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError(f"Prepared JSONL contains a row without conversation_id: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Prepared JSONL contains duplicate conversation IDs: {path}")
    return ids


def write_ids(path: Path, ids: set[str] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in sorted(set(ids))), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit prepared Qwen JSONL without loading a model")
    parser.add_argument("--language-counts", type=Path, help="Print canonical bucket counts for this prepared JSONL")
    parser.add_argument("--extract-ids", type=Path, help="Extract sorted unique IDs from this prepared JSONL")
    parser.add_argument("--v1-test", type=Path, help="V1 prepared test JSONL for intersection")
    parser.add_argument("--v2-test", type=Path, help="V2 prepared test JSONL for intersection")
    parser.add_argument("--output", type=Path, help="Output ID manifest for --extract-ids or test intersection")
    args = parser.parse_args()
    if args.language_counts:
        rows = read_rows(args.language_counts)
        counts = Counter(qwen_language_bucket(row) for row in rows)
        result = {
            "path": str(args.language_counts),
            "conversation_count": len(rows),
            "bucket_counts": {bucket: counts.get(bucket, 0) for bucket in ("english", "tamil", "tanglish", "tamil-english-mixed")},
            "canonical_language_counts": Counter(resolve_prepared_language(row) for row in rows),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False, default=dict))
        return 0
    if args.extract_ids:
        if not args.output:
            parser.error("--extract-ids requires --output")
        ids = ids_from_rows(args.extract_ids)
        write_ids(args.output, ids)
        print(json.dumps({"output": str(args.output), "count": len(ids), "sha256": ids_manifest_sha256(ids)}, indent=2))
        return 0
    if args.v1_test or args.v2_test:
        if not args.v1_test or not args.v2_test or not args.output:
            parser.error("--v1-test and --v2-test together require --output")
        common = set(ids_from_rows(args.v1_test)) & set(ids_from_rows(args.v2_test))
        write_ids(args.output, common)
        print(json.dumps({"output": str(args.output), "count": len(common), "sha256": ids_manifest_sha256(common)}, indent=2))
        return 0
    parser.error("choose --language-counts, --extract-ids, or --v1-test/--v2-test")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Small local parity check for adapter, merged HF, and optional GGUF exports."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_train_vm import apply_template, generate_samples, stratified_generation_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hf_model(path: str, adapter: Path | None, offline: bool):
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=offline, low_cpu_mem_usage=True)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Qwen adapter/merged/GGUF export parity")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--merged-model", required=True, type=Path)
    parser.add_argument("--prepared-test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if bool(args.gguf) != bool(args.llama_cli):
        parser.error("--gguf and --llama-cli must be supplied together")
    rows = stratified_generation_rows(read_jsonl(args.prepared_test), args.sample_count, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.offline, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_model = hf_model(args.base_model, args.adapter, args.offline)
    adapter_samples = generate_samples(adapter_model, tokenizer, rows, len(rows), args.max_new_tokens, False)
    del adapter_model
    merged_tokenizer = AutoTokenizer.from_pretrained(str(args.merged_model), local_files_only=args.offline, trust_remote_code=False)
    if merged_tokenizer.pad_token_id is None:
        merged_tokenizer.pad_token = merged_tokenizer.eos_token
    merged_model = hf_model(str(args.merged_model), None, args.offline)
    merged_samples = generate_samples(merged_model, merged_tokenizer, rows, len(rows), args.max_new_tokens, False)
    del merged_model
    merged_by_id = {item["conversation_id"]: item for item in merged_samples}
    adapter_by_id = {item["conversation_id"]: item for item in adapter_samples}
    samples = []
    for row in rows:
        item = adapter_by_id.get(row["conversation_id"], {})
        merged = merged_by_id.get(row["conversation_id"], {})
        result = {
            "conversation_id": row["conversation_id"],
            "language": row.get("language"),
            "expected": item.get("expected", ""),
            "adapter_answer": item.get("generated", ""),
            "merged_answer": merged.get("generated", ""),
            "adapter_metrics": item,
            "merged_metrics": merged,
        }
        if args.gguf:
            prompt = apply_template(tokenizer, row["messages"][:-1], tokenize=False, add_generation_prompt=True)
            completed = subprocess.run(
                [str(args.llama_cli), "-m", str(args.gguf), "-p", str(prompt), "-n", str(args.max_new_tokens), "--temp", "0"],
                check=True, capture_output=True, text=True,
            )
            result["gguf_answer"] = completed.stdout.strip()
        samples.append(result)
    report = {
        "report_type": "qwen_export_parity",
        "evaluation_only": True,
        "sample_count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "deterministic": {"seed": args.seed, "do_sample": False, "enable_thinking": False},
        "samples": samples,
        "parity_claim": "No parity claim is made automatically; inspect answers and metrics for regressions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "sample_count": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

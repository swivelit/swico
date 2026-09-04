#!/usr/bin/env python3
"""Export an existing Swico PEFT adapter; this command does not train."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_train_vm import BASE_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an existing Qwen PEFT adapter (no training)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", type=Path, required=True, help="Existing PEFT adapter directory")
    parser.add_argument("--output", type=Path, required=True, help="Merged Hugging Face model directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing merged export")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--convert-script", type=Path, help="Optional llama.cpp convert_hf_to_gguf.py")
    parser.add_argument("--gguf-output", type=Path, help="Optional GGUF output path")
    parser.add_argument("--quantize-binary", type=Path, help="Optional llama.cpp llama-quantize binary")
    parser.add_argument("--quantization", default="Q4_K_M")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is non-empty; choose a new path or pass --overwrite: {args.output}")
    if args.gguf_output and not args.convert_script:
        raise SystemExit("--gguf-output requires --convert-script")
    print("EXPORT ONLY: loading base model and existing adapter; no training will run.")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.offline,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.offline, trust_remote_code=False)
    tokenizer.save_pretrained(args.output)
    result = {"training": False, "adapter": str(args.adapter), "merged_hf_model": str(args.output), "gguf": None}
    if args.convert_script:
        if not args.gguf_output:
            raise SystemExit("--convert-script also requires --gguf-output")
        args.gguf_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(args.convert_script), str(args.output), "--outfile", str(args.gguf_output)],
            check=True,
        )
        gguf_path = args.gguf_output
        if args.quantize_binary:
            quantized = gguf_path.with_name(f"{gguf_path.stem}-{args.quantization}.gguf")
            subprocess.run([str(args.quantize_binary), str(gguf_path), str(quantized), args.quantization], check=True)
            result["gguf_quantized"] = str(quantized)
        result["gguf"] = str(gguf_path)
    report_path = args.output / "export_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

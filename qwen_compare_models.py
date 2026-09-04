#!/usr/bin/env python3
"""Deterministic, evaluation-only Qwen base/adapter/champion comparison.

This script never calls Trainer and never updates weights. Models are loaded
and released sequentially so a CPU VM can compare a candidate safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_train_vm import (
    BASE_MODEL,
    CANONICAL_LANGUAGES,
    LANGUAGE_BUCKETS,
    assistant_only_tokens,
    generate_samples,
    resolve_prepared_language,
    stable_hash,
    stratified_generation_rows,
    summarize_generation_samples,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(base_model: str, adapter: Path | None, offline: bool):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=offline,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model


def teacher_forced_loss(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    max_seq_length: int,
) -> dict[str, Any]:
    totals = {language: {"loss_sum": 0.0, "token_count": 0} for language in CANONICAL_LANGUAGES}
    for row in rows:
        ids, labels = assistant_only_tokens(tokenizer, row["messages"], max_seq_length)
        input_ids = torch.tensor([ids], dtype=torch.long)
        label_ids = torch.tensor([labels], dtype=torch.long)
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention)
        logits = outputs.logits[:, :-1, :].float()
        shifted = label_ids[:, 1:]
        mask = shifted != -100
        if not bool(mask.any()):
            continue
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape_as(shifted)
        language = resolve_prepared_language(row)
        totals[language]["loss_sum"] += float(losses[mask].sum())
        totals[language]["token_count"] += int(mask.sum())
    result = {}
    total_loss = 0.0
    total_tokens = 0
    for language, values in totals.items():
        loss = values["loss_sum"] / values["token_count"] if values["token_count"] else None
        result[language] = {
            "supervised_token_count": values["token_count"],
            "loss": loss,
            "perplexity": math.exp(min(loss, 20.0)) if loss is not None else None,
        }
        total_loss += values["loss_sum"]
        total_tokens += values["token_count"]
    overall_loss = total_loss / total_tokens if total_tokens else None
    result["overall"] = {
        "supervised_token_count": total_tokens,
        "loss": overall_loss,
        "perplexity": math.exp(min(overall_loss, 20.0)) if overall_loss is not None else None,
    }
    return result


def metric_non_regression(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """A conservative objective-metric gate; it is not a factuality judge."""
    for language in LANGUAGE_BUCKETS:
        ref = reference.get("by_language", {}).get(language, {})
        cand = candidate.get("by_language", {}).get(language, {})
        if not ref.get("sample_count") or not cand.get("sample_count"):
            return False
        if cand.get("termination_rate", 0.0) < ref.get("termination_rate", 0.0):
            return False
        if cand.get("max_token_hit_rate", 1.0) > ref.get("max_token_hit_rate", 1.0):
            return False
        if cand.get("mean_repeated_4gram_ratio", 1.0) > ref.get("mean_repeated_4gram_ratio", 1.0):
            return False
        if cand.get("script_adherence_rate", 0.0) < ref.get("script_adherence_rate", 0.0):
            return False
    return True


def metric_strict_improvement(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    ref = reference.get("overall", {})
    cand = candidate.get("overall", {})
    return (
        cand.get("termination_rate", 0.0) > ref.get("termination_rate", 0.0)
        or cand.get("max_token_hit_rate", 1.0) < ref.get("max_token_hit_rate", 1.0)
        or cand.get("mean_repeated_4gram_ratio", 1.0) < ref.get("mean_repeated_4gram_ratio", 1.0)
        or cand.get("script_adherence_rate", 0.0) > ref.get("script_adherence_rate", 0.0)
    )


def candidate_status_from_metrics(
    base_generation: dict[str, Any],
    adapter_generation: dict[str, Any],
    champion_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative status object from fakeable objective metrics."""
    beats_base = metric_non_regression(base_generation, adapter_generation) and metric_strict_improvement(
        base_generation, adapter_generation
    )
    does_not_regress_champion = (
        metric_non_regression(champion_generation, adapter_generation)
        if champion_generation is not None
        else None
    )
    automatic_gate = bool(
        adapter_generation["health"]["healthy"]
        and beats_base
        and (does_not_regress_champion is not False)
    )
    return {
        "training_completed": None,
        "generation_healthy": adapter_generation["health"]["healthy"],
        "beats_base": beats_base,
        "does_not_regress_champion": does_not_regress_champion,
        "manual_quality_review_required": True,
        "automatic_promotion_gate_passed": automatic_gate,
        "promotion_eligible": False,
        "reason": "Automatic metrics are regression gates only; factual quality still requires manual review.",
    }


def compare_one_model(
    label: str,
    base_model: str,
    adapter: Path | None,
    tokenizer,
    rows: list[dict[str, Any]],
    count: int,
    max_new_tokens: int,
    seed: int,
    offline: bool,
    include_loss: bool,
    max_seq_length: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    selected = stratified_generation_rows(rows, count, seed)
    model = load_model(base_model, adapter, offline)
    samples = generate_samples(model, tokenizer, selected, len(selected), max_new_tokens, False)
    generation = summarize_generation_samples(samples)
    losses = teacher_forced_loss(model, tokenizer, selected, max_seq_length) if include_loss else None
    model_report = {
        "label": label,
        "adapter_path": str(adapter) if adapter else None,
        "generation_evaluation": generation,
        "teacher_forced_loss": losses,
        "sample_count": len(samples),
    }
    sample_map = {sample["conversation_id"]: sample for sample in samples}
    del model
    gc.collect()
    return model_report, sample_map


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen base/adapter comparison (evaluation only)",
        "",
        "No training or weight updates were performed. Models were loaded sequentially.",
        "",
        f"- Prepared test set: `{report['prepared_test']}`",
        f"- Base model: `{report['base_model']}`",
        f"- Samples: {report['generation_sample_count']}",
        f"- max_new_tokens: {report['generation_max_new_tokens']}",
        f"- Manual quality review required: **{report['candidate_status']['manual_quality_review_required']}**",
        "",
        "## Model metrics",
        "",
    ]
    for label, values in report["models"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"Generation: `{values['generation_evaluation']}`")
        if values.get("teacher_forced_loss"):
            lines.append(f"Teacher-forced loss/perplexity: `{values['teacher_forced_loss']}`")
        lines.append("")
    lines.extend(["## Same-prompt generations", ""])
    for sample in report["samples"]:
        lines.extend(
            [
                f"### {sample['conversation_id']} ({sample['language']})",
                f"Prompt: {sample['prompt']}",
                "",
                f"Expected: {sample['expected']}",
                "",
                f"Base: {sample['base_answer']}",
                "",
                f"Adapter: {sample['adapter_answer']}",
                "",
            ]
        )
        if "champion_answer" in sample:
            lines.extend([f"Champion: {sample['champion_answer']}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation-only Qwen base/adapter/champion comparison")
    parser.add_argument("--prepared-test", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--champion-adapter", type=Path, default=Path(os.environ["SWICO_QWEN_CHAMPION_ADAPTER"]) if os.environ.get("SWICO_QWEN_CHAMPION_ADAPTER") else None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-loss", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.sample_count <= 0 or args.max_new_tokens <= 0 or args.max_seq_length <= 0:
        parser.error("sample-count, max-new-tokens and max-seq-length must be positive")
    return args


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.prepared_test)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.offline, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    models: dict[str, Any] = {}
    sample_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for label, adapter in (
        ("base", None),
        ("adapter", args.candidate_adapter),
        ("champion", args.champion_adapter),
    ):
        if label == "champion" and adapter is None:
            continue
        model_report, sample_map = compare_one_model(
            label, args.base_model, adapter, tokenizer, rows, args.sample_count,
            args.max_new_tokens, args.seed, args.offline, args.include_loss, args.max_seq_length,
        )
        models[label] = model_report
        sample_maps[label] = sample_map
    base_generation = models["base"]["generation_evaluation"]
    adapter_generation = models["adapter"]["generation_evaluation"]
    has_champion = "champion" in models
    candidate_status = candidate_status_from_metrics(
        base_generation,
        adapter_generation,
        models["champion"]["generation_evaluation"] if has_champion else None,
    )
    sample_ids = sorted(sample_maps["adapter"], key=lambda value: stable_hash(value, args.seed + 2))
    samples = []
    for conversation_id in sample_ids:
        adapter_sample = sample_maps["adapter"][conversation_id]
        base_sample = sample_maps["base"].get(conversation_id, {})
        item = {
            "conversation_id": conversation_id,
            "language": adapter_sample.get("language"),
            "prompt": adapter_sample.get("prompt_last_message", ""),
            "expected": adapter_sample.get("expected", ""),
            "base_answer": base_sample.get("generated", ""),
            "adapter_answer": adapter_sample.get("generated", ""),
            "base_metrics": base_sample,
            "adapter_metrics": adapter_sample,
        }
        if has_champion:
            champion_sample = sample_maps["champion"].get(conversation_id, {})
            item["champion_answer"] = champion_sample.get("generated", "")
            item["champion_metrics"] = champion_sample
        samples.append(item)
    report = {
        "report_type": "qwen_base_adapter_champion_comparison",
        "evaluation_only": True,
        "prepared_test": str(args.prepared_test),
        "base_model": args.base_model,
        "generation_sample_count": args.sample_count,
        "generation_max_new_tokens": args.max_new_tokens,
        "deterministic": {"seed": args.seed, "do_sample": False, "enable_thinking": False, "same_selected_rows": True},
        "models": models,
        "samples": samples,
        "candidate_status": candidate_status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "comparison_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output_dir / "comparison_report.json"), "markdown": str(args.output_dir / "comparison_report.md"), "candidate_status": report["candidate_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

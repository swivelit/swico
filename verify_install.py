#!/usr/bin/env python3
"""Offline compatibility check for the installed training stack."""

from __future__ import annotations

import inspect
import logging
import tempfile
from pathlib import Path

from sentence_transformers import SentenceTransformerTrainer
from transformers import EarlyStoppingCallback, TrainerCallback

import train_vm


with tempfile.TemporaryDirectory(prefix="swico-training-verify-") as directory:
    root = Path(directory)
    args = train_vm.make_training_args(
        output_dir=root / "trainer",
        epochs=3.0,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        logging_steps=25,
        save_total_limit=2,
        primary_metric="validation_cosine_ndcg@10",
        load_best_model_at_end=True,
        seed=42,
        use_bf16=False,
    )

    callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001,
    )
    trainer_parameters = inspect.signature(SentenceTransformerTrainer.__init__).parameters
    assert "callbacks" in trainer_parameters
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "validation_cosine_ndcg@10"
    assert str(args.eval_strategy).lower().endswith("epoch")
    assert str(args.save_strategy).lower().endswith("epoch")
    assert callback.early_stopping_patience == 2

    guard = train_vm.ResourceGuardCallback(
        output_path=root / "guard.json",
        logger=logging.getLogger("verify"),
        interval_steps=5,
        emergency_available_memory_gib=0.01,
        max_process_rss_gib=None,
    )
    assert isinstance(guard, TrainerCallback)

    parsed = train_vm.parse_args()
    profile = train_vm.profile_with_overrides(parsed)
    train_vm.validate_profile(profile, parsed)
    parsed.output_root = root / "run-root"
    parsed.output = None
    parsed.run_mode = "auto"
    parsed.run_id = None
    parsed.run_label = "verify"

    first = train_vm.resolve_run_context(parsed, profile)
    train_vm.atomic_write_json(
        first.output / "run_manifest.json",
        {"invocation_fingerprint": first.invocation_fingerprint},
    )
    resumed = train_vm.resolve_run_context(parsed, profile)
    assert resumed.resumed is True
    assert resumed.output == first.output

    train_vm.atomic_write_json(
        first.output / "run_state.json",
        {"final_evaluation_complete": True, "completed_at": "verified"},
    )
    second = train_vm.resolve_run_context(parsed, profile)
    assert second.resumed is False
    assert second.output != first.output

print("Swico trainer compatibility check passed.")

#!/usr/bin/env python3
"""Offline compatibility check for the installed training stack."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from sentence_transformers import SentenceTransformerTrainer
from transformers import EarlyStoppingCallback

import train_vm


with tempfile.TemporaryDirectory(prefix="swico-training-verify-") as directory:
    args = train_vm.make_training_args(
        output_dir=Path(directory),
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

print("Swico trainer compatibility check passed.")

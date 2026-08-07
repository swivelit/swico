
## 2026-08-07 - Qwen smoke tokenization fix

- Increased the Qwen smoke profile sequence length from 256 to 512 tokens so multilingual conversations do not lose assistant supervision during truncation.
- Avoid requesting tokenizer assistant masks when the active Qwen chat template does not expose generation spans, removing the misleading Transformers warning and using the compatibility fallback directly.
# Changelog

## 4.0.0 — timestamped experiments and adaptive resource protection

- Added timestamped run directories under `SWICO_OUTPUT_ROOT/runs/`.
- Added automatic compatible-incomplete-run resume and automatic new-run creation after completion.
- Added `latest`, `latest-completed` and text pointer files.
- Removed the requirement to delete a fixed output directory between experiments.
- Clarified that `SWICO_STAGE1_EPOCHS` and `SWICO_STAGE2_EPOCHS` are maximum caps.
- Added safe memory-aware tuning of evaluation/mining batch and chunk controls.
- Added explicit aggressive training-batch tuning as an opt-in quality-sensitive mode.
- Added a runtime memory guard that requests checkpointing and a graceful pause under memory pressure.
- Added base-model validation protection so a degraded trained candidate is not promoted.
- Added persistent `autotune.json` and `run_manifest.json` reports.
- Preserved strict env parsing, early stopping, best-checkpoint restoration, stage selection and configuration-safe resume.

## 3.0.0 — environment configuration and early stopping

- Added typed `training.env` and strict validation.
- Added validation NDCG early stopping and best-model restoration.
- Added callback-state restoration and incompatible-resume rejection.

## 2026-08-07 — dual-model CPU training

- Kept the advanced multilingual-E5 retrieval trainer and pointed its default data path at the bundled E5 CSV.
- Added `qwen_train_vm.py` for Qwen/Qwen3-0.6B conversational LoRA SFT.
- Added conversation-level leakage-resistant splitting and assistant-only causal-LM labels.
- Added Qwen BF16 detection, gradient accumulation/checkpointing, early stopping, best checkpoint loading, resume, memory guards and held-out evaluation.
- Added adapter export and optional merged-model export.
- Added `qwen_training.env`, `qwen_training.env.example`, `run_qwen_training.sh`, and `run_e5_training.sh`.
- Added PEFT to the CPU requirements and setup verification.
- Added Qwen source-contract tests and documentation.

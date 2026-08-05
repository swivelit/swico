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

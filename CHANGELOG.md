# Changelog

## 3.0.0 — environment configuration and early stopping

- Added typed `training.env` and `training.env.example` files.
- Added strict validation for unknown keys, malformed booleans, numeric ranges, split ratios and contradictory settings.
- Added precedence: CLI > shell environment > env file > profile defaults.
- Added `auto` values for profile-controlled settings.
- Added `--print-config` and custom `--env-file` support.
- Exposed model, data, optimizer, scheduler, batch, evaluation, mining, HNSW, Matryoshka, runtime and preflight controls through the env file.
- Added validation NDCG@10 early stopping for both training stages.
- Added configurable patience and minimum improvement.
- Added automatic best-checkpoint restoration before stage model saving.
- Added callback-state restoration on resume.
- Kept stage-1-versus-stage-2 validation selection as an independent safeguard.
- Added comprehensive run-configuration persistence and incompatible-resume rejection.
- Removed the Transformers 5 warmup deprecation path.
- Added installation-time unit tests and real API compatibility verification.
- Added stage reports for best metric, best checkpoint, completed epochs and early-stop status.

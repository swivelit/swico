# Validation record

## Static and repository checks

- Python syntax compilation for trainer, configuration, verifier and tests
- Bash syntax validation for setup and launcher scripts
- strict env-file parsing and unknown-key rejection
- AST verification that public function calls match keyword signatures
- source checks for early stopping and best-model restoration
- source checks for timestamped run allocation and compatible resume
- source checks that safe autotuning leaves quality-sensitive settings unchanged
- source checks for the runtime resource guard and baseline quality gate

## Runtime checks required on the target VM

Run:

```bash
./setup_vm.sh
SWICO_PROFILE=smoke ./run_vm_training.sh
```

The previous v3 smoke log already demonstrated working CPU BF16, training, validation, saving and reload on the target VM. Version 4 changes run allocation, resource planning and quality promotion, so one v4 smoke run is required.

After completion:

```bash
find training_artifacts/e5-small-swico/runs -maxdepth 1 -mindepth 1 -type d -printf '%f\n'
cat training_artifacts/e5-small-swico/latest/autotune.json
cat training_artifacts/e5-small-swico/latest/run_manifest.json
cat training_artifacts/e5-small-swico/latest/reports/final_report.md
```

Run the smoke command a second time. It must create a different timestamped directory rather than reuse the completed first run.

## Qwen3-0.6B LoRA checks

Static validation also covers:

- Qwen trainer Python syntax
- Qwen launcher Bash syntax
- strict registration of every `SWICO_QWEN_*` environment key
- conversation-level train/validation/test splitting
- assistant-only label construction
- LoRA target configuration
- early stopping / best model selection
- resumable timestamped Qwen runs
- memory guard callback
- adapter and final-report export paths
- native Qwen assistant-mask and end-of-message supervision, including truncation protection
- stratified 30-sample generation metrics and unhealthy-candidate reporting

Target-VM validation:

```bash
./setup_vm.sh
./run_qwen_training.sh --print-config
SWICO_QWEN_PREPARE_ONLY=true ./run_qwen_training.sh
SWICO_QWEN_PROFILE=smoke ./run_qwen_training.sh
```

After the smoke run:

```bash
cat training_artifacts/qwen3-0.6b-swico/latest/reports/final_report.md
cat training_artifacts/qwen3-0.6b-swico/latest/run_config.json
cat training_artifacts/qwen3-0.6b-swico/latest/system.json
```

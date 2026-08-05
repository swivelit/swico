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

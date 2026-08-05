# Validation record

## Repository checks completed before packaging

- Python syntax compilation for `train_vm.py`, `training_config.py`, `verify_install.py` and tests
- Bash syntax checks for `setup_vm.sh` and `run_vm_training.sh`
- seven unit/source-contract tests passed
- strict env-file parsing, shell override precedence, `auto` values, boolean parsing and unknown-key rejection tested
- function-call keyword contracts checked through the Python AST
- early-stopping callback, best-model arguments, epoch-aligned save/evaluation strategy and Transformers-v5 warmup compatibility checked in source contracts
- all 90,162 source rows processed through the updated cleaning and component-isolated splitting path
- resulting rows: train 45,012; validation 8,750; test 8,751
- zero exact normalized query, passage and connected-component overlap verified across train, validation and test
- simulated Transformers-5 argument construction selected float `warmup_steps`, epoch evaluation/saving and validation NDCG best-model selection

## Installation-time checks included in the package

`./setup_vm.sh` now performs the following after dependency installation:

```bash
python -m unittest discover -s tests -v
python verify_install.py
```

`verify_install.py` instantiates the real installed `SentenceTransformerTrainingArguments`, checks the installed trainer's callback support, validates epoch-aligned evaluation and saving, confirms best-model loading and constructs `EarlyStoppingCallback` without downloading a model.

## Required VM validation

Run:

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh
```

The user's previous v2.1.1 smoke run already established that this VM supports CPU BF16 and can train, evaluate, save and reload the E5 model. Version 3 changes configuration and stopping behavior, so one new smoke run is still required before starting the long VM profile.

After the smoke test, inspect:

```bash
cat training_artifacts/e5-small-swico/reports/stage1.json
```

For the one-epoch smoke profile, early stopping is configured but intentionally inactive because there is only one possible validation epoch. The full `vm` profile activates early stopping because its maximum epoch counts are greater than one.

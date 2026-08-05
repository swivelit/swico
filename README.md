# Swico CPU Retrieval Model Training

This repository fine-tunes `intfloat/multilingual-e5-small` for Swico retrieval/RAG on a CPU-only Google Compute Engine VM. It creates a domain-adapted embedding model; it does not train a general-purpose chat LLM from scratch.

## Target machine

The default `vm` profile is designed for:

- 4 physical CPU cores / 8 logical CPUs
- Intel Xeon Platinum with AVX2, AVX-512 and CPU BF16 support
- about 29 GiB RAM
- no GPU
- Debian 13 / Python 3.13

## Professional configuration model

All operational settings and training hyperparameters are exposed in:

```text
training.env
```

You should not edit Python source to tune a run.

Configuration precedence is:

1. command-line argument
2. shell environment variable
3. `training.env`
4. selected profile default

Values marked `auto` are inherited from the selected profile. This makes the following command safe even though `training.env` normally selects `vm`:

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh
```

The strict configuration loader rejects misspelled `SWICO_` keys, invalid booleans, malformed number lists, unsafe ranges, incompatible split fractions and contradictory early-stopping settings before training starts.

### Edit the configuration

```bash
nano training.env
```

Save with `Ctrl+O`, press `Enter`, then exit with `Ctrl+X`.

Validate and print the complete effective configuration without reading the dataset or downloading a model:

```bash
./run_vm_training.sh --print-config
```

Use a different configuration file when needed:

```bash
./run_vm_training.sh --env-file experiments/run-02.env --print-config
```

## Early stopping and overfitting protection

The previous version did not stop a stage early. It only compared the completed stage-1 and stage-2 models afterward.

Version 3 adds validation-driven early stopping inside both stages:

- validation runs at the end of every epoch
- the monitored metric is validation cosine NDCG@10
- `SWICO_EARLY_STOPPING_PATIENCE=2` allows two consecutive validation evaluations without sufficient improvement
- `SWICO_EARLY_STOPPING_THRESHOLD=0.001` requires an absolute improvement greater than 0.001
- `SWICO_LOAD_BEST_MODEL_AT_END=true` restores the best checkpoint before the stage model is saved
- early-stopping callback state is restored when resuming from a checkpoint
- stage 1 and stage 2 are still compared after training, and the better validation model is promoted to `models/final/`
- the held-out test split is used only for the final unbiased report, not for stopping or model selection

Early stopping detects a lack of validation improvement; it cannot mathematically prove that every form of overfitting has been eliminated. The combination of component-isolated splits, best-checkpoint restoration, early stopping and stage-level selection is the appropriate safeguard for this pipeline.

Default maximum epochs are intentionally upper bounds rather than mandatory work:

- smoke: stage 1 = 1, stage 2 disabled
- vm: stage 1 = 5, stage 2 = 4
- full: stage 1 = 5, stage 2 = 4

Training can finish earlier when the validation metric stops improving.

## Training architecture

The pipeline includes:

- E5 `query: ` and `passage: ` formatting
- repeated-boilerplate removal and exact-pair deduplication
- connected-component train/validation/test splitting with zero normalized query, passage or component leakage
- partial fine-tuning of the final encoder layers
- no-duplicate batches
- Multiple Negatives Ranking Loss
- configurable Matryoshka dimensions and weights
- stage-1 clean-pair training
- stage-2 guarded hard-negative curriculum
- FAISS HNSW candidate retrieval with deterministic fallback
- validation-based early stopping and best-model restoration
- validation selection between stage 1 and stage 2
- bounded retrieval evaluation, confidence calibration and latency reporting
- resumable checkpoints with callback-state restoration
- strict run-configuration matching before resume
- atomic JSON state/report writes
- disk-efficient final-model hard links with copy fallback
- completed-checkpoint and intermediate-model cleanup

## Install or upgrade the VM environment

Run inside the extracted repository:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv unzip
chmod +x setup_vm.sh run_vm_training.sh verify_install.py
./setup_vm.sh
```

It is safe to rerun `./setup_vm.sh` over the existing `.venv`. It installs any new requirement, runs the unit tests and performs an offline compatibility check against the installed Sentence Transformers and Transformers APIs.

## Smoke test

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh
```

A successful smoke run ends with:

```text
Training pipeline complete. Final model: .../models/final
```

Remove only the smoke output before the real run:

```bash
rm -rf training_artifacts/e5-small-swico
```

## Real VM training

Review the effective configuration:

```bash
./run_vm_training.sh --print-config
```

Start in the background:

```bash
nohup ./run_vm_training.sh > training.log 2>&1 &
```

Monitor:

```bash
tail -f training.log
```

Press `Ctrl+C` to leave the log view. The background training process continues.

Check process, memory and disk:

```bash
ps aux | grep '[t]rain_vm.py'
free -h
df -h /
du -sh .venv .hf_cache training_artifacts training.log 2>/dev/null
```

## Resume behavior

Rerun the same command:

```bash
nohup ./run_vm_training.sh > training.log 2>&1 &
```

The pipeline resumes from the latest compatible checkpoint. If a training-critical env value changed, it refuses to resume and instructs you to use a new output directory or explicitly overwrite the old run. This prevents mixing checkpoints created with different learning rates, data splits, model layers or loss settings.

For a new experiment, change:

```text
SWICO_OUTPUT_DIR=training_artifacts/e5-small-swico-run-02
```

## Important `training.env` controls

Common values:

```text
SWICO_PROFILE=vm
SWICO_MAX_TRAIN_ROWS=auto
SWICO_BATCH_SIZE=auto
SWICO_MAX_SEQ_LENGTH=auto
SWICO_TRAINABLE_LAYERS=auto
SWICO_STAGE1_EPOCHS=auto
SWICO_STAGE2_EPOCHS=auto
SWICO_STAGE1_LR=0.00002
SWICO_STAGE2_LR=0.000008
SWICO_WEIGHT_DECAY=0.01
SWICO_WARMUP_RATIO=0.06
SWICO_EARLY_STOPPING=true
SWICO_EARLY_STOPPING_PATIENCE=2
SWICO_EARLY_STOPPING_THRESHOLD=0.001
SWICO_LOAD_BEST_MODEL_AT_END=true
```

All supported controls, including split ratios, mining parameters, HNSW settings, loss scale, Matryoshka weights, evaluation sizes, CPU threads and preflight limits, are documented directly in `training.env`.

## Outputs

```text
training_artifacts/e5-small-swico/models/final/
training_artifacts/e5-small-swico/reports/final_report.md
training_artifacts/e5-small-swico/reports/final_report.json
training_artifacts/e5-small-swico/reports/stage1.json
training_artifacts/e5-small-swico/reports/stage2.json
training_artifacts/e5-small-swico/run_config.json
training_artifacts/e5-small-swico/run_state.json
training_artifacts/e5-small-swico/training.log
```

Each stage report records:

- maximum and completed epochs
- whether early stopping occurred
- monitored validation metric
- best validation score
- best checkpoint path
- training loss and runtime
- effective batch size

## Stop cleanly

```bash
pkill -INT -f 'train_vm.py'
```

Run the same training command again to resume.

## Start over intentionally

Either change `SWICO_OUTPUT_DIR`, or run:

```bash
./run_vm_training.sh --overwrite-output
```

`--overwrite-output` refuses to delete protected paths such as `/`, your home directory or the repository root.

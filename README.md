# Swico CPU Retrieval Model Training

This repository fine-tunes `intfloat/multilingual-e5-small` for Swico retrieval/RAG on a CPU-only Google Compute Engine VM. It creates a domain-adapted embedding model; it does not train a general-purpose chat LLM from scratch.

## Target machine

The default `vm` profile is designed for:

- 4 physical CPU cores / 8 logical CPUs
- Intel Xeon Platinum with AVX2, AVX-512 and CPU BF16 support
- about 29 GiB RAM
- no GPU
- Debian 13 / Python 3.13

## Epoch controls

The maximum epoch caps are controlled by:

```text
SWICO_STAGE1_EPOCHS
SWICO_STAGE2_EPOCHS
```

With `auto`, the selected profile supplies the values:

- `smoke`: stage 1 = 1, stage 2 = 0
- `vm`: stage 1 = 5, stage 2 = 4
- `full`: stage 1 = 5, stage 2 = 4

These are maximums, not mandatory work. When early stopping is enabled, validation runs at the end of every epoch and training stops after the configured number of non-improving validation evaluations.

```text
SWICO_EARLY_STOPPING=true
SWICO_EARLY_STOPPING_PATIENCE=2
SWICO_EARLY_STOPPING_THRESHOLD=0.001
SWICO_LOAD_BEST_MODEL_AT_END=true
```

The best validation checkpoint is restored before a stage model is saved. The test set is not used for stopping or model selection.

## Timestamped run directories

The default output root is:

```text
training_artifacts/e5-small-swico
```

Every new completed experiment receives its own UTC timestamped directory:

```text
training_artifacts/e5-small-swico/runs/20260805T092530Z_vm/
```

Optional labels appear at the end:

```text
SWICO_RUN_LABEL=larger-eval-batch
```

which creates a name such as:

```text
20260805T092530Z_vm_larger-eval-batch
```

The default lifecycle is:

```text
SWICO_RUN_MODE=auto
SWICO_RESUME=true
```

`auto` resumes the newest compatible incomplete run. When the newest compatible run is complete, the next command creates a new timestamped run. Completed runs are never silently reused or deleted.

Convenience links are maintained:

```text
training_artifacts/e5-small-swico/latest
training_artifacts/e5-small-swico/latest-completed
```

Text pointer files are also written for filesystems where symlinks are unavailable.

To force a new run even when an incomplete compatible run exists:

```bash
SWICO_RUN_MODE=new ./run_vm_training.sh
```

To require an existing compatible incomplete run:

```bash
SWICO_RUN_MODE=resume-latest ./run_vm_training.sh
```

## Safe adaptive tuning

Not every hyperparameter should be increased automatically. Epochs, learning rate, sequence length, trainable layers, loss scale, hard-negative semantics and the physical training batch can change model quality. Increasing all of them is not a valid or accuracy-preserving optimization.

The default planner therefore uses:

```text
SWICO_AUTOTUNE=true
SWICO_AUTOTUNE_MODE=safe
```

Safe mode uses current available RAM and configured reserve to increase only throughput/resource controls that do not change the training objective:

- evaluation encoding batch
- hard-negative encoding batch
- mining chunk size
- evaluation corpus chunk size

It does not alter:

- maximum epochs
- learning rates
- sequence length
- trainable layers
- loss or Matryoshka settings
- hard-negative selection thresholds
- physical training batch

The resolved plan is saved in every run as:

```text
autotune.json
```

Aggressive mode may raise the physical training batch:

```text
SWICO_AUTOTUNE_MODE=aggressive
SWICO_AUTOTUNE_TRAIN_BATCH=true
```

This is opt-in because Multiple Negatives Ranking Loss uses the physical batch as the negative set. Changing it may improve or reduce quality; identical accuracy cannot be guaranteed.

## Memory protection

Preflight refuses to start below the configured RAM and disk reserves. During training, a resource callback checks system available memory and process RSS every few optimizer steps:

```text
SWICO_MEMORY_GUARD=true
SWICO_MEMORY_GUARD_INTERVAL_STEPS=5
SWICO_EMERGENCY_AVAILABLE_MEMORY_GIB=4.0
SWICO_MAX_PROCESS_RSS_GIB=auto
```

When the emergency threshold is reached, it requests a checkpoint and gracefully pauses the run. Rerunning the same command resumes the compatible timestamped run. This substantially reduces OOM risk, but no userspace program can guarantee survival from a sudden kernel OOM kill or another process consuming all RAM between checks.

## Accuracy protection

The original base model is evaluated on the same validation bundle before fine-tuning. The final candidate is selected from stage 1, stage 2 and the original base model:

```text
SWICO_BASELINE_PROTECTION=true
SWICO_MIN_VALIDATION_GAIN=0.0
```

If every trained candidate is worse than the base validation NDCG, the base model is promoted instead. This protects the measured held-out validation score; it cannot guarantee identical behavior on every unseen real-world query.

## Configuration

All operational controls and hyperparameters are in:

```text
training.env
```

Precedence is:

1. command-line argument
2. shell environment variable
3. `training.env`
4. profile default

Values marked `auto` inherit a profile or automatic decision. Strict mode rejects misspelled `SWICO_` keys.

Edit:

```bash
nano training.env
```

Validate without reading the dataset or downloading a model:

```bash
./run_vm_training.sh --print-config
```

## Install or upgrade

```bash
sudo apt-get update
sudo apt-get install -y python3-venv unzip
chmod +x setup_vm.sh run_vm_training.sh verify_install.py
./setup_vm.sh
```

It is safe to rerun over the existing `.venv` and `.hf_cache`.

## Smoke test

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh
```

There is no need to delete the smoke output. The next completed run receives a different timestamped directory.

Inspect the newest run:

```bash
cat training_artifacts/e5-small-swico/latest/reports/final_report.md
```

## Real training

```bash
nohup ./run_vm_training.sh > launcher.log 2>&1 &
```

Monitor the trainer-owned log inside the active run:

```bash
tail -f training_artifacts/e5-small-swico/latest/training.log
```

Check resources:

```bash
free -h
df -h /
ps aux | grep '[t]rain_vm.py'
```

## Important defaults

```text
SWICO_STAGE1_EPOCHS=auto
SWICO_STAGE2_EPOCHS=auto
SWICO_EARLY_STOPPING=true
SWICO_EARLY_STOPPING_PATIENCE=2
SWICO_RUN_MODE=auto
SWICO_AUTOTUNE_MODE=safe
SWICO_AUTOTUNE_TRAIN_BATCH=false
SWICO_MEMORY_GUARD=true
SWICO_BASELINE_PROTECTION=true
```

## Run outputs

Each run contains:

```text
models/final/
reports/final_report.md
reports/final_report.json
reports/stage1.json
reports/stage2.json
autotune.json
run_manifest.json
run_config.json
run_state.json
system.json
training.log
```

## Stop and resume

Stop cleanly:

```bash
pkill -INT -f 'train_vm.py'
```

Resume:

```bash
nohup ./run_vm_training.sh > launcher.log 2>&1 &
```

The `auto` run mode resumes only an incomplete run whose configured training fingerprint matches. A changed learning rate, data split, model layer count or other training-critical value creates a new timestamped experiment instead of mixing incompatible checkpoints.

---

# Qwen3-0.6B Conversational Training

The repository also trains `Qwen/Qwen3-0.6B` as the generative conversational half of the Swico stack. E5 remains the retriever; Qwen generates responses.

## Dataset

The bundled conversational dataset is:

```text
data/qwen_dataset - qwen_dataset.csv
```

Required CSV columns:

```text
conversation_id,turn_index,role,content,language
```

Messages with the same `conversation_id` are grouped into one conversation. Splitting happens at the complete-conversation level so turns from one conversation cannot leak across train, validation and test sets.

## Qwen training method

The Qwen trainer uses:

- Qwen's own chat template
- normal chat mode (`enable_thinking=false` when supported by the tokenizer)
- assistant-response-only causal language-model loss
- LoRA / PEFT rather than full-weight fine-tuning
- validation loss for early stopping and best-checkpoint restoration
- held-out test loss and perplexity
- deterministic held-out sample generations
- CPU BF16 when the machine supports it
- gradient accumulation and optional gradient checkpointing
- timestamped compatible resume
- memory guard and disk/RAM preflight checks

The default LoRA targets are:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

## Qwen configuration

Edit:

```bash
nano qwen_training.env
```

Validate configuration without loading the model:

```bash
./run_qwen_training.sh --print-config
```

Prepare and validate the dataset without downloading Qwen:

```bash
SWICO_QWEN_PREPARE_ONLY=true ./run_qwen_training.sh
```

## Qwen smoke test

```bash
SWICO_QWEN_PROFILE=smoke ./run_qwen_training.sh
```

Monitor:

```bash
tail -f training_artifacts/qwen3-0.6b-swico/latest/training.log
```

Inspect the report:

```bash
cat training_artifacts/qwen3-0.6b-swico/latest/reports/final_report.md
```

## Qwen real training

```bash
nohup ./run_qwen_training.sh > qwen-launcher.log 2>&1 &
```

The default `vm` profile is intentionally conservative for an 8-logical-CPU / ~29 GiB RAM VM:

```text
max conversations = 12000
epoch cap = 3
physical batch = 1
gradient accumulation = 16
max sequence length = 512
LoRA rank = 8
LoRA alpha = 16
LoRA dropout = 0.05
```

Early stopping can finish before the epoch cap.

## Qwen outputs

Each Qwen run stores:

```text
training_artifacts/qwen3-0.6b-swico/runs/<timestamp>_<profile>/
  models/adapter/
  prepared/train.jsonl
  prepared/validation.jsonl
  prepared/test.jsonl
  reports/final_report.md
  reports/final_report.json
  run_manifest.json
  run_config.json
  run_state.json
  system.json
  training.log
```

By default only the LoRA adapter is exported. This is the recommended CPU-VM behavior because it minimizes disk and memory use. To also create a merged standalone Qwen model after training:

```text
SWICO_QWEN_MERGE_ADAPTER=true
```

Merging temporarily consumes more RAM.

## Which trainer should I run?

E5 retriever:

```bash
./run_e5_training.sh
```

Qwen conversational generator:

```bash
./run_qwen_training.sh
```

You should train and evaluate them independently. At application runtime, E5 retrieves relevant context and the fine-tuned Qwen model uses that context to produce the conversational response.

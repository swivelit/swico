# Swico CPU Retrieval Model Training

This repository fine-tunes `intfloat/multilingual-e5-small` for Swico retrieval/RAG on a CPU-only VM. It produces a domain-adapted embedding model; it does not train a general-purpose chat LLM from scratch.

## Target VM

The default `vm` profile is designed for the supplied Google Compute Engine machine:

- 4 physical CPU cores / 8 logical CPUs
- Intel Xeon Platinum 8581C
- AVX2, AVX-512, VNNI, AMX and BF16 instructions
- about 29 GiB RAM
- no GPU and no swap
- about 6.6 GiB free root-disk space before installation

## What was wrong with the previous pipeline

The old `train_vm.py` used 16 CPU threads, 80 epochs, batch-256 mining/encoding, four data-loader workers, full-corpus evaluation during training, multiple large checkpoints, and duplicate ONNX/Hugging Face exports. Those choices could make this 4-core VM extremely slow and could exhaust its 10 GiB root disk.

The uploaded repository also contained the same dataset twice. The ZIP copy used the malformed name `train_augmented .csv`. The revised package keeps only the canonical `data.csv`.

## Uploaded dataset audit

The revised preprocessing path was run across all 90,162 uploaded rows:

- 66,822 rows had repeated synthetic boilerplate removed
- 89,408 repeated sentences were removed
- 87,497 unique cleaned query/passage pairs remained
- recommended VM split: 45,012 training rows, 8,750 validation rows, 8,751 test rows
- query overlap between splits: 0
- passage overlap between splits: 0
- connected-component overlap between splits: 0

The split groups linked queries and passages before splitting, so identical normalized queries or shared normalized passages cannot silently cross into validation/test.

## New training architecture

`train_vm.py` now provides:

- automatic CPU thread control for the 8 exposed logical CPUs
- automatic native CPU BF16 autocast on this Xeon, with `--no-bf16` fallback
- E5 `query: ` and `passage: ` prefixes
- exact-pair deduplication and repeated-boilerplate removal
- leakage-resistant connected-component splitting
- complete-component sampling when the VM profile caps training rows
- partial fine-tuning of only the final four encoder layers
- real batch size 64 with no duplicate samples inside a batch
- Multiple Negatives Ranking Loss
- Matryoshka training at 384, 256 and 128 dimensions
- stage 1 clean-pair training
- stage 2 guarded hard-negative training
- approximate FAISS HNSW hard-negative search instead of full brute-force mining
- automatic validation comparison of stage 1 versus stage 2
- final-model selection based on held-out validation NDCG
- confidence calibration from validation only, never from the test split
- bounded retrieval evaluation and search-latency measurements
- one resumable checkpoint per stage
- disk-efficient hard links for final model promotion
- automatic removal of finished checkpoints and intermediate models
- no ONNX export or duplicate model export during training

## Install

Run these commands inside the extracted repository:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
chmod +x setup_vm.sh run_vm_training.sh
./setup_vm.sh
```

The installer uses the official CPU-only PyTorch index, disables pip caching, and stops if less than 5 GiB is free before installation.

## Run the smoke test first

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh
```

The smoke profile uses 2,000 rows and verifies downloading, preprocessing, training, checkpointing, model saving, and evaluation.

Remove its output before the real run:

```bash
rm -rf training_artifacts/e5-small-swico
```

## Run the recommended VM profile

Foreground:

```bash
SWICO_PROFILE=vm ./run_vm_training.sh
```

Persistent shell run:

```bash
nohup env SWICO_PROFILE=vm ./run_vm_training.sh > training.log 2>&1 &
tail -f training.log
```

The recommended profile uses up to 45,000 component-complete training rows, batch size 64, maximum sequence length 192, the final four encoder layers, one clean-pair epoch, and one guarded hard-negative epoch.

## Resume after interruption

Run the identical command again:

```bash
SWICO_PROFILE=vm ./run_vm_training.sh
```

The pipeline resumes from its newest checkpoint, reuses prepared data and hard negatives, and skips completed stages.

## Full-data profile

After the VM profile succeeds, the complete training split can be run separately:

```bash
SWICO_PROFILE=full ./run_vm_training.sh --output training_artifacts/e5-small-swico-full
```

## Prepare and audit without training

```bash
.venv/bin/python train_vm.py --profile vm --prepare-only
```

## Important outputs

```text
training_artifacts/e5-small-swico/models/final/
training_artifacts/e5-small-swico/reports/final_report.md
training_artifacts/e5-small-swico/reports/final_report.json
training_artifacts/e5-small-swico/training.log
training_artifacts/e5-small-swico/run_state.json
training_artifacts/e5-small-swico/prepared/
```

Use `models/final/` as the Sentence Transformers model directory. The report compares 384-, 256-, and 128-dimensional embeddings. Keep 384 dimensions unless the held-out metrics show that a smaller dimension is acceptable.

## Monitoring commands

```bash
ps aux | grep '[t]rain_vm.py'
tail -f training.log
df -h /
du -sh .venv .hf_cache training_artifacts 2>/dev/null
```

Stop cleanly:

```bash
pkill -INT -f 'train_vm.py'
```

Start over:

```bash
rm -rf training_artifacts/e5-small-swico
SWICO_PROFILE=vm ./run_vm_training.sh
```

## Safe overrides

```bash
./run_vm_training.sh --batch-size 32
./run_vm_training.sh --max-train-rows 30000
./run_vm_training.sh --trainable-layers 3
./run_vm_training.sh --no-hard-negatives
./run_vm_training.sh --no-bf16
./run_vm_training.sh --offline
```

Use `--batch-size 32 --no-bf16` only if the native BF16 smoke test fails on the VM. Do not add ONNX export during training on this disk. Export or quantize only after the final model is complete and free space has been checked.

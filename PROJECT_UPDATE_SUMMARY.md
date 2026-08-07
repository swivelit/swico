# Swico dual-model training update

## Trainers

- `train_vm.py` — advanced multilingual-E5 retrieval/RAG fine-tuning
- `qwen_train_vm.py` — advanced Qwen/Qwen3-0.6B conversational LoRA SFT

## Data

- E5: `data/e5_dataset - e5_dataset.csv`
- Qwen: `data/qwen_dataset - qwen_dataset.csv`

## First VM setup

```bash
chmod +x setup_vm.sh run_vm_training.sh run_e5_training.sh run_qwen_training.sh verify_install.py train_vm.py qwen_train_vm.py
./setup_vm.sh
```

## Validate configurations

```bash
./run_e5_training.sh --print-config
./run_qwen_training.sh --print-config
```

## Smoke tests

```bash
SWICO_PROFILE=smoke ./run_e5_training.sh
SWICO_QWEN_PROFILE=smoke ./run_qwen_training.sh
```

## Full VM-profile runs

```bash
nohup ./run_e5_training.sh > e5-launcher.log 2>&1 &
nohup ./run_qwen_training.sh > qwen-launcher.log 2>&1 &
```

Run them one at a time on the same CPU VM to avoid RAM/CPU contention.

## Main outputs

E5:

```text
training_artifacts/e5-small-swico/latest/models/final/
training_artifacts/e5-small-swico/latest/reports/final_report.md
```

Qwen:

```text
training_artifacts/qwen3-0.6b-swico/latest/models/adapter/
training_artifacts/qwen3-0.6b-swico/latest/reports/final_report.md
```

# Validation record

## Completed before packaging

- `python3 -m py_compile train_vm.py`
- `bash -n setup_vm.sh`
- `bash -n run_vm_training.sh`
- parsed the complete 90,162-row `data.csv`
- removed repeated synthetic boilerplate and exact duplicate pairs
- created component-isolated train, validation and test splits
- confirmed zero exact normalized query, passage and component overlap between splits
- exercised E5 formatting, trainer argument construction, final-layer freezing, Matryoshka dimensions, bounded retrieval metrics, NumPy exact-search fallback, hard-negative mining and hard-negative cache behavior with local test doubles
- verified model-tree promotion using hard links with copy fallback

## Resulting dataset audit

- original rows: 90,162
- cleaned unique pairs: 87,497
- VM training rows: 45,012
- validation rows: 8,750
- test rows: 8,751
- rows changed by repeated-boilerplate removal: 66,822
- repeated sentences removed: 89,408

## Required VM validation

A real end-to-end fine-tuning run was not executed in the packaging sandbox because the Sentence Transformers stack and base-model files were not available through that sandbox's package/model network. Run the included smoke profile on the target VM before the full VM profile:

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh
```

If CPU BF16 is rejected by an operation, rerun the smoke profile with:

```bash
SWICO_PROFILE=smoke ./run_vm_training.sh --no-bf16
```

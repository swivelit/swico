# Qwen CPU LoRA runtime fixes

This build includes the earlier multi-turn assistant-labeling fix and adds the runtime fix for the latest smoke-training failure:

`AttributeError` from `transformers.generation.utils.generate()` while accessing `inputs_tensor.shape[0]`.

## Latest root cause

The Qwen smoke run completed tokenization and LoRA training successfully, then failed during held-out sample generation. Modern Transformers can return a `BatchEncoding` object from `apply_chat_template`. The trainer treated that whole object as `input_ids`, so `model.generate()` received a tokenizer container instead of a PyTorch tensor.

## Latest fix

`qwen_train_vm.py` now:

- requests `return_dict=True` for generation prompt tokenization;
- extracts `input_ids` and `attention_mask` from mapping-like `BatchEncoding` outputs;
- accepts direct tensor/list outputs as a defensive fallback;
- normalizes generation inputs to 2-D PyTorch tensors;
- creates an attention mask when the tokenizer does not return one;
- enables KV cache for inference-only generation;
- removes `EarlyStoppingCallback` after training and before held-out test evaluation, preventing the misleading `eval_loss` warning caused by the `test_` metric prefix.

## Existing assistant-labeling fix retained

The trainer still uses temporary assistant-boundary markers to derive exact assistant-only labels for multi-turn Qwen conversations. Those markers are removed before tokenization and are never trained on.

## Validation

- Full unit-test suite: 20 passed.
- Runtime regression test verifies a BatchEncoding-like object is unpacked to tensors before `model.generate()`.
- Multi-turn assistant-labeling regression test remains green.
- Python compilation check passed.

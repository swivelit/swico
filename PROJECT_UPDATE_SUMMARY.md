# Qwen CPU LoRA runtime and termination fixes

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

## Assistant termination supervision

The trainer first uses the active Qwen tokenizer's native assistant generation mask. It labels assistant content and the tokenizer-defined Qwen EOS/end-of-message token after every assistant span, while system/user text, role headers and disabled-thinking wrapper text remain masked. Older Transformers templates use the marker/offset compatibility path; markers are removed before tokenization and are never trained on.

This prevents the common failure where a good answer is followed by unconstrained generation until `max_new_tokens` because the model was never trained to emit its own turn terminator.

## Validation

- Full unit-test suite: run on the target VM after installing `requirements-cpu.txt`.
- Runtime regression test verifies a BatchEncoding-like object is unpacked to tensors before `model.generate()`.
- Multi-turn assistant content/end-of-message, system/user masking, thinking-wrapper masking and truncation regressions are covered.
- VM generation evaluation now uses 30 deterministic language-stratified samples with termination and max-token health thresholds.
- Python compilation check passed.

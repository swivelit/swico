# Qwen CPU LoRA runtime and pipeline hardening

The Qwen pipeline now evaluates generation with `max_new_tokens=256` by default: 4 deterministic language-stratified smoke samples and 40 VM samples. The former 128-token setting is retained only as historical context; 384 is not used as a default.

Legacy V1 prepared rows such as `en,ta` are now resolved by their meaningful non-system language (`ta`), not by the first metadata item. A model-free audit utility can verify benchmark bucket counts and create canonical frozen-ID manifests.

Split assignment now uses conversation IDs rather than mutable conversation-content digests. Prompt normalization and answer edits therefore do not move a conversation between partitions; frozen evaluation IDs can be forced into the test quota without expanding it unnecessarily.

Preparation derives one canonical language from non-system rows, so the English system-row label cannot contaminate language categories. The ta-en system prompt is normalized to Tamil-script-plus-English semantics, Tanglish remains Latin-script Tamil, and script/style checks are recorded.

Preparation audits assistant responses with a conservative word-level repeated-4gram threshold (`>0.30`), excludes only clearly pathological conversations, and writes `prepared/data_quality.json`. User and assistant answer text is never automatically rewritten.

Qwen runs also write tokenizer pre/post length, truncation, and supervised-token retention statistics by language. This is available in tokenizer-only mode. The real audit decision is now an explicit V2 VM sequence length of 768; 1024 is not selected.

Generation health now requires all four language buckets and checks termination, max-token hits, repetition, and script adherence per bucket. `status=completed` remains a technical training result; candidate promotion is separate and always requires manual quality review.

`qwen_compare_models.py` compares base Qwen, a candidate adapter, and an optional champion adapter on the exact same deterministic prepared test samples without retraining. `export_qwen_adapter.py` merges an existing adapter for deployment, with optional llama.cpp conversion/quantization; `qwen_validate_exports.py` provides local adapter/merged/GGUF parity checks.

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
- Historical VM generation evaluation used 30 deterministic language-stratified samples; current default is 40 with per-language health thresholds.
- Python compilation check passed.

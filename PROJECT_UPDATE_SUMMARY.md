# Qwen assistant-labeling fix

This build fixes the Qwen3 smoke-training failure:

`ERROR: Could not locate assistant response 2 in rendered Qwen chat template`

## Root cause

The previous implementation tried to find each raw assistant response inside the fully rendered Qwen chat string. That is fragile because Qwen3 can add or transform chat-template boundary text around assistant turns.

## Fix

`qwen_train_vm.py` now:

- injects temporary unique boundary markers around assistant message content;
- renders the Qwen chat template only once;
- removes those temporary markers before tokenization;
- derives exact assistant character spans from the marker positions;
- maps those spans through the tokenizer offset mapping;
- masks all non-assistant tokens with `-100`;
- never trains on the temporary markers;
- keeps the existing right-side sequence truncation and final safety check.

The markers are used only internally to calculate labels and are absent from the final model input.

## Validation

- Full unit-test suite: 17 passed.
- Added a runtime regression test covering two assistant replies in one conversation.
- Python compilation check passed.

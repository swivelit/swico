from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "qwen_train_vm.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
WANTED = {"apply_template", "generate_samples"}
NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
MODULE = ast.Module(body=NODES, type_ignores=[])
ast.fix_missing_locations(MODULE)
NS: dict[str, object] = {"torch": torch, "Any": Any}
exec(compile(MODULE, str(ROOT / "qwen_train_vm.py"), "exec"), NS)
generate_samples = NS["generate_samples"]


class FakeBatchEncoding:
    """Mapping-like tokenizer output that intentionally has no .shape attribute."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.data = {"input_ids": input_ids, "attention_mask": attention_mask}

    def keys(self):
        return self.data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str):
        return self.data[key]

    def get(self, key: str, default=None):
        return self.data.get(key, default)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, enable_thinking=False, **kwargs):
        self.last_kwargs = kwargs
        assert tokenize is True
        assert kwargs.get("return_tensors") == "pt"
        assert kwargs.get("return_dict") is True
        return FakeBatchEncoding(
            input_ids=torch.tensor([[10, 11, 12]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        )

    def decode(self, tokens, skip_special_tokens=True):
        return "generated answer"


class FakeModel:
    def eval(self):
        return self

    def generate(self, *, input_ids, attention_mask, **kwargs):
        assert torch.is_tensor(input_ids)
        assert torch.is_tensor(attention_mask)
        assert tuple(input_ids.shape) == (1, 3)
        assert tuple(attention_mask.shape) == (1, 3)
        return torch.cat([input_ids, torch.tensor([[99]], dtype=torch.long)], dim=1)


class QwenGenerationRuntimeTests(unittest.TestCase):
    def test_generation_unpacks_batch_encoding_before_model_generate(self) -> None:
        rows = [
            {
                "conversation_id": "conv-1",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        ]
        samples = generate_samples(FakeModel(), FakeTokenizer(), rows, 1, 16, False)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["generated"], "generated answer")
        self.assertEqual(samples[0]["expected"], "Hi")


if __name__ == "__main__":
    unittest.main()

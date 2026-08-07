from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "qwen_train_vm.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
WANTED = {"apply_template", "assistant_only_tokens"}
NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
MODULE = ast.Module(body=NODES, type_ignores=[])
ast.fix_missing_locations(MODULE)
NS: dict[str, object] = {}
exec(compile(MODULE, str(ROOT / "qwen_train_vm.py"), "exec"), NS)
assistant_only_tokens = NS["assistant_only_tokens"]


class FakeBatch(dict):
    pass


class FakeQwenTokenizer:
    def apply_chat_template(self, messages, tokenize=False, enable_thinking=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts.append(f"<|im_start|>{role}\n")
            if role == "assistant":
                parts.append("<think>\n\n</think>\n\n")
            parts.append(content)
            parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, truncation=False):
        ids = [ord(ch) for ch in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return FakeBatch(input_ids=ids, offset_mapping=offsets)


class QwenAssistantMaskRuntimeTests(unittest.TestCase):
    def test_multiturn_assistant_content_is_labeled_without_raw_text_search(self) -> None:
        messages = [
            {"role": "system", "content": "Be useful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
        ids, labels = assistant_only_tokens(FakeQwenTokenizer(), messages, 4096)
        labeled = "".join(chr(token_id) for token_id, label in zip(ids, labels) if label != -100)
        self.assertEqual(labeled, "First answerSecond answer")
        self.assertNotIn("<think>", labeled)
        self.assertNotIn("assistant", labeled)


if __name__ == "__main__":
    unittest.main()

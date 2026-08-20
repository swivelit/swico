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


class NativeMaskQwenTokenizer:
    eos_token_id = 2
    eos_token = "<|im_end|>"

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, truncation=False):
        if text == self.eos_token:
            return FakeBatch(input_ids=[self.eos_token_id])
        ids = [ord(char) for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return FakeBatch(input_ids=ids, offset_mapping=offsets)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if not tokenize:
            return "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)

        ids = []
        assistant_masks = []
        for message in messages:
            ids.extend([100 if message["role"] == "system" else 101 if message["role"] == "user" else 102])
            assistant_masks.append(0)
            if message["role"] == "assistant":
                # These simulate Qwen's disabled-thinking wrapper. Native
                # generation spans start at content, not at wrapper text.
                ids.extend([300, 301])
                assistant_masks.extend([0, 0])
                content_ids = [ord(char) for char in message["content"]]
                ids.extend(content_ids)
                assistant_masks.extend([1] * len(content_ids))
                ids.append(self.eos_token_id)
                assistant_masks.append(0)
            else:
                content_ids = [400 + ord(char) for char in message["content"]]
                ids.extend(content_ids)
                assistant_masks.extend([0] * len(content_ids))
            ids.append(103)
            assistant_masks.append(0)
        return FakeBatch(input_ids=ids, assistant_masks=assistant_masks)


class QwenAssistantMaskRuntimeTests(unittest.TestCase):
    def test_content_and_termination_are_labeled_but_prompt_and_thinking_are_not(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        ids, labels = assistant_only_tokens(NativeMaskQwenTokenizer(), messages, 4096)
        labeled = [token_id for token_id, label in zip(ids, labels) if label != -100]
        self.assertEqual(labeled, [ord(char) for char in "answer"] + [2])
        self.assertNotIn(300, labeled)
        self.assertNotIn(301, labeled)
        self.assertNotIn(400 + ord("s"), labeled)
        self.assertNotIn(400 + ord("q"), labeled)

    def test_each_assistant_turn_receives_termination_supervision(self) -> None:
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second"},
        ]
        ids, labels = assistant_only_tokens(NativeMaskQwenTokenizer(), messages, 4096)
        self.assertEqual(sum(label == 2 for label in labels), 2)
        self.assertEqual(
            [token_id for token_id, label in zip(ids, labels) if label != -100],
            [*map(ord, "first"), 2, *map(ord, "second"), 2],
        )

    def test_truncation_never_silently_returns_prompt_only_labels(self) -> None:
        messages = [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "later prompt"},
        ]
        with self.assertRaisesRegex(ValueError, "no assistant content or termination"):
            assistant_only_tokens(NativeMaskQwenTokenizer(), messages, 1)


if __name__ == "__main__":
    unittest.main()

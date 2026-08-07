from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "qwen_train_vm.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class QwenSourceContractTests(unittest.TestCase):
    def test_lora_and_assistant_only_training_are_wired(self) -> None:
        self.assertIn("LoraConfig", SOURCE)
        self.assertIn('task_type="CAUSAL_LM"', SOURCE)
        self.assertIn("assistant_only_tokens", SOURCE)
        self.assertIn("SWICO_QWEN_ASSISTANT_BOUNDARY", SOURCE)
        self.assertIn("enable_thinking=False", SOURCE)

    def test_conversation_level_split_and_leakage_check_are_wired(self) -> None:
        self.assertIn("split_conversations", SOURCE)
        self.assertIn('"conversation_id_overlap"', SOURCE)
        self.assertIn("Conversation leakage detected", SOURCE)

    def test_early_stopping_resume_and_memory_guard_are_wired(self) -> None:
        self.assertIn("EarlyStoppingCallback", SOURCE)
        self.assertIn("get_last_checkpoint", SOURCE)
        self.assertIn("class QwenResourceGuardCallback", SOURCE)
        self.assertIn("control.should_training_stop = True", SOURCE)

    def test_adapter_and_reports_are_exported(self) -> None:
        self.assertIn('output / "models" / "adapter"', SOURCE)
        self.assertIn('reports_dir / "final_report.json"', SOURCE)
        self.assertIn('reports_dir / "final_report.md"', SOURCE)

    def test_required_csv_columns_are_explicit(self) -> None:
        self.assertIn('"conversation_id", "turn_index", "role", "content", "language"', SOURCE)


if __name__ == "__main__":
    unittest.main()

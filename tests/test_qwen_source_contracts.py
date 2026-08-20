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
        self.assertIn("return_assistant_tokens_mask=True", SOURCE)
        self.assertIn("termination_ids", SOURCE)
        self.assertIn("no assistant content or termination tokens after tokenization/truncation", SOURCE)

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

    def test_generation_unpacks_batch_encoding_to_tensors(self) -> None:
        self.assertIn('return_dict=True', SOURCE)
        self.assertIn('"input_ids" in rendered', SOURCE)
        self.assertIn('torch.ones_like(input_ids)', SOURCE)
        self.assertIn('use_cache=True', SOURCE)

    def test_step_evaluation_and_stratified_generation_reporting_are_wired(self) -> None:
        self.assertIn('"save_strategy": "steps"', SOURCE)
        self.assertIn('"eval_strategy": "steps"', SOURCE)
        self.assertIn("stratified_generation_rows", SOURCE)
        self.assertIn("summarize_generation_samples", SOURCE)
        self.assertIn('"max_token_hit_rate"', SOURCE)
        self.assertIn('"unhealthy_reasons"', SOURCE)

    def test_early_stopping_callback_is_removed_before_test_evaluation(self) -> None:
        self.assertIn('trainer.remove_callback(EarlyStoppingCallback)', SOURCE)


if __name__ == "__main__":
    unittest.main()

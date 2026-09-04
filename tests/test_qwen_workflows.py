from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class QwenWorkflowContractTests(unittest.TestCase):
    def test_champion_gate_is_fakeable_without_loading_models(self) -> None:
        source = (ROOT / "qwen_compare_models.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted = {"metric_non_regression", "metric_strict_improvement", "candidate_status_from_metrics"}
        module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
        ast.fix_missing_locations(module)
        ns: dict[str, object] = {
            "Any": Any,
            "LANGUAGE_BUCKETS": ("english", "tamil", "tanglish", "tamil-english-mixed"),
        }
        exec(compile(module, str(ROOT / "qwen_compare_models.py"), "exec"), ns)
        def metrics(repetition: float) -> dict[str, Any]:
            return {"overall": {"termination_rate": 1.0, "max_token_hit_rate": 0.0, "mean_repeated_4gram_ratio": repetition, "script_adherence_rate": 1.0}, "by_language": {language: {"sample_count": 1, "termination_rate": 1.0, "max_token_hit_rate": 0.0, "mean_repeated_4gram_ratio": repetition, "script_adherence_rate": 1.0} for language in ns["LANGUAGE_BUCKETS"]}, "health": {"healthy": True}}
        status = ns["candidate_status_from_metrics"](metrics(0.2), metrics(0.1), metrics(0.05))
        self.assertTrue(status["beats_base"])
        self.assertFalse(status["does_not_regress_champion"])
        self.assertTrue(status["manual_quality_review_required"])
        self.assertFalse(status["promotion_eligible"])

    def test_comparison_is_evaluation_only_and_supports_optional_champion(self) -> None:
        source = (ROOT / "qwen_compare_models.py").read_text(encoding="utf-8")
        self.assertIn('"evaluation_only": True', source)
        self.assertIn("champion-adapter", source)
        self.assertIn("released sequentially", source)
        self.assertIn("manual_quality_review_required", source)
        self.assertNotIn("Trainer(", source)

    def test_export_workflow_is_existing_adapter_only_and_safe_by_default(self) -> None:
        source = (ROOT / "export_qwen_adapter.py").read_text(encoding="utf-8")
        self.assertIn("EXPORT ONLY", source)
        self.assertIn("merge_and_unload", source)
        self.assertIn("--overwrite", source)
        self.assertIn("convert_hf_to_gguf", (ROOT / "README.md").read_text(encoding="utf-8"))

    def test_candidate_status_is_not_training_completion(self) -> None:
        source = (ROOT / "qwen_train_vm.py").read_text(encoding="utf-8")
        self.assertIn('"status": "completed"', source)
        self.assertIn('"candidate_status"', source)
        self.assertIn('"promotion_eligible": False', source)

    def test_env_registers_all_new_qwen_keys(self) -> None:
        config = (ROOT / "training_config.py").read_text(encoding="utf-8")
        for key in (
            "SWICO_QWEN_DATA_QUALITY_REPETITION_THRESHOLD",
            "SWICO_QWEN_AUDIT_TOKENIZATION",
            "SWICO_QWEN_HEALTH_MIN_TERMINATION_RATE",
            "SWICO_QWEN_HEALTH_MAX_TOKEN_HIT_RATE",
            "SWICO_QWEN_HEALTH_MAX_REPEATED_4GRAM_RATIO",
            "SWICO_QWEN_HEALTH_REQUIRE_SCRIPT_ADHERENCE",
        ):
            self.assertIn(key, config)


if __name__ == "__main__":
    unittest.main()

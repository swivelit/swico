from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "train_vm.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class SourceContractTests(unittest.TestCase):
    def test_early_stopping_and_best_model_are_wired(self) -> None:
        self.assertIn("EarlyStoppingCallback", SOURCE)
        self.assertIn('"load_best_model_at_end": load_best_model_at_end', SOURCE)
        self.assertIn('kwargs["metric_for_best_model"] = primary_metric', SOURCE)
        self.assertIn('"save_strategy": "epoch"', SOURCE)

    def test_no_legacy_transformers_v5_warmup_warning_path(self) -> None:
        self.assertIn('transformers_major >= 5 and "warmup_steps" in supported', SOURCE)

    def test_public_function_calls_match_keyword_signatures(self) -> None:
        functions = {
            node.name: node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
        }
        checked = {
            "prepare_dataset",
            "build_eval_bundle",
            "mine_hard_negatives",
            "make_loss",
            "make_training_args",
            "train_stage",
            "detailed_retrieval_metrics",
        }
        for target in checked:
            function = functions[target]
            parameters = {argument.arg for argument in function.args.args}
            for call in ast.walk(TREE):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == target
                ):
                    continue
                keywords = {keyword.arg for keyword in call.keywords if keyword.arg}
                positional = {
                    argument.arg
                    for argument in function.args.args[: len(call.args)]
                }
                self.assertFalse(keywords - parameters, (target, call.lineno))
                self.assertFalse(parameters - keywords - positional, (target, call.lineno))

    def test_timestamped_runs_and_latest_links_are_wired(self) -> None:
        self.assertIn("_allocate_unique_timestamped_run", SOURCE)
        self.assertIn('strftime("%Y%m%dT%H%M%SZ")', SOURCE)
        self.assertIn('context.root / "latest-completed"', SOURCE)
        self.assertIn("_latest_compatible_incomplete_run", SOURCE)
        self.assertIn("completed_runs_are_never_reused", SOURCE)

    def test_safe_autotune_does_not_change_quality_controls(self) -> None:
        function = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_adaptive_resource_plan"
        )
        segment = ast.get_source_segment(SOURCE, function) or ""
        self.assertIn("quality_preserving_safe_mode", segment)
        self.assertIn('args.autotune_mode == "aggressive"', segment)
        self.assertNotIn("stage1_epochs =", segment)
        self.assertNotIn("stage2_epochs =", segment)
        self.assertNotIn("stage1_lr =", segment)
        self.assertNotIn("stage2_lr =", segment)
        self.assertNotIn("max_seq_length =", segment)
        self.assertNotIn("trainable_layers =", segment)

    def test_memory_guard_and_baseline_gate_are_wired(self) -> None:
        self.assertIn("class ResourceGuardCallback", SOURCE)
        self.assertIn("control.should_training_stop = True", SOURCE)
        self.assertIn("ResourceStopRequested", SOURCE)
        self.assertIn("promote_validation_winner", SOURCE)
        self.assertIn('selected = "base"', SOURCE)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

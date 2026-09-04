from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "qwen_train_vm.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
WANTED = {
    "stable_hash", "file_sha256", "ids_manifest_sha256", "read_frozen_eval_ids", "parse_conversation_cap", "conversation_cap_source", "conversation_cap_request",
    "normalize_content", "canonical_language", "resolve_prepared_language", "has_tamil_script", "has_latin_letters", "validate_language_style",
    "word_tokens", "repeated_4gram_ratio", "validate_and_group_conversations", "audit_conversation_quality",
    "qwen_language_bucket", "stratified_generation_rows", "summarize_generation_samples",
    "split_conversations", "apply_template", "assistant_only_tokens", "_tokenization_summary", "tokenize_split", "build_tokenization_audit", "training_fingerprint",
}
NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
MODULE = ast.Module(body=NODES, type_ignores=[])
ast.fix_missing_locations(MODULE)


class FakeDataset:
    @classmethod
    def from_dict(cls, payload):
        return payload


NS: dict[str, object] = {
    "Any": Any, "Sequence": list, "argparse": argparse, "hashlib": hashlib, "json": json,
    "math": math, "re": re, "pd": pd, "Dataset": FakeDataset,
    "ALL_CONVERSATIONS": -1,
    "CANONICAL_LANGUAGES": ("en", "ta", "tanglish", "ta-en"),
    "CANONICAL_SYSTEM_PROMPTS": {
        "en": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer in English.",
        "ta": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer in Tamil.",
        "tanglish": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer in tanglish.",
        "ta-en": "You are Swico, a helpful AI assistant. Understand the user's question and provide a simple, clear, accurate, and relevant answer using a natural mix of Tamil script and English that matches the user's language style.",
    },
    "REQUIRED_COLUMNS": ("conversation_id", "turn_index", "role", "content", "language"),
    "VALID_ROLES": {"system", "user", "assistant"},
    "DEFAULT_DATA_QUALITY_REPETITION_THRESHOLD": 0.30,
    "LANGUAGE_BUCKETS": ("english", "tamil", "tanglish", "tamil-english-mixed"),
}
exec(compile(MODULE, str(ROOT / "qwen_train_vm.py"), "exec"), NS)


class QwenFunctions:
    def __getattr__(self, name):
        return NS[name]


qwen = QwenFunctions()


class QwenHardeningTests(unittest.TestCase):
    def test_generation_defaults_and_cap_sentinel(self) -> None:
        self.assertIn("SWICO_QWEN_GENERATION_MAX_NEW_TOKENS", SOURCE)
        self.assertIn("default=env_int(\"SWICO_QWEN_GENERATION_MAX_NEW_TOKENS\", 256)", SOURCE)
        self.assertEqual(qwen.parse_conversation_cap("all"), qwen.ALL_CONVERSATIONS)
        self.assertEqual(qwen.conversation_cap_source(qwen.ALL_CONVERSATIONS), "explicit all")
        self.assertEqual(qwen.conversation_cap_source(100), "explicit integer")
        with self.assertRaises(argparse.ArgumentTypeError):
            qwen.parse_conversation_cap("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            qwen.parse_conversation_cap("banana")

    def test_canonical_language_and_ta_en_prompt_normalization(self) -> None:
        rows = []
        examples = {
            "en": ("hello, how are you?", "I am fine."),
            "ta": ("தமிழில் சொல்லுங்கள்", "நான் உதவுகிறேன்."),
            "tanglish": ("Tamil-la sollunga", "Naan help panren."),
            "ta-en": ("விண்ணப்பம் online-la எப்படி?", "Portal-la apply பண்ணலாம்."),
        }
        for number, (language, (user, assistant)) in enumerate(examples.items()):
            cid = f"c-{number}"
            rows.extend([
                {"conversation_id": cid, "turn_index": "0", "role": "system", "content": "old prompt", "language": "en"},
                {"conversation_id": cid, "turn_index": "1", "role": "user", "content": user, "language": language},
                {"conversation_id": cid, "turn_index": "2", "role": "assistant", "content": assistant, "language": language},
            ])
        conversations = qwen.validate_and_group_conversations(pd.DataFrame(rows))
        by_language = {item["language"]: item for item in conversations}
        self.assertEqual(by_language["ta-en"]["messages"][0]["content"], qwen.CANONICAL_SYSTEM_PROMPTS["ta-en"])
        self.assertEqual(sum(item["system_prompt_normalized_count"] for item in conversations), 4)
        self.assertEqual(qwen.canonical_language("tamil-english-mixed"), "ta-en")
        self.assertEqual(qwen.qwen_language_bucket({"language": "ta-en"}), "tamil-english-mixed")

    def test_legacy_language_metadata_resolves_to_meaningful_non_system_language(self) -> None:
        expected = {
            "en": "english",
            "en,ta": "tamil",
            "en,tanglish": "tanglish",
            "en,ta-en": "tamil-english-mixed",
            "ta": "tamil",
            "tanglish": "tanglish",
            "ta-en": "tamil-english-mixed",
        }
        for metadata, bucket in expected.items():
            self.assertEqual(qwen.qwen_language_bucket({"conversation_id": metadata, "language": metadata}), bucket)
        with self.assertRaises(ValueError):
            qwen.qwen_language_bucket({"conversation_id": "ambiguous", "language": "en,ta,tanglish"})
        with self.assertRaises(ValueError):
            qwen.qwen_language_bucket({"conversation_id": "malformed", "language": "en,"})

    def test_system_en_does_not_make_tamil_conversation_multilingual(self) -> None:
        frame = pd.DataFrame([
            {"conversation_id": "ta", "turn_index": "0", "role": "system", "content": "English", "language": "en"},
            {"conversation_id": "ta", "turn_index": "1", "role": "user", "content": "தமிழ் கேள்வி", "language": "ta"},
            {"conversation_id": "ta", "turn_index": "2", "role": "assistant", "content": "தமிழ் பதில்", "language": "ta"},
            {"conversation_id": "en", "turn_index": "0", "role": "system", "content": "English", "language": "en"},
            {"conversation_id": "en", "turn_index": "1", "role": "user", "content": "English question", "language": "en"},
            {"conversation_id": "en", "turn_index": "2", "role": "assistant", "content": "English answer", "language": "en"},
            {"conversation_id": "x", "turn_index": "0", "role": "system", "content": "English", "language": "en"},
            {"conversation_id": "x", "turn_index": "1", "role": "user", "content": "Tanglish question", "language": "tanglish"},
            {"conversation_id": "x", "turn_index": "2", "role": "assistant", "content": "Tanglish answer", "language": "tanglish"},
        ])
        conversations = qwen.validate_and_group_conversations(frame)
        self.assertEqual({item["language"] for item in conversations}, {"ta", "en", "tanglish"})

    def test_repetition_quality_exclusion_is_reported(self) -> None:
        repeated = "one two three four " * 10
        conversations = [
            {"conversation_id": "bad", "language": "tanglish", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": repeated}]},
            {"conversation_id": "good", "language": "en", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "one two three four five"}]},
        ]
        kept, report = qwen.audit_conversation_quality(conversations)
        self.assertEqual([item["conversation_id"] for item in kept], ["good"])
        self.assertEqual(report["conversations_excluded"], 1)
        self.assertEqual(report["by_language"]["tanglish"]["assistant_messages_excluded"], 1)
        self.assertEqual(report["suspicious_assistant_messages"][0]["reason"], "word-level repeated-4gram ratio above severe threshold")

    def test_tokenization_statistics_capture_truncation_and_retention(self) -> None:
        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                if not tokenize:
                    return ""
                ids = [1]
                masks = [0]
                for message in messages:
                    values = list(range(10, 10 + len(message["content"])))
                    ids.extend(values)
                    masks.extend([1 if message["role"] == "assistant" else 0] * len(values))
                return {"input_ids": ids, "assistant_masks": masks}

        diagnostics = {}
        rows = [{"language": "ta", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]}]
        qwen.tokenize_split(Tokenizer(), rows, 4, diagnostics=diagnostics)
        audit = qwen.build_tokenization_audit(diagnostics, max_seq_length=4, split_name="test")
        ta = audit["by_language"]["ta"]
        self.assertEqual(ta["conversation_count"], 1)
        self.assertEqual(ta["conversations_truncated"], 1)
        self.assertLess(ta["supervised_token_retention_ratio"], 1.0)
        self.assertIn("p95", ta["pre_truncation_token_count"])

    def test_language_health_requires_all_buckets_and_checks_repetition(self) -> None:
        samples = [
            {"language": language, "eos_or_end_of_message_produced": True, "max_token_hit": False, "max_new_tokens_reached": False, "output_token_count": 4, "generation_time_seconds": 1.0, "tokens_per_second": 4.0, "repeated_4gram_ratio": 0.0, "script_style_adherent": True}
            for language in ("english", "tamil", "tanglish", "tamil-english-mixed")
        ]
        healthy = qwen.summarize_generation_samples(samples)
        self.assertTrue(healthy["health"]["healthy"])
        unhealthy = qwen.summarize_generation_samples(samples[:3])
        self.assertIn("missing language bucket: tamil-english-mixed", unhealthy["health"]["unhealthy_reasons"])
        samples[-1]["repeated_4gram_ratio"] = 0.4
        repeated = qwen.summarize_generation_samples(samples)
        self.assertIn("tamil-english-mixed repeated-4gram ratio above threshold", repeated["health"]["unhealthy_reasons"])

    def test_stratified_selection_and_training_fingerprint_ignore_eval_only_settings(self) -> None:
        rows = [{"conversation_id": f"{language}-{index}", "language": language} for language in ("en", "ta", "tanglish", "ta-en") for index in range(5)]
        first = qwen.stratified_generation_rows(rows, 4, 42)
        second = qwen.stratified_generation_rows(rows, 4, 42)
        self.assertEqual([r["conversation_id"] for r in first], [r["conversation_id"] for r in second])
        self.assertEqual({qwen.qwen_language_bucket(r) for r in first}, set(qwen.LANGUAGE_BUCKETS))
        config = {"profile": {"eval_generation_samples": 4, "max_seq_length": 512}, "generation_sample_count": 4, "generation_max_new_tokens": 128, "learning_rate": 5e-5}
        changed = {"profile": {"eval_generation_samples": 40, "max_seq_length": 512}, "generation_sample_count": 40, "generation_max_new_tokens": 256, "learning_rate": 5e-5}
        self.assertEqual(qwen.training_fingerprint(config), qwen.training_fingerprint(changed))

    def test_split_membership_uses_ids_not_mutable_content_and_stays_disjoint(self) -> None:
        rows = [{"conversation_id": f"id-{index}", "digest": f"digest-{index}", "language": "en"} for index in range(30)]
        original = qwen.split_conversations(rows, 42, (0.8, 0.1, 0.1), None)
        changed = [dict(row, digest=f"changed-{row['conversation_id']}") for row in rows]
        rewritten = qwen.split_conversations(changed, 42, (0.8, 0.1, 0.1), None)
        assignment = {row["conversation_id"]: split for split, values in original.items() for row in values}
        rewritten_assignment = {row["conversation_id"]: split for split, values in rewritten.items() for row in values}
        self.assertEqual(assignment, rewritten_assignment)
        self.assertEqual(set(assignment), {row["conversation_id"] for row in rows})
        train_ids = {row["conversation_id"] for row in original["train"]}
        validation_ids = {row["conversation_id"] for row in original["validation"]}
        test_ids = {row["conversation_id"] for row in original["test"]}
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        remaining = [row for row in rows if row["conversation_id"] != "id-29"]
        reduced = qwen.split_conversations(remaining, 42, (0.8, 0.1, 0.1), None)
        reduced_assignment = {row["conversation_id"]: split for split, values in reduced.items() for row in values}
        self.assertEqual({key: value for key, value in assignment.items() if key != "id-29"}, reduced_assignment)
        other_seed = qwen.split_conversations(rows, 99, (0.8, 0.1, 0.1), None)
        other_assignment = {row["conversation_id"]: split for split, values in other_seed.items() for row in values}
        self.assertNotEqual(assignment, other_assignment)

    def test_frozen_ids_are_forced_into_test_quota(self) -> None:
        rows = [{"conversation_id": f"id-{index}", "digest": f"digest-{index}", "language": "en"} for index in range(30)]
        splits = qwen.split_conversations(rows, 42, (0.8, 0.1, 0.1), None, frozen_eval_ids={"id-1", "id-2"})
        self.assertIn("id-1", {row["conversation_id"] for row in splits["test"]})
        self.assertIn("id-2", {row["conversation_id"] for row in splits["test"]})
        self.assertNotIn("id-1", {row["conversation_id"] for row in splits["train"] + splits["validation"]})

    def test_frozen_id_manifest_rejects_duplicates_and_is_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen-ids.txt"
            path.write_text("id-2\nid-1\nid-2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                qwen.read_frozen_eval_ids(path)
            path.write_text("id-2\nid-1\n", encoding="utf-8")
            ids, meta = qwen.read_frozen_eval_ids(path)
            self.assertEqual(ids, {"id-1", "id-2"})
            self.assertEqual(meta["sha256"], qwen.file_sha256(path))
            self.assertEqual(len(qwen.ids_manifest_sha256(ids)), 64)


if __name__ == "__main__":
    unittest.main()

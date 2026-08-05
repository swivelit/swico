from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_config import (
    ConfigError,
    env_bool,
    env_float_tuple,
    env_optional_int,
    initialize_environment,
)


class TrainingConfigTests(unittest.TestCase):
    def test_env_file_loads_without_overriding_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.env"
            path.write_text(
                "SWICO_CONFIG_STRICT=true\nSWICO_PROFILE=vm\nSWICO_CPU_THREADS=4\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"SWICO_CPU_THREADS": "6"},
                clear=True,
            ):
                loaded = initialize_environment(["--env-file", str(path)])
                self.assertEqual(loaded, path.resolve())
                self.assertEqual(os.environ["SWICO_PROFILE"], "vm")
                self.assertEqual(os.environ["SWICO_CPU_THREADS"], "6")

    def test_unknown_swico_key_is_rejected_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.env"
            path.write_text(
                "SWICO_CONFIG_STRICT=true\nSWICO_BATCH_SZE=32\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ConfigError, "SWICO_BATCH_SZE"):
                    initialize_environment(["--env-file", str(path)])

    def test_auto_optional_value_and_typed_lists(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SWICO_MAX_TRAIN_ROWS": "auto",
                "SWICO_MATRYOSHKA_WEIGHTS": "1.0, 0.35, 0.2",
                "SWICO_RESUME": "yes",
            },
            clear=True,
        ):
            self.assertIsNone(env_optional_int("SWICO_MAX_TRAIN_ROWS"))
            self.assertEqual(
                env_float_tuple("SWICO_MATRYOSHKA_WEIGHTS", (1.0,)),
                (1.0, 0.35, 0.2),
            )
            self.assertTrue(env_bool("SWICO_RESUME", False))

    def test_missing_explicit_env_file_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigError, "does not exist"):
                initialize_environment(["--env-file", "/definitely/missing/training.env"])


if __name__ == "__main__":
    unittest.main()

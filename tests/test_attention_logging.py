"""Backend reporting must be visible, deduplicated and independent of RNG state."""

import importlib.util
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers.utils import import_utils

from src.utils.attention_logging import (
    log_attention_backend,
    warn_flash_attention_fallback,
)

ROOT = Path(__file__).parent.parent
ATTENTION_PATHS = ("src/utils/attention.py", "src/wan/modules/attention.py")
LOGGER = "src.utils.attention_logging"


class AttentionLoggingTests(unittest.TestCase):
    def setUp(self):
        log_attention_backend.cache_clear()
        warn_flash_attention_fallback.cache_clear()

    def tearDown(self):
        log_attention_backend.cache_clear()
        warn_flash_attention_fallback.cache_clear()

    def import_attention_modules(self, *, available, error=None):
        """Exercise both real module initializers with controlled FA4 availability."""
        import builtins

        original_import = builtins.__import__

        def import_with_fa4(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "flash_attn.cute":
                if error is not None:
                    raise type(error)(str(error))
                return SimpleNamespace(cute=SimpleNamespace())
            return original_import(name, globals, locals, fromlist, level)

        modules = []
        with (
            patch.object(import_utils, "is_flash_attn_4_available", return_value=available),
            patch("builtins.__import__", side_effect=import_with_fa4),
        ):
            for index, relative in enumerate(ATTENTION_PATHS):
                spec = importlib.util.spec_from_file_location(f"attention_logging_test_{index}", ROOT / relative)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules.append(module)
        return modules

    def test_fa4_logs_once_across_both_modules(self):
        with self.assertLogs(LOGGER, level="INFO") as captured:
            modules = self.import_attention_modules(available=True)
        self.assertTrue(all(module.FLASH_ATTN_4_AVAILABLE for module in modules))
        self.assertEqual([record.getMessage() for record in captured.records], ["[attention] backend=FA4"])
        self.assertEqual(captured.records[0].levelname, "INFO")

    def test_unavailable_fa4_reports_sdpa_without_warning(self):
        with self.assertLogs(LOGGER, level="INFO") as captured:
            modules = self.import_attention_modules(available=False)
        self.assertFalse(any(module.FLASH_ATTN_4_AVAILABLE for module in modules))
        self.assertEqual([record.getMessage() for record in captured.records], ["[attention] backend=PyTorch SDPA"])
        self.assertEqual(captured.records[0].levelname, "INFO")

    def test_import_failures_warn_once_and_preserve_rng(self):
        for missing in ("fcntl", "triton"):
            with self.subTest(missing=missing):
                log_attention_backend.cache_clear()
                warn_flash_attention_fallback.cache_clear()
                cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                python_state = random.getstate()
                numpy_state = np.random.get_state()
                cpu_state = torch.get_rng_state().clone()
                with self.assertLogs(LOGGER, level="INFO") as captured:
                    modules = self.import_attention_modules(
                        available=True, error=ModuleNotFoundError(f"No module named '{missing}'")
                    )
                self.assertFalse(any(module.FLASH_ATTN_4_AVAILABLE for module in modules))
                warnings = [record for record in captured.records if record.levelname == "WARNING"]
                self.assertEqual(len(warnings), 1)
                self.assertIn(f"ModuleNotFoundError: No module named '{missing}'", warnings[0].getMessage())
                self.assertIn("falling back to PyTorch SDPA", warnings[0].getMessage())
                self.assertIn("numerical results may differ", warnings[0].getMessage())
                self.assertEqual(captured.records[-1].getMessage(), "[attention] backend=PyTorch SDPA")
                self.assertEqual(len(captured.records), 2)
                self.assertEqual(random.getstate(), python_state)
                self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
                current_numpy_state = np.random.get_state()
                self.assertEqual(current_numpy_state[0], numpy_state[0])
                np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
                self.assertEqual(current_numpy_state[2:], numpy_state[2:])
                if cuda_states:
                    self.assertTrue(all(
                        torch.equal(before, after)
                        for before, after in zip(cuda_states, torch.cuda.get_rng_state_all(), strict=True)
                    ))

    def test_runtime_errors_still_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "kernel error"):
            self.import_attention_modules(available=True, error=RuntimeError("kernel error"))


if __name__ == "__main__":
    unittest.main()

"""Report attention backend selection without changing backend availability."""

import logging
from functools import cache, lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def log_attention_backend(flash_attn_available: bool) -> None:
    """Log each selected backend once per process across both attention modules."""
    backend = "FA4" if flash_attn_available else "PyTorch SDPA"
    logger.info("[attention] backend=%s", backend)


@cache
def warn_flash_attention_fallback(error_type: str, error_message: str) -> None:
    """Report each distinct import failure once, without retaining its traceback."""
    logger.warning(
        "[attention] FA4 import failed (%s: %s); falling back to PyTorch SDPA. "
        "Performance and numerical results may differ.",
        error_type,
        error_message,
    )

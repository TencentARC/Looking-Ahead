"""Lightweight checkpoint state-dict key normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_generator_fsdp_key(key: str) -> str:
    """Map supported inner/outer FSDP generator namespaces to canonical keys."""

    while key.startswith("_fsdp_wrapped_module."):
        key = key[len("_fsdp_wrapped_module.") :]
    while "._fsdp_wrapped_module." in key:
        key = key.replace("._fsdp_wrapped_module.", ".")
    return key


def normalize_generator_fsdp_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a generator state dict using the unwrapped canonical namespace."""

    normalized: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for key, value in state_dict.items():
        canonical = normalize_generator_fsdp_key(key)
        if canonical in normalized:
            raise ValueError(
                "FSDP checkpoint key normalization collision for "
                f"{origins[canonical]!r} and {key!r} -> {canonical!r}."
            )
        normalized[canonical] = value
        origins[canonical] = key
    return normalized

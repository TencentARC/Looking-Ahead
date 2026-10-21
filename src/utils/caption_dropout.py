"""Deterministic, RNG-isolated caption dropout for diffusion training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import torch

_CAPTION_DROPOUT_STREAM_ID = 0x4341_5054_494F_4E01


def validate_caption_dropout_probability(value: object) -> float:
    """Return a validated caption-dropout probability in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("caption_dropout_p must be a non-boolean real number.")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("caption_dropout_p must be finite and within [0, 1].")
    return probability


def derive_caption_dropout_seed(
    base_seed: int,
    global_rank: int,
    global_step: int,
) -> int:
    """Derive a stable, position-addressed seed without consuming global RNG.

    The address deliberately includes rank and global training step. Batch-slot
    identity is supplied by the stable position within the mask vector sampled
    from this seed.
    """
    components = (base_seed, global_rank, global_step, _CAPTION_DROPOUT_STREAM_ID)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in components):
        raise TypeError("Caption-dropout seed components must be non-boolean integers.")
    if global_rank < 0 or global_step < 0:
        raise ValueError("Caption-dropout seed address requires global_rank and global_step >= 0.")

    mask = (1 << 64) - 1
    value = 0x510E_527F_ADE6_82D1
    for component in components:
        value ^= (component & mask) + 0x9E37_79B9_7F4A_7C15 + ((value << 6) & mask)
        value ^= value >> 30
        value = (value * 0xBF58_476D_1CE4_E5B9) & mask
        value ^= value >> 27
        value = (value * 0x94D0_49BB_1331_11EB) & mask
        value ^= value >> 31
    return value & ((1 << 63) - 1)


def sample_caption_dropout_mask(
    probability: object,
    batch_size: int,
    *,
    base_seed: int,
    global_rank: int,
    global_step: int,
) -> torch.Tensor:
    """Sample a per-sample CPU mask from an isolated generator.

    ``p=0`` and ``p=1`` use exact deterministic fast paths. For intermediate
    probabilities, a fresh CPU generator is addressed by seed/rank/step, so
    this function never reads or advances the process-global CPU or CUDA RNG.
    """
    probability = validate_caption_dropout_probability(probability)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("caption-dropout batch_size must be a non-boolean integer.")
    if batch_size <= 0:
        raise ValueError("caption-dropout batch_size must be positive.")

    seed = derive_caption_dropout_seed(
        base_seed=base_seed,
        global_rank=global_rank,
        global_step=global_step,
    )
    if probability == 0.0:
        return torch.zeros(batch_size, dtype=torch.bool)
    if probability == 1.0:
        return torch.ones(batch_size, dtype=torch.bool)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(batch_size, generator=generator) < probability


def replace_prompt_embeds_with_negative(
    conditional_dict: Mapping[str, object],
    unconditional_dict: Mapping[str, object],
    dropout_mask: torch.Tensor,
) -> dict[str, object]:
    """Return conditioning with selected prompt-embedding rows replaced.

    Only ``prompt_embeds`` is changed. Other conditioning fields and both input
    mappings remain untouched.
    """
    if not torch.is_tensor(dropout_mask):
        raise TypeError("caption-dropout mask must be a torch.Tensor.")
    if dropout_mask.dtype != torch.bool or dropout_mask.ndim != 1:
        raise ValueError("caption-dropout mask must be a one-dimensional bool tensor.")

    conditional = conditional_dict.get("prompt_embeds")
    negative = unconditional_dict.get("prompt_embeds")
    if not torch.is_tensor(conditional) or not torch.is_tensor(negative):
        raise KeyError("Both conditioning dictionaries must contain tensor prompt_embeds.")
    if conditional.shape != negative.shape:
        raise ValueError(
            "Conditional and negative prompt_embeds must have identical shapes, "
            f"got {tuple(conditional.shape)} and {tuple(negative.shape)}."
        )
    if conditional.ndim < 1 or conditional.shape[0] != dropout_mask.numel():
        raise ValueError(
            "caption-dropout mask length must equal the prompt_embeds batch dimension."
        )
    if conditional.device != negative.device or conditional.dtype != negative.dtype:
        raise ValueError("Conditional and negative prompt_embeds must share device and dtype.")

    broadcast_shape = (dropout_mask.numel(),) + (1,) * (conditional.ndim - 1)
    selected = torch.where(
        dropout_mask.to(device=conditional.device).reshape(broadcast_shape),
        negative,
        conditional,
    )
    result = dict(conditional_dict)
    result["prompt_embeds"] = selected
    return result

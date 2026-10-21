"""Training-only compatibility for retaining a subset of predictive offsets."""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

_BRANCH_OFFSET = re.compile(
    r"^(?:model\.)?predictive_multi_chunk_branches\."
    r"(?:(?:clean|noisy)\.)?offset_([1-9][0-9]*)\."
)


def retain_predictive_checkpoint_offsets(
    state_dict: Mapping[str, Any],
    target_offsets: Sequence[int],
    *,
    logger: logging.Logger,
    label: str,
) -> tuple[Mapping[str, Any], bool]:
    """Filter canonical branch keys only for a nonempty proper offset subset.

    Callers normalize FSDP keys first and retain all existing family checks.
    The returned flag requires strict validation of the retained state; it must
    not be used with the ordinary missing-aux initialization fallback.
    Shared owners, metadata and unrecognized keys are preserved for validation.
    No tensor is copied and the input mapping is never mutated.
    """
    key_offsets = {
        key: int(match.group(1))
        for key in state_dict
        if (match := _BRANCH_OFFSET.match(key)) is not None
    }
    checkpoint_offsets = set(key_offsets.values())
    wanted = set(target_offsets)
    if not wanted or not wanted < checkpoint_offsets:
        return state_dict, False

    dropped = checkpoint_offsets - wanted
    retained = {
        key: value for key, value in state_dict.items() if key_offsets.get(key) not in dropped
    }
    logger.info(
        "%s offset subset: checkpoint_offsets=%s training_offsets=%s "
        "dropped_offsets=%s dropped_tensors=%d",
        label,
        sorted(checkpoint_offsets),
        sorted(wanted),
        sorted(dropped),
        len(state_dict) - len(retained),
    )
    return retained, True

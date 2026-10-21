"""Ordinary TF/FCP initialization compatibility; full resume is separate."""

import logging
import re

from src.utils.checkpoint_state import normalize_generator_fsdp_state_dict
from src.utils.predictive_checkpoint import retain_predictive_checkpoint_offsets
from src.utils.predictive_multi_chunk_config import (
    is_predictive_source_layers_metadata_key,
    validate_aux_shared_across_offsets_checkpoint,
    validate_mlp_source_layer_checkpoint,
    validate_offset_source_mode_checkpoint,
    validate_source_unshared_scope_checkpoint,
)


def is_predictive_aux_key(key):
    return (
        "predictive_multi_chunk_branches" in key.split(".")
        or "predictive_multi_chunk_shared_direct_modules" in key.split(".")
        or is_predictive_source_layers_metadata_key(key)
    )


def prepare_predictive_checkpoint(state, config, expected_shapes, *, label, ema=False):
    state = normalize_generator_fsdp_state_dict(state)
    if any(key.endswith("predictive_history_context_marker") for key in state):
        raise RuntimeError(f"{label}: history-context checkpoint is incompatible with FCP")
    predictive = validate_offset_source_mode_checkpoint(state, label=label)
    validate_source_unshared_scope_checkpoint(state, label=label)
    validate_aux_shared_across_offsets_checkpoint(state, label=label)
    validate_mlp_source_layer_checkpoint(
        state,
        source_layers=config["source_layers"],
        label=label,
    )
    state, subset = retain_predictive_checkpoint_offsets(
        state, config["future_offsets"], logger=logging.getLogger(__name__), label=label
    )
    if predictive:
        present_offsets = {
            int(match.group(1))
            for key in state
            if (match := re.search(r"predictive_multi_chunk_branches\.offset_(\d+)\.", key))
        }
        if not set(config["future_offsets"]) <= present_offsets:
            raise RuntimeError(f"{label}: requested offsets absent from checkpoint")
    if ema:
        state = {k: v for k, v in state.items() if not is_predictive_source_layers_metadata_key(k)}
        expected_shapes = {
            k: v
            for k, v in expected_shapes.items()
            if not is_predictive_source_layers_metadata_key(k)
        }
    missing = set(expected_shapes) - state.keys()
    unexpected = state.keys() - set(expected_shapes)
    bad_missing = [
        k
        for k in missing
        if subset
        or not is_predictive_aux_key(k)
        or (predictive and (".hidden_adapter." in k or is_predictive_source_layers_metadata_key(k)))
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"{label}: missing required keys={sorted(bad_missing)}; "
            f"unexpected keys={sorted(unexpected)}"
        )
    for key, value in state.items():
        if not hasattr(value, "shape") or tuple(value.shape) != expected_shapes[key]:
            raise RuntimeError(
                f"{label}: incompatible shape for {key}; expected {expected_shapes[key]}"
            )
    if missing:
        logging.getLogger(__name__).warning(
            "%s keeps %d initialized auxiliary tensors (TF warm start or compatible depth increase)",
            label,
            len(missing),
        )
    return state, predictive

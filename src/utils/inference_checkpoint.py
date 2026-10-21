"""Strict ordinary backbone loading with live/default-single-EMA selection."""

import logging
from collections.abc import Mapping
from pathlib import Path

from src.utils.checkpoint_state import normalize_generator_fsdp_state_dict
from src.utils.ckpt_cache import load_ckpt
from src.utils.wan_model_family import validate_wan_checkpoint_family

logger = logging.getLogger(__name__)
_AUX_NAMES = {
    "predictive_multi_chunk_branches",
    "predictive_multi_chunk_shared_direct_modules",
    "predictive_multi_chunk_source_layers_metadata",
    "predictive_history_context_marker",
    "predictive_offset_source_mode_marker",
    "predictive_source_unshared_scope_marker",
    "predictive_aux_shared_across_offsets_marker",
}


def filter_inference_backbone(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not any(component in _AUX_NAMES for component in key.split("."))
    }


def load_generator_checkpoint(generator, config):
    path = config.checkpoint_path
    if not path:
        if config.use_ema:
            raise FileNotFoundError("EMA requested but no training checkpoint was selected.")
        logger.info("No training checkpoint was loaded; using the pipeline's initial weights.")
        return None
    if config.use_ema:
        sibling = Path(path).with_name("model_ema.pt")
        if sibling.is_file():
            selected = str(sibling)
            payload = load_ckpt(selected, logger=logger, label=sibling.name)
            state_dict = payload.get("generator_ema", payload)
        else:
            selected = str(path)
            payload = load_ckpt(selected, logger=logger, label=Path(path).name)
            if "generator_ema" not in payload:
                raise FileNotFoundError(f"EMA unavailable: no {sibling} or embedded generator_ema.")
            state_dict = payload["generator_ema"]
    else:
        selected = str(path)
        payload = load_ckpt(selected, logger=logger, label=Path(path).name)
        state_dict = payload["generator"]
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{selected}: generator state must be a mapping.")
    state_dict = normalize_generator_fsdp_state_dict(state_dict)
    validate_wan_checkpoint_family(
        state_dict, model_name=config.model_name, consumer_name="inference generator checkpoint"
    )
    filtered = filter_inference_backbone(state_dict)
    logger.info(
        "Loading %s; filtered %d training-only tensors", selected, len(state_dict) - len(filtered)
    )
    # PyTorch 2.10 supports assign; all compatibility errors remain fatal.
    generator.load_state_dict(filtered, strict=True, assign=True)
    logger.info("Strict backbone load complete: %s", selected)
    return selected

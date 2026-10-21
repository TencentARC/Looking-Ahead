"""FCP geometry and checkpoint identity checks.

Excluded checkpoint families are recognized for rejection, without retaining their
model construction or execution branches. Public inputs use future_prediction.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from omegaconf import OmegaConf

PREDICTIVE_SOURCE_LAYERS_METADATA_BUFFER = "predictive_multi_chunk_source_layers_metadata"
PREDICTIVE_OFFSET_SOURCE_MODE_MARKER = "predictive_offset_source_mode_marker"
PREDICTIVE_CHAINED_SOURCE_MODE_MARKER_VALUE = 1
PREDICTIVE_SOURCE_UNSHARED_SCOPE_MARKER = "predictive_source_unshared_scope_marker"
PREDICTIVE_SOURCE_CONDITIONING_SCOPE_MARKER_VALUE = 1
PREDICTIVE_AUX_SHARED_ACROSS_OFFSETS_MARKER = "predictive_aux_shared_across_offsets_marker"
PREDICTIVE_DIRECT_MODULE_NAMES = (
    "patch_embedding",
    "time_embedding",
    "time_projection",
    "text_embedding",
    "head",
)
PREDICTIVE_SHARED_DIRECT_MODULES_NAMESPACE = "predictive_multi_chunk_shared_direct_modules"


def is_predictive_source_layers_metadata_key(key: str) -> bool:
    """Return whether ``key`` stores the ordered MLP source-layer signature."""
    return key == PREDICTIVE_SOURCE_LAYERS_METADATA_BUFFER or key.endswith(
        f".{PREDICTIVE_SOURCE_LAYERS_METADATA_BUFFER}"
    )


def is_predictive_offset_source_mode_marker_key(key: str) -> bool:
    """Return whether ``key`` stores chained offset-source identity."""
    return key == PREDICTIVE_OFFSET_SOURCE_MODE_MARKER or key.endswith(
        f".{PREDICTIVE_OFFSET_SOURCE_MODE_MARKER}"
    )


def is_predictive_source_unshared_scope_marker_key(key: str) -> bool:
    """Return whether ``key`` stores source-conditioning scope identity."""
    return key == PREDICTIVE_SOURCE_UNSHARED_SCOPE_MARKER or key.endswith(
        f".{PREDICTIVE_SOURCE_UNSHARED_SCOPE_MARKER}"
    )


def is_predictive_aux_shared_across_offsets_marker_key(key: str) -> bool:
    """Return whether ``key`` stores cross-offset direct-module identity."""
    return key == PREDICTIVE_AUX_SHARED_ACROSS_OFFSETS_MARKER or key.endswith(
        f".{PREDICTIVE_AUX_SHARED_ACROSS_OFFSETS_MARKER}"
    )


def checkpoint_has_predictive_aux_state(state_dict: Mapping) -> bool:
    """Return whether a checkpoint contains any predictive aux namespace."""
    return any(
        ".predictive_multi_chunk_branches." in key
        or key.startswith("predictive_multi_chunk_branches.")
        or key.startswith("model.predictive_multi_chunk_branches.")
        or f".{PREDICTIVE_SHARED_DIRECT_MODULES_NAMESPACE}." in key
        or key.startswith(f"{PREDICTIVE_SHARED_DIRECT_MODULES_NAMESPACE}.")
        or key.startswith(f"model.{PREDICTIVE_SHARED_DIRECT_MODULES_NAMESPACE}.")
        or is_predictive_source_layers_metadata_key(key)
        or is_predictive_offset_source_mode_marker_key(key)
        or is_predictive_source_unshared_scope_marker_key(key)
        or is_predictive_aux_shared_across_offsets_marker_key(key)
        for key in state_dict
    )


def validate_aux_shared_across_offsets_checkpoint(
    state_dict: Mapping,
    *,
    label: str,
) -> bool:
    """Require exact cross-offset direct-module checkpoint-family identity."""
    source_is_predictive = checkpoint_has_predictive_aux_state(state_dict)
    if not source_is_predictive:
        return False

    marker_keys = [
        key for key in state_dict if is_predictive_aux_shared_across_offsets_marker_key(key)
    ]
    if len(marker_keys) > 1:
        raise RuntimeError(
            f"{label} contains duplicate predictive aux_shared_across_offsets markers: "
            f"{marker_keys}."
        )
    checkpoint_value = 0
    if marker_keys:
        marker = state_dict[marker_keys[0]]
        if (
            not hasattr(marker, "detach")
            or marker.numel() != 1
            or marker.ndim != 0
            or marker.is_floating_point()
            or marker.is_complex()
            or str(marker.dtype) == "torch.bool"
        ):
            raise RuntimeError(f"{label} predictive aux_shared_across_offsets marker is malformed.")
        checkpoint_value = marker.detach().cpu().item()
        max_value = (1 << len(PREDICTIVE_DIRECT_MODULE_NAMES)) - 1
        if (
            isinstance(checkpoint_value, bool)
            or checkpoint_value <= 0
            or checkpoint_value > max_value
        ):
            raise RuntimeError(
                f"{label} predictive aux_shared_across_offsets marker has unknown "
                f"value {checkpoint_value!r}."
            )

    if checkpoint_value != 0:
        raise RuntimeError(
            f"{label} predictive aux_shared_across_offsets mismatch: "
            f"checkpoint_marker={checkpoint_value}, runtime_marker=0."
        )
    return True


def validate_offset_source_mode_checkpoint(
    state_dict: Mapping,
    *,
    label: str,
) -> bool:
    """Require exact direct/chained identity for predictive warm starts."""
    source_is_predictive = checkpoint_has_predictive_aux_state(state_dict)
    if not source_is_predictive:
        return False

    marker_keys = [key for key in state_dict if is_predictive_offset_source_mode_marker_key(key)]
    if len(marker_keys) > 1:
        raise RuntimeError(
            f"{label} contains duplicate predictive offset-source mode markers: {marker_keys}."
        )
    checkpoint_mode = "direct"
    if marker_keys:
        marker = state_dict[marker_keys[0]]
        if (
            not hasattr(marker, "detach")
            or marker.numel() != 1
            or marker.ndim != 0
            or marker.is_floating_point()
            or marker.is_complex()
        ):
            raise RuntimeError(f"{label} predictive offset-source mode marker is malformed.")
        value = marker.detach().cpu().item()
        if isinstance(value, bool) or value != PREDICTIVE_CHAINED_SOURCE_MODE_MARKER_VALUE:
            raise RuntimeError(
                f"{label} predictive offset-source mode marker has unknown value {value!r}."
            )
        checkpoint_mode = "chained"
    if checkpoint_mode != "direct":
        raise RuntimeError(
            f"{label} predictive offset-source mode mismatch: "
            f"checkpoint={checkpoint_mode}, runtime=direct."
        )
    return True


def validate_source_unshared_scope_checkpoint(
    state_dict: Mapping,
    *,
    label: str,
) -> bool:
    """Require exact legacy/source-conditioning checkpoint-family identity."""
    source_is_predictive = checkpoint_has_predictive_aux_state(state_dict)
    if not source_is_predictive:
        return False
    marker_keys = [key for key in state_dict if is_predictive_source_unshared_scope_marker_key(key)]
    if len(marker_keys) > 1:
        raise RuntimeError(
            f"{label} contains duplicate predictive source-unshared scope markers: {marker_keys}."
        )
    checkpoint_scope = "source_conditioning" if marker_keys else None
    if marker_keys:
        marker = state_dict[marker_keys[0]]
        if (
            not hasattr(marker, "detach")
            or marker.numel() != 1
            or marker.ndim != 0
            or marker.is_floating_point()
            or marker.is_complex()
            or str(marker.dtype) == "torch.bool"
        ):
            raise RuntimeError(f"{label} predictive source-unshared scope marker is malformed.")
        value = marker.detach().cpu().item()
        if isinstance(value, bool) or value != PREDICTIVE_SOURCE_CONDITIONING_SCOPE_MARKER_VALUE:
            raise RuntimeError(
                f"{label} predictive source-unshared scope marker has unknown value {value!r}."
            )
    if checkpoint_scope is not None:
        raise RuntimeError(
            f"{label} predictive source-unshared scope mismatch: "
            f"checkpoint={checkpoint_scope or 'legacy'}, "
            f"runtime=legacy."
        )
    return True


def validate_mlp_source_layer_checkpoint(
    state_dict: Mapping,
    *,
    source_layers: Sequence[int],
    label: str,
) -> bool:
    """Validate ordered source-layer metadata for an MLP target.

    Returns true when the source checkpoint is predictive. A checkpoint with
    no predictive aux state is a valid non-predictive warm start and returns
    false without requiring metadata.
    """
    source_is_predictive = checkpoint_has_predictive_aux_state(state_dict)
    if not source_is_predictive:
        return source_is_predictive

    metadata_keys = [key for key in state_dict if is_predictive_source_layers_metadata_key(key)]
    if len(metadata_keys) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one ordered source-layer metadata tensor "
            f"for an MLP target; found {metadata_keys}."
        )
    metadata = state_dict[metadata_keys[0]]
    if not hasattr(metadata, "detach"):
        raise RuntimeError(f"{label} source-layer metadata must be a tensor.")
    if metadata.ndim != 1:
        raise RuntimeError(f"{label} source-layer metadata must be a one-dimensional tensor.")
    metadata_values = metadata.detach().cpu().tolist()
    if any(isinstance(value, bool) or not isinstance(value, int) for value in metadata_values):
        raise RuntimeError(f"{label} source-layer metadata must contain integers.")
    checkpoint_layers = list(metadata_values)
    expected_layers = [int(value) for value in source_layers]
    if checkpoint_layers != expected_layers:
        raise RuntimeError(
            f"{label} source_layers mismatch: checkpoint={checkpoint_layers}, "
            f"runtime={expected_layers}."
        )
    return True


@dataclass(frozen=True)
class PredictiveChunkLayout:
    num_frames: int
    block_size: int
    independent_first_frame: bool
    regular_frame_start: int
    num_regular_chunks: int

    @property
    def first_regular_chunk_index(self):
        return int(self.independent_first_frame)

    def valid_pair_count(self, future_offset):
        return self.num_regular_chunks - future_offset

    def regular_pair_indices(self, future_offset):
        return tuple(
            (self.first_regular_chunk_index + i, self.first_regular_chunk_index + i + future_offset)
            for i in range(self.valid_pair_count(future_offset))
        )


def build_predictive_chunk_layout(
    *, num_frames, block_size, independent_first_frame, consumer_name="FCP"
):
    start = int(independent_first_frame)
    if block_size <= 0 or num_frames <= start or (num_frames - start) % block_size:
        raise ValueError(
            f"{consumer_name}: incompatible chunk layout T={num_frames}, "
            f"block={block_size}, independent_first_frame={independent_first_frame}"
        )
    return PredictiveChunkLayout(
        num_frames, block_size, independent_first_frame, start, (num_frames - start) // block_size
    )


def read_predictive_multi_chunk_config(args, model):
    # The public resolver validates names and value domains before model loading.
    cfg = OmegaConf.to_container(args.future_prediction, resolve=True)
    if not cfg["enabled"]:
        raise ValueError("FCP requires future_prediction.enabled=true")
    layout = build_predictive_chunk_layout(
        num_frames=args.image_or_video_shape[1],
        block_size=args.num_frame_per_block,
        independent_first_frame=args.independent_first_frame,
    )
    layers = cfg["source_layers"]
    if layers[-1] >= len(model.blocks):
        raise ValueError("future_prediction.source_layers exceeds the loaded backbone")
    depth = cfg["mini_dit_depth"]
    init_source = cfg.get("init_source")
    if cfg["init_mode"] == "copy":
        if init_source is None:
            if len(layers) != 1:
                raise ValueError("future_prediction.init_source is required for multi-layer copy")
            init_source = list(range(layers[0] + 1, layers[0] + 1 + depth))
        if (
            len(init_source) != depth
            or len(set(init_source)) != depth
            or any(i < 0 or i >= len(model.blocks) for i in init_source)
        ):
            raise ValueError("future_prediction.init_source is incompatible with loaded blocks")
    elif "init_source" in cfg:
        raise ValueError("future_prediction.init_source must be absent for random initialization")
    hidden_dim = round(cfg["source_layer_fusion"]["hidden_ratio"] * model.dim)
    if hidden_dim <= 0:
        raise ValueError("future_prediction.source_layer_fusion.hidden_ratio gives zero width")
    if any(layout.valid_pair_count(offset) <= 0 for offset in cfg["future_offsets"]):
        raise ValueError("future_prediction.future_offsets leaves no regular pair")
    return {
        **cfg,
        "init_source": init_source,
        "source_layer_fusion": {**cfg["source_layer_fusion"], "hidden_dim": hidden_dim},
        "source_context_streams": cfg["source_context"]["streams"],
        "num_frame_per_block": args.num_frame_per_block,
        "independent_first_frame": args.independent_first_frame,
        "num_chunks": layout.num_regular_chunks,
    }


def configure_predictive_multi_chunk_from_config(model, config):
    model.configure_predictive_multi_chunk(config)

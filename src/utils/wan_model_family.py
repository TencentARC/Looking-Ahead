"""Canonical Wan model-family capabilities and latent geometry.

This module describes model facts only. It does not mutate an OmegaConf config
and therefore does not turn a ``model_name`` override into an implicit family
switch. Family-specific YAML files remain responsible for their complete data,
checkpoint, shape, and experiment settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch


@dataclass(frozen=True)
class WanModelFamily:
    """Geometry and project capability contract for one Wan checkpoint family."""

    model_name: str
    version: str
    task: str
    latent_channels: int
    native_latent_frames: int
    native_pixel_frames: int
    native_height: int
    native_width: int
    vae_temporal_stride: int
    vae_spatial_stride: int
    default_fps: int
    causal: bool
    predictive: bool
    condition_only_first_frame: bool

    @property
    def default_frame_size(self) -> tuple[int, int, int, int]:
        return (self.native_pixel_frames, 3, self.native_height, self.native_width)

    @property
    def default_latent_size(self) -> tuple[int, int, int, int]:
        return (
            self.native_latent_frames,
            self.latent_channels,
            self.native_height // self.vae_spatial_stride,
            self.native_width // self.vae_spatial_stride,
        )

    @property
    def vae_compression_factor(self) -> tuple[int, int, int]:
        return (
            self.vae_temporal_stride,
            self.vae_spatial_stride,
            self.vae_spatial_stride,
        )


_MODEL_FAMILIES = {
    "Wan2.1-T2V-1.3B": WanModelFamily(
        "Wan2.1-T2V-1.3B",
        "2.1",
        "t2v",
        16,
        21,
        81,
        480,
        832,
        4,
        8,
        16,
        True,
        True,
        True,
    ),
    "Wan2.1-T2V-14B": WanModelFamily(
        "Wan2.1-T2V-14B",
        "2.1",
        "t2v",
        16,
        21,
        81,
        480,
        832,
        4,
        8,
        16,
        True,
        True,
        True,
    ),
    "Wan2.2-TI2V-5B": WanModelFamily(
        "Wan2.2-TI2V-5B",
        "2.2",
        "ti2v",
        48,
        31,
        121,
        704,
        1280,
        4,
        16,
        24,
        True,
        True,
        True,
    ),
}


def get_wan_model_family(model_name: str) -> WanModelFamily:
    """Return a known family contract or reject an unsupported checkpoint name."""
    try:
        return _MODEL_FAMILIES[str(model_name)]
    except KeyError as error:
        supported = ", ".join(sorted(_MODEL_FAMILIES))
        raise ValueError(
            f"Unsupported Wan model_name={model_name!r}; supported values: {supported}."
        ) from error


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def validate_wan_model_geometry(
    config: Any,
    *,
    consumer_name: str,
    require_batch_one: bool = False,
) -> WanModelFamily:
    """Validate explicit config geometry without deriving or rewriting values."""
    family = get_wan_model_family(_config_value(config, "model_name"))
    shape = _config_value(config, "image_or_video_shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 5:
        raise ValueError(
            f"{consumer_name}: image_or_video_shape must be [B,T,C,H,W]; got {shape!r}."
        )
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in shape):
        raise TypeError(
            f"{consumer_name}: image_or_video_shape values must be integers; got {shape!r}."
        )
    batch, frames, channels, height, width = map(int, shape)
    if batch <= 0 or frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"{consumer_name}: image_or_video_shape values must be positive; got {shape!r}."
        )
    if require_batch_one and batch != 1:
        raise ValueError(f"{consumer_name}: requires batch dimension 1; got B={batch}.")
    if channels != family.latent_channels:
        raise ValueError(
            f"{consumer_name}: {family.model_name} requires latent C={family.latent_channels}; "
            f"got image_or_video_shape={list(shape)!r}. Select a complete family-specific YAML "
            "instead of overriding model_name alone."
        )
    fps = _config_value(config, "fps", family.default_fps)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (Integral, float))
        or int(fps) != fps
        or fps <= 0
    ):
        raise ValueError(f"{consumer_name}: fps must be a positive integer; got {fps!r}.")
    return family


def require_wan_capability(
    config: Any,
    capability: str,
    *,
    consumer_name: str,
) -> WanModelFamily:
    """Validate geometry and require a named boolean family capability."""
    family = validate_wan_model_geometry(config, consumer_name=consumer_name)
    if not hasattr(family, capability):
        raise ValueError(f"Unknown Wan family capability {capability!r}.")
    if not bool(getattr(family, capability)):
        raise ValueError(
            f"{consumer_name}: model_name={family.model_name!r} does not support {capability}."
        )
    return family


def validate_wan_checkpoint_family(
    state_dict: Mapping[str, Any],
    *,
    model_name: str,
    consumer_name: str,
) -> None:
    """Reject an identifiable cross-family checkpoint before state loading.

    Historical checkpoints do not carry explicit family metadata, so the
    diffusion patch-embedding input channel is the stable discriminator across
    dense, predictive, EMA, MoT, and FSDP namespaces. If a partial checkpoint
    omits that tensor, the existing strict state-dict validation remains
    authoritative. Recognizable Wan2.1 patch output widths additionally
    distinguish 1.3B from 14B, which share the same 16-channel VAE. This does
    not reinterpret unknown/reduced architectures or migrate checkpoint keys.
    """
    family = get_wan_model_family(model_name)
    for key, value in state_dict.items():
        if not key.endswith("patch_embedding.weight") or not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2:
            continue
        checkpoint_channels = int(value.shape[1])
        if checkpoint_channels != family.latent_channels:
            raise ValueError(
                f"{consumer_name}: checkpoint tensor {key!r} has latent C="
                f"{checkpoint_channels}, but model_name={model_name!r} requires "
                f"C={family.latent_channels}. Cross-family Wan2.1/Wan2.2 "
                "checkpoint loading is not supported."
            )
        wan21_widths = {"Wan2.1-T2V-1.3B": 1536, "Wan2.1-T2V-14B": 5120}
        checkpoint_width = int(value.shape[0])
        if (
            model_name in wan21_widths
            and checkpoint_width in wan21_widths.values()
            and checkpoint_width != wan21_widths[model_name]
        ):
            raise ValueError(
                f"{consumer_name}: checkpoint tensor {key!r} has hidden width "
                f"{checkpoint_width}, but model_name={model_name!r} requires "
                f"{wan21_widths[model_name]}. Cross-size Wan2.1 checkpoint loading "
                "is not supported."
            )


def validate_wan_latent_tensor(
    latent: torch.Tensor,
    config: Any,
    *,
    consumer_name: str,
) -> None:
    """Validate a project-layout ``[B,T,C,H,W]`` latent against the YAML."""
    family = validate_wan_model_geometry(config, consumer_name=consumer_name)
    if latent.ndim != 5:
        raise ValueError(
            f"{consumer_name}: latent must have shape [B,T,C,H,W]; got {tuple(latent.shape)!r}."
        )
    configured = tuple(int(value) for value in _config_value(config, "image_or_video_shape"))
    actual = tuple(int(value) for value in latent.shape)
    expected_tail = configured[1:]
    if actual[1:] != expected_tail:
        raise ValueError(
            f"{consumer_name}: latent shape {actual!r} is incompatible with "
            f"model_name={family.model_name!r} and configured [T,C,H,W]="
            f"{expected_tail!r}. Use a family-compatible dataset and YAML."
        )

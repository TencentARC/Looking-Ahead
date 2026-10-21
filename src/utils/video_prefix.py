"""Causal video-prefix geometry shared by input validation and ordinary sampling."""

import math

import torch

from src.utils.wan_model_family import get_wan_model_family


def output_geometry(config):
    family = get_wan_model_family(config["model_name"])
    if config["num_output_frames"] > 0:
        pixel_frames = config["num_output_frames"]
    else:
        duration = config["duration"]
        seconds = int(duration[:-1]) * {"s": 1, "m": 60, "h": 3600}[duration[-1].lower()]
        pixel_frames = int(config.get("fps", family.default_fps) * seconds)
    latent_frames = 1 + math.ceil((pixel_frames - 1) / family.vae_temporal_stride)
    chunk = config["num_frame_per_block"]
    if config["independent_first_frame"]:
        latent_frames = 1 + math.ceil((latent_frames - 1) / chunk) * chunk
    else:
        latent_frames = math.ceil(latent_frames / chunk) * chunk
    return pixel_frames, latent_frames


def prefix_geometry(config):
    """Return (K, condition pixels); duration includes the prefix and visible tail."""
    count = config["v2v_context_latents"]
    block = config["num_frame_per_block"]
    if type(count) is not int or count < 1:
        raise ValueError("v2v_context_latents: must be a positive non-boolean integer")
    if config["independent_first_frame"]:
        if count <= 1 or (count - 1) % block:
            raise ValueError("v2v_context_latents: requires K=1+M*block, M>=1")
    elif count % block:
        raise ValueError("v2v_context_latents: requires K=M*block, M>=1")
    family = get_wan_model_family(config["model_name"])
    condition_frames = 1 + family.vae_temporal_stride * (count - 1)
    pixels, latents = output_geometry(config)
    if pixels <= condition_frames or latents <= count:
        raise ValueError("v2v_context_latents: requested output must include visible continuation")
    if (latents - count) % block:
        raise ValueError("v2v_context_latents: generated tail must be chunk-aligned")
    return count, condition_frames


def automatic_prefix_geometry(config, available_frames):
    """Choose the longest aligned context while leaving a generated chunk."""
    if type(available_frames) is not int or available_frames < 1:
        raise ValueError("video_path: no usable frames at the configured FPS")
    family = get_wan_model_family(config["model_name"])
    pixels, latents = output_geometry(config)
    block = config["num_frame_per_block"]
    maximum = min(
        1 + (available_frames - 1) // family.vae_temporal_stride,
        1 + (pixels - 2) // family.vae_temporal_stride,
        latents - block,
    )
    if config["independent_first_frame"]:
        count = 1 + (maximum - 1) // block * block
        minimum = 1 + block
    else:
        count = maximum // block * block
        minimum = block
    if count < minimum:
        raise ValueError(
            "video_path: video is too short for an aligned V2V context "
            "at the configured FPS with visible continuation"
        )
    return prefix_geometry({**config, "v2v_context_latents": count})


def select_prefix(clean, count):
    """Use only the stored clean prefix; the unseen suffix never conditions generation."""
    if clean.shape[1] < count:
        raise ValueError(f"v2v_context_latents: requested {count}, sample has {clean.shape[1]}")
    prefix = clean[:, :count]
    if not torch.isfinite(prefix).all():
        raise ValueError("video prefix contains non-finite latent values")
    return prefix.contiguous()

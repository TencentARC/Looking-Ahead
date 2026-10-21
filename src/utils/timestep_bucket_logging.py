"""Chunk-wise timestep-bucket metrics for diffusion training logs."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from src.utils.timestep_buckets import (
    TIMESTEP_BUCKET_COUNT,
    physical_timestep_bucket_bounds,
    validate_timestep_bucket_count,
)


def validate_timestep_bucket_log_interval(value: object) -> int:
    """Validate the positive optimizer-step window used by bucket logging."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("timestep_bucket_log_interval must be a positive integer.")
    if value <= 0:
        raise ValueError("timestep_bucket_log_interval must be greater than zero.")
    return value


def physical_timestep_bucket_ids(
    timestep: torch.Tensor,
    *,
    max_timestep: int | float,
    bucket_count: int = TIMESTEP_BUCKET_COUNT,
) -> torch.Tensor:
    """Map physical scheduler timesteps to the shared canonical buckets."""
    bucket_count = validate_timestep_bucket_count(bucket_count)
    if isinstance(max_timestep, bool) or not isinstance(max_timestep, (int, float)):
        raise TypeError("max_timestep must be positive.")
    if max_timestep <= 0:
        raise ValueError("max_timestep must be greater than zero.")

    bucket_ids = torch.floor(timestep.detach().to(torch.float64) * bucket_count / max_timestep).to(
        torch.long
    )
    return bucket_ids.clamp(0, bucket_count - 1)


def build_chunkwise_loss_observations(
    frame_loss: torch.Tensor,
    timestep: torch.Tensor,
    *,
    num_frame_per_block: int,
    independent_first_frame: bool,
    exclude_first_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce frame losses to the timestep-sharing units used for logging.

    Args:
        frame_loss: Scheduler-weighted main loss in ``[B, T]``.
        timestep: Canonical physical scheduler timesteps in ``[B, T]``.
        num_frame_per_block: Number of regular latent frames sharing one
            timestep.
        independent_first_frame: Whether frame zero is a one-frame timestep
            unit outside the regular chunk layout.
        exclude_first_frame: Whether frame zero is conditioning-only and does
            not participate in the main generator loss.

    Returns:
        Flattened ``(chunk_loss, chunk_timestep)`` tensors. Every regular
        chunk contributes one mean loss and one physical timestep. An included
        independent first frame contributes one additional observation.
    """
    if frame_loss.ndim != 2 or timestep.ndim != 2:
        raise ValueError("frame_loss and timestep must both have shape [B, T].")
    if frame_loss.shape != timestep.shape:
        raise ValueError(
            "frame_loss and timestep must have identical shapes, got "
            f"{tuple(frame_loss.shape)} and {tuple(timestep.shape)}."
        )
    if isinstance(num_frame_per_block, bool) or not isinstance(num_frame_per_block, int):
        raise TypeError("num_frame_per_block must be a positive integer.")
    if num_frame_per_block <= 0:
        raise ValueError("num_frame_per_block must be greater than zero.")
    if exclude_first_frame and not independent_first_frame:
        raise ValueError("exclude_first_frame requires independent_first_frame=true.")

    if independent_first_frame:
        first_loss = frame_loss[:, :1]
        first_timestep = timestep[:, :1]
        regular_loss = frame_loss[:, 1:]
        regular_timestep = timestep[:, 1:]
    else:
        first_loss = frame_loss[:, :0]
        first_timestep = timestep[:, :0]
        regular_loss = frame_loss
        regular_timestep = timestep

    regular_frames = regular_loss.shape[1]
    if regular_frames % num_frame_per_block != 0:
        raise ValueError(
            "Regular frame count must be divisible by num_frame_per_block, got "
            f"T_regular={regular_frames}, F={num_frame_per_block}."
        )

    batch_size = frame_loss.shape[0]
    regular_chunks = regular_frames // num_frame_per_block
    regular_chunk_loss = regular_loss.reshape(
        batch_size,
        regular_chunks,
        num_frame_per_block,
    ).mean(dim=2)
    regular_chunk_timestep = regular_timestep.reshape(
        batch_size,
        regular_chunks,
        num_frame_per_block,
    )[:, :, 0]

    if independent_first_frame and not exclude_first_frame:
        chunk_loss = torch.cat((first_loss, regular_chunk_loss), dim=1)
        chunk_timestep = torch.cat((first_timestep, regular_chunk_timestep), dim=1)
    else:
        chunk_loss = regular_chunk_loss
        chunk_timestep = regular_chunk_timestep
    return chunk_loss.flatten().detach(), chunk_timestep.flatten().detach()


class TimestepBucketAccumulator:
    """Accumulate and globally reduce ten chunk-wise timestep loss buckets."""

    def __init__(
        self,
        *,
        interval: int,
        max_timestep: int,
        device: torch.device | int | str,
    ) -> None:
        self.interval = validate_timestep_bucket_log_interval(interval)
        if isinstance(max_timestep, bool) or not isinstance(max_timestep, int):
            raise TypeError("internal timestep scale must be a positive integer.")
        if max_timestep <= 0:
            raise ValueError("internal timestep scale must be greater than zero.")
        self.max_timestep = max_timestep
        self._sums = torch.zeros(
            TIMESTEP_BUCKET_COUNT,
            dtype=torch.float64,
            device=device,
        )
        self._counts = torch.zeros_like(self._sums)
        self.steps_accumulated = 0

    def update(self, chunk_loss: torch.Tensor, chunk_timestep: torch.Tensor) -> None:
        """Add one optimizer step without synchronizing the device."""
        if chunk_loss.ndim != 1 or chunk_timestep.ndim != 1:
            raise ValueError("chunk_loss and chunk_timestep must be one-dimensional.")
        if chunk_loss.shape != chunk_timestep.shape:
            raise ValueError(
                "chunk_loss and chunk_timestep must have identical shapes, got "
                f"{tuple(chunk_loss.shape)} and {tuple(chunk_timestep.shape)}."
            )
        if chunk_loss.device != self._sums.device or chunk_timestep.device != self._sums.device:
            raise ValueError("Timestep bucket observations must be on the accumulator device.")

        bucket_ids = physical_timestep_bucket_ids(
            chunk_timestep,
            max_timestep=self.max_timestep,
        )
        self._sums.scatter_add_(
            0,
            bucket_ids,
            chunk_loss.detach().to(torch.float64),
        )
        self._counts.scatter_add_(
            0,
            bucket_ids,
            torch.ones_like(chunk_loss, dtype=torch.float64),
        )
        self.steps_accumulated += 1

    @property
    def ready(self) -> bool:
        """Whether the configured number of optimizer steps has accumulated."""
        return self.steps_accumulated >= self.interval

    def flush(self) -> tuple[torch.Tensor, int] | None:
        """All-reduce sums/counts, reset the local window, and return packed stats."""
        if self.steps_accumulated == 0:
            return None
        packed = torch.stack((self._sums, self._counts))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        window_steps = self.steps_accumulated
        self._sums.zero_()
        self._counts.zero_()
        self.steps_accumulated = 0
        return packed, window_steps

    def format_metrics(
        self,
        packed: torch.Tensor,
        *,
        loss_prefix: str = "generator_loss_timestep_bucket",
        count_prefix: str = "generator_loss_timestep_bucket_count",
    ) -> dict[str, float | int]:
        """Convert globally reduced stats to stable W&B metric names."""
        if packed.shape != (2, TIMESTEP_BUCKET_COUNT):
            raise ValueError(
                "Packed timestep bucket stats must have shape "
                f"[2, {TIMESTEP_BUCKET_COUNT}], got {tuple(packed.shape)}."
            )
        for name, prefix in (("loss_prefix", loss_prefix), ("count_prefix", count_prefix)):
            if not isinstance(prefix, str) or not prefix or prefix.endswith("/"):
                raise ValueError(
                    f"{name} must be a non-empty metric prefix without a trailing '/'."
                )
        sums = packed[0].detach().cpu()
        counts = packed[1].detach().cpu()
        metrics: dict[str, float | int] = {}
        endpoint_width = max(1, len(str(self.max_timestep)))
        for bucket_index in range(TIMESTEP_BUCKET_COUNT):
            lower_float, upper_float = physical_timestep_bucket_bounds(
                bucket_index,
                max_timestep=self.max_timestep,
            )
            lower = round(lower_float)
            upper = round(upper_float)
            label = f"{lower:0{endpoint_width}d}_{upper:0{endpoint_width}d}"
            count = int(counts[bucket_index].item())
            loss = float(sums[bucket_index].item() / count) if count > 0 else math.nan
            metrics[f"{loss_prefix}/{label}"] = loss
            metrics[f"{count_prefix}/{label}"] = count
        overall_count = int(counts.sum().item())
        overall_loss = float(sums.sum().item() / overall_count) if overall_count > 0 else math.nan
        metrics[f"{loss_prefix}/overall"] = overall_loss
        metrics[f"{count_prefix}/overall"] = overall_count
        return metrics


def build_predictive_aux_chunkwise_loss_observations(
    frame_loss: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce predictive auxiliary frame losses to future-chunk observations.

    Args:
        frame_loss: Scheduler-weighted auxiliary loss in
            ``[B, valid_chunks, block_size]`` before source/offset averaging.
        timestep: Physical auxiliary target timesteps with the same shape.

    Returns:
        Flattened detached ``(chunk_loss, chunk_timestep)`` tensors. Each valid
        future chunk contributes its block-mean loss and its shared timestep.
    """
    if frame_loss.ndim != 3 or timestep.ndim != 3:
        raise ValueError(
            "Predictive auxiliary frame loss and timestep must both have shape "
            "[B, valid_chunks, block_size]."
        )
    if frame_loss.shape != timestep.shape:
        raise ValueError(
            "Predictive auxiliary frame loss and timestep must have identical shapes, got "
            f"{tuple(frame_loss.shape)} and {tuple(timestep.shape)}."
        )
    if frame_loss.shape[2] == 0:
        raise ValueError("Predictive auxiliary block_size must be greater than zero.")

    chunk_loss = frame_loss.mean(dim=2)
    chunk_timestep = timestep[:, :, 0]
    return chunk_loss.flatten().detach(), chunk_timestep.flatten().detach()

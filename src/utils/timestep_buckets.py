"""Shared physical-timestep bucket definitions for training and diagnostics."""

from __future__ import annotations

TIMESTEP_BUCKET_COUNT = 10


def validate_timestep_bucket_count(value: object) -> int:
    """Return a positive integer bucket count or raise a precise error."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("timestep bucket count must be a positive integer.")
    if value <= 0:
        raise ValueError("timestep bucket count must be greater than zero.")
    return value


def physical_timestep_bucket_bounds(
    bucket_index: int,
    *,
    max_timestep: int | float,
    bucket_count: int = TIMESTEP_BUCKET_COUNT,
) -> tuple[float, float]:
    """Return the canonical half-open physical-timestep interval for one bucket.

    The last interval also contains ``max_timestep`` because runtime bucket IDs
    clamp that endpoint into the final bucket.
    """
    bucket_count = validate_timestep_bucket_count(bucket_count)
    if isinstance(bucket_index, bool) or not isinstance(bucket_index, int):
        raise TypeError("bucket_index must be an integer.")
    if not 0 <= bucket_index < bucket_count:
        raise ValueError(f"bucket_index must be in [0, {bucket_count}); got {bucket_index!r}.")
    if isinstance(max_timestep, bool) or not isinstance(max_timestep, (int, float)):
        raise TypeError("max_timestep must be positive.")
    if max_timestep <= 0:
        raise ValueError("max_timestep must be greater than zero.")

    width = float(max_timestep) / bucket_count
    return bucket_index * width, (bucket_index + 1) * width

"""Shared training-snapshot scheduling helpers."""

from __future__ import annotations


def should_run_training_snapshot(config, step: int) -> bool:
    """Return whether this step should execute visualization inference.

    ``--no_visualize`` is a hard opt-out for generated video snapshots. Besides
    honoring the CLI contract, this keeps trainers from leaving the VAE on
    CPU/FP32 and then invoking a GPU/BF16 snapshot decode.
    """

    snapshot_iters = int(config.snapshot_iters)
    return (
        not bool(config.no_visualize) and snapshot_iters != -1 and int(step) % snapshot_iters == 0
    )

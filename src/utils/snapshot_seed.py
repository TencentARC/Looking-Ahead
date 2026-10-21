"""Isolated preview RNG address, independent of training noise."""

_MASK64 = (1 << 64) - 1
_TORCH_SEED_MODULUS = (1 << 63) - 1
_SNAPSHOT_STREAM_ID = 0x434F4646534E4150


def _mix_u64(value: int) -> int:
    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value & _MASK64


def derive_condition_only_snapshot_seed(
    *,
    base_seed: int,
    global_rank: int,
    global_step: int,
) -> int:
    """Derive the isolated snapshot-noise seed from stable position fields."""
    if global_rank < 0 or global_step < 0:
        raise ValueError("global_rank and global_step must be non-negative.")
    mixed = _mix_u64(int(base_seed) ^ _SNAPSHOT_STREAM_ID)
    mixed = _mix_u64(mixed ^ int(global_rank))
    mixed = _mix_u64(mixed ^ int(global_step))
    return mixed % _TORCH_SEED_MODULUS

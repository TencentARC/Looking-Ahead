"""Strict-resume protocol for the single-EMA checkpoint format.

Only boundary metadata/config and the next batch are hashed. External inputs and
large DCP/rank payloads are never checksum-scanned.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

FORMAT = "looking_ahead.training_resume"
SCHEMA_VERSION = 1
BOUNDARY_NAME = "next_optimizer_update"
METADATA_FILENAME = "metadata.json"
COMPLETE_FILENAME = "COMPLETE"
_RANK_STATE_TEMPLATE = "rank_{rank:05d}.pt"
_OPERATIONAL_CONFIG_PATHS = frozenset(
    {
        "resume_checkpoint_path",
        "save_resume_checkpoint",
        "max_iters",
        "log_iters",
        "max_checkpoints",
        "expdir",
        "checkpoint_dir",
        "snapshot_dir",
        "log_dir",
        "wandb_save_dir",
        "disable_wandb",
        "wandb_host",
        "wandb_entity",
        "wandb_project",
        "config_name",
    }
)
_LEGACY_PROVENANCE_PATHS = frozenset({"checkpoint_path", "generator_ckpt"})


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_config(config: Any) -> dict[str, Any]:
    """Return the resolved public config; credentials are environment-only."""

    if OmegaConf.is_config(config):
        result = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, Mapping):
        result = dict(config)
    else:
        raise TypeError("Exact-resume config must be an OmegaConf or mapping.")
    result = _jsonable(result)
    if not isinstance(result, dict):
        raise TypeError("Resolved exact-resume config must be a mapping.")
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(_flatten(item, child))
        if not value:
            result[prefix] = []
        return result
    return {prefix: value}


def validate_config_compatibility(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    saved_step: int,
) -> None:
    """Apply the confirmed operational allowlist and emit a deterministic difference report."""

    saved_flat = _flatten(saved)
    current_flat = _flatten(current)
    differences: list[str] = []
    all_paths = sorted(set(saved_flat) | set(current_flat))
    for path in all_paths:
        old = saved_flat.get(path, "<missing>")
        new = current_flat.get(path, "<missing>")
        if old == new:
            continue
        root = path.split(".", 1)[0].split("[", 1)[0]
        if root in _OPERATIONAL_CONFIG_PATHS:
            continue
        if root in _LEGACY_PROVENANCE_PATHS:
            if new is None or new == old:
                continue
            differences.append(f"{path}: legacy provenance must be null or {old!r}, got {new!r}")
            continue
        if path.startswith("data_sources[") and path.rsplit(".", 1)[-1] in {
            "data_path",
            "annotation_path",
        }:
            continue
        differences.append(f"{path}: saved={old!r}, current={new!r}")

    old_max = saved.get("max_iters", -1)
    new_max = current.get("max_iters", -1)
    if old_max == -1:
        if new_max != -1:
            differences.append(f"max_iters: saved run is unbounded (-1), current={new_max!r}")
    elif new_max != -1 and (new_max < old_max or new_max <= saved_step):
        differences.append(
            "max_iters: must remain equal/increase and exceed saved step "
            f"{saved_step}; saved={old_max!r}, current={new_max!r}"
        )

    for field in sorted(_LEGACY_PROVENANCE_PATHS):
        old = saved.get(field, None)
        new = current.get(field, None)
        if new is not None and new != old:
            message = f"{field}: legacy provenance must be null or {old!r}, got {new!r}"
            if message not in differences:
                differences.append(message)

    if differences:
        joined = "\n  - ".join(differences)
        raise ValueError(f"Exact-resume configuration mismatch:\n  - {joined}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def current_topology(config: Any) -> dict[str, Any]:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    node_rank = int(os.environ.get("GROUP_RANK", rank // max(local_world_size, 1)))
    return {
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_count": world_size // max(local_world_size, 1),
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "distributed_backend": dist.get_backend() if dist.is_initialized() else None,
        "sharding_strategy": str(_cfg_get(config, "sharding_strategy", "full")),
        "batch_size": int(_cfg_get(config, "batch_size", 1)),
        "num_workers": int(_cfg_get(config, "num_workers", 8)),
    }


def _expected_topologies(config: Any, world_size: int) -> list[dict[str, Any]]:
    """Derive the fixed logical rank layout without an object collective."""

    local = current_topology(config)
    local_world_size = int(local["local_world_size"])
    if local_world_size <= 0 or world_size % local_world_size != 0:
        raise ValueError(
            "Exact-resume topology requires world_size to be divisible by "
            f"LOCAL_WORLD_SIZE; got {world_size} and {local_world_size}."
        )
    return [
        {
            **local,
            "rank": rank,
            "local_rank": rank % local_world_size,
            "node_rank": rank // local_world_size,
        }
        for rank in range(world_size)
    ]


def validate_topology(saved: Sequence[Mapping[str, Any]], config: Any) -> None:
    current = current_topology(config)
    rank = current["rank"]
    if rank >= len(saved):
        raise ValueError(f"Exact-resume topology mismatch: rank {rank} absent from saved topology.")
    if len(saved) != current["world_size"]:
        raise ValueError(
            "Exact-resume topology mismatch: "
            f"saved world_size={len(saved)}, current={current['world_size']}."
        )
    expected = dict(saved[rank])
    if expected != current:
        raise ValueError(
            "Exact-resume logical topology mismatch for rank "
            f"{rank}: saved={expected!r}, current={current!r}."
        )


def capture_cpu_replay_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def restore_cpu_replay_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def _probe_rng_state(state: Mapping[str, Any]) -> dict[str, str]:
    py_rng = random.Random()
    py_rng.setstate(state["python"])
    python_probe = [py_rng.getrandbits(64) for _ in range(8)]

    np_rng = np.random.RandomState()
    np_rng.set_state(state["numpy"])
    numpy_probe = np_rng.bytes(64)

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.set_state(state["torch_cpu"])
    cpu_probe = torch.rand(16, generator=cpu_generator, dtype=torch.float64).numpy().tobytes()

    result = {
        "python": _sha256_bytes(repr(python_probe).encode("ascii")),
        "numpy": _sha256_bytes(numpy_probe),
        "torch_cpu": _sha256_bytes(cpu_probe),
    }
    for index, cuda_state in enumerate(state.get("torch_cuda", [])):
        generator = torch.Generator(device=f"cuda:{index}")
        generator.set_state(cuda_state)
        probe = (
            torch.rand(
                16,
                generator=generator,
                device=f"cuda:{index}",
                dtype=torch.float32,
            )
            .cpu()
            .numpy()
            .tobytes()
        )
        result[f"torch_cuda:{index}"] = _sha256_bytes(probe)
    return result


def capture_rng_state() -> dict[str, Any]:
    state = capture_cpu_replay_rng()
    state["torch_cuda"] = [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
    state["probe"] = _probe_rng_state(state)
    return state


def assert_rng_state_unchanged(state: Mapping[str, Any]) -> None:
    current = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [item.cpu() for item in torch.cuda.get_rng_state_all()],
    }
    actual_probe = _probe_rng_state(current)
    if actual_probe != state["probe"]:
        raise RuntimeError(
            "Exact-resume checkpoint capture advanced a training RNG stream: "
            f"before={state['probe']!r}, after={actual_probe!r}."
        )


def restore_and_validate_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state.get("torch_cuda", []))
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError(
            "Exact-resume CUDA RNG device-count mismatch: "
            f"saved={len(cuda_states)}, current={torch.cuda.device_count()}."
        )
    torch.cuda.set_rng_state_all(cuda_states)
    current_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [item.cpu() for item in torch.cuda.get_rng_state_all()],
    }
    actual_probe = _probe_rng_state(current_state)
    if actual_probe != state["probe"]:
        raise RuntimeError(
            f"Exact-resume RNG probe mismatch: saved={state['probe']!r}, restored={actual_probe!r}."
        )


def batch_identity(batch: Any) -> str:
    """Return a stable content identity for an accepted or prefetched batch."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"numpy\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence\0")
            for item in value:
                update(item)
        else:
            digest.update(type(value).__name__.encode("ascii", errors="replace"))
            digest.update(b"\0")
            digest.update(repr(value).encode("utf-8", errors="replace"))
            digest.update(b"\0")

    update(batch)
    return digest.hexdigest()


def _accumulator_state(accumulator: Any) -> dict[str, Any]:
    return {
        "interval": int(accumulator.interval),
        "max_timestep": int(accumulator.max_timestep),
        "sums": accumulator._sums.detach().cpu().clone(),
        "counts": accumulator._counts.detach().cpu().clone(),
        "steps_accumulated": int(accumulator.steps_accumulated),
    }


def _restore_accumulator_state(accumulator: Any, saved: Mapping[str, Any]) -> None:
    if int(saved["interval"]) != int(accumulator.interval):
        raise ValueError("Exact-resume timestep-bucket interval mismatch.")
    if int(saved["max_timestep"]) != int(accumulator.max_timestep):
        raise ValueError("Exact-resume timestep-bucket maximum mismatch.")
    for field, target in (("sums", accumulator._sums), ("counts", accumulator._counts)):
        source = saved[field]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(f"Exact-resume timestep-bucket {field} shape mismatch.")
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    accumulator.steps_accumulated = int(saved["steps_accumulated"])


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _collective_device() -> torch.device:
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _raise_if_any_rank_failed(stage: str, local_error: str | None) -> None:
    """Propagate rank-local failure using one fixed-size tensor collective."""

    if not dist.is_initialized():
        if local_error is not None:
            raise RuntimeError(f"Exact-resume {stage} failed: {local_error}")
        return

    succeeded = torch.tensor(
        int(local_error is None),
        dtype=torch.int32,
        device=_collective_device(),
    )
    dist.all_reduce(succeeded, op=dist.ReduceOp.MIN)
    if not bool(succeeded.item()):
        detail = local_error or "another rank reported a local failure"
        raise RuntimeError(f"Exact-resume {stage} failed: {detail}")


def _broadcast_save_token(rank: int) -> str:
    """Broadcast a UUID as a fixed 16-byte tensor, never as a Python object."""

    if not dist.is_initialized():
        return uuid.uuid4().hex
    device = _collective_device()
    if rank == 0:
        token = torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=device)
    else:
        token = torch.empty(16, dtype=torch.uint8, device=device)
    dist.broadcast(token, src=0)
    return bytes(token.cpu().tolist()).hex()


def _checkpoint_process_group() -> Any:
    """Keep DCP's Python-object coordination off the NCCL/GPU data plane."""

    if not dist.is_initialized() or dist.get_backend() != "nccl":
        return None
    try:
        return dist.new_group(backend="gloo")
    except RuntimeError as error:
        raise RuntimeError(
            "Exact resume on an NCCL training group requires an auxiliary Gloo "
            "process group for bounded CPU-side DCP coordination."
        ) from error


def _commit_rank_local_directory(temporary: Path, final: Path) -> None:
    """Commit a local shard, tolerating a peer's rename on a shared filesystem."""

    try:
        os.replace(temporary, final)
    except OSError:
        # On a shared filesystem all ranks address the same directory. One
        # peer wins the rename and the others observe the already-finalized
        # path. On rank-local filesystems every rank takes the normal branch.
        if temporary.exists() or not final.is_dir():
            raise


def _config_digest(config):
    return _sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())


def _validate_envelope(value, *, label):
    if (
        value.get("format") != FORMAT
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"{label}: expected {FORMAT} schema_version={SCHEMA_VERSION}")


def resolve_bundle_path(path):
    """Select only the explicit directory or its immediate resume_state child."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Resume bundle directory not found: {root}")

    def has_layout(candidate):
        return any(
            (candidate / name).exists()
            for name in (METADATA_FILENAME, COMPLETE_FILENAME, "distributed_state")
        ) or any(candidate.glob("rank_*.pt"))

    direct = has_layout(root)
    child = root / "resume_state"
    if direct and has_layout(child):
        raise ValueError(f"Ambiguous resume path (direct and child layouts): {root}")
    return root if direct else child


def _load_and_validate_bundle(bundle):
    if not (bundle / COMPLETE_FILENAME).is_file():
        raise RuntimeError(f"Resume bundle is incomplete (missing COMPLETE): {bundle}")
    encoded = (bundle / METADATA_FILENAME).read_bytes()
    metadata = json.loads(encoded)
    completion = json.loads((bundle / COMPLETE_FILENAME).read_text())
    _validate_envelope(metadata, label="metadata")
    _validate_envelope(completion, label="COMPLETE")
    if completion.get("metadata_sha256") != _sha256_bytes(encoded):
        raise RuntimeError("Resume metadata checksum mismatch")
    if metadata.get("config_sha256") != _config_digest(metadata.get("resolved_config")):
        raise RuntimeError("Resume resolved-config digest mismatch")
    if metadata.get("boundary") != BOUNDARY_NAME:
        raise ValueError("Unsupported resume boundary")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Invalid resume completed step")
    if completion.get("step") != step:
        raise ValueError("Resume completion step disagreement")
    topologies = metadata["topology"]
    expected = [_RANK_STATE_TEMPLATE.format(rank=rank) for rank in range(len(topologies))]
    if (
        not expected
        or metadata.get("rank_state_files") != expected
        or sorted(p.name for p in bundle.glob("rank_*.pt")) != expected
    ):
        raise RuntimeError("Resume rank-file coverage mismatch; collect all rank-local pieces")
    if [item["rank"] for item in topologies] != list(range(len(topologies))):
        raise ValueError("Resume topology rank coverage mismatch")
    # Inspect bounded DCP metadata and existence, never hash large tensor files.
    dcp_metadata = dcp.FileSystemReader(bundle / "distributed_state").read_metadata()
    payload_model_schema = {
        key.removeprefix("model."): {
            "shape": list(value.size),
            "dtype": str(value.properties.dtype),
        }
        for key, value in dcp_metadata.state_dict_metadata.items()
        if key.startswith("model.") and isinstance(value, dcp.TensorStorageMetadata)
    }
    if payload_model_schema != metadata["model_schema"]:
        raise ValueError("Resume DCP model schema disagrees with metadata")
    for storage in dcp_metadata.storage_data.values():
        if not (bundle / "distributed_state" / storage.relative_path).is_file():
            raise RuntimeError(f"Resume DCP payload is missing: {storage.relative_path}")
    return metadata


def _trainer_rank_state(trainer):
    return {
        "timestep_bucket": _accumulator_state(trainer.timestep_bucket_accumulator),
        "predictive_timestep_buckets": {
            key: _accumulator_state(value)
            for key, value in trainer.predictive_timestep_accumulators.items()
        },
        "preview_cached": trainer._val_batch_cache is not None,
    }


def _restore_trainer_rank_state(trainer, saved):
    _restore_accumulator_state(trainer.timestep_bucket_accumulator, saved["timestep_bucket"])
    auxiliary = saved["predictive_timestep_buckets"]
    if set(auxiliary) != set(trainer.predictive_timestep_accumulators):
        raise ValueError("Resume predictive timestep-bucket keys mismatch")
    for key, value in trainer.predictive_timestep_accumulators.items():
        _restore_accumulator_state(value, auxiliary[key])


def _ema_rank_state(trainer):
    ema = trainer.generator_ema
    if ema is None:
        return None
    return {
        "decay": ema.decay,
        "num_updates": ema.num_updates,
        "shadow": {key: value.detach().cpu().clone() for key, value in ema.shadow.items()},
    }


def _restore_ema_rank_state(trainer, saved):
    expected_active = (
        trainer.config.ema_weight > 0 and trainer.step >= trainer.config.ema_start_step
    )
    if (saved is not None) != expected_active:
        raise ValueError("Resume active single-EMA state mismatch")
    if saved is None:
        trainer.generator_ema = None
        return
    if saved["decay"] != trainer.config.ema_weight or saved["num_updates"] < 0:
        raise ValueError("Resume EMA controller mismatch")
    if trainer.generator_ema is None:
        trainer._start_generator_ema()
    ema = trainer.generator_ema
    if set(saved["shadow"]) != set(ema.shadow):
        raise ValueError("Resume EMA tensor set mismatch")
    for key, tensor in saved["shadow"].items():
        target = ema.shadow[key]
        if tensor.shape != target.shape or tensor.dtype != torch.float32:
            raise ValueError(f"Resume EMA tensor schema mismatch: {key}")
    # The rank already owns its complete local shadow. Copy into that storage
    # after validating every tensor, avoiding two additional GPU-sized shadows
    # from a device-converted mapping followed by load_state_dict's clones.
    with torch.no_grad():
        for key, tensor in saved["shadow"].items():
            ema.shadow[key].copy_(tensor)
    ema.num_updates = int(saved["num_updates"])


def _model_schema(state):
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in state.items()
        if torch.is_tensor(value)
    }


class ExactResumeManager:
    """Rank-local save/restore with distributed commit and public format checks."""

    def __init__(self, config, *, logger):
        self.config = config
        self.logger = logger
        self.save_enabled = config.save_resume_checkpoint
        self.resume_path = None
        self.metadata = None
        self.resolved_config = canonical_config(config)
        error = None
        try:
            if config.resume_checkpoint_path is not None:
                self.resume_path = resolve_bundle_path(config.resume_checkpoint_path)
                self.metadata = _load_and_validate_bundle(self.resume_path)
                validate_config_compatibility(
                    self.metadata["resolved_config"],
                    self.resolved_config,
                    saved_step=self.metadata["step"],
                )
                validate_topology(self.metadata["topology"], config)
                for field in _LEGACY_PROVENANCE_PATHS:
                    if self.resolved_config.get(field) is None:
                        self.resolved_config[field] = self.metadata["resolved_config"].get(field)
                self.logger.info("Resolved resume bundle: %s", self.resume_path)
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
        _raise_if_any_rank_failed("preflight", error)
        self._checkpoint_group = _checkpoint_process_group()

    def save(
        self,
        trainer,
        *,
        checkpoint_dir,
        data_cursor,
        data_replay_initial_rng,
        data_loader_initial_rng,
        next_batch_sha256,
    ):
        if not self.save_enabled:
            raise RuntimeError("Full-state saving is disabled")
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        final = Path(checkpoint_dir) / "resume_state"
        temporary = Path(checkpoint_dir) / f".resume_state.incomplete-{_broadcast_save_token(rank)}"
        error = None
        try:
            if final.exists():
                raise FileExistsError(f"Resume bundle already exists: {final}")
            temporary.mkdir(parents=True, exist_ok=True)
        except Exception as caught:
            error = str(caught)
        _raise_if_any_rank_failed("directory setup", error)
        boundary_rng = capture_rng_state()
        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
        model_state, optimizer_state = get_state_dict(
            trainer.model.generator, trainer.generator_optimizer, options=options
        )
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=temporary / "distributed_state",
            process_group=self._checkpoint_group,
        )
        error = None
        try:
            rank_state = {
                "format": FORMAT,
                "schema_version": SCHEMA_VERSION,
                "rank": rank,
                "step": int(trainer.step),
                "data_cursor": int(data_cursor),
                "data_replay_initial_rng": data_replay_initial_rng,
                "data_loader_initial_rng": data_loader_initial_rng,
                "next_batch_sha256": next_batch_sha256,
                "rng": boundary_rng,
                "ema": _ema_rank_state(trainer),
                "trainer_state": _trainer_rank_state(trainer),
                "effective_lrs": [
                    float(group["lr"]) for group in trainer.generator_optimizer.param_groups
                ],
            }
            torch.save(rank_state, temporary / _RANK_STATE_TEMPLATE.format(rank=rank))
            assert_rng_state_unchanged(boundary_rng)
        except Exception as caught:
            error = str(caught)
        _raise_if_any_rank_failed("rank-state write/RNG capture", error)
        error = None
        if rank == 0:
            try:
                metadata = {
                    "format": FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "boundary": BOUNDARY_NAME,
                    "step": int(trainer.step),
                    "resolved_config": self.resolved_config,
                    "config_sha256": _config_digest(self.resolved_config),
                    "topology": _expected_topologies(trainer.config, world_size),
                    "optimizer": {
                        "name": "adamw",
                        "param_group_count": len(trainer.generator_optimizer.param_groups),
                    },
                    "model_schema": _model_schema(model_state),
                    "ema": {
                        "active": trainer.generator_ema is not None,
                        "num_updates": trainer.generator_ema.num_updates
                        if trainer.generator_ema
                        else None,
                    },
                    "rank_state_files": [
                        _RANK_STATE_TEMPLATE.format(rank=item) for item in range(world_size)
                    ],
                    "parent": {"bundle": str(self.resume_path), "step": self.metadata["step"]}
                    if self.metadata
                    else None,
                    "logging": {
                        "wandb_run_id": getattr(trainer.wandb_run, "id", None),
                        "config_name": trainer.config.config_name,
                    },
                }
                _write_json(temporary / METADATA_FILENAME, metadata)
            except Exception as caught:
                error = str(caught)
        _raise_if_any_rank_failed("metadata write", error)
        error = None
        try:
            _commit_rank_local_directory(temporary, final)
        except Exception as caught:
            error = str(caught)
        _raise_if_any_rank_failed("local commit", error)
        error = None
        if rank == 0:
            try:
                completion = {
                    "format": FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "step": int(trainer.step),
                    "metadata_sha256": _sha256_bytes((final / METADATA_FILENAME).read_bytes()),
                }
                sibling = final / f".COMPLETE-{uuid.uuid4().hex}"
                _write_json(sibling, completion)
                os.replace(sibling, final / COMPLETE_FILENAME)
            except Exception as caught:
                error = str(caught)
        _raise_if_any_rank_failed("completion publication", error)
        self.logger.info("Full training-state bundle committed: %s", final)
        return final

    def restore(self, trainer):
        metadata = self.metadata
        rank = dist.get_rank() if dist.is_initialized() else 0
        error = None
        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
        model_state, optimizer_state = get_state_dict(
            trainer.model.generator, trainer.generator_optimizer, options=options
        )
        try:
            rank_state = torch.load(
                self.resume_path / _RANK_STATE_TEMPLATE.format(rank=rank),
                map_location="cpu",
                weights_only=False,
            )
            _validate_envelope(rank_state, label="rank state")
            if rank_state["rank"] != rank or rank_state["step"] != metadata["step"]:
                raise ValueError("Resume rank/step envelope disagreement")
            if rank_state["data_cursor"] != metadata["step"]:
                raise ValueError("Resume training data cursor disagreement")
            if metadata["model_schema"] != _model_schema(model_state):
                raise ValueError("Resume model schema mismatch before tensor application")
            if metadata["optimizer"] != {
                "name": "adamw",
                "param_group_count": len(trainer.generator_optimizer.param_groups),
            }:
                raise ValueError("Resume optimizer manifest mismatch")
            if metadata["ema"]["active"] != (rank_state["ema"] is not None) or (
                rank_state["ema"] is not None
                and metadata["ema"]["num_updates"] != rank_state["ema"]["num_updates"]
            ):
                raise ValueError("Resume EMA manifest disagreement")
        except Exception as caught:
            error = str(caught)
        _raise_if_any_rank_failed("rank/schema preflight", error)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(
            state,
            checkpoint_id=self.resume_path / "distributed_state",
            process_group=self._checkpoint_group,
        )
        error = None
        try:
            incompatible = set_state_dict(
                trainer.model.generator,
                trainer.generator_optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
                options=options,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"Resume model keys mismatch: {incompatible}")
            trainer.step = metadata["step"]
            _restore_ema_rank_state(trainer, rank_state["ema"])
            _restore_trainer_rank_state(trainer, rank_state["trainer_state"])
            if [
                float(group["lr"]) for group in trainer.generator_optimizer.param_groups
            ] != rank_state["effective_lrs"]:
                raise RuntimeError("Resume effective LR mismatch after optimizer restoration")
        except Exception as caught:
            error = str(caught)
        _raise_if_any_rank_failed("state application", error)
        return rank_state


def prune_completed_checkpoints(root, keep, manager, *, logger):
    """Rank-zero's existing completion evidence classifies shared/local fragments.

    Broadcast eligible numeric steps, then prune each local tree. Peers need no
    local COMPLETE and rank zero never probes peer filesystems.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    removable = []
    if rank == 0:
        completed = []
        for path in root.glob("checkpoint_model_*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.removeprefix("checkpoint_model_"))
                bundle = path / "resume_state"
                encoded = (bundle / METADATA_FILENAME).read_bytes()
                metadata = json.loads(encoded)
                completion = json.loads((bundle / COMPLETE_FILENAME).read_text())
                _validate_envelope(metadata, label="retention metadata")
                _validate_envelope(completion, label="retention COMPLETE")
                if (
                    metadata["step"] == step == completion["step"]
                    and completion["metadata_sha256"] == _sha256_bytes(encoded)
                    and metadata["config_sha256"] == _config_digest(metadata["resolved_config"])
                ):
                    completed.append((step, path.name))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # Incomplete/unclassified remnants are never eligible.
        removable = [name for _, name in sorted(completed)[:-keep]]
    names = [removable]
    if dist.is_initialized():
        dist.broadcast_object_list(names, src=0, group=manager._checkpoint_group)
    import shutil

    error = None
    for name in names[0]:
        try:
            shutil.rmtree(root / name)
        except FileNotFoundError:
            pass  # A peer may already have pruned this shared tree.
        except OSError as caught:
            error = str(caught)
    _raise_if_any_rank_failed("retention", error)
    logger.info("Completed checkpoint retention removed: %s", names[0])

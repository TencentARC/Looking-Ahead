"""FSDP readers/writers and one scalar EMA; no implicit fabric overrides."""

import os
import tempfile
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from src.utils.checkpoint_state import (
    normalize_generator_fsdp_key,
    normalize_generator_fsdp_state_dict,
)


def launch_distributed_job():
    """Direct Python uses a local process group; torchrun uses its own rendezvous."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if dist.is_initialized():
        return
    kwargs = {
        "backend": "nccl",
        "device_id": torch.device("cuda", local_rank),
        "timeout": timedelta(minutes=30),
    }
    if "RANK" in os.environ:
        dist.init_process_group(**kwargs)
    else:
        descriptor, filename = tempfile.mkstemp(prefix="looking-ahead-pg-")
        os.close(descriptor)
        os.unlink(filename)
        dist.init_process_group(init_method=f"file://{filename}", rank=0, world_size=1, **kwargs)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def fsdp_state_dict(model):
    """All ranks collect; only global rank zero retains ordinary weight tensors."""
    checkpoint = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    return checkpoint if is_main_process() else {}


def fsdp_wrap(module, *, sharding_strategy, mixed_precision, sync_module_states=False):
    precision = (
        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        if mixed_precision
        else None
    )
    strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]
    return FSDP(
        module,
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=int(5e7)),
        sharding_strategy=strategy,
        mixed_precision=precision,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=sync_module_states,
    )


def _fsdp_sharding_strategy(module: torch.nn.Module):
    strategy = getattr(module, "sharding_strategy", None)
    if strategy is not None:
        return strategy
    fsdp_state = getattr(module, "_fsdp_state", None)
    if fsdp_state is not None:
        return getattr(fsdp_state, "sharding_strategy", None)
    return None


def _effective_no_shard(module: torch.nn.Module) -> bool:
    """Return true when all FSDP wrappers use effective ``NO_SHARD``.

    A plain module is treated like ``NO_SHARD`` for EMA serialization helpers;
    this keeps the direct save/load path testable without requiring CUDA/FSDP.
    """
    fsdp_modules = [submodule for submodule in module.modules() if isinstance(submodule, FSDP)]
    if not fsdp_modules:
        return True
    return all(
        _fsdp_sharding_strategy(submodule) == ShardingStrategy.NO_SHARD
        for submodule in fsdp_modules
    )


class EMA_FSDP:
    """
    Exponential Moving Average for an FSDP-wrapped model.

    Design notes
    ------------
    * **Sharded, on-GPU shadow.** Each rank stores an EMA shadow of *only its
      own local shard* of every ``flat_param`` (FSDP1) / ``DTensor`` (FSDP2),
      keeping the buffer in fp32 on the same device as the live parameter.
      This means ``update`` is a pure local elementwise op -- no
      ``summon_full_params``, no all-gather, no host-device traffic, no
      cross-rank synchronization -- so it is essentially free compared to
      the optimizer step.
    * **Collective only on save.** ``full_state_dict`` is the only path that
      materializes full EMA tensors for checkpointing. Sharded FSDP uses
      ``summon_full_params``; effective ``NO_SHARD`` returns the already-full
      local EMA shadows directly. It should only run at checkpoint time, never
      per-step.
    * Non-floating-point buffers (e.g. ``num_batches_tracked``) are never
      averaged.
    """

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        decay: float = 0.999,
    ):
        self.decay = decay
        self.num_updates = 0

        # name -> fp32 tensor on the *same device & with the same local
        # shape* as the corresponding live (sharded) parameter. Each rank
        # only holds its own shard.
        self.shadow: dict = {}
        self._is_main = is_main_process()

        self._init_shadow(fsdp_module)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _iter_named_tensors(self, module):
        yield from module.named_parameters()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        # No collective: each rank just clones its own shard in fp32. For
        # FSDP1 the local shard is a 1-D ``flat_param`` slice; for FSDP2 it
        # is a ``DTensor`` whose ``_local_tensor`` we read transparently
        # (``detach().clone()`` on a DTensor preserves the DTensor wrapping,
        # which is exactly what we want for arithmetic later).
        for n, t in self._iter_named_tensors(fsdp_module):
            self.shadow[n] = t.detach().to(dtype=torch.float32).clone()

    # ------------------------------------------------------------------
    # Update -- per-step hot path
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, fsdp_module):
        """Per-rank local EMA update. **No** collective communication.

        Cost is one fp32 mul_+add_ over the local shard, which is dominated
        by memory bandwidth and negligible next to the optimizer step.
        """
        self.num_updates += 1
        d = self.decay
        one_minus_d = 1.0 - d

        for n, t in self._iter_named_tensors(fsdp_module):
            shadow = self.shadow.get(n)
            if shadow is None:
                # New / unseen parameter -- initialize on the fly.
                self.shadow[n] = t.detach().to(dtype=torch.float32).clone()
                continue
            # Cast live param to fp32 *on device* (cheap if already fp32,
            # otherwise a fused kernel). No host transfer, no collective.
            src = t.detach()
            if src.dtype != torch.float32:
                src = src.to(dtype=torch.float32)
            shadow.mul_(d).add_(src, alpha=one_minus_d)

    # ------------------------------------------------------------------
    # (De)serialization helpers
    # ------------------------------------------------------------------
    def state_dict(self):
        """Return the *local-shard* shadow on every rank.

        Note: this is **not** a full state dict. Use :meth:`full_state_dict`
        if you want the rank-0-only fully-gathered version (e.g. for
        checkpointing). Each rank's tensors live on its own device.
        """
        return self.shadow

    @torch.no_grad()
    def load_full_state_dict(
        self, full_sd, fsdp_module, *, allow_missing_aux=False, expected_shapes=None
    ):
        """Restore ordinary EMA from each rank's validated full source state.

        All-rank writeback avoids PyTorch's unsupported rank0-only writeback.
        The temporary swap restores the original independent live shards.
        """
        full_sd = normalize_generator_fsdp_state_dict(full_sd)
        expected = {
            normalize_generator_fsdp_key(name): tensor
            for name, tensor in self._iter_named_tensors(fsdp_module)
        }
        missing = set(expected) - set(full_sd)
        unexpected = set(full_sd) - set(expected)
        if unexpected or (missing and not allow_missing_aux):
            raise RuntimeError("EMA checkpoint has missing or unexpected parameter keys.")
        if allow_missing_aux:
            from src.utils.predictive_training_checkpoint import is_predictive_aux_key

            if any(not is_predictive_aux_key(k) for k in missing):
                raise RuntimeError("EMA checkpoint is missing backbone parameters")
        if expected_shapes is not None:
            for name, value in full_sd.items():
                if tuple(value.shape) != expected_shapes[name]:
                    raise RuntimeError(f"EMA checkpoint shape mismatch: {name}")
        if _effective_no_shard(fsdp_module):
            for name, tensor in expected.items():
                if name in full_sd and full_sd[name].shape != tensor.shape:
                    raise RuntimeError(f"EMA checkpoint shape mismatch: {name}")
            new_shadow = {}
            for n, t in self._iter_named_tensors(fsdp_module):
                canonical_name = normalize_generator_fsdp_key(n)
                src = full_sd.get(canonical_name, t)
                new_shadow[n] = (
                    src.detach()
                    .to(
                        device=t.device,
                        dtype=torch.float32,
                    )
                    .clone()
                )
            self.shadow = new_shadow
            return

        # Backup live shards (each rank saves its own).
        backup = {n: t.data.clone() for n, t in self._iter_named_tensors(fsdp_module)}

        try:
            with FSDP.summon_full_params(
                fsdp_module, writeback=True, rank0_only=False, offload_to_cpu=False
            ):
                for name, tensor in self._iter_named_tensors(fsdp_module):
                    canonical_name = normalize_generator_fsdp_key(name)
                    if canonical_name in full_sd:
                        tensor.copy_(
                            full_sd[canonical_name].to(dtype=tensor.dtype, device=tensor.device)
                        )
            new_shadow = {
                name: tensor.detach().to(dtype=torch.float32).clone()
                for name, tensor in self._iter_named_tensors(fsdp_module)
            }
            self.shadow = new_shadow
        finally:
            for name, tensor in self._iter_named_tensors(fsdp_module):
                tensor.data.copy_(backup[name])

    @torch.no_grad()
    def full_state_dict(self, fsdp_module):
        """Return the *full* EMA state dict on rank 0 (empty dict elsewhere).

        This should only be used at checkpoint time. Effective ``NO_SHARD``
        returns the already-full EMA shadows directly. Sharded FSDP temporarily
        swaps the live FSDP parameters for the EMA shadows, summons them in
        full on rank 0, copies them out, then restores the live weights -- all
        without persisting any change.
        """
        if _effective_no_shard(fsdp_module):
            if not self._is_main:
                return {}
            return {
                normalize_generator_fsdp_key(n): shadow.detach()
                .to("cpu", dtype=torch.float32)
                .clone()
                for n, shadow in self.shadow.items()
            }

        # Clone the live local shards before copying EMA into their storage.
        # ``use_orig_params=True`` exposes views into FSDP's FlatParameter, so
        # retaining ``t.data`` here would only retain an alias. FSDP may reuse
        # and overwrite that storage while summoning full parameters, turning
        # an apparent restore into an accidental live <- EMA switch.
        backup = {}
        for n, t in self._iter_named_tensors(fsdp_module):
            shadow = self.shadow.get(n)
            if shadow is None:
                continue
            backup[n] = t.detach().clone()
            t.copy_(shadow.to(device=t.device, dtype=t.dtype), non_blocking=True)

        # CPU FSDP parameters are already host-resident. Asking PyTorch to
        # offload a rank0-only summon back to the same device can hang under
        # Gloo, while CUDA training still benefits from the original offload.
        offload_to_cpu = any(t.device.type != "cpu" for t in backup.values())

        try:
            with FSDP.summon_full_params(
                fsdp_module,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=offload_to_cpu,
            ):
                if not self._is_main:
                    out = {}
                else:
                    out = {}
                    for n, t in self._iter_named_tensors(fsdp_module):
                        out[normalize_generator_fsdp_key(n)] = (
                            t.detach().to(device="cpu", dtype=torch.float32).clone()
                        )
        finally:
            # Restore the independent live-shard copies regardless of what
            # happened above. Copying into the post-context parameter views
            # preserves FSDP's storage/view invariants.
            for n, t in self._iter_named_tensors(fsdp_module):
                if n in backup:
                    t.copy_(backup[n], non_blocking=True)

        return out

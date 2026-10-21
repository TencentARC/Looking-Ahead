"""FCP flow objective with independent future timesteps."""

import logging
import os
from dataclasses import dataclass

import torch

from src.model.fm import FlowMatching
from src.utils.predictive_multi_chunk_config import (
    build_predictive_chunk_layout,
    configure_predictive_multi_chunk_from_config,
    read_predictive_multi_chunk_config,
)
from src.utils.predictive_training_checkpoint import prepare_predictive_checkpoint
from src.utils.timestep_bucket_logging import (
    build_chunkwise_loss_observations,
    build_predictive_aux_chunkwise_loss_observations,
)
from src.utils.wan_model_family import validate_wan_checkpoint_family


@dataclass
class PredictiveLossComponents:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    aux_loss: torch.Tensor
    log_dict: dict


class PredictiveFlowMatching(FlowMatching):
    _predictive_consumer_name = "FCP"
    _aux_future_timestep_stream_id = 0xA17F_71A5_7E9D_51E1

    def __init__(self, args, device):
        self._independent_aux_scheduler = None
        self._loaded_live_checkpoint_has_predictive_aux = None
        super().__init__(args, device)
        # Validate the physical interval against the effective scheduler eagerly.
        self._build_aux_future_timestep_candidate_indices()

    def _after_generator_initialized(self, args):
        cfg = read_predictive_multi_chunk_config(args, self.generator.model)
        self.predictive_multi_chunk_config = cfg
        self.generator.model.num_frame_per_block = args.num_frame_per_block
        self.generator.model.independent_first_frame = args.independent_first_frame
        configure_predictive_multi_chunk_from_config(self.generator.model, cfg)
        self._predictive_checkpoint_shapes = {
            k: tuple(v.shape) for k, v in self.generator.state_dict().items()
        }

    def _prepare_checkpoint(self, state, *, ema=False):
        validate_wan_checkpoint_family(
            state,
            model_name=self.args.model_name,
            consumer_name="FCP EMA" if ema else "FCP training",
        )
        return prepare_predictive_checkpoint(
            state,
            self.predictive_multi_chunk_config,
            self._predictive_checkpoint_shapes,
            label="FCP EMA" if ema else "FCP training",
            ema=ema,
        )

    def load_generator_checkpoint_state_dict(self, state_dict):
        state, predictive = self._prepare_checkpoint(state_dict)
        self.generator.load_state_dict(
            state, strict=False
        )  # All exceptions checked before mutation.
        self._loaded_live_checkpoint_has_predictive_aux = predictive

    def load_generator_ema_checkpoint_state_dict(self, ema, state_dict, generator):
        state, predictive = self._prepare_checkpoint(state_dict, ema=True)
        live = self._loaded_live_checkpoint_has_predictive_aux
        if live is not None and live != predictive:
            raise RuntimeError("FCP live/EMA checkpoint identity conflict")
        ema.load_full_state_dict(
            state,
            generator,
            allow_missing_aux=True,
            expected_shapes=self._predictive_checkpoint_shapes,
        )

    def _runtime_predictive_chunk_layout(self, num_frame: int):
        """Validate runtime frames against the parser-normalized chunk layout."""
        cfg = self.predictive_multi_chunk_config
        layout = build_predictive_chunk_layout(
            num_frames=num_frame,
            block_size=cfg["num_frame_per_block"],
            independent_first_frame=cfg.get("independent_first_frame", False),
            consumer_name=self._predictive_consumer_name,
        )
        if layout.num_regular_chunks != cfg["num_chunks"]:
            raise ValueError(
                f"{self._predictive_consumer_name} runtime/config chunk mismatch: "
                f"runtime N={layout.num_regular_chunks}, configured N={cfg['num_chunks']}."
            )
        return layout

    def _get_aux_future_scheduler(self):
        """Return the effective scheduler for independent predictive targets."""
        config = getattr(self, "predictive_multi_chunk_config", {})
        aux_shift = config.get("aux_future_timestep_shift")
        if aux_shift is None:
            return self.scheduler
        if getattr(self, "_independent_aux_scheduler", None) is None:
            self._independent_aux_scheduler = self.scheduler.clone_with_shift(aux_shift)
            logging.info(
                "Predictive independent aux scheduler enabled: main_shift=%s, aux_shift=%s",
                self.scheduler.shift,
                aux_shift,
            )
        return self._independent_aux_scheduler

    @staticmethod
    def _mix_aux_future_timestep_seed(*components: int) -> int:
        """Mix position components into one deterministic 64-bit seed."""
        mask = (1 << 64) - 1
        value = 0x9E37_79B9_7F4A_7C15
        for component in components:
            value ^= (
                (int(component) & mask)
                + 0x9E37_79B9_7F4A_7C15
                + ((value << 6) & mask)
                + (value >> 2)
            )
            value &= mask
        value = (value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9 & mask
        value = (value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB & mask
        return int(value ^ (value >> 31))

    @staticmethod
    def _generator_device(device) -> torch.device:
        if isinstance(device, int):
            return torch.device("cuda", device)
        return torch.device(device)

    def _sample_aux_future_timestep_indices(
        self,
        *,
        batch_size: int,
        valid_chunks: int,
        block_size: int,
        future_offset: int,
        global_step: int,
        global_rank: int,
        training_seed: int,
        first_regular_chunk_index: int = 0,
        candidate_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample chunk-shared aux timestep indices with isolated per-position seeds."""
        device = self._generator_device(self.device)
        if candidate_indices is not None:
            if candidate_indices.ndim != 1 or candidate_indices.numel() == 0:
                raise ValueError("Aux future timestep candidate indices must be non-empty and 1D.")
            candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
        indices = torch.empty(
            batch_size,
            valid_chunks,
            block_size,
            device=device,
            dtype=torch.long,
        )
        for batch_slot in range(batch_size):
            for retained_position in range(valid_chunks):
                source_chunk_index = retained_position
                global_source_chunk_index = first_regular_chunk_index + source_chunk_index
                seed_components = [
                    training_seed,
                    global_rank,
                    global_step,
                    batch_slot,
                    global_source_chunk_index,
                    future_offset,
                ]
                seed_components.append(self._aux_future_timestep_stream_id)
                seed = self._mix_aux_future_timestep_seed(*seed_components)
                generator = torch.Generator(device=device).manual_seed(seed)
                if candidate_indices is None:
                    timestep_index = torch.randint(
                        0,
                        self._get_aux_future_scheduler().num_train_timesteps,
                        (1,),
                        device=device,
                        dtype=torch.long,
                        generator=generator,
                    )
                else:
                    ordinal = torch.randint(
                        0,
                        candidate_indices.numel(),
                        (1,),
                        device=device,
                        dtype=torch.long,
                        generator=generator,
                    )
                    timestep_index = candidate_indices[ordinal]
                indices[batch_slot, retained_position, :] = timestep_index
        return indices

    def _build_aux_future_timestep_candidate_indices_for_bounds(
        self,
        bounds: list[float],
        *,
        field_name: str,
    ) -> torch.Tensor:
        """Return scheduler indices inside one configured closed physical-t interval."""
        lower, upper = bounds
        timesteps = self._get_aux_future_scheduler().timesteps
        if timesteps.ndim != 1 or timesteps.numel() == 0:
            raise ValueError(
                f"{field_name} requires a non-empty one-dimensional scheduler.timesteps table."
            )

        candidate_indices = torch.nonzero(
            (timesteps >= lower) & (timesteps <= upper),
            as_tuple=False,
        ).flatten()
        if candidate_indices.numel() == 0:
            table_min = float(timesteps.min().item())
            table_max = float(timesteps.max().item())
            raise ValueError(
                f"{field_name} "
                f"[{lower}, {upper}] selects no scheduler timestep; runtime "
                f"scheduler.timesteps spans [{table_min}, {table_max}] across "
                f"{timesteps.numel()} entries."
            )
        return candidate_indices.to(device=self._generator_device(self.device))

    def _build_aux_future_timestep_candidate_indices(self) -> torch.Tensor:
        """Return scheduler indices inside the legacy global physical-t interval."""
        return self._build_aux_future_timestep_candidate_indices_for_bounds(
            self.predictive_multi_chunk_config["aux_future_timestep_range"],
            field_name="future_prediction.aux_future_timestep_range",
        )

    def _build_independent_aux_future_inputs(
        self,
        *,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        batch_size: int,
        num_frame: int,
        global_step: int,
        global_rank: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Build offset-keyed aux noisy chunks, targets and timesteps."""
        cfg = self.predictive_multi_chunk_config
        block_size = cfg["num_frame_per_block"]
        if clean_latent.shape[1] != num_frame or noise.shape[1] != num_frame:
            raise ValueError(
                "Independent predictive aux inputs must match the runtime temporal length: "
                f"clean={clean_latent.shape[1]}, noise={noise.shape[1]}, expected={num_frame}."
            )
        layout = self._runtime_predictive_chunk_layout(num_frame)
        num_chunks = layout.num_regular_chunks
        training_seed = int(getattr(self.args, "seed", 0))
        regular_start = layout.regular_frame_start
        clean_chunks = clean_latent[:, regular_start:].unflatten(1, (num_chunks, block_size))
        noise_chunks = noise[:, regular_start:].unflatten(1, (num_chunks, block_size))
        candidate_indices = self._build_aux_future_timestep_candidate_indices()

        aux_noisy_chunks_by_offset = {}
        aux_training_target_by_offset = {}
        aux_timestep_by_offset = {}
        for offset in cfg["future_offsets"]:
            valid_chunks = layout.valid_pair_count(offset)
            if valid_chunks < 1:
                raise ValueError(
                    f"{self._predictive_consumer_name} has no retained regular predictive "
                    f"pair for offset {offset}: T={num_frame}, F={block_size}, N={num_chunks}, "
                    f"independent_first_frame={layout.independent_first_frame}, "
                    f"valid_pair_count="
                    f"{valid_chunks}."
                )
            timestep_indices = self._sample_aux_future_timestep_indices(
                batch_size=batch_size,
                valid_chunks=valid_chunks,
                block_size=block_size,
                future_offset=int(offset),
                global_step=global_step,
                global_rank=global_rank,
                training_seed=training_seed,
                first_regular_chunk_index=layout.first_regular_chunk_index,
                candidate_indices=candidate_indices,
            )
            aux_timestep = (
                self._get_aux_future_scheduler()
                .timesteps[timestep_indices]
                .to(
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            clean_future = clean_chunks[:, offset : offset + valid_chunks]
            noise_future = noise_chunks[:, offset : offset + valid_chunks]
            flat_clean = clean_future.reshape(
                batch_size * valid_chunks * block_size,
                *clean_latent.shape[2:],
            )
            flat_noise = noise_future.reshape_as(flat_clean)
            aux_noisy = (
                self._get_aux_future_scheduler()
                .add_noise(
                    flat_clean,
                    flat_noise,
                    aux_timestep.reshape(-1),
                )
                .reshape(batch_size, valid_chunks, block_size, *clean_latent.shape[2:])
            )
            aux_target = self._get_aux_future_scheduler().training_target(
                clean_future,
                noise_future,
                aux_timestep,
            )
            aux_noisy_chunks_by_offset[int(offset)] = aux_noisy
            aux_training_target_by_offset[int(offset)] = aux_target
            aux_timestep_by_offset[int(offset)] = aux_timestep
        return aux_noisy_chunks_by_offset, aux_training_target_by_offset, aux_timestep_by_offset

    def _auxiliary_losses(self, aux_outputs, targets, timesteps):
        cfg = self.predictive_multi_chunk_config
        scheduler = self._get_aux_future_scheduler()
        total = next(iter(targets.values())).new_zeros(())
        log, losses, times = {}, {}, {}
        for offset, weight in zip(cfg["future_offsets"], cfg["offset_loss_weights"], strict=True):
            expected = cfg["num_chunks"] - offset
            if aux_outputs[offset]["valid_chunks"] != expected:
                raise ValueError(f"FCP valid pair mismatch for offset {offset}")
            target, timestep = targets[offset], timesteps[offset]
            source_losses = []
            for source in cfg["source_context_streams"]:
                pred = aux_outputs[offset]["flow_pred_by_source"][source]
                if pred.shape != target.shape:
                    raise ValueError(f"FCP prediction/target shape mismatch for {source}/{offset}")
                frame_loss = (pred.float() - target.float()).square().mean((3, 4, 5))
                frame_loss = frame_loss * scheduler.training_weight(timestep.flatten()).reshape(
                    *timestep.shape
                )
                chunk_loss, chunk_time = build_predictive_aux_chunkwise_loss_observations(
                    frame_loss, timestep
                )
                losses[(source, offset)], times[(source, offset)] = chunk_loss, chunk_time
                source_loss = frame_loss.mean()
                source_losses.append(source_loss)
                log[f"pred_{source}_{offset}_loss"] = source_loss.detach()
            offset_loss = torch.stack(source_losses).mean()
            total = total + weight * offset_loss
            log[f"pred_{offset}_loss"] = offset_loss.detach()
            log[f"valid_{offset}"] = float(expected)
        log["_predictive_aux_chunk_loss"] = losses
        log["_predictive_aux_chunk_timestep"] = times
        return total, log

    def generator_loss_components(
        self,
        image_or_video_shape,
        conditional_dict,
        unconditional_dict,
        clean_latent,
        initial_latent=None,
        global_step=None,
        global_rank=None,
    ):
        noise = torch.randn_like(clean_latent)
        batch, frames = image_or_video_shape[:2]
        index = self._get_timestep(0, 1000, batch, frames, self.num_frame_per_block)
        canonical = self.scheduler.timesteps[index]
        timestep = canonical.to(dtype=self.dtype)
        noisy = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1), timestep.flatten()
        ).unflatten(0, (batch, frames))
        target = self.scheduler.training_target(clean_latent, noise, timestep)
        aux_noisy, aux_target, aux_timestep = self._build_independent_aux_future_inputs(
            clean_latent=clean_latent,
            noise=noise,
            batch_size=batch,
            num_frame=frames,
            global_step=0 if global_step is None else int(global_step),
            global_rank=0 if global_rank is None else int(global_rank),
        )
        flow, x0, outputs = self.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent,
            return_predictive_aux=True,
            predictive_aux_noisy_chunks_by_offset=aux_noisy,
            predictive_aux_timestep_chunks_by_offset=aux_timestep,
        )
        frame_loss = self._flow_matching_frame_loss(flow, target, timestep, batch, frames)
        main, metrics = self._reduce_main_frame_loss(frame_loss)
        chunk_loss, chunk_time = build_chunkwise_loss_observations(
            frame_loss,
            canonical,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            exclude_first_frame=self.condition_only_first_frame,
        )
        aux, aux_metrics = self._auxiliary_losses(outputs, aux_target, aux_timestep)
        total = main + aux
        log = {
            "x0": clean_latent.detach(),
            "x0_pred": x0.detach(),
            "_generator_loss": main.detach(),
            "pred_total_loss": total.detach(),
            "_generator_chunk_loss": chunk_loss,
            "_generator_chunk_timestep": chunk_time,
            **metrics,
            **aux_metrics,
        }
        if os.environ.get("DEBUG") == "1":
            log.update(
                {
                    f"debug_aux_timestep_mean_{offset}": value.detach().float().mean()
                    for offset, value in aux_timestep.items()
                }
            )
        return PredictiveLossComponents(total, main, aux, log)

    def generator_loss(self, *args, **kwargs):
        components = self.generator_loss_components(*args, **kwargs)
        return components.total_loss, components.log_dict

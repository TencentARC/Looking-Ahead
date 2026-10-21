"""Selected TF/FCP diffusion trainer, single EMA and strict training-state resume."""

import gc
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torchcodec.encoders import VideoEncoder

from src.utils.caption_dropout import (
    replace_prompt_embeds_with_negative,
    sample_caption_dropout_mask,
)
from src.utils.checkpoint_state import normalize_generator_fsdp_state_dict
from src.utils.ckpt_cache import load_ckpt
from src.utils.distributed import (
    EMA_FSDP,
    barrier,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
)
from src.utils.optimizer import (
    LinearLrWarmup,
    build_generator_optimizer,
    optimizer_learning_rate_metrics,
)
from src.utils.timestep_bucket_logging import TimestepBucketAccumulator
from src.utils.train_dataset import build_training_dataloader
from src.utils.training_snapshot import should_run_training_snapshot
from src.utils.wan_model_family import validate_wan_latent_tensor


def select_generator_checkpoint_state_dict(
    payload, *, require_raw_generator_checkpoint, checkpoint_path
):
    if "generator" in payload:
        return payload["generator"]
    if "model" in payload:
        return payload["model"]
    if "generator_ema" in payload:
        if require_raw_generator_checkpoint:
            raise RuntimeError(
                f"Raw generator checkpoint required; only generator_ema: {checkpoint_path}"
            )
        return payload["generator_ema"]
    return payload


def initialize_wandb(config):
    """Normal production initialization is online; failures propagate unchanged."""
    wandb.login(host=config.wandb_host, key=os.environ.get("WANDB_API_KEY"))
    return wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        name=config.config_name,
        mode="online",
        entity=config.wandb_entity,
        project=config.wandb_project,
        dir=config.wandb_save_dir,
    )


def reject_existing_targets(paths, *, device, label):
    conflict = next((str(path) for path in paths if path.exists()), None)
    vote = torch.tensor(int(conflict is not None), device=device)
    if dist.is_initialized():
        dist.all_reduce(vote, op=dist.ReduceOp.MAX)
    if vote.item():
        raise FileExistsError(
            f"{label}: existing target on a participating filesystem: {conflict or 'another rank'}"
        )


def propagate_write_failure(error, *, device, label):
    """Writers report failure before the next collective or any retention step."""
    failed = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        if error is not None:
            raise error
        raise RuntimeError(f"{label}: writing failed on another participating rank")


def preview_manifest(batch, *, step, global_rank):
    samples = []
    for index, prompt in enumerate(batch["prompts"]):
        stem = f"rank_{global_rank:04d}_sample_{index:04d}"
        samples.append(
            {
                "index": index,
                "prompt": prompt,
                "videos": {kind: f"{stem}_{kind}.mp4" for kind in ("gt", "gen")},
            }
        )
    return {"step": step, "global_rank": global_rank, "samples": samples}


class DiffusionTrainer:
    def __init__(self, config):
        if config.future_prediction.enabled:
            from src.model.predictive_fm import PredictiveFlowMatching as Objective
        else:
            from src.model.fm import FlowMatching as Objective

        self.config = config
        self.step = 0
        self.logger = logging.getLogger("looking_ahead")
        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.is_main_process = self.global_rank == 0
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.exact_resume_manager = None
        if config.save_resume_checkpoint or config.resume_checkpoint_path is not None:
            from src.utils.exact_resume import ExactResumeManager

            self.exact_resume_manager = ExactResumeManager(config, logger=self.logger)
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        seed = int(config.seed) + self.global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.wandb_run = (
            initialize_wandb(config) if self.is_main_process and not config.disable_wandb else None
        )
        self.output_path = Path(config.checkpoint_dir)
        self.snapshot_path = Path(config.snapshot_dir)
        os.environ.setdefault(
            "WAN_VIS_CAUSAL_MASK_DIR", str(self.snapshot_path / "causal_mask_vis")
        )
        self.model = Objective(config, self.device)
        if config.generator_ckpt and config.resume_checkpoint_path is None:
            payload = load_ckpt(config.generator_ckpt)
            state = select_generator_checkpoint_state_dict(
                payload,
                require_raw_generator_checkpoint=config.require_raw_generator_checkpoint,
                checkpoint_path=config.generator_ckpt,
            )
            self.model.load_generator_checkpoint_state_dict(state)
        from src.utils.grad_clip import (
            bind_predictive_grad_clip_plan,
            build_predictive_grad_clip_plan,
        )

        plan = (
            build_predictive_grad_clip_plan(self.model.generator)
            if config.separate_predictive_grad_clip
            else None
        )
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            sync_module_states=config.generator_fsdp_sync_module_states,
        )
        self.predictive_grad_clip_plan = (
            bind_predictive_grad_clip_plan(plan, self.model.generator) if plan is not None else None
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
        )
        if not config.no_visualize:
            self.model.vae.to(device=self.device, dtype=self.dtype)
        self.generator_optimizer = build_generator_optimizer(config, self.model.generator)
        self.lr_warmup = LinearLrWarmup(
            self.generator_optimizer,
            config.lr_warmup_steps,
            config.lr_drop_steps,
            config.lr_drop_gamma,
        )
        self.generator_ema = None
        if config.ema_weight > 0 and self.step >= config.ema_start_step:
            self._start_generator_ema()
            if config.generator_ckpt and config.resume_checkpoint_path is None:
                sibling = Path(config.generator_ckpt).with_name("model_ema.pt")
                if sibling.is_file():
                    payload = load_ckpt(str(sibling))
                    state = normalize_generator_fsdp_state_dict(
                        payload.get("generator_ema", payload)
                    )
                    self.model.load_generator_ema_checkpoint_state_dict(
                        self.generator_ema, state, self.model.generator
                    )
        self._train_dataloader = build_training_dataloader(config, is_train=True)
        self._train_iterator = None
        self._exact_prefetched_batch = None
        self._exact_data_cursor = 0
        self._exact_data_replay_initial_rng = None
        self._exact_loader_initial_rng = None
        self.val_dataloader = build_training_dataloader(config, is_train=False)
        self._val_batch_cache = None
        self.unconditional_dict = None
        self.timestep_bucket_accumulator = TimestepBucketAccumulator(
            interval=config.timestep_bucket_log_interval, max_timestep=1000, device=self.device
        )
        self.predictive_timestep_accumulators = {
            (source, int(offset)): TimestepBucketAccumulator(
                interval=config.timestep_bucket_log_interval, max_timestep=1000, device=self.device
            )
            for source in (
                config.future_prediction.source_context.streams
                if config.future_prediction.enabled
                else []
            )
            for offset in config.future_prediction.get("future_offsets", [])
        }
        self.logger.info(
            "Training initialized rank=%d world_size=%d seed=%d optimizer=adamw",
            self.global_rank,
            dist.get_world_size(),
            seed,
        )

        if (
            self.exact_resume_manager is not None
            and self.exact_resume_manager.resume_path is not None
        ):
            self._restore_exact_training_state()

    def _read_training_batch(self):
        if self._train_iterator is None:
            if (
                self.exact_resume_manager is not None
                and self._exact_data_replay_initial_rng is None
            ):
                from src.utils.exact_resume import capture_cpu_replay_rng

                self._exact_data_replay_initial_rng = capture_cpu_replay_rng()
                self._exact_loader_initial_rng = (
                    self._train_dataloader.generator.get_state().clone()
                )
            self._train_iterator = iter(self._train_dataloader)
        try:
            return next(self._train_iterator)
        except StopIteration:
            # Finite streams cycle; resampled exhaustion is a RuntimeError.
            self._train_iterator = iter(self._train_dataloader)
            return next(self._train_iterator)

    def _next_training_batch(self):
        if self._exact_prefetched_batch is None:
            batch = self._read_training_batch()
        else:
            batch = self._exact_prefetched_batch
            self._exact_prefetched_batch = None
        if self.exact_resume_manager is not None:
            self._exact_data_cursor += 1
        return batch

    def _restore_exact_training_state(self):
        from src.utils.exact_resume import (
            _raise_if_any_rank_failed,
            batch_identity,
            restore_and_validate_rng_state,
            restore_cpu_replay_rng,
        )

        manager = self.exact_resume_manager
        state = manager.restore(self)
        error = None
        try:
            # Replay each reader's deterministic stream using its original loader
            # generator. Warm caches before restoring boundary RNG, deliberately last.
            restore_cpu_replay_rng(state["data_replay_initial_rng"])
            self._train_dataloader.generator.set_state(state["data_loader_initial_rng"])
            self._train_iterator = None
            self._exact_data_replay_initial_rng = state["data_replay_initial_rng"]
            self._exact_loader_initial_rng = state["data_loader_initial_rng"]
            for _ in range(state["data_cursor"]):
                self._read_training_batch()
            next_batch = self._read_training_batch()
            actual = batch_identity(next_batch)
            if actual != state["next_batch_sha256"]:
                raise RuntimeError(
                    f"Resume next-batch identity mismatch: saved={state['next_batch_sha256']}, replayed={actual}"
                )
            self._exact_data_cursor = state["data_cursor"]
            self._exact_prefetched_batch = next_batch
            if state["trainer_state"]["preview_cached"]:
                self._get_val_batch()
            with torch.no_grad():
                self.unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * self.config.batch_size
                )
                self.unconditional_dict = {
                    key: value.detach() for key, value in self.unconditional_dict.items()
                }
            if self.wandb_run is not None:
                self.wandb_run.config.update(
                    {
                        "exact_resume_parent_bundle": str(manager.resume_path),
                        "exact_resume_parent_step": self.step,
                        "exact_resume_parent_wandb_run_id": manager.metadata["logging"].get(
                            "wandb_run_id"
                        ),
                    },
                    allow_val_change=True,
                )
            restore_and_validate_rng_state(state["rng"])
        except Exception as caught:
            error = str(caught)
        _raise_if_any_rank_failed("data replay/RNG restore", error)
        self.logger.info(
            "Exact training state restored from %s at completed step %d",
            manager.resume_path,
            self.step,
        )

    def _save_exact_resume_state(self):
        from src.utils.exact_resume import batch_identity

        if self._exact_prefetched_batch is None:
            self._exact_prefetched_batch = self._read_training_batch()
        directory = self.output_path / f"checkpoint_model_{self.step:06d}"
        self.exact_resume_manager.save(
            self,
            checkpoint_dir=directory,
            data_cursor=self._exact_data_cursor,
            data_replay_initial_rng=self._exact_data_replay_initial_rng,
            data_loader_initial_rng=self._exact_loader_initial_rng,
            next_batch_sha256=batch_identity(self._exact_prefetched_batch),
        )
        self._prune_old_checkpoints()
        barrier()

    def _start_generator_ema(self):
        self.generator_ema = EMA_FSDP(self.model.generator, decay=float(self.config.ema_weight))
        self.logger.info(
            "EMA started at completed step %d decay=%s", self.step, self.config.ema_weight
        )

    def _update_generator_ema(self):
        if self.config.ema_weight <= 0 or self.step < self.config.ema_start_step:
            return
        if self.generator_ema is None:
            self._start_generator_ema()
        else:
            self.generator_ema.update(self.model.generator)

    def _log_metrics(self, metrics):
        if not self.is_main_process:
            return
        if not self.config.disable_wandb:
            wandb.log(metrics, step=self.step)
        self.logger.info("step=%d metrics=%s", self.step, metrics)

    def _flush_timestep_bucket_metrics(self):
        result = self.timestep_bucket_accumulator.flush()
        if result is not None:
            packed, window_steps = result
            self._log_metrics(
                {
                    **self.timestep_bucket_accumulator.format_metrics(packed),
                    "timestep_bucket_window_steps": window_steps,
                }
            )

        for (source, offset), accumulator in self.predictive_timestep_accumulators.items():
            result = accumulator.flush()
            if result is not None:
                packed, _ = result
                self._log_metrics(
                    accumulator.format_metrics(
                        packed,
                        loss_prefix=f"pred_aux_loss_timestep_bucket/{source}/offset_{offset}",
                        count_prefix=f"pred_aux_loss_timestep_bucket_count/{source}/offset_{offset}",
                    )
                )

    def train_one_step(self, batch):
        clean = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        clean = clean[:, : self.config.image_or_video_shape[1]]
        validate_wan_latent_tensor(clean, self.config, consumer_name="training batch")
        prompts = batch["prompts"]
        batch_size = len(prompts)
        with torch.no_grad():
            conditional = self.model.text_encoder(text_prompts=prompts)
            if (
                self.unconditional_dict is None
                or self.unconditional_dict["prompt_embeds"].shape[0] != batch_size
            ):
                self.unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                self.unconditional_dict = {
                    key: value.detach() for key, value in self.unconditional_dict.items()
                }
            mask = sample_caption_dropout_mask(
                self.config.caption_dropout_p,
                batch_size,
                base_seed=int(self.config.seed),
                global_rank=self.global_rank,
                global_step=self.step,
            )
            if self.config.caption_dropout_p > 0:
                conditional = replace_prompt_embeds_with_negative(
                    conditional, self.unconditional_dict, mask
                )
        self.lr_warmup.apply(self.step)
        shape = list(self.config.image_or_video_shape)
        shape[0] = batch_size
        loss, log_dict = self.model.generator_loss(
            shape,
            conditional,
            self.unconditional_dict,
            clean,
            global_step=self.step,
            global_rank=self.global_rank,
        )
        self.generator_optimizer.zero_grad()
        loss.backward()
        clip_metrics = {}
        if self.predictive_grad_clip_plan is not None:
            from src.utils.grad_clip import clip_predictive_grad_norms_

            norm, main_norm, aux_norm = clip_predictive_grad_norms_(
                self.model.generator,
                self.predictive_grad_clip_plan,
                max_main_norm=self.config.max_grad_norm,
                max_predictive_norm=self.config.max_predictive_grad_norm,
            )
            clip_metrics = {
                "main_grad_norm": main_norm.item(),
                "predictive_grad_norm": aux_norm.item(),
            }
        else:
            norm = self.model.generator.clip_grad_norm_(self.config.max_grad_norm)
        self.generator_optimizer.step()
        self.step += 1
        self.timestep_bucket_accumulator.update(
            log_dict["_generator_chunk_loss"], log_dict["_generator_chunk_timestep"]
        )
        for key, accumulator in self.predictive_timestep_accumulators.items():
            accumulator.update(
                log_dict["_predictive_aux_chunk_loss"][key],
                log_dict["_predictive_aux_chunk_timestep"][key],
            )
        metrics = {
            **clip_metrics,
            "generator_loss": log_dict.get("_generator_loss", loss.detach()).item(),
            "grad_norm": norm.item(),
            **optimizer_learning_rate_metrics(self.generator_optimizer),
        }
        if self.config.caption_dropout_p > 0:
            metrics["caption_dropout_fraction"] = mask.float().mean().item()
        for key, value in log_dict.items():
            if not key.startswith("_") and torch.is_tensor(value) and value.numel() == 1:
                metrics[key] = value.item()
        self._log_metrics(metrics)
        if self.timestep_bucket_accumulator.ready:
            self._flush_timestep_bucket_metrics()
        if self.step % self.config.gc_interval == 0:
            gc.collect()
        return metrics

    def _prune_old_checkpoints(self):
        keep = self.config.max_checkpoints
        if keep is None or keep <= 0:
            return
        if self.config.save_resume_checkpoint:
            from src.utils.exact_resume import prune_completed_checkpoints

            prune_completed_checkpoints(
                self.output_path, keep, self.exact_resume_manager, logger=self.logger
            )
            return
        directories = sorted(
            path for path in self.output_path.glob("checkpoint_model_*") if path.is_dir()
        )
        for path in directories[:-keep]:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            except OSError as error:
                self.logger.warning("Failed to prune %s: %s", path, error)

    def save(self):
        directory = self.output_path / f"checkpoint_model_{self.step:06d}"
        reject_existing_targets([directory], device=self.device, label="checkpoint save")
        state = fsdp_state_dict(self.model.generator)
        error = None
        if self.is_main_process:
            try:
                directory.mkdir(parents=True)
                torch.save({"generator": state}, directory / "model.pt")
                self.logger.info("Live weights saved: %s", directory / "model.pt")
            except Exception as caught:
                error = caught
        propagate_write_failure(error, device=self.device, label="live checkpoint")
        if self.generator_ema is not None and self.step > self.config.ema_start_step:
            ema_state = self.generator_ema.full_state_dict(self.model.generator)
            # Source-ordered integer metadata is copied, never EMA-averaged.
            if self.config.future_prediction.enabled:
                ema_state["model.predictive_multi_chunk_source_layers_metadata"] = (
                    self.model.generator.model.predictive_multi_chunk_source_layers_metadata.detach()
                    .cpu()
                    .clone()
                )
            error = None
            if self.is_main_process:
                try:
                    torch.save({"generator_ema": ema_state}, directory / "model_ema.pt")
                    self.logger.info("EMA weights saved: %s", directory / "model_ema.pt")
                except Exception as caught:
                    error = caught
            propagate_write_failure(error, device=self.device, label="EMA checkpoint")
        barrier()
        if not self.config.save_resume_checkpoint and self.is_main_process:
            self._prune_old_checkpoints()

    def _get_val_batch(self):
        if self._val_batch_cache is None:
            self._val_batch_cache = next(iter(self.val_dataloader))
        return self._val_batch_cache

    def save_video_from_latent(self, tensor, path, *, is_latent):
        if is_latent:
            tensor = self.model.vae.decode_to_pixel(
                tensor.to(self.device, self.dtype), use_cache=False
            )
            tensor = tensor * 127.5 + 127.5
        else:
            tensor = tensor * 255
        video = tensor.clamp(0, 255).byte().cpu().squeeze(0)
        VideoEncoder(video, frame_rate=self.config.get("fps", self.model.vae.default_fps)).to_file(
            str(path)
        )

    @torch.no_grad()
    def snapshot(self):
        batch = self._get_val_batch()
        manifest = preview_manifest(batch, step=self.step, global_rank=self.global_rank)
        directory = self.snapshot_path / f"{self.step:06d}"
        metadata = directory / f"rank_{self.global_rank:04d}_samples.json"
        targets = [metadata] + [
            directory / name for sample in manifest["samples"] for name in sample["videos"].values()
        ]
        reject_existing_targets(targets, device=self.device, label="training preview")
        directory.mkdir(parents=True, exist_ok=True)
        self.model.generator.eval()
        try:
            video = self.model.inference(
                batch,
                list(self.config.image_or_video_shape),
                global_step=self.step,
                global_rank=self.global_rank,
            )
            error = None
            try:
                for sample in manifest["samples"]:
                    index = sample["index"]
                    clean = batch["clean_latent"][
                        index : index + 1, : self.config.image_or_video_shape[1]
                    ]
                    self.save_video_from_latent(
                        clean, directory / sample["videos"]["gt"], is_latent=True
                    )
                    self.save_video_from_latent(
                        video[index : index + 1],
                        directory / sample["videos"]["gen"],
                        is_latent=False,
                    )
                with metadata.open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, indent=2)
            except Exception as caught:
                error = caught
            propagate_write_failure(error, device=self.device, label="training preview")
        finally:
            self.model.generator.train()
        self.logger.info("Training preview complete: %s", metadata)

    def train(self):
        while self.config.max_iters == -1 or self.step < self.config.max_iters:
            start = time.perf_counter()
            if self.step % 20 == 0:
                torch.cuda.empty_cache()
            self.train_one_step(self._next_training_batch())
            self._update_generator_ema()
            if not self.config.no_save and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
            if should_run_training_snapshot(self.config, self.step):
                self.snapshot()
                torch.cuda.empty_cache()
            barrier()
            self._log_metrics({"step_time": time.perf_counter() - start})
            if self.config.save_resume_checkpoint and self.step % self.config.log_iters == 0:
                self._save_exact_resume_state()
        self._flush_timestep_bucket_metrics()
        self.logger.info("Training complete after %d optimizer updates", self.step)

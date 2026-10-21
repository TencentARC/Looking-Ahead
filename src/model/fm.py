"""Teacher-forced flow-matching objective and ordinary previews."""

import logging
from pathlib import Path

import torch

from src.model.base import BaseModel
from src.pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from src.utils.checkpoint_state import normalize_generator_fsdp_state_dict
from src.utils.ckpt_cache import load_ckpt
from src.utils.snapshot_seed import derive_condition_only_snapshot_seed
from src.utils.timestep_bucket_logging import build_chunkwise_loss_observations
from src.utils.wan_model_family import validate_wan_checkpoint_family
from src.utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
    isolated_torch_cpu_rng,
)


class FlowMatching(BaseModel):
    def __init__(self, args, device):
        self.condition_only_first_frame = args.condition_only_first_frame
        self.independent_first_frame = args.independent_first_frame
        self.num_frame_per_block = args.num_frame_per_block
        super().__init__(args, device)
        self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.generator.model.independent_first_frame = self.independent_first_frame
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

    def _initialize_models(self, args, device):
        def initialize_generator_and_aux():
            self.generator = WanDiffusionWrapper(
                model_name=args.model_name,
                latent_shape=args.image_or_video_shape,
                load_diffusion_pretrained=args.load_diffusion_pretrained,
                initialization_seed=int(args.seed) if args.load_diffusion_pretrained else None,
                use_predictive_causal_model=args.future_prediction.enabled,
                **args.model_kwargs,
            )
            self._after_generator_initialized(args)
            self.generator.requires_grad_(True)

        if args.load_diffusion_pretrained:
            initialize_generator_and_aux()
        else:
            # Random initialization isolates both backbone and auxiliary construction
            # under one shared seed, then restores the rank's training CPU RNG.
            with isolated_torch_cpu_rng(int(args.seed)):
                initialize_generator_and_aux()
        self.text_encoder = WanTextEncoder(model_name=args.model_name).requires_grad_(False)
        self.vae = WanVAEWrapper(model_name=args.model_name).requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        if args.checkpoint_path and args.resume_checkpoint_path is None:
            payload = load_ckpt(args.checkpoint_path)
            # Objective warm start prefers embedded EMA, then sibling EMA,
            # then live. Trainer generator_ckpt is a separate live-first path.
            if "generator_ema" in payload:
                state = payload["generator_ema"]
            else:
                sibling = Path(args.checkpoint_path).with_name("model_ema.pt")
                if sibling.is_file():
                    ema_payload = load_ckpt(str(sibling))
                    state = ema_payload.get("generator_ema", ema_payload)
                else:
                    state = payload["generator"]
            self.load_generator_checkpoint_state_dict(state)
            logging.info("Objective warm start loaded from %s", args.checkpoint_path)

    def _after_generator_initialized(self, args):
        """Configure method-specific modules before ordinary checkpoint loading."""

    def load_generator_ema_checkpoint_state_dict(self, ema, state, generator):
        ema.load_full_state_dict(state, generator)

    def load_generator_checkpoint_state_dict(self, state_dict):
        state = normalize_generator_fsdp_state_dict(state_dict)
        validate_wan_checkpoint_family(
            state, model_name=self.args.model_name, consumer_name="training generator checkpoint"
        )
        self.generator.load_state_dict(state, strict=True)

    def _flow_matching_frame_loss(
        self, flow_pred, training_target, timestep, batch_size, num_frame
    ):
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))
        return loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))

    def _reduce_main_frame_loss(self, main_frame_loss):
        if not self.condition_only_first_frame:
            return main_frame_loss.mean(), {}
        return main_frame_loss[:, 1:].mean(), {
            "main_c0_loss": main_frame_loss[:, 0].mean().detach()
        }

    def generator_loss(
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
        batch_size, num_frame = image_or_video_shape[:2]
        index = self._get_timestep(0, 1000, batch_size, num_frame, self.num_frame_per_block)
        canonical_timestep = self.scheduler.timesteps[index]
        timestep = canonical_timestep.to(dtype=self.dtype)
        noisy = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1), timestep.flatten()
        ).unflatten(0, (batch_size, num_frame))
        target = self.scheduler.training_target(clean_latent, noise, timestep)
        # Zero context augmentation: no extra draw and no augmentation interface.
        flow, x0 = self.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent,
        )
        frame_loss = self._flow_matching_frame_loss(flow, target, timestep, batch_size, num_frame)
        loss, metrics = self._reduce_main_frame_loss(frame_loss)
        chunk_loss, chunk_timestep = build_chunkwise_loss_observations(
            frame_loss,
            canonical_timestep,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            exclude_first_frame=self.condition_only_first_frame,
        )
        return loss, {
            "x0": clean_latent.detach(),
            "x0_pred": x0.detach(),
            "_generator_chunk_loss": chunk_loss,
            "_generator_chunk_timestep": chunk_timestep,
            **metrics,
        }

    @torch.no_grad()
    def inference(
        self, batch, image_or_video_shape, sanity_check=False, global_step=0, global_rank=0
    ):
        clean = batch["clean_latent"][:, : image_or_video_shape[1]].to(self.device, self.dtype)
        if self.inference_pipeline is None:
            self.inference_pipeline = CausalDiffusionInferencePipeline(
                self.args,
                self.device,
                generator=self.generator,
                text_encoder=self.text_encoder,
                vae=self.vae,
            )
        shape = list(clean.shape)
        if not sanity_check:
            shape[1] = (shape[1] - 1) * 2 + 1 if self.independent_first_frame else shape[1] * 2
        context_frames = 1 if self.independent_first_frame else self.num_frame_per_block
        shape[1] -= context_frames
        if self.condition_only_first_frame:
            seed = derive_condition_only_snapshot_seed(
                base_seed=int(self.args.seed), global_rank=global_rank, global_step=global_step
            )
            generator = torch.Generator(device=clean.device).manual_seed(seed)
            noise = torch.randn(shape, device=clean.device, dtype=clean.dtype, generator=generator)
        else:
            noise = torch.randn(shape, device=clean.device, dtype=clean.dtype)
        try:
            return self.inference_pipeline.inference(
                noise=noise, text_prompts=batch["prompts"], initial_latent=clean[:, :context_frames]
            )
        finally:
            self.inference_pipeline.reset_cache()

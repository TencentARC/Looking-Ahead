"""Ordinary causal TI2V and shared training-preview pipeline."""

import logging
import math

import torch
import torch.distributed as dist
from tqdm import tqdm

from src.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from src.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from src.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


def validate_condition_only_inference_inputs(
    *, enabled, initial_latent, generated_frames, num_frame_per_block, consumer_name
):
    if initial_latent is None or initial_latent.ndim != 5 or initial_latent.shape[1] < 1:
        raise ValueError(f"{consumer_name}: context must have shape [B,T,C,H,W].")
    if enabled and (initial_latent.shape[1] - 1) % num_frame_per_block:
        raise ValueError(f"{consumer_name}: condition-only context must satisfy T=1+M*F.")
    if generated_frames < 0 or generated_frames % num_frame_per_block:
        raise ValueError(f"{consumer_name}: generated tail must be chunk-aligned.")


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        self.args = args
        self.condition_only_first_frame = args.condition_only_first_frame
        self.generator = (
            generator
            if generator is not None
            else WanDiffusionWrapper(
                model_name=args.model_name,
                latent_shape=args.image_or_video_shape,
                load_diffusion_pretrained=args.load_diffusion_pretrained,
                initialization_seed=int(args.seed),
                **args.model_kwargs,
            ).to(device)
        )
        self.text_encoder = (
            text_encoder if text_encoder is not None else WanTextEncoder(args.model_name).to(device)
        )
        self.vae = vae if vae is not None else WanVAEWrapper(args.model_name).to(device)
        self.num_train_timesteps = 1000
        self.sampling_steps = args.sampling_steps
        self.sample_solver = args.sample_solver
        self.shift = args.timestep_shift
        model = self.generator.model
        self.num_heads = model.config.num_heads
        self.head_dim = model.config.dim // self.num_heads
        self.text_len = model.config.text_len
        self.num_transformer_blocks = model.config.num_layers
        self.frame_seq_length = self.generator.frame_seq_len
        self.kv_cache_pos = self.kv_cache_neg = None
        self.crossattn_cache_pos = self.crossattn_cache_neg = None
        self.num_frame_per_block = args.num_frame_per_block
        self.independent_first_frame = args.independent_first_frame
        model.num_frame_per_block = self.num_frame_per_block
        model.independent_first_frame = self.independent_first_frame
        self.local_attn_size = model.local_attn_size
        logger.info("KV inference with %d frames per block", self.num_frame_per_block)

    def _iter_crossattn_cache_lists(self):
        yield self.crossattn_cache_pos
        yield self.crossattn_cache_neg

    @staticmethod
    def _reset_crossattn_cache_list(cache_list):
        for block_cache in cache_list:
            block_cache["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        initial_latent: torch.Tensor | None = None,
        start_frame_index: int | None = 0,
        return_latents: bool = False,
        return_video: bool = True,
    ) -> torch.Tensor:
        """Generate a causal video window on GPU.

        Args:
            noise: Initial noise for generated latent frames,
                ``[B, T_noise, C, H, W]``. It excludes ``initial_latent``.
            text_prompts: List of ``B`` prompts.
            initial_latent: Required context latents
                ``[B, T_init, C, H, W]``. ``T_init=1`` is I2V; larger values
                are video extension/context.
            start_frame_index: Absolute latent-frame index of the first frame
                in this window. Used only for RoPE/KV positions; cache-local
                indices still begin at zero.
            return_latents: If true, return ``(video, output_latents)``.
            return_video: If false, skip VAE decode and return latents only.

        Returns:
            Pixel video ``[B, T_pixel, 3, H_pixel, W_pixel]`` in ``[0, 1]``;
            latents ``[B, T_init + T_noise, C, H, W]`` when requested; or
            latents alone when ``return_video=False``.

        Compatibility:
            Text plus supplied C0 is required. The generated tail must contain
            a whole number of chunks.
        """
        validate_condition_only_inference_inputs(
            enabled=bool(getattr(self, "condition_only_first_frame", False)),
            initial_latent=initial_latent,
            generated_frames=int(noise.shape[1]),
            num_frame_per_block=self.num_frame_per_block,
            consumer_name="CausalDiffusionInferencePipeline.inference",
        )
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1]
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            for cache_list in self._iter_crossattn_cache_lists():
                self._reset_crossattn_cache_list(cache_list)
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
        current_start_frame = start_frame_index
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
        if self.independent_first_frame:
            assert (num_input_frames - 1) % self.num_frame_per_block == 0
            num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
            output[:, :1].copy_(initial_latent[:, :1].detach())
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            current_start_frame += 1
            cache_start_frame += 1
        else:
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block
        for _ in range(num_input_blocks):
            current_ref_latents = initial_latent[
                :, cache_start_frame : cache_start_frame + self.num_frame_per_block
            ]
            output[:, cache_start_frame : cache_start_frame + self.num_frame_per_block].copy_(
                current_ref_latents.detach()
            )
            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            current_start_frame += self.num_frame_per_block
            cache_start_frame += self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks
        with tqdm(
            total=sum(all_num_frames), disable=dist.is_initialized() and dist.get_rank() != 0
        ) as pbar:
            pbar.set_description("Processing latent frames")
            for current_num_frames in all_num_frames:
                noisy_input = noise[
                    :,
                    cache_start_frame - num_input_frames : cache_start_frame
                    - num_input_frames
                    + current_num_frames,
                ]
                latents = noisy_input
                sample_scheduler = self._initialize_sample_scheduler(noise)
                for _, t in enumerate(tqdm(sample_scheduler.timesteps, disable=True)):
                    latent_model_input = latents
                    timestep = t * torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                    )
                    flow_pred_cond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                    )
                    flow_pred_uncond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=unconditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                    )
                    flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                        flow_pred_cond - flow_pred_uncond
                    )
                    temp_x0 = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]
                    latents = temp_x0
                logger.debug(
                    f"kv_cache local/global_end_index: {self.kv_cache_pos[0]['local_end_index'].item()}/{self.kv_cache_pos[0]['global_end_index'].item()}"
                )
                output[:, cache_start_frame : cache_start_frame + current_num_frames].copy_(
                    latents.detach()
                )
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                current_start_frame += current_num_frames
                cache_start_frame += current_num_frames
                pbar.update(current_num_frames)
                pbar.set_description("Processing latent frames")
        if return_video:
            # The next call re-creates the KV caches. Freeing them first leaves their
            # memory to the VAE decode (about 12 GiB for 5B at 1024x576).
            self.reset_cache()
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            if return_latents:
                return (video, output)
            else:
                return video
        else:
            return output

    def inference_long(
        self,
        noise: torch.Tensor,
        text_prompts: list[str],
        initial_latent: torch.Tensor | None = None,
        start_frame_index: int | None = 0,
        return_latents: bool = False,
        return_video: bool = True,
        vae_decode_chunk: int = 32,
    ) -> torch.Tensor:
        """Generate a causal window with CPU latent buffering and chunked VAE decode.

        Args are the same as :meth:`inference`, plus:
            vae_decode_chunk: Number of latent frames per VAE decode call.

        Returns:
            Same return shapes as :meth:`inference`.

        Compatibility:
            The output latent buffer lives on CPU pinned memory to reduce GPU
            peak usage. VAE decoding uses ``use_cache=True`` and therefore
            currently requires batch size ``1`` in ``WanVAEWrapper``.
        """
        validate_condition_only_inference_inputs(
            enabled=bool(getattr(self, "condition_only_first_frame", False)),
            initial_latent=initial_latent,
            generated_frames=int(noise.shape[1]),
            num_frame_per_block=self.num_frame_per_block,
            consumer_name="CausalDiffusionInferencePipeline.inference_long",
        )
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1]
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device="cpu",
            dtype=noise.dtype,
            pin_memory=torch.cuda.is_available(),
        )
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            for cache_list in self._iter_crossattn_cache_lists():
                self._reset_crossattn_cache_list(cache_list)
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
        # CPU copy/cat below must not read pending asynchronous D2H transfers.
        # Keep D2H blocking; H2D decode transfers can remain asynchronous.
        current_start_frame = start_frame_index
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
        if self.independent_first_frame:
            assert (num_input_frames - 1) % self.num_frame_per_block == 0
            num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
            output[:, :1].copy_(initial_latent[:, :1].detach().to("cpu"))
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            current_start_frame += 1
            cache_start_frame += 1
        else:
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block
        for _ in range(num_input_blocks):
            current_ref_latents = initial_latent[
                :, cache_start_frame : cache_start_frame + self.num_frame_per_block
            ]
            output[:, cache_start_frame : cache_start_frame + self.num_frame_per_block].copy_(
                current_ref_latents.detach().to("cpu")
            )
            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )
            current_start_frame += self.num_frame_per_block
            cache_start_frame += self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks
        with tqdm(
            total=sum(all_num_frames), disable=dist.is_initialized() and dist.get_rank() != 0
        ) as pbar:
            pbar.set_description("Processing latent frames")
            for current_num_frames in all_num_frames:
                noisy_input = noise[
                    :,
                    cache_start_frame - num_input_frames : cache_start_frame
                    - num_input_frames
                    + current_num_frames,
                ]
                latents = noisy_input
                sample_scheduler = self._initialize_sample_scheduler(noise)
                for _, t in enumerate(tqdm(sample_scheduler.timesteps, disable=True)):
                    latent_model_input = latents
                    timestep = t * torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                    )
                    flow_pred_cond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                    )
                    flow_pred_uncond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=unconditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                    )
                    flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                        flow_pred_cond - flow_pred_uncond
                    )
                    temp_x0 = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]
                    latents = temp_x0
                logger.debug(
                    f"kv_cache local/global_end_index: {self.kv_cache_pos[0]['local_end_index'].item()}/{self.kv_cache_pos[0]['global_end_index'].item()}"
                )
                output[:, cache_start_frame : cache_start_frame + current_num_frames].copy_(
                    latents.detach().to("cpu")
                )
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                current_start_frame += current_num_frames
                cache_start_frame += current_num_frames
                pbar.set_description("Processing latent frames")
                pbar.update(current_num_frames)
                del latents, noisy_input, latent_model_input
                del flow_pred, flow_pred_cond, flow_pred_uncond, temp_x0
                del timestep, sample_scheduler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if return_video:
            vae_device = noise.device
            chunk = max(1, int(vae_decode_chunk))
            total_lat = output.shape[1]
            pixel_chunks = []
            try:
                for s in range(0, total_lat, chunk):
                    e = min(s + chunk, total_lat)
                    lat_chunk = output[:, s:e].to(vae_device, non_blocking=True)
                    pix_chunk = self.vae.decode_to_pixel(lat_chunk, use_cache=True)
                    pixel_chunks.append(pix_chunk.detach().to("cpu"))
                    del lat_chunk, pix_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                self.vae.model.clear_cache()
            video = torch.cat(pixel_chunks, dim=1)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            if return_latents:
                return (video, output)
            else:
                return video
        else:
            return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize per-GPU self-attention KV caches.

        Args:
            batch_size: ``B``.
            dtype: Cache tensor dtype, usually matching the latent noise dtype.
            device: CUDA device.

        Cache shape:
            For every transformer block, stores ``k`` and ``v`` tensors with
            shape ``[B, local_attn_size * frame_seq_len, num_heads, head_dim]``
            plus scalar ``global_end_index`` / ``local_end_index`` counters.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        kv_cache_size = self.local_attn_size * self.frame_seq_length
        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, self.num_heads, self.head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, self.num_heads, self.head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
            kv_cache_neg.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, self.num_heads, self.head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, self.num_heads, self.head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache_pos = kv_cache_pos
        self.kv_cache_neg = kv_cache_neg

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize per-GPU text cross-attention caches.

        Shape per transformer block is
        ``[B, text_len, num_heads, head_dim]`` for both ``k`` and ``v``. The
        ``is_init`` flag lets the Wan block reuse text K/V across timesteps.
        """

        def make_cache_list():
            cache_list = []
            for _ in range(self.num_transformer_blocks):
                cache_list.append(
                    {
                        "k": torch.zeros(
                            [batch_size, self.text_len, self.num_heads, self.head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, self.text_len, self.num_heads, self.head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "is_init": False,
                    }
                )
            return cache_list

        crossattn_cache_pos = make_cache_list()
        crossattn_cache_neg = make_cache_list()
        self.crossattn_cache_pos = crossattn_cache_pos
        self.crossattn_cache_neg = crossattn_cache_neg

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1, shift=None):
        """Create the per-block sampler on ``noise.device``."""
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if isinstance(shift, bool):
            raise ValueError("Scheduler shift must be finite and positive.")
        effective_shift = self.shift if shift is None else float(shift)
        if effective_shift <= 0 or not math.isfinite(effective_shift):
            raise ValueError("Scheduler shift must be finite and positive.")
        if self.sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(shift=1)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=effective_shift
            )
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(shift=1)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, effective_shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=noise.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler

    def reset_cache(self):
        """Drop KV/cross-attention caches and release cached CUDA blocks."""
        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        torch.cuda.empty_cache()

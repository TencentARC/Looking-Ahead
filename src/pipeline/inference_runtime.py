"""Text with image/video-prefix sampling and distributed existence voting."""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torchcodec.encoders import VideoEncoder
from torchvision import transforms

from src.dataset.image_preprocessing import ResizeCenterCrop
from src.dataset.single_image import SingleImageDataset, read_prompt_file
from src.dataset.single_video import SingleVideoDataset
from src.dataset.text import TextImagePairDataset
from src.pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from src.utils.inference_checkpoint import load_generator_checkpoint
from src.utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from src.utils.train_dataset import build_train_dataset
from src.utils.video_prefix import output_geometry, select_prefix
from src.utils.wan_model_family import get_wan_model_family

logger = logging.getLogger(__name__)


def output_exists_on_all_ranks(path, device):
    exists = Path(path).exists()
    if dist.is_initialized() and dist.get_world_size() > 1:
        flag = torch.tensor([int(exists)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        exists = bool(flag.item())
    return exists


@torch.no_grad()
def run_inference(config):
    single_dataset = None
    family = get_wan_model_family(config.model_name)
    if config.get("image_path") is not None:
        h, w = config.image_or_video_shape[-2:]
        # Read one prompt/image and report changes before loading GPU models.
        prompt = (
            read_prompt_file(config.prompt_file)
            if config.get("prompt_file") is not None
            else config.prompt
        )
        single_dataset = SingleImageDataset(
            config.image_path,
            prompt,
            height=h * family.vae_spatial_stride,
            width=w * family.vae_spatial_stride,
        )
    v2v = config.get("v2v", False)
    # Config geometry was validated by the resolver; only sample bounds remain below.
    context_count = config.v2v_context_latents if v2v else 1
    automatic_context = v2v and context_count is None
    condition_frames = (
        None if automatic_context else 1 + family.vae_temporal_stride * (context_count - 1)
    )
    if config.get("video_path") is not None:
        h, w = config.image_or_video_shape[-2:]
        prompt = (
            read_prompt_file(config.prompt_file)
            if config.prompt_file is not None
            else config.prompt
        )
        single_dataset = SingleVideoDataset(
            config.video_path,
            prompt,
            height=h * family.vae_spatial_stride,
            width=w * family.vae_spatial_stride,
            fps=config.get("fps", family.default_fps),
            condition_frames=condition_frames,
            auto_context_config=config if automatic_context else None,
        )
        if automatic_context:
            context_count = single_dataset.context_latents
            condition_frames = single_dataset.report["condition_frames"]
        single_dataset.report["context_latents"] = context_count
    distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # Construct on CPU, assign strict ordinary weights, then transfer once.
    pipeline = CausalDiffusionInferencePipeline(config, device="cpu")
    loaded_checkpoint = load_generator_checkpoint(pipeline.generator, config)
    pipeline.to(dtype=torch.bfloat16)
    free_memory = get_cuda_free_memory_gb(device)
    if free_memory < 40:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    pixel_frames, latent_frames = output_geometry(config)
    height, width = pipeline.generator.frame_shape[-2:]
    _, channels, latent_height, latent_width = pipeline.generator.latent_shape
    limit = int(config.test_sample_limit)
    per_rank_limit = -1 if limit < 0 else (limit + world_size - 1) // world_size
    if config.use_train_data:
        dataset = build_train_dataset(config, max_samples=per_rank_limit)
    else:
        if dist.is_initialized():
            raise ValueError("External-image TI2V requires direct non-distributed Python.")
        transform = transforms.Compose(
            [
                ResizeCenterCrop(height, width),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        if single_dataset is not None:
            dataset = single_dataset
            per_rank_limit = 0 if limit == 0 else 1
        else:
            dataset = TextImagePairDataset(
                config.data_path, transform=transform, eval_first_n=limit
            )
            per_rank_limit = len(dataset)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
    output = Path(config.output_path)
    output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    completed = 0
    for index, batch in enumerate(loader):
        if per_rank_limit >= 0 and completed >= per_rank_limit:
            break
        prompt = batch["prompts"][0]
        if config.use_train_data and world_size > 1:
            stem = f"r{rank:02d}-{index:05d}"
        else:
            global_index = int(batch["idx"][0]) if "idx" in batch else index
            stem = f"{global_index:05d}"
        sample_stem = f"{stem}-{config.seed}"
        if v2v:
            sample_stem += f"-v2v-k={context_count}"
        video_path = output / f"{sample_stem}.mp4"
        if output_exists_on_all_ranks(video_path, device):
            completed += 1
            logger.info("Skipped existing video %s (%d/%s)", video_path, completed, per_rank_limit)
            continue
        if single_dataset is not None:
            report_path = output / f"{sample_stem}.input.json"
            report_path.write_text(
                json.dumps(single_dataset.report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("Image preprocessing report: %s", report_path)
        image_for_save = None
        if config.use_train_data:
            clean = batch["clean_latent"]
            if clean.ndim != 5 or tuple(clean.shape[2:]) != (channels, latent_height, latent_width):
                raise ValueError(
                    f"TI2V latent input incompatible with model geometry: {tuple(clean.shape)}"
                )
            prefix = select_prefix(clean, context_count) if v2v else clean[:, :1].contiguous()
            initial = prefix.to(device=device, dtype=torch.bfloat16)
        else:
            image = (batch["video"] if v2v else batch["image"].unsqueeze(2)).to(
                device=device, dtype=torch.bfloat16
            )
            initial = pipeline.vae.encode_to_latent(image).to(dtype=torch.bfloat16)
            if initial.shape[1] != context_count:
                raise ValueError("video_path: encoded prefix length differs from requested context")
            image_for_save = image[0, :, 0].float().cpu()
        noise_generator = torch.Generator(device=device).manual_seed(int(config.seed) % (2**63))
        noise = torch.randn(
            (1, latent_frames - context_count, channels, latent_height, latent_width),
            device=device,
            dtype=torch.bfloat16,
            generator=noise_generator,
        )
        logger.info(
            "Generating %s; context=%s, total latent frames=%d",
            video_path,
            tuple(initial.shape),
            latent_frames,
        )
        long_path = latent_frames >= config.long_inference_threshold
        logger.info(
            "Sampling path: %s; context_latents=%d; generated_latents=%d",
            "long" if long_path else "short",
            context_count,
            latent_frames - context_count,
        )
        if long_path:
            video = pipeline.inference_long(
                noise, [prompt], initial_latent=initial, vae_decode_chunk=config.vae_decode_chunk
            )
        else:
            video = pipeline.inference(noise, [prompt], initial_latent=initial)
        pixels = (video[0, :pixel_frames].clamp(0, 1) * 255).byte().cpu()
        VideoEncoder(pixels, frame_rate=config.get("fps", pipeline.generator.default_fps)).to_file(
            str(video_path)
        )
        if v2v:
            fps = config.get("fps", pipeline.generator.default_fps)
            for folder, interval in (
                ("conditions", slice(0, condition_frames)),
                ("continuations", slice(condition_frames, pixel_frames)),
            ):
                destination = output / folder / f"{sample_stem}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                VideoEncoder(pixels[interval].contiguous(), frame_rate=fps).to_file(
                    str(destination)
                )
            manifest_dir = output / "manifests"
            manifest_dir.mkdir(exist_ok=True)
            manifest = {
                "mode": "v2v",
                "context_latents": context_count,
                "requested_context_latents": config.v2v_context_latents,
                "context_selection": (
                    single_dataset.report["context_selection"]
                    if single_dataset is not None
                    else "latent_prefix"
                ),
                "condition_frames": condition_frames,
                "output_frames": pixel_frames,
                "continuation_frames": pixel_frames - condition_frames,
                "total_latents": latent_frames,
                "fps": fps,
                "seed": config.seed,
                "model_name": config.model_name,
                "checkpoint_path": config.checkpoint_path,
                "loaded_checkpoint_path": loaded_checkpoint,
                "use_ema": config.use_ema,
                "source_id": config.get("inference_train_data_source_id"),
                "video_path": config.get("video_path"),
                "sample_metadata": {
                    key: value[0] for key, value in batch.get("metadata", {}).items()
                },
                "prompt": prompt,
                "condition_interval": [0, condition_frames],
                "continuation_interval": [condition_frames, pixel_frames],
            }
            (manifest_dir / f"{sample_stem}.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        (output / f"{stem}.txt").write_text(prompt + "\n", encoding="utf-8")
        if image_for_save is None:
            image_for_save = (
                pipeline.vae.decode_to_pixel(initial[:, :1], use_cache=False)[0, 0].float().cpu()
            )
            pipeline.vae.model.clear_cache()
        png = ((image_for_save + 1) / 2).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(png).save(output / f"{sample_stem}.png")
        pipeline.vae.model.clear_cache()
        completed += 1
        logger.info("Completed %s (%d/%s)", video_path, completed, per_rank_limit)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

"""Video context sampled at model FPS: an explicit prefix or an automatic tail."""

import logging
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import functional as TF

from src.dataset.image_preprocessing import resize_center_crop
from src.utils.video_prefix import automatic_prefix_geometry

logger = logging.getLogger(__name__)


class SingleVideoDataset(Dataset):
    """Decode only the selected context interval for causal VAE encoding."""

    def __init__(
        self,
        video_path,
        prompt,
        *,
        height,
        width,
        fps,
        condition_frames,
        auto_context_config=None,
    ):
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video_path: video file not found: {path}")
        decoder = VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=1)
        start = decoder.metadata.begin_stream_seconds
        automatic = condition_frames is None
        self.context_latents = None
        if automatic:
            if auto_context_config is None:
                raise ValueError("Automatic video context requires model/output geometry")
            # Use the actual final frame PTS, including variable-rate videos. Anchoring
            # to stream end minus 1/FPS can silently omit the last source frame.
            last_pts = decoder.get_frame_at(len(decoder) - 1).pts_seconds
            available = math.floor((last_pts - start) * fps + 1e-7) + 1
            self.context_latents, condition_frames = automatic_prefix_geometry(
                auto_context_config, available
            )
            # Step back from the last frame in frame units, then divide once. With
            # last_pts - k / fps, the rounding of k / fps can leave a time just before
            # a frame's PTS, and get_frames_played_at then repeats the previous frame.
            times = [
                max(start, (last_pts * fps - (condition_frames - 1 - index)) / fps)
                for index in range(condition_frames)
            ]
        else:
            times = [start + index / fps for index in range(condition_frames)]
            if times[-1] >= decoder.metadata.end_stream_seconds:
                raise ValueError(
                    "video_path: video is too short for the requested context at model FPS"
                )
        batch = decoder.get_frames_played_at(times)
        frames = batch.data
        images = []
        geometry = None
        for frame in frames:
            image, geometry = resize_center_crop(TF.to_pil_image(frame), height, width)
            images.append(TF.normalize(TF.to_tensor(image), [0.5], [0.5]))
        self.video = torch.stack(images, dim=1)  # [C,T,H,W], selected context only.
        self.prompt = prompt
        self.report = {
            "source_path": str(path),
            "source_fps": decoder.metadata.average_fps,
            "source_size": {"width": frames.shape[-1], "height": frames.shape[-2]},
            "input_size": {"width": width, "height": height},
            "fps": fps,
            "condition_frames": condition_frames,
            "context_selection": "auto_tail" if automatic else "explicit_prefix",
            "sample_times_seconds": times,
            "source_pts_seconds": batch.pts_seconds.tolist(),
            "preprocessing": (
                "sample trailing frames at model FPS; RGB resize and center crop"
                if automatic
                else "sample leading frames at model FPS; RGB resize and center crop"
            ),
            **geometry,
        }
        if automatic:
            self.report.update(
                context_latents=self.context_latents,
                available_frames_at_target_fps=available,
                source_last_frame_pts_seconds=last_pts,
            )
        logger.info("Video context preprocessing: %s", self.report)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {"video": self.video, "prompts": self.prompt, "idx": 0}

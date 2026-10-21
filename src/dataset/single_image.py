"""One image and prompt for TI2V, with an explicit preprocessing report."""

import logging
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.dataset.image_preprocessing import resize_center_crop

logger = logging.getLogger(__name__)

EXIF_OPERATIONS = {
    2: "horizontal flip",
    3: "180-degree rotation",
    4: "vertical flip",
    5: "transpose across the main diagonal",
    6: "90-degree clockwise rotation",
    7: "transpose across the opposite diagonal",
    8: "90-degree counterclockwise rotation",
}


def read_prompt_file(filename):
    """Read a whole UTF-8 text file as one prompt; keep interior line breaks."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"prompt_file: text file not found: {path}")
    try:
        prompt = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"prompt_file: cannot read {path} as UTF-8 text: {error}") from error
    if not prompt:
        raise ValueError(f"prompt_file: file contains no prompt text: {path}")
    logger.info("Single-image prompt file: %s (UTF-8)", path)
    return prompt


class SingleImageDataset(Dataset):
    """Load and prepare one supplied C0 without a metadata JSON or layout convention."""

    def __init__(self, image_path, prompt, *, height, width):
        self.path = Path(image_path)
        self.prompt = prompt
        if not self.path.is_file():
            raise FileNotFoundError(f"image_path: image file not found: {self.path}")
        operations = []
        try:
            with Image.open(self.path) as source:
                original_size, original_mode = source.size, source.mode
                frames = getattr(source, "n_frames", 1)
                if frames > 1:
                    operations.append(f"Selected first frame (0) from {frames} frames.")
                orientation = source.getexif().get(274, 1)
                image = ImageOps.exif_transpose(source)
                if orientation in EXIF_OPERATIONS:
                    operations.append(
                        f"Applied EXIF orientation {orientation}: {EXIF_OPERATIONS[orientation]}."
                    )
                oriented_size = image.size
                has_alpha = "A" in image.getbands() or "transparency" in image.info
                if image.mode != "RGB":
                    detail = (
                        " Alpha channel discarded; no background compositing." if has_alpha else ""
                    )
                    operations.append(f"Converted {image.mode} to RGB.{detail}")
                    image = image.convert("RGB")
                elif has_alpha:
                    operations.append(
                        "Discarded RGB color-key transparency; RGB pixels retained, no background compositing."
                    )
                # Copy/decode while the source stream is still open.
                image.load()
        except (OSError, ValueError) as error:
            raise ValueError(f"image_path: cannot decode {self.path}: {error}") from error
        aspect_changed = oriented_size[0] * height != width * oriented_size[1]
        image, geometry = resize_center_crop(image, height, width)
        resized = geometry["resized_size"]
        if oriented_size != (resized["width"], resized["height"]):
            operations.append(
                f"Resized {oriented_size[0]}x{oriented_size[1]} to "
                f"{resized['width']}x{resized['height']} preserving aspect ratio "
                "using bilinear interpolation with antialiasing (rounded to whole pixels)."
            )
        if geometry["cropped"]:
            operations.append(
                f"Center-cropped box {geometry['crop_box']} from the resized image "
                f"to {width}x{height}; edge content discarded, no stretching or padding."
            )
        if not operations:
            operations.append("No orientation, RGB conversion, resize, or crop needed.")
        self.image = TF.normalize(TF.to_tensor(image), [0.5], [0.5])
        operations.append(
            "Converted to float32 CHW tensor; pixel range [0,255] normalized to [-1,1]."
        )
        self.report = {
            "source_path": str(self.path.resolve()),
            "source_mode": original_mode,
            "source_size": {"width": original_size[0], "height": original_size[1]},
            "exif_orientation": orientation,
            "oriented_size": {"width": oriented_size[0], "height": oriented_size[1]},
            "input_mode": "RGB",
            "input_size": {"width": width, "height": height},
            "aspect_ratio_changed": aspect_changed,
            "preprocessing": "resize_center_crop",
            **geometry,
            "operations": operations,
        }
        logger.info(
            "Single-image preprocessing: %s; %s %dx%d -> RGB %dx%d",
            self.path,
            original_mode,
            *original_size,
            width,
            height,
        )
        for operation in operations:
            logger.info("  %s", operation)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {"image": self.image, "prompts": self.prompt, "idx": 0}

"""Shared aspect-preserving resize and center crop for external images."""

from torchvision.transforms import functional as TF


def resize_center_crop(image, height, width):
    """Scale to cover the target, then crop its center; report pixel geometry."""
    source_width, source_height = image.size
    if source_width * height >= width * source_height:
        resized_height = height
        resized_width = (source_width * height + source_height - 1) // source_height
    else:
        resized_width = width
        resized_height = (source_height * width + source_width - 1) // source_width
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    image = TF.resize(image, [resized_height, resized_width], antialias=True)
    image = TF.crop(image, top, left, height, width)
    return image, {
        "resized_size": {"width": resized_width, "height": resized_height},
        "crop_box": [left, top, left + width, top + height],
        "cropped": resized_width != width or resized_height != height,
    }


class ResizeCenterCrop:
    """Callable transform using the same preprocessing as single-image input."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        return resize_center_crop(image, self.height, self.width)[0]

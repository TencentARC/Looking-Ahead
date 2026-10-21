import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TextImagePairDataset(Dataset):
    """Image-prompt dataset for I2V inference.

    ``data_dir`` contains one ``target_crop_info_*.json`` metadata file and an
    aspect-ratio subdirectory holding images. Returned images are optionally
    transformed by the caller into tensors before VAE encoding.
    """

    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob("target_crop_info_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split("_")[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Load one I2V conditioning image and prompt metadata.

        Returns:
            Dict with ``image`` (PIL image or transformed tensor), ``prompts``
            string, ``target_bbox`` ``[x1, y1, x2, y2]``, ``target_ratio``,
            ``type``, ``origin_size`` ``(width, height)`` and stable ``idx``.
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item["file_name"]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "prompts": item["caption"],
            "target_bbox": item["target_crop"]["target_bbox"],
            "target_ratio": item["target_crop"]["target_ratio"],
            "type": item["type"],
            "origin_size": (item["origin_width"], item["origin_height"]),
            "idx": idx,
        }

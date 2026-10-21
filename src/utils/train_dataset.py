"""Finite source selection for TI2V and deterministic training/preview loaders."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.dataset.multi_source import (
    build_multi_source_tensor_webdataset,
    validate_multi_source_selection,
)
from src.dataset.tensor import PytorchTensorWebDataset
from src.dataset.utils import seed_webdataset_worker


@dataclass(frozen=True)
class ResolvedTrainDataSource:
    source_id: str
    data_paths: tuple[str, ...]
    annotation_path: str

    @property
    def primary_data_path(self):
        return self.data_paths[0]


def resolve_train_data_selection(config):
    sources = validate_multi_source_selection(
        config.data_sources,
        legacy_fields={"data_path": config.data_path, "annotation_path": config.annotation_path},
    )
    source_id = config.inference_train_data_source_id
    if source_id is None:
        if len(sources) != 1:
            raise ValueError("Multiple data_sources require --train_data_source_id <id>.")
        selected = sources[0]
    else:
        selected = next((source for source in sources if source.source_id == source_id), None)
        if selected is None:
            raise ValueError(f"Unknown inference train-data source ID {source_id!r}.")
    return ResolvedTrainDataSource(
        selected.source_id, (selected.data_path,), selected.annotation_path
    )


def build_train_dataset(config, max_samples=-1, *, selection=None):
    """Build a finite inference stream; reported length does not limit iteration."""
    selected = selection or resolve_train_data_selection(config)
    rank = dist.get_rank() if dist.is_initialized() else 0
    return PytorchTensorWebDataset(
        data_path=selected.primary_data_path,
        annotation_path=selected.annotation_path,
        shuffle=0,
        shardshuffle=False,
        resampled=False,
        length=max_samples if max_samples >= 0 else None,
        seed=int(config.seed),
        global_rank=rank,
    )


def build_training_dataset(config, *, is_train):
    validate_multi_source_selection(
        config.data_sources,
        legacy_fields={"data_path": config.data_path, "annotation_path": config.annotation_path},
    )
    if int(config.seed) == 0:
        raise ValueError("Deterministic training requires a nonzero seed.")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    length = 100000000 if is_train else world_size * int(config.val_batch_size)
    return build_multi_source_tensor_webdataset(
        config.data_sources,
        is_train=is_train,
        seed=int(config.seed),
        global_rank=rank,
        length=length,
        target_frame_num=int(config.image_or_video_shape[1]),
        sample_shuffle=int(config.wds_sample_shuffle),
        sample_shuffle_initial=int(config.wds_sample_shuffle_initial),
    )


def build_training_dataloader(config, *, is_train):
    dataset = build_training_dataset(config, is_train=is_train)
    workers = int(config.num_workers if is_train else config.val_num_workers)
    generator = torch.Generator(device="cpu").manual_seed(dataset.dataloader_seed)
    kwargs = {"prefetch_factor": 2} if workers > 0 else {}
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size if is_train else config.val_batch_size),
        num_workers=workers,
        drop_last=is_train,
        pin_memory=False,
        persistent_workers=False,
        in_order=True,
        generator=generator,
        worker_init_fn=seed_webdataset_worker,
        **kwargs,
    )

"""Weighted sample-level mixing for multiple latent WebDataset sources."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import IterableDataset

from .tensor import PytorchTensorWebDataset
from .utils import (
    DEFAULT_WDS_SAMPLE_SHUFFLE,
    DEFAULT_WDS_SAMPLE_SHUFFLE_INITIAL,
    derive_webdataset_seed,
)

logger = logging.getLogger(__name__)

_WDS_SOURCE_CHOICE_STREAM_ID = 0x5744_5305
_WDS_SOURCE_NAMESPACE_STRIDE = 0x100


@dataclass(frozen=True)
class WebDatasetSource:
    """Validated configuration for one independently indexed WebDataset."""

    source_id: str
    data_path: str
    annotation_path: str
    weight: float


def normalize_webdataset_sources(raw_sources: Any) -> list[WebDatasetSource]:
    """Validate the public ``data_sources`` config and normalize its weights."""
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
        raise TypeError("data_sources must be a non-empty list of mappings.")
    if not raw_sources:
        raise ValueError("data_sources must contain at least one WebDataset source.")

    sources: list[WebDatasetSource] = []
    seen_ids: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise TypeError(f"data_sources[{index}] must be a mapping.")

        source_id = raw_source.get("id", f"source_{index}")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"data_sources[{index}].id must be a non-empty string.")
        source_id = source_id.strip()
        if source_id in seen_ids:
            raise ValueError(f"Duplicate data source id: {source_id!r}.")
        seen_ids.add(source_id)

        data_path = raw_source.get("data_path")
        if not isinstance(data_path, str) or not data_path:
            raise ValueError(f"data_sources[{index}].data_path must be a non-empty string.")
        if not data_path.rstrip("/\\").lower().endswith("shards"):
            raise ValueError(
                f"data_sources[{index}].data_path must name a directory ending "
                f"in 'shards', got {data_path!r}."
            )

        annotation_path = raw_source.get("annotation_path")
        if not isinstance(annotation_path, str) or not annotation_path:
            raise ValueError(f"data_sources[{index}].annotation_path must be a non-empty string.")

        weight = raw_source.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"data_sources[{index}].weight must be a positive number.")
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"data_sources[{index}].weight must be finite and greater than zero.")

        sources.append(
            WebDatasetSource(
                source_id=source_id,
                data_path=data_path,
                annotation_path=annotation_path,
                weight=weight,
            )
        )

    weight_sum = sum(source.weight for source in sources)
    return [
        WebDatasetSource(
            source_id=source.source_id,
            data_path=source.data_path,
            annotation_path=source.annotation_path,
            weight=source.weight / weight_sum,
        )
        for source in sources
    ]


def validate_multi_source_selection(
    raw_sources: Any,
    *,
    legacy_fields: Mapping[str, Any],
) -> list[WebDatasetSource]:
    """Validate opt-in selection and reject ambiguous legacy fields."""
    conflicting_fields = [name for name, value in legacy_fields.items() if value is not None]
    if conflicting_fields:
        joined = ", ".join(conflicting_fields)
        raise ValueError(
            f"data_sources cannot be combined with non-null legacy field(s): {joined}."
        )
    return normalize_webdataset_sources(raw_sources)


class WeightedMultiSourceDataset(IterableDataset):
    """Mix independent iterable datasets at sample granularity.

    Training sources are expected to be infinite resampled streams. For finite
    validation sources, an exhausted source is removed from the choice set and
    iteration ends after every source has been consumed once.
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        sources: Sequence[WebDatasetSource],
        *,
        seed: int,
        global_rank: int,
        resampled: bool,
        length: int | None = None,
    ) -> None:
        super().__init__()
        if len(datasets) != len(sources) or not datasets:
            raise ValueError("datasets and sources must have the same non-zero length.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Multi-source seed must be a non-boolean integer.")
        if isinstance(global_rank, bool) or not isinstance(global_rank, int):
            raise TypeError("Multi-source global_rank must be a non-boolean integer.")
        if global_rank < 0:
            raise ValueError("Multi-source global_rank must be non-negative.")

        self.datasets = list(datasets)
        self.sources = list(sources)
        self._weights = [source.weight for source in sources]
        self._deterministic = True
        self._seed = seed
        self._global_rank = global_rank
        self._resampled = resampled
        if length is None:
            self._length = sum(len(dataset) for dataset in self.datasets)
        else:
            self._length = min(
                int(length),
                sum(len(dataset) for dataset in self.datasets),
            )
        logger.info(
            "Initialized %s with sources=%s, resampled=%s, deterministic=%s, reported_length=%d",
            type(self).__name__,
            [
                {
                    "id": source.source_id,
                    "weight": source.weight,
                    "samples": len(dataset),
                }
                for source, dataset in zip(
                    self.sources,
                    self.datasets,
                    strict=True,
                )
            ],
            self._resampled,
            self._deterministic,
            self._length,
        )

    def _worker_address(self) -> tuple[int, int]:
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            return 0, 1
        return worker.id, worker.num_workers

    def _choice_seed(self) -> int:
        worker_id, num_workers = self._worker_address()
        return derive_webdataset_seed(
            self._seed,
            self._global_rank,
            worker_id,
            num_workers,
            _WDS_SOURCE_CHOICE_STREAM_ID,
        )

    @property
    def dataloader_seed(self) -> int | None:
        """Return an isolated DataLoader seed for deterministic mixing."""
        return derive_webdataset_seed(
            self._seed,
            self._global_rank,
            worker_id=0,
            num_workers=1,
            stream_id=_WDS_SOURCE_CHOICE_STREAM_ID + 1,
        )

    def _make_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._choice_seed())
        return generator

    def _choose_source(
        self,
        generator: torch.Generator,
        active: Sequence[bool],
    ) -> int:
        total_weight = sum(
            weight for weight, is_active in zip(self._weights, active, strict=True) if is_active
        )
        if total_weight <= 0:
            raise RuntimeError("No active WebDataset source remains.")
        target = torch.rand((), generator=generator, dtype=torch.float64).item() * total_weight
        cumulative = 0.0
        last_active = -1
        for index, (weight, is_active) in enumerate(zip(self._weights, active, strict=True)):
            if not is_active:
                continue
            last_active = index
            cumulative += weight
            if target < cumulative:
                return index
        return last_active

    def _tag_sample(self, sample: dict[str, Any], source_index: int) -> dict[str, Any]:
        if not isinstance(sample, dict):
            raise TypeError(
                f"WebDataset source {self.sources[source_index].source_id!r} "
                f"yielded {type(sample).__name__}, expected dict."
            )
        tagged = dict(sample)
        metadata = dict(tagged.get("metadata", {}))
        metadata.update(
            {
                "source_id": self.sources[source_index].source_id,
                "source_index": source_index,
                "source_weight": self.sources[source_index].weight,
            }
        )
        tagged["metadata"] = metadata
        return tagged

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators: list[Iterator[dict[str, Any]]] = [iter(dataset) for dataset in self.datasets]
        active = [True] * len(iterators)
        generator = self._make_generator()

        while any(active):
            source_index = self._choose_source(generator, active)
            try:
                sample = next(iterators[source_index])
            except StopIteration:
                if self._resampled:
                    source_id = self.sources[source_index].source_id
                    raise RuntimeError(
                        f"Resampled WebDataset source {source_id!r} exhausted unexpectedly."
                    ) from None
                active[source_index] = False
                continue
            yield self._tag_sample(sample, source_index)

    def __len__(self) -> int:
        return self._length


def build_multi_source_tensor_webdataset(
    raw_sources: Any,
    *,
    is_train: bool,
    seed: int,
    global_rank: int,
    length: int | None,
    target_frame_num: int | None = None,
    sample_shuffle: int = DEFAULT_WDS_SAMPLE_SHUFFLE,
    sample_shuffle_initial: int = DEFAULT_WDS_SAMPLE_SHUFFLE_INITIAL,
) -> WeightedMultiSourceDataset:
    """Build one latent WebDataset pipeline per source plus a weighted mixer."""
    sources = normalize_webdataset_sources(raw_sources)
    datasets: list[PytorchTensorWebDataset] = []
    for source_index, source in enumerate(sources):
        datasets.append(
            PytorchTensorWebDataset(
                data_path=source.data_path,
                annotation_path=source.annotation_path,
                target_frame_num=target_frame_num,
                shuffle=sample_shuffle if is_train else 0,
                shuffle_initial=sample_shuffle_initial,
                shardshuffle=is_train,
                resampled=is_train,
                seed=seed,
                global_rank=global_rank,
                seed_namespace=(source_index + 1) * _WDS_SOURCE_NAMESPACE_STRIDE,
            )
        )

    return WeightedMultiSourceDataset(
        datasets,
        sources,
        seed=seed,
        global_rank=global_rank,
        resampled=is_train,
        length=length,
    )

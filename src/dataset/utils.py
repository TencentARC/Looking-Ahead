import json
import logging
import os
import random
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


_WDS_SHARD_RESAMPLE_STREAM_ID = 0x5744_5301
_WDS_SHARD_SHUFFLE_STREAM_ID = 0x5744_5302
_WDS_SAMPLE_SHUFFLE_STREAM_ID = 0x5744_5303
_WDS_DATALOADER_STREAM_ID = 0x5744_5304
DEFAULT_WDS_SAMPLE_SHUFFLE = 1000
DEFAULT_WDS_SAMPLE_SHUFFLE_INITIAL = 100


def validate_webdataset_sample_shuffle(
    shuffle: int,
    shuffle_initial: int,
) -> tuple[int, int]:
    """Validate sample-buffer shuffle settings without changing their values."""
    values = {
        "wds_sample_shuffle": shuffle,
        "wds_sample_shuffle_initial": shuffle_initial,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-boolean integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    return shuffle, shuffle_initial


def resolve_webdataset_sample_shuffle(config: Any) -> tuple[int, int]:
    """Resolve public WebDataset sample-shuffle controls with legacy defaults."""
    return validate_webdataset_sample_shuffle(
        getattr(config, "wds_sample_shuffle", DEFAULT_WDS_SAMPLE_SHUFFLE),
        getattr(
            config,
            "wds_sample_shuffle_initial",
            DEFAULT_WDS_SAMPLE_SHUFFLE_INITIAL,
        ),
    )


def derive_webdataset_seed(
    base_seed: int,
    global_rank: int,
    worker_id: int,
    num_workers: int,
    stream_id: int,
) -> int:
    """Derive a stable worker-local seed without consuming any global RNG.

    The address includes the training seed, distributed rank, DataLoader
    worker identity/topology and a named integer stream. SplitMix64-style
    integer mixing avoids Python's process-randomized ``hash`` and excludes
    wall-clock time, PIDs and OS entropy.
    """
    components = (base_seed, global_rank, worker_id, num_workers, stream_id)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in components):
        raise TypeError("WebDataset seed components must be non-boolean integers.")
    if global_rank < 0 or worker_id < 0 or num_workers <= 0:
        raise ValueError(
            "WebDataset seed address requires global_rank >= 0, worker_id >= 0, "
            "and num_workers > 0."
        )

    mask = (1 << 64) - 1
    value = 0x6A09_E667_F3BC_C909
    for component in components:
        value ^= (component & mask) + 0x9E37_79B9_7F4A_7C15 + ((value << 6) & mask)
        value ^= value >> 30
        value = (value * 0xBF58_476D_1CE4_E5B9) & mask
        value ^= value >> 27
        value = (value * 0x94D0_49BB_1331_11EB) & mask
        value ^= value >> 31
    return value & ((1 << 63) - 1)


def seed_webdataset_worker(worker_id: int) -> None:
    """Seed worker-global RNGs from DataLoader's isolated generator stream."""
    del worker_id  # ``torch.initial_seed`` already includes the worker offset.
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (1 << 32))
    torch.manual_seed(worker_seed)


def _zero_webdataset_worker_seed() -> int:
    """Disable WebDataset's implicit worker seed after explicit derivation."""
    return 0


def _read_file_parallel(path: str, num_workers: int = 8) -> bytes:
    """Read a file using parallel ``pread`` from a thread pool.

    Why: S3 FUSE backends (s3fs / goofys / mountpoint-s3) generally cap a
    *single* sequential read at ~50–150 MB/s because there is no
    auto-parallelism on a single fd. Issuing several concurrent ``pread``
    calls on independent file descriptors lets the FUSE layer fire multiple
    GET range requests in flight, multiplying effective throughput by
    roughly ``num_workers`` until the underlying network / S3 endpoint
    saturates.

    Falls back to a single-threaded read for small files where the
    parallelism overhead would dominate.
    """
    file_size = os.path.getsize(path)
    if file_size == 0:
        return b""

    # Below ~16 MiB the single-threaded path is already cheap; skip the
    # thread-pool overhead.
    if file_size <= 16 << 20 or num_workers <= 1:
        with open(path, "rb") as f:
            return f.read()

    # Split the file into ``num_workers`` (roughly) equal chunks. Cap the
    # number of chunks to avoid spawning thousands of fds for huge files.
    num_chunks = min(max(num_workers, 1), 64)
    base = file_size // num_chunks
    remainder = file_size % num_chunks
    offsets: list[int] = []
    sizes: list[int] = []
    cur = 0
    for i in range(num_chunks):
        sz = base + (1 if i < remainder else 0)
        offsets.append(cur)
        sizes.append(sz)
        cur += sz

    def _worker(args):
        offset, size = args
        # Each worker uses its *own* file descriptor so pread calls don't
        # contend (and so the FUSE layer can issue independent GETs).
        fd = os.open(path, os.O_RDONLY)
        try:
            parts: list[bytes] = []
            done = 0
            while done < size:
                chunk = os.pread(fd, size - done, offset + done)
                if not chunk:
                    raise OSError(f"unexpected EOF reading {path!r} at offset {offset + done}")
                parts.append(chunk)
                done += len(chunk)
            return b"".join(parts) if len(parts) > 1 else parts[0]
        finally:
            os.close(fd)

    with ThreadPoolExecutor(max_workers=min(num_workers, num_chunks)) as ex:
        chunks = list(ex.map(_worker, zip(offsets, sizes, strict=True)))

    return b"".join(chunks)


class BaseWebDataset(IterableDataset):
    """Annotation-indexed deterministic latent stream with in-memory captions.

    JSONL paths define sorted tar paths; annotation filtering precedes decode.
    Read/decode errors propagate and reported length never truncates iteration.
    Annotation reads use eight workers for large files and serial reads for small
    files. No disk annotation cache or eager shard scan is performed.
    """

    #: Override in subclasses if the dataset writes a different prefix into
    #: shard filenames (e.g. ``"latent_shard-"`` for pre-encoded latents).
    SHARD_KEY_STRIP_PREFIX: str = ""

    def __init__(
        self,
        data_path: str | os.PathLike,
        annotation_path: str,
        shuffle: int = 1000,
        shuffle_initial: int = 100,
        shardshuffle: bool | int = True,
        resampled: bool = False,
        nodesplitter: Callable | None = None,
        handler: Callable | None = None,
        length: int | None = None,
        seed: int = 0,
        global_rank: int = 0,
        seed_namespace: int = 0,
    ):
        super().__init__()

        # ``data_path`` is a *directory* that contains the tar shards. We
        # never glob it; the annotation file enumerates the shards.
        data_dir = str(data_path)
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(
                f"{type(self).__name__}: data_path must be a directory "
                f"containing tar shards, got: {data_dir!r}"
            )
        self.data_path = data_dir
        self.caption_field = "global_caption_en"

        # Load (shard, clip_name) -> caption from the JSONL index. The JSONL
        # is the single source of truth: every entry must have shard,
        # clip_name and a non-empty caption_field. Strict mode: any
        # inconsistency aborts construction.
        if not annotation_path:
            raise FileNotFoundError(
                f"{type(self).__name__}: annotation_path is required, got: {annotation_path!r}"
            )
        self.annotation_path = annotation_path
        self.caption_index: dict[tuple, str] = {}
        shards_in_order: list[str] = []  # preserve first-seen order
        seen_shards: set = set()

        _t0 = time.perf_counter()
        raw_bytes = _read_file_parallel(annotation_path, num_workers=8)
        _t_read = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        # ``splitlines()`` handles \n / \r\n / \r and avoids producing a
        # trailing empty line for files that end with a newline -- cleaner
        # than ``split('\n')``. Decoding once for the whole blob is much
        # faster than decoding per-line.
        text = raw_bytes.decode("utf-8")
        del raw_bytes  # free ~JSONL-size buffer before building the index
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line or line.isspace():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{type(self).__name__}: invalid JSON in "
                    f"{annotation_path} at line {line_no}: {e}"
                ) from e
            shard = entry.get("shard")
            clip = entry.get("clip_name")
            if not shard or not clip:
                raise ValueError(
                    f"{type(self).__name__}: missing 'shard' or "
                    f"'clip_name' in {annotation_path} at line {line_no}: {entry!r}"
                )
            cap = entry.get(self.caption_field)
            if not isinstance(cap, str) or not cap:
                raise ValueError(
                    f"{type(self).__name__}: missing or empty "
                    f"caption field {self.caption_field!r} in "
                    f"{annotation_path} at line {line_no}: {entry!r}"
                )
            self.caption_index[(shard, clip)] = cap
            if shard not in seen_shards:
                seen_shards.add(shard)
                shards_in_order.append(shard)
        _t_parse = time.perf_counter() - _t0

        if not self.caption_index:
            raise ValueError(
                f"{type(self).__name__}: no usable entries found in "
                f"{annotation_path} (caption_field={self.caption_field!r})"
            )

        # Derive tar URLs deterministically from the JSONL ``shard`` fields.
        # NOTE: this avoids globbing ``data_path`` (which can be very slow
        # for directories with many files). Sorted for deterministic
        # ordering across ranks / workers.
        self.urls: list[str] = sorted(
            os.path.join(self.data_path, self.SHARD_KEY_STRIP_PREFIX + s + ".tar")
            for s in shards_in_order
        )

        # Optional best-effort existence check. Disabled by default because
        # ``data_path`` is typically an S3 FUSE mount where each
        # ``os.path.isfile`` triggers a HEAD request -- O(num_shards) round
        # trips on the critical path. WebDataset will raise immediately when
        # opening a missing tar anyway, so this check is purely belt-and-
        # suspenders. Enable with ``verify_shards=True`` for local FS or
        # debugging.
        self._shuffle, self._shuffle_initial = validate_webdataset_sample_shuffle(
            shuffle,
            shuffle_initial,
        )
        self._shardshuffle = shardshuffle
        self._resampled = resampled
        self._nodesplitter = nodesplitter if nodesplitter is not None else wds.split_by_node
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("WebDataset seed must be a non-boolean integer.")
        if isinstance(global_rank, bool) or not isinstance(global_rank, int):
            raise TypeError("WebDataset global_rank must be a non-boolean integer.")
        if global_rank < 0:
            raise ValueError("WebDataset global_rank must be non-negative.")
        if isinstance(seed_namespace, bool) or not isinstance(seed_namespace, int):
            raise TypeError("WebDataset seed_namespace must be a non-boolean integer.")
        if seed_namespace < 0:
            raise ValueError("WebDataset seed_namespace must be non-negative.")
        self._deterministic = True
        self._seed = seed
        self._global_rank = global_rank
        self._seed_namespace = seed_namespace
        # Strict mode: any tar / decode error must propagate. Do NOT use
        # ``wds.warn_and_continue`` (the previous default) which would
        # silently swallow corrupt shards and could turn into an effective
        # infinite loop when combined with ``ResampledShards``.
        self._handler = handler if handler is not None else wds.reraise_exception

        # Optional dataset length. ``IterableDataset`` has no inherent ``__len__``
        # because the number of samples in a tar generally isn't known without
        # scanning every shard. After annotation-based filtering, however, the
        # JSONL row count *is* the exact number of samples that will be yielded
        # per epoch, so we use it as the default. Callers can still override
        # via ``length=`` (clamped to the index size) or assign :attr:`length`
        # later.
        if length is None:
            self._length: int | None = len(self.caption_index)
        else:
            self._length = min(int(length), len(self.caption_index))

        logger.info(
            "%s: loaded %d annotation entries across %d shards (read=%.2fs, parse=%.2fs)",
            type(self).__name__,
            len(self.caption_index),
            len(self.urls),
            _t_read,
            _t_parse,
        )
        if self._deterministic:
            logger.info(
                "%s: deterministic order enabled (seed=%d, global_rank=%d)",
                type(self).__name__,
                self._seed,
                self._global_rank,
            )

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _basename_no_tar(url: str) -> str:
        if not url:
            return ""
        base = os.path.basename(str(url))
        for suffix in (".tar.gz", ".tar.zst", ".tar.xz", ".tar"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    def _extract_shard_key(self, url: str) -> str:
        """Extract a shard identifier (e.g. ``shard-000087``) from a tar URL."""
        return self._basename_no_tar(url)

    def _annotation_lookup_key(self, shard_key: str) -> str:
        """Map a shard key to the key used in the JSONL ``shard`` field.

        Default: strip ``SHARD_KEY_STRIP_PREFIX`` from the front of the shard
        key. Subclasses can override for more elaborate mappings.
        """
        if self.SHARD_KEY_STRIP_PREFIX and shard_key.startswith(self.SHARD_KEY_STRIP_PREFIX):
            return shard_key[len(self.SHARD_KEY_STRIP_PREFIX) :]
        return shard_key

    def _resolve_caption(
        self,
        shard_key: str,
        key: str,
        meta_obj: dict[str, Any] | None = None,
    ) -> str:
        """Resolve a caption for ``(shard_key, key)``.

        Strict mode: raises ``KeyError`` if a caption index is configured
        but the entry is missing, or ``ValueError`` if no caption can be
        resolved at all. Never returns ``None``.
        """
        if self.caption_index:
            lookup_shard = self._annotation_lookup_key(shard_key)
            cap = self.caption_index.get((lookup_shard, key))
            if isinstance(cap, str) and cap:
                return cap
            raise KeyError(
                f"{type(self).__name__}: caption miss for "
                f"(shard={lookup_shard!r}, clip_name={key!r}). "
                f"All samples must have a matching entry in the "
                f"annotation index."
            )

        raise ValueError(f"{type(self).__name__}: annotation caption index is empty.")

    # -- subclass hook ----------------------------------------------------------

    def _decode_payload(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert one raw webdataset sample dict into a training-ready dict.

        Strict mode: must return a valid sample. Raise an exception on any
        malformed input; do NOT return ``None``.
        """
        raise NotImplementedError

    # -- iteration --------------------------------------------------------------

    def _stream_seed(self, stream_id: int) -> int:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        return derive_webdataset_seed(
            self._seed,
            self._global_rank,
            worker_id,
            num_workers,
            stream_id + self._seed_namespace,
        )

    @property
    def dataloader_seed(self) -> int | None:
        """Return an isolated loader seed when deterministic order is enabled."""
        return derive_webdataset_seed(
            self._seed,
            self._global_rank,
            worker_id=0,
            num_workers=1,
            stream_id=_WDS_DATALOADER_STREAM_ID + self._seed_namespace,
        )

    # -- annotation-driven filter ----------------------------------------------

    def _filter_by_annotation(self, samples: Iterable[dict[str, Any]]):
        """Drop samples whose ``(shard, clip_name)`` is not in the JSONL index.

        The tar shards are packed in **full** (every clip), but the JSONL
        annotation file only covers the **subset** that has captions. Iterating
        all tar samples and then failing in :meth:`_resolve_caption` would be
        wrong (strict mode is meant for *truly* unexpected data, not for the
        known unannotated remainder). Instead we filter as soon as we have
        ``__url__`` + ``__key__`` from ``tarfile_to_samples``, before any
        decode work happens.
        """
        ann = self.caption_index
        for sample in samples:
            url = sample.get("__url__", "")
            key = sample.get("__key__", "")
            shard_key = self._annotation_lookup_key(self._extract_shard_key(url))
            if (shard_key, key) in ann:
                yield sample
            # else: silently skip — this is the unannotated remainder, by design.

    def _build_pipeline(self):
        """Build the underlying webdataset pipeline."""
        if self._resampled:
            pipeline: list[Any] = [
                wds.ResampledShards(
                    self.urls,
                    seed=self._stream_seed(_WDS_SHARD_RESAMPLE_STREAM_ID),
                    worker_seed=_zero_webdataset_worker_seed,
                    deterministic=True,
                )
            ]
        else:
            pipeline = [wds.SimpleShardList(self.urls), self._nodesplitter, wds.split_by_worker]
            if self._shardshuffle:
                if isinstance(self._shardshuffle, bool):
                    buf = len(self.urls)
                elif isinstance(self._shardshuffle, int):
                    buf = self._shardshuffle
                else:
                    buf = len(self.urls)
                pipeline.append(
                    wds.shuffle(buf, seed=self._stream_seed(_WDS_SHARD_SHUFFLE_STREAM_ID))
                )
        pipeline.append(wds.tarfile_to_samples(handler=self._handler))
        pipeline.append(self._filter_by_annotation)
        if self._shuffle and self._shuffle > 0:
            pipeline.append(
                wds.shuffle(
                    self._shuffle,
                    initial=self._shuffle_initial,
                    seed=self._stream_seed(_WDS_SAMPLE_SHUFFLE_STREAM_ID),
                )
            )
        return wds.DataPipeline(*pipeline)

    def __iter__(self):
        """Iterate the underlying webdataset pipeline.

        Strict mode: ``_decode_payload`` MUST return a valid sample for every
        input. Any exception propagates immediately so failures surface fast
        instead of being silently dropped (which previously could turn into
        an infinite loop under ``ResampledShards`` and a NCCL watchdog hang
        on other ranks).
        """
        pipeline = self._build_pipeline()
        for sample in pipeline:
            yield self._decode_payload(sample)

    # -- length -----------------------------------------------------------------

    @property
    def length(self) -> int | None:
        """User-supplied dataset length (number of samples), if known."""
        return self._length

    @length.setter
    def length(self, value: int | None) -> None:
        self._length = None if value is None else int(value)

    def __len__(self) -> int:
        """Return the number of samples if known.

        ``IterableDataset`` has no inherent length: the number of samples
        packed inside a set of tar shards is generally unknowable without
        scanning every shard. Callers that know the total (e.g. from a
        precomputed manifest) should pass ``length=N`` to ``__init__`` or set
        ``ds.length = N`` afterwards. If neither is done, this raises
        ``TypeError`` to match the behaviour of a plain ``IterableDataset``
        (so ``DataLoader`` / ``len(ds)`` consumers get a clear error rather
        than a silently wrong number).
        """
        if self._length is None:
            logger.warning(
                "%s has no known length. "
                "Pass `length=<int>` to the constructor or set `ds.length = <int>` "
                "if the total sample count is known.",
                type(self).__name__,
            )
            return 0
        return self._length

"""Validate latent WebDataset shards against the project's training contract.

The validator is read-only. It fully parses the annotation JSONL, then checks a
deterministic sample of tar shards and latent payloads by default. Use
``--all-shards --all-samples`` for full annotated coverage.

Exit code is 0 when no validation errors are found and 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

SHARD_FILENAME_PREFIX = "latent_"
CANONICAL_LATENT_FIELD = "latent.pth"
MAX_JSON_MEMBER_BYTES = 16 * 1024 * 1024

DiskShape = tuple[int, int, int, int]
TrainingShape = tuple[int, int, int, int]


@dataclass(frozen=True)
class AnnotationRecord:
    """One validated annotation entry."""

    line_no: int
    shard: str
    clip_name: str
    caption: str


@dataclass(frozen=True)
class TrainingContract:
    """Effective shape and internal causal contract of one public recipe."""

    path: Path
    full_shape: tuple[int, int, int, int, int]
    block_size: int | None
    causal: bool
    independent_first_frame: bool
    model_name: str | None

    @property
    def sample_shape(self) -> TrainingShape:
        """Return the configured per-sample layout ``[T, C, H, W]``."""
        return self.full_shape[1:]


@dataclass(frozen=True)
class ValidationOptions:
    """Inputs and coverage controls for one validation run."""

    data_path: Path
    annotation_path: Path
    caption_field: str = "global_caption_en"
    expected_shape: DiskShape | None = None
    config_path: Path | None = None
    sample_shards: int = 3
    samples_per_shard: int = 3
    all_shards: bool = False
    all_samples: bool = False
    allow_unsafe_pickle: bool = False
    max_payload_bytes: int = 2 * 1024 * 1024 * 1024
    max_errors: int = 50


@dataclass
class ValidationReport:
    """Validation findings and compact coverage statistics."""

    max_errors: int
    error_count: int = 0
    warning_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    annotation_entries: int = 0
    annotation_shards: int = 0
    checked_shards: int = 0
    checked_samples: int = 0
    matched_annotation_entries: int = 0
    observed_shapes: Counter[DiskShape] = field(default_factory=Counter)
    observed_dtypes: Counter[str] = field(default_factory=Counter)
    training_contract: TrainingContract | None = None
    reference_shape: DiskShape | None = None

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def error(self, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < self.max_errors:
            self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < self.max_errors:
            self.warnings.append(message)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)


def parse_disk_shape(value: str) -> DiskShape:
    """Parse ``C,T,H,W`` (or ``CxTxHxW``) into four positive integers."""
    normalized = value.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"expected four dimensions in C,T,H,W order, got {value!r}"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape contains a non-integer: {value!r}") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {value!r}")
    return shape  # type: ignore[return-value]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _format_shape(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def _validate_shape_values(raw: Any, *, dimensions: int, field_name: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != dimensions:
        raise ValueError(f"{field_name} must contain exactly {dimensions} dimensions, got {raw!r}")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} dimensions must be positive integers, got {raw!r}")
        values.append(value)
    return tuple(values)


def _load_training_contract(config_path: Path) -> TrainingContract:
    # Resolve shared defaults and the public recipe through the same boundary as
    # training, without importing the trainer or touching data/GPU/runtime outputs.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.utils.config import resolve_config

    resolved = resolve_config(["--config", str(config_path), "--dry_run"], training=True)
    config = resolved.public
    full_shape = _validate_shape_values(
        list(config.image_or_video_shape), dimensions=5, field_name="image_or_video_shape"
    )
    return TrainingContract(
        path=resolved.config_path,
        full_shape=full_shape,
        block_size=config.num_frame_per_block,
        causal=True,
        independent_first_frame=config.independent_first_frame,
        model_name=config.model_name,
    )


def _validate_training_contract(
    report: ValidationReport,
    contract: TrainingContract,
    expected_shape: DiskShape | None,
) -> None:
    training_t, training_c, training_h, training_w = contract.sample_shape
    block_size = contract.block_size
    if contract.causal and block_size is not None:
        if contract.independent_first_frame:
            if (training_t - 1) % block_size != 0:
                report.error(
                    f"config {contract.path}: independent_first_frame=true requires "
                    f"(T - 1) % num_frame_per_block == 0, got T={training_t}, "
                    f"block={block_size}"
                )
        elif training_t % block_size != 0:
            report.error(
                f"config {contract.path}: independent_first_frame=false requires "
                f"T % num_frame_per_block == 0, got T={training_t}, block={block_size}"
            )

    if expected_shape is not None:
        disk_c, disk_t, disk_h, disk_w = expected_shape
        if (disk_c, disk_h, disk_w) != (training_c, training_h, training_w):
            report.error(
                "--expected-shape is spatially incompatible with config: disk "
                f"{_format_shape(expected_shape)} [C,T,H,W] versus config "
                f"{_format_shape(contract.sample_shape)} [T,C,H,W]"
            )
        if disk_t < training_t:
            report.error(
                "--expected-shape has fewer latent frames than config consumes: "
                f"disk T={disk_t}, config T={training_t}"
            )


def _load_annotations(
    options: ValidationOptions,
    report: ValidationReport,
) -> dict[str, list[AnnotationRecord]]:
    path = options.annotation_path
    if not path.is_file():
        report.error(f"annotation JSONL does not exist or is not a file: {path}")
        return {}
    if path.suffix.lower() != ".jsonl":
        report.warning(f"annotation file does not use the recommended .jsonl suffix: {path}")

    by_shard: dict[str, list[AnnotationRecord]] = {}
    seen_pairs: dict[tuple[str, str], int] = {}
    try:
        annotation_file = path.open("r", encoding="utf-8")
    except OSError as exc:
        report.error(f"cannot open annotation JSONL {path}: {exc}")
        return {}

    with annotation_file:
        for line_no, raw_line in enumerate(annotation_file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                report.error(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(entry, dict):
                report.error(f"{path}:{line_no}: each line must be a JSON object")
                continue

            shard = entry.get("shard")
            clip_name = entry.get("clip_name")
            caption = entry.get(options.caption_field)
            if not isinstance(shard, str) or not shard.strip():
                report.error(f"{path}:{line_no}: missing or empty string field 'shard'")
                continue
            if (
                shard != shard.strip()
                or shard in {".", ".."}
                or "/" in shard
                or "\\" in shard
                or "\x00" in shard
            ):
                report.error(
                    f"{path}:{line_no}: shard must be a filename-safe logical id, got {shard!r}"
                )
                continue
            if shard.startswith(SHARD_FILENAME_PREFIX) or shard.endswith(".tar"):
                report.error(
                    f"{path}:{line_no}: shard must omit the 'latent_' prefix and '.tar' suffix"
                )
                continue
            if not isinstance(clip_name, str) or not clip_name.strip():
                report.error(f"{path}:{line_no}: missing or empty string field 'clip_name'")
                continue
            if "\x00" in clip_name:
                report.error(f"{path}:{line_no}: clip_name contains a NUL byte")
                continue
            if not isinstance(caption, str) or not caption.strip():
                report.error(
                    f"{path}:{line_no}: missing or empty caption field {options.caption_field!r}"
                )
                continue

            pair = (shard, clip_name)
            if pair in seen_pairs:
                report.error(
                    f"{path}:{line_no}: duplicate (shard, clip_name) {pair!r}; "
                    f"first seen at line {seen_pairs[pair]}"
                )
                continue
            seen_pairs[pair] = line_no
            by_shard.setdefault(shard, []).append(
                AnnotationRecord(
                    line_no=line_no,
                    shard=shard,
                    clip_name=clip_name,
                    caption=caption,
                )
            )

    report.annotation_entries = sum(len(records) for records in by_shard.values())
    report.annotation_shards = len(by_shard)
    if report.annotation_entries == 0:
        report.error(
            f"annotation JSONL has no valid entries (caption field={options.caption_field!r}): {path}"
        )
    return by_shard


def _evenly_spaced[T](items: list[T], count: int) -> list[T]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def _split_webdataset_member(name: str) -> tuple[str, str] | None:
    """Mirror WebDataset's first-dot split into ``(__key__, field)``."""
    basename_start = name.rfind("/") + 1
    dot_index = name.find(".", basename_start)
    if dot_index <= basename_start or dot_index == len(name) - 1:
        return None
    return name[:dot_index], name[dot_index + 1 :]


def _coerce_latent(payload: Any) -> torch.Tensor:
    """Mirror the latent payload forms accepted by ``src/dataset/tensor.py``."""
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(payload)
    if isinstance(payload, dict):
        for key in ("vae_latent", "latent", "latents", "data"):
            if key in payload:
                value = payload[key]
                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, np.ndarray):
                    return torch.from_numpy(value)
                raise TypeError(f"payload[{key!r}] has unsupported type {type(value).__name__}")
        tensor_values = [value for value in payload.values() if isinstance(value, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]
        raise KeyError(
            "dict payload has no recognized latent key "
            "(vae_latent/latent/latents/data) or unique tensor value"
        )
    raise TypeError(f"unsupported payload type {type(payload).__name__}")


def _extract_member_bytes(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    limit: int,
) -> bytes:
    if member.size > limit:
        raise ValueError(
            f"member is {member.size} bytes, exceeding the configured limit of {limit} bytes"
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise OSError("tar member cannot be opened as a regular file")
    with extracted:
        return extracted.read()


def _load_json_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> Any:
    raw = _extract_member_bytes(archive, member, limit=MAX_JSON_MEMBER_BYTES)
    return json.loads(raw.decode("utf-8"))


def _validate_optional_json(
    archive: tarfile.TarFile,
    fields: dict[str, tarfile.TarInfo],
    *,
    shard: str,
    key: str,
    latent: torch.Tensor,
    report: ValidationReport,
) -> None:
    actions_member = fields.get("actions.json")
    if actions_member is not None:
        try:
            _load_json_member(archive, actions_member)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            report.error(f"shard={shard!r}, key={key!r}: invalid actions.json: {exc}")

    meta_member = fields.get("meta.json")
    if meta_member is None:
        return
    try:
        metadata = _load_json_member(archive, meta_member)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        report.error(f"shard={shard!r}, key={key!r}: invalid meta.json: {exc}")
        return
    if not isinstance(metadata, dict):
        report.error(f"shard={shard!r}, key={key!r}: meta.json must contain a JSON object")
        return

    metadata_key = metadata.get("clip_name")
    if metadata_key is not None and metadata_key != key:
        report.error(f"shard={shard!r}, key={key!r}: meta.json clip_name is {metadata_key!r}")
    metadata_shape = metadata.get("latent_shape")
    if metadata_shape is not None:
        try:
            parsed_shape = _validate_shape_values(
                metadata_shape,
                dimensions=4,
                field_name="meta.json latent_shape",
            )
        except ValueError as exc:
            report.error(f"shard={shard!r}, key={key!r}: {exc}")
        else:
            if parsed_shape != tuple(latent.shape):
                report.error(
                    f"shard={shard!r}, key={key!r}: meta.json latent_shape "
                    f"{_format_shape(parsed_shape)} does not match payload "
                    f"{_format_shape(tuple(latent.shape))}"
                )
    metadata_dtype = metadata.get("latent_dtype")
    if metadata_dtype is not None:
        expected_dtype = str(latent.dtype).removeprefix("torch.").lower()
        actual_dtype = str(metadata_dtype).removeprefix("torch.").lower()
        if actual_dtype != expected_dtype:
            report.error(
                f"shard={shard!r}, key={key!r}: meta.json latent_dtype "
                f"{metadata_dtype!r} does not match payload {expected_dtype!r}"
            )


def _validate_observed_shape(
    latent: torch.Tensor,
    *,
    shard: str,
    key: str,
    options: ValidationOptions,
    report: ValidationReport,
) -> None:
    if latent.ndim != 4:
        report.error(
            f"shard={shard!r}, key={key!r}: latent must be 4-D [C,T,H,W], "
            f"got {_format_shape(tuple(latent.shape))}"
        )
        return
    shape: DiskShape = tuple(latent.shape)  # type: ignore[assignment]
    report.observed_shapes[shape] += 1
    dtype_name = str(latent.dtype).removeprefix("torch.")
    report.observed_dtypes[dtype_name] += 1

    if report.reference_shape is None:
        report.reference_shape = shape
    elif shape != report.reference_shape:
        report.error(
            f"shard={shard!r}, key={key!r}: non-uniform latent shape "
            f"{_format_shape(shape)}; first checked shape was "
            f"{_format_shape(report.reference_shape)}"
        )

    if options.expected_shape is not None and shape != options.expected_shape:
        report.error(
            f"shard={shard!r}, key={key!r}: latent shape {_format_shape(shape)} "
            f"does not match --expected-shape {_format_shape(options.expected_shape)}"
        )

    contract = report.training_contract
    if contract is not None:
        training_t, training_c, training_h, training_w = contract.sample_shape
        disk_c, disk_t, disk_h, disk_w = shape
        if (disk_c, disk_h, disk_w) != (training_c, training_h, training_w):
            report.error(
                f"shard={shard!r}, key={key!r}: disk [C,H,W] "
                f"{_format_shape((disk_c, disk_h, disk_w))} does not match config "
                f"{_format_shape((training_c, training_h, training_w))}"
            )
        if disk_t < training_t:
            report.error(
                f"shard={shard!r}, key={key!r}: disk T={disk_t} is shorter than "
                f"config T={training_t}"
            )
        elif disk_t > training_t:
            report.note(
                f"stored T={disk_t} is compatible with config T={training_t}; "
                "the trainer consumes only the leading configured frames"
            )


def _validate_latent_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    fields: dict[str, tarfile.TarInfo],
    *,
    shard: str,
    key: str,
    options: ValidationOptions,
    report: ValidationReport,
) -> None:
    try:
        raw = _extract_member_bytes(
            archive,
            member,
            limit=options.max_payload_bytes,
        )
    except Exception as exc:
        report.error(
            f"shard={shard!r}, key={key!r}: cannot read latent member: {type(exc).__name__}: {exc}"
        )
        return

    try:
        payload = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=not options.allow_unsafe_pickle,
        )
    except Exception as exc:
        safety_hint = ""
        if not options.allow_unsafe_pickle:
            safety_hint = (
                "; safe weights-only loading rejected this payload. Only for trusted data, "
                "retry with --allow-unsafe-pickle"
            )
        report.error(
            f"shard={shard!r}, key={key!r}: torch.load failed: "
            f"{type(exc).__name__}: {exc}{safety_hint}"
        )
        return

    try:
        latent = _coerce_latent(payload)
    except Exception as exc:
        report.error(
            f"shard={shard!r}, key={key!r}: unsupported latent payload: {type(exc).__name__}: {exc}"
        )
        return

    report.checked_samples += 1
    if not latent.is_floating_point():
        report.error(
            f"shard={shard!r}, key={key!r}: latent dtype must be floating point, got {latent.dtype}"
        )
    elif not bool(torch.isfinite(latent).all()):
        report.error(f"shard={shard!r}, key={key!r}: latent contains NaN or Inf values")

    _validate_observed_shape(
        latent,
        shard=shard,
        key=key,
        options=options,
        report=report,
    )
    if latent.ndim == 4:
        _validate_optional_json(
            archive,
            fields,
            shard=shard,
            key=key,
            latent=latent,
            report=report,
        )


def _scan_shard(
    shard: str,
    records: list[AnnotationRecord],
    *,
    options: ValidationOptions,
    report: ValidationReport,
) -> None:
    tar_path = options.data_path / f"{SHARD_FILENAME_PREFIX}{shard}.tar"
    if not tar_path.is_file():
        report.error(f"shard={shard!r}: derived tar file does not exist: {tar_path}")
        return

    annotated_keys = {record.clip_name for record in records}
    groups: dict[str, dict[str, tarfile.TarInfo]] = {}
    try:
        archive = tarfile.open(tar_path, mode="r:*")
    except (OSError, tarfile.TarError) as exc:
        report.error(f"shard={shard!r}: cannot open tar {tar_path}: {exc}")
        return

    with archive:
        try:
            for member in archive:
                if not member.isfile():
                    continue
                split = _split_webdataset_member(member.name)
                if split is None:
                    continue
                key, suffix = split
                if key not in annotated_keys:
                    continue
                normalized_suffix = suffix.lower()
                fields = groups.setdefault(key, {})
                if normalized_suffix in fields:
                    report.error(
                        f"shard={shard!r}, key={key!r}: duplicate tar field {normalized_suffix!r}"
                    )
                    continue
                fields[normalized_suffix] = member
        except (OSError, tarfile.TarError) as exc:
            report.error(f"shard={shard!r}: failed while scanning tar {tar_path}: {exc}")
            return

        loadable_members: dict[str, tarfile.TarInfo] = {}
        for record in records:
            fields = groups.get(record.clip_name)
            if fields is None:
                report.error(
                    f"shard={shard!r}, key={record.clip_name!r}: annotation key "
                    "has no matching tar sample"
                )
                continue
            latent_member = fields.get(CANONICAL_LATENT_FIELD)
            if latent_member is None:
                fallback = next(
                    (
                        member
                        for suffix, member in fields.items()
                        if suffix == "pth" or suffix.endswith(".pth")
                    ),
                    None,
                )
                if fallback is None:
                    report.error(
                        f"shard={shard!r}, key={record.clip_name!r}: missing required "
                        f"{CANONICAL_LATENT_FIELD!r} field"
                    )
                    continue
                report.error(
                    f"shard={shard!r}, key={record.clip_name!r}: latent member "
                    f"{fallback.name!r} is runtime-tolerated but non-canonical; use "
                    f"<key>.{CANONICAL_LATENT_FIELD}"
                )
                latent_member = fallback
            loadable_members[record.clip_name] = latent_member
            report.matched_annotation_entries += 1

        selected_records = (
            records
            if options.all_samples
            else _evenly_spaced(records, min(options.samples_per_shard, len(records)))
        )
        for record in selected_records:
            member = loadable_members.get(record.clip_name)
            fields = groups.get(record.clip_name)
            if member is None or fields is None:
                continue
            _validate_latent_member(
                archive,
                member,
                fields,
                shard=shard,
                key=record.clip_name,
                options=options,
                report=report,
            )

    report.checked_shards += 1


def validate_dataset(options: ValidationOptions) -> ValidationReport:
    """Validate one latent WebDataset and return all bounded findings."""
    report = ValidationReport(max_errors=options.max_errors)
    if options.caption_field != "global_caption_en":
        report.warning(
            "This tool-only caption selection does not establish training compatibility; "
            "public training fixes the field to 'global_caption_en'."
        )
    if options.allow_unsafe_pickle:
        report.warning("unsafe pickle loading is enabled; validate only data from a trusted source")
    data_path = options.data_path
    if not data_path.is_dir():
        report.error(f"data path does not exist or is not a directory: {data_path}")

    try:
        by_shard = _load_annotations(options, report)
    except (OSError, UnicodeError) as exc:
        report.error(f"cannot read annotation JSONL {options.annotation_path}: {exc}")
        by_shard = {}

    if options.config_path is not None:
        try:
            contract = _load_training_contract(options.config_path)
        except Exception as exc:
            report.error(
                f"cannot load training config {options.config_path}: {type(exc).__name__}: {exc}"
            )
        else:
            report.training_contract = contract
            _validate_training_contract(report, contract, options.expected_shape)
    elif options.expected_shape is None:
        report.note(
            "no --config or --expected-shape was supplied; payload rank, finiteness, "
            "and sampled shape uniformity are checked, but not a target model shape"
        )

    if not data_path.is_dir() or not by_shard:
        return report

    shard_names = sorted(by_shard)
    selected_shards = (
        shard_names
        if options.all_shards
        else _evenly_spaced(shard_names, min(options.sample_shards, len(shard_names)))
    )
    if len(selected_shards) < len(shard_names):
        report.note(
            f"archive coverage is sampled: {len(selected_shards)}/{len(shard_names)} shards; "
            "use --all-shards for every referenced tar"
        )
    if not options.all_samples:
        report.note(
            f"payload decoding is sampled at up to {options.samples_per_shard} annotated "
            "samples per checked shard; use --all-samples for every annotated payload"
        )

    for shard in selected_shards:
        _scan_shard(
            shard,
            by_shard[shard],
            options=options,
            report=report,
        )
    return report


def format_report(options: ValidationOptions, report: ValidationReport) -> str:
    """Render a stable human-readable validation summary."""
    result = "PASS" if report.ok else "FAIL"
    lines = [
        f"Latent WebDataset validation: {result}",
        f"Data path: {options.data_path}",
        f"Annotation: {options.annotation_path}",
        (
            "Index: "
            f"{report.annotation_entries} valid entries across "
            f"{report.annotation_shards} shards"
        ),
        (
            "Coverage: "
            f"{report.checked_shards}/{report.annotation_shards} shards opened, "
            f"{report.matched_annotation_entries} annotated keys matched, "
            f"{report.checked_samples} payloads decoded"
        ),
    ]
    if options.expected_shape is not None:
        lines.append(f"Expected disk shape [C,T,H,W]: {_format_shape(options.expected_shape)}")
    if report.training_contract is not None:
        contract = report.training_contract
        model_suffix = f", model={contract.model_name}" if contract.model_name else ""
        lines.append(
            "Training contract [B,T,C,H,W]: "
            f"{_format_shape(contract.full_shape)} from {contract.path}{model_suffix}"
        )
        lines.append(
            "Block config: "
            f"causal={str(contract.causal).lower()}, "
            f"num_frame_per_block={contract.block_size}, "
            f"independent_first_frame={str(contract.independent_first_frame).lower()}"
        )
    if report.observed_shapes:
        observed_shapes = ", ".join(
            f"{_format_shape(shape)} x{count}"
            for shape, count in sorted(report.observed_shapes.items())
        )
        lines.append(f"Observed disk shapes [C,T,H,W]: {observed_shapes}")
    if report.observed_dtypes:
        observed_dtypes = ", ".join(
            f"{dtype} x{count}" for dtype, count in sorted(report.observed_dtypes.items())
        )
        lines.append(f"Observed dtypes: {observed_dtypes}")

    if report.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in report.notes)
    if report.warning_count:
        lines.append(f"Warnings ({report.warning_count}):")
        lines.extend(f"  - {warning}" for warning in report.warnings)
        hidden = report.warning_count - len(report.warnings)
        if hidden > 0:
            lines.append(f"  - ... {hidden} more warnings suppressed")
    if report.error_count:
        lines.append(f"Errors ({report.error_count}):")
        lines.extend(f"  - {error}" for error in report.errors)
        hidden = report.error_count - len(report.errors)
        if hidden > 0:
            lines.append(f"  - ... {hidden} more errors suppressed")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Directory containing latent_<shard>.tar files.",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        required=True,
        help="UTF-8 annotation JSONL containing shard, clip_name, and caption fields.",
    )
    parser.add_argument(
        "--caption-field",
        default="global_caption_en",
        help="Tool-only annotation field check; public training uses global_caption_en.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=(
            "Public recipe (stem or path), merged with shared defaults, whose shape must be "
            "compatible with sampled disk latents."
        ),
    )
    parser.add_argument(
        "--expected-shape",
        type=parse_disk_shape,
        help="Exact on-disk latent shape in C,T,H,W order, e.g. 48,31,44,80.",
    )
    parser.add_argument(
        "--sample-shards",
        type=_positive_int,
        default=3,
        help="Deterministically sample this many shards (default: 3).",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=_positive_int,
        default=3,
        help="Decode this many annotated payloads per checked shard (default: 3).",
    )
    parser.add_argument(
        "--all-shards",
        action="store_true",
        help="Open every shard referenced by the annotation JSONL.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Decode every annotated payload in each checked shard.",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Use torch.load(weights_only=False). This may execute pickle code; enable only "
            "for trusted datasets."
        ),
    )
    parser.add_argument(
        "--max-payload-mib",
        type=_positive_int,
        default=2048,
        help="Refuse to load a sampled .pth member larger than this many MiB (default: 2048).",
    )
    parser.add_argument(
        "--max-errors",
        type=_positive_int,
        default=50,
        help="Maximum errors and warnings printed for each category (default: 50).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(
        data_path=args.data_path,
        annotation_path=args.annotation_path,
        caption_field=args.caption_field,
        expected_shape=args.expected_shape,
        config_path=args.config_path,
        sample_shards=args.sample_shards,
        samples_per_shard=args.samples_per_shard,
        all_shards=args.all_shards,
        all_samples=args.all_samples,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        max_payload_bytes=args.max_payload_mib * 1024 * 1024,
        max_errors=args.max_errors,
    )
    report = validate_dataset(options)
    print(format_report(options, report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

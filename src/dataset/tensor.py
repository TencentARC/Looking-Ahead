import io
import json
import logging
import os
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import torch

from .utils import BaseWebDataset

logger = logging.getLogger(__name__)


def _load_pth_bytes(raw: bytes) -> Any:
    """Load a torch.save-serialized object from raw bytes."""
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def _coerce_latent(obj: Any) -> torch.Tensor:
    """Extract a latent tensor from the deserialized .pth payload.

    Strict mode: raises ``TypeError`` / ``KeyError`` if no tensor can be
    extracted. Accepts:

        - a bare ``torch.Tensor`` (returned as-is)
        - a numpy array -> converted to tensor
        - a dict containing one of the common keys: ``vae_latent``,
          ``latent``, ``latents``, ``data`` -> returns that value
        - a dict with exactly one tensor value -> returns it

    Returns:
        A tensor that downstream decoders expect to be 4-D ``[C, T, H, W]``;
        dimensionality is validated by :meth:`PytorchTensorWebDataset._decode_payload`.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for k in ("vae_latent", "latent", "latents", "data"):
            if k in obj:
                v = obj[k]
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, np.ndarray):
                    return torch.from_numpy(v)
                raise TypeError(
                    f"_coerce_latent: payload[{k!r}] has unsupported type {type(v).__name__}"
                )
        # Fallback: if there's exactly one tensor value, take it.
        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensor_vals) == 1:
            return tensor_vals[0]
        raise KeyError(
            f"_coerce_latent: dict payload has no recognized latent key "
            f"(expected one of vae_latent/latent/latents/data); "
            f"keys={list(obj.keys())!r}"
        )
    raise TypeError(f"_coerce_latent: unsupported payload type {type(obj).__name__}")


class PytorchTensorWebDataset(BaseWebDataset):
    """Pre-encoded latent tar input, with JSONL captions as the authority.

    Store one <key>.latent.pth in [C,T,H,W]; yield float32 [T,C,H,W].
    Optional meta.json is parsed; action payloads are ignored. The runtime has
    no payload-layout selector, custom transform, or disk annotation cache.
    target_frame_num crops a prefix and raises on short samples.
    """

    SHARD_KEY_STRIP_PREFIX = "latent_"

    def __init__(
        self,
        data_path: Union[str, "os.PathLike[str]"],
        annotation_path: str,
        target_frame_num: int | None = None,
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
        super().__init__(
            data_path=data_path,
            annotation_path=annotation_path,
            shuffle=shuffle,
            shuffle_initial=shuffle_initial,
            shardshuffle=shardshuffle,
            resampled=resampled,
            nodesplitter=nodesplitter,
            handler=handler,
            length=length,
            seed=seed,
            global_rank=global_rank,
            seed_namespace=seed_namespace,
        )

        if isinstance(target_frame_num, bool) or (
            target_frame_num is not None and not isinstance(target_frame_num, int)
        ):
            raise TypeError("target_frame_num must be a positive integer or None.")
        if target_frame_num is not None and target_frame_num <= 0:
            raise ValueError("target_frame_num must be greater than zero.")
        self.target_frame_num = target_frame_num

        logger.info(
            "Initialized %s with %d shard(s) "
            "(annotation_path=%r, caption_index_size=%d, cast_dtype=%s, "
            "target_frame_num=%s)",
            self.__class__.__name__,
            len(self.urls),
            self.annotation_path,
            len(self.caption_index),
            torch.float32,
            self.target_frame_num,
        )

    # -- helpers ----------------------------------------------------------------

    def _decode_payload(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Decode one WebDataset sample into the trainer sample contract.

        Args:
            sample: Raw WebDataset sample containing a latent ``.pth`` payload,
                optional ``actions.json`` and optional ``meta.json``.

        Returns:
            Dict with ``prompts`` string, ``clean_latent`` float tensor
            ``[T, C, H, W]`` and metadata keys. On-disk latent layout must be
            ``[C, T, H, W]`` and is permuted before returning.

        Raises:
            ``KeyError`` / ``TypeError`` / ``ValueError`` on malformed samples.
            The dataset is intentionally strict so shard/index drift is visible.
        """
        key = sample.get("__key__", "")
        url = sample.get("__url__", "")
        shard_key = self._extract_shard_key(url)

        latent_raw: bytes | None = None
        meta_obj: dict[str, Any] = {}

        for k, v in sample.items():
            if k.startswith("__"):
                continue
            kl = k.lower()
            if latent_raw is None and (
                kl == "latent.pth"
                or kl.endswith(".latent.pth")
                or kl == "pth"
                or kl.endswith(".pth")
            ):
                latent_raw = v
            elif kl == "meta.json" or kl.endswith(".meta.json"):
                meta_raw = v
                if isinstance(meta_raw, (bytes, bytearray)):
                    meta_raw = meta_raw.decode("utf-8", errors="replace")
                meta_obj = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})

        if latent_raw is None:
            raise KeyError(
                f"{type(self).__name__}: no latent .pth payload found in "
                f"sample (url={url!r}, key={key!r}, fields={list(sample.keys())!r})"
            )
        if not isinstance(latent_raw, (bytes, bytearray)):
            raise TypeError(
                f"{type(self).__name__}: latent payload has unexpected "
                f"type {type(latent_raw).__name__} "
                f"(url={url!r}, key={key!r}); expected raw bytes. "
                f"Did webdataset auto-decode the .pth file?"
            )

        prompt = self._resolve_caption(shard_key, key, meta_obj)

        payload = _load_pth_bytes(bytes(latent_raw))
        latent = _coerce_latent(payload)

        latent = latent.to(dtype=torch.float32)

        # On-disk layout is (C, T, H, W); align with the trainer layout (T, C, H, W).
        if latent.dim() != 4:
            raise ValueError(
                f"{type(self).__name__}: latent must be 4-D (C, T, H, W), "
                f"got shape {tuple(latent.shape)} (url={url!r}, key={key!r})"
            )
        latent = latent.permute(1, 0, 2, 3).contiguous()

        if self.target_frame_num is not None:
            actual_frame_num = latent.shape[0]
            if actual_frame_num < self.target_frame_num:
                raise ValueError(
                    f"{type(self).__name__}: latent requires at least "
                    f"{self.target_frame_num} frame(s), got {actual_frame_num} "
                    f"(url={url!r}, key={key!r}); short samples are not padded."
                )
            latent = latent[: self.target_frame_num].contiguous()

        result: dict[str, Any] = {
            "prompts": prompt,
            "clean_latent": latent,
            "metadata": {
                "key": key,
                "shard_key": shard_key,
            },
        }
        return result

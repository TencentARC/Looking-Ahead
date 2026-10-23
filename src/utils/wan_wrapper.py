"""Wan text/VAE/backbone readers for the selected causal TF/FCP interfaces."""

import logging
import os
import time
from contextlib import contextmanager, nullcontext

import torch

try:
    import fcntl
except ImportError:  # Windows has no flock; the base model then loads from WAN_ROOT directly.
    fcntl = None

from src.utils.scheduler import FlowMatchScheduler
from src.utils.wan_model_family import get_wan_model_family
from src.wan.modules.t5 import umt5_xxl
from src.wan.modules.tokenizers import HuggingfaceTokenizer
from src.wan.modules.vae import _video_vae
from src.wan22.modules.vae2_2 import _video_vae as _video_vae_2_2

WAN_ROOT = os.environ.get("WAN_ROOT", "./weights/Wan-AI")
WAN_LOCAL_CACHE = os.environ.get("WAN_LOCAL_CACHE", "/tmp/Wan-AI")
WAN_LOCAL_CACHE_ENABLE = os.environ.get("WAN_LOCAL_CACHE_ENABLE", "1").lower() not in (
    "0",
    "false",
    "no",
)
WAN_LOCAL_CACHE_WORKERS = int(os.environ.get("WAN_LOCAL_CACHE_WORKERS", "16"))
_RESOLVED_MODEL_DIRS = {}


def _flatten_video_timestep(
    timestep: torch.Tensor, *, batch_size: int, frame_count: int
) -> torch.Tensor:
    """Return one timestep per flattened ``[batch, frame]`` element."""
    if timestep.ndim == 1 and timestep.shape[0] == batch_size:
        timestep = timestep[:, None]
    if timestep.ndim != 2 or timestep.shape[0] != batch_size:
        raise ValueError(
            "Video timestep must have shape [B], [B, 1], or [B, T]; "
            f"got {tuple(timestep.shape)} for B={batch_size}, T={frame_count}."
        )
    if timestep.shape[1] == 1:
        timestep = timestep.expand(batch_size, frame_count)
    elif timestep.shape[1] != frame_count:
        raise ValueError(
            "Video timestep frame dimension must be 1 or match the latent frames; "
            f"got {timestep.shape[1]} for T={frame_count}."
        )
    return timestep.reshape(batch_size * frame_count)


@contextmanager
def isolated_torch_cpu_rng(seed: int):
    """Temporarily seed CPU initialization without advancing global RNG state.

    Training seeds normally differ by rank. Randomly initialized model
    parameters, however, must start from one shared seed before FSDP shards the
    module. This context installs the state of an isolated CPU generator and
    restores the caller's CPU RNG state on exit. It does not read or modify any
    CUDA generator, including the global stream used for training noise and
    timestep sampling.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Diffusion initialization seed must be a non-boolean integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        yield


def _initialize_wan_diffusion_model(
    model_class,
    pretrained_model_name_or_path: str,
    *,
    load_diffusion_pretrained: bool,
    subfolder: str | None = None,
    model_kwargs: dict | None = None,
):
    """Create a Wan diffusion module with or without pretrained DiT weights.

    The random-initialization path still reads the upstream ``config.json`` so
    model-family dimensions remain authoritative, but it never calls
    ``from_pretrained`` and therefore does not deserialize the diffusion
    checkpoint. Text-encoder and VAE loading are owned by their separate
    wrappers and are unaffected.
    """
    if not isinstance(load_diffusion_pretrained, bool):
        raise TypeError("load_diffusion_pretrained must be boolean.")
    model_kwargs = dict(model_kwargs or {})
    location_kwargs = {}
    if subfolder is not None:
        location_kwargs["subfolder"] = subfolder

    if load_diffusion_pretrained:
        return model_class.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **location_kwargs,
            **model_kwargs,
        )

    model_config = model_class.load_config(
        pretrained_model_name_or_path,
        **location_kwargs,
    )
    return model_class.from_config(model_config, **model_kwargs)


def _parallel_copy_tree(src_dir: str, dst_dir: str, num_workers: int) -> None:
    """Copy ``src_dir`` -> ``dst_dir`` recursively using a thread pool.

    Equivalent in effect to ``cp -r src_dir/. dst_dir/``: directory
    structure is mirrored, and individual files are copied in parallel
    with ``shutil.copy2`` (preserves mtime / mode). Existing destination
    files with the same size are skipped, so this is resumable across
    interrupted runs.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. Mirror the directory tree first (cheap, single-threaded).
    for root, dirs, _files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(target_root, exist_ok=True)
        # ``dirs`` is iterated implicitly via os.walk recursion.
        del dirs

    # 2. Build the (src_path, dst_path) work list.
    work: list = []
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        for fn in files:
            sp = os.path.join(root, fn)
            dp = os.path.join(target_root, fn)
            # Skip files that already look fully copied.
            try:
                if os.path.exists(dp) and os.path.getsize(dp) == os.path.getsize(sp):
                    continue
            except OSError:
                pass
            work.append((sp, dp))

    if not work:
        return

    def _one(sp_dp):
        sp, dp = sp_dp
        # ``copy2`` preserves metadata (mtime / mode), which keeps the
        # sentinel-based cache reuse predictable across runs.
        shutil.copy2(sp, dp)
        return os.path.getsize(dp)

    workers = max(1, min(num_workers, len(work)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, item) for item in work]
        for f in as_completed(futs):
            # Surface the first error (if any); ThreadPoolExecutor will
            # cancel queued futures once we propagate.
            f.result()


def _resolve_wan_model_dir(model_name: str) -> str:
    """Return the directory that should be used as ``WAN_ROOT/<model_name>``.

    Behaviour:

    * If staging is disabled (``WAN_LOCAL_CACHE_ENABLE=0``) or
      ``WAN_ROOT`` already points inside ``WAN_LOCAL_CACHE`` (i.e. the
      source is *already* on local fast storage), we just return
      ``os.path.join(WAN_ROOT, model_name)`` and load directly.
    * Otherwise, we stage the *single* model subdirectory we need to
      local disk under ``WAN_LOCAL_CACHE/<model_name>/`` using a
      thread-pooled file copy and return that path. We deliberately do
      NOT mirror all of ``WAN_ROOT`` (the parent ``Wan-AI/`` tree is
      huge and contains many models we don't use this run).

    Concurrency: a ``.wan_cache_done`` sentinel inside the staged
    directory marks a successful copy. We hold an ``flock`` on a sibling
    lockfile while copying so multiple ranks/workers in the same node
    don't all try to pull the same ~30 GiB twice.
    """
    cached = _RESOLVED_MODEL_DIRS.get(model_name)
    if cached is not None:
        return cached

    src = os.path.join(WAN_ROOT, model_name)

    # Skip staging when:
    #   * the user opted out, or
    #   * WAN_ROOT already lives inside WAN_LOCAL_CACHE (no point
    #     copying onto itself), or
    #   * the source dir doesn't exist (let the downstream loader raise
    #     a clear error against the original path).
    try:
        wan_root_abs = os.path.realpath(WAN_ROOT)
        cache_root_abs = os.path.realpath(WAN_LOCAL_CACHE)
    except OSError:
        wan_root_abs, cache_root_abs = WAN_ROOT, WAN_LOCAL_CACHE
    already_local = wan_root_abs == cache_root_abs or wan_root_abs.startswith(
        cache_root_abs.rstrip("/") + "/"
    )
    if fcntl is None or not WAN_LOCAL_CACHE_ENABLE or already_local or not os.path.isdir(src):
        _RESOLVED_MODEL_DIRS[model_name] = src
        return src

    dst = os.path.join(WAN_LOCAL_CACHE, model_name)
    sentinel = os.path.join(dst, ".wan_cache_done")
    lock_path = os.path.join(WAN_LOCAL_CACHE, f".{model_name.replace('/', '_')}.lock")
    os.makedirs(WAN_LOCAL_CACHE, exist_ok=True)

    # Fast path: another process / a previous run already staged it.
    if os.path.isfile(sentinel):
        logging.info(f"[wan-cache] using cached {dst}")
        _RESOLVED_MODEL_DIRS[model_name] = dst
        return dst

    os.makedirs(dst, exist_ok=True)
    logging.info(
        f"[wan-cache] staging {src} -> {dst} "
        f"(parallel copy, workers={WAN_LOCAL_CACHE_WORKERS}; this happens "
        f"once per node; subsequent runs reuse the cache)"
    )
    _t = time.perf_counter()
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check after acquiring the lock: a sibling process may
            # have finished while we were waiting.
            if os.path.isfile(sentinel):
                logging.info(
                    f"[wan-cache] another process finished staging while we "
                    f"waited for the lock; reusing {dst}"
                )
            else:
                try:
                    _parallel_copy_tree(src, dst, WAN_LOCAL_CACHE_WORKERS)
                except Exception as e:  # noqa: BLE001
                    logging.warning(
                        f"[wan-cache] parallel copy failed ({type(e).__name__}: "
                        f"{e}); falling back to source path {src}"
                    )
                    _RESOLVED_MODEL_DIRS[model_name] = src
                    return src
                # Mark success so future runs / wrappers hit the fast path.
                with open(sentinel, "w") as f:
                    f.write(f"copied {src} -> {dst}\n")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    logging.info(f"[wan-cache] staged {model_name} in {time.perf_counter() - _t:.1f}s -> {dst}")
    _RESOLVED_MODEL_DIRS[model_name] = dst
    return dst


class WanVAEWrapper(torch.nn.Module):
    """Project-facing wrapper around Wan VAE encode/decode APIs.

    Public methods use project video/latent layout ``[B, T, C, H, W]`` for
    latents and ``[B, C, T, H, W]`` for raw pixels passed into
    :meth:`encode_to_latent`. The underlying Wan VAE consumes
    ``[B, C, T, H, W]`` and normalizes latents with model-specific mean/std.
    """

    def __init__(self, model_name="Wan2.1-T2V-1.3B"):
        """Load the Wan VAE and model-specific latent normalization stats.

        Args:
            model_name: Wan checkpoint family. Determines VAE checkpoint file,
                latent channel count, temporal/spatial compression and default
                FPS.
        """
        super().__init__()
        get_wan_model_family(model_name)
        self.model_name = model_name
        self.dtype = torch.bfloat16

        # init param
        if model_name == "Wan2.2-TI2V-5B":
            self.default_fps = 24
            self.mean = torch.tensor(
                [
                    -0.2289,
                    -0.0052,
                    -0.1323,
                    -0.2339,
                    -0.2799,
                    0.0174,
                    0.1838,
                    0.1557,
                    -0.1382,
                    0.0542,
                    0.2813,
                    0.0891,
                    0.1570,
                    -0.0098,
                    0.0375,
                    -0.1825,
                    -0.2246,
                    -0.1207,
                    -0.0698,
                    0.5109,
                    0.2665,
                    -0.2108,
                    -0.2158,
                    0.2502,
                    -0.2055,
                    -0.0322,
                    0.1109,
                    0.1567,
                    -0.0729,
                    0.0899,
                    -0.2799,
                    -0.1230,
                    -0.0313,
                    -0.1649,
                    0.0117,
                    0.0723,
                    -0.2839,
                    -0.2083,
                    -0.0520,
                    0.3748,
                    0.0152,
                    0.1957,
                    0.1433,
                    -0.2944,
                    0.3573,
                    -0.0548,
                    -0.1681,
                    -0.0667,
                ],
                dtype=torch.float32,
            )
            self.std = torch.tensor(
                [
                    0.4765,
                    1.0364,
                    0.4514,
                    1.1677,
                    0.5313,
                    0.4990,
                    0.4818,
                    0.5013,
                    0.8158,
                    1.0344,
                    0.5894,
                    1.0901,
                    0.6885,
                    0.6165,
                    0.8454,
                    0.4978,
                    0.5759,
                    0.3523,
                    0.7135,
                    0.6804,
                    0.5833,
                    1.4146,
                    0.8986,
                    0.5659,
                    0.7069,
                    0.5338,
                    0.4889,
                    0.4917,
                    0.4069,
                    0.4999,
                    0.6866,
                    0.4093,
                    0.5709,
                    0.6065,
                    0.6415,
                    0.4944,
                    0.5726,
                    1.2042,
                    0.5458,
                    1.6887,
                    0.3971,
                    1.0600,
                    0.3943,
                    0.5537,
                    0.5444,
                    0.4089,
                    0.7468,
                    0.7744,
                ],
                dtype=torch.float32,
            )
        else:
            self.default_fps = 16
            self.mean = torch.tensor(
                [
                    -0.7571,
                    -0.7089,
                    -0.9113,
                    0.1075,
                    -0.1745,
                    0.9653,
                    -0.1517,
                    1.5508,
                    0.4134,
                    -0.0715,
                    0.5517,
                    -0.3632,
                    -0.1922,
                    -0.9497,
                    0.2503,
                    -0.2921,
                ],
                dtype=torch.float32,
            )
            self.std = torch.tensor(
                [
                    2.8184,
                    1.4541,
                    2.3275,
                    2.6558,
                    1.2196,
                    1.7708,
                    2.6052,
                    2.0743,
                    3.2687,
                    2.1526,
                    2.8652,
                    1.5579,
                    1.6382,
                    1.1253,
                    2.8251,
                    1.9160,
                ],
                dtype=torch.float32,
            )

        # init model
        if model_name == "Wan2.2-TI2V-5B":
            vae_path = os.path.join(_resolve_wan_model_dir(model_name), "Wan2.2_VAE.pth")
            logging.info(f"[WanVAEWrapper] Loading VAE ({model_name}) from {vae_path}")
            _t0 = time.perf_counter()
            self.model = (
                _video_vae_2_2(
                    pretrained_path=vae_path,
                    z_dim=48,
                    dim=160,
                    dim_mult=[1, 2, 4, 4],
                    temperal_downsample=[False, True, True],
                )
                .eval()
                .requires_grad_(False)
            )
            logging.info(
                f"[WanVAEWrapper] VAE ({model_name}) loaded in {time.perf_counter() - _t0:.2f}s"
            )
        else:
            vae_path = os.path.join(_resolve_wan_model_dir(model_name), "Wan2.1_VAE.pth")
            logging.info(f"[WanVAEWrapper] Loading VAE ({model_name}) from {vae_path}")
            _t0 = time.perf_counter()
            self.model = (
                _video_vae(
                    pretrained_path=vae_path,
                    z_dim=16,
                )
                .eval()
                .requires_grad_(False)
            )
            logging.info(
                f"[WanVAEWrapper] VAE ({model_name}) loaded in {time.perf_counter() - _t0:.2f}s"
            )
            self.vae_stride = (4, 8, 8)
            self.target_video_length = 81

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode raw pixel videos into project-layout latents.

        Args:
            pixel: Tensor ``[B, C=3, T_pixel, H_pixel, W_pixel]`` in the Wan
                VAE value range (typically ``[-1, 1]``).

        Returns:
            Latents ``[B, T_latent, C_latent, H_latent, W_latent]`` as
            ``float32``.
        """
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Decode project-layout latents back to pixel videos.

        Args:
            latent: Tensor ``[B, T_latent, C_latent, H_latent, W_latent]``.
            use_cache: Use the Wan VAE streaming temporal cache. This is used
                by long-video inference and currently requires ``B == 1``.

        Returns:
            Pixel tensor ``[B, T_pixel, C=3, H_pixel, W_pixel]`` in ``[-1, 1]``.

        Compatibility:
            Callers using ``use_cache=True`` must clear ``self.model``'s cache
            after a full decode sequence.
        """
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanTextEncoder(torch.nn.Module):
    """umT5 text encoder wrapper returning Wan-compatible prompt embeddings."""

    def __init__(self, model_name="Wan2.1-T2V-1.3B") -> None:
        """Load tokenizer and T5 encoder for a Wan checkpoint family.

        The encoder skeleton is built on the ``meta`` device and populated with
        ``load_state_dict(assign=True)`` to avoid expensive random init and
        extra host-side copies before moving to GPU.
        """
        super().__init__()
        get_wan_model_family(model_name)
        self.model_name = model_name

        _model_dir = _resolve_wan_model_dir(self.model_name)
        t5_ckpt_path = os.path.join(_model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path = os.path.join(_model_dir, "google/umt5-xxl/")
        logging.info(f"[WanTextEncoder] Loading umT5-xxl ({model_name}) from {t5_ckpt_path}")
        _t0 = time.perf_counter()

        # ----- Phase 1: build the encoder skeleton on the ``meta`` device ----
        # Profiling showed the original ~1 minute init was *not* dominated by
        # the 11.4 GiB file read (often ~190 MB/s, well below the local
        # storage's real bandwidth). The bulk of the time was actually
        # CPU-side work that gets thrown away the moment we
        # ``load_state_dict``:
        #   * ``T5Encoder.__init__`` ends with ``self.apply(init_weights)``,
        #     which calls ``nn.init.normal_`` on every Linear weight (and
        #     ``Embedding``'s default reset). For umt5-xxl that's ~5.7 B
        #     parameters of single-threaded RNG writes (tens of GiB of
        #     stores). Those values are immediately overwritten by the
        #     ckpt -> pure waste.
        #   * ``_t5`` then does an extra ``model.to(dtype, device)`` walk
        #     over every parameter.
        # Constructing on ``device='meta'`` gives every parameter a shape +
        # dtype but *no storage*: the ``nn.init`` calls become no-ops (init
        # on meta tensors does nothing), no RNG, no allocation. We then use
        # ``load_state_dict(..., assign=True)`` to swap in the ckpt's real
        # storages without copying.
        _t_build = time.perf_counter()
        with torch.device("meta"):
            self.text_encoder = umt5_xxl(
                dtype=torch.bfloat16,
                device=torch.device("meta"),
            )
        self.text_encoder.eval().requires_grad_(False)
        logging.info(
            f"[WanTextEncoder] meta-built skeleton in {time.perf_counter() - _t_build:.2f}s"
        )

        # ----- Phase 2: read the ckpt -----------------------------------------
        # ``mmap=True`` returns file-backed views; the actual bytes are pulled
        # in lazily when ``assign=True`` re-points the parameters at them.
        # ``weights_only=True`` skips arbitrary pickle execution.
        _t_read = time.perf_counter()
        sd = torch.load(t5_ckpt_path, map_location="cpu", mmap=True, weights_only=True)
        logging.info(
            f"[WanTextEncoder] torch.load (mmap) returned {len(sd)} tensors "
            f"in {time.perf_counter() - _t_read:.2f}s"
        )

        # ----- Phase 3: assign storages into the meta module ------------------
        # ``assign=True`` is what actually materialises the parameters: the
        # meta-tensor params get their storage replaced by the (bf16, mmap'd)
        # ckpt tensors. This is also what triggers the real disk read,
        # so its wall time is the meaningful "load" number to watch.
        _t_assign = time.perf_counter()
        self.text_encoder.load_state_dict(sd, assign=True)
        logging.info(
            f"[WanTextEncoder] load_state_dict(assign=True) in "
            f"{time.perf_counter() - _t_assign:.2f}s"
        )
        del sd

        # ----- Phase 4: tokenizer (small, but still log it for visibility) ---
        _t_tok = time.perf_counter()
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")
        logging.info(f"[WanTextEncoder] tokenizer in {time.perf_counter() - _t_tok:.2f}s")

        logging.info(
            f"[WanTextEncoder] umT5-xxl ({model_name}) loaded in {time.perf_counter() - _t0:.2f}s"
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: list[str]) -> dict:
        """Encode prompts into padded text embeddings.

        Args:
            text_prompts: List of prompt strings, length ``B``.

        Returns:
            ``{"prompt_embeds": context}`` where ``context`` is
            ``[B, text_len, dim]`` on the current CUDA device. Padding tokens
            are zeroed for downstream attention-cache reuse.
        """
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens, strict=False):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanDiffusionWrapper(torch.nn.Module):
    """Project [B,T,C,H,W] interface around the selected causal Wan backbone."""

    def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=5.0,
        local_attn_size=-1,
        latent_shape=None,
        sink_size=0,
        use_predictive_causal_model=False,
        load_diffusion_pretrained=True,
        initialization_seed=0,
    ):
        super().__init__()
        family = get_wan_model_family(model_name)
        self.model_name = model_name
        self.load_diffusion_pretrained = load_diffusion_pretrained
        self.default_frame_size = family.default_frame_size
        self.default_latent_size = family.default_latent_size
        self.vae_compression_factor = family.vae_compression_factor
        self.default_fps = family.default_fps
        self.latent_shape = (
            tuple(latent_shape[-4:]) if latent_shape is not None else self.default_latent_size
        )
        if len(self.latent_shape) != 4 or self.latent_shape[1] != family.latent_channels:
            raise ValueError(f"{model_name} requires [T,{family.latent_channels},H,W] latents.")
        self.frame_shape = (
            (self.latent_shape[0] - 1) * family.vae_temporal_stride + 1,
            3,
            self.latent_shape[2] * family.vae_spatial_stride,
            self.latent_shape[3] * family.vae_spatial_stride,
        )
        self.frame_seq_len = self.latent_shape[-2] * self.latent_shape[-1] // 4
        if local_attn_size == -1:
            local_attn_size = self.latent_shape[0]
        if use_predictive_causal_model:
            from src.utils.predictive_causal_model import PredictiveCausalWanModel as ModelClass
        else:
            from src.utils.causal_model import CausalWanModel as ModelClass
        location = _resolve_wan_model_dir(model_name)
        logging.info("Loading %s (%s) from %s", ModelClass.__name__, model_name, location)
        # Objective-level random construction owns the shared backbone+aux RNG
        # context; standalone wrappers keep their isolated initialization.
        initialization_context = (
            nullcontext()
            if initialization_seed is None
            else isolated_torch_cpu_rng(initialization_seed)
        )
        with initialization_context:
            self.model = _initialize_wan_diffusion_model(
                ModelClass,
                location,
                load_diffusion_pretrained=load_diffusion_pretrained,
                model_kwargs={
                    "local_attn_size": local_attn_size,
                    "sink_size": sink_size,
                    "train_frame_size": self.latent_shape[0],
                    "model_type": family.task,
                },
            )
        self.model.eval()
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0)
        self.scheduler.set_timesteps(1000, training=True)

    def enable_gradient_checkpointing(self):
        self.model.enable_gradient_checkpointing()

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        *,
        kv_cache=None,
        crossattn_cache=None,
        current_start=None,
        cache_start=None,
        clean_x=None,
        return_predictive_aux=False,
        predictive_aux_noisy_chunks_by_offset=None,
        predictive_aux_timestep_chunks_by_offset=None,
    ):
        kwargs = {"t": timestep, "context": conditional_dict["prompt_embeds"]}
        if kv_cache is not None:
            kwargs.update(
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
            )
        elif clean_x is not None:
            kwargs.update(clean_x=clean_x.permute(0, 2, 1, 3, 4))
        if return_predictive_aux:
            kwargs["return_predictive_aux"] = True
            if predictive_aux_noisy_chunks_by_offset is not None:
                kwargs["aux_noisy_chunks_by_offset"] = {
                    int(offset): chunks.permute(0, 3, 1, 2, 4, 5).contiguous()
                    for offset, chunks in predictive_aux_noisy_chunks_by_offset.items()
                }
            if predictive_aux_timestep_chunks_by_offset is not None:
                kwargs["aux_timestep_chunks_by_offset"] = predictive_aux_timestep_chunks_by_offset
        output = self.model(noisy_image_or_video.permute(0, 2, 1, 3, 4), **kwargs)
        flow_pred, auxiliary = output if return_predictive_aux else (output, None)
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            _flatten_video_timestep(
                timestep, batch_size=flow_pred.shape[0], frame_count=flow_pred.shape[1]
            ),
        ).unflatten(0, flow_pred.shape[:2])
        return (flow_pred, pred_x0, auxiliary) if return_predictive_aux else (flow_pred, pred_x0)

    def get_scheduler(self):
        return self.scheduler

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = (
            x.double().to(flow_pred.device)
            for x in [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

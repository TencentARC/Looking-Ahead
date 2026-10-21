"""Local NVMe staging + loading helpers for slow-filesystem-backed checkpoints.

Why this module exists
----------------------

Repeated checkpoint reads from network or FUSE filesystems can be slow.
Training initialization and inference share these helpers to stage checkpoint
files in a local cache and reuse them across runs.

What it does
------------

* :func:`stage_ckpt_to_local` optionally copies a checkpoint file or its
  regular sibling files to a local cache root and returns the resulting path.
  Callers can choose their own ``torch.load`` semantics.
* :func:`load_ckpt` is the convenience wrapper used by training and inference:
  stages, then ``torch.load(..., mmap=True, weights_only=True)`` with a
  graceful fallback chain and explicit timing logs.

Both honour the same env vars:

* ``WAN_CKPT_LOCAL_CACHE``           - cache root (default
  ``/tmp/wan-ckpt-cache``).
* ``WAN_CKPT_LOCAL_CACHE_ENABLE``    - set to ``0`` / ``false`` to bypass
  staging entirely.
* ``WAN_CKPT_LOCAL_CACHE_BUDGET_GIB`` - LRU eviction budget in GiB
  (default 200).
* ``WAN_CKPT_REMOTE_PREFIXES``       - optional comma-separated list of
  path prefixes whose checkpoints should be staged. When **unset**,
  staging applies to *every* source path outside the cache root.
  When set, only paths starting
  with one of the prefixes are staged - useful if some of your
  checkpoints already live on fast local disk and you want to skip the
  copy for those.

Concurrency
-----------

Multiple ranks / processes on the same node may all enter staging at
once. We serialise per-checkpoint via ``fcntl.flock`` on a sibling
lockfile and gate completion on a ``.stage_done`` sentinel file inside
the staged dir. Whoever wins the race performs the copy; everyone else
hits the fast path on wakeup.

Failure handling
----------------

Every external dependency has a fall-through branch that logs a warning
and returns the *original* path. The caller's load will then succeed,
just slowly. This is intentional: a flaky cache must never block a run.
"""

from __future__ import annotations

import logging
import os
import time

import torch

__all__ = [
    "stage_ckpt_to_local",
    "load_ckpt",
]

# --- Configuration (env-driven) ------------------------------------------

_CKPT_LOCAL_CACHE_DIR = os.environ.get("WAN_CKPT_LOCAL_CACHE", "/tmp/wan-ckpt-cache")
_CKPT_LOCAL_CACHE_ENABLE = os.environ.get("WAN_CKPT_LOCAL_CACHE_ENABLE", "1").lower() not in (
    "0",
    "false",
    "no",
)
# Cache budget in GiB; configurable through the environment.
_CKPT_LOCAL_CACHE_BUDGET_BYTES = int(
    float(os.environ.get("WAN_CKPT_LOCAL_CACHE_BUDGET_GIB", "200")) * (1024**3)
)

# Optional whitelist: only paths under one of these prefixes are staged.
# Configure via the comma-separated ``WAN_CKPT_REMOTE_PREFIXES`` env var
# (e.g. ``WAN_CKPT_REMOTE_PREFIXES=/mnt/s3/,/mnt/nfs/``). When unset,
# *every* source path is eligible for staging (we still skip paths that
# are already inside the cache dir, see ``_should_stage`` below).
_REMOTE_FUSE_PREFIX = tuple(
    p for p in os.environ.get("WAN_CKPT_REMOTE_PREFIXES", "").split(",") if p
)


def _should_stage(path: str) -> bool:
    """Return True iff ``path`` should be copied to the local cache.

    Rules:

    * If staging is disabled via env var -> False.
    * If ``WAN_CKPT_REMOTE_PREFIXES`` is set, only paths under one of the
      configured prefixes are staged. This preserves the original opt-in
      whitelist behaviour for users who want it.
    * Otherwise (default), every path is eligible *except* paths that
      already live inside the local cache directory - copying ``cache/x``
      to ``cache/y`` would be pointless and would also break the
      "directory hash -> staged subdir" mapping.
    """
    if not _CKPT_LOCAL_CACHE_ENABLE:
        return False
    if _REMOTE_FUSE_PREFIX:
        return path.startswith(_REMOTE_FUSE_PREFIX)
    # Default: stage everything that isn't already in the cache root.
    try:
        path_abs = os.path.realpath(path)
        cache_abs = os.path.realpath(_CKPT_LOCAL_CACHE_DIR)
    except OSError:
        return True
    return not path_abs.startswith(cache_abs.rstrip("/") + "/")


def _get_logger(logger):
    """Return a usable logger.

    Callers in ``inference.py`` already have a structured logger; the
    trainer/model code just imports ``logging`` at module level. Accepting
    ``None`` lets us slot into both without forcing every call site to
    construct one.
    """
    return logger if logger is not None else logging.getLogger(__name__)


# --- Public API ----------------------------------------------------------


def stage_ckpt_to_local(path: str, *, logger=None, copy_siblings: bool = True) -> str:
    """Mirror a slow-filesystem-backed checkpoint to a local cache; return new path.

    Staging is enabled by default (``WAN_CKPT_LOCAL_CACHE_ENABLE=1``) and
    applies to every source path unless ``WAN_CKPT_REMOTE_PREFIXES`` is
    set, in which case only paths under one of the configured prefixes
    are staged. Paths that already live inside the cache root are
    returned as-is to avoid self-copies. See :func:`_should_stage` for
    the full predicate.

    When staging applies:

    1. With ``copy_siblings=True`` (the trainer default), hash the parent
       directory and copy every regular sibling so warm-start EMA lookup sees
       the complete checkpoint. With ``False`` (used by inference), hash the
       parent plus selected filename and copy only that file.
    3. Write ``.stage_done`` to mark completion; subsequent calls hit the
       fast path and ``utime`` the sentinel so LRU reflects "most
       recently *used*" rather than "most recently *fetched*".
    4. After a successful stage, evict oldest staged dirs until the cache
       size is back under :data:`_CKPT_LOCAL_CACHE_BUDGET_BYTES`.

    On any error (copy failure, disk full, ...) returns the original
    ``path`` so the caller's ``torch.load`` still works.
    """
    log = _get_logger(logger)

    if not _should_stage(path):
        return path

    import fcntl
    import hashlib
    import shutil

    src_dir = os.path.dirname(path)
    fname = os.path.basename(path)
    scope_key = src_dir if copy_siblings else f"{src_dir}\0{fname}"
    dir_key = hashlib.sha1(scope_key.encode("utf-8")).hexdigest()[:16]
    scope_suffix = (
        os.path.basename(src_dir) if copy_siblings else f"{os.path.basename(src_dir)}__{fname}"
    )
    cache_subdir = os.path.join(_CKPT_LOCAL_CACHE_DIR, f"{dir_key}__{scope_suffix}")
    sentinel = os.path.join(cache_subdir, ".stage_done")
    lock_path = os.path.join(_CKPT_LOCAL_CACHE_DIR, f".{dir_key}.lock")
    staged_path = os.path.join(cache_subdir, fname)

    try:
        os.makedirs(_CKPT_LOCAL_CACHE_DIR, exist_ok=True)
    except OSError as e:
        log.warning(
            f"[ckpt-cache] cannot create cache dir {_CKPT_LOCAL_CACHE_DIR} "
            f"({e}); falling back to source path."
        )
        return path

    # Fast path: directory already staged.
    if os.path.isfile(sentinel) and os.path.isfile(staged_path):
        try:
            os.utime(sentinel, None)  # bump LRU mtime
        except OSError:
            pass
        log.info(f"[ckpt-cache] hit: {staged_path}")
        return staged_path

    log.info(
        f"[ckpt-cache] staging {src_dir} -> {cache_subdir} "
        f"(once per ckpt; reused on subsequent runs)"
    )
    _t = time.perf_counter()
    os.makedirs(cache_subdir, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check after lock acquisition: a sibling rank may have
            # finished while we were waiting.
            if os.path.isfile(sentinel) and os.path.isfile(staged_path):
                log.info(
                    "[ckpt-cache] another process finished staging while "
                    f"we waited; using {staged_path}"
                )
            else:
                try:
                    if copy_siblings:
                        for entry in os.scandir(src_dir):
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            shutil.copy2(entry.path, os.path.join(cache_subdir, entry.name))
                    else:
                        shutil.copy2(path, staged_path)
                except OSError as e:
                    log.warning(
                        f"[ckpt-cache] copy failed ({e}); falling back to source path {path}"
                    )
                    return path
                if not os.path.isfile(staged_path):
                    log.warning(
                        f"[ckpt-cache] expected staged file {staged_path} "
                        f"missing after copy; falling back to {path}"
                    )
                    return path
                with open(sentinel, "w") as f:
                    f.write(f"staged {src_dir} -> {cache_subdir}\n")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    dt = time.perf_counter() - _t
    try:
        sz = os.path.getsize(staged_path) / (1024**3)
        rate = f", {sz * 1024 / dt:.0f} MiB/s" if dt > 0 else ""
        log.info(f"[ckpt-cache] staged {sz:.1f} GiB in {dt:.1f}s{rate} -> {staged_path}")
    except OSError:
        log.info(f"[ckpt-cache] staged in {dt:.1f}s -> {staged_path}")

    _evict_ckpt_cache_lru(logger=log, keep_subdir=cache_subdir)
    return staged_path


def load_ckpt(
    path: str,
    *,
    logger=None,
    label: str = "ckpt",
    map_location: str = "cpu",
) -> dict:
    """Stage to local NVMe (if remote) and ``torch.load`` with mmap fallback.

    Convenience wrapper for callers that want the full inference-style
    fast path:

    - ``mmap=True`` so tensor storages are demand-paged from the file
      rather than copied into Python memory up-front.
    - ``weights_only=True`` skips arbitrary pickle execution. Falls back
      to ``weights_only=False`` if the file contains non-tensor objects
      (older checkpoints occasionally do), then to a non-mmap full-read
      if mmap fails for any reason.

    Callers that need to control the load flags themselves (e.g. trainer
    code that wants a full-read for predictable timing) should call
    :func:`stage_ckpt_to_local` directly and run their own ``torch.load``.
    """
    log = _get_logger(logger)

    path = stage_ckpt_to_local(path, logger=log, copy_siblings=False)

    file_size_mb: float | None = None
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        pass

    size_str = f" ({file_size_mb:.1f} MiB)" if file_size_mb is not None else ""
    log.info(f"[ckpt] Loading {label} from {path}{size_str}")
    t0 = time.perf_counter()

    try:
        blob = torch.load(path, map_location=map_location, mmap=True, weights_only=True)
        mode = "mmap+weights_only"
    except Exception as e_safe:  # noqa: BLE001
        log.warning(
            f"[ckpt] weights_only=True load failed ({type(e_safe).__name__}: "
            f"{e_safe}); retrying without it."
        )
        try:
            blob = torch.load(path, map_location=map_location, mmap=True, weights_only=False)
            mode = "mmap"
        except Exception as e_mmap:  # noqa: BLE001
            log.warning(
                f"[ckpt] mmap=True load failed ({type(e_mmap).__name__}: "
                f"{e_mmap}); retrying with full read."
            )
            blob = torch.load(path, map_location=map_location, weights_only=False)
            mode = "full-read"

    dt = time.perf_counter() - t0
    rate = f", {file_size_mb / dt:.1f} MiB/s" if file_size_mb is not None and dt > 0 else ""
    log.info(f"[ckpt] {label} loaded in {dt:.2f}s [{mode}]{rate}")
    return blob


# --- Internal: LRU eviction ----------------------------------------------


def _evict_ckpt_cache_lru(*, logger, keep_subdir: str | None = None) -> None:
    """LRU-evict staged ckpt dirs until total size <= budget.

    Order is by ``.stage_done`` mtime, oldest first. ``keep_subdir`` is
    never evicted (typically the one we just staged). Best-effort: any
    I/O error during scanning or removal is logged and ignored.
    """
    try:
        entries: list[tuple[float, str, int]] = []  # (mtime, dir, bytes)
        with os.scandir(_CKPT_LOCAL_CACHE_DIR) as it:
            for de in it:
                if not de.is_dir(follow_symlinks=False):
                    continue
                sub = de.path
                sentinel = os.path.join(sub, ".stage_done")
                if not os.path.isfile(sentinel):
                    continue  # in-progress or unrelated; skip
                try:
                    mt = os.path.getmtime(sentinel)
                except OSError:
                    continue
                total = 0
                for root, _, files in os.walk(sub):
                    for fn in files:
                        try:
                            total += os.path.getsize(os.path.join(root, fn))
                        except OSError:
                            pass
                entries.append((mt, sub, total))
    except FileNotFoundError:
        return

    total_bytes = sum(e[2] for e in entries)
    if total_bytes <= _CKPT_LOCAL_CACHE_BUDGET_BYTES:
        return

    import shutil as _shutil

    entries.sort(key=lambda e: e[0])  # oldest first
    budget = _CKPT_LOCAL_CACHE_BUDGET_BYTES
    over = total_bytes - budget
    logger.info(
        f"[ckpt-cache] LRU eviction: {total_bytes / 1024**3:.1f} GiB > "
        f"{budget / 1024**3:.1f} GiB budget, freeing "
        f"{over / 1024**3:.1f} GiB"
    )
    for _mt, sub, sz in entries:
        if total_bytes <= budget:
            break
        if keep_subdir is not None and os.path.realpath(sub) == os.path.realpath(keep_subdir):
            continue
        try:
            _shutil.rmtree(sub)
            total_bytes -= sz
            logger.info(f"[ckpt-cache]   evicted {sub} ({sz / 1024**3:.1f} GiB)")
        except OSError as e:
            logger.warning(f"[ckpt-cache]   failed to evict {sub}: {e}")

"""Launcher log preparation uses the same final training configuration as train.py."""

import os
import tempfile
from pathlib import Path

from src.utils.config import resolve_config


def prepare_launcher_logs(argv=None, *, node_rank=0, world_size=1):
    resolved = resolve_config(argv, training=True, world_size=world_size)
    if type(node_rank) is not int or node_rank < 0:
        raise ValueError("RANK must be a nonnegative node rank")
    root = Path(resolved.public.expdir) / "logs" / "torchrun" / f"node_{node_rank}"
    root.mkdir(parents=True, exist_ok=True)
    return str(Path(tempfile.mkdtemp(prefix="run_", dir=root)).absolute())


if __name__ == "__main__":
    # run.sh supplies process count to this one helper invocation. Its parent
    # WORLD_SIZE remains the node count and torchrun owns worker WORLD_SIZE.
    print(
        prepare_launcher_logs(
            node_rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    )

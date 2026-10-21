"""Node-local startup snapshots and inference output naming (no tensor loading)."""

import logging
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


def setup_startup_logging(config, *, training):
    """Append one local file per global rank; the trainer records successful restore."""
    if training:
        directory = Path(config.log_dir)
    else:
        output = Path(config.output_path)
        directory = output.parent / "logs" / output.name
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0")) if "LOCAL_RANK" in os.environ else 0
    logger = logging.getLogger("looking_ahead")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()
    for handler in (
        logging.FileHandler(directory / f"rank_{rank}.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        root_logger.addHandler(handler)
    logger.info(
        "Starting session config=%s resume_source=%s",
        config.config_name,
        config.resume_checkpoint_path if training else None,
    )
    return logger


def configure_debug_environment():
    if os.environ.get("DEBUG") == "1":
        os.environ["WAN_VIS_CAUSAL_MASK"] = "1"
    else:
        os.environ.setdefault("WAN_VIS_CAUSAL_MASK", "0")


def publish_config(config, destination: Path) -> None:
    """LOCAL_RANK=0 publishes a closed temporary YAML by same-directory replacement."""
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix=".startup-",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            OmegaConf.save(config, stream, resolve=True)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_inference_paths(config) -> None:
    """Resolve inference paths without loading or validating checkpoint contents."""
    checkpoint = config.checkpoint_path
    if checkpoint is not None:
        checkpoint_name = Path(checkpoint).parent.name or "checkpoint_model_init"
    else:
        root = Path(config.expdir) / "checkpoints"
        if config.checkpoint_name is not None:
            checkpoint_name = config.checkpoint_name
            checkpoint = str(root / checkpoint_name / "model.pt")
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(
                    f"checkpoint_name: specified checkpoint not found: {checkpoint}"
                )
        else:
            candidates = sorted(root.glob("checkpoint_model_*")) if root.is_dir() else []
            checkpoint = str(candidates[-1] / "model.pt") if candidates else None
            checkpoint_name = candidates[-1].name if candidates else "checkpoint_model_init"
    if checkpoint is None and config.use_ema:
        raise FileNotFoundError("use_ema: no training checkpoint was requested or discovered")
    if checkpoint is None:
        print("[inference] No training checkpoint loaded; retaining pipeline initial weights.")
    config.checkpoint_path = checkpoint
    if config.output_path is not None:
        return
    data_path = config.get("image_path") or config.get("video_path") or config.data_path
    if config.use_train_data:
        data_path = next(
            source.data_path
            for source in config.data_sources
            if source.id == config.inference_train_data_source_id
        )
    suffix = ""
    if config.model_kwargs.sink_size > 0:
        suffix += f"-sink={config.model_kwargs.sink_size}"
    if config.use_ema:
        suffix += "-ema"
    if config.use_train_data:
        suffix += "-train_data"
    if config.get("v2v", False):
        context = config.v2v_context_latents
        suffix += f"-v2v-k={context if context is not None else 'auto'}"
    if config.i2v:
        suffix += "-i2v"
    if config.get("image_path") is not None:
        suffix += "-single_image"
    config.output_path = str(
        Path(config.expdir)
        / "inference"
        / f"{checkpoint_name}-{Path(data_path).stem}"
        / f"test_{config.duration}{suffix}"
    )

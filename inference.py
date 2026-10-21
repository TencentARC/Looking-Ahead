"""Text + supplied image/video-prefix ordinary inference entry."""

from pathlib import Path

from omegaconf import OmegaConf

from src.utils.config import resolve_config
from src.utils.startup import (
    configure_debug_environment,
    publish_config,
    resolve_inference_paths,
    setup_startup_logging,
)


def parse_config(argv=None, *, action=None):
    return resolve_config(argv, training=False, action=action)


def run_inference(resolved):
    from src.pipeline.inference_runtime import run_inference as run

    run(resolved.public)


def main(argv=None, *, action=None):
    resolved = parse_config(argv, action=action)
    if resolved.dry_run:
        print(OmegaConf.to_yaml(resolved.public))
        return
    config = resolved.public
    resolve_inference_paths(config)
    output = Path(config.output_path)
    publish_config(config, output.parent / "configs" / f"{output.name}.yaml")
    configure_debug_environment()
    setup_startup_logging(config, training=False)
    run_inference(resolved)


if __name__ == "__main__":
    main()

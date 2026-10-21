"""TF/FCP training with ordinary checkpoints and strict full-state resume."""

from pathlib import Path

from omegaconf import OmegaConf

from src.utils.config import resolve_config
from src.utils.startup import configure_debug_environment, publish_config, setup_startup_logging


def parse_config(argv=None):
    return resolve_config(argv, training=True)


def run_training(resolved):
    from src.trainer.diffusion import DiffusionTrainer

    trainer = DiffusionTrainer(resolved.public)
    trainer.train()
    import wandb

    wandb.finish()


def main(argv=None):
    resolved = parse_config(argv)
    if resolved.dry_run:
        print(OmegaConf.to_yaml(resolved.public))
        return
    publish_config(resolved.public, Path(resolved.public.expdir) / "configs/training_config.yaml")
    configure_debug_environment()
    setup_startup_logging(resolved.public, training=True)
    run_training(resolved)


if __name__ == "__main__":
    main()

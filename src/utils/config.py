"""Shared YAML/dotlist/flag resolution before any runtime side effects."""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.cli import PROJECT_ROOT, derive_expdir, resolve_config_path
from src.utils.config_schema import validate_names
from src.utils.config_validation import validate_config


@dataclass(frozen=True)
class ResolvedConfig:
    public: DictConfig
    config_path: Path
    dry_run: bool

    @property
    def model_cls(self) -> str:
        return "PredictiveFlowMatching" if self.public.future_prediction.enabled else "FlowMatching"

    @property
    def model_internal_kwargs(self) -> dict:
        return {
            "use_predictive_causal_model": self.public.future_prediction.enabled,
            "use_infinite_rope": self.public.model_name.startswith("Wan2.1-"),
        }


def build_parser(*, training: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Looking-Ahead training"
        if training
        else "Looking-Ahead text + image/video-prefix inference",
        allow_abbrev=False,
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config_path",
        "--config",
        required=True,
        help="Config stem under configs/, project-relative path, or absolute path",
    )
    parser.add_argument("--exp_dir", "--expdir", dest="expdir")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and print the final public config; no snapshots/model/data/GPU",
    )
    if training:
        parser.add_argument("--no_save", action="store_true")
        parser.add_argument("--no_visualize", action="store_true")
        parser.add_argument("--disable_wandb", "--disable-wandb", action="store_true")
        parser.add_argument("--wandb_save_dir", "--wandb-save-dir")
    else:
        parser.add_argument("--checkpoint_path", "--ckpt")
        parser.add_argument("--checkpoint_name", "--ckpt_name")
        parser.add_argument("--output_path")
        parser.add_argument("--data_path", "--data", help="External-image metadata directory")
        parser.add_argument(
            "--image_path",
            help="One first-frame image; requires --prompt or --prompt_file, no JSON",
        )
        prompts = parser.add_mutually_exclusive_group()
        prompts.add_argument(
            "--prompt", help="Text and motion description for --image_path or --video_path"
        )
        prompts.add_argument(
            "--prompt_file", help="UTF-8 .txt prompt for --image_path or --video_path"
        )
        parser.add_argument("--duration", help="Positive whole duration, e.g. 5s, 1m, 1h")
        parser.add_argument("--num_output_frames", type=int)
        parser.add_argument(
            "--video_path", help="One video prefix; requires a prompt and context length"
        )
        parser.add_argument(
            "--v2v", action="store_true", help="Continue a clean video-latent prefix"
        )
        parser.add_argument(
            "--v2v_context_latents",
            type=int,
            help="Context length in latent frames; omit with --video_path to select the longest valid tail",
        )
        parser.add_argument("--seed", type=int)
        parser.add_argument("--test_sample_limit", type=int)
        parser.add_argument(
            "--i2v",
            action="store_true",
            help="Supply text + C0 for metadata/latent input; implicit with --image_path",
        )
        parser.add_argument(
            "--use_train_data",
            action="store_true",
            help="Use the first clean latent frame of one configured finite source",
        )
        parser.add_argument(
            "--train_data_source_id",
            "--train-data-source-id",
            dest="inference_train_data_source_id",
        )
        parser.add_argument(
            "--sink_size",
            "--sink",
            dest="model_kwargs.sink_size",
            type=int,
            help="Retained prefix length in latent frames",
        )
        weights = parser.add_mutually_exclusive_group()
        weights.add_argument("--use_ema", "--ema", dest="use_ema", action="store_true")
        weights.add_argument("--no_ema", dest="use_ema", action="store_false")
    return parser


def _layer(config, label):
    data = OmegaConf.to_container(config, resolve=False)
    try:
        validate_names(data)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error
    return config


def _dotlist(tokens):
    layers = []
    for token in tokens:
        key, separator, _ = token.partition("=")
        if not separator or not re.fullmatch(
            r"[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)*", key
        ):
            raise ValueError(f"Invalid argument {token!r}; use a registered flag or key=value")
        # Each occurrence is checked, even if the same key is overridden later.
        layers.append(_layer(OmegaConf.from_dotlist([token]), token))
    return layers


def resolve_config(
    argv=None, *, training: bool, action: str | None = None, world_size: int | None = None
) -> ResolvedConfig:
    parser = build_parser(training=training)
    args, tokens = parser.parse_known_args(argv)
    flags = vars(args).copy()
    path = resolve_config_path(flags.pop("config_path"))
    dry_run = flags.pop("dry_run", False)
    defaults = _layer(OmegaConf.load(PROJECT_ROOT / "configs/default_config.yaml"), "defaults")
    recipe = _layer(OmegaConf.load(path), str(path))
    action_defaults = (
        {}
        if training
        else {
            "seed": 42,
            "test_sample_limit": 32 if action == "test" else -1,
            "image_path": None,
            "prompt": None,
            "prompt_file": None,
            "video_path": None,
            "v2v": False,
            "v2v_context_latents": None,
            "model_kwargs": {"sink_size": 3},
        }
    )
    explicit = OmegaConf.create({})
    for key, value in flags.items():
        OmegaConf.update(explicit, key, value, merge=True)
    layers = [defaults, OmegaConf.create(action_defaults), recipe, *_dotlist(tokens), explicit]
    merged = OmegaConf.merge(*layers)
    data = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    validate_names(data)
    if world_size is None:
        # Parent WORLD_SIZE denotes nodes; only worker WORLD_SIZE denotes processes.
        world_size = int(os.environ.get("WORLD_SIZE", "1")) if "LOCAL_RANK" in os.environ else 1
    validate_config(data, training=training, world_size=world_size)
    if (
        not training
        and not data["use_train_data"]
        and (action is not None or "LOCAL_RANK" in os.environ)
    ):
        raise ValueError(
            "i2v: external-image input requires direct non-distributed Python execution"
        )
    if training and os.environ.get("DEBUG") == "1":
        data["disable_wandb"] = True
    data["expdir"] = data["expdir"] if data["expdir"] is not None else derive_expdir(path)
    if data["wandb_save_dir"] is None:
        data["wandb_save_dir"] = str(Path(data["expdir"]) / "wandb")
    data["config_name"] = path.stem
    if training:
        for key, child in (
            ("checkpoint_dir", "checkpoints"),
            ("snapshot_dir", "snapshots"),
            ("log_dir", "logs"),
        ):
            data[key] = str(Path(data["expdir"]) / child)
    public = OmegaConf.create(data)
    OmegaConf.set_struct(public, True)
    return ResolvedConfig(public, path, dry_run)

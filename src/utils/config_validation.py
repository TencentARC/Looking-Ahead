"""Configuration-only validation for TF/FCP workflows.

No dataset, checkpoint, model, distributed group or CUDA is initialized here.
The narrow auxiliary range check uses a CPU timestep table to preserve the scheduler's
float32 endpoint semantics; the usual full range needs no tensor imports.
"""

import math
import re

# channels, backbone blocks, hidden width, FPS; these are family facts, not recipes.
FAMILIES = {
    "Wan2.1-T2V-1.3B": (16, 30, 1536, 16),
    "Wan2.1-T2V-14B": (16, 40, 5120, 16),
    "Wan2.2-TI2V-5B": (48, 30, 3072, 24),
}


def fail(field, rule):
    raise ValueError(f"{field}: {rule}")


def integer(value, field, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        fail(
            field,
            f"must be a non-boolean integer >= {minimum}"
            if minimum is not None
            else "must be a non-boolean integer",
        )
    return value


def number(value, field, minimum=None, maximum=None, *, strict_min=False, strict_max=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(field, "must be a finite non-boolean number")
    if minimum is not None and (value < minimum or (strict_min and value == minimum)):
        fail(field, f"must be {'>' if strict_min else '>='} {minimum}")
    if maximum is not None and (value > maximum or (strict_max and value == maximum)):
        fail(field, f"must be {'<' if strict_max else '<='} {maximum}")
    return value


def string(value, field, *, nullable=False, empty=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or (not empty and not value.strip()):
        fail(field, "must be a string" if empty else "must be a nonempty string")


def choice(value, field, choices):
    if value not in choices:
        fail(field, f"must be one of {', '.join(map(str, choices))}")


def items(value, field, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        fail(field, "must be a nonempty list" if nonempty else "must be a list")
    return value


def increasing(values, field, *, minimum=0, nonempty=False):
    items(values, field, nonempty=nonempty)
    for value in values:
        integer(value, field, minimum)
    if any(left >= right for left, right in zip(values, values[1:], strict=False)):
        fail(field, "must be strictly increasing with no duplicates")


def validate_sources(config, *, required):
    sources = config.get("data_sources")
    if sources is None and not required:
        return
    items(sources, "data_sources", nonempty=True)
    seen = set()
    for index, source in enumerate(sources):
        prefix = f"data_sources[{index}]"
        source_id = source.get("id", f"source_{index}")
        string(source_id, f"{prefix}.id")
        source_id = source_id.strip()
        if source_id in seen:
            fail(f"{prefix}.id", f"duplicate source ID {source_id!r}")
        seen.add(source_id)
        source["id"] = source_id
        for key in ("data_path", "annotation_path"):
            string(source.get(key), f"{prefix}.{key}")
        if not source["data_path"].rstrip("/\\").lower().endswith("shards"):
            fail(f"{prefix}.data_path", "must name a directory ending in 'shards'")
        number(source.get("weight"), f"{prefix}.weight", 0, strict_min=True)
    if required:
        for key in ("data_path", "annotation_path"):
            if config.get(key) is not None:
                fail(key, "must be null/omitted when using data_sources")


def validate_auxiliary_range(bounds, shift):
    field = "future_prediction.aux_future_timestep_range"
    items(bounds, field)
    if len(bounds) != 2:
        fail(field, "requires two finite nonnegative endpoints")
    lower, upper = [number(value, field, 0) for value in bounds]
    if lower > upper:
        fail(field, "lower endpoint must not exceed upper endpoint")
    # Every retained training schedule starts at 1000. Avoid importing torch for
    # the reference interval; a narrower interval needs the exact CPU arithmetic.
    if lower <= 1000 <= upper:
        return
    import torch

    sigmas = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32, device="cpu")[:-1]
    timesteps = (shift * sigmas / (1 + (shift - 1) * sigmas)) * 1000
    if not ((timesteps >= lower) & (timesteps <= upper)).any().item():
        fail(field, "closed interval selects no actual training scheduler timestep")


def validate_future_prediction(config, family, chunks):
    fp = config["future_prediction"]
    prefix = "future_prediction."
    if type(fp["enabled"]) is not bool:
        fail(prefix + "enabled", "must be boolean")
    if fp["enabled"]:
        for field in ("future_offsets", "offset_loss_weights", "source_layers", "mini_dit_depth"):
            if field not in fp:
                fail(
                    prefix + field, "is required for FCP; method changes do not construct a recipe"
                )
    offsets = fp.get("future_offsets")
    if "future_offsets" in fp:
        increasing(offsets, prefix + "future_offsets", minimum=1, nonempty=True)
        if offsets[-1] >= chunks:
            fail(prefix + "future_offsets", f"every offset must be below {chunks} regular chunks")
    if "offset_loss_weights" in fp:
        weights = items(fp["offset_loss_weights"], prefix + "offset_loss_weights", nonempty=True)
        if offsets is not None and len(weights) != len(offsets):
            fail(prefix + "offset_loss_weights", "requires one weight per offset")
        for weight in weights:
            number(weight, prefix + "offset_loss_weights", 0)
        if not any(weight > 0 for weight in weights):
            fail(prefix + "offset_loss_weights", "at least one weight must be positive")
    layers = fp.get("source_layers")
    if "source_layers" in fp:
        increasing(layers, prefix + "source_layers", nonempty=True)
        if layers[-1] >= family[1]:
            fail(prefix + "source_layers", f"indices must be below {family[1]}")
    depth = None
    if "mini_dit_depth" in fp:
        depth = integer(fp["mini_dit_depth"], prefix + "mini_dit_depth", 1)
    mode = fp["init_mode"]
    choice(mode, prefix + "init_mode", ("copy", "random"))
    if mode == "random":
        if "init_source" in fp:
            fail(
                prefix + "init_source", "must be absent for random initialization (null is invalid)"
            )
    elif fp["enabled"] or "init_source" in fp:
        if depth is None:
            fail(prefix + "mini_dit_depth", "is required with init_source")
        if "init_source" not in fp:
            if len(layers) != 1:
                fail(prefix + "init_source", "is required for multi-layer copy initialization")
            indices = list(range(layers[0] + 1, layers[0] + 1 + depth))
        else:
            indices = items(fp["init_source"], prefix + "init_source", nonempty=True)
            if len(indices) == 1 and isinstance(indices[0], str):
                match = re.fullmatch(r"\s*(-?\d+)\s*:\s*(-?\d+)\s*", indices[0])
                if not match:
                    fail(prefix + "init_source", "range must use start:end (end exclusive)")
                start, end = map(int, match.groups())
                if not 0 <= start < end <= family[1]:
                    fail(prefix + "init_source", "range must lie inside backbone blocks")
                indices = list(range(start, end))
        for index in indices:
            integer(index, prefix + "init_source", 0)
            if index >= family[1]:
                fail(prefix + "init_source", f"indices must be below {family[1]}")
        if len(indices) != depth or len(set(indices)) != len(indices):
            fail(prefix + "init_source", "distinct index count must equal mini_dit_depth")
        fp["init_source"] = indices
    fusion = fp["source_layer_fusion"]
    choice(fusion["mode"], prefix + "source_layer_fusion.mode", ("mlp",))
    choice(fusion["activation"], prefix + "source_layer_fusion.activation", ("gelu",))
    ratio = number(
        fusion["hidden_ratio"], prefix + "source_layer_fusion.hidden_ratio", 0, strict_min=True
    )
    if not math.isfinite(ratio * family[2]) or round(ratio * family[2]) <= 0:
        fail(
            prefix + "source_layer_fusion.hidden_ratio", "must produce a positive finite MLP width"
        )
    streams = fp["source_context"].get("streams")
    if streams is None:
        streams = ["noisy"]
    items(streams, prefix + "source_context.streams", nonempty=True)
    if any(not isinstance(value, str) or value not in ("clean", "noisy") for value in streams):
        fail(prefix + "source_context.streams", "only clean and noisy are supported")
    if len(set(streams)) != len(streams):
        fail(prefix + "source_context.streams", "duplicates are not allowed")
    fp["source_context"]["streams"] = [name for name in ("clean", "noisy") if name in streams]
    shift = fp.get("aux_future_timestep_shift", config["model_kwargs"]["timestep_shift"])
    number(shift, prefix + "aux_future_timestep_shift", 0, strict_min=True)
    validate_auxiliary_range(fp["aux_future_timestep_range"], shift)


def validate_config(config: dict, *, training: bool, world_size: int = 1) -> None:
    """Validate and normalize one final merged public config before side effects."""
    for key in (
        "image_or_video_shape",
        "num_frame_per_block",
        "ema_weight",
        "ema_start_step",
        "mixed_precision",
        "gradient_checkpointing",
        "sharding_strategy",
        "log_iters",
        "num_workers",
        "guidance_scale",
    ):
        if key not in config:
            fail(key, "is required; select a complete training recipe")
    bool_fields = """load_diffusion_pretrained generator_fsdp_sync_module_states
    independent_first_frame condition_only_first_frame i2v mixed_precision gradient_checkpointing
    separate_predictive_grad_clip save_resume_checkpoint require_raw_generator_checkpoint
    no_save no_visualize disable_wandb use_ema use_train_data""".split()
    for key in bool_fields:
        if type(config[key]) is not bool:
            fail(key, "must be boolean")
    model = config["model_name"]
    if not isinstance(model, str) or model not in FAMILIES:
        fail("model_name", f"must be one of {', '.join(FAMILIES)}")
    family = FAMILIES[model]
    positive_ints = """batch_size val_batch_size log_iters gc_interval
    timestep_bucket_log_interval num_frame_per_block sampling_steps
    long_inference_threshold vae_decode_chunk""".split()
    for key in positive_ints:
        integer(config[key], key, 1)
    for key in (
        "num_workers",
        "val_num_workers",
        "lr_warmup_steps",
        "ema_start_step",
        "wds_sample_shuffle",
        "wds_sample_shuffle_initial",
    ):
        integer(config[key], key, 0)
    for key in ("max_iters", "snapshot_iters", "num_output_frames"):
        value = integer(config[key], key)
        if value != -1 and value <= 0:
            fail(key, "must be positive or -1")
    integer(config["test_sample_limit"], "test_sample_limit", -1)
    if config["max_checkpoints"] is not None:
        integer(config["max_checkpoints"], "max_checkpoints")
    seed = integer(config["seed"], "seed", 0)
    integer(world_size, "world_size", 1)
    if seed > 2**32 - 1 or (training and (seed == 0 or seed + world_size - 1 > 2**32 - 1)):
        fail("seed", "must fit NumPy uint32 for every rank; training also requires nonzero")
    string(config["optimizer"], "optimizer")
    config["optimizer"] = config["optimizer"].strip().lower()
    choice(config["optimizer"], "optimizer", ("adamw",))
    choice(config["ema_checkpoint_mode"], "ema_checkpoint_mode", ("default",))
    choice(config["sample_solver"], "sample_solver", ("unipc", "dpm++"))
    choice(
        config["sharding_strategy"],
        "sharding_strategy",
        ("full", "hybrid_full", "hybrid_zero2", "no_shard"),
    )
    for key in ("generator_fsdp_wrap_strategy", "text_encoder_fsdp_wrap_strategy"):
        if key in config:
            choice(config[key], key, ("size",))
    adamw = config["optimizer_config"]["adamw"]
    for key in ("lr", "weight_decay"):
        number(adamw[key], f"optimizer_config.adamw.{key}", 0)
    for key in ("beta1", "beta2"):
        number(adamw[key], f"optimizer_config.adamw.{key}", 0, 1, strict_max=True)
    increasing(config["lr_drop_steps"], "lr_drop_steps", minimum=1)
    if any(step <= config["lr_warmup_steps"] for step in config["lr_drop_steps"]):
        fail("lr_drop_steps", "each node must be after lr_warmup_steps")
    number(config["lr_drop_gamma"], "lr_drop_gamma", 0, 1, strict_min=True, strict_max=True)
    number(config["ema_weight"], "ema_weight", 0, 1, strict_max=True)
    number(config["caption_dropout_p"], "caption_dropout_p", 0, 1)
    number(config["guidance_scale"], "guidance_scale", 0)
    number(config["max_grad_norm"], "max_grad_norm", 0, strict_min=True)
    number(config["timestep_shift"], "timestep_shift", 0, strict_min=True)
    number(
        config["model_kwargs"]["timestep_shift"], "model_kwargs.timestep_shift", 0, strict_min=True
    )
    string(config["negative_prompt"], "negative_prompt", empty=True)
    for key in ("wandb_host", "wandb_project"):
        string(config[key], key)
    for key in (
        "wandb_entity",
        "wandb_save_dir",
        "expdir",
        "checkpoint_path",
        "generator_ckpt",
        "resume_checkpoint_path",
        "data_path",
        "annotation_path",
        "output_path",
        "checkpoint_name",
        "inference_train_data_source_id",
    ):
        string(config[key], key, nullable=True)
    shape = items(config["image_or_video_shape"], "image_or_video_shape")
    if len(shape) != 5:
        fail("image_or_video_shape", "must be [B,T,C,H,W]")
    for value in shape:
        integer(value, "image_or_video_shape", 1)
    _, frames, channels, height, width = shape
    if channels != family[0]:
        fail("image_or_video_shape", f"{model} requires C={family[0]}")
    if height % 2 or width % 2:
        fail("image_or_video_shape", "H and W must be divisible by the spatial patch size 2")
    independent = config["independent_first_frame"]
    if config["condition_only_first_frame"] and not independent:
        fail("condition_only_first_frame", "requires independent_first_frame=true")
    regular = frames - int(independent)
    block = config["num_frame_per_block"]
    if regular <= 0 or regular % block:
        fail("image_or_video_shape", "T must contain positive whole regular chunks after C0")
    fps = config.get("fps", family[3])
    number(fps, "fps", 0, strict_min=True)
    if int(fps) != fps:
        fail("fps", "must be integer-valued")
    kwargs = config["model_kwargs"]
    window = integer(kwargs["local_attn_size"], "model_kwargs.local_attn_size")
    if window == -1:
        window = frames
    elif window <= 0:
        fail("model_kwargs.local_attn_size", "must be positive or -1 (automatic)")
    sink = integer(kwargs["sink_size"], "model_kwargs.sink_size", 0)
    if sink + block > window:
        fail("model_kwargs.local_attn_size", "must fit sink_size plus an incoming regular chunk")
    validate_future_prediction(config, family, regular // block)
    predictive = config["future_prediction"]["enabled"]
    separate = config["separate_predictive_grad_clip"]
    if separate and not predictive:
        fail("separate_predictive_grad_clip", "requires future_prediction.enabled=true")
    if separate:
        number(config["max_predictive_grad_norm"], "max_predictive_grad_norm", 0, strict_min=True)
    elif config["max_predictive_grad_norm"] is not None:
        fail("max_predictive_grad_norm", "must be null/absent when separate clipping is off")
    if config["save_resume_checkpoint"] and config["no_save"]:
        fail("save_resume_checkpoint", "cannot be enabled with no_save=true")
    single_image = config.get("image_path") is not None
    if training:
        for key in (
            "image_path",
            "video_path",
            "prompt",
            "prompt_file",
            "v2v",
            "v2v_context_latents",
        ):
            if key in config:
                fail(key, "is an inference-only input")
    else:
        string(config.get("image_path"), "image_path", nullable=True)
        string(config.get("prompt"), "prompt", nullable=True)
        string(config.get("prompt_file"), "prompt_file", nullable=True)
        string(config.get("video_path"), "video_path", nullable=True)
        single_video = config.get("video_path") is not None
        if single_image and single_video:
            fail("video_path", "cannot combine with image_path")
        if type(config["v2v"]) is not bool:
            fail("v2v", "must be boolean")
        if single_video:
            config["v2v"] = True
        v2v = config["v2v"]
        if v2v and config["i2v"]:
            fail("v2v", "cannot combine with i2v")
        if v2v and single_image:
            fail("v2v", "cannot combine with image_path")
        if v2v and not (single_video or config["use_train_data"]):
            fail("v2v", "requires video_path or use_train_data=true")
        if single_video:
            if config["data_path"] is not None or config["use_train_data"]:
                fail("video_path", "cannot combine with data_path or use_train_data")
            if config["inference_train_data_source_id"] is not None:
                fail("video_path", "cannot combine with train_data_source_id")
        if not v2v and config["v2v_context_latents"] is not None:
            fail("v2v_context_latents", "requires v2v=true or video_path")
        inline_prompt = config.get("prompt") is not None
        file_prompt = config.get("prompt_file") is not None
        if inline_prompt and file_prompt:
            fail("prompt/prompt_file", "are mutually exclusive; choose only one")
        if file_prompt and not config["prompt_file"].lower().endswith(".txt"):
            fail("prompt_file", "must name a .txt file")
        if (single_image or single_video) != (inline_prompt or file_prompt):
            fail(
                "image_path/--prompt",
                "provide both --image_path or --video_path and either --prompt or --prompt_file",
            )
        if single_image:
            if config["data_path"] is not None:
                fail("image_path", "cannot combine with data_path; select one input route")
            if config["use_train_data"] or config["inference_train_data_source_id"] is not None:
                fail("image_path", "cannot combine with use_train_data or train_data_source_id")
            # A single supplied first frame selects TI2V without a redundant --i2v flag.
            config["i2v"] = True
    validate_sources(config, required=training or config["use_train_data"])
    duration = config["duration"]
    string(duration, "duration")
    if config["num_output_frames"] == -1 and not re.fullmatch(r"[1-9][0-9]*[smhSMH]", duration):
        fail("duration", "must be a positive whole duration with s/m/h units")
    if training:
        return
    if not config["i2v"] and not config["v2v"]:
        fail("i2v", "ordinary inference requires text plus supplied C0; enable i2v")
    if not config["v2v"] and not independent and block != 1:
        fail(
            "num_frame_per_block",
            "single-C0 inference without an independent first frame requires 1",
        )
    if config["v2v"]:
        from src.utils.video_prefix import prefix_geometry

        if single_video and config["v2v_context_latents"] is None:
            # Metadata-dependent tail selection happens before GPU/model loading.
            minimum = 1 + block if independent else block
            prefix_geometry({**config, "v2v_context_latents": minimum})
        else:
            prefix_geometry(config)
    source_id = config["inference_train_data_source_id"]
    if config["use_train_data"]:
        sources = config["data_sources"]
        if source_id is None and len(sources) == 1:
            source_id = sources[0]["id"]
        if source_id not in {source["id"] for source in sources}:
            fail("inference_train_data_source_id", "select one configured source for inference")
        config["inference_train_data_source_id"] = source_id
    else:
        if source_id is not None:
            fail("inference_train_data_source_id", "requires use_train_data=true")
        if not single_image and not single_video:
            string(config["data_path"], "data_path")
        if world_size != 1:
            fail("i2v", "external-image input requires non-distributed Python execution")

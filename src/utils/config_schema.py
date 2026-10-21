"""Public input boundary for the selected TF/FCP workflows.

Only names belong here: validate every input layer before merging, so a later
override cannot hide a removed control. Value/combination checks run on the final
configuration. Lists are fields, never dotlist-indexable containers.
"""

SCHEMA = dict.fromkeys(
    """
model_name load_diffusion_pretrained generator_fsdp_sync_module_states
independent_first_frame condition_only_first_frame optimizer lr_warmup_steps
lr_drop_steps lr_drop_gamma i2v gc_interval timestep_bucket_log_interval seed
wds_sample_shuffle wds_sample_shuffle_initial inference_train_data_source_id
max_iters snapshot_iters checkpoint_path resume_checkpoint_path save_resume_checkpoint
batch_size val_batch_size test_sample_limit caption_dropout_p negative_prompt
max_grad_norm separate_predictive_grad_clip max_predictive_grad_norm max_checkpoints
ema_checkpoint_mode ema_weight ema_start_step timestep_shift guidance_scale
mixed_precision wandb_host wandb_entity wandb_project sharding_strategy log_iters
num_workers image_or_video_shape gradient_checkpointing num_frame_per_block
data_path annotation_path fps generator_ckpt require_raw_generator_checkpoint
sampling_steps sample_solver val_num_workers no_save no_visualize disable_wandb
expdir wandb_save_dir checkpoint_name output_path duration num_output_frames
use_ema use_train_data long_inference_threshold vae_decode_chunk image_path prompt prompt_file
v2v v2v_context_latents video_path
""".split()
)
SCHEMA.update(
    {
        "optimizer_config": {"adamw": dict.fromkeys(["lr", "beta1", "beta2", "weight_decay"])},
        "model_kwargs": dict.fromkeys(["timestep_shift", "sink_size", "local_attn_size"]),
        "future_prediction": {
            **dict.fromkeys(
                """enabled future_offsets offset_loss_weights source_layers
        mini_dit_depth init_mode init_source aux_future_timestep_range
        aux_future_timestep_shift""".split()
            ),
            "source_context": {"streams": None},
            "source_layer_fusion": dict.fromkeys(["mode", "activation", "hidden_ratio"]),
        },
        "data_sources": [dict.fromkeys(["id", "data_path", "annotation_path", "weight"])],
        # Public FSDP wrapping strategies remain configurable.
        "generator_fsdp_wrap_strategy": None,
        "text_encoder_fsdp_wrap_strategy": None,
    }
)


def validate_names(data: dict, schema: dict = SCHEMA, prefix: str = "") -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            raise ValueError(f"{path}: unknown, removed, fixed or derived public field")
        child = schema[key]
        if isinstance(child, dict):
            validate_names(value, child, path)
        elif isinstance(child, list):
            if value is None:
                continue  # A null default is allowed; active data routes require a list.
            if not isinstance(value, list):
                raise ValueError(f"{path} must be a list of mappings")
            for index, entry in enumerate(value):
                validate_names(entry, child[0], f"{path}[{index}]")

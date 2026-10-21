# Inference

This guide covers text-and-image-to-video (TI2V) generation and text-and-video-to-video (TV2V) continuation. TI2V conditions on text and a supplied first frame, while TV2V conditions on text and a supplied video segment.

Run the commands from the repository root after preparing the environment and
model assets described in the [README](../README.md). The examples use `uv run python`; `python` also works when the project environment is already activated.

## Contents

- [Models and local assets](#models-and-local-assets)
- [Sampling](#sampling)
    - [TV2V: continue a video](#tv2v-continue-a-video)
        - [Automatic context selection](#automatic-context-selection)
        - [How automatic prefix length is selected](#how-automatic-prefix-length-is-selected)
        - [Explicit prefix length](#explicit-prefix-length)
    - [TI2V: animate an image](#ti2v-animate-an-image)
        - [Single image and prompt](#single-image-and-prompt)
        - [Batch images](#batch-images)
- [Long rollouts](#long-rollouts)
- [Outputs and preprocessing reports](#outputs-and-preprocessing-reports)

## Models and local assets

The released recipes use these defaults. Resolution is shown as width × height.

| Model | TF config | FCP config | Resolution | FPS |
| --- | --- | --- | --- | --- |
| 1.3B | `tf_wan21_t2v_1.3b` | `fcp_wan21_t2v_1.3b` | 768×432 | 16 |
| 5B | `tf_wan22_ti2v_5b` | `fcp_wan22_ti2v_5b` | 1024×576 | 24 |

Base model directories belong under `weights/Wan-AI/`. Each matching project
EMA checkpoint belongs under `weights/Looking-Ahead/<config>/model_ema.pt`.
Both base components and a matching project checkpoint are needed for the
examples below.

## Sampling

Both TV2V and TI2V use `inference.py` as the entry point. The input option
automatically selects the mode: `--video_path` enables TV2V, while
`--image_path` enables TI2V.

| Parameter | Default / behavior |
| --- | --- |
| `--config` / `--config_path` | Required recipe stem or YAML path. |
| `--checkpoint_path` | Explicit selected checkpoint; use a matching `model_ema.pt` with `--use_ema`. |
| `--use_ema` | Load EMA weights; include this flag with the released checkpoints. |
| `--image_path` | Single-image TI2V input; requires text. |
| `--video_path` | Single-video TV2V input; requires text. |
| `--prompt_file` / `--prompt` | Mutually exclusive prompt-file or inline-text inputs. |
| `--duration` | `5s`; positive whole duration with `s`, `m` or `h`. Includes the visual context. |
| `--num_output_frames` | `-1` follows duration; a positive integer takes precedence over duration. |
| `--seed` | `42`; integer from 0 to 2³²−1. |
| `--sink` / `--sink_size` | `3` latent frames retained as the attention sink. `0` disables it. |
| `--output_path` | Output directory; choose a fresh one for a new input or changed settings. |

The attention sink controls retained attention-cache frames. It does not set the
TV2V context length; that is selected automatically or controlled with K.

When no explicit checkpoint path is supplied, `--exp_dir` selects a training
experiment directory. `--checkpoint_name` selects a named checkpoint below its
`checkpoints/` directory; otherwise discovery uses the last lexicographically
sorted `checkpoint_model_*` entry. Explicit `--checkpoint_path` takes precedence.
For EMA, loading first checks for a sibling `model_ema.pt`, then embedded
`generator_ema` in the selected file. Loading fails if EMA weights are unavailable.
Explicit matching paths are recommended for TF/FCP comparisons.

We did not vary the settings below in our experiments. If you understand their
effects, you can customize them using `key=value` arguments:

| Override | Released inference default | Supported values |
| --- | --- | --- |
| `sampling_steps` | `50` | Positive integer; steps per generated chunk. |
| `sample_solver` | `unipc` | `unipc` or `dpm++`. |
| `guidance_scale` | `3.0` | Finite nonnegative CFG scale. |
| `negative_prompt` | Shared negative-prompt string | A string; quote values containing spaces. |
| `timestep_shift` | `5.0` | Finite positive inference scheduler shift. |
| `fps` | 16 for 1.3B; 24 for 5B | Positive integer-valued FPS; also controls video input sampling. |
| `long_inference_threshold` | `256` latent frames | Positive integer; dispatches to the long path at or above this length. |
| `vae_decode_chunk` | `32` latent frames | Positive integer; decode chunk size for the long path. |

Values are merged from shared/inference defaults, the selected recipe and then
`key=value` overrides. Explicit registered flags take precedence over the same
setting in `key=value`, regardless of their order in the command.

> [!IMPORTANT]
> Input images and video frames are automatically resized with their aspect ratio
> preserved and center-cropped to match the model's resolution. Content near the
> edges may be cropped out.

Single-image, MP4 and batch-image inputs share the same spatial resize/crop
function. Each image or selected video frame is:

1. Resized with its aspect ratio preserved until both target dimensions are
   covered. Dimensions are rounded up to whole pixels.
2. Center-cropped to the recipe's target width and height.
3. Converted to a float32 tensor and normalized to `[-1, 1]` before model-side
   precision conversion and VAE encoding.

Resizing uses bilinear interpolation with antialiasing. Image proportions are
preserved, but edge content can be discarded by the crop. Inputs already at the
target size retain their spatial geometry. Original files remain unchanged.

Single-image and MP4 routes save `<sample>.input.json` with original dimensions,
resized dimensions, target dimensions, crop coordinates and a `cropped` flag.
Video reports also include source/target FPS, sampling timestamps, actual source
frame timestamps, selection mode and resolved K. Batch images share the spatial
transform but do not currently emit this per-image JSON report.

### TV2V: continue a video

#### Automatic context selection

This example continues the milk-pouring input with the 5B FCP EMA model:

```bash
uv run python inference.py \
  --config fcp_wan22_ti2v_5b \
  --checkpoint_path weights/Looking-Ahead/fcp_wan22_ti2v_5b/model_ema.pt \
  --use_ema \
  --video_path examples/inference/v2v/milk_pouring.mp4 \
  --prompt_file examples/inference/v2v/milk_pouring.txt \
  --duration 5s --seed 42 --sink 3 \
  --output_path outputs/v2v/fcp_milk_pouring
```

The same flat directory also contains `dominoes.mp4` and `dominoes.txt`.
Replace `milk_pouring` in both input paths and choose another output directory
to use that example.

The duration is the **total output length, including the selected context**.
For this 5B example, 120 output frames contain 25 context frames and 95 generated
continuation frames.

The same MP4 can be used directly with 1.3B:

```bash
uv run python inference.py \
  --config fcp_wan21_t2v_1.3b \
  --checkpoint_path weights/Looking-Ahead/fcp_wan21_t2v_1.3b/model_ema.pt \
  --use_ema \
  --video_path examples/inference/v2v/milk_pouring.mp4 \
  --prompt_file examples/inference/v2v/milk_pouring.txt \
  --duration 5s --seed 42 --sink 3 \
  --output_path outputs/v2v/fcp_milk_pouring_1.3b
```

It is sampled at 16 FPS, resized to 768×432, and uses 13 context frames followed
by 67 continuation frames. No separately converted input file is required.

To override sampling settings for this TV2V example:

```bash
uv run python inference.py \
  --config fcp_wan22_ti2v_5b \
  --checkpoint_path weights/Looking-Ahead/fcp_wan22_ti2v_5b/model_ema.pt \
  --use_ema \
  --video_path examples/inference/v2v/milk_pouring.mp4 \
  --prompt_file examples/inference/v2v/milk_pouring.txt \
  --duration 5s --seed 42 \
  --output_path outputs/v2v/fcp_milk_custom \
  sampling_steps=30 sample_solver=dpm++ guidance_scale=4.0 \
  timestep_shift=5.0 negative_prompt='blur, distorted motion'
```

#### How automatic prefix length is selected

When `--v2v_context_latents` is omitted, the MP4 route:

1. Finds the original video's actual last frame timestamp.
2. Places sampling timestamps backwards from that frame at intervals of
   `1 / target_fps`, then supplies the selected frames in chronological order.
3. Chooses the longest context supported by the video that satisfies the VAE
   and model chunk alignment rules.
4. Limits that context to leave visible continuation and at least one generated
   latent chunk within the requested output length.

Here, **prefix** means the conditioning segment at the beginning of the output;
it can come from the end of the source video. Automatic mode discards the
unneeded beginning of the source video and retains its last frame.

For the released recipes, the VAE temporal stride is 4, the regular chunk size
is 3 latent frames, and the first frame is independent. Therefore:

```text
K = 1 + 3 × M, where M >= 1
condition pixel frames = 1 + 4 × (K - 1)
```

Valid K values are `4, 7, 10, ...`, corresponding to `13, 25, 37, ...` condition
frames. K is a count of **latent frames**, rather than pixel frames or seconds.

Both bundled MP4s contain 25 frames at 24 FPS, with frame timestamps from
0.0 to 1.0 seconds. For a 5-second output:

| Recipe FPS | Automatic K | Condition frames | Selected source timestamps |
| --- | --- | --- | --- |
| 24 (5B) | 7 | 25 | 0.0–1.0 seconds |
| 16 (1.3B) | 4 | 13 | 0.25–1.0 seconds |

Time sampling selects the original frame displayed at each requested timestamp.
A lower target FPS skips some source frames; a higher target FPS can select the
same source frame more than once. No motion interpolation is performed, and the
source MP4 is not rewritten. For variable-frame-rate input, the anchor remains
the actual last frame timestamp.

The target FPS comes from the final `fps` setting, falling back to the model
family default when absent. It controls both input time sampling and output
encoding. A long input can be truncated further by the total output-length
limit. A video that cannot supply the shortest valid context fails rather than
being padded or converted to single-image conditioning.

#### Explicit prefix length

Pass `--v2v_context_latents K` to select a fixed context length. **Explicit K
retains the existing leading-prefix behavior:** it reads from the beginning of
the video, whereas automatic mode selects a segment ending at the last frame.

For example, adding `--v2v_context_latents 7` to the 5B command selects its first
25 frames. The current 1.3B example needs K=4: K=7 at 16 FPS would request a frame
at 1.5 seconds, beyond the supplied input. Invalid alignment, insufficient input
length or an output with no visible continuation fails; explicit K is never
silently shortened. Use a new output directory when changing this setting.

### TI2V: animate an image

#### Single image and prompt

```bash
uv run python inference.py \
  --config fcp_wan21_t2v_1.3b \
  --checkpoint_path weights/Looking-Ahead/fcp_wan21_t2v_1.3b/model_ema.pt \
  --use_ema \
  --image_path examples/inference/i2v/sailboat_lake.png \
  --prompt_file examples/inference/i2v/sailboat_lake.txt \
  --duration 5s --seed 42 \
  --output_path outputs/i2v/fcp_sailboat_lake
```

`--image_path` automatically enables TI2V and uses the supplied image as C0.

The flat `examples/inference/i2v/` directory contains six same-name PNG/TXT pairs:
`sailboat_lake`, `cat_window`, `paper_fan_open`, `book_page_turn`, `can_crush` and
`pendulum`. The fan, book and can examples were selected with 5B; the pendulum
example was selected with 1.3B. Use the matching config and checkpoint when
trying those reference settings. A model name containing `T2V` does not enable
prompt-only sampling; a supplied visual condition is required.

For `--image_path`, EXIF orientation is applied and the image is converted to
RGB. Alpha is discarded, and only the first frame of an animated image is used.
These operations are included in its preprocessing report.

#### Batch images

The batch example contains the original sailboat and cat pairs:

```text
examples/inference/i2v_batch/
├── target_crop_info_1672-941.json
└── 1672-941/
    ├── sailboat_lake.png
    └── cat_window.png
```

```bash
uv run python inference.py \
  --config fcp_wan21_t2v_1.3b \
  --checkpoint_path weights/Looking-Ahead/fcp_wan21_t2v_1.3b/model_ema.pt \
  --use_ema --i2v \
  --data_path examples/inference/i2v_batch \
  --test_sample_limit 2 --duration 5s --seed 42 \
  --output_path outputs/i2v/fcp_batch
```

Use exactly one `target_crop_info_<suffix>.json` file and a matching `<suffix>/`
image subdirectory. The JSON is an array of records with `file_name`, `caption`,
`target_crop` (`target_bbox` and `target_ratio`), `type`, `origin_width` and
`origin_height`; the supplied file is a complete example. Actual spatial
preprocessing uses the loaded image and model resolution, with center cropping;
metadata bounding boxes do not override that transform.

The plain `i2v/` folder is for `--image_path`, rather than this metadata route.
The provided batch metadata covers two images, not all six single-image pairs.

## Long rollouts

> [!CAUTION]
> Our released models were trained on 5-second video clips without dedicated
> anti-drifting techniques. They are therefore not specifically optimized for
> longer rollouts, and generation quality beyond the training duration is not
> guaranteed.

For rollouts beyond the training duration, inference currently uses the
following defaults:

1. **Bounded latent context.** The rolling KV cache retains at most the training
   sequence length in latent frames, given by `image_or_video_shape[1]`: 22 for
   1.3B and 31 for 5B. This limit includes the sink and the current chunk. Older
   non-sink entries are evicted as generation proceeds. The window is controlled
   by `model_kwargs.local_attn_size`; its default value of `-1` resolves to the
   configured training length.
2. **Attention sink.** The default `--sink 3` preserves the first three latent
   frames in the cache while the remaining context rolls forward. It can be
   changed with `--sink` or `model_kwargs.sink_size`.
3. **Window-relative temporal RoPE.** Both model families assign temporal RoPE
   positions relative to the current cache window and reapply RoPE to cached
   keys as the window rolls. Temporal positions are shifted back and clamped to
   the training range when necessary, rather than growing indefinitely with the
   full rollout. Spatial RoPE is unchanged.

These settings are our initial defaults, not an optimized solution for long
rollouts. If you understand their implications and your requirements, you can
adjust the cache window and sink settings or modify the RoPE implementation.

In preliminary experiments, incorporating simple anti-drifting techniques
inspired by [Helios](https://github.com/PKU-YuanGroup/Helios) improved rollout
stability. We plan to investigate this direction more systematically in future
work.

## Outputs and preprocessing reports

For the first 5B milk example, output files are organized as:

```text
outputs/v2v/fcp_milk_pouring/
├── 00000-42-v2v-k=7.mp4
├── 00000-42-v2v-k=7.png
├── 00000-42-v2v-k=7.input.json
├── 00000.txt
├── conditions/00000-42-v2v-k=7.mp4
├── continuations/00000-42-v2v-k=7.mp4
└── manifests/00000-42-v2v-k=7.json
```

The main MP4 includes both the context and continuation. `conditions/` and
`continuations/` are slices of that decoded output. The PNG shows the processed
first frame of the selected context. The TXT records its prompt. TI2V uses the
same naming without the `-v2v-k=K` suffix and without TV2V companion directories.

The TV2V manifest records the requested and resolved context length, selection
mode (`auto_tail` or `explicit_prefix`), frame counts, FPS,
seed, selected weights and prompt. Automatic mode preserves `null` as the
requested K in the saved configuration; reports contain the actual resolved K.

With `--output_path outputs/v2v/fcp_milk_pouring`, the effective public config is
saved to `outputs/v2v/configs/fcp_milk_pouring.yaml`, and console messages are
also written to `outputs/v2v/logs/fcp_milk_pouring/rank_0.log`.

An existing main MP4 is skipped based on its presence alone. Its companion files
are not regenerated or repaired. Use a fresh output directory when changing
inputs, prefix selection, weights or other settings. Seed alone does not
guarantee identical results across runtime environments.

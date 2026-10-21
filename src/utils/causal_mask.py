import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def _compute_block_edges(
    length: int, block_size: int, independent_first_frame: int = 0, segment_starts=None
):
    """
    生成 block 边界点列表 [b0, b1, b2, ...]，相邻两个边界构成一个 block [b_i, b_{i+1})。

    支持多段（segment）布局：``segment_starts`` 给出每个段的起点（如 teacher forcing
    中 clean 段起点 0、noise 段起点 num_frames*frame_seqlen）。每个段内部从段起点开始
    按 ``block_size`` 切分；若 ``independent_first_frame > 0``，则每个段的第一个 block
    长度为 ``independent_first_frame``。

    Returns
    -------
    edges : list[int]
        长度 = block 数 + 1，满足 edges[0] == 0、edges[-1] == length，且单调递增。
    is_independent : list[bool]
        长度 = block 数，标记每个 block 是否是某个段的独立首帧 block。
    """
    if block_size is None or block_size <= 0:
        return ([0, length], [False])
    if segment_starts is None or len(segment_starts) == 0:
        segment_starts = [0]
    segs = sorted({int(s) for s in segment_starts if 0 <= int(s) < length})
    if not segs or segs[0] != 0:
        segs = [0] + [s for s in segs if s != 0]
    seg_ends = segs[1:] + [length]
    edges = [0]
    is_independent = []
    has_indep = independent_first_frame is not None and independent_first_frame > 0
    for s_start, s_end in zip(segs, seg_ends, strict=False):
        cur = s_start
        if has_indep:
            first_end = min(cur + independent_first_frame, s_end)
            if first_end > cur:
                edges.append(first_end)
                is_independent.append(True)
                cur = first_end
            if cur >= s_end:
                continue
        while cur < s_end:
            nxt = min(cur + block_size, s_end)
            edges.append(nxt)
            is_independent.append(False)
            cur = nxt
    return (edges, is_independent)


def mask_to_attention_with_grid(
    mask,
    upscale=32,
    grid_color=(0, 0, 0),
    block_size=None,
    highlight_color=(255, 0, 0),
    highlight_alpha=1.0,
    highlight_only_active=True,
    block_grid_color=None,
    history_size: int = 0,
    history_grid_color=(0, 255, 0),
    history_alpha=0.35,
    independent_first_frame: int = 0,
    clean_length: int = 0,
):
    """
    可视化 attention masking 矩阵。

    Parameters
    ----------
    mask : np.ndarray
        HxW 的 0/1 mask。

        注意：本项目里 ``mask`` 通常来自
        ``BlockMask.to_dense().squeeze(0).squeeze(0).cpu().numpy()``，
        FlexAttention 的 ``BlockMask`` dense 出来后是 *block 粒度*
        （H = total_token_length / FLEX_BLOCK_SIZE，FLEX_BLOCK_SIZE 默认 128）。
        当 ``frame_seqlen == FLEX_BLOCK_SIZE`` 时，``mask`` 的 1 像素 = 1 帧；
        所有 ``block_size`` / ``clean_length`` / ``independent_first_frame``
        参数都按 *帧* 单位（即 mask 的行/列像素位置）传入。

    upscale : int
        每个 cell 放大的像素大小

    grid_color : tuple
        普通网格颜色

    block_size : int | None
        block 粒度，单位为 **帧**（即 ``num_frame_per_block``）。

    highlight_color : tuple
        对角 block 高亮颜色

    highlight_alpha : float
        对角 block 高亮混合系数

    highlight_only_active : bool
        True:
            仅高亮 mask>0 区域
        False:
            整个 block 区域高亮

    block_grid_color : tuple | None
        block 外框颜色

    history_size : int
        block-wise history 数量（block 个数）

    history_grid_color : tuple
        history block 内部染色颜色

    history_alpha : float
        history block 混合透明度

    independent_first_frame : int
        > 0 时，每个段的前 ``independent_first_frame`` *帧* 视为一个独立
        block（对应 i2v 模式下被分离出来的第一帧），其余帧再按 ``block_size``
        切分。等于 0 时表示常规均匀分块。
        注意：该独立首帧 block 不计入 history_size 统计，也不会被染上
        history 颜色。

    clean_length : int
        > 0 时，表示输入 mask 是 teacher forcing 双段布局
        ``[clean (clean_length 帧) | noise]``，clean 段和 noise 段会各自
        独立按 ``block_size`` 切分。若同时指定了 ``independent_first_frame > 0``，
        每个段的首个 block 长度为 ``independent_first_frame`` *帧*（用于 i2v）。
        等于 0 时退化为单段布局。
    """
    H, W = mask.shape
    block_size_tok = block_size
    indep_tok = (
        independent_first_frame if independent_first_frame and independent_first_frame > 0 else 0
    )
    clean_length_tok = clean_length if clean_length and clean_length > 0 else 0
    cmap = plt.get_cmap("plasma")
    colored = cmap(mask.astype(np.float32))[:, :, :3]
    img = (colored * 255).astype(np.uint8)
    img = img.astype(np.float32)
    if block_size_tok is not None and block_size_tok > 0 and (history_size > 0):
        history_color = np.array(history_grid_color, dtype=np.float32)
        seg_starts = [0]
        if clean_length_tok and clean_length_tok > 0:
            seg_starts.append(clean_length_tok)
        row_edges, row_is_indep = _compute_block_edges(H, block_size_tok, indep_tok, seg_starts)
        col_edges, col_is_indep = _compute_block_edges(W, block_size_tok, indep_tok, seg_starts)
        n_row_blocks = len(row_edges) - 1
        n_col_blocks = len(col_edges) - 1
        n_diag_blocks = min(n_row_blocks, n_col_blocks)
        for br in range(n_diag_blocks):
            valid_history_found = 0
            for bc in range(br - 1, -1, -1):
                if col_is_indep[bc]:
                    continue
                r0 = row_edges[br]
                r1 = row_edges[br + 1]
                c0 = col_edges[bc]
                c1 = col_edges[bc + 1]
                sub_mask = mask[r0:r1, c0:c1] > 0
                if not np.any(sub_mask):
                    continue
                valid_history_found += 1
                if valid_history_found > history_size:
                    break
                region = img[r0:r1, c0:c1]
                if highlight_only_active:
                    blended = region.copy()
                    blended[sub_mask] = (1.0 - history_alpha) * region[
                        sub_mask
                    ] + history_alpha * history_color
                else:
                    blended = (1.0 - history_alpha) * region + history_alpha * history_color
                img[r0:r1, c0:c1] = blended
    if block_size_tok is not None and block_size_tok > 0:
        hl = np.array(highlight_color, dtype=np.float32)
        seg_starts = [0]
        if clean_length_tok and clean_length_tok > 0:
            seg_starts.append(clean_length_tok)
        row_edges, _ = _compute_block_edges(H, block_size_tok, indep_tok, seg_starts)
        col_edges, _ = _compute_block_edges(W, block_size_tok, indep_tok, seg_starts)
        n_blocks = min(len(row_edges) - 1, len(col_edges) - 1)
        for i in range(n_blocks):
            r0 = row_edges[i]
            r1 = row_edges[i + 1]
            c0 = col_edges[i]
            c1 = col_edges[i + 1]
            region = img[r0:r1, c0:c1]
            if highlight_only_active:
                sub_mask = mask[r0:r1, c0:c1] > 0
                if not np.any(sub_mask):
                    continue
                blended = region.copy()
                blended[sub_mask] = (1.0 - highlight_alpha) * region[
                    sub_mask
                ] + highlight_alpha * hl
            else:
                blended = (1.0 - highlight_alpha) * region + highlight_alpha * hl
            img[r0:r1, c0:c1] = blended
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((W * upscale, H * upscale), resample=Image.NEAREST)
    draw = ImageDraw.Draw(pil_img)
    for x in range(0, W * upscale, upscale):
        draw.line([(x, 0), (x, H * upscale)], fill=grid_color)
    for y in range(0, H * upscale, upscale):
        draw.line([(0, y), (W * upscale, y)], fill=grid_color)
    if block_size_tok is not None and block_size_tok > 0 and (block_grid_color is not None):
        seg_starts = [0]
        if clean_length_tok and clean_length_tok > 0:
            seg_starts.append(clean_length_tok)
        row_edges, _ = _compute_block_edges(H, block_size_tok, indep_tok, seg_starts)
        col_edges, _ = _compute_block_edges(W, block_size_tok, indep_tok, seg_starts)
        n_blocks = min(len(row_edges) - 1, len(col_edges) - 1)
        for i in range(n_blocks):
            x0 = col_edges[i] * upscale
            y0 = row_edges[i] * upscale
            x1 = col_edges[i + 1] * upscale - 1
            y1 = row_edges[i + 1] * upscale - 1
            draw.rectangle([(x0, y0), (x1, y1)], outline=block_grid_color, width=2)
    return pil_img


def visualize_causal_mask(
    mode: str,
    *,
    num_frames: int,
    num_frame_per_block: int,
    local_attn_size: int = -1,
    sink_size: int = 0,
    independent_first_frame: bool = False,
    out_dir: str,
    vis_frame_seqlen: int = 128,
    upscale: int = 16,
    filename: str | None = None,
):
    """
    Build a *toy* causal mask with the same block / window / sink layout as the
    real model but a tiny ``frame_seqlen`` (default 128 instead of ~1560), then
    render it via :func:`mask_to_attention_with_grid` and save to ``out_dir``.

    The real ``self.block_mask`` is way too large to materialise as a dense
    matrix (e.g. ``(21*1560)^2`` ~ 1e9 entries), so we visualise an equivalent
    *structure-preserving* mini version which is enough to inspect the block
    pattern (causal window, sink, independent first frame, TF clean/noise
    layout, etc.).

    The arguments forwarded to :func:`mask_to_attention_with_grid` are now
    expressed in **frames** (matching the convention used by
    ``notebooks/jupyter_causal_mask.ipynb``):
    ``block_size = num_frame_per_block``, ``clean_length = num_frames``,
    ``independent_first_frame = 0 or 1``, ``history_size`` counts past
    *blocks* within the local-attention window (excluding the sink and the
    current block). ``frame_seqlen`` is forwarded so the renderer can map
    these frame-unit args to token-unit positions on the dense mask.

    Parameters
    ----------
    mode : {"TF"}
        Teacher-forcing two-segment layout.
    out_dir : str
        Directory the PNG is saved to. Created if missing.
    filename : str | None
        Override output filename. If None, a descriptive name is generated.

    Returns
    -------
    str
        Full path to the saved PNG.
    """
    import os

    mode = mode.upper()
    assert mode == "TF", f"unknown mode {mode!r}"
    kwargs = {
        "device": "cpu",
        "num_frames": num_frames,
        "frame_seqlen": vis_frame_seqlen,
        "num_frame_per_block": num_frame_per_block,
        "local_attn_size": local_attn_size,
        "sink_size": sink_size,
        "independent_first_frame": independent_first_frame,
        "_compile": False,
    }
    mask = CausalMaskMixin._prepare_teacher_forcing_mask(**kwargs)
    dense_mask = mask.to_dense().squeeze(0).squeeze(0).cpu().numpy()
    if local_attn_size > 0:
        history_size = max(0, (local_attn_size - sink_size) // num_frame_per_block - 1)
    else:
        history_size = 0
    pil_img = mask_to_attention_with_grid(
        dense_mask,
        upscale=upscale,
        block_size=num_frame_per_block,
        history_size=history_size,
        independent_first_frame=int(bool(independent_first_frame)),
        clean_length=num_frames,
    )
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"{mode}_F{num_frames}_B{num_frame_per_block}_L{local_attn_size}_K{sink_size}_I{int(bool(independent_first_frame))}.png"
    out_path = os.path.join(out_dir, filename)
    pil_img.save(out_path)
    return out_path


class CausalMaskMixin:
    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        independent_first_frame: bool = False,
        _compile: bool = True,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
            [clean frames] [noisy frames]
        where each segment is laid out as

            independent_first_frame=False:
                [N latent frame] [N latent frame] ... [N latent frame]
            independent_first_frame=True:
                [1 latent frame] [N latent frame] ... [N latent frame]
            (N = num_frame_per_block; the 1-frame block is the i2v image
            conditioning frame)

        We use flex attention to construct the attention mask.
        """
        if local_attn_size == -1:
            local_attn_size = num_frames
        assert local_attn_size >= num_frame_per_block, (
            f"local_attn_size ({local_attn_size}) should be greater than or equal to num_frame_per_block ({num_frame_per_block})"
        )
        assert sink_size <= local_attn_size - num_frame_per_block, (
            f"sink_size ({sink_size}) should be less than or equal to local_attn_size ({local_attn_size}) - num_frame_per_block ({num_frame_per_block})"
        )
        if independent_first_frame:
            assert (num_frames - 1) % num_frame_per_block == 0, (
                f"For independent_first_frame, (num_frames - 1) % num_frame_per_block == 0, got num_frames={num_frames} and num_frame_per_block={num_frame_per_block}"
            )
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        clean_ends = num_frames * frame_seqlen
        attention_block_size = num_frame_per_block * frame_seqlen
        attention_window_size = local_attn_size * frame_seqlen
        attention_sink_size = sink_size * frame_seqlen
        B_size = attention_block_size
        W_size = attention_window_size
        S_size = attention_sink_size
        CE = clean_ends
        pad_start = total_length
        I_off = frame_seqlen if independent_first_frame else 0

        def attention_mask(b, h, q_idx, kv_idx):
            is_clean_q = q_idx < CE
            is_noise_q = (q_idx >= CE) & (q_idx < pad_start)
            is_clean_first = is_clean_q & (q_idx < I_off)
            q_clean_rel = q_idx - I_off
            q_clean_block_end_regular = I_off + (q_clean_rel // B_size + 1) * B_size
            c_end = torch.where(is_clean_first, I_off, q_clean_block_end_regular)
            c_start = c_end - W_size + S_size
            C0 = (kv_idx >= c_start) & (kv_idx < c_end)
            C0_Sink = (kv_idx < S_size) & (kv_idx < c_start)
            clean_mask = is_clean_q & (C0 | C0_Sink)
            q_noise_rel = q_idx - CE
            is_noise_first = is_noise_q & (q_noise_rel < I_off)
            nb_regular = (q_noise_rel - I_off) // B_size
            n_start_regular = CE + I_off + nb_regular * B_size
            n_end_regular = n_start_regular + B_size
            n_start = torch.where(is_noise_first, CE, n_start_regular)
            n_end = torch.where(is_noise_first, CE + I_off, n_end_regular)
            nc_end_regular = I_off + nb_regular * B_size
            nc_start_regular = I_off + (nb_regular + 1) * B_size - W_size + S_size
            nc_end = torch.where(is_noise_first, torch.zeros_like(nc_end_regular), nc_end_regular)
            nc_start_first = I_off - W_size + S_size
            nc_start = torch.where(is_noise_first, nc_start_first, nc_start_regular)
            C1 = (kv_idx >= n_start) & (kv_idx < n_end)
            C2 = (kv_idx >= nc_start) & (kv_idx < nc_end)
            C1_Sink = (kv_idx < S_size) & (kv_idx < nc_start)
            noise_mask = is_noise_q & (C1 | C2 | C1_Sink)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=_compile,
            device=device,
        )
        return block_mask


if __name__ == "__main__":
    device = "cpu"
    kwargs = {"device": device, "num_frames": 21, "frame_seqlen": 128, "num_frame_per_block": 3}
    local_attn_size = 6
    sink_size = 1
    mask = CausalMaskMixin._prepare_teacher_forcing_mask(
        **kwargs, local_attn_size=local_attn_size, sink_size=sink_size
    )
    dense_mask = mask.to_dense().squeeze(0).squeeze(0).cpu().numpy()
    mask_image = mask_to_attention_with_grid(
        dense_mask,
        upscale=4,
        block_size=kwargs["num_frame_per_block"],
        clean_length=kwargs["num_frames"],
    )
    mask_image.save("teacher_forcing_mask.png")
    kwargs_i2v = dict(kwargs, num_frames=22)
    mask_i2v = CausalMaskMixin._prepare_teacher_forcing_mask(
        **kwargs_i2v,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        independent_first_frame=True,
    )
    dense_mask_i2v = mask_i2v.to_dense().squeeze(0).squeeze(0).cpu().numpy()
    mask_image_i2v = mask_to_attention_with_grid(
        dense_mask_i2v,
        upscale=4,
        block_size=kwargs_i2v["num_frame_per_block"],
        independent_first_frame=1,
        clean_length=kwargs_i2v["num_frames"],
    )
    mask_image_i2v.save("teacher_forcing_mask_i2v.png")

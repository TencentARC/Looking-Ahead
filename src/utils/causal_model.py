import logging
import math
import os

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import flex_attention

from src.utils.causal_mask import CausalMaskMixin, visualize_causal_mask
from src.utils.causal_rope import causal_rope_apply
from src.utils.distributed import is_main_process
from src.wan.modules.model import WAN_CROSSATTENTION_CLASSES
from src.wan22.modules.model import (
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

from .attention import attention

flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")


class CausalWanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        train_frame_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-06,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.train_frame_size = train_frame_size
        self.inference_frame_size = local_attn_size if local_attn_size != -1 else train_frame_size
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return (q, k, v)

        q, k, v = qkv_fn(x)
        if kv_cache is None:
            is_tf = s == seq_lens[0].item() * 2
            if not is_tf:
                raise ValueError(
                    "Uncached attention requires a teacher-forcing clean/noisy sequence."
                )
            q_chunk = torch.chunk(q, 2, dim=1)
            k_chunk = torch.chunk(k, 2, dim=1)
            roped_query = []
            roped_key = []
            for ii in range(2):
                rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                roped_query.append(rq)
                roped_key.append(rk)
            roped_query = torch.cat(roped_query, dim=1)
            roped_key = torch.cat(roped_key, dim=1)
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [
                    roped_query,
                    torch.zeros(
                        [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )
            padded_roped_key = torch.cat(
                [
                    roped_key,
                    torch.zeros(
                        [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                        device=k.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )
            padded_v = torch.cat(
                [
                    v,
                    torch.zeros(
                        [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask,
            )
            if padded_length > 0:
                x = x[:, :, :-padded_length]
            x = x.transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            max_attention_size = self.inference_frame_size * frame_seqlen
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            if (
                self.local_attn_size != -1
                and current_end > kv_cache["global_end_index"].item()
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):
                num_evicted_tokens = (
                    num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                )
                num_rolled_tokens = (
                    kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                )
                rolling_dst = slice(sink_tokens, sink_tokens + num_rolled_tokens)
                rolling_src = slice(
                    sink_tokens + num_evicted_tokens,
                    sink_tokens + num_evicted_tokens + num_rolled_tokens,
                )
                kv_cache["k"][:, rolling_dst] = kv_cache["k"][:, rolling_src].clone()
                kv_cache["v"][:, rolling_dst] = kv_cache["v"][:, rolling_src].clone()
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v
            local_start_frame_index = local_start_index // frame_seqlen
            local_end_frame_index = local_end_index // frame_seqlen
            assert local_start_frame_index + grid_sizes[0][0].item() == local_end_frame_index, (
                f"local_start_frame_index: {local_start_frame_index}, local_end_frame_index: {local_end_frame_index}, grid_sizes[0][0]: {grid_sizes[0][0].item()}"
            )
            query_relative_indices = torch.arange(
                local_start_frame_index, local_end_frame_index, device=q.device
            )
            cache_relative_indices = torch.arange(0, local_end_frame_index, device=k.device)
            if self.train_frame_size == -1:
                query_shift = 0
                cache_shift = 0
            else:
                max_train_frame_index = self.train_frame_size - 1
                query_shift = max(query_relative_indices.max().item() - max_train_frame_index, 0)
                cache_shift = max(cache_relative_indices.max().item() - max_train_frame_index, 0)
            query_relative_indices = torch.clamp(query_relative_indices - query_shift, min=0)
            cache_relative_indices = torch.clamp(cache_relative_indices - cache_shift, min=0)
            if (
                self.local_attn_size != -1
                and current_end > kv_cache["global_end_index"].item()
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):
                tail_start = max(sink_tokens, local_end_index - max_attention_size + sink_tokens)
                tail_end = local_end_index
                sink_end = min(kv_cache["local_end_index"].item(), sink_tokens)
                if sink_end > 0:
                    k = torch.cat(
                        [kv_cache["k"][:, :sink_end], kv_cache["k"][:, tail_start:tail_end]], dim=1
                    )
                    v = torch.cat(
                        [kv_cache["v"][:, :sink_end], kv_cache["v"][:, tail_start:tail_end]], dim=1
                    )
                else:
                    k = kv_cache["k"][:, tail_start:tail_end]
                    v = kv_cache["v"][:, tail_start:tail_end]
            else:
                k = kv_cache["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
                v = kv_cache["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
            ).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, relative_frame_indices=cache_relative_indices
            ).type_as(v)
            x = attention(roped_query, roped_key, v)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        model_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-06,
        train_frame_size: int = 21,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            train_frame_size=train_frame_size,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        WanCrossAttentionClass = WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"]
        self.cross_attn = WanCrossAttentionClass(
            dim=dim, num_heads=num_heads, window_size=(-1, -1), qk_norm=qk_norm, eps=eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
    ):
        """
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-06):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        shift, scale = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        x = self.head(x * (1 + scale) + shift)
        return x


class CausalWanModel(ModelMixin, ConfigMixin, CausalMaskMixin):
    """
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["CausalWanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=21,
        sink_size=0,
        train_frame_size=21,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-06,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = False,
    ):
        """
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Selected backbone family: 't2v' or 'ti2v'; sampling requires supplied C0.
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to 21):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            train_frame_size (`int`, *optional*, defaults to 21):
                Number of frames in training sequences, used for block-relative RoPE in inference
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """
        super().__init__()
        assert model_type in ["t2v", "ti2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.train_frame_size = train_frame_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        block_cls = CausalWanAttentionBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    model_type=model_type,
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    train_frame_size=train_frame_size,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask = None

    def _forward_inference(
        self,
        x,
        t,
        context,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
    ):
        """
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        patch_embedding = self.patch_embedding
        x = [patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(x)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.block_mask,
        }

        def create_custom_forward(module):

            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        output = torch.stack(x)
        return output

    def _forward_train(self, x, t, context, clean_x=None):
        """
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if clean_x is None:
            raise ValueError("Teacher-forcing forward requires clean_x.")
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if self.block_mask is None:
            self.block_mask = self._prepare_teacher_forcing_mask(
                device,
                num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
                independent_first_frame=self.independent_first_frame,
            )
            _vis_flag = os.environ.get("WAN_VIS_CAUSAL_MASK", "0")
            if _vis_flag and _vis_flag != "0":
                if is_main_process():
                    if not hasattr(self, "_causal_mask_vis_done"):
                        self._causal_mask_vis_done = set()
                    mode = "TF"
                    vis_key = (mode, int(x.shape[2]), bool(self.independent_first_frame))
                    if vis_key not in self._causal_mask_vis_done:
                        out_dir = os.environ.get(
                            "WAN_VIS_CAUSAL_MASK_DIR", os.path.join("outputs", "causal_mask_vis")
                        )
                        try:
                            saved_path = visualize_causal_mask(
                                mode=mode,
                                num_frames=int(x.shape[2]),
                                num_frame_per_block=self.num_frame_per_block,
                                local_attn_size=self.local_attn_size,
                                sink_size=self.sink_size,
                                independent_first_frame=bool(self.independent_first_frame),
                                out_dir=out_dir,
                            )
                            logging.info(
                                "[causal_mask_vis] saved %s mask (num_frames=%d, num_frame_per_block=%d, local_attn_size=%d, sink_size=%d, independent_first_frame=%s) -> %s",
                                mode,
                                int(x.shape[2]),
                                self.num_frame_per_block,
                                self.local_attn_size,
                                self.sink_size,
                                bool(self.independent_first_frame),
                                saved_path,
                            )
                        except Exception as _e:
                            logging.warning("[causal_mask_vis] failed: %s", _e)
                        self._causal_mask_vis_done.add(vis_key)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1) for u in x]
        )
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )
        clean_patch_embedding = self.patch_embedding
        clean_x = [clean_patch_embedding(u.unsqueeze(0)) for u in clean_x]
        clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
        seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
        clean_x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1)
                for u in clean_x
            ]
        )
        x = torch.cat([clean_x, x], dim=1)
        aug_t = torch.zeros_like(t)
        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
        )
        e0_clean = (
            self.time_projection(e_clean)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )
        e0 = torch.cat([e0_clean, e0], dim=1)
        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context,
            "context_lens": context_lens,
            "block_mask": self.block_mask,
        }

        def create_custom_forward(module):

            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False
                )
            else:
                x = block(x, **kwargs)
        x = x[:, x.shape[1] // 2 :]
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
            out.append(u)
        return out

    def init_weights(self):
        """
        Initialize model parameters using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

"""FCP with shared source streams and per-offset AdaLN mini DiTs.

The causal backbone/cached preview route is shared with TF. Auxiliary modules
retain checkpoint-compatible serialized names and are used only during teacher-forced training.
"""

import copy
import logging
import os

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.utils.attention import attention
from src.utils.causal_mask import visualize_causal_mask
from src.utils.causal_model import CausalHead, CausalWanModel
from src.utils.distributed import is_main_process
from src.utils.predictive_multi_chunk_config import build_predictive_chunk_layout
from src.wan22.modules.model import WanLayerNorm, WanRMSNorm, rope_apply, sinusoidal_embedding_1d


class PredictiveSourceAdaLNProjection(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class PredictiveMultiChunkSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs):
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        out = attention(
            rope_apply(q, grid_sizes, freqs).type_as(v),
            rope_apply(k, grid_sizes, freqs).type_as(v),
            v,
            causal=False,
            window_size=(-1, -1),
        )
        return self.o(out.type_as(v).flatten(2))


class PredictiveMultiChunkAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = PredictiveMultiChunkSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.source_adaln_projection = PredictiveSourceAdaLNProjection(dim, 6 * dim)

    def forward(self, x, e, grid_sizes, freqs, source_tokens):
        if source_tokens.shape != x.shape:
            raise ValueError("FCP AdaLN source and future token shapes must match")
        frames, seq = e.shape[1], x.shape[1] // e.shape[1]
        source_mod = (
            self.source_adaln_projection(source_tokens)
            .unflatten(1, (frames, seq))
            .unflatten(-1, (6, self.dim))
        )
        e = (self.modulation.view(1, 1, 1, 6, self.dim) + e.unsqueeze(2) + source_mod).unbind(3)
        y = self.self_attn(
            (self.norm1(x).unflatten(1, (frames, seq)) * (1 + e[1]) + e[0]).flatten(1, 2),
            grid_sizes,
            freqs,
        )
        x = x + (y.unflatten(1, (frames, seq)) * e[2]).flatten(1, 2)
        y = self.ffn((self.norm2(x).unflatten(1, (frames, seq)) * (1 + e[4]) + e[3]).flatten(1, 2))
        return x + (y.unflatten(1, (frames, seq)) * e[5]).flatten(1, 2)


class PredictiveMultiChunkBranch(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        dim = model.dim
        self.hidden_adapter = nn.Sequential(
            nn.Linear(
                len(config["source_layers"]) * dim, config["source_layer_fusion"]["hidden_dim"]
            ),
            nn.GELU(),
            nn.Linear(config["source_layer_fusion"]["hidden_dim"], dim),
        )
        self.blocks = nn.ModuleList(
            [
                PredictiveMultiChunkAttentionBlock(
                    dim, model.ffn_dim, model.num_heads, model.qk_norm, model.eps
                )
                for _ in range(config["mini_dit_depth"])
            ]
        )
        _init_predictive_module(self)
        if config["init_mode"] == "copy":
            for block, index in zip(self.blocks, config["init_source"], strict=True):
                main = model.blocks[index].state_dict()
                target = block.state_dict()
                selected = {k: v for k, v in main.items() if k in target}
                missing, unexpected = block.load_state_dict(selected, strict=False)
                if unexpected or any(not k.startswith("source_adaln_projection.") for k in missing):
                    raise RuntimeError("FCP mini DiT copy is incompatible with backbone blocks")
                block.source_adaln_projection.reset_parameters()
        # Fresh private modules preserve source initialization order. Do not
        # recopy these modules after a subsequent TF checkpoint warm start.
        for name in ("patch_embedding", "time_embedding", "time_projection", "head"):
            original = getattr(model, name)
            if name == "patch_embedding":
                module = nn.Conv3d(
                    model.in_dim, dim, kernel_size=model.patch_size, stride=model.patch_size
                )
            elif name == "time_embedding":
                module = nn.Sequential(
                    nn.Linear(model.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
                )
            elif name == "time_projection":
                module = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
            else:
                module = CausalHead(dim, model.out_dim, model.patch_size, model.eps)
            reference = next(original.parameters())
            module.to(device=reference.device, dtype=reference.dtype)
            if config["init_mode"] == "copy":
                module.load_state_dict(original.state_dict(), strict=True)
            else:
                _init_predictive_module(module)
            setattr(self, name, module)
        reference = next(model.parameters())
        self.to(device=reference.device, dtype=reference.dtype)
        self.freq_dim, self.dim = model.freq_dim, dim

    def forward(
        self,
        hidden_tokens,
        future_latents,
        future_t,
        grid_sizes,
        freqs,
        gradient_checkpointing=False,
    ):
        dtype = hidden_tokens.dtype
        patch = self.patch_embedding(future_latents.to(self.patch_embedding.weight.dtype))
        if tuple(patch.shape[2:]) != tuple(grid_sizes[0].tolist()):
            raise ValueError("FCP future latent patch grid mismatch")
        x = patch.flatten(2).transpose(1, 2).to(dtype)
        embeds = sinusoidal_embedding_1d(self.freq_dim, future_t.flatten()).to(
            self.time_embedding[0].weight.dtype
        )
        embeds = self.time_embedding(embeds)
        e = (
            self.time_projection(embeds)
            .unflatten(1, (6, self.dim))
            .unflatten(0, future_t.shape)
            .to(dtype)
        )
        head_e = embeds.unflatten(0, future_t.shape).unsqueeze(2).to(dtype)
        source = self.hidden_adapter(hidden_tokens).to(dtype)
        for block in self.blocks:
            if torch.is_grad_enabled() and gradient_checkpointing:
                x = checkpoint(block, x, e, grid_sizes, freqs, source, use_reentrant=False)
            else:
                x = block(x, e, grid_sizes, freqs, source)
        return self.head(x.to(self.head.head.weight.dtype), head_e.to(self.head.head.weight.dtype))


def _init_predictive_module(module):
    for child in module.modules():
        if isinstance(child, PredictiveSourceAdaLNProjection):
            child.reset_parameters()
        elif isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class PredictiveCausalWanModel(CausalWanModel):
    def configure_predictive_multi_chunk(self, config):
        if hasattr(self, "predictive_multi_chunk_branches"):
            raise RuntimeError("FCP branches are already configured")
        self.predictive_multi_chunk_config = copy.deepcopy(config)
        self.predictive_multi_chunk_branches = nn.ModuleDict(
            {
                f"offset_{offset}": PredictiveMultiChunkBranch(self, config)
                for offset in config["future_offsets"]
            }
        )
        self.register_buffer(
            "predictive_multi_chunk_source_layers_metadata",
            torch.tensor(
                config["source_layers"], dtype=torch.int64, device=next(self.parameters()).device
            ),
        )

    def _forward_predictive_multi_chunk(
        self, sources, grid_sizes, aux_noisy_chunks_by_offset, aux_timestep_chunks_by_offset
    ):
        cfg = self.predictive_multi_chunk_config
        batch = sources[cfg["source_layers"][0]].shape[0]
        frames = int(grid_sizes[0, 0])
        layout = build_predictive_chunk_layout(
            num_frames=frames,
            block_size=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
        )
        if layout.num_regular_chunks != cfg["num_chunks"]:
            raise ValueError("FCP runtime and configured chunk counts differ")
        frame_seq = int(grid_sizes[0, 1] * grid_sizes[0, 2])
        noisy_tokens = frames * frame_seq
        regular_start = layout.regular_frame_start * frame_seq
        hidden = torch.cat([sources[layer] for layer in cfg["source_layers"]], dim=-1)
        chunk_tokens = self.num_frame_per_block * frame_seq
        source_chunks = {
            "clean": hidden[:, regular_start:noisy_tokens],
            "noisy": hidden[:, noisy_tokens + regular_start : 2 * noisy_tokens],
        }
        source_chunks = {
            name: value.reshape(batch, layout.num_regular_chunks, chunk_tokens, hidden.shape[-1])
            for name, value in source_chunks.items()
        }
        chunk_grid = grid_sizes.clone()
        chunk_grid[:, 0] = self.num_frame_per_block
        outputs = {}
        for offset in cfg["future_offsets"]:
            valid = layout.valid_pair_count(offset)
            if (
                offset not in aux_noisy_chunks_by_offset
                or offset not in aux_timestep_chunks_by_offset
            ):
                raise ValueError(f"Missing independent FCP inputs for offset {offset}")
            future = aux_noisy_chunks_by_offset[offset]
            future_t = aux_timestep_chunks_by_offset[offset]
            expected = (batch, self.in_dim, valid, self.num_frame_per_block)
            if tuple(future.shape[:4]) != expected or tuple(future_t.shape) != (
                batch,
                valid,
                self.num_frame_per_block,
            ):
                raise ValueError(f"Invalid FCP auxiliary input shapes for offset {offset}")
            future = future.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * valid, self.in_dim, self.num_frame_per_block, *future.shape[-2:]
            )
            grids = chunk_grid.repeat_interleave(valid, dim=0)
            predictions = {}
            branch = self.predictive_multi_chunk_branches[f"offset_{offset}"]
            for name in cfg["source_context_streams"]:
                current = source_chunks[name][:, :valid].reshape(
                    batch * valid, chunk_tokens, hidden.shape[-1]
                )
                head = branch(
                    current,
                    future,
                    future_t.reshape(batch * valid, self.num_frame_per_block),
                    grids,
                    self.freqs,
                    self.gradient_checkpointing,
                )
                prediction = torch.stack(self.unpatchify(head, grids)).permute(0, 2, 1, 3, 4)
                predictions[name] = prediction.unflatten(0, (batch, valid))
            outputs[offset] = {"flow_pred_by_source": predictions, "valid_chunks": valid}
        return outputs

    def _forward_train(
        self,
        x,
        t,
        context,
        clean_x=None,
        return_predictive_aux=False,
        aux_noisy_chunks_by_offset=None,
        aux_timestep_chunks_by_offset=None,
    ):
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

        sources, auxiliary = {}, None
        selected = (
            self.predictive_multi_chunk_config["source_layers"] if return_predictive_aux else []
        )
        for index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False
                )
            else:
                x = block(x, **kwargs)
            if index in selected:
                sources[index] = x
                if index == selected[-1]:
                    auxiliary = self._forward_predictive_multi_chunk(
                        sources,
                        grid_sizes,
                        aux_noisy_chunks_by_offset,
                        aux_timestep_chunks_by_offset,
                    )
        x = x[:, x.shape[1] // 2 :]
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        output = torch.stack(x)
        return (output, auxiliary) if return_predictive_aux else output

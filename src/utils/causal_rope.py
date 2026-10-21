import torch


def rope_time_delta_multiply(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: int) -> None:
    """
    k_chunk: [B, L_sink, H, D] (view of the sink portion of K, with RoPE already applied)
    freqs  : [1024, C/2] complex (self.freqs)
    delta_frames: how many frames to shift the sink to the left (negative) / right (positive)
    In-place, multiplies only the time-axis channels by exp(i * ω * delta_frames)
    """
    if delta_frames == 0:
        return

    B, L, H, D = k_chunk.shape
    assert D % 2 == 0
    c = D // 2
    t_c = c - 2 * (c // 3)  # time channel complex dim
    h_c = c // 3
    w_c = c // 3
    # freqs -> time / height / width split
    freqs_t, _, _ = freqs.split([t_c, h_c, w_c], dim=1)  # [1024, t_c], complex

    #  Complex rotation factor corresponding to the delta (for the time axis)
    shift = abs(int(delta_frames))
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos
    mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    mult = mult.view(1, 1, 1, t_c)  # [1,1,1,t_c]

    # Convert only the time-axis channels to complex and multiply (in-place)
    time_ri = k_chunk[..., : 2 * t_c]  # [B,L,H,2*t_c]
    time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))  # [(B*L*H), t_c]
    time_cx = time_cx * mult.to(time_cx.dtype)  # delta rotate
    time_ri_new = torch.view_as_real(time_cx).reshape(B, L, H, t_c, 2).flatten(-2)  # [B,L,H,2*t_c]
    time_ri.copy_(time_ri_new.to(time_ri.dtype))  # in-place


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    """
    Apply causal RoPE (Rotary Position Embedding) to input tensor. Adapted from Infinity-RoPE (https://arxiv.org/abs/2511.20649)

    Args:
        x: Input tensor of shape [B, L, num_heads, head_dim]
        grid_sizes: Tensor of shape [B, 3] containing (F, H, W)
        freqs: RoPE frequencies
        start_frame: Starting frame index for sequential RoPE (default: 0)
        relative_frame_indices: Optional tensor of shape [F] specifying explicit frame indices
                               for Block-Relativistic RoPE. If provided, overrides start_frame.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        # override f with the length of provided relative frame indices for Block-Relativistic RoPE
        if relative_frame_indices is not None:
            f = len(relative_frame_indices)

        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

        # Use relative_frame_indices if provided (Block-Relativistic RoPE),
        # otherwise use sequential indices starting from start_frame
        if relative_frame_indices is not None:
            # relative_frame_indices should be a tensor of shape [f] with explicit frame indices
            frame_indices = relative_frame_indices.long()
        else:
            frame_indices = slice(start_frame, start_frame + f)

        freqs_i = torch.cat(
            [
                freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

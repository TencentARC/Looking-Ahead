"""Causal objective base and chunkwise timestep sampling."""

import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.inference_pipeline = None
        self._initialize_models(args, device)

    def _initialize_models(self, args, device):
        raise NotImplementedError

    def _get_timestep(self, min_timestep, max_timestep, batch_size, num_frame, num_frame_per_block):
        # Preserve the full per-frame draw before chunk grouping, including C0.
        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.long,
        )
        if self.independent_first_frame:
            regular = timestep[:, 1:].reshape(batch_size, -1, num_frame_per_block)
            regular[:, :, 1:] = regular[:, :, :1]
            return torch.cat((timestep[:, :1], regular.reshape(batch_size, -1)), dim=1)
        timestep = timestep.reshape(batch_size, -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, :1]
        return timestep.reshape(batch_size, num_frame)

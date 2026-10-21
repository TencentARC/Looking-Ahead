# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
# Convert dpm solver for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for Looking-Ahead: fixed flow scale and selected order-2 solver only.
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma

def retrieve_timesteps(scheduler, *, device=None, sigmas):
    """Set the selected DPM++ sigma schedule and return its timesteps."""
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps, len(scheduler.timesteps)

class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """Order-2 flow-matching sampler adapted from diffusers/Wan.

Only the selected deterministic algorithm and solver are retained."""
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(self, shift: Optional[float]=1.0):
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / 1000, 1000)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * 1000
        self.model_outputs = [None] * 2
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_timesteps(self, num_inference_steps: Union[int, None]=None, device: Union[str, torch.device]=None, sigmas: Optional[List[float]]=None, shift: Optional[Union[float, None]]=None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]
        if shift is None:
            shift = self.config.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_last = 0
        timesteps = sigmas * 1000
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * 2
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

    def _sigma_to_alpha_sigma_t(self, sigma):
        return (1 - sigma, sigma)

    def convert_model_output(self, model_output: torch.Tensor, *, sample: torch.Tensor) -> torch.Tensor:
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        return x0_pred

    def dpm_solver_first_order_update(self, model_output: torch.Tensor, *, sample: torch.Tensor) -> torch.Tensor:
        sigma_t, sigma_s = (self.sigmas[self.step_index + 1], self.sigmas[self.step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        x_t = sigma_t / sigma_s * sample - alpha_t * (torch.exp(-h) - 1.0) * model_output
        return x_t

    def multistep_dpm_solver_second_order_update(self, model_output_list: List[torch.Tensor], *, sample: torch.Tensor) -> torch.Tensor:
        sigma_t, sigma_s0, sigma_s1 = (self.sigmas[self.step_index + 1], self.sigmas[self.step_index], self.sigmas[self.step_index - 1])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        m0, m1 = (model_output_list[-1], model_output_list[-2])
        h, h_0 = (lambda_t - lambda_s0, lambda_s0 - lambda_s1)
        r0 = h_0 / h
        D0, D1 = (m0, 1.0 / r0 * (m0 - m1))
        x_t = sigma_t / sigma_s0 * sample - alpha_t * (torch.exp(-h) - 1.0) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output: torch.Tensor, timestep: Union[int, torch.Tensor], sample: torch.Tensor, return_dict: bool=True) -> Union[SchedulerOutput, Tuple]:
        """Advance the selected deterministic flow sampler by one step."""
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
        if self.step_index is None:
            self._init_step_index(timestep)
        lower_order_final = self.step_index == len(self.timesteps) - 1
        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(2 - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        sample = sample.to(torch.float32)
        if self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample)
        else:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample)
        if self.lower_order_nums < 2:
            self.lower_order_nums += 1
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

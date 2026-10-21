import torch


class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002):
        self.num_train_timesteps = 1000
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def clone_with_shift(self, shift: float):
        """Clone this schedule while replacing only ``shift``."""
        clone = type(self)(
            num_inference_steps=len(self.timesteps),
            shift=shift,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
        )
        training = hasattr(self, "linear_timesteps_weights")
        clone.set_timesteps(len(self.timesteps), training=training)
        clone.sigmas = clone.sigmas.to(device=self.sigmas.device, dtype=self.sigmas.dtype)
        clone.timesteps = clone.timesteps.to(
            device=self.timesteps.device, dtype=self.timesteps.dtype
        )
        if training:
            clone.linear_timesteps_weights = clone.linear_timesteps_weights.to(
                device=self.linear_timesteps_weights.device,
                dtype=self.linear_timesteps_weights.dtype,
            )
        return clone

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

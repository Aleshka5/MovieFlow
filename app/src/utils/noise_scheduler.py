from __future__ import annotations

import torch


class LinearNoiseScheduler:
    """Линейный DDPM scheduler (betas) для обучения DiT."""

    def __init__(
        self,
        *,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device | str,
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps должен быть > 0.")
        betas = torch.linspace(float(beta_start), float(beta_end), num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = int(num_train_timesteps)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * clean + sqrt_one_minus_alpha * noise

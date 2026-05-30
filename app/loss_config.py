from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LossConfig:
    weight_diffusion: float = 1.0
    weight_detail: float = 0.05
    weight_charbonnier: float = 0.0
    weight_fft: float = 0.0
    weight_temporal: float = 0.0
    charbonnier_epsilon: float = 1e-3
    fft_epsilon: float = 1e-6
    fft_use_log_magnitude: bool = True
    temporal_warmup_steps: int = 0
    temporal_detach_previous: bool = True

    def temporal_warmup_scale(self, global_step: int) -> float:
        if self.temporal_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(max(global_step, 0) + 1) / float(self.temporal_warmup_steps))

    @classmethod
    def from_settings(cls, settings: Any) -> "LossConfig":
        return cls(
            weight_diffusion=float(settings.loss_weight_diffusion),
            weight_detail=float(settings.loss_weight_detail),
            weight_charbonnier=float(settings.loss_weight_charbonnier),
            weight_fft=float(settings.loss_weight_fft),
            weight_temporal=float(settings.loss_weight_temporal),
            charbonnier_epsilon=float(settings.loss_charbonnier_epsilon),
            fft_epsilon=float(settings.loss_fft_epsilon),
            fft_use_log_magnitude=bool(settings.loss_fft_use_log_magnitude),
            temporal_warmup_steps=int(settings.loss_temporal_warmup_steps),
            temporal_detach_previous=bool(settings.loss_temporal_detach_previous),
        )

    def mlflow_param_dict(self) -> dict[str, Any]:
        return {
            "loss_weight_diffusion": self.weight_diffusion,
            "loss_weight_detail": self.weight_detail,
            "loss_weight_charbonnier": self.weight_charbonnier,
            "loss_weight_fft": self.weight_fft,
            "loss_weight_temporal": self.weight_temporal,
            "loss_charbonnier_epsilon": self.charbonnier_epsilon,
            "loss_fft_epsilon": self.fft_epsilon,
            "loss_fft_use_log_magnitude": self.fft_use_log_magnitude,
            "loss_temporal_warmup_steps": self.temporal_warmup_steps,
            "loss_temporal_detach_previous": self.temporal_detach_previous,
        }

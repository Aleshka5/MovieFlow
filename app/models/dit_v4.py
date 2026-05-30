from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from app.models.dit_v3 import DiTV3Model, DiTV3ModelConfig


def _parse_mask_columns_from_right(value: object) -> tuple[int, ...]:
    if value is None:
        return (1, 3, 5, 7)
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)  # type: ignore[arg-type]


@dataclass(slots=True)
class DiTV4ModelConfig(DiTV3ModelConfig):
    side_mask_columns_from_right: tuple[int, ...] = (1, 3, 5, 7)

    @classmethod
    def from_settings(cls, settings, architecture_name: str = "dit_v4") -> "DiTV4ModelConfig":
        return cls(
            architecture_name=architecture_name,
            latent_channels=settings.latent_channels,
            condition_height=settings.condition_height,
            condition_width=settings.condition_width,
            query_height=settings.query_height,
            query_width=settings.query_width,
            hidden_size=settings.hidden_size,
            num_attention_heads=settings.num_attention_heads,
            num_transformer_blocks=settings.num_transformer_blocks,
            mlp_ratio=settings.mlp_ratio,
            dropout=settings.dropout,
            query_patch_size=getattr(settings, "dit_v2_query_patch_size", 2),
            condition_patch_size=getattr(settings, "dit_v2_condition_patch_size", 2),
            previous_patch_size=getattr(settings, "dit_v3_previous_patch_size", 2),
            previous_cross_start_block=getattr(settings, "dit_v3_previous_cross_start_block", -1),
            side_mask_columns_from_right=_parse_mask_columns_from_right(
                getattr(settings, "dit_v4_side_mask_columns_from_right", (1, 3, 5, 7))
            ),
        )


class DiTV4Model(DiTV3Model):
    """DiT v4: v3 cross-attention with pre-masked previous side latents."""

    config: DiTV4ModelConfig

    def __init__(self, config: DiTV4ModelConfig) -> None:
        super().__init__(config)
        invalid_offsets = [offset for offset in config.side_mask_columns_from_right if offset <= 0]
        if invalid_offsets:
            raise ValueError(
                "side_mask_columns_from_right должен содержать положительные 1-based offsets, "
                f"получено: {invalid_offsets}"
            )

    def mask_sides_latents(self, latents: Tensor) -> Tensor:
        self._validate_channels_first(
            latents,
            source="previous_sides_latents",
            expected_height=self.config.query_height,
            expected_width=self.config.query_width,
        )
        masked = latents.clone()
        width = masked.shape[-1]
        for offset_from_right in self.config.side_mask_columns_from_right:
            column_index = width - offset_from_right
            if column_index >= 0:
                masked[..., column_index] = 0
        return masked

    def forward(
        self,
        noisy_query_latents: Tensor,
        condition_latents: Tensor,
        previous_sides_latents: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        return super().forward(
            noisy_query_latents=noisy_query_latents,
            condition_latents=condition_latents,
            previous_sides_latents=self.mask_sides_latents(previous_sides_latents),
            timesteps=timesteps,
        )

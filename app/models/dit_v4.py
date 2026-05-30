from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from app.models.dit_v2 import _modulate
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
        self.first_frame_embedding = nn.Embedding(2, config.hidden_size)
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

    def _first_frame_states(
        self,
        is_first_frame: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if is_first_frame is None:
            first_frame_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            if not isinstance(is_first_frame, Tensor):
                raise TypeError(
                    "is_first_frame должен быть torch.Tensor с bool/int значениями по batch."
                )
            first_frame_values = is_first_frame.to(device=device)
            if first_frame_values.ndim == 0:
                first_frame_values = first_frame_values.reshape(1)
            if first_frame_values.ndim == 2 and first_frame_values.shape[1] == 1:
                first_frame_values = first_frame_values.reshape(-1)
            if first_frame_values.shape[0] == 1 and batch_size > 1:
                first_frame_values = first_frame_values.expand(batch_size)
            if first_frame_values.ndim != 1 or first_frame_values.shape[0] != batch_size:
                raise ValueError(
                    "Размер is_first_frame должен совпадать с batch size: "
                    f"is_first_frame={tuple(first_frame_values.shape)}, batch={batch_size}."
                )
            first_frame_ids = first_frame_values.to(dtype=torch.bool).to(dtype=torch.long)
        return self.first_frame_embedding(first_frame_ids).to(dtype=dtype)

    def _adaln_states(
        self,
        timestep_states: Tensor,
        condition_states: Tensor,
        first_frame_states: Tensor,
    ) -> Tensor:
        return timestep_states + condition_states.mean(dim=1) + first_frame_states

    def forward(
        self,
        noisy_query_latents: Tensor,
        condition_latents: Tensor,
        previous_sides_latents: Tensor,
        timesteps: Tensor,
        is_first_frame: Tensor | None = None,
    ) -> Tensor:
        self._validate_channels_first(
            noisy_query_latents,
            source="noisy_query_latents",
            expected_height=self.config.query_height,
            expected_width=self.config.query_width,
        )
        self._validate_channels_first(
            condition_latents,
            source="condition_latents",
            expected_height=self.config.condition_height,
            expected_width=self.config.condition_width,
        )
        previous_sides_latents = self.mask_sides_latents(previous_sides_latents)

        query_patches = self._patchify(
            noisy_query_latents,
            patch_embedding=self.query_patch_embedding,
            patch_size=self.config.query_patch_size,
            source="noisy_query_latents",
        )
        condition_patches = self._patchify(
            condition_latents,
            patch_embedding=self.condition_patch_embedding,
            patch_size=self.config.condition_patch_size,
            source="condition_latents",
        )
        previous_patches = self._patchify(
            previous_sides_latents,
            patch_embedding=self.previous_patch_embedding,
            patch_size=self.config.previous_patch_size,
            source="previous_sides_latents",
        )

        query_states = self._tokens_from_patches(query_patches)
        condition_states = self._tokens_from_patches(condition_patches)
        previous_states = self._tokens_from_patches(previous_patches)

        query_positional = self.query_positional_embedding.to(dtype=query_states.dtype)
        condition_positional = self.condition_positional_embedding.to(dtype=condition_states.dtype)
        previous_positional = self.previous_positional_embedding.to(dtype=previous_states.dtype)
        query_states = query_states + query_positional
        condition_states = condition_states + condition_positional
        previous_states = previous_states + previous_positional

        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] == 1 and query_states.shape[0] > 1:
            timesteps = timesteps.expand(query_states.shape[0])
        if timesteps.shape[0] != query_states.shape[0]:
            raise ValueError(
                "Размер timesteps должен совпадать с batch size: "
                f"timesteps={timesteps.shape[0]}, batch={query_states.shape[0]}."
            )

        timestep_states = self._sinusoidal_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.config.hidden_size,
        ).to(dtype=query_states.dtype)
        timestep_states = self.timestep_mlp(timestep_states)
        first_frame_states = self._first_frame_states(
            is_first_frame,
            batch_size=query_states.shape[0],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        adaln_states = self._adaln_states(timestep_states, condition_states, first_frame_states)

        for block_index, block in enumerate(self.blocks):
            query_states = block(
                query_states,
                condition_states,
                previous_states,
                adaln_states,
                use_previous_cross=block_index >= self.previous_cross_start_block,
            )

        shift, scale = self.final_modulation(adaln_states).chunk(2, dim=1)
        query_states = _modulate(self.final_norm(query_states), shift, scale)
        return self._unpatchify(
            query_states,
            output_height=self.config.query_height,
            output_width=self.config.query_width,
        )

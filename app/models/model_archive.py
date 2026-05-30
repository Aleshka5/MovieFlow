from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from app.models.dit_v2 import BaseModelConfig, DiTV2Model, DiTV2ModelConfig
from app.models.dit_v3 import DiTV3Model, DiTV3ModelConfig
from app.models.dit_v4 import DiTV4Model, DiTV4ModelConfig, _parse_mask_columns_from_right


@dataclass(frozen=True, slots=True)
class ModelArchiveEntry:
    architecture_name: str
    config_cls: type[BaseModelConfig]
    model_cls: type[nn.Module]


MODEL_ARCHIVE: dict[str, ModelArchiveEntry] = {
    "dit_v2": ModelArchiveEntry(
        architecture_name="dit_v2",
        config_cls=DiTV2ModelConfig,
        model_cls=DiTV2Model,
    ),
    "dit_v3": ModelArchiveEntry(
        architecture_name="dit_v3",
        config_cls=DiTV3ModelConfig,
        model_cls=DiTV3Model,
    ),
    "dit_v4": ModelArchiveEntry(
        architecture_name="dit_v4",
        config_cls=DiTV4ModelConfig,
        model_cls=DiTV4Model,
    ),
}


def list_architectures() -> tuple[str, ...]:
    return tuple(sorted(MODEL_ARCHIVE.keys()))


def get_model_entry(architecture_name: str) -> ModelArchiveEntry:
    normalized = architecture_name.lower()
    if normalized not in MODEL_ARCHIVE:
        available = ", ".join(list_architectures())
        raise ValueError(
            f"Неизвестная архитектура '{architecture_name}'. Доступные варианты: {available}."
        )
    return MODEL_ARCHIVE[normalized]


def build_model(config: BaseModelConfig) -> nn.Module:
    entry = get_model_entry(config.architecture_name)
    if not isinstance(config, entry.config_cls):
        raise TypeError(
            f"Конфиг архитектуры '{entry.architecture_name}' должен быть типа "
            f"{entry.config_cls.__name__}, получено: {type(config).__name__}."
        )
    return entry.model_cls(config)


def build_config_from_settings(settings: Any, architecture_name: str) -> BaseModelConfig:
    entry = get_model_entry(architecture_name)
    if entry.config_cls is DiTV2ModelConfig:
        return DiTV2ModelConfig.from_settings(settings, architecture_name=entry.architecture_name)
    if entry.config_cls is DiTV3ModelConfig:
        return DiTV3ModelConfig.from_settings(settings, architecture_name=entry.architecture_name)
    if entry.config_cls is DiTV4ModelConfig:
        return DiTV4ModelConfig.from_settings(settings, architecture_name=entry.architecture_name)
    raise NotImplementedError(
        f"Создание конфига из Settings для архитектуры '{entry.architecture_name}' не реализовано."
    )


def _parse_int(params: dict[str, str], key: str, fallback: int) -> int:
    value = params.get(key)
    return int(value) if value is not None else fallback


def _parse_float(params: dict[str, str], key: str, fallback: float) -> float:
    value = params.get(key)
    return float(value) if value is not None else fallback


def build_config_from_run_params(
    settings: Any, architecture_name: str, run_params: dict[str, str]
) -> BaseModelConfig:
    entry = get_model_entry(architecture_name)
    common = {
        "architecture_name": entry.architecture_name,
        "latent_channels": _parse_int(run_params, "latent_channels", settings.latent_channels),
    }
    if entry.config_cls is DiTV2ModelConfig:
        return DiTV2ModelConfig(
            **common,
            condition_height=_parse_int(run_params, "condition_height", settings.condition_height),
            condition_width=_parse_int(run_params, "condition_width", settings.condition_width),
            query_height=_parse_int(run_params, "query_height", settings.query_height),
            query_width=_parse_int(run_params, "query_width", settings.query_width),
            hidden_size=_parse_int(run_params, "hidden_size", settings.hidden_size),
            num_attention_heads=_parse_int(
                run_params, "num_attention_heads", settings.num_attention_heads
            ),
            num_transformer_blocks=_parse_int(
                run_params, "num_transformer_blocks", settings.num_transformer_blocks
            ),
            mlp_ratio=_parse_float(run_params, "mlp_ratio", settings.mlp_ratio),
            dropout=_parse_float(run_params, "dropout", settings.dropout),
            query_patch_size=_parse_int(
                run_params,
                "query_patch_size",
                getattr(settings, "dit_v2_query_patch_size", 2),
            ),
            condition_patch_size=_parse_int(
                run_params,
                "condition_patch_size",
                getattr(settings, "dit_v2_condition_patch_size", 2),
            ),
        )
    if entry.config_cls is DiTV3ModelConfig:
        return DiTV3ModelConfig(
            **common,
            condition_height=_parse_int(run_params, "condition_height", settings.condition_height),
            condition_width=_parse_int(run_params, "condition_width", settings.condition_width),
            query_height=_parse_int(run_params, "query_height", settings.query_height),
            query_width=_parse_int(run_params, "query_width", settings.query_width),
            hidden_size=_parse_int(run_params, "hidden_size", settings.hidden_size),
            num_attention_heads=_parse_int(
                run_params, "num_attention_heads", settings.num_attention_heads
            ),
            num_transformer_blocks=_parse_int(
                run_params, "num_transformer_blocks", settings.num_transformer_blocks
            ),
            mlp_ratio=_parse_float(run_params, "mlp_ratio", settings.mlp_ratio),
            dropout=_parse_float(run_params, "dropout", settings.dropout),
            query_patch_size=_parse_int(
                run_params,
                "query_patch_size",
                getattr(settings, "dit_v2_query_patch_size", 2),
            ),
            condition_patch_size=_parse_int(
                run_params,
                "condition_patch_size",
                getattr(settings, "dit_v2_condition_patch_size", 2),
            ),
            previous_patch_size=_parse_int(
                run_params,
                "previous_patch_size",
                getattr(settings, "dit_v3_previous_patch_size", 2),
            ),
            previous_cross_start_block=_parse_int(
                run_params,
                "previous_cross_start_block",
                getattr(settings, "dit_v3_previous_cross_start_block", -1),
            ),
        )
    if entry.config_cls is DiTV4ModelConfig:
        return DiTV4ModelConfig(
            **common,
            condition_height=_parse_int(run_params, "condition_height", settings.condition_height),
            condition_width=_parse_int(run_params, "condition_width", settings.condition_width),
            query_height=_parse_int(run_params, "query_height", settings.query_height),
            query_width=_parse_int(run_params, "query_width", settings.query_width),
            hidden_size=_parse_int(run_params, "hidden_size", settings.hidden_size),
            num_attention_heads=_parse_int(
                run_params, "num_attention_heads", settings.num_attention_heads
            ),
            num_transformer_blocks=_parse_int(
                run_params, "num_transformer_blocks", settings.num_transformer_blocks
            ),
            mlp_ratio=_parse_float(run_params, "mlp_ratio", settings.mlp_ratio),
            dropout=_parse_float(run_params, "dropout", settings.dropout),
            query_patch_size=_parse_int(
                run_params,
                "query_patch_size",
                getattr(settings, "dit_v2_query_patch_size", 2),
            ),
            condition_patch_size=_parse_int(
                run_params,
                "condition_patch_size",
                getattr(settings, "dit_v2_condition_patch_size", 2),
            ),
            previous_patch_size=_parse_int(
                run_params,
                "previous_patch_size",
                getattr(settings, "dit_v3_previous_patch_size", 2),
            ),
            previous_cross_start_block=_parse_int(
                run_params,
                "previous_cross_start_block",
                getattr(settings, "dit_v3_previous_cross_start_block", -1),
            ),
            side_mask_columns_from_right=_parse_mask_columns_from_right(
                run_params.get(
                    "side_mask_columns_from_right",
                    getattr(settings, "dit_v4_side_mask_columns_from_right", "1,3,5,7"),
                )
            ),
        )
    raise NotImplementedError(
        f"Построение конфига из run_params для '{entry.architecture_name}' не реализовано."
    )

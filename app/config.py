from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.loss_config import LossConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    movies_bucket_name: str = Field(default="movies", alias="MOVIES_BUCKET_NAME")
    mlflow_bucket_name: str = Field(default="mlflow", alias="MLFLOW_BUCKET_NAME")

    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="ru-1", alias="AWS_DEFAULT_REGION")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    mlflow_s3_endpoint_url: str | None = Field(default=None, alias="MLFLOW_S3_ENDPOINT_URL")

    gpu_support: bool = Field(default=True, alias="GPU_SUPPORT")
    datasets_folder: Path | None = Field(default=None, alias="DATASETS_FOLDER")

    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_registry_uri: str | None = Field(default=None, alias="MLFLOW_REGISTRY_URI")
    mlflow_tracking_username: str | None = Field(
        default=None,
        alias="MLFLOW_TRACKING_USERNAME",
    )
    mlflow_tracking_password: str | None = Field(
        default=None,
        alias="MLFLOW_TRACKING_PASSWORD",
    )
    mlflow_experiment_name: str = Field(default="DiT", alias="MLFLOW_EXPERIMENT_NAME")
    mlflow_registered_model_name: str = Field(
        default="DiTModel",
        alias="MLFLOW_REGISTERED_MODEL_NAME",
    )
    dataset_download_dir: Path = Field(
        default=PROJECT_ROOT / "dataset", alias="DATASET_DOWNLOAD_DIR"
    )
    mlflow_dataset_artifact_path: str = Field(
        default="scene_sft",
        alias="MLFLOW_DATASET_ARTIFACT_PATH",
    )
    mlflow_dataset_run_ids: str = Field(
        default="",
        alias="MLFLOW_DATASET_RUN_IDS",
    )

    latent_channels: int = Field(default=16, alias="LATENT_CHANNELS")
    model_architecture: str = Field(default="dit_v2", alias="MODEL_ARCHITECTURE")
    condition_key: str = Field(default="source_image_center", alias="CONDITION_KEY")
    target_key: str = Field(default="source_image_sides", alias="TARGET_KEY")
    previous_sides_key: str = Field(default="previous_frame_sides", alias="PREVIOUS_SIDES_KEY")
    condition_height: int = Field(default=30, alias="CONDITION_HEIGHT")
    condition_width: int = Field(default=27, alias="CONDITION_WIDTH")
    query_height: int = Field(default=30, alias="QUERY_HEIGHT")
    query_width: int = Field(default=8, alias="QUERY_WIDTH")
    hidden_size: int = Field(default=256, alias="HIDDEN_SIZE")
    num_attention_heads: int = Field(default=8, alias="NUM_ATTENTION_HEADS")
    num_transformer_blocks: int = Field(default=16, alias="NUM_TRANSFORMER_BLOCKS")
    mlp_ratio: float = Field(default=4.0, alias="MLP_RATIO")
    dropout: float = Field(default=0.05, alias="DROPOUT")
    dit_v2_query_patch_size: int = Field(default=2, alias="DIT_V2_QUERY_PATCH_SIZE")
    dit_v2_condition_patch_size: int = Field(default=3, alias="DIT_V2_CONDITION_PATCH_SIZE")
    dit_v3_previous_patch_size: int = Field(default=2, alias="DIT_V3_PREVIOUS_PATCH_SIZE")
    dit_v3_previous_cross_start_block: int = Field(
        default=-1, alias="DIT_V3_PREVIOUS_CROSS_START_BLOCK"
    )
    dit_v4_side_mask_columns_from_right: str = Field(
        default="1,3,5,7",
        alias="DIT_V4_SIDE_MASK_COLUMNS_FROM_RIGHT",
    )
    use_autocast: bool = Field(default=True, alias="USE_AUTOCAST")
    autocast_dtype: str = Field(default="bfloat16", alias="AUTOCAST_DTYPE")

    train_batch_size: int = Field(default=8, alias="TRAIN_BATCH_SIZE")
    train_num_workers: int = Field(default=0, alias="TRAIN_NUM_WORKERS")
    train_max_steps: int = Field(default=10000, alias="TRAIN_MAX_STEPS")
    val_every_n_logs: int = Field(default=0, alias="VAL_EVERY_N_LOGS")
    preview_every_n_steps: int = Field(default=2000, alias="PREVIEW_EVERY_N_STEPS")
    preview_images_count: int = Field(default=4, alias="PREVIEW_IMAGES_COUNT")
    learning_rate: float = Field(default=1e-4, alias="LEARNING_RATE")
    weight_decay: float = Field(default=1e-2, alias="WEIGHT_DECAY")
    grad_clip_norm: float = Field(default=1.0, alias="GRAD_CLIP_NORM")
    log_every_steps: int = Field(default=10, alias="LOG_EVERY_STEPS")
    checkpoint_every_steps: int = Field(default=5000, alias="CHECKPOINT_EVERY_STEPS")
    seed: int = Field(default=42, alias="SEED")
    prediction_type: str = Field(default="epsilon_v_hybrid", alias="PREDICTION_TYPE")
    epsilon_v_hybrid_lambda: float = Field(default=0.5, alias="EPSILON_V_HYBRID_LAMBDA")
    min_snr_gamma: float = Field(default=5.0, alias="MIN_SNR_GAMMA")
    loss_weight_diffusion: float = Field(default=1.0, alias="LOSS_WEIGHT_DIFFUSION")
    loss_weight_detail: float = Field(
        default=0.05,
        validation_alias=AliasChoices("LOSS_WEIGHT_DETAIL", "DETAIL_LOSS_WEIGHT"),
    )
    loss_weight_charbonnier: float = Field(default=0.0, alias="LOSS_WEIGHT_CHARBONNIER")
    loss_weight_fft: float = Field(default=0.0, alias="LOSS_WEIGHT_FFT")
    loss_weight_temporal: float = Field(default=0.0, alias="LOSS_WEIGHT_TEMPORAL")
    loss_charbonnier_epsilon: float = Field(default=1e-3, alias="LOSS_CHARBONNIER_EPSILON")
    loss_fft_epsilon: float = Field(default=1e-6, alias="LOSS_FFT_EPSILON")
    loss_fft_use_log_magnitude: bool = Field(default=True, alias="LOSS_FFT_USE_LOG_MAGNITUDE")
    loss_temporal_warmup_steps: int = Field(default=0, alias="LOSS_TEMPORAL_WARMUP_STEPS")
    loss_temporal_detach_previous: bool = Field(default=True, alias="LOSS_TEMPORAL_DETACH_PREVIOUS")
    use_ema: bool = Field(default=True, alias="USE_EMA")
    ema_decay: float = Field(default=0.999, alias="EMA_DECAY")

    num_train_timesteps: int = Field(default=1000, alias="NUM_TRAIN_TIMESTEPS")
    beta_start: float = Field(default=1e-4, alias="BETA_START")
    beta_end: float = Field(default=2e-2, alias="BETA_END")

    @property
    def default_device(self) -> str:
        return "cuda" if self.gpu_support else "cpu"

    @property
    def dataset_run_ids(self) -> list[str]:
        return [
            run_id.strip() for run_id in self.mlflow_dataset_run_ids.split(",") if run_id.strip()
        ]

    @property
    def loss_config(self) -> LossConfig:
        return LossConfig.from_settings(self)

    @property
    def detail_loss_weight(self) -> float:
        # Backward-compatible alias for older training scripts.
        return float(self.loss_weight_detail)

    def mlflow_param_dict(self) -> dict[str, Any]:
        params = {
            "latent_channels": self.latent_channels,
            "architecture_name": self.model_architecture,
            "condition_key": self.condition_key,
            "target_key": self.target_key,
            "previous_sides_key": self.previous_sides_key,
            "condition_height": self.condition_height,
            "condition_width": self.condition_width,
            "query_height": self.query_height,
            "query_width": self.query_width,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "query_patch_size": self.dit_v2_query_patch_size,
            "condition_patch_size": self.dit_v2_condition_patch_size,
            "previous_patch_size": self.dit_v3_previous_patch_size,
            "previous_cross_start_block": self.dit_v3_previous_cross_start_block,
            "side_mask_columns_from_right": self.dit_v4_side_mask_columns_from_right,
            "use_autocast": self.use_autocast,
            "autocast_dtype": self.autocast_dtype,
            "train_batch_size": self.train_batch_size,
            "train_num_workers": self.train_num_workers,
            "train_max_steps": self.train_max_steps,
            "val_every_n_logs": self.val_every_n_logs,
            "preview_every_n_steps": self.preview_every_n_steps,
            "preview_images_count": self.preview_images_count,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "log_every_steps": self.log_every_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "seed": self.seed,
            "prediction_type": self.prediction_type,
            "epsilon_v_hybrid_lambda": self.epsilon_v_hybrid_lambda,
            "min_snr_gamma": self.min_snr_gamma,
            "detail_loss_weight": self.detail_loss_weight,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "mlflow_registered_model_name": self.mlflow_registered_model_name,
        }
        params.update(self.loss_config.mlflow_param_dict())
        return params


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

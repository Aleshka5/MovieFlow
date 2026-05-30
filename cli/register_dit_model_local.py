from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import get_settings
from app.models.model_archive import build_config_from_run_params, build_model
from app.src.repositories.mlflow import MLflowRepository


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Регистрация локальной DiT-модели в MLflow Model Registry "
            "по путям к весам и JSON-конфигу."
        )
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        required=True,
        help="Путь к локальному файлу весов (.pt/.pth).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Путь к JSON-конфигу (run params), из которого восстанавливается архитектура.",
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=settings.mlflow_registered_model_name,
        help="Имя модели в MLflow Model Registry.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="",
        help=(
            "Явное имя архитектуры (например, dit_v2/dit_v3). "
            "Если не задано, берется из config-path -> architecture_name."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="register-local-dit-model",
        help="Имя MLflow run.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=settings.mlflow_tracking_uri,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--registry-uri",
        type=str,
        default=settings.mlflow_registry_uri,
        help="MLflow registry URI.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=settings.mlflow_experiment_name,
        help="Имя MLflow experiment.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="dit_final",
        help="Artifact path для регистрации pytorch flavor модели.",
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Строгая загрузка state_dict в модель (по умолчанию: включено).",
    )
    return parser.parse_args()


def _load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in config, got: {type(payload).__name__}")
    return payload


def _to_run_params(raw_config: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in raw_config.items()}


def _extract_state_dict(weights_payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(weights_payload, dict):
        if "model_state_dict" in weights_payload and isinstance(
            weights_payload["model_state_dict"], dict
        ):
            return weights_payload["model_state_dict"]
        if "state_dict" in weights_payload and isinstance(weights_payload["state_dict"], dict):
            return weights_payload["state_dict"]
        if weights_payload and all(
            isinstance(value, torch.Tensor) for value in weights_payload.values()
        ):
            return weights_payload
    raise ValueError(
        "Не удалось извлечь state_dict из файла весов. "
        "Ожидается checkpoint с ключом model_state_dict/state_dict "
        "или напрямую словарь параметров модели."
    )


def _build_input_examples(
    *,
    latent_channels: int,
    condition_height: int,
    condition_width: int,
    query_height: int,
    query_width: int,
    num_train_timesteps: int,
    architecture_name: str,
) -> dict[str, np.ndarray]:
    input_examples = {
        "noisy_query_latents": np.random.randn(
            1, latent_channels, query_height, query_width
        ).astype(np.float32),
        "condition_latents": np.random.randn(
            1, latent_channels, condition_height, condition_width
        ).astype(np.float32),
        "timesteps": np.array([max(0, num_train_timesteps // 2)], dtype=np.int64),
    }
    if architecture_name.strip().lower() in {"dit_v3", "dit_v4"}:
        input_examples["previous_sides_latents"] = np.random.randn(
            1, latent_channels, query_height, query_width
        ).astype(np.float32)
    return input_examples


def main() -> None:
    args = _parse_args()
    settings = get_settings()

    weights_path = args.weights_path.resolve()
    config_path = args.config_path.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    raw_config = _load_json_config(config_path)
    run_params = _to_run_params(raw_config)

    architecture_name = (
        args.architecture.strip()
        or run_params.get("architecture_name", "").strip()
        or settings.model_architecture
    )
    model_config = build_config_from_run_params(
        settings=settings,
        architecture_name=architecture_name,
        run_params=run_params,
    )
    model = build_model(model_config)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=bool(args.strict_load))
    model.eval()

    if missing_keys:
        _log(f"[weights] missing_keys={len(missing_keys)}")
    if unexpected_keys:
        _log(f"[weights] unexpected_keys={len(unexpected_keys)}")
    _log(
        "Веса загружены: "
        f"architecture={model_config.architecture_name}, strict={bool(args.strict_load)}"
    )

    mlflow_repo = MLflowRepository(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        experiment_name=args.experiment_name,
        settings=settings,
    )
    input_examples = _build_input_examples(
        latent_channels=int(getattr(model_config, "latent_channels")),
        condition_height=int(getattr(model_config, "condition_height")),
        condition_width=int(getattr(model_config, "condition_width")),
        query_height=int(getattr(model_config, "query_height")),
        query_width=int(getattr(model_config, "query_width")),
        num_train_timesteps=int(
            run_params.get("num_train_timesteps", settings.num_train_timesteps)
        ),
        architecture_name=model_config.architecture_name,
    )

    with mlflow_repo.start_run(
        run_name=args.run_name, tags={"pipeline": "local-model-registration"}
    ):
        mlflow_repo.log_params(
            {
                "register_source": "local_paths",
                "weights_file_name": weights_path.name,
                "weights_file_path": str(weights_path),
                "config_file_name": config_path.name,
                "config_file_path": str(config_path),
                "architecture_name": model_config.architecture_name,
                "registered_model_name": args.registered_model_name,
                "strict_load": bool(args.strict_load),
            }
        )
        mlflow_repo.log_config(raw_config, artifact_path="configs/local_registration_config.json")
        mlflow_repo.mlflow.log_artifact(str(weights_path), artifact_path="source_artifacts")
        mlflow_repo.mlflow.log_artifact(str(config_path), artifact_path="source_artifacts")

        registration = mlflow_repo.register_final_model(
            model=model,
            registered_model_name=args.registered_model_name,
            model_config=raw_config,
            input_examples=input_examples,
            architecture_name=model_config.architecture_name,
            artifact_path=args.artifact_path,
        )
        mlflow_repo.log_params(
            {
                "registered_model_name": registration["name"],
                "registered_model_version": registration["version"],
                "registered_model_uri": registration["model_uri"],
            }
        )

    _log(
        "Model registered in MLflow Registry: "
        f"{registration['name']} v{registration['version']} ({registration['model_uri']})"
    )


if __name__ == "__main__":
    main()

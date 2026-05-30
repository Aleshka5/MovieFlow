from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from app.config import Settings


def download_encoded_dataset(settings: Settings, *, destination: Path | None = None) -> Path:
    """Скачивает encoded SFT-артефакты из MLflow runs в локальную папку."""
    run_ids = settings.dataset_run_ids
    if not run_ids:
        raise ValueError(
            "Не задан MLFLOW_DATASET_RUN_IDS. Укажите run id(ы) через запятую или передайте --dataset-dir."
        )

    target_dir = destination or settings.dataset_download_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    for run_id in run_ids:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=settings.mlflow_dataset_artifact_path,
            dst_path=str(target_dir / run_id),
        )
        _ = client.get_run(run_id)
        print(f"[dataset] downloaded run={run_id} -> {local_path}", flush=True)

    return target_dir

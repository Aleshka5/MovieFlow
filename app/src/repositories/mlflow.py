from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.exceptions import MlflowException, RestException
from mlflow.models import Model, infer_signature
from mlflow.tracking import MlflowClient as MlflowTrackingClient

from app.config import Settings, get_settings
# from app.src.repositories import BaseModelRegistry

_log = logging.getLogger(__name__)

_SECRET_PARTS = ("secret", "password", "token", "key")


class MLflowRepository:
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        experiment_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracking_uri = tracking_uri or self.settings.mlflow_tracking_uri
        self.registry_uri = registry_uri or self.settings.mlflow_registry_uri or self.tracking_uri
        self.experiment_name = experiment_name or self.settings.mlflow_experiment_name
        self._configure()
        self._ensure_experiment_exists()

    def _configure(self) -> None:
        if not self.tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI must be set")
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        self._set_env_default("MLFLOW_TRACKING_URI", self.tracking_uri)
        self._set_env_default("MLFLOW_TRACKING_USERNAME", self.settings.mlflow_tracking_username)
        self._set_env_default("MLFLOW_TRACKING_PASSWORD", self.settings.mlflow_tracking_password)
        if self.registry_uri:
            self._set_env_default("MLFLOW_REGISTRY_URI", self.registry_uri)
        self._set_env_default("MLFLOW_S3_ENDPOINT_URL", self.settings.mlflow_s3_endpoint_url)
        self._set_env_default("AWS_ACCESS_KEY_ID", self.settings.aws_access_key_id)
        self._set_env_default("AWS_SECRET_ACCESS_KEY", self.settings.aws_secret_access_key)
        self._set_env_default("AWS_DEFAULT_REGION", self.settings.aws_default_region)

    @staticmethod
    def _set_env_default(name: str, value: str | None) -> None:
        if value and not os.environ.get(name):
            os.environ[name] = value

    @staticmethod
    def _normalize_param_value(value: Any) -> str:
        if isinstance(value, (bool, int, float, str)) or value is None:
            return str(value)
        if isinstance(value, Path):
            return str(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _sanitize_key(key: str) -> bool:
        lowered = key.lower()
        return not any(token in lowered for token in _SECRET_PARTS)

    def _ensure_experiment_exists(self) -> str:
        mlflow.set_experiment(self.experiment_name)
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            raise RuntimeError(f"Failed to create or resolve experiment {self.experiment_name!r}")
        return exp.experiment_id

    @contextmanager
    def start_run(self, *, run_name: str | None = None, tags: dict[str, Any] | None = None):
        clean_tags = {k: self._normalize_param_value(v) for k, v in (tags or {}).items()}
        with mlflow.start_run(run_name=run_name, tags=clean_tags) as run:
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        payload = {
            k: self._normalize_param_value(v) for k, v in params.items() if self._sanitize_key(k)
        }
        if payload:
            mlflow.log_params(payload)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        payload = {k: float(v) for k, v in metrics.items()}
        if payload:
            mlflow.log_metrics(payload, step=step)

    def log_config(
        self, config: dict[str, Any], artifact_path: str = "configs/train_config.json"
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mlflow-config-") as tmp:
            file_path = Path(tmp) / "train_config.json"
            file_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            mlflow.log_artifact(str(file_path), artifact_path=str(Path(artifact_path).parent))

    @property
    def mlflow(self):
        return mlflow

    def log_checkpoint(
        self,
        model_state: dict[str, Any] | None = None,
        *,
        step: int,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if model_state is None:
            if model is None:
                raise ValueError("log_checkpoint: передайте model_state или model.")
            model_state = {
                "step": step,
                "model_state_dict": model.state_dict(),
            }
            if optimizer is not None:
                model_state["optimizer_state_dict"] = optimizer.state_dict()
            if extra:
                model_state.update(extra)
        with tempfile.TemporaryDirectory(prefix="mlflow-ckpt-") as tmp:
            checkpoint_name = f"step_{step:08d}.pt"
            path = Path(tmp) / checkpoint_name
            torch.save(model_state, path)
            mlflow.log_artifact(str(path), artifact_path="checkpoints")
        return f"checkpoints/{checkpoint_name}"

    def log_model_weights(
        self,
        *,
        model: torch.nn.Module,
        artifact_relpath: str = "weights/final_model.pt",
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="mlflow-weights-") as tmp:
            path = Path(tmp) / Path(artifact_relpath).name
            torch.save(model.cpu().state_dict(), path)
            artifact_dir = str(Path(artifact_relpath).parent)
            if artifact_dir and artifact_dir != ".":
                mlflow.log_artifact(str(path), artifact_path=artifact_dir)
            else:
                mlflow.log_artifact(str(path))
        return artifact_relpath

    def log_preview_image(
        self,
        image_rgb: np.ndarray,
        *,
        file_name: str,
        artifact_path: str = "previews",
        run_id: str | None = None,
    ) -> str:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected image [H, W, 3], got shape={image_rgb.shape}")
        image = np.clip(image_rgb, 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory(prefix="mlflow-preview-") as tmp:
            path = Path(tmp) / file_name
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(path), image_bgr)
            if not ok:
                raise RuntimeError(f"Failed to encode preview image: {path}")
            if run_id is None:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
            else:
                client = MlflowTrackingClient(tracking_uri=self.tracking_uri)
                client.log_artifact(run_id, str(path), artifact_path=artifact_path)
        return f"{artifact_path}/{file_name}"

    def register_final_model(
        self,
        *,
        model: torch.nn.Module,
        registered_model_name: str,
        model_config: dict[str, Any] | None = None,
        input_examples: dict[str, np.ndarray] | None = None,
        architecture_name: str | None = None,
        artifact_path: str = "model",
    ) -> dict[str, str]:
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("register_final_model requires an active mlflow run")

        cpu_model = model.cpu().eval()
        if input_examples is None:
            input_examples = {}

        with tempfile.TemporaryDirectory(prefix="mlflow-final-weights-") as tmp:
            final_weights = Path(tmp) / "final_model.pt"
            torch.save(cpu_model.state_dict(), final_weights)
            mlflow.log_artifact(str(final_weights), artifact_path="weights")

        output_numpy: np.ndarray | None = None
        signature = None
        if input_examples:
            tensors = {
                key: torch.from_numpy(np.asarray(value))
                for key, value in input_examples.items()
            }
            with torch.inference_mode():
                if {"high_res", "side", "low_res", "camera"}.issubset(tensors):
                    sample_output = cpu_model(
                        tensors["high_res"],
                        tensors["side"],
                        tensors["low_res"],
                        tensors["camera"],
                    )
                elif {"noisy_query_latents", "condition_latents", "timesteps"}.issubset(tensors):
                    timesteps = tensors["timesteps"]
                    if timesteps.dtype != torch.long:
                        timesteps = timesteps.to(torch.long).reshape(-1)
                    model_inputs: dict[str, torch.Tensor] = {
                        "noisy_query_latents": tensors["noisy_query_latents"].float(),
                        "condition_latents": tensors["condition_latents"].float(),
                        "timesteps": timesteps,
                    }
                    if "previous_sides_latents" in tensors:
                        model_inputs["previous_sides_latents"] = tensors["previous_sides_latents"].float()
                    elif (
                        architecture_name is not None
                        and architecture_name.strip().lower() in {"dit_v3", "dit_v4"}
                    ):
                        model_inputs["previous_sides_latents"] = torch.zeros_like(
                            model_inputs["noisy_query_latents"]
                        )
                    sample_output = cpu_model(**model_inputs)
                else:
                    sample_output = cpu_model(
                        **{key: value.float() if value.is_floating_point() else value for key, value in tensors.items()}
                    )
            output_numpy = sample_output.detach().cpu().numpy()
            try:
                if {"high_res", "side", "low_res", "camera"}.issubset(input_examples):
                    signature = infer_signature(
                        [
                            input_examples["high_res"],
                            input_examples["side"],
                            input_examples["low_res"],
                            input_examples["camera"],
                        ],
                        output_numpy,
                    )
                elif {"noisy_query_latents", "condition_latents", "timesteps"}.issubset(input_examples):
                    signature_inputs: dict[str, np.ndarray] = {
                        "noisy_query_latents": input_examples["noisy_query_latents"],
                        "condition_latents": input_examples["condition_latents"],
                        "timesteps": input_examples["timesteps"],
                    }
                    if "previous_sides_latents" in input_examples:
                        signature_inputs["previous_sides_latents"] = input_examples[
                            "previous_sides_latents"
                        ]
                    elif (
                        architecture_name is not None
                        and architecture_name.strip().lower() in {"dit_v3", "dit_v4"}
                    ):
                        signature_inputs["previous_sides_latents"] = np.zeros_like(
                            input_examples["noisy_query_latents"]
                        )
                    signature = infer_signature(signature_inputs, output_numpy)
            except Exception:  # noqa: BLE001
                _log.warning("Could not infer MLflow signature for model.")

        is_dit = architecture_name is not None and architecture_name.lower().startswith("dit")
        metadata = {
            "framework": "pytorch",
            "model_type": "dit_side_diffusion" if is_dit else "frame_side_generator",
            "architecture_name": architecture_name or "unknown",
            "input_shapes": json.dumps(
                {key: list(np.asarray(value).shape) for key, value in input_examples.items()},
                ensure_ascii=False,
            ),
            "output_shape": str(tuple(output_numpy.shape)) if output_numpy is not None else "unknown",
            "model_config": json.dumps(model_config or {}, ensure_ascii=False, default=str),
        }
        model_artifact_name = Path(artifact_path).name or "model"
        try:
            info = mlflow.pytorch.log_model(
                pytorch_model=cpu_model,
                name=model_artifact_name,
                registered_model_name=registered_model_name,
                signature=signature,
                metadata=metadata,
                pip_requirements=["torch", "mlflow", "numpy", "cloudpickle"],
            )
        except TypeError:
            info = mlflow.pytorch.log_model(
                pytorch_model=cpu_model,
                artifact_path=model_artifact_name,
                registered_model_name=registered_model_name,
                signature=signature,
                metadata=metadata,
                pip_requirements=["torch", "mlflow", "numpy", "cloudpickle"],
            )

        client = MlflowTrackingClient(tracking_uri=self.tracking_uri)
        safe_name = registered_model_name.replace("'", "\\'")
        versions = list(client.search_model_versions(filter_string=f"name='{safe_name}'"))
        run_id = active_run.info.run_id
        by_run = [v for v in versions if v.run_id == run_id]
        if by_run:
            latest = max(by_run, key=lambda v: int(v.version))
            version = str(latest.version)
            model_uri = f"models:/{registered_model_name}/{version}"
        else:
            version = "unknown"
            model_uri = info.model_uri

        self.log_params(
            {
                "registered_model_name": registered_model_name,
                "registered_model_version": version,
                "registered_model_uri": model_uri,
            }
        )
        return {"name": registered_model_name, "version": version, "model_uri": model_uri}

    def resolve_model_uri(self, model_ref: str) -> str:
        model_ref = model_ref.strip()
        if model_ref.startswith("runs:/"):
            return model_ref
        if model_ref.startswith("models:/"):
            rest = model_ref.removeprefix("models:/").strip()
            if not rest:
                raise ValueError("Empty models:/ uri")
            if "/" in rest or "@" in rest:
                return model_ref
            return self._registry_uri_with_version(rest)
        return self._registry_uri_with_version(model_ref)

    def _registry_uri_with_version(self, registered_name: str) -> str:
        client = MlflowTrackingClient(self.tracking_uri)
        safe = registered_name.replace("'", "\\'")
        versions = list(
            client.search_model_versions(filter_string=f"name='{safe}'", max_results=200)
        )
        if not versions:
            raise FileNotFoundError(f"Model {registered_name!r} has no versions in Model Registry.")

        def _stage_key(v: Any) -> tuple[int, int]:
            stage = (v.current_stage or "None").strip()
            order = {"Production": 0, "Staging": 1, "None": 2, "Archived": 3}.get(stage, 9)
            return (order, -int(v.version))

        best = min(versions, key=_stage_key)
        return f"models:/{registered_name}/{best.version}"

    @staticmethod
    def _find_file(root: Path, filename: str) -> Path | None:
        for direct in (root / filename, root / "model" / filename, root / "weights" / filename):
            if direct.is_file():
                return direct
        for path in root.rglob(filename):
            if path.is_file():
                return path
        return None

    def get_model(
        self,
        model_id: str,
        *,
        map_location: str | torch.device = "cpu",
        try_mlflow_pytorch_flavor: bool = True,
    ) -> Any:
        uri = self.resolve_model_uri(model_id)
        if try_mlflow_pytorch_flavor:
            try:
                model = mlflow.pytorch.load_model(uri, map_location=map_location)
                model.eval()
                return model
            except (MlflowException, RestException, OSError, RuntimeError, ValueError):
                pass

        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        root = Path(local_dir)
        scripted_path = self._find_file(root, "model_best_scripted.pt")
        if scripted_path is not None:
            model = torch.jit.load(str(scripted_path), map_location=map_location)
            model.eval()
            return model

        final_weights = self._find_file(root, "final_model.pt")
        if final_weights is not None:
            raise RuntimeError(
                "Found final_model.pt (state_dict only). Use mlflow.pytorch artifacts for direct loading."
            )
        raise FileNotFoundError(f"Unable to load model artifacts for {uri!r}")

    def get_registered_model_by_name(
        self,
        model_name: str,
        *,
        version: str | int | None = None,
        stage: str | None = None,
        map_location: str | torch.device = "cpu",
        try_mlflow_pytorch_flavor: bool = True,
    ) -> Any:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty.")
        if version is not None and stage is not None:
            raise ValueError("Specify only one of version or stage.")

        if version is not None:
            model_id = f"models:/{model_name}/{version}"
        elif stage is not None:
            stage_name = stage.strip()
            if not stage_name:
                raise ValueError("stage must be non-empty when provided.")
            model_id = f"models:/{model_name}@{stage_name}"
        else:
            model_id = model_name

        return self.get_model(
            model_id,
            map_location=map_location,
            try_mlflow_pytorch_flavor=try_mlflow_pytorch_flavor,
        )


class MLFlowClient(MLflowRepository):
    def get_model_input_size(self, model_id: str) -> tuple[int, int] | None:
        """
        Возвращает (height, width), сохранённые в metadata модели MLflow.
        Поддерживает старый формат metadata.image_size = "<height>x<width>"
        и новый metadata.input_shape = "(B, C, H, W)".
        """
        uri = self.resolve_model_uri(model_id)
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        root = Path(local_dir)
        mlmodel_path = self._find_file(root, "MLmodel")
        if mlmodel_path is None:
            return None

        model_meta = Model.load(str(mlmodel_path))
        metadata = model_meta.metadata or {}
        image_size = metadata.get("image_size")
        if isinstance(image_size, str):
            parts = image_size.lower().strip().split("x")
            if len(parts) == 2:
                try:
                    height = int(parts[0].strip())
                    width = int(parts[1].strip())
                except ValueError:
                    return None
                if height > 0 and width > 0:
                    return (height, width)

        input_shape = metadata.get("input_shape")
        if not isinstance(input_shape, str):
            return None
        cleaned = input_shape.strip().removeprefix("(").removesuffix(")")
        chunks = [x.strip() for x in cleaned.split(",") if x.strip()]
        if len(chunks) < 4:
            return None
        try:
            height = int(chunks[-2])
            width = int(chunks[-1])
        except ValueError:
            return None
        if height <= 0 or width <= 0:
            return None
        return (height, width)

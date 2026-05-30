from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as functional
from mlflow.models import Model

from app.config import get_settings
from app.decoder_api_config import get_decoder_api_config
from app.models.model_archive import BaseModelConfig, build_config_from_run_params
from app.src.repositories.mlflow import MLflowRepository
from app.src.utils.latent_decoder_api import LatentDecoderAPIClient
from app.src.utils.noise_scheduler import LinearNoiseScheduler
from app.src.utils.video_reader.sequential_access_reader import SequentialAccessReader


@dataclass(slots=True)
class SamplingConfig:
    prediction_type: str
    num_train_timesteps: int
    beta_start: float
    beta_end: float


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _print_frames_progress(processed: int, total: int, started_at: float) -> None:
    if total <= 0:
        return
    progress = min(max(processed / total, 0.0), 1.0)
    bar_width = 30
    filled = int(round(progress * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = max(time.perf_counter() - started_at, 1e-8)
    fps = processed / elapsed if processed > 0 else 0.0
    eta_sec = int((total - processed) / max(fps, 1e-8)) if processed < total else 0
    print(
        f"\r[frames] |{bar}| {progress * 100:6.2f}% "
        f"{processed}/{total} fps={fps:6.2f} eta={eta_sec:4d}s",
        end="",
        flush=True,
    )
    if processed >= total:
        print("", flush=True)


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Инференс DiT из MLflow Registry: генерация боковых областей по центру кадра и "
            "сборка результирующего mp4."
        )
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        required=True,
        help="Путь к исходному видео.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        required=True,
        help="Путь к выходному mp4 с восстановленными боковинами.",
    )
    parser.add_argument(
        "--model-ref",
        type=str,
        default=settings.mlflow_registered_model_name,
        help=(
            "Ссылка на модель в MLflow: имя, models:/name/version, models:/name@stage или runs:/..."
        ),
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
        "--model-cache-dir",
        type=Path,
        default=Path(".cache") / "mlflow_models",
        help=(
            "Локальный кэш артефактов модели. При повторном запуске с тем же model-ref модель "
            "не скачивается повторно."
        ),
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Принудительно заново скачать артефакты модели в кэш.",
    )
    parser.add_argument(
        "--sampler",
        choices=("ddpm", "ddim"),
        default="ddim",
        help="Резолвер денойзинга.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=80,
        help="Количество шагов инференса.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Базовый seed генерации.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="FPS выходного видео. <=0 означает взять из входного.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=560,
        help="Ширина кадра после чтения (вход нормализуется до этого размера).",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=240,
        help="Высота кадра после чтения (вход нормализуется до этого размера).",
    )
    parser.add_argument(
        "--center-width",
        type=int,
        default=432,
        help="Ширина центральной области, подаваемой в модель.",
    )
    parser.add_argument(
        "--second-frame-prev-sides-source",
        choices=("generated", "first_frame_ref"),
        default="generated",
        help=(
            "Источник previous_sides_latents для 2-го кадра: "
            "'generated' — из генерации 1-го кадра, "
            "'first_frame_ref' — из референсных боковин 1-го кадра."
        ),
    )
    parser.add_argument(
        "--decoder-api-base-url",
        type=str,
        default=get_decoder_api_config().decoder_api_base_url,
        help="Базовый URL encoder/decoder API.",
    )
    parser.add_argument(
        "--decoder-api-timeout-sec",
        type=float,
        default=get_decoder_api_config().decoder_api_timeout_sec,
        help="Таймаут запросов к API.",
    )
    parser.add_argument(
        "--disable-decoder-readiness-check",
        action="store_true",
        help="Не выполнять предварительную проверку /readiness.",
    )
    return parser.parse_args()


def _sanitize_prediction_type(raw_value: str) -> str:
    value = raw_value.strip().lower()
    if value not in {"epsilon", "v", "epsilon_v_hybrid"}:
        raise ValueError(f"Неподдерживаемый prediction_type: {raw_value}")
    return value


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_file(root: Path, filename: str) -> Path | None:
    for direct in (root / filename, root / "model" / filename, root / "weights" / filename):
        if direct.is_file():
            return direct
    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def _read_model_config_from_mlmodel(local_model_dir: Path) -> dict[str, str]:
    mlmodel_path = _find_file(local_model_dir, "MLmodel")
    if mlmodel_path is None:
        return {}
    metadata = Model.load(str(mlmodel_path)).metadata or {}
    raw_model_config = metadata.get("model_config")
    if isinstance(raw_model_config, dict):
        return {str(key): str(value) for key, value in raw_model_config.items()}
    if isinstance(raw_model_config, str):
        try:
            parsed = json.loads(raw_model_config)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
    return {}


def _download_model_artifacts_cached(
    *,
    repo: MLflowRepository,
    model_uri: str,
    cache_dir: Path,
    force_redownload: bool,
) -> Path:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(model_uri.encode("utf-8")).hexdigest()
    model_cache_dir = cache_dir / cache_key
    marker_path = model_cache_dir / "cache_meta.json"

    if model_cache_dir.exists() and not force_redownload and marker_path.is_file():
        cached_meta = json.loads(marker_path.read_text(encoding="utf-8"))
        cached_path = Path(cached_meta.get("local_model_dir", ""))
        if cached_meta.get("model_uri") == model_uri and cached_path.is_dir():
            _log(f"Использую модель из локального кэша: {cached_path}")
            return cached_path

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Скачиваю артефакты модели из MLflow: {model_uri}")
    local_model_dir = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(model_cache_dir))
    ).resolve()
    marker_payload = {"model_uri": model_uri, "local_model_dir": str(local_model_dir)}
    marker_path.write_text(
        json.dumps(marker_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"Артефакты модели сохранены в кэш: {local_model_dir}")
    return local_model_dir


def _resolve_inference_timesteps(
    *, train_steps: int, inference_steps: int, device: torch.device
) -> torch.Tensor:
    if train_steps <= 0:
        raise ValueError("num_train_timesteps должен быть > 0.")
    if inference_steps <= 0:
        raise ValueError("num_inference_steps должен быть > 0.")
    raw = torch.linspace(train_steps - 1, 0, inference_steps, dtype=torch.float32)
    timesteps = raw.round().to(torch.long).unique_consecutive()
    if timesteps.numel() == 0:
        raise RuntimeError("Не удалось построить последовательность timesteps для инференса.")
    return timesteps.to(device)


def _predict_epsilon(
    *,
    model_output: torch.Tensor,
    current_latents: torch.Tensor,
    timestep: int,
    scheduler: LinearNoiseScheduler,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type in {"epsilon", "epsilon_v_hybrid"}:
        return model_output
    if prediction_type == "v":
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[timestep].view(1, 1, 1, 1)
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(1, 1, 1, 1)
        return sqrt_one_minus_alpha * current_latents + sqrt_alpha * model_output
    raise ValueError(f"Неподдерживаемый prediction_type: {prediction_type}")


def _step_ddim_like(
    *,
    current_latents: torch.Tensor,
    eps_prediction: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_bar_t)
    x0_prediction = (current_latents - sqrt_one_minus_alpha_t * eps_prediction) / (
        sqrt_alpha_bar_t + 1e-8
    )
    if alpha_bar_prev >= 0.999999:
        return x0_prediction

    sigma = (
        float(eta)
        * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
        * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / alpha_bar_prev, min=0.0))
    )
    direction = (
        torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma * sigma, min=0.0)) * eps_prediction
    )
    noise = torch.randn_like(current_latents) if float(sigma.item()) > 0.0 else 0.0
    return torch.sqrt(alpha_bar_prev) * x0_prediction + direction + sigma * noise


def _sample_query_sides(
    *,
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    condition_latents: torch.Tensor,
    latent_channels: int,
    query_height: int,
    query_width: int,
    num_inference_steps: int,
    sampler: str,
    prediction_type: str,
    seed: int,
    previous_sides_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    device = condition_latents.device
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    query_latents = torch.randn(
        (condition_latents.shape[0], latent_channels, query_height, query_width),
        generator=generator,
        device=device,
        dtype=condition_latents.dtype,
    )
    timesteps = _resolve_inference_timesteps(
        train_steps=scheduler.num_train_timesteps,
        inference_steps=num_inference_steps,
        device=device,
    )

    sampler_eta = 1.0 if sampler == "ddpm" else 0.0
    for index, timestep_tensor in enumerate(timesteps):
        timestep = int(timestep_tensor.item())
        model_timesteps = torch.full(
            (condition_latents.shape[0],),
            timestep,
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            model_inputs = {
                "noisy_query_latents": query_latents,
                "condition_latents": condition_latents,
                "timesteps": model_timesteps,
            }
            if previous_sides_latents is not None:
                model_inputs["previous_sides_latents"] = previous_sides_latents
            model_output = model(**model_inputs)
        eps_prediction = _predict_epsilon(
            model_output=model_output,
            current_latents=query_latents,
            timestep=timestep,
            scheduler=scheduler,
            prediction_type=prediction_type,
        )
        alpha_bar_t = scheduler.alphas_cumprod[timestep]
        if index + 1 < timesteps.numel():
            prev_timestep = int(timesteps[index + 1].item())
            alpha_bar_prev = scheduler.alphas_cumprod[prev_timestep]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device, dtype=query_latents.dtype)
        query_latents = _step_ddim_like(
            current_latents=query_latents,
            eps_prediction=eps_prediction,
            alpha_bar_t=alpha_bar_t,
            alpha_bar_prev=alpha_bar_prev,
            eta=sampler_eta,
        )
    return query_latents


def _prepare_condition_latents(
    *,
    decoder_client: LatentDecoderAPIClient,
    center_rgb: np.ndarray,
    model_config: BaseModelConfig,
    device: torch.device,
) -> torch.Tensor:
    encoded = decoder_client.encode_single_frame(center_rgb)
    condition = torch.from_numpy(np.asarray(encoded, dtype=np.float32)).unsqueeze(0).to(device)
    if condition.ndim != 4:
        raise ValueError(
            f"Encoder API должен возвращать латенты формы [C, H, W] или [1, C, H, W], получено {tuple(condition.shape)}"
        )
    if condition.shape[1] != model_config.latent_channels:
        raise ValueError(
            f"Число latent channels не совпадает с моделью: encoder={condition.shape[1]}, "
            f"model={model_config.latent_channels}"
        )
    expected_hw = (model_config.condition_height, model_config.condition_width)
    if condition.shape[-2:] != expected_hw:
        condition = functional.interpolate(
            condition,
            size=expected_hw,
            mode="bilinear",
            align_corners=False,
        )
    return condition


def _prepare_reference_side_latents(
    *,
    decoder_client: LatentDecoderAPIClient,
    source_frame_rgb: np.ndarray,
    center_width: int,
    model_config: BaseModelConfig,
    device: torch.device,
) -> torch.Tensor:
    frame_height, frame_width, _ = source_frame_rgb.shape
    center_start = (frame_width - center_width) // 2
    center_end = center_start + center_width
    if center_start < 0 or center_end > frame_width:
        raise ValueError(
            "Некорректная геометрия center для извлечения референсных боковин: "
            f"frame_width={frame_width}, center_width={center_width}"
        )
    side_reference_rgb = np.concatenate(
        (source_frame_rgb[:, :center_start, :], source_frame_rgb[:, center_end:, :]),
        axis=1,
    )
    if side_reference_rgb.shape[0] != frame_height or side_reference_rgb.shape[1] <= 0:
        raise ValueError(
            "Не удалось извлечь референсные боковины кадра: "
            f"shape={side_reference_rgb.shape}"
        )

    encoded = decoder_client.encode_single_frame(side_reference_rgb)
    previous = torch.from_numpy(np.asarray(encoded, dtype=np.float32)).unsqueeze(0).to(device)
    if previous.ndim != 4:
        raise ValueError(
            "Encoder API должен возвращать латенты формы [C, H, W] или [1, C, H, W], "
            f"получено {tuple(previous.shape)}"
        )
    if previous.shape[1] != model_config.latent_channels:
        raise ValueError(
            "Число latent channels не совпадает с моделью для previous_sides: "
            f"encoder={previous.shape[1]}, model={model_config.latent_channels}"
        )
    expected_hw = (model_config.query_height, model_config.query_width)
    if previous.shape[-2:] != expected_hw:
        previous = functional.interpolate(
            previous,
            size=expected_hw,
            mode="bilinear",
            align_corners=False,
        )
    return previous


def _model_requires_previous_sides(model: torch.nn.Module) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "previous_sides_latents" in signature.parameters


def _resolve_output_fps(input_video_path: Path, forced_fps: float) -> float:
    if forced_fps > 0.0:
        return float(forced_fps)
    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео для чтения fps: {input_video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if fps > 0.0 else 25.0


def _resolve_model_and_configs(
    *,
    args: argparse.Namespace,
    settings: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, BaseModelConfig, SamplingConfig]:
    mlflow_repo = MLflowRepository(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        experiment_name=settings.mlflow_experiment_name,
        settings=settings,
    )
    model_uri = mlflow_repo.resolve_model_uri(args.model_ref)
    local_model_dir = _download_model_artifacts_cached(
        repo=mlflow_repo,
        model_uri=model_uri,
        cache_dir=args.model_cache_dir,
        force_redownload=bool(args.force_redownload),
    )
    run_params = _read_model_config_from_mlmodel(local_model_dir)
    architecture_name = run_params.get("architecture_name", settings.model_architecture)
    model_config = build_config_from_run_params(
        settings=settings,
        architecture_name=architecture_name,
        run_params=run_params,
    )

    prediction_type = _sanitize_prediction_type(
        run_params.get("prediction_type", settings.prediction_type)
    )
    sampling_config = SamplingConfig(
        prediction_type=prediction_type,
        num_train_timesteps=int(
            run_params.get("num_train_timesteps", settings.num_train_timesteps)
        ),
        beta_start=float(run_params.get("beta_start", settings.beta_start)),
        beta_end=float(run_params.get("beta_end", settings.beta_end)),
    )

    model = mlflow.pytorch.load_model(str(local_model_dir), map_location=device)
    model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    _log(
        "Модель загружена: "
        f"uri={model_uri}, arch={model_config.architecture_name}, "
        f"prediction_type={sampling_config.prediction_type}"
    )
    return model, model_config, sampling_config


def _compose_frame_with_generated_sides(
    *,
    source_frame_rgb: np.ndarray,
    center_width: int,
    generated_sides_rgb: np.ndarray,
) -> np.ndarray:
    height, full_width, _ = source_frame_rgb.shape
    side_total_width = full_width - center_width
    if side_total_width <= 0:
        raise ValueError(
            f"center_width must be smaller than frame width: {center_width} >= {full_width}"
        )
    left_width = side_total_width // 2
    right_width = side_total_width - left_width
    center_start = left_width
    center_end = center_start + center_width
    center = source_frame_rgb[:, center_start:center_end, :]

    if generated_sides_rgb.shape[:2] != (height, side_total_width):
        generated_sides_rgb = cv2.resize(
            generated_sides_rgb,
            (side_total_width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    left = generated_sides_rgb[:, :left_width, :]
    right = generated_sides_rgb[:, side_total_width - right_width :, :]
    return np.concatenate([left, center, right], axis=1)


def _validate_geometry(frame_width: int, frame_height: int, center_width: int) -> None:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width/frame_height должны быть > 0.")
    if center_width <= 0 or center_width > frame_width:
        raise ValueError("center_width должен быть > 0 и <= frame_width.")
    if (frame_width - center_width) % 2 != 0:
        raise ValueError("frame_width - center_width должен быть четным для симметричных боковин.")


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    _validate_geometry(args.frame_width, args.frame_height, args.center_width)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    _log(f"Устройство инференса: {device}")
    model, model_config, sampling_config = _resolve_model_and_configs(
        args=args,
        settings=settings,
        device=device,
    )
    model_requires_previous_sides = _model_requires_previous_sides(model)
    if model_requires_previous_sides and model_config.architecture_name not in {"dit_v3", "dit_v4"}:
        _log(
            "[warn] Forward модели требует previous_sides_latents, "
            f"но в metadata architecture_name={model_config.architecture_name!r}."
        )
    use_previous_sides = model_requires_previous_sides or model_config.architecture_name == "dit_v3"

    decoder_client = LatentDecoderAPIClient(
        base_url=args.decoder_api_base_url,
        timeout_sec=float(args.decoder_api_timeout_sec),
    )
    if not args.disable_decoder_readiness_check:
        response = decoder_client.readiness()
        response.raise_for_status()
        _log("Decoder API readiness: OK")

    scheduler = LinearNoiseScheduler(
        num_train_timesteps=sampling_config.num_train_timesteps,
        beta_start=sampling_config.beta_start,
        beta_end=sampling_config.beta_end,
        device=device,
    )
    output_fps = _resolve_output_fps(args.input_video, float(args.fps))
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (args.frame_width, args.frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть writer для выходного файла: {args.output_video}")

    _log(
        "Старт инференса видео: "
        f"input={args.input_video}, output={args.output_video}, "
        f"sampler={args.sampler}, steps={args.num_inference_steps}"
    )
    center_start = (args.frame_width - args.center_width) // 2
    center_end = center_start + args.center_width

    processed = 0
    started_at = time.perf_counter()
    previous_generated_side_latents: torch.Tensor | None = None
    first_frame_reference_side_latents: torch.Tensor | None = None
    with SequentialAccessReader(
        str(args.input_video),
        height=args.frame_height,
        width=args.frame_width,
        normalize=False,
    ) as reader:
        total = reader.total_frames
        for frame_index in range(total):
            window = reader.read_window(1)
            if window.shape[0] == 0:
                break
            source_chw = window[0]
            source_frame_rgb = source_chw.permute(1, 2, 0).contiguous().cpu().numpy()
            center_rgb = source_frame_rgb[:, center_start:center_end, :]
            condition_latents = _prepare_condition_latents(
                decoder_client=decoder_client,
                center_rgb=center_rgb,
                model_config=model_config,
                device=device,
            )
            previous_sides_latents = None
            if use_previous_sides:
                if frame_index == 0:
                    # Для первого кадра реального предыдущего контекста нет.
                    previous_sides_latents = torch.zeros(
                        (
                            condition_latents.shape[0],
                            model_config.latent_channels,
                            model_config.query_height,
                            model_config.query_width,
                        ),
                        device=device,
                        dtype=condition_latents.dtype,
                    )
                elif (
                    frame_index == 1
                    and args.second_frame_prev_sides_source == "first_frame_ref"
                    and first_frame_reference_side_latents is not None
                ):
                    previous_sides_latents = first_frame_reference_side_latents.to(
                        device=device, dtype=condition_latents.dtype
                    )
                else:
                    if previous_generated_side_latents is None:
                        raise RuntimeError(
                            "Ожидались previous_sides_latents из предыдущей генерации, "
                            "но они отсутствуют."
                        )
                    previous_sides_latents = previous_generated_side_latents.to(
                        device=device, dtype=condition_latents.dtype
                    )

            generated_side_latents = _sample_query_sides(
                model=model,
                scheduler=scheduler,
                condition_latents=condition_latents,
                latent_channels=model_config.latent_channels,
                query_height=model_config.query_height,
                query_width=model_config.query_width,
                num_inference_steps=args.num_inference_steps,
                sampler=args.sampler,
                prediction_type=sampling_config.prediction_type,
                seed=args.seed + frame_index,
                previous_sides_latents=previous_sides_latents,
            )
            if use_previous_sides:
                if frame_index == 0 and args.second_frame_prev_sides_source == "first_frame_ref":
                    first_frame_reference_side_latents = _prepare_reference_side_latents(
                        decoder_client=decoder_client,
                        source_frame_rgb=source_frame_rgb,
                        center_width=args.center_width,
                        model_config=model_config,
                        device=device,
                    )
                previous_generated_side_latents = generated_side_latents.detach()
            decoded_sides = decoder_client.decode_tensor_batch(generated_side_latents)[0]
            fused_frame_rgb = _compose_frame_with_generated_sides(
                source_frame_rgb=source_frame_rgb,
                center_width=args.center_width,
                generated_sides_rgb=decoded_sides,
            )
            fused_frame_bgr = cv2.cvtColor(fused_frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(fused_frame_bgr)
            processed += 1
            _print_frames_progress(processed=processed, total=total, started_at=started_at)

    writer.release()
    _log(f"Готово. Видео сохранено: {args.output_video}")


if __name__ == "__main__":
    main()

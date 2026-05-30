from __future__ import annotations

import argparse
import copy
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from app.decoder_api_config import get_decoder_api_config
from app.config import get_settings
from app.loss_config import LossConfig
from app.models.model_archive import build_config_from_settings, build_model, list_architectures
from app.src.data.sft_dataset import create_sft_dataloader
from app.src.repositories.mlflow import MLflowRepository
from app.src.utils.latent_decoder_api import LatentDecoderAPIClient
from app.src.utils.mlflow_dataset import download_encoded_dataset
from app.src.utils.noise_scheduler import LinearNoiseScheduler
from app.src.utils.sft_reader import discover_sft_files

settings = get_settings()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение DiT с логированием в MLflow.")
    parser.add_argument(
        "--architecture",
        type=str,
        default=settings.model_architecture,
        choices=list_architectures(),
        help="Имя архитектуры модели из model archive.",
    )
    parser.add_argument("--run-name", type=str, default="baseline-dit-training-local")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help=(
            "Количество GPU для обучения. 0 = авто. "
            "Для multi-GPU запускайте через torchrun --nproc_per_node=<N>."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=settings.datasets_folder,
        help="Папка с датасетом или один .sft-файл; при папке .sft ищутся рекурсивно во всех подпапках.",
    )
    parser.add_argument(
        "--dataset-recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Искать .sft рекурсивно в подпапках dataset-dir (по умолчанию: включено).",
    )
    parser.add_argument("--tracking-uri", type=str, default=settings.mlflow_tracking_uri)
    parser.add_argument("--registry-uri", type=str, default=settings.mlflow_registry_uri)
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=settings.mlflow_registered_model_name,
        help="Имя модели в MLflow Model Registry для финальной версии.",
    )
    parser.add_argument(
        "--preview-every-n-epochs",
        type=int,
        default=0,
        help="Период тестового прогона для визуализации (каждые N эпох). 0 отключает.",
    )
    parser.add_argument(
        "--preview-every-n-steps",
        type=int,
        default=settings.preview_every_n_steps,
        help=(
            "Период тестового прогона для визуализации (каждые N global steps). "
            "Если > 0, имеет приоритет над preview-every-n-epochs."
        ),
    )
    parser.add_argument(
        "--preview-images-count",
        type=int,
        default=settings.preview_images_count,
        help="Количество фиксированных изображений M для тестового прогона.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(gpu_support: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Переменная окружения {name} должна быть int, получено: {raw!r}") from exc


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _supports_first_frame_conditioning(model: nn.Module) -> bool:
    return hasattr(_unwrap_model(model), "first_frame_embedding")


def _is_first_frame_from_frame_ids(frame_ids: torch.Tensor | None) -> torch.Tensor | None:
    if frame_ids is None:
        return None
    return frame_ids.reshape(-1) == 0


def _maybe_add_first_frame_input(
    *,
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    frame_ids: torch.Tensor | None,
) -> None:
    if not _supports_first_frame_conditioning(model):
        return
    is_first_frame = _is_first_frame_from_frame_ids(frame_ids)
    if is_first_frame is None:
        raise ValueError("Для DiT v4 требуется frame_id, чтобы передать is_first_frame в AdaLN.")
    model_inputs["is_first_frame"] = is_first_frame


def _is_distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _init_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    requested_num_gpus = int(args.num_gpus)

    if requested_num_gpus < 0:
        raise ValueError("--num-gpus должен быть >= 0.")

    if world_size > 1:
        if requested_num_gpus > 0 and requested_num_gpus != world_size:
            raise ValueError(
                "--num-gpus должен совпадать с WORLD_SIZE при запуске через torchrun. "
                f"Получено num_gpus={requested_num_gpus}, WORLD_SIZE={world_size}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DDP требует CUDA. GPU не обнаружен.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size

    if requested_num_gpus > 1:
        available_gpus = torch.cuda.device_count()
        if available_gpus < requested_num_gpus:
            raise ValueError(
                f"Запрошено --num-gpus={requested_num_gpus}, но доступно только {available_gpus}."
            )
        raise ValueError(
            "Для запуска на нескольких GPU используйте torchrun, например: "
            f"torchrun --nproc_per_node={requested_num_gpus} cli/train_dit.py --num-gpus {requested_num_gpus}"
        )

    return False, 0, 0, 1


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source_model: nn.Module) -> None:
        source_parameters = dict(source_model.named_parameters())
        for name, ema_parameter in self.model.named_parameters():
            source_parameter = source_parameters[name]
            ema_parameter.data.mul_(self.decay).add_(source_parameter.data, alpha=1.0 - self.decay)
        source_buffers = dict(source_model.named_buffers())
        for name, ema_buffer in self.model.named_buffers():
            ema_buffer.copy_(source_buffers[name])


def _extract_schedule_coefficients(
    scheduler: LinearNoiseScheduler, timesteps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_alpha = scheduler.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    return sqrt_alpha, sqrt_one_minus_alpha


def _snr_weights(
    *,
    scheduler: LinearNoiseScheduler,
    timesteps: torch.Tensor,
    prediction_type: str,
    min_snr_gamma: float,
) -> torch.Tensor:
    snr = scheduler.alphas_cumprod[timesteps] / (1.0 - scheduler.alphas_cumprod[timesteps] + 1e-8)
    clipped_snr = torch.minimum(snr, torch.full_like(snr, float(min_snr_gamma)))
    if prediction_type == "v":
        return (clipped_snr / (snr + 1.0)).detach()
    return (clipped_snr / (snr + 1e-8)).detach()


def _prediction_to_eps(
    *,
    model_output: torch.Tensor,
    noisy_target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: LinearNoiseScheduler,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type in {"epsilon", "epsilon_v_hybrid"}:
        return model_output
    if prediction_type == "v":
        sqrt_alpha, sqrt_one_minus_alpha = _extract_schedule_coefficients(scheduler, timesteps)
        return sqrt_one_minus_alpha * noisy_target + sqrt_alpha * model_output
    raise ValueError(f"Неподдерживаемый prediction_type: {prediction_type}")


def _compute_losses(
    *,
    model_output: torch.Tensor,
    noise: torch.Tensor,
    clean_target: torch.Tensor,
    noisy_target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: LinearNoiseScheduler,
    prediction_type: str,
    epsilon_v_hybrid_lambda: float,
    min_snr_gamma: float,
    loss_config: LossConfig,
    global_step: int,
    previous_sides: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sqrt_alpha, sqrt_one_minus_alpha = _extract_schedule_coefficients(scheduler, timesteps)
    eps_prediction = _prediction_to_eps(
        model_output=model_output,
        noisy_target=noisy_target,
        timesteps=timesteps,
        scheduler=scheduler,
        prediction_type=prediction_type,
    )
    v_target = sqrt_alpha * noise - sqrt_one_minus_alpha * clean_target

    eps_loss_sample = (
        functional.mse_loss(eps_prediction, noise, reduction="none").flatten(1).mean(dim=1)
    )
    if prediction_type == "v":
        v_prediction = model_output
        diffusion_loss_sample = (
            functional.mse_loss(v_prediction, v_target, reduction="none").flatten(1).mean(dim=1)
        )
    elif prediction_type == "epsilon_v_hybrid":
        v_prediction = sqrt_alpha * eps_prediction - sqrt_one_minus_alpha * clean_target
        v_loss_sample = (
            functional.mse_loss(v_prediction, v_target, reduction="none").flatten(1).mean(dim=1)
        )
        diffusion_loss_sample = (
            float(epsilon_v_hybrid_lambda) * eps_loss_sample
            + (1.0 - float(epsilon_v_hybrid_lambda)) * v_loss_sample
        )
    elif prediction_type == "epsilon":
        diffusion_loss_sample = eps_loss_sample
    else:
        raise ValueError(f"Неподдерживаемый prediction_type: {prediction_type}")

    weights = _snr_weights(
        scheduler=scheduler,
        timesteps=timesteps,
        prediction_type=prediction_type,
        min_snr_gamma=min_snr_gamma,
    )
    diffusion_loss = (weights * diffusion_loss_sample).mean()

    x0_prediction = (noisy_target - sqrt_one_minus_alpha * eps_prediction) / (sqrt_alpha + 1e-8)
    grad_x_pred = x0_prediction[:, :, :, 1:] - x0_prediction[:, :, :, :-1]
    grad_x_true = clean_target[:, :, :, 1:] - clean_target[:, :, :, :-1]
    grad_y_pred = x0_prediction[:, :, 1:, :] - x0_prediction[:, :, :-1, :]
    grad_y_true = clean_target[:, :, 1:, :] - clean_target[:, :, :-1, :]
    detail_loss = functional.l1_loss(grad_x_pred, grad_x_true) + functional.l1_loss(
        grad_y_pred, grad_y_true
    )

    charbonnier_loss = torch.sqrt(
        (x0_prediction - clean_target).pow(2) + float(loss_config.charbonnier_epsilon) ** 2
    ).mean()

    predicted_fft = torch.fft.rfft2(x0_prediction.float(), dim=(-2, -1))
    target_fft = torch.fft.rfft2(clean_target.float(), dim=(-2, -1))
    predicted_magnitude = predicted_fft.abs()
    target_magnitude = target_fft.abs()
    if loss_config.fft_use_log_magnitude:
        predicted_magnitude = torch.log(predicted_magnitude + float(loss_config.fft_epsilon))
        target_magnitude = torch.log(target_magnitude + float(loss_config.fft_epsilon))
    fft_loss = functional.l1_loss(predicted_magnitude, target_magnitude)

    if previous_sides is not None:
        temporal_target = (
            previous_sides.detach() if loss_config.temporal_detach_previous else previous_sides
        )
        temporal_loss_sample = torch.sqrt(
            (x0_prediction - temporal_target).pow(2) + float(loss_config.charbonnier_epsilon) ** 2
        ).flatten(1).mean(dim=1)
        temporal_scale = loss_config.temporal_warmup_scale(global_step)
        temporal_loss = temporal_loss_sample.mean() * temporal_scale
    else:
        temporal_loss = torch.zeros_like(diffusion_loss)

    total_loss = (
        float(loss_config.weight_diffusion) * diffusion_loss
        + float(loss_config.weight_detail) * detail_loss
        + float(loss_config.weight_charbonnier) * charbonnier_loss
        + float(loss_config.weight_fft) * fft_loss
        + float(loss_config.weight_temporal) * temporal_loss
    )
    return (
        total_loss,
        diffusion_loss,
        detail_loss,
        charbonnier_loss,
        fft_loss,
        temporal_loss,
        eps_prediction,
    )


def compute_noise_metrics(
    *,
    model_output: torch.Tensor,
    noise: torch.Tensor,
    clean_target: torch.Tensor,
    noisy_target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: LinearNoiseScheduler,
    prediction_type: str,
    epsilon_v_hybrid_lambda: float,
    min_snr_gamma: float,
    loss_config: LossConfig,
    global_step: int,
    previous_sides: torch.Tensor | None,
) -> dict[str, float]:
    (
        total_loss,
        diffusion_loss,
        detail_loss,
        charbonnier_loss,
        fft_loss,
        temporal_loss,
        eps_prediction,
    ) = _compute_losses(
        model_output=model_output,
        noise=noise,
        clean_target=clean_target,
        noisy_target=noisy_target,
        timesteps=timesteps,
        scheduler=scheduler,
        prediction_type=prediction_type,
        epsilon_v_hybrid_lambda=epsilon_v_hybrid_lambda,
        min_snr_gamma=min_snr_gamma,
        loss_config=loss_config,
        global_step=global_step,
        previous_sides=previous_sides,
    )
    loss_mse = float(diffusion_loss.item())
    loss_total = float(total_loss.item())
    loss_detail = float(detail_loss.item())
    loss_charbonnier = float(charbonnier_loss.item())
    loss_fft = float(fft_loss.item())
    loss_temporal = float(temporal_loss.item())
    loss_l1 = float(functional.l1_loss(eps_prediction, noise).item())
    cosine_similarity = float(
        functional.cosine_similarity(
            eps_prediction.flatten(1),
            noise.flatten(1),
            dim=1,
        )
        .mean()
        .item()
    )
    predicted_noise_std = float(eps_prediction.std().item())
    target_noise_std = float(noise.std().item())
    timestep_mean = float(timesteps.float().mean().item())
    sqrt_alpha, sqrt_one_minus_alpha = _extract_schedule_coefficients(scheduler, timesteps)
    reconstructed_x0 = (noisy_target - sqrt_one_minus_alpha * eps_prediction) / (sqrt_alpha + 1e-8)
    x0_reconstruction_mse = float(functional.mse_loss(reconstructed_x0, clean_target).item())
    snr = scheduler.alphas_cumprod[timesteps] / (1.0 - scheduler.alphas_cumprod[timesteps] + 1e-8)
    snr_mean = float(snr.mean().item())
    return {
        "loss_total": loss_total,
        "loss_mse": loss_mse,
        "loss_detail": loss_detail,
        "loss_charbonnier": loss_charbonnier,
        "loss_fft": loss_fft,
        "loss_temporal": loss_temporal,
        "loss_l1": loss_l1,
        "noise_cosine_similarity": cosine_similarity,
        "x0_reconstruction_mse": x0_reconstruction_mse,
        "noise_pred_std": predicted_noise_std,
        "noise_target_std": target_noise_std,
        "timestep_mean": timestep_mean,
        "snr_mean": snr_mean,
    }


def _resolve_autocast_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(
        f"AUTOCAST_DTYPE должен быть одним из: bfloat16|bf16|float16|fp16|half. Получено: {value!r}"
    )


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    return str(dtype)


def evaluate_on_validation(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scheduler: LinearNoiseScheduler,
    device: torch.device,
    use_cuda: bool,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    prediction_type: str,
    epsilon_v_hybrid_lambda: float,
    min_snr_gamma: float,
    loss_config: LossConfig,
    global_step: int,
    use_previous_sides: bool,
) -> dict[str, float]:
    model.eval()
    start_time = time.perf_counter()
    weighted_sums: dict[str, float] = {}
    samples_seen = 0
    batches_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            condition = batch["condition"].to(device, non_blocking=use_cuda)
            clean_target = batch["target"].to(device, non_blocking=use_cuda)
            previous_sides = (
                batch["previous_sides"].to(device, non_blocking=use_cuda) if use_previous_sides else None
            )
            frame_ids = batch.get("frame_id")
            frame_ids = frame_ids.to(device, non_blocking=use_cuda) if frame_ids is not None else None
            timesteps = scheduler.sample_timesteps(batch_size=clean_target.shape[0], device=device)
            noise = torch.randn_like(clean_target)
            noisy_target = scheduler.add_noise(clean_target, noise, timesteps)
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                model_inputs = {
                    "noisy_query_latents": noisy_target,
                    "condition_latents": condition,
                    "timesteps": timesteps,
                }
                if previous_sides is not None:
                    model_inputs["previous_sides_latents"] = previous_sides
                _maybe_add_first_frame_input(
                    model=model,
                    model_inputs=model_inputs,
                    frame_ids=frame_ids,
                )
                model_output = model(**model_inputs)
                batch_metrics = compute_noise_metrics(
                    model_output=model_output,
                    noise=noise,
                    clean_target=clean_target,
                    noisy_target=noisy_target,
                    timesteps=timesteps,
                    scheduler=scheduler,
                    prediction_type=prediction_type,
                    epsilon_v_hybrid_lambda=epsilon_v_hybrid_lambda,
                    min_snr_gamma=min_snr_gamma,
                    loss_config=loss_config,
                    global_step=global_step,
                    previous_sides=previous_sides,
                )
            batch_size = clean_target.shape[0]
            samples_seen += batch_size
            batches_seen += 1
            for key, value in batch_metrics.items():
                weighted_sums[key] = weighted_sums.get(key, 0.0) + value * batch_size

    if samples_seen == 0:
        return {"num_samples": 0.0, "num_batches": float(batches_seen)}

    duration = max(time.perf_counter() - start_time, 1e-8)
    aggregated = {key: value / samples_seen for key, value in weighted_sums.items()}
    aggregated["num_samples"] = float(samples_seen)
    aggregated["num_batches"] = float(batches_seen)
    aggregated["steps_per_second"] = float(batches_seen / duration)
    aggregated["samples_per_second"] = float(samples_seen / duration)
    return aggregated


def _collect_fixed_preview_batch(
    *,
    dataloader: torch.utils.data.DataLoader,
    max_images: int,
) -> dict[str, torch.Tensor] | None:
    if max_images <= 0:
        return None

    fixed_condition: list[torch.Tensor] = []
    fixed_target: list[torch.Tensor] = []
    fixed_previous_sides: list[torch.Tensor] = []
    fixed_frame_ids: list[torch.Tensor] = []
    has_previous_sides = False
    has_frame_ids = False
    collected = 0
    for batch in dataloader:
        condition = batch["condition"]
        target = batch["target"]
        previous_sides = batch.get("previous_sides")
        frame_ids = batch.get("frame_id")
        batch_size = condition.shape[0]
        take_count = min(max_images - collected, batch_size)
        if take_count <= 0:
            break
        fixed_condition.append(condition[:take_count].detach().cpu())
        fixed_target.append(target[:take_count].detach().cpu())
        if previous_sides is not None:
            has_previous_sides = True
            fixed_previous_sides.append(previous_sides[:take_count].detach().cpu())
        if frame_ids is not None:
            has_frame_ids = True
            fixed_frame_ids.append(frame_ids[:take_count].detach().cpu())
        collected += take_count
        if collected >= max_images:
            break

    if collected == 0:
        return None

    preview_batch = {
        "condition": torch.cat(fixed_condition, dim=0),
        "target": torch.cat(fixed_target, dim=0),
    }
    if has_previous_sides:
        preview_batch["previous_sides"] = torch.cat(fixed_previous_sides, dim=0)
    if has_frame_ids:
        preview_batch["frame_id"] = torch.cat(fixed_frame_ids, dim=0)
    return preview_batch


def _latent_to_display_image(latent: torch.Tensor) -> np.ndarray:
    array = latent.detach().cpu().float().numpy()
    if array.ndim != 3:
        raise ValueError(f"Ожидается тензор [C, H, W], получено: {tuple(array.shape)}")
    if array.shape[0] >= 3:
        rgb = array[:3]
    else:
        rgb = np.repeat(array[:1], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb - rgb.min()
    denom = float(rgb.max()) + 1e-8
    rgb = rgb / denom
    return rgb


def _mask_previous_sides_for_preview(
    model: nn.Module,
    previous_sides: torch.Tensor,
) -> torch.Tensor | None:
    unwrapped = _unwrap_model(model)
    mask_fn = getattr(unwrapped, "mask_sides_latents", None)
    if callable(mask_fn):
        return mask_fn(previous_sides)
    return None


def _run_preview_sampling(
    *,
    model: nn.Module,
    scheduler: LinearNoiseScheduler,
    condition_latents: torch.Tensor,
    initial_noise: torch.Tensor,
    device: torch.device,
    use_cuda: bool,
    preview_steps: list[int],
    prediction_type: str,
    previous_sides_latents: torch.Tensor | None = None,
    is_first_frame: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    valid_steps = sorted(
        {step for step in preview_steps if 1 <= step <= scheduler.num_train_timesteps}
    )
    if not valid_steps:
        return {}

    model.eval()
    with torch.no_grad():
        condition = condition_latents.to(device, non_blocking=use_cuda)
        noisy_target = initial_noise.to(device, non_blocking=use_cuda)
        batch_size = condition.shape[0]
        snapshots: dict[int, torch.Tensor] = {}

        for denoise_step in range(1, scheduler.num_train_timesteps + 1):
            timestep_value = scheduler.num_train_timesteps - denoise_step
            timesteps = torch.full(
                (batch_size,),
                timestep_value,
                device=device,
                dtype=torch.long,
            )
            model_inputs = {
                "noisy_query_latents": noisy_target,
                "condition_latents": condition,
                "timesteps": timesteps,
            }
            if previous_sides_latents is not None:
                previous_sides = previous_sides_latents.to(device, non_blocking=use_cuda)
                model_inputs["previous_sides_latents"] = previous_sides
            if is_first_frame is not None:
                model_inputs["is_first_frame"] = is_first_frame.to(device, non_blocking=use_cuda)
            model_output = model(**model_inputs)
            predicted_noise = _prediction_to_eps(
                model_output=model_output,
                noisy_target=noisy_target,
                timesteps=timesteps,
                scheduler=scheduler,
                prediction_type=prediction_type,
            )
            sqrt_alpha_t = scheduler.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = scheduler.sqrt_one_minus_alphas_cumprod[timesteps].view(
                -1, 1, 1, 1
            )
            x0_prediction = (noisy_target - sqrt_one_minus_alpha_t * predicted_noise) / (
                sqrt_alpha_t + 1e-8
            )

            if timestep_value > 0:
                previous_timesteps = timesteps - 1
                sqrt_alpha_prev = scheduler.sqrt_alphas_cumprod[previous_timesteps].view(
                    -1, 1, 1, 1
                )
                sqrt_one_minus_alpha_prev = scheduler.sqrt_one_minus_alphas_cumprod[
                    previous_timesteps
                ].view(-1, 1, 1, 1)
                noisy_target = (
                    sqrt_alpha_prev * x0_prediction + sqrt_one_minus_alpha_prev * predicted_noise
                )
            else:
                noisy_target = x0_prediction

            if denoise_step in valid_steps:
                snapshots[denoise_step] = noisy_target.detach().cpu()
                if len(snapshots) == len(valid_steps):
                    break

    return snapshots


def _log_preview_figure(
    *,
    mlflow_repo: MLflowRepository,
    epoch_index: int,
    global_step: int,
    clean_targets: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
    preview_steps: list[int],
) -> None:
    steps_to_plot = [step for step in preview_steps if step in snapshots]
    if clean_targets.numel() == 0 or not steps_to_plot:
        return

    rows = clean_targets.shape[0]
    cols = 1 + len(steps_to_plot)
    figure, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for row_index in range(rows):
        axes[row_index][0].imshow(_latent_to_display_image(clean_targets[row_index]))
        axes[row_index][0].set_axis_off()
        if row_index == 0:
            axes[row_index][0].set_title("original", fontsize=10)
        for col_index, step in enumerate(steps_to_plot, start=1):
            axes[row_index][col_index].imshow(_latent_to_display_image(snapshots[step][row_index]))
            axes[row_index][col_index].set_axis_off()
            if row_index == 0:
                axes[row_index][col_index].set_title(f"step {step}", fontsize=10)
    figure.tight_layout()
    mlflow_repo.mlflow.log_figure(
        figure,
        artifact_file=f"preview/epoch_{epoch_index + 1:04d}_global_step_{global_step:08d}.png",
    )
    plt.close(figure)


def _log_preview_figure_decoded(
    *,
    mlflow_repo: MLflowRepository,
    decoder_client: LatentDecoderAPIClient | None,
    epoch_index: int,
    global_step: int,
    clean_targets: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
    preview_steps: list[int],
    condition_latents: torch.Tensor | None = None,
    masked_previous_sides_latents: torch.Tensor | None = None,
) -> None:
    if decoder_client is None:
        return
    steps_to_plot = [step for step in preview_steps if step in snapshots]
    if clean_targets.numel() == 0 or not steps_to_plot:
        return

    try:
        decoded_condition = (
            decoder_client.decode_tensor_batch(condition_latents)
            if condition_latents is not None
            else None
        )
        decoded_masked_previous_sides = (
            decoder_client.decode_tensor_batch(masked_previous_sides_latents)
            if masked_previous_sides_latents is not None
            else None
        )
        decoded_target = decoder_client.decode_tensor_batch(clean_targets)
        decoded_snapshots = {
            step: decoder_client.decode_tensor_batch(snapshots[step]) for step in steps_to_plot
        }
    except Exception as exc:  # noqa: BLE001
        _log(f"[preview] decode via API failed: {exc}")
        return

    rows = clean_targets.shape[0]
    leading_images: list[tuple[str, np.ndarray]] = []
    if decoded_condition is not None:
        leading_images.append(("input center decoded", decoded_condition))
    if decoded_masked_previous_sides is not None:
        leading_images.append(("masked sides decoded", decoded_masked_previous_sides))
    leading_images.append(("target decoded", decoded_target))

    cols = len(leading_images) + len(steps_to_plot)
    figure, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for row_index in range(rows):
        for col_index, (title, decoded_batch) in enumerate(leading_images):
            axes[row_index][col_index].imshow(decoded_batch[row_index])
            axes[row_index][col_index].set_axis_off()
            if row_index == 0:
                axes[row_index][col_index].set_title(title, fontsize=10)
        for col_index, step in enumerate(steps_to_plot, start=len(leading_images)):
            axes[row_index][col_index].imshow(decoded_snapshots[step][row_index])
            axes[row_index][col_index].set_axis_off()
            if row_index == 0:
                axes[row_index][col_index].set_title(f"step {step} decoded", fontsize=10)
    figure.tight_layout()
    mlflow_repo.mlflow.log_figure(
        figure,
        artifact_file=f"preview/epoch_{epoch_index + 1:04d}_global_step_{global_step:08d}_decoded.png",
    )
    plt.close(figure)


def _positional_component_indices(hidden_size: int) -> list[tuple[int, str]]:
    half = hidden_size // 2
    quarter = max(half // 2, 1)
    candidates = (
        (0, "y sin"),
        (quarter, "y cos"),
        (half, "x sin"),
        (half + quarter, "x cos"),
    )
    seen: set[int] = set()
    result: list[tuple[int, str]] = []
    for index, title in candidates:
        if 0 <= index < hidden_size and index not in seen:
            result.append((index, title))
            seen.add(index)
    return result


def _log_positional_encoding_preview(
    *,
    mlflow_repo: MLflowRepository,
    model: nn.Module,
    global_step: int,
) -> None:
    base_model = _unwrap_model(model)
    embedding_specs = (
        (
            "query",
            getattr(base_model, "query_positional_embedding", None),
            getattr(base_model, "query_patches_h", None),
            getattr(base_model, "query_patches_w", None),
        ),
        (
            "condition",
            getattr(base_model, "condition_positional_embedding", None),
            getattr(base_model, "condition_patches_h", None),
            getattr(base_model, "condition_patches_w", None),
        ),
        (
            "previous",
            getattr(base_model, "previous_positional_embedding", None),
            getattr(base_model, "previous_patches_h", None),
            getattr(base_model, "previous_patches_w", None),
        ),
    )
    rows: list[tuple[str, torch.Tensor, int, int]] = []
    for name, embedding, patches_h, patches_w in embedding_specs:
        if embedding is None or patches_h is None or patches_w is None:
            continue
        if embedding.ndim != 3 or embedding.shape[0] != 1:
            continue
        if embedding.shape[1] != int(patches_h) * int(patches_w):
            continue
        rows.append((name, embedding.detach().float().cpu()[0], int(patches_h), int(patches_w)))

    if not rows:
        return

    component_indices = _positional_component_indices(rows[0][1].shape[1])
    if not component_indices:
        return

    figure, axes = plt.subplots(
        len(rows),
        len(component_indices),
        figsize=(3.0 * len(component_indices), 2.8 * len(rows)),
        squeeze=False,
    )
    for row_index, (name, embedding, patches_h, patches_w) in enumerate(rows):
        for col_index, (component_index, component_title) in enumerate(component_indices):
            grid = embedding[:, component_index].view(patches_h, patches_w).numpy()
            axes[row_index][col_index].imshow(grid, cmap="coolwarm", interpolation="nearest")
            axes[row_index][col_index].set_axis_off()
            axes[row_index][col_index].set_title(
                f"{name}: {component_title} d{component_index}",
                fontsize=9,
            )
    figure.tight_layout()
    mlflow_repo.mlflow.log_figure(
        figure,
        artifact_file=f"preview/positional_encoding_global_step_{global_step:08d}.png",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size = _init_distributed(args)
    is_main_process = rank == 0

    try:
        settings = get_settings()
        if is_distributed and settings.train_num_workers > 0:
            raise ValueError(
                "Для DDP с текущим IterableDataset установите TRAIN_NUM_WORKERS=0."
            )

        set_seed(settings.seed + rank)
        prediction_type = settings.prediction_type.strip().lower()
        if prediction_type not in {"epsilon", "v", "epsilon_v_hybrid"}:
            raise ValueError(
                "PREDICTION_TYPE должен быть одним из: epsilon, v, epsilon_v_hybrid. "
                f"Получено: {settings.prediction_type}"
            )
        epsilon_v_hybrid_lambda = float(min(max(settings.epsilon_v_hybrid_lambda, 0.0), 1.0))

        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            use_cuda = True
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            use_cuda = True
        else:
            device = torch.device("cpu")
            use_cuda = False

        autocast_enabled = bool(settings.use_autocast and use_cuda)
        autocast_dtype = _resolve_autocast_dtype(settings.autocast_dtype)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=autocast_enabled and autocast_dtype is torch.float16,
        )

        if is_main_process:
            if autocast_enabled:
                _log(f"[amp] enabled with dtype={_torch_dtype_name(autocast_dtype)}")
            else:
                _log("[amp] disabled")
            _log(
                f"[ddp] enabled={is_distributed}, rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )

        if args.dataset_dir:
            dataset_dir = Path(args.dataset_dir)
        elif settings.datasets_folder is not None:
            dataset_dir = Path(settings.datasets_folder)
        elif settings.dataset_run_ids:
            if is_distributed:
                raise ValueError(
                    "DDP режим требует локально доступный dataset-dir для всех процессов. "
                    "Укажите --dataset-dir или DATASETS_FOLDER."
                )
            if is_main_process:
                _log("DATASETS_FOLDER не задан — скачиваем encoded SFT из MLflow...")
            dataset_dir = download_encoded_dataset(settings)
        else:
            raise ValueError(
                "Не задан путь к датасету. Укажите DATASETS_FOLDER, --dataset-dir "
                "или MLFLOW_DATASET_RUN_IDS для скачивания encoded_sft."
            )

        sft_files = discover_sft_files(dataset_dir, recursive=args.dataset_recursive)
        search_mode = (
            "рекурсивно (включая подпапки)" if args.dataset_recursive else "только в корне папки"
        )
        if is_main_process:
            _log(f"Найдено {len(sft_files)} .sft файл(ов) в {dataset_dir} — поиск {search_mode}")
            preview_limit = 20
            for file_path in sft_files[:preview_limit]:
                _log(f"  - {file_path}")
            if len(sft_files) > preview_limit:
                _log(f"  ... и ещё {len(sft_files) - preview_limit}")

        use_previous_sides = args.architecture.strip().lower() in {"dit_v3", "dit_v4"}
        previous_sides_key = settings.previous_sides_key if use_previous_sides else None
        if is_main_process:
            _log(
                "DiT dataset keys: "
                f"condition={settings.condition_key!r}, target={settings.target_key!r}, "
                f"previous={previous_sides_key!r}"
            )

        train_dataloader = create_sft_dataloader(
            dataset_dir=dataset_dir,
            condition_key=settings.condition_key,
            target_key=settings.target_key,
            previous_sides_key=previous_sides_key,
            batch_size=settings.train_batch_size,
            num_workers=settings.train_num_workers,
            latent_channels=settings.latent_channels,
            condition_height=settings.condition_height,
            condition_width=settings.condition_width,
            target_height=settings.query_height,
            target_width=settings.query_width,
            pin_memory=False,
            split="train",
            train_ratio=0.9,
            split_seed=settings.seed,
            recursive=args.dataset_recursive,
            distributed_rank=rank if is_distributed else 0,
            distributed_world_size=world_size if is_distributed else 1,
            drop_last=is_distributed,
        )
        val_dataloader = create_sft_dataloader(
            dataset_dir=dataset_dir,
            condition_key=settings.condition_key,
            target_key=settings.target_key,
            previous_sides_key=previous_sides_key,
            batch_size=settings.train_batch_size,
            num_workers=settings.train_num_workers,
            latent_channels=settings.latent_channels,
            condition_height=settings.condition_height,
            condition_width=settings.condition_width,
            target_height=settings.query_height,
            target_width=settings.query_width,
            pin_memory=False,
            split="val",
            train_ratio=0.9,
            split_seed=settings.seed,
            recursive=args.dataset_recursive,
            distributed_rank=0,
            distributed_world_size=1,
            drop_last=False,
        )

        model_config = build_config_from_settings(settings, architecture_name=args.architecture)
        use_previous_sides = model_config.architecture_name in {"dit_v3", "dit_v4"}
        base_model = build_model(model_config).to(device)
        use_first_frame_conditioning = _supports_first_frame_conditioning(base_model)
        ema = ExponentialMovingAverage(base_model, decay=settings.ema_decay) if settings.use_ema else None
        model: nn.Module
        if is_distributed:
            model = DistributedDataParallel(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            model = base_model

        total_params = sum(parameter.numel() for parameter in base_model.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad
        )
        max_steps = int(settings.train_max_steps)
        if max_steps <= 0:
            raise ValueError(
                "TRAIN_MAX_STEPS должен быть > 0. "
                "Остановка обучения выполняется по числу global steps."
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        scheduler = LinearNoiseScheduler(
            num_train_timesteps=settings.num_train_timesteps,
            beta_start=settings.beta_start,
            beta_end=settings.beta_end,
            device=device,
        )

        mlflow_repo = (
            MLflowRepository(
                tracking_uri=args.tracking_uri,
                registry_uri=args.registry_uri,
                experiment_name=settings.mlflow_experiment_name,
            )
            if is_main_process
            else None
        )
        decoder_api_config = get_decoder_api_config()
        decoder_client: LatentDecoderAPIClient | None = None
        if is_main_process and decoder_api_config.decoder_api_enabled:
            decoder_client = LatentDecoderAPIClient(
                base_url=decoder_api_config.decoder_api_base_url,
                timeout_sec=decoder_api_config.decoder_api_timeout_sec,
            )
            if decoder_api_config.decoder_api_check_readiness:
                try:
                    readiness_response = decoder_client.readiness()
                    readiness_response.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    _log(f"[preview] Decoder API is unavailable ({exc}), decoded preview disabled.")
                    decoder_client = None

        global_step = 0
        loss_config = settings.loss_config
        preview_interval_epochs = max(args.preview_every_n_epochs, 0)
        preview_interval_steps = max(args.preview_every_n_steps, 0)
        preview_images_count = max(args.preview_images_count, 0)
        preview_steps = [1000]
        model_runtime_params = {
            "query_patch_size": getattr(model_config, "query_patch_size", -1),
            "condition_patch_size": getattr(model_config, "condition_patch_size", -1),
            "previous_patch_size": getattr(model_config, "previous_patch_size", -1),
            "previous_cross_start_block": getattr(model_config, "previous_cross_start_block", -1),
            "side_mask_columns_from_right": ",".join(
                str(value)
                for value in getattr(model_config, "side_mask_columns_from_right", ())
            ),
            "input_normalization": bool(getattr(model_config, "input_normalization", False)),
            "update_condition_tokens": bool(getattr(model_config, "update_condition_tokens", False)),
            "backbone_type": getattr(model_config, "backbone_type", "n/a"),
            "first_frame_conditioning": use_first_frame_conditioning,
            "use_autocast": autocast_enabled,
            "autocast_dtype": _torch_dtype_name(autocast_dtype),
            "ddp_enabled": is_distributed,
            "ddp_world_size": world_size,
            "cli_num_gpus": int(args.num_gpus),
        }
        val_every_n_logs = max(int(settings.val_every_n_logs), 0)
        if is_main_process:
            if preview_interval_epochs > 0 and preview_interval_steps == 0:
                _log(
                    "[preview] preview-every-n-epochs устарел в step-based цикле. "
                    "Используйте preview-every-n-steps; preview будет отключен."
                )
            _log(
                "[val] schedule: "
                + (
                    f"evaluate every {val_every_n_logs} step(s)"
                    if val_every_n_logs > 0
                    else "disabled during training (only final validation)"
                )
            )
            _log(f"Training progress mode: by global steps ({max_steps} steps = 100%).")

        run_context = (
            mlflow_repo.start_run(run_name=args.run_name, tags={"pipeline": "dit-training"})
            if mlflow_repo is not None
            else nullcontext()
        )
        registration: dict[str, str] | None = None
        with run_context:
            run_params = settings.mlflow_param_dict()
            run_params.update(
                {
                    "architecture_name": model_config.architecture_name,
                    "model_params_total": total_params,
                    "model_params_trainable": trainable_params,
                    "train_val_split_ratio": "0.9/0.1",
                    "lr_scheduler": "constant",
                    "lr_scheduler_mode": "disabled",
                    "lr_scheduler_total_steps": 0,
                    "lr_scheduler_eta_min": settings.learning_rate,
                    "preview_every_n_epochs": preview_interval_epochs,
                    "preview_every_n_steps": preview_interval_steps,
                    "preview_images_count": preview_images_count,
                    "preview_steps": ",".join(str(step) for step in preview_steps),
                    "prediction_type": prediction_type,
                    "epsilon_v_hybrid_lambda": epsilon_v_hybrid_lambda,
                    "min_snr_gamma": settings.min_snr_gamma,
                    "use_ema": settings.use_ema,
                    "ema_decay": settings.ema_decay,
                    "val_every_n_logs": val_every_n_logs,
                    **loss_config.mlflow_param_dict(),
                    **model_runtime_params,
                }
            )
            if mlflow_repo is not None:
                mlflow_repo.log_params(run_params)
                mlflow_repo.log_config(
                    {
                        "run_name": args.run_name,
                        "dataset_dir": str(dataset_dir),
                        "device": str(device),
                        "architecture_name": model_config.architecture_name,
                        "model_params_total": total_params,
                        "model_params_trainable": trainable_params,
                        "preview_every_n_epochs": preview_interval_epochs,
                        "preview_every_n_steps": preview_interval_steps,
                        "preview_images_count": preview_images_count,
                        "preview_steps": preview_steps,
                        "prediction_type": prediction_type,
                        "epsilon_v_hybrid_lambda": epsilon_v_hybrid_lambda,
                        "min_snr_gamma": settings.min_snr_gamma,
                        "use_ema": settings.use_ema,
                        "ema_decay": settings.ema_decay,
                        "val_every_n_logs": val_every_n_logs,
                        **model_runtime_params,
                        **settings.mlflow_param_dict(),
                    }
                )

            fixed_preview_batch = (
                _collect_fixed_preview_batch(
                    dataloader=val_dataloader,
                    max_images=preview_images_count,
                )
                if is_main_process
                else None
            )
            if is_main_process and fixed_preview_batch is None and preview_interval_steps > 0:
                _log("[preview] Не удалось собрать фиксированный набор изображений, визуализация отключена.")
            if fixed_preview_batch is not None and preview_interval_steps > 0:
                preview_generator = torch.Generator(device="cpu").manual_seed(settings.seed)
                fixed_preview_noise = torch.randn(
                    fixed_preview_batch["target"].shape,
                    generator=preview_generator,
                    dtype=fixed_preview_batch["target"].dtype,
                )
            else:
                fixed_preview_noise = None
            if mlflow_repo is not None and preview_interval_steps > 0:
                _log_positional_encoding_preview(
                    mlflow_repo=mlflow_repo,
                    model=_unwrap_model(model),
                    global_step=global_step,
                )

            last_val_metrics: dict[str, float] = {}
            train_iterator = iter(train_dataloader)
            while global_step < max_steps:
                try:
                    batch = next(train_iterator)

                    step_start_time = time.perf_counter()
                    model.train()
                    condition = batch["condition"].to(device, non_blocking=use_cuda)
                    clean_target = batch["target"].to(device, non_blocking=use_cuda)
                    previous_sides = (
                        batch["previous_sides"].to(device, non_blocking=use_cuda)
                        if use_previous_sides
                        else None
                    )
                    frame_ids = batch.get("frame_id")
                    frame_ids = (
                        frame_ids.to(device, non_blocking=use_cuda)
                        if frame_ids is not None
                        else None
                    )

                    timesteps = scheduler.sample_timesteps(
                        batch_size=clean_target.shape[0], device=device
                    )
                    noise = torch.randn_like(clean_target)
                    noisy_target = scheduler.add_noise(clean_target, noise, timesteps)

                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
                        model_inputs = {
                            "noisy_query_latents": noisy_target,
                            "condition_latents": condition,
                            "timesteps": timesteps,
                        }
                        if previous_sides is not None:
                            model_inputs["previous_sides_latents"] = previous_sides
                        _maybe_add_first_frame_input(
                            model=model,
                            model_inputs=model_inputs,
                            frame_ids=frame_ids,
                        )
                        model_output = model(**model_inputs)
                        (
                            loss,
                            diffusion_loss,
                            detail_loss,
                            charbonnier_loss,
                            fft_loss,
                            temporal_loss,
                            _,
                        ) = _compute_losses(
                            model_output=model_output,
                            noise=noise,
                            clean_target=clean_target,
                            noisy_target=noisy_target,
                            timesteps=timesteps,
                            scheduler=scheduler,
                            prediction_type=prediction_type,
                            epsilon_v_hybrid_lambda=epsilon_v_hybrid_lambda,
                            min_snr_gamma=settings.min_snr_gamma,
                            loss_config=loss_config,
                            global_step=global_step,
                            previous_sides=previous_sides,
                        )

                    optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), settings.grad_clip_norm
                            ).item()
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), settings.grad_clip_norm
                            ).item()
                        )
                        optimizer.step()
                    if ema is not None:
                        ema.update(_unwrap_model(model))

                    global_step += 1

                    if global_step % settings.log_every_steps == 0:
                        with torch.no_grad():
                            train_metrics = compute_noise_metrics(
                                model_output=model_output,
                                noise=noise,
                                clean_target=clean_target,
                                noisy_target=noisy_target,
                                timesteps=timesteps,
                                scheduler=scheduler,
                                prediction_type=prediction_type,
                                epsilon_v_hybrid_lambda=epsilon_v_hybrid_lambda,
                                min_snr_gamma=settings.min_snr_gamma,
                                loss_config=loss_config,
                                global_step=global_step,
                                previous_sides=previous_sides,
                            )
                            train_metrics["loss_total_backprop"] = float(loss.item())
                            train_metrics["loss_mse_backprop"] = float(diffusion_loss.item())
                            train_metrics["loss_detail_backprop"] = float(detail_loss.item())
                            train_metrics["loss_charbonnier_backprop"] = float(charbonnier_loss.item())
                            train_metrics["loss_fft_backprop"] = float(fft_loss.item())
                            train_metrics["loss_temporal_backprop"] = float(temporal_loss.item())

                        step_duration = max(time.perf_counter() - step_start_time, 1e-8)
                        train_metrics.update(
                            {
                                "gradient_norm": grad_norm,
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "steps_per_second": 1.0 / step_duration,
                                "samples_per_second": clean_target.shape[0] / step_duration,
                            }
                        )
                        train_metrics["progress_percent"] = min(
                            (global_step / max_steps) * 100.0,
                            100.0,
                        )
                        if use_cuda:
                            train_metrics["gpu_memory_allocated_mb"] = float(
                                torch.cuda.memory_allocated(device=device)
                            ) / (1024**2)
                            train_metrics["gpu_memory_reserved_mb"] = float(
                                torch.cuda.memory_reserved(device=device)
                            ) / (1024**2)

                        val_metrics: dict[str, float] = {}
                        should_run_val_now = (
                            val_every_n_logs > 0 and global_step % val_every_n_logs == 0
                        )
                        if should_run_val_now and is_main_process:
                            eval_model = ema.model if ema is not None else _unwrap_model(model)
                            val_metrics = evaluate_on_validation(
                                model=eval_model,
                                dataloader=val_dataloader,
                                scheduler=scheduler,
                                device=device,
                                use_cuda=use_cuda,
                                autocast_enabled=autocast_enabled,
                                autocast_dtype=autocast_dtype,
                                prediction_type=prediction_type,
                                epsilon_v_hybrid_lambda=epsilon_v_hybrid_lambda,
                                min_snr_gamma=settings.min_snr_gamma,
                                loss_config=loss_config,
                                global_step=global_step,
                                use_previous_sides=use_previous_sides,
                            )
                            last_val_metrics = val_metrics

                        if mlflow_repo is not None:
                            prefixed_train_metrics = {
                                f"train_{key}": float(value) for key, value in train_metrics.items()
                            }
                            prefixed_val_metrics = (
                                {f"val_{key}": float(value) for key, value in val_metrics.items()}
                                if val_metrics
                                else {}
                            )
                            mlflow_repo.log_metrics(
                                metrics={**prefixed_train_metrics, **prefixed_val_metrics},
                                step=global_step,
                            )

                        if is_main_process:
                            current_val_metrics = val_metrics or last_val_metrics
                            _log(
                                f"[train] progress {train_metrics['progress_percent']:.2f}% "
                                f"({global_step}/{max_steps} steps), "
                                f"train_loss={train_metrics['loss_mse']:.6f}, "
                                f"val_loss={current_val_metrics.get('loss_mse', float('nan')):.6f}"
                            )
                            if "loss_mse" in current_val_metrics and val_every_n_logs > 0:
                                _log(
                                    f"[train/val] step={global_step} "
                                    f"train_loss={train_metrics['loss_mse']:.6f}, "
                                    f"val_loss={current_val_metrics['loss_mse']:.6f}"
                                )

                    if (
                        mlflow_repo is not None
                        and global_step % settings.checkpoint_every_steps == 0
                    ):
                        mlflow_repo.log_checkpoint(
                            model=_unwrap_model(model),
                            optimizer=optimizer,
                            step=global_step,
                            extra={"architecture_name": model_config.architecture_name},
                        )

                    if (
                        mlflow_repo is not None
                        and preview_interval_steps > 0
                        and global_step % preview_interval_steps == 0
                        and fixed_preview_batch is not None
                        and fixed_preview_noise is not None
                    ):
                        preview_model = ema.model if ema is not None else _unwrap_model(model)
                        preview_previous_sides = (
                            fixed_preview_batch["previous_sides"] if use_previous_sides else None
                        )
                        preview_masked_previous_sides = (
                            _mask_previous_sides_for_preview(preview_model, preview_previous_sides)
                            if preview_previous_sides is not None
                            else None
                        )
                        preview_is_first_frame = (
                            _is_first_frame_from_frame_ids(fixed_preview_batch.get("frame_id"))
                            if _supports_first_frame_conditioning(preview_model)
                            else None
                        )
                        preview_snapshots = _run_preview_sampling(
                            model=preview_model,
                            scheduler=scheduler,
                            condition_latents=fixed_preview_batch["condition"],
                            initial_noise=fixed_preview_noise,
                            device=device,
                            use_cuda=use_cuda,
                            preview_steps=preview_steps,
                            prediction_type=prediction_type,
                            previous_sides_latents=preview_previous_sides,
                            is_first_frame=preview_is_first_frame,
                        )
                        _log_preview_figure(
                            mlflow_repo=mlflow_repo,
                            epoch_index=0,
                            global_step=global_step,
                            clean_targets=fixed_preview_batch["target"],
                            snapshots=preview_snapshots,
                            preview_steps=preview_steps,
                        )
                        _log_preview_figure_decoded(
                            mlflow_repo=mlflow_repo,
                            decoder_client=decoder_client,
                            epoch_index=0,
                            global_step=global_step,
                            clean_targets=fixed_preview_batch["target"],
                            snapshots=preview_snapshots,
                            preview_steps=preview_steps,
                            condition_latents=fixed_preview_batch["condition"],
                            masked_previous_sides_latents=preview_masked_previous_sides,
                        )
                        _log(
                            f"[preview] logged denoising figure for step={global_step}, "
                            f"images={fixed_preview_batch['target'].shape[0]}"
                        )

                    # Раннее освобождение ссылок уменьшает пиковое потребление памяти между шагами.
                    del (
                        batch,
                        condition,
                        clean_target,
                        frame_ids,
                        timesteps,
                        noise,
                        noisy_target,
                        model_output,
                        loss,
                        grad_norm,
                        diffusion_loss,
                        detail_loss,
                        charbonnier_loss,
                        fft_loss,
                        temporal_loss,
                    )
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    continue

            if val_every_n_logs == 0 and is_main_process:
                eval_model = ema.model if ema is not None else _unwrap_model(model)
                final_val_metrics = evaluate_on_validation(
                    model=eval_model,
                    dataloader=val_dataloader,
                    scheduler=scheduler,
                    device=device,
                    use_cuda=use_cuda,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    prediction_type=prediction_type,
                    epsilon_v_hybrid_lambda=epsilon_v_hybrid_lambda,
                    min_snr_gamma=settings.min_snr_gamma,
                    loss_config=loss_config,
                    global_step=global_step,
                    use_previous_sides=use_previous_sides,
                )
                if mlflow_repo is not None:
                    mlflow_repo.log_metrics(
                        metrics={
                            f"val_{key}": float(value) for key, value in final_val_metrics.items()
                        },
                        step=global_step,
                    )
                if "loss_mse" in final_val_metrics:
                    _log(
                        f"[val] final step={global_step} "
                        f"val_loss={final_val_metrics['loss_mse']:.6f}"
                    )

            if mlflow_repo is not None:
                export_model = ema.model if ema is not None else _unwrap_model(model)
                mlflow_repo.log_model_weights(model=export_model, artifact_relpath="weights/final_model.pt")
                dit_input_examples = {
                    "noisy_query_latents": torch.randn(
                        1,
                        settings.latent_channels,
                        settings.query_height,
                        settings.query_width,
                    )
                    .numpy()
                    .astype(np.float32),
                    "condition_latents": torch.randn(
                        1,
                        settings.latent_channels,
                        settings.condition_height,
                        settings.condition_width,
                    )
                    .numpy()
                    .astype(np.float32),
                    "timesteps": np.array([settings.num_train_timesteps // 2], dtype=np.int64),
                }
                if use_previous_sides:
                    dit_input_examples["previous_sides_latents"] = (
                        torch.randn(
                            1,
                            settings.latent_channels,
                            settings.query_height,
                            settings.query_width,
                        )
                        .numpy()
                        .astype(np.float32)
                    )
                if use_first_frame_conditioning:
                    dit_input_examples["is_first_frame"] = np.array([False], dtype=np.bool_)
                registration = mlflow_repo.register_final_model(
                    model=export_model,
                    registered_model_name=args.registered_model_name,
                    artifact_path="dit_final",
                    model_config=run_params,
                    input_examples=dit_input_examples,
                    architecture_name=model_config.architecture_name,
                )
                mlflow_repo.log_params(
                    {
                        "registered_model_name": registration["name"],
                        "registered_model_version": registration["version"],
                        "registered_model_uri": registration["model_uri"],
                    }
                )
                mlflow_repo.log_metrics(metrics={"final_step": float(global_step)}, step=global_step)

        if is_main_process and registration is not None:
            _log(
                "Registered model in MLflow Registry: "
                f"{registration['name']} v{registration['version']} ({registration['model_uri']})"
            )
        if is_main_process:
            _log(f"Training finished. Global steps: {global_step}")
    finally:
        if _is_distributed_ready():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

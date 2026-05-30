from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window_2d = kernel[:, None] @ kernel[None, :]
    return window_2d


def _ssim_per_channel(pred: Tensor, target: Tensor, *, window_size: int = 7) -> Tensor:
    """SSIM для одного канала, pred/target: [B, 1, H, W] в диапазоне [0, 1]."""
    if pred.shape[-1] < window_size or pred.shape[-2] < window_size:
        return torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype)

    window = _gaussian_window(window_size, sigma=1.5, device=pred.device, dtype=pred.dtype)
    window = window.expand(1, 1, window_size, window_size)
    padding = window_size // 2

    c1 = 0.01**2
    c2 = 0.03**2

    mu_pred = F.conv2d(pred, window, padding=padding)
    mu_target = F.conv2d(target, window, padding=padding)
    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean(dim=(1, 2, 3))


def compute_ssim(pred: Tensor, target: Tensor) -> Tensor:
    """Средний SSIM по batch и каналам."""
    per_channel = [_ssim_per_channel(pred[:, i : i + 1], target[:, i : i + 1]) for i in range(pred.shape[1])]
    return torch.stack(per_channel, dim=1).mean(dim=1)


def compute_psnr(pred: Tensor, target: Tensor, *, max_val: float = 1.0) -> Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10((max_val**2) / (mse + 1e-8))


def compute_pearson_corr(pred: Tensor, target: Tensor) -> Tensor:
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((pred_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1) + 1e-8)
    return numerator / denominator


def compute_advanced_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Метрики качества реконструкции латентных карт sides."""
    with torch.no_grad():
        mae = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        rmse = torch.sqrt(mse)
        rel_l1 = ((pred - target).abs() / (target.abs() + 1e-3)).mean()
        psnr = compute_psnr(pred, target).mean()
        ssim = compute_ssim(pred, target).mean()
        corr = compute_pearson_corr(pred, target).mean()

    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "rel_l1": float(rel_l1.item()),
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
        "pearson_corr": float(corr.item()),
    }


def compute_gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 по пространственным градиентам — сохраняет резкие границы в diff map."""
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

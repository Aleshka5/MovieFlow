from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from app.src.utils.video_reader.sequential_access_reader import SequentialAccessReader


class MidasDepthEstimator:
    def __init__(self, *, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model: torch.nn.Module | None = None
        self.transform: Any | None = None

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Запрошен --depth-device cuda, но CUDA недоступна")
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self) -> None:
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_name, trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise RuntimeError(
                    "Для MiDaS не найден пакет 'timm'. Установите зависимость: uv add timm"
                ) from exc
            raise RuntimeError("Не удалось загрузить MiDaS: отсутствует python-зависимость.") from exc
        except Exception as exc:
            raise RuntimeError(
                "Не удалось загрузить MiDaS из torch.hub. "
                "Проверьте интернет/кэш torch и попробуйте позже."
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        if "small" in self.model_name.lower():
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    def estimate_depth_norm(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            raise RuntimeError("Depth estimator не инициализирован")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
        depth = prediction[0].detach().cpu().numpy()
        return cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)


def _collect_video_paths(videos_dir: Path) -> list[Path]:
    patterns = ("*.mp4", "*.mkv", "*.avi", "*.mov")
    videos = sorted(path for pattern in patterns for path in videos_dir.rglob(pattern) if path.is_file())
    if not videos:
        raise SystemExit(f"В папке не найдено видеофайлов: {videos_dir}")
    return videos


def _tensor_chw_rgb_to_bgr(frame_chw_rgb: torch.Tensor) -> np.ndarray:
    frame_hwc_rgb = frame_chw_rgb.permute(1, 2, 0).contiguous().cpu().numpy()
    if frame_hwc_rgb.dtype != np.uint8:
        frame_hwc_rgb = np.clip(frame_hwc_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_hwc_rgb, cv2.COLOR_RGB2BGR)


def _edge_width(width: int, edge_ratio: float) -> int:
    return max(1, int(round(width * edge_ratio)))


def _edge_arrays_2d(values: np.ndarray, edge_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    edge = _edge_width(values.shape[1], edge_ratio)
    left = values[:, :edge]
    right = values[:, values.shape[1] - edge :]
    return left, right


def _safe_float(value: float | np.floating[Any]) -> float:
    return float(value)


def _diff_stats(source_bgr: np.ndarray, next_bgr: np.ndarray, edge_ratio: float) -> dict[str, float]:
    diff_bgr = cv2.absdiff(source_bgr, next_bgr)
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    low_freq = cv2.GaussianBlur(diff_gray, (0, 0), sigmaX=1.5)
    high_freq = np.abs(diff_gray - low_freq)
    left_edge, right_edge = _edge_arrays_2d(diff_gray, edge_ratio=edge_ratio)

    mean_mag = _safe_float(np.mean(diff_gray))
    noise_ratio = _safe_float(np.mean(high_freq) / (mean_mag + 1e-6))
    structure_ratio = _safe_float(np.mean(np.abs(low_freq)) / (mean_mag + 1e-6))
    return {
        "mean": mean_mag,
        "median": _safe_float(np.median(diff_gray)),
        "std": _safe_float(np.std(diff_gray)),
        "p95": _safe_float(np.percentile(diff_gray, 95.0)),
        "max": _safe_float(np.max(diff_gray)),
        "edge_mean_left": _safe_float(np.mean(left_edge)),
        "edge_mean_right": _safe_float(np.mean(right_edge)),
        "edge_max_left": _safe_float(np.max(left_edge)),
        "edge_max_right": _safe_float(np.max(right_edge)),
        "noise_ratio": noise_ratio,
        "structure_ratio": structure_ratio,
    }


def _flow_stats(flow: np.ndarray, edge_ratio: float) -> dict[str, float]:
    magnitude = np.hypot(flow[..., 0], flow[..., 1]).astype(np.float32)
    mean_vector = np.mean(flow.reshape(-1, 2), axis=0)
    mean_magnitude = _safe_float(np.mean(magnitude))
    coherence = _safe_float(float(np.hypot(mean_vector[0], mean_vector[1])) / (mean_magnitude + 1e-6))
    left_edge, right_edge = _edge_arrays_2d(magnitude, edge_ratio=edge_ratio)
    low_freq = cv2.GaussianBlur(magnitude, (0, 0), sigmaX=1.5)
    high_freq = np.abs(magnitude - low_freq)

    return {
        "mean_magnitude": mean_magnitude,
        "median_magnitude": _safe_float(np.median(magnitude)),
        "std_magnitude": _safe_float(np.std(magnitude)),
        "p95_magnitude": _safe_float(np.percentile(magnitude, 95.0)),
        "max_magnitude": _safe_float(np.max(magnitude)),
        "mean_dx": _safe_float(mean_vector[0]),
        "mean_dy": _safe_float(mean_vector[1]),
        "coherence": coherence,
        "edge_mean_left": _safe_float(np.mean(left_edge)),
        "edge_mean_right": _safe_float(np.mean(right_edge)),
        "edge_max_left": _safe_float(np.max(left_edge)),
        "edge_max_right": _safe_float(np.max(right_edge)),
        "noise_ratio": _safe_float(np.mean(high_freq) / (mean_magnitude + 1e-6)),
        "structure_ratio": _safe_float(np.mean(np.abs(low_freq)) / (mean_magnitude + 1e-6)),
    }


def _depth_stats(depth_norm: np.ndarray, edge_ratio: float) -> dict[str, float]:
    left_edge, right_edge = _edge_arrays_2d(depth_norm, edge_ratio=edge_ratio)
    return {
        "mean": _safe_float(np.mean(depth_norm)),
        "min": _safe_float(np.min(depth_norm)),
        "max": _safe_float(np.max(depth_norm)),
        "edge_mean_left": _safe_float(np.mean(left_edge)),
        "edge_mean_right": _safe_float(np.mean(right_edge)),
        "edge_p95_left": _safe_float(np.percentile(left_edge, 95.0)),
        "edge_p95_right": _safe_float(np.percentile(right_edge, 95.0)),
    }


def _compute_flow(source_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev=source_gray,
        next=next_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=10,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def _new_video_payload() -> dict[str, Any]:
    return {
        "frame_count": 0,
        "camera_move_vector": [],
        "camera_move_track": [],
        "camera_speed": [],
        "brightness_mean": [],
        "brightness_median": [],
        "depth_mean": [],
        "depth_min": [],
        "depth_max": [],
        "depth_edge_mean_left": [],
        "depth_edge_mean_right": [],
        "depth_edge_p95_left": [],
        "depth_edge_p95_right": [],
        "diff_mean": [],
        "diff_median": [],
        "diff_std": [],
        "diff_p95": [],
        "diff_max": [],
        "diff_edge_mean_left": [],
        "diff_edge_mean_right": [],
        "diff_edge_max_left": [],
        "diff_edge_max_right": [],
        "diff_noise_ratio": [],
        "diff_structure_ratio": [],
        "optiflow_mean_magnitude": [],
        "optiflow_median_magnitude": [],
        "optiflow_std_magnitude": [],
        "optiflow_p95_magnitude": [],
        "optiflow_max_magnitude": [],
        "optiflow_mean_dx": [],
        "optiflow_mean_dy": [],
        "optiflow_coherence": [],
        "optiflow_edge_mean_left": [],
        "optiflow_edge_mean_right": [],
        "optiflow_edge_max_left": [],
        "optiflow_edge_max_right": [],
        "optiflow_noise_ratio": [],
        "optiflow_structure_ratio": [],
    }


def _append_frame_metrics(
    payload: dict[str, Any],
    *,
    brightness: dict[str, float],
    depth: dict[str, float],
    diff: dict[str, float],
    flow: dict[str, float],
    camera_dx: float,
    camera_dy: float,
    track_x: float,
    track_y: float,
) -> None:
    payload["frame_count"] += 1
    payload["camera_move_vector"].append([camera_dx, camera_dy])
    payload["camera_move_track"].append([track_x, track_y])
    payload["camera_speed"].append(_safe_float(np.hypot(camera_dx, camera_dy)))
    payload["brightness_mean"].append(brightness["mean"])
    payload["brightness_median"].append(brightness["median"])

    payload["depth_mean"].append(depth["mean"])
    payload["depth_min"].append(depth["min"])
    payload["depth_max"].append(depth["max"])
    payload["depth_edge_mean_left"].append(depth["edge_mean_left"])
    payload["depth_edge_mean_right"].append(depth["edge_mean_right"])
    payload["depth_edge_p95_left"].append(depth["edge_p95_left"])
    payload["depth_edge_p95_right"].append(depth["edge_p95_right"])

    payload["diff_mean"].append(diff["mean"])
    payload["diff_median"].append(diff["median"])
    payload["diff_std"].append(diff["std"])
    payload["diff_p95"].append(diff["p95"])
    payload["diff_max"].append(diff["max"])
    payload["diff_edge_mean_left"].append(diff["edge_mean_left"])
    payload["diff_edge_mean_right"].append(diff["edge_mean_right"])
    payload["diff_edge_max_left"].append(diff["edge_max_left"])
    payload["diff_edge_max_right"].append(diff["edge_max_right"])
    payload["diff_noise_ratio"].append(diff["noise_ratio"])
    payload["diff_structure_ratio"].append(diff["structure_ratio"])

    payload["optiflow_mean_magnitude"].append(flow["mean_magnitude"])
    payload["optiflow_median_magnitude"].append(flow["median_magnitude"])
    payload["optiflow_std_magnitude"].append(flow["std_magnitude"])
    payload["optiflow_p95_magnitude"].append(flow["p95_magnitude"])
    payload["optiflow_max_magnitude"].append(flow["max_magnitude"])
    payload["optiflow_mean_dx"].append(flow["mean_dx"])
    payload["optiflow_mean_dy"].append(flow["mean_dy"])
    payload["optiflow_coherence"].append(flow["coherence"])
    payload["optiflow_edge_mean_left"].append(flow["edge_mean_left"])
    payload["optiflow_edge_mean_right"].append(flow["edge_mean_right"])
    payload["optiflow_edge_max_left"].append(flow["edge_max_left"])
    payload["optiflow_edge_max_right"].append(flow["edge_max_right"])
    payload["optiflow_noise_ratio"].append(flow["noise_ratio"])
    payload["optiflow_structure_ratio"].append(flow["structure_ratio"])


def _compute_video_analytics(
    *,
    video_path: Path,
    width: int,
    height: int,
    edge_ratio: float,
    depth_estimator: MidasDepthEstimator,
) -> dict[str, Any]:
    payload = _new_video_payload()
    track_x = 0.0
    track_y = 0.0

    with SequentialAccessReader(path=str(video_path), width=width, height=height, normalize=False) as reader:
        initial_window = reader.read_window(2)
        if initial_window.shape[0] == 0:
            return payload

        current = initial_window[0]
        next_frame = initial_window[1] if initial_window.shape[0] > 1 else None

        while True:
            source_bgr = _tensor_chw_rgb_to_bgr(current)
            next_bgr = source_bgr if next_frame is None else _tensor_chw_rgb_to_bgr(next_frame)

            gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            brightness = {
                "mean": _safe_float(np.mean(gray)),
                "median": _safe_float(np.median(gray)),
            }

            depth_norm = depth_estimator.estimate_depth_norm(source_bgr)
            depth_stats = _depth_stats(depth_norm, edge_ratio=edge_ratio)
            diff_stats = _diff_stats(source_bgr, next_bgr, edge_ratio=edge_ratio)

            if next_frame is None:
                flow = np.zeros((source_bgr.shape[0], source_bgr.shape[1], 2), dtype=np.float32)
            else:
                flow = _compute_flow(source_bgr, next_bgr)
            flow_stats = _flow_stats(flow, edge_ratio=edge_ratio)

            inv_depth_weight = 1.0 - depth_norm
            weighted_flow = flow * inv_depth_weight[..., None]
            camera_dx = _safe_float(-np.median(weighted_flow[..., 0]))
            camera_dy = _safe_float(-np.median(weighted_flow[..., 1]))
            track_x += camera_dx
            track_y += camera_dy

            _append_frame_metrics(
                payload,
                brightness=brightness,
                depth=depth_stats,
                diff=diff_stats,
                flow=flow_stats,
                camera_dx=camera_dx,
                camera_dy=camera_dy,
                track_x=track_x,
                track_y=track_y,
            )

            if next_frame is None:
                break

            current = next_frame
            batch = reader.read_window(1)
            next_frame = batch[0] if batch.shape[0] > 0 else None

    return payload


def run_build_scene_analytics(
    *,
    videos_dir: Path,
    output_path: Path,
    width: int,
    height: int,
    edge_ratio: float,
    max_videos: int | None,
    depth_model: str,
    depth_device: str,
) -> Path:
    if not videos_dir.is_dir():
        raise SystemExit(f"--videos-dir должен указывать на существующую папку: {videos_dir}")
    if width <= 0 or height <= 0:
        raise SystemExit("--width и --height должны быть > 0")
    if not 0.0 < edge_ratio <= 0.5:
        raise SystemExit("--edge-ratio должен быть в диапазоне (0, 0.5]")
    if max_videos is not None and max_videos <= 0:
        raise SystemExit("--max-videos должен быть > 0")

    videos = _collect_video_paths(videos_dir)
    if max_videos is not None:
        videos = videos[:max_videos]

    logger.info("Загрузка MiDaS: {} (device={})", depth_model, depth_device)
    depth_estimator = MidasDepthEstimator(model_name=depth_model, device=depth_device)
    depth_estimator.load()

    result: dict[str, Any] = {}
    for video_path in tqdm(videos, desc="Scene analytics", unit="video"):
        rel_name = video_path.relative_to(videos_dir).as_posix()
        result[rel_name] = _compute_video_analytics(
            video_path=video_path,
            width=width,
            height=height,
            edge_ratio=edge_ratio,
            depth_estimator=depth_estimator,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Собрать JSON-аналитику по сценам: трек движения камеры, "
            "яркость, depth-метрики, diff-метрики и optical-flow метрики по каждому кадру."
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Папка с видео-сценами (1 файл = 1 сцена, подпапки поддерживаются)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Путь к выходному JSON-файлу аналитики",
    )
    parser.add_argument("--width", type=int, default=560, help="Ширина кадров перед аналитикой")
    parser.add_argument("--height", type=int, default=240, help="Высота кадров перед аналитикой")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Лимит количества сканируемых видео (по порядку сортировки имен)",
    )
    parser.add_argument(
        "--edge-ratio",
        type=float,
        default=0.1,
        help="Доля ширины для левого/правого края карты (например, 0.1 = 10%%)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="MiDaS_small",
        choices=("MiDaS_small", "DPT_Hybrid"),
        help="MiDaS модель для depth-аналитики",
    )
    parser.add_argument(
        "--depth-device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Устройство для MiDaS",
    )
    args = parser.parse_args()

    output = run_build_scene_analytics(
        videos_dir=args.videos_dir,
        output_path=args.output_path,
        width=args.width,
        height=args.height,
        edge_ratio=args.edge_ratio,
        max_videos=args.max_videos,
        depth_model=args.depth_model,
        depth_device=args.depth_device,
    )
    logger.info("Готово: {}", output)


if __name__ == "__main__":
    main()

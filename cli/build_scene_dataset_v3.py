from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from safetensors.torch import save_file
from tqdm import tqdm

from app.src.utils.video_reader.random_access_reader import RandomAccessReader
from app.src.utils.video_reader.sequential_access_reader import SequentialAccessReader


@dataclass
class NormalizationConfig:
    frame_width: int
    frame_height: int


class MidasDepthEstimator:
    def __init__(self, *, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model: torch.nn.Module | None = None
        self.transform = None

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
            raise RuntimeError(
                "Не удалось загрузить MiDaS: отсутствует python-зависимость."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Не удалось загрузить MiDaS из torch.hub. "
                "Проверьте интернет/кэш torch и попробуйте --disable-depth."
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        if "small" in self.model_name.lower():
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    def estimate_depth_norm(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        depth_norm = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
        # Ближние объекты получают меньший вес, фон — больший (как в render_optical_flow).
        inv_depth_weight = 1.0 - depth_norm
        return depth_norm, inv_depth_weight


def _collect_video_paths(videos_dir: Path, *, video_prefix: str | None) -> list[Path]:
    patterns = ("*.mp4", "*.mkv", "*.avi", "*.mov")
    videos = sorted(
        path for pattern in patterns for path in videos_dir.glob(pattern) if path.is_file()
    )
    if video_prefix:
        videos = [path for path in videos if path.name.startswith(video_prefix)]
    if not videos:
        hint = f" с префиксом {video_prefix!r}" if video_prefix else ""
        raise SystemExit(f"В папке не найдено видеофайлов{hint}: {videos_dir}")
    return videos


def _compute_sample_frame_indices(total_frames: int, n_samples: int) -> list[int]:
    """
    Равномерная выборка по длине ролика: всегда первый и последний кадр,
    остальные — равномерно между ними (np.linspace).
    """
    if total_frames <= 0:
        return []
    if n_samples <= 0:
        raise ValueError(f"n_samples должен быть > 0, получено: {n_samples}")

    last = total_frames - 1
    if last == 0:
        return [0]

    if n_samples == 1:
        return [0]

    raw = np.linspace(0, last, num=n_samples, dtype=np.int64)
    raw[0] = 0
    raw[-1] = last

    out: list[int] = []
    seen: set[int] = set()
    for idx in raw.tolist():
        frame_id = int(idx)
        if frame_id not in seen:
            seen.add(frame_id)
            out.append(frame_id)
    return out


def _crop_center(frame_bgr: np.ndarray, center_width: int) -> np.ndarray:
    frame_width = frame_bgr.shape[1]
    left = (frame_width - center_width) // 2
    right = left + center_width
    return frame_bgr[:, left:right, :]


def _split_center_and_sides(
    frame: np.ndarray,
    *,
    center_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_width = frame.shape[1]
    left = (frame_width - center_width) // 2
    right = left + center_width
    center = frame[:, left:right, ...]
    sides = np.concatenate((frame[:, :left, ...], frame[:, right:, ...]), axis=1)
    return center, sides


def _resize_flow_with_scale(flow: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = flow.shape[:2]
    resized = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_AREA)
    resized[..., 0] *= float(target_w) / float(src_w)
    resized[..., 1] *= float(target_h) / float(src_h)
    return resized.astype(np.float32)


def _normalize_rgb_bgr_to_chw01(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(chw).contiguous()


def _normalize_depth_to_chw01(depth_map: np.ndarray) -> torch.Tensor:
    depth = np.clip(depth_map.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(depth[None, ...]).contiguous()


def _normalize_flow_to_chw(flow: np.ndarray, cfg: NormalizationConfig) -> torch.Tensor:
    normalized = flow.astype(np.float32).copy()
    normalized[..., 0] /= float(max(1, cfg.frame_width))
    normalized[..., 1] /= float(max(1, cfg.frame_height))
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).contiguous()


def _normalize_camera_move(vec_xy: np.ndarray, cfg: NormalizationConfig) -> torch.Tensor:
    norm = np.array([max(1, cfg.frame_width), max(1, cfg.frame_height)], dtype=np.float32)
    return torch.from_numpy((vec_xy.astype(np.float32) / norm).copy()).contiguous()


def _zero_frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _zero_flow(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 2), dtype=np.float32)


def _tensor_chw_rgb_to_bgr(frame_chw_rgb: torch.Tensor) -> np.ndarray:
    frame_hwc_rgb = frame_chw_rgb.permute(1, 2, 0).contiguous().cpu().numpy()
    if frame_hwc_rgb.dtype != np.uint8:
        frame_hwc_rgb = np.clip(frame_hwc_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_hwc_rgb, cv2.COLOR_RGB2BGR)


def _buffer_frame_or_zero(
    buffer: deque[torch.Tensor],
    offset: int,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, bool]:
    if offset >= len(buffer):
        return _zero_frame(height, width), True
    return _tensor_chw_rgb_to_bgr(buffer[offset]), False


def _previous_frame_bgr(
    previous_frame: torch.Tensor | None,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if previous_frame is None:
        return _zero_frame(height, width)
    return _tensor_chw_rgb_to_bgr(previous_frame)


def _worker_compute_flows(
    task: tuple[
        int,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        int,
        int,
        int,
        int,
        float,
        int,
    ],
) -> tuple[int, np.ndarray, np.ndarray]:
    (
        local_idx,
        source_center_bgr,
        step2_center_bgr,
        step4_center_bgr,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    ) = task
    source_gray = cv2.cvtColor(source_center_bgr, cv2.COLOR_BGR2GRAY)
    h, w = source_gray.shape[:2]
    half_h = max(1, h // 2)
    half_w = max(1, w // 2)

    flow_step2 = _zero_flow(h, w)
    if step2_center_bgr is not None:
        step2_gray = cv2.cvtColor(step2_center_bgr, cv2.COLOR_BGR2GRAY)
        flow_step2 = cv2.calcOpticalFlowFarneback(
            prev=source_gray,
            next=step2_gray,
            flow=None,
            pyr_scale=0.5,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        ).astype(np.float32)

    flow_step4_half = _zero_flow(half_h, half_w)
    if step4_center_bgr is not None:
        step4_gray = cv2.cvtColor(step4_center_bgr, cv2.COLOR_BGR2GRAY)
        flow_step4_full = cv2.calcOpticalFlowFarneback(
            prev=source_gray,
            next=step4_gray,
            flow=None,
            pyr_scale=0.5,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        ).astype(np.float32)
        flow_step4_half = _resize_flow_with_scale(flow_step4_full, half_w, half_h)

    return local_idx, flow_step2, flow_step4_half


def _drain_completed(
    *,
    in_flight: dict[Future, int],
    ready: dict[int, tuple[np.ndarray, np.ndarray]],
    wait_any: bool,
) -> None:
    if not in_flight:
        return
    if wait_any:
        done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
        done_futures = list(done)
    else:
        done_futures = [f for f in list(in_flight.keys()) if f.done()]
        if not done_futures:
            return
    for future in done_futures:
        in_flight.pop(future, None)
        idx, flow_step2, flow_step4_half = future.result()
        ready[idx] = (flow_step2, flow_step4_half)


def _flush_scene_rows(
    *,
    next_idx: int,
    flow_ready: dict[int, tuple[np.ndarray, np.ndarray]],
    base_rows: dict[int, dict[str, torch.Tensor | int | np.ndarray]],
    sink: dict[str, list[torch.Tensor | int]],
) -> int:
    while next_idx in base_rows and next_idx in flow_ready:
        row = base_rows.pop(next_idx)
        flow_step2, flow_step4_half = flow_ready.pop(next_idx)

        inv_depth_weight_half = row["inv_depth_weight_half"]
        assert isinstance(inv_depth_weight_half, np.ndarray)
        depth_weighted_flow = flow_step4_half * inv_depth_weight_half[..., None]
        # Камера движется в сторону, противоположную движению пикселей.
        dx = -float(np.median(depth_weighted_flow[..., 0]))
        dy = -float(np.median(depth_weighted_flow[..., 1]))
        camera_vec = np.array([dx, dy], dtype=np.float32)

        flow2_cfg = NormalizationConfig(
            frame_width=flow_step2.shape[1], frame_height=flow_step2.shape[0]
        )
        flow4_cfg = NormalizationConfig(
            frame_width=flow_step4_half.shape[1], frame_height=flow_step4_half.shape[0]
        )
        camera_cfg = NormalizationConfig(
            frame_width=flow_step4_half.shape[1],
            frame_height=flow_step4_half.shape[0],
        )

        sink["source_image_center"].append(row["source_image_center"])
        sink["source_image_sides"].append(row["source_image_sides"])
        sink["opti_map_1"].append(_normalize_flow_to_chw(flow_step2, flow2_cfg))
        sink["opti_map_2"].append(_normalize_flow_to_chw(flow_step4_half, flow4_cfg))
        sink["depth_map"].append(row["depth_map"])
        sink["camera_move_vector"].append(_normalize_camera_move(camera_vec, camera_cfg))
        sink["previous_frame_center"].append(row["previous_frame_center"])
        sink["previous_frame_sides"].append(row["previous_frame_sides"])
        sink["scene_logical_id"].append(row["scene_logical_id"])
        sink["frame_id"].append(row["frame_id"])
        next_idx += 1
    return next_idx


def _stack_scene_rows(rows: dict[str, list[torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    tensor_keys = (
        "source_image_center",
        "source_image_sides",
        "opti_map_1",
        "opti_map_2",
        "depth_map",
        "camera_move_vector",
        "previous_frame_center",
        "previous_frame_sides",
    )
    for key in tensor_keys:
        values = rows[key]
        if not values:
            continue
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        result[key] = torch.stack(tensors, dim=0).contiguous()

    scene_ids = torch.tensor(rows["scene_logical_id"], dtype=torch.int32)
    frame_ids = torch.tensor(rows["frame_id"], dtype=torch.int32)
    result["scene_logical_id"] = scene_ids.contiguous()
    result["frame_id"] = frame_ids.contiguous()
    return result


def _concat_scene_tensors(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        return {}
    keys = tuple(items[0].keys())
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        out[key] = torch.cat([item[key] for item in items], dim=0).contiguous()
    return out


def _save_dataset_part(
    *,
    output_dir: Path,
    part_idx: int,
    payload: dict[str, torch.Tensor],
) -> Path:
    out_path = output_dir / f"scene_dataset_part_{part_idx:05d}.sft"
    save_file(payload, str(out_path))
    return out_path


def _enqueue_scene_frame(
    *,
    frame_buffer: deque[torch.Tensor],
    local_idx: int,
    frame_id: int,
    scene_logical_id: int,
    height: int,
    width: int,
    center_width: int,
    depth_estimator: MidasDepthEstimator | None,
    depth_h: int,
    depth_w: int,
    pool: ProcessPoolExecutor | None,
    in_flight: dict[Future, int],
    flow_ready: dict[int, tuple[np.ndarray, np.ndarray]],
    base_rows: dict[int, dict[str, torch.Tensor | int | np.ndarray]],
    farneback_flags: int,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    previous_frame: torch.Tensor | None,
) -> None:
    source_bgr, source_missing = _buffer_frame_or_zero(frame_buffer, 0, height=height, width=width)
    prev_bgr = _previous_frame_bgr(previous_frame, height=height, width=width)
    next2_bgr, next2_missing = _buffer_frame_or_zero(frame_buffer, 2, height=height, width=width)
    next4_bgr, next4_missing = _buffer_frame_or_zero(frame_buffer, 4, height=height, width=width)

    source_center_bgr = _crop_center(source_bgr, center_width)
    prev_center_bgr = _crop_center(prev_bgr, center_width)
    source_sides_bgr = _split_center_and_sides(source_bgr, center_width=center_width)[1]
    prev_sides_bgr = _split_center_and_sides(prev_bgr, center_width=center_width)[1]

    if depth_estimator is None or source_missing:
        depth_half = np.zeros((depth_h, depth_w), dtype=np.float32)
        inv_depth_weight_half = np.ones((depth_h, depth_w), dtype=np.float32)
    else:
        depth_source = cv2.resize(
            source_center_bgr, (depth_w, depth_h), interpolation=cv2.INTER_AREA
        )
        depth_half, inv_depth_weight_half = depth_estimator.estimate_depth_norm(depth_source)

    base_rows[local_idx] = {
        "source_image_center": _normalize_rgb_bgr_to_chw01(source_center_bgr),
        "source_image_sides": _normalize_rgb_bgr_to_chw01(source_sides_bgr),
        "depth_map": _normalize_depth_to_chw01(depth_half),
        "inv_depth_weight_half": inv_depth_weight_half,
        "previous_frame_center": _normalize_rgb_bgr_to_chw01(prev_center_bgr),
        "previous_frame_sides": _normalize_rgb_bgr_to_chw01(prev_sides_bgr),
        "scene_logical_id": scene_logical_id,
        "frame_id": frame_id,
    }

    task = (
        local_idx,
        source_center_bgr,
        None if next2_missing else _crop_center(next2_bgr, center_width),
        None if next4_missing else _crop_center(next4_bgr, center_width),
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        farneback_flags,
    )
    if pool is None:
        idx, flow1, flow2 = _worker_compute_flows(task)
        flow_ready[idx] = (flow1, flow2)
    else:
        future = pool.submit(_worker_compute_flows, task)
        in_flight[future] = local_idx


def _process_scene_video(
    *,
    video_path: Path,
    scene_logical_id: int,
    width: int,
    height: int,
    center_width: int,
    depth_estimator: MidasDepthEstimator | None,
    pool: ProcessPoolExecutor | None,
    max_in_flight: int,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    use_gaussian: bool,
    samples_per_video: int | None,
) -> dict[str, torch.Tensor]:
    scene_rows: dict[str, list[torch.Tensor | int]] = {
        "source_image_center": [],
        "source_image_sides": [],
        "opti_map_1": [],
        "opti_map_2": [],
        "depth_map": [],
        "camera_move_vector": [],
        "previous_frame_center": [],
        "previous_frame_sides": [],
        "scene_logical_id": [],
        "frame_id": [],
    }

    required_window = 5
    flow_ready: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    base_rows: dict[int, dict[str, torch.Tensor | int | np.ndarray]] = {}
    in_flight: dict[Future, int] = {}
    next_idx = 0
    farneback_flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian else 0
    depth_h = max(1, height // 2)
    depth_w = max(1, center_width // 2)

    reader_cls = RandomAccessReader if samples_per_video is not None else SequentialAccessReader
    with reader_cls(
        path=str(video_path),
        width=width,
        height=height,
        normalize=False,
    ) as reader:
        total_frames = reader.total_frames
        if total_frames <= 0:
            return _stack_scene_rows(scene_rows)

        if samples_per_video is not None:
            frame_ids = _compute_sample_frame_indices(total_frames, samples_per_video)
            for local_idx, frame_id in enumerate(frame_ids):
                window = reader.read_window(frame_id, required_window)
                frame_buffer: deque[torch.Tensor] = deque(maxlen=required_window)
                for idx in range(window.shape[0]):
                    frame_buffer.append(window[idx])

                previous_frame: torch.Tensor | None = None
                if frame_id > 0:
                    prev_window = reader.read_window(frame_id - 1, 1)
                    if prev_window.shape[0] > 0:
                        previous_frame = prev_window[0]

                _enqueue_scene_frame(
                    frame_buffer=frame_buffer,
                    local_idx=local_idx,
                    frame_id=frame_id,
                    scene_logical_id=scene_logical_id,
                    height=height,
                    width=width,
                    center_width=center_width,
                    depth_estimator=depth_estimator,
                    depth_h=depth_h,
                    depth_w=depth_w,
                    pool=pool,
                    in_flight=in_flight,
                    flow_ready=flow_ready,
                    base_rows=base_rows,
                    farneback_flags=farneback_flags,
                    levels=levels,
                    winsize=winsize,
                    iterations=iterations,
                    poly_n=poly_n,
                    poly_sigma=poly_sigma,
                    previous_frame=previous_frame,
                )
                _drain_completed(in_flight=in_flight, ready=flow_ready, wait_any=False)
                next_idx = _flush_scene_rows(
                    next_idx=next_idx,
                    flow_ready=flow_ready,
                    base_rows=base_rows,
                    sink=scene_rows,
                )
                while len(in_flight) >= max_in_flight:
                    _drain_completed(in_flight=in_flight, ready=flow_ready, wait_any=True)
                    next_idx = _flush_scene_rows(
                        next_idx=next_idx,
                        flow_ready=flow_ready,
                        base_rows=base_rows,
                        sink=scene_rows,
                    )
        else:
            initial_batch = reader.read_window(required_window)
            if initial_batch.shape[0] == 0:
                return _stack_scene_rows(scene_rows)

            frame_buffer = deque(
                [initial_batch[idx] for idx in range(initial_batch.shape[0])],
                maxlen=required_window,
            )
            local_idx = 0
            previous_frame_tensor: torch.Tensor | None = None

            while True:
                _enqueue_scene_frame(
                    frame_buffer=frame_buffer,
                    local_idx=local_idx,
                    frame_id=local_idx,
                    scene_logical_id=scene_logical_id,
                    height=height,
                    width=width,
                    center_width=center_width,
                    depth_estimator=depth_estimator,
                    depth_h=depth_h,
                    depth_w=depth_w,
                    pool=pool,
                    in_flight=in_flight,
                    flow_ready=flow_ready,
                    base_rows=base_rows,
                    farneback_flags=farneback_flags,
                    levels=levels,
                    winsize=winsize,
                    iterations=iterations,
                    poly_n=poly_n,
                    poly_sigma=poly_sigma,
                    previous_frame=previous_frame_tensor,
                )
                _drain_completed(in_flight=in_flight, ready=flow_ready, wait_any=False)
                next_idx = _flush_scene_rows(
                    next_idx=next_idx,
                    flow_ready=flow_ready,
                    base_rows=base_rows,
                    sink=scene_rows,
                )

                while len(in_flight) >= max_in_flight:
                    _drain_completed(in_flight=in_flight, ready=flow_ready, wait_any=True)
                    next_idx = _flush_scene_rows(
                        next_idx=next_idx,
                        flow_ready=flow_ready,
                        base_rows=base_rows,
                        sink=scene_rows,
                    )

                local_idx += 1
                if local_idx >= total_frames:
                    break

                previous_frame_tensor = frame_buffer[0]
                batch = reader.read_window(1)
                if batch.shape[0] == 0:
                    break
                frame_buffer.append(batch[0])

        while in_flight:
            _drain_completed(in_flight=in_flight, ready=flow_ready, wait_any=True)
            next_idx = _flush_scene_rows(
                next_idx=next_idx,
                flow_ready=flow_ready,
                base_rows=base_rows,
                sink=scene_rows,
            )
        _flush_scene_rows(
            next_idx=next_idx,
            flow_ready=flow_ready,
            base_rows=base_rows,
            sink=scene_rows,
        )
    return _stack_scene_rows(scene_rows)


def run_build_scene_dataset(
    *,
    videos_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    center_width: int,
    part_size_frames: int,
    video_prefix: str | None,
    samples_per_video: int,
    disable_depth: bool,
    depth_model: str,
    depth_device: str,
    num_workers: int,
    max_in_flight: int,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    use_gaussian: bool,
) -> list[Path]:
    if not videos_dir.is_dir():
        raise SystemExit(f"--videos-dir должен указывать на существующую папку: {videos_dir}")
    if center_width > width:
        raise SystemExit("--center-width должен быть <= --width")
    if (width - center_width) % 2 != 0:
        raise SystemExit(
            "Разница между --width и --center-width должна быть четной для симметричного crop"
        )
    if part_size_frames <= 0:
        raise SystemExit("--part-size-frames должен быть > 0")
    if samples_per_video <= 0:
        raise SystemExit("--samples-per-video должен быть > 0")

    videos = _collect_video_paths(videos_dir, video_prefix=video_prefix)
    logger.info(
        "Видео к обработке: {} (prefix={!r}, samples_per_video={})",
        len(videos),
        video_prefix or "",
        samples_per_video,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    max_in_flight = max(num_workers, max_in_flight)

    depth_estimator: MidasDepthEstimator | None = None
    if not disable_depth:
        depth_estimator = MidasDepthEstimator(model_name=depth_model, device=depth_device)
        logger.info("Загрузка MiDaS: {} (device={})", depth_model, depth_estimator.device)
        depth_estimator.load()

    saved_parts: list[Path] = []
    bucket_tensors: list[dict[str, torch.Tensor]] = []
    bucket_frames = 0
    part_idx = 1
    scene_logical_id = 0

    pool: ProcessPoolExecutor | None = None
    if num_workers > 1:
        pool = ProcessPoolExecutor(max_workers=num_workers)

    try:
        for video_path in tqdm(videos, desc="Scene videos", unit="video"):
            scene_payload = _process_scene_video(
                video_path=video_path,
                scene_logical_id=scene_logical_id,
                width=width,
                height=height,
                center_width=center_width,
                depth_estimator=depth_estimator,
                pool=pool,
                max_in_flight=max_in_flight,
                levels=levels,
                winsize=winsize,
                iterations=iterations,
                poly_n=poly_n,
                poly_sigma=poly_sigma,
                use_gaussian=use_gaussian,
                samples_per_video=samples_per_video,
            )
            scene_frames = int(scene_payload["frame_id"].shape[0])
            if scene_frames == 0:
                logger.warning("Пропуск пустого видео: {}", video_path.name)
                continue

            logger.info(
                "Сцена {} <- {} (frames={})",
                scene_logical_id,
                video_path.name,
                scene_frames,
            )
            bucket_tensors.append(scene_payload)
            bucket_frames += scene_frames
            scene_logical_id += 1

            if bucket_frames >= part_size_frames:
                part_payload = _concat_scene_tensors(bucket_tensors)
                out_path = _save_dataset_part(
                    output_dir=output_dir,
                    part_idx=part_idx,
                    payload=part_payload,
                )
                logger.info(
                    "Сохранена часть {}: frames={}, scenes={}",
                    out_path.name,
                    int(part_payload["frame_id"].shape[0]),
                    int(part_payload["scene_logical_id"].unique().numel()),
                )
                saved_parts.append(out_path)
                part_idx += 1
                bucket_tensors = []
                bucket_frames = 0
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    if bucket_tensors:
        part_payload = _concat_scene_tensors(bucket_tensors)
        out_path = _save_dataset_part(
            output_dir=output_dir,
            part_idx=part_idx,
            payload=part_payload,
        )
        logger.info(
            "Сохранена часть {}: frames={}, scenes={}",
            out_path.name,
            int(part_payload["frame_id"].shape[0]),
            int(part_payload["scene_logical_id"].unique().numel()),
        )
        saved_parts.append(out_path)

    return saved_parts


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
            "Собрать scene dataset (.sft) с полями source_image_center/source_image_sides/opti/depth/diff/"
            "camera_move_vector/previous_frame_center/previous_frame_sides + scene_logical_id/frame_id."
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Папка с видео, уже разрезанными по logical scenes (1 файл = 1 сцена)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Папка для .sft частей")
    parser.add_argument("--width", type=int, required=True, help="Ширина W для source_image")
    parser.add_argument("--height", type=int, required=True, help="Высота H для source_image")
    parser.add_argument(
        "--center-width",
        type=int,
        required=True,
        help="Ширина W_C для центрального crop",
    )
    parser.add_argument(
        "--part-size-frames",
        type=int,
        default=512,
        help="Целевой размер одной части датасета в количестве кадров",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default="E01_",
        help="Обрабатывать только файлы с этим префиксом имени (пустая строка — все видео)",
    )
    parser.add_argument(
        "--samples-per-video",
        type=int,
        default=2,
        help=(
            "Сколько кадров брать из каждого ролика: первый, последний и равномерно "
            "распределённые между ними"
        ),
    )
    parser.add_argument(
        "--disable-depth",
        action="store_true",
        help="Отключить расчёт depth_map (вместо него пишутся нули)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="MiDaS_small",
        choices=("MiDaS_small", "DPT_Hybrid"),
        help="MiDaS модель для depth_map",
    )
    parser.add_argument(
        "--depth-device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Устройство для MiDaS",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Количество worker-процессов для расчета optical flow",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=16,
        help="Максимум flow-задач в очереди",
    )
    parser.add_argument("--levels", type=int, default=3, help="Число уровней пирамиды Farneback")
    parser.add_argument("--winsize", type=int, default=10, help="Размер окна усреднения Farneback")
    parser.add_argument("--iterations", type=int, default=3, help="Итерации Farneback")
    parser.add_argument(
        "--poly-n", type=int, default=5, help="Размер окрестности Farneback (5 или 7)"
    )
    parser.add_argument("--poly-sigma", type=float, default=1.2, help="Sigma Farneback")
    parser.add_argument(
        "--use-gaussian",
        action="store_true",
        help="Использовать OPTFLOW_FARNEBACK_GAUSSIAN",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width и --height должны быть > 0")
    if args.center_width <= 0:
        raise SystemExit("--center-width должен быть > 0")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers должен быть > 0")
    if args.max_in_flight <= 0:
        raise SystemExit("--max-in-flight должен быть > 0")
    if args.levels <= 0 or args.winsize <= 0 or args.iterations <= 0:
        raise SystemExit("--levels, --winsize и --iterations должны быть > 0")
    if args.poly_n not in (5, 7):
        raise SystemExit("--poly-n должен быть 5 или 7")
    if args.poly_sigma <= 0:
        raise SystemExit("--poly-sigma должен быть > 0")

    video_prefix = args.video_prefix or None

    saved_parts = run_build_scene_dataset(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        center_width=args.center_width,
        part_size_frames=args.part_size_frames,
        video_prefix=video_prefix,
        samples_per_video=args.samples_per_video,
        disable_depth=args.disable_depth,
        depth_model=args.depth_model,
        depth_device=args.depth_device,
        num_workers=args.num_workers,
        max_in_flight=args.max_in_flight,
        levels=args.levels,
        winsize=args.winsize,
        iterations=args.iterations,
        poly_n=args.poly_n,
        poly_sigma=args.poly_sigma,
        use_gaussian=args.use_gaussian,
    )
    logger.info("Готово: сохранено частей {}", len(saved_parts))


if __name__ == "__main__":
    main()

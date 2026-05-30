from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from app.src.utils.video_reader.sequential_access_reader import SequentialAccessReader


def _tensor_chw_rgb_to_bgr(frame_chw_rgb: torch.Tensor) -> np.ndarray:
    if frame_chw_rgb.ndim != 3 or frame_chw_rgb.shape[0] != 3:
        raise ValueError(f"Ожидался тензор [3, H, W], получено: {tuple(frame_chw_rgb.shape)}")
    frame_hwc_rgb = frame_chw_rgb.permute(1, 2, 0).contiguous().cpu().numpy()
    if frame_hwc_rgb.dtype != np.uint8:
        frame_hwc_rgb = np.clip(frame_hwc_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_hwc_rgb, cv2.COLOR_RGB2BGR)


def _flow_to_bgr_visual(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _upsample_flow(flow: np.ndarray, *, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = flow.shape[:2]
    upsampled = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    upsampled[..., 0] *= float(target_w) / float(src_w)
    upsampled[..., 1] *= float(target_h) / float(src_h)
    return upsampled


def _difference_to_bgr_visual(prev_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(prev_bgr, next_bgr)
    diff_f = diff.astype(np.float32)
    scale = float(np.percentile(diff_f, 99.0))
    if scale < 1.0:
        scale = 1.0
    return np.clip(diff_f * (255.0 / scale), 0, 255).astype(np.uint8)


def _warp_frame_with_flow(frame_bgr: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame_bgr.shape[:2] != flow.shape[:2]:
        raise ValueError(
            "Размеры кадра и flow не совпадают: "
            f"{frame_bgr.shape[:2]} vs {flow.shape[:2]}"
        )
    h, w = frame_bgr.shape[:2]
    ys, xs = np.indices((h, w), dtype=np.float32)
    dst_x = np.rint(xs + flow[..., 0]).astype(np.int32)
    dst_y = np.rint(ys + flow[..., 1]).astype(np.int32)
    valid = (dst_x >= 0) & (dst_x < w) & (dst_y >= 0) & (dst_y < h)
    warped = np.zeros_like(frame_bgr)
    assigned = np.zeros((h, w), dtype=bool)
    src_x = xs.astype(np.int32)
    src_y = ys.astype(np.int32)
    warped[dst_y[valid], dst_x[valid]] = frame_bgr[src_y[valid], src_x[valid]]
    assigned[dst_y[valid], dst_x[valid]] = True
    return warped, assigned


def _camera_direction_label(dx: float, dy: float, eps: float = 0.15) -> str:
    parts: list[str] = []
    if dx > eps:
        parts.append("RIGHT")
    elif dx < -eps:
        parts.append("LEFT")
    if dy > eps:
        parts.append("DOWN")
    elif dy < -eps:
        parts.append("UP")
    if not parts:
        return "STABLE"
    return "+".join(parts)


def _overlay_camera_motion(
    frame_bgr: np.ndarray,
    *,
    dx: float,
    dy: float,
    speed: float,
    arrow_scale: float,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    origin = (w // 2, h // 2)
    tip_x = int(np.clip(origin[0] + dx * arrow_scale, 0, w - 1))
    tip_y = int(np.clip(origin[1] + dy * arrow_scale, 0, h - 1))
    cv2.arrowedLine(out, origin, (tip_x, tip_y), (0, 255, 255), 2, tipLength=0.28)
    label = _camera_direction_label(dx, dy)
    cv2.putText(
        out,
        f"cam: {label}",
        (10, max(18, int(h * 0.045))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"v={speed:.2f}px/f  dx={dx:.2f} dy={dy:.2f}",
        (10, max(36, int(h * 0.09))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _compose_grid_frame(
    *,
    main_frame: np.ndarray,
    flow_map_step1: np.ndarray,
    flow_map_step4: np.ndarray,
    depth_map: np.ndarray,
    diff_map: np.ndarray,
    warped_frame: np.ndarray,
    dx: float,
    dy: float,
    speed: float,
    arrow_scale: float,
) -> np.ndarray:
    top_left = _overlay_camera_motion(
        main_frame, dx=dx, dy=dy, speed=speed, arrow_scale=arrow_scale
    )
    top_right = depth_map
    mid_left = flow_map_step1
    mid_right = flow_map_step4
    bottom_left = diff_map
    bottom_right = warped_frame
    top_row = np.concatenate((top_left, top_right), axis=1)
    middle_row = np.concatenate((mid_left, mid_right), axis=1)
    bottom_row = np.concatenate((bottom_left, bottom_right), axis=1)
    return np.concatenate((top_row, middle_row, bottom_row), axis=0)


class MidasDepthEstimator:
    def __init__(self, *, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model: Any | None = None
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
            raise RuntimeError(
                "Не удалось загрузить MiDaS: отсутствует python-зависимость. "
                "Проверьте stack trace и установите требуемый пакет."
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

    def estimate(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        depth_norm_f32 = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
        # Инвертируем: ближние объекты (обычно выше в MiDaS) получают меньший вес,
        # фон — больший.
        inv_depth_weight = 1.0 - depth_norm_f32
        depth_vis = cv2.applyColorMap(
            np.clip(depth_norm_f32 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        return depth_vis, inv_depth_weight


def _worker_compute_flow_maps(
    task: tuple[
        int,
        np.ndarray,
        np.ndarray,
        list[np.ndarray],
        float,
        int,
        int,
        int,
        int,
        float,
        int,
    ],
) -> tuple[int, np.ndarray, np.ndarray]:
    (
        pair_idx,
        prev_bgr,
        curr_main_bgr,
        curr_speed_bgr_list,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    ) = task

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_main_gray = cv2.cvtColor(curr_main_bgr, cv2.COLOR_BGR2GRAY)

    flow_step1 = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_main_gray,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    full_h, full_w = prev_gray.shape[:2]
    half_w = max(1, int(round(full_w * 0.5)))
    half_h = max(1, int(round(full_h * 0.5)))
    prev_gray_half = cv2.resize(prev_gray, (half_w, half_h), interpolation=cv2.INTER_AREA)
    flow_step4_half_acc = np.zeros((half_h, half_w, 2), dtype=np.float32)
    for curr_speed_bgr in curr_speed_bgr_list:
        curr_speed_gray_full = cv2.cvtColor(curr_speed_bgr, cv2.COLOR_BGR2GRAY)
        curr_speed_gray_half = cv2.resize(
            curr_speed_gray_full,
            (half_w, half_h),
            interpolation=cv2.INTER_AREA,
        )
        flow_step4_half_acc += cv2.calcOpticalFlowFarneback(
            prev=prev_gray_half,
            next=curr_speed_gray_half,
            flow=None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        )
    flow_step4_half = flow_step4_half_acc / float(len(curr_speed_bgr_list))
    return pair_idx, flow_step1, flow_step4_half


def _drain_completed(
    *,
    in_flight: dict[Future, int],
    flow_ready: dict[int, tuple[np.ndarray, np.ndarray]],
    wait_any: bool,
) -> None:
    if not in_flight:
        return
    if wait_any:
        done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
        done_futures = list(done)
    else:
        done_futures = [fut for fut in list(in_flight.keys()) if fut.done()]
        if not done_futures:
            return
    for fut in done_futures:
        in_flight.pop(fut, None)
        pair_idx, flow_step1, flow_step4 = fut.result()
        flow_ready[pair_idx] = (flow_step1, flow_step4)


def _flush_ready(
    *,
    writer: cv2.VideoWriter,
    next_write_idx: int,
    base_ready: dict[int, np.ndarray],
    depth_ready: dict[int, np.ndarray],
    depth_weight_ready_half: dict[int, np.ndarray],
    diff_ready: dict[int, np.ndarray],
    flow_ready: dict[int, tuple[np.ndarray, np.ndarray]],
    pbar: tqdm,
    arrow_scale: float,
    prev_warped_frame: np.ndarray | None,
) -> tuple[int, np.ndarray | None]:
    while (
        next_write_idx in base_ready
        and next_write_idx in depth_ready
        and next_write_idx in depth_weight_ready_half
        and next_write_idx in diff_ready
        and next_write_idx in flow_ready
    ):
        main_frame = base_ready.pop(next_write_idx)
        depth_map = depth_ready.pop(next_write_idx)
        inv_depth_weight_half = depth_weight_ready_half.pop(next_write_idx)
        diff_map = diff_ready.pop(next_write_idx)
        flow_step1, flow_step4_half = flow_ready.pop(next_write_idx)
        depth_weighted_flow_for_camera = flow_step4_half * inv_depth_weight_half[..., None]
        # Камера движется в сторону, противоположную движению пикселей.
        dx = -float(np.median(depth_weighted_flow_for_camera[..., 0]))
        dy = -float(np.median(depth_weighted_flow_for_camera[..., 1]))
        speed = float(np.hypot(dx, dy))
        target_w, target_h = main_frame.shape[1], main_frame.shape[0]
        flow_step4 = _upsample_flow(flow_step4_half, target_w=target_w, target_h=target_h)
        warp_source = prev_warped_frame if prev_warped_frame is not None else main_frame
        warped_by_flow, filled_mask = _warp_frame_with_flow(warp_source, flow_step4)
        warped_by_flow[~filled_mask] = main_frame[~filled_mask]
        prev_warped_frame = warped_by_flow

        composed = _compose_grid_frame(
            main_frame=main_frame,
            flow_map_step1=_flow_to_bgr_visual(flow_step1),
            flow_map_step4=_flow_to_bgr_visual(flow_step4),
            depth_map=depth_map,
            diff_map=diff_map,
            warped_frame=warped_by_flow,
            dx=dx,
            dy=dy,
            speed=speed,
            arrow_scale=arrow_scale,
        )
        writer.write(composed)
        pbar.update(1)
        next_write_idx += 1
    return next_write_idx, prev_warped_frame


def _resolve_max_in_flight(num_workers: int, max_in_flight: int | None) -> int:
    if max_in_flight is None:
        return max(2, num_workers * 2)
    if max_in_flight < num_workers:
        raise ValueError("--max-in-flight должен быть >= --num-workers")
    return max_in_flight


def render_optical_flow_video(
    *,
    video_path: Path,
    output_path: Path,
    resize_scale: float,
    frame_step: int,
    speed_map_step: int,
    speed_map_avg_frames: int,
    codec: str,
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    use_gaussian: bool,
    num_workers: int,
    max_in_flight: int | None,
    max_output_frames: int | None,
    disable_depth: bool,
    depth_model: str,
    depth_device: str,
    camera_arrow_scale: float,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if len(codec) != 4:
        raise ValueError("--codec должен содержать 4 символа")
    if not 0.0 < resize_scale <= 1.0:
        raise ValueError("--resize-scale должен быть в диапазоне (0, 1]")
    if frame_step <= 0:
        raise ValueError("--frame-step должен быть > 0")
    if speed_map_step <= 0:
        raise ValueError("--speed-map-step должен быть > 0")
    if speed_map_avg_frames <= 0:
        raise ValueError("--speed-map-avg-frames должен быть > 0")
    if not 0.0 < pyr_scale < 1.0:
        raise ValueError("--pyr-scale должен быть в диапазоне (0, 1)")
    if levels <= 0:
        raise ValueError("--levels должен быть > 0")
    if winsize <= 0:
        raise ValueError("--winsize должен быть > 0")
    if iterations <= 0:
        raise ValueError("--iterations должен быть > 0")
    if poly_n not in (5, 7):
        raise ValueError("--poly-n должен быть 5 или 7 (ограничение Farneback)")
    if num_workers <= 0:
        raise ValueError("--num-workers должен быть > 0")
    if camera_arrow_scale <= 0:
        raise ValueError("--camera-arrow-scale должен быть > 0")
    if max_output_frames is not None and max_output_frames <= 0:
        raise ValueError("--max-output-frames должен быть > 0 или None")
    max_in_flight = _resolve_max_in_flight(num_workers, max_in_flight)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("Не удалось определить ширину/высоту входного видео")
    max_step = max(frame_step, speed_map_step)
    required_window = max_step + speed_map_avg_frames
    if total_frames < required_window:
        raise RuntimeError(
            f"Недостаточно кадров: total={total_frames}, нужно минимум {required_window} "
            f"для frame-step={frame_step}, speed-map-step={speed_map_step} "
            f"и speed-map-avg-frames={speed_map_avg_frames}"
        )
    if fps <= 0:
        fps = 25.0

    scaled_width = max(1, int(round(width * resize_scale)))
    scaled_height = max(1, int(round(height * resize_scale)))
    expected_pairs = total_frames - (max_step + speed_map_avg_frames - 1)
    if max_output_frames is not None:
        expected_pairs = min(expected_pairs, max_output_frames)
    out_width = scaled_width * 2
    out_height = scaled_height * 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть writer для выходного файла: {output_path}")

    farneback_flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian else 0
    logger.info(
        "Pipeline: workers={}, max_in_flight={}, frame_step={}, speed_map_step={}, speed_map_avg_frames={}",
        num_workers,
        max_in_flight,
        frame_step,
        speed_map_step,
        speed_map_avg_frames,
    )

    depth_estimator: MidasDepthEstimator | None = None
    if not disable_depth:
        depth_estimator = MidasDepthEstimator(model_name=depth_model, device=depth_device)
        logger.info(
            "Загрузка depth модели MiDaS: {} (device={})", depth_model, depth_estimator.device
        )
        depth_estimator.load()

    base_ready: dict[int, np.ndarray] = {}
    depth_ready: dict[int, np.ndarray] = {}
    depth_weight_ready_half: dict[int, np.ndarray] = {}
    diff_ready: dict[int, np.ndarray] = {}
    flow_ready: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    in_flight: dict[Future, int] = {}

    with SequentialAccessReader(
        path=str(video_path),
        height=scaled_height,
        width=scaled_width,
        normalize=False,
    ) as reader:
        initial_batch = reader.read_window(required_window)
        if initial_batch.shape[0] < required_window:
            raise RuntimeError(
                f"Недостаточно кадров для max_step={max_step}: "
                f"доступно {initial_batch.shape[0]}, нужно минимум {required_window}"
            )

        frame_buffer: deque[torch.Tensor] = deque(
            [initial_batch[idx] for idx in range(initial_batch.shape[0])],
            maxlen=required_window,
        )

        pool: ProcessPoolExecutor | None = None
        if num_workers > 1:
            pool = ProcessPoolExecutor(max_workers=num_workers)

        pair_idx = 0
        next_write_idx = 0
        prev_warped_frame: np.ndarray | None = None
        try:
            with tqdm(total=expected_pairs, desc="Optical flow", unit="pair") as pbar:
                while True:
                    if pair_idx >= expected_pairs:
                        break
                    prev_bgr = _tensor_chw_rgb_to_bgr(frame_buffer[0])
                    curr_main_bgr = _tensor_chw_rgb_to_bgr(frame_buffer[frame_step])
                    curr_speed_bgr_list = [
                        _tensor_chw_rgb_to_bgr(frame_buffer[speed_map_step + offset])
                        for offset in range(speed_map_avg_frames)
                    ]
                    next_bgr = _tensor_chw_rgb_to_bgr(frame_buffer[1])

                    if depth_estimator is not None:
                        depth_h = max(1, int(round(prev_bgr.shape[0] * 0.5)))
                        depth_w = max(1, int(round(prev_bgr.shape[1] * 0.5)))
                        prev_bgr_half = cv2.resize(
                            prev_bgr, (depth_w, depth_h), interpolation=cv2.INTER_AREA
                        )
                        depth_map_half, inv_depth_weight_half = depth_estimator.estimate(
                            prev_bgr_half
                        )
                        depth_map = cv2.resize(
                            depth_map_half,
                            (prev_bgr.shape[1], prev_bgr.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        depth_map = np.zeros_like(prev_bgr)
                        depth_h = max(1, int(round(prev_bgr.shape[0] * 0.5)))
                        depth_w = max(1, int(round(prev_bgr.shape[1] * 0.5)))
                        inv_depth_weight_half = np.ones((depth_h, depth_w), dtype=np.float32)
                        cv2.putText(
                            depth_map,
                            "Depth disabled",
                            (10, max(24, int(depth_map.shape[0] * 0.08))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    diff_map = _difference_to_bgr_visual(prev_bgr, next_bgr)
                    base_ready[pair_idx] = prev_bgr
                    depth_ready[pair_idx] = depth_map
                    depth_weight_ready_half[pair_idx] = inv_depth_weight_half
                    diff_ready[pair_idx] = diff_map

                    task = (
                        pair_idx,
                        prev_bgr,
                        curr_main_bgr,
                        curr_speed_bgr_list,
                        pyr_scale,
                        levels,
                        winsize,
                        iterations,
                        poly_n,
                        poly_sigma,
                        farneback_flags,
                    )
                    if pool is None:
                        idx, flow_map, speed_map = _worker_compute_flow_maps(task)
                        flow_ready[idx] = (flow_map, speed_map)
                    else:
                        fut = pool.submit(_worker_compute_flow_maps, task)
                        in_flight[fut] = pair_idx
                    pair_idx += 1

                    _drain_completed(in_flight=in_flight, flow_ready=flow_ready, wait_any=False)
                    next_write_idx, prev_warped_frame = _flush_ready(
                        writer=writer,
                        next_write_idx=next_write_idx,
                        base_ready=base_ready,
                        depth_ready=depth_ready,
                        depth_weight_ready_half=depth_weight_ready_half,
                        diff_ready=diff_ready,
                        flow_ready=flow_ready,
                        pbar=pbar,
                        arrow_scale=camera_arrow_scale,
                        prev_warped_frame=prev_warped_frame,
                    )

                    batch = reader.read_window(1)
                    if batch.shape[0] == 0:
                        break
                    frame_buffer.append(batch[0])

                    while len(in_flight) >= max_in_flight:
                        _drain_completed(in_flight=in_flight, flow_ready=flow_ready, wait_any=True)
                        next_write_idx, prev_warped_frame = _flush_ready(
                            writer=writer,
                            next_write_idx=next_write_idx,
                            base_ready=base_ready,
                            depth_ready=depth_ready,
                            depth_weight_ready_half=depth_weight_ready_half,
                            diff_ready=diff_ready,
                            flow_ready=flow_ready,
                            pbar=pbar,
                            arrow_scale=camera_arrow_scale,
                            prev_warped_frame=prev_warped_frame,
                        )

                while in_flight:
                    _drain_completed(in_flight=in_flight, flow_ready=flow_ready, wait_any=True)
                    next_write_idx, prev_warped_frame = _flush_ready(
                        writer=writer,
                        next_write_idx=next_write_idx,
                        base_ready=base_ready,
                        depth_ready=depth_ready,
                        depth_weight_ready_half=depth_weight_ready_half,
                        diff_ready=diff_ready,
                        flow_ready=flow_ready,
                        pbar=pbar,
                        arrow_scale=camera_arrow_scale,
                        prev_warped_frame=prev_warped_frame,
                    )
                next_write_idx, prev_warped_frame = _flush_ready(
                    writer=writer,
                    next_write_idx=next_write_idx,
                    base_ready=base_ready,
                    depth_ready=depth_ready,
                    depth_weight_ready_half=depth_weight_ready_half,
                    diff_ready=diff_ready,
                    flow_ready=flow_ready,
                    pbar=pbar,
                    arrow_scale=camera_arrow_scale,
                    prev_warped_frame=prev_warped_frame,
                )
                if next_write_idx != expected_pairs:
                    raise RuntimeError(
                        f"Неполная запись кадров: written={next_write_idx}, expected={expected_pairs}"
                    )
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=False)

    writer.release()
    logger.info(
        "Готово: {} (pairs={}, frame_step={}, speed_map_step={}, speed_map_avg_frames={}, depth={})",
        output_path,
        expected_pairs,
        frame_step,
        speed_map_step,
        speed_map_avg_frames,
        "off" if disable_depth else depth_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "3x2 визуализация: "
            "ROW1=[video+camera, depth(0.5->upsample)], "
            "ROW2=[flow_step1(scale=1.0), flow_stepN(avg@0.5->upsample)], "
            "ROW3=[color_diff(curr,next), depth_weighted_flow_for_camera]."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True, help="Путь к входному видео")
    parser.add_argument("--output-path", type=Path, required=True, help="Путь к выходному видео")
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=1.0,
        help="Масштаб ресайза входных кадров перед обработкой (0..1]",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Шаг для основного optical flow (TR): diff(i, i+frame-step)",
    )
    parser.add_argument(
        "--speed-map-step",
        type=int,
        default=4,
        help="Шаг для второго optical flow: diff(i, i+speed-map-step)",
    )
    parser.add_argument(
        "--speed-map-avg-frames",
        type=int,
        default=2,
        help="Усреднение второго optical flow по N кадрам (i+step ... i+step+N-1)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="4-символьный кодек OpenCV VideoWriter (по умолчанию mp4v)",
    )
    parser.add_argument("--pyr-scale", type=float, default=0.5, help="Масштаб пирамиды Farneback")
    parser.add_argument("--levels", type=int, default=3, help="Число уровней пирамиды")
    parser.add_argument("--winsize", type=int, default=10, help="Размер окна усреднения")
    parser.add_argument("--iterations", type=int, default=3, help="Итерации на уровень пирамиды")
    parser.add_argument("--poly-n", type=int, default=5, help="Размер окрестности (5 или 7)")
    parser.add_argument("--poly-sigma", type=float, default=1.2, help="Sigma для Farneback")
    parser.add_argument(
        "--use-gaussian",
        action="store_true",
        help="Использовать OPTFLOW_FARNEBACK_GAUSSIAN",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Количество worker-процессов для расчёта flow",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Максимум задач в очереди reader->workers (по умолчанию 2 * num-workers)",
    )
    parser.add_argument(
        "--max-output-frames",
        type=int,
        default=None,
        help="Лимит генерируемых кадров. Если не указан, рендерится всё видео",
    )
    parser.add_argument(
        "--disable-depth",
        action="store_true",
        help="Отключить depth map (если не хотите качать/грузить MiDaS)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="MiDaS_small",
        choices=("MiDaS_small", "DPT_Hybrid"),
        help="MiDaS модель для depth карты",
    )
    parser.add_argument(
        "--depth-device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Устройство для depth модели",
    )
    parser.add_argument(
        "--camera-arrow-scale",
        type=float,
        default=12.0,
        help="Масштаб стрелки глобального движения камеры",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    render_optical_flow_video(
        video_path=args.video_path,
        output_path=args.output_path,
        resize_scale=args.resize_scale,
        frame_step=args.frame_step,
        speed_map_step=args.speed_map_step,
        speed_map_avg_frames=args.speed_map_avg_frames,
        codec=args.codec,
        pyr_scale=args.pyr_scale,
        levels=args.levels,
        winsize=args.winsize,
        iterations=args.iterations,
        poly_n=args.poly_n,
        poly_sigma=args.poly_sigma,
        use_gaussian=args.use_gaussian,
        num_workers=args.num_workers,
        max_in_flight=args.max_in_flight,
        max_output_frames=args.max_output_frames,
        disable_depth=args.disable_depth,
        depth_model=args.depth_model,
        depth_device=args.depth_device,
        camera_arrow_scale=args.camera_arrow_scale,
    )


if __name__ == "__main__":
    main()

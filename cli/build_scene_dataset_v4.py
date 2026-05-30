from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from safetensors.torch import save_file
from tqdm import tqdm

from app.src.utils.video_reader.random_access_reader import RandomAccessReader
from app.src.utils.video_reader.sequential_access_reader import SequentialAccessReader


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
    """Равномерная выборка по длине ролика: первый, последний и точки между ними."""
    if total_frames <= 0:
        return []
    if n_samples <= 0:
        raise ValueError(f"n_samples должен быть > 0, получено: {n_samples}")

    last = total_frames - 1
    if last == 0 or n_samples == 1:
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


def _normalize_rgb_bgr_to_chw01(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(chw).contiguous()


def _zero_frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


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
) -> np.ndarray:
    if offset >= len(buffer):
        return _zero_frame(height, width)
    return _tensor_chw_rgb_to_bgr(buffer[offset])


def _previous_frame_bgr(
    previous_frame: torch.Tensor | None,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if previous_frame is None:
        return _zero_frame(height, width)
    return _tensor_chw_rgb_to_bgr(previous_frame)


def _split_frame_into_half_views(
    frame_bgr: np.ndarray,
    *,
    center_width: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return left view and mirrored-right view as (center_half, side_half)."""
    frame_width = frame_bgr.shape[1]
    side_width = (frame_width - center_width) // 2
    center_left = side_width
    center_mid = center_left + center_width // 2
    center_right = center_left + center_width

    left_view = (
        frame_bgr[:, center_left:center_mid, :],
        frame_bgr[:, :side_width, :],
    )
    right_view = (
        cv2.flip(frame_bgr[:, center_mid:center_right, :], 1),
        cv2.flip(frame_bgr[:, center_right:, :], 1),
    )
    return left_view, right_view


def _append_half_frame_rows(
    *,
    rows: dict[str, list[torch.Tensor | int]],
    source_bgr: np.ndarray,
    previous_bgr: np.ndarray,
    center_width: int,
    scene_logical_id: int,
    frame_id: int,
) -> None:
    source_views = _split_frame_into_half_views(source_bgr, center_width=center_width)
    previous_views = _split_frame_into_half_views(previous_bgr, center_width=center_width)

    for (source_center, source_side), (previous_center, previous_side) in zip(
        source_views,
        previous_views,
        strict=True,
    ):
        rows["source_image_center"].append(_normalize_rgb_bgr_to_chw01(source_center))
        rows["source_image_sides"].append(_normalize_rgb_bgr_to_chw01(source_side))
        rows["previous_frame_center"].append(_normalize_rgb_bgr_to_chw01(previous_center))
        rows["previous_frame_sides"].append(_normalize_rgb_bgr_to_chw01(previous_side))
        rows["scene_logical_id"].append(scene_logical_id)
        rows["frame_id"].append(frame_id)


def _stack_scene_rows(rows: dict[str, list[torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    tensor_keys = (
        "source_image_center",
        "source_image_sides",
        "previous_frame_center",
        "previous_frame_sides",
    )
    for key in tensor_keys:
        values = rows[key]
        if values:
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            result[key] = torch.stack(tensors, dim=0).contiguous()

    result["scene_logical_id"] = torch.tensor(rows["scene_logical_id"], dtype=torch.int32)
    result["frame_id"] = torch.tensor(rows["frame_id"], dtype=torch.int32)
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


def _process_scene_video(
    *,
    video_path: Path,
    scene_logical_id: int,
    width: int,
    height: int,
    center_width: int,
    samples_per_video: int | None,
) -> dict[str, torch.Tensor]:
    scene_rows: dict[str, list[torch.Tensor | int]] = {
        "source_image_center": [],
        "source_image_sides": [],
        "previous_frame_center": [],
        "previous_frame_sides": [],
        "scene_logical_id": [],
        "frame_id": [],
    }

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
            for frame_id in frame_ids:
                window = reader.read_window(frame_id, 1)
                frame_buffer: deque[torch.Tensor] = deque(
                    [window[idx] for idx in range(window.shape[0])],
                    maxlen=1,
                )
                source_bgr = _buffer_frame_or_zero(
                    frame_buffer,
                    0,
                    height=height,
                    width=width,
                )

                previous_frame: torch.Tensor | None = None
                if frame_id > 0:
                    previous_window = reader.read_window(frame_id - 1, 1)
                    if previous_window.shape[0] > 0:
                        previous_frame = previous_window[0]
                previous_bgr = _previous_frame_bgr(
                    previous_frame,
                    height=height,
                    width=width,
                )

                _append_half_frame_rows(
                    rows=scene_rows,
                    source_bgr=source_bgr,
                    previous_bgr=previous_bgr,
                    center_width=center_width,
                    scene_logical_id=scene_logical_id,
                    frame_id=frame_id,
                )
        else:
            previous_frame: torch.Tensor | None = None
            frame_id = 0
            while True:
                batch = reader.read_window(1)
                if batch.shape[0] == 0:
                    break

                source_frame = batch[0]
                source_bgr = _tensor_chw_rgb_to_bgr(source_frame)
                previous_bgr = _previous_frame_bgr(
                    previous_frame,
                    height=height,
                    width=width,
                )

                _append_half_frame_rows(
                    rows=scene_rows,
                    source_bgr=source_bgr,
                    previous_bgr=previous_bgr,
                    center_width=center_width,
                    scene_logical_id=scene_logical_id,
                    frame_id=frame_id,
                )

                previous_frame = source_frame
                frame_id += 1

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
) -> list[Path]:
    if not videos_dir.is_dir():
        raise SystemExit(f"--videos-dir должен указывать на существующую папку: {videos_dir}")
    if center_width > width:
        raise SystemExit("--center-width должен быть <= --width")
    if center_width % 2 != 0:
        raise SystemExit("--center-width должен быть четным, чтобы делить центр пополам")
    if (width - center_width) % 2 != 0:
        raise SystemExit(
            "Разница между --width и --center-width должна быть четной для симметричного crop"
        )
    if (width - center_width) <= 0:
        raise SystemExit("--width должен быть больше --center-width, чтобы существовали боковые части")
    if part_size_frames <= 0:
        raise SystemExit("--part-size-frames должен быть > 0")
    if samples_per_video <= 0:
        raise SystemExit("--samples-per-video должен быть > 0")

    videos = _collect_video_paths(videos_dir, video_prefix=video_prefix)
    logger.info(
        "Видео к обработке: {} (prefix={!r}, samples_per_video={}, half-frame augmentation=2x)",
        len(videos),
        video_prefix or "",
        samples_per_video,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_parts: list[Path] = []
    bucket_tensors: list[dict[str, torch.Tensor]] = []
    bucket_frames = 0
    part_idx = 1
    scene_logical_id = 0

    for video_path in tqdm(videos, desc="Scene videos", unit="video"):
        scene_payload = _process_scene_video(
            video_path=video_path,
            scene_logical_id=scene_logical_id,
            width=width,
            height=height,
            center_width=center_width,
            samples_per_video=samples_per_video,
        )
        scene_frames = int(scene_payload["frame_id"].shape[0])
        if scene_frames == 0:
            logger.warning("Пропуск пустого видео: {}", video_path.name)
            continue

        logger.info(
            "Сцена {} <- {} (samples={})",
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
                "Сохранена часть {}: samples={}, scenes={}",
                out_path.name,
                int(part_payload["frame_id"].shape[0]),
                int(part_payload["scene_logical_id"].unique().numel()),
            )
            saved_parts.append(out_path)
            part_idx += 1
            bucket_tensors = []
            bucket_frames = 0

    if bucket_tensors:
        part_payload = _concat_scene_tensors(bucket_tensors)
        out_path = _save_dataset_part(
            output_dir=output_dir,
            part_idx=part_idx,
            payload=part_payload,
        )
        logger.info(
            "Сохранена часть {}: samples={}, scenes={}",
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
            "Собрать v4 scene dataset (.sft): source/previous center-half и side-half. "
            "Правые половины отражаются и сохраняются в те же ключи, что и левые."
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
        help="Ширина центрального crop до деления пополам",
    )
    parser.add_argument(
        "--part-size-frames",
        type=int,
        default=512,
        help="Целевой размер одной части датасета в количестве samples после 2x augmentation",
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
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width и --height должны быть > 0")
    if args.center_width <= 0:
        raise SystemExit("--center-width должен быть > 0")

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
    )
    logger.info("Готово: сохранено частей {}", len(saved_parts))


if __name__ == "__main__":
    main()

"""
Чтение видео с последовательным доступом к кадрам.

Используется как базовый ридер: открытие/закрытие, декод, resize, RGB/CHW и
опциональная нормализация кадров для модели.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class SequentialAccessReader:
    def __init__(self, path: str, height: int, width: int, normalize: bool = False) -> None:
        self.path = path
        self.height = int(height)
        self.width = int(width)
        self.normalize = bool(normalize)
        self.current_frame = 0

        self._cap: cv2.VideoCapture | None = None
        self._total_frames: int | None = None

    def __enter__(self) -> "SequentialAccessReader":
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.path}")
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._total_frames = None

    @staticmethod
    def normalize_rgb_chw_uint8_for_model(frames: torch.Tensor) -> torch.Tensor:
        """
        Единая нормализация входов модели: uint8 RGB CHW -> float32 [0..1].
        """
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(f"Ожидался тензор [N, C, H, W], получено: {tuple(frames.shape)}")
        if frames.shape[1] != 3:
            raise ValueError(f"Ожидалось 3 канала (RGB), получено: {frames.shape[1]}")
        if frames.dtype == torch.uint8:
            return frames.float().div_(255.0)
        if torch.is_floating_point(frames):
            min_value = float(frames.min().item())
            max_value = float(frames.max().item())
            if min_value < 0.0 or max_value > 1.0:
                raise ValueError(
                    "Float-данные для модели должны быть в [0..1], "
                    f"получено min={min_value}, max={max_value}"
                )
            return frames if frames.dtype == torch.float32 else frames.float()
        raise ValueError(f"Неподдерживаемый dtype для нормализации: {frames.dtype}")

    def _assert_opened(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Видео не открыто. Используйте контекстный менеджер with.")
        return self._cap

    @property
    def total_frames(self) -> int:
        cap = self._assert_opened()
        if self._total_frames is None:
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._total_frames

    def _empty_batch(self) -> torch.Tensor:
        dtype = torch.float32 if self.normalize else torch.uint8
        return torch.zeros((0, 3, self.height, self.width), dtype=dtype)

    def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height))
        chw = torch.from_numpy(np.transpose(frame, (2, 0, 1)).copy())
        if self.normalize:
            return self.normalize_rgb_chw_uint8_for_model(chw)[0]
        return chw

    def _read_window_from_current(self, window_size: int) -> torch.Tensor:
        if window_size <= 0:
            return self._empty_batch()

        total_frames = self.total_frames
        if total_frames <= 0:
            return self._empty_batch()

        cap = self._assert_opened()
        frames_list: list[torch.Tensor] = []
        for _ in range(window_size):
            ok, frame = cap.read()
            if not ok:
                break
            frames_list.append(self._prepare_frame(frame))
            self.current_frame += 1

        if not frames_list:
            return self._empty_batch()
        return torch.stack(frames_list, dim=0).contiguous()

    def read_window(self, window_size: int) -> torch.Tensor:
        """
        Читает подряд до window_size кадров из текущей позиции декодера.
        """
        return self._read_window_from_current(int(window_size))

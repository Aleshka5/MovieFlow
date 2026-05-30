"""
Чтение видео с произвольным доступом к кадрам.

Нужен веб-аннотатору: пользователь может прыгать по timeline и смотреть
пару соседних кадров в любой точке ролика.
"""

from __future__ import annotations

import cv2
import torch

from app.src.utils.video_reader.sequential_access_reader import SequentialAccessReader


class RandomAccessReader(SequentialAccessReader):
    """
    OpenCV-ридер без кэша и оптимизаций.

    Каждый запрос делает seek + decode — проще отлаживать, достаточно для UI.
    """

    def __init__(self, path: str, height: int, width: int, normalize: bool = False) -> None:
        # Единый размер кадра для UI и последующего пайплайна (get_cuts, модель).
        super().__init__(path=path, height=height, width=width, normalize=normalize)

    def seek(self, frame_id: int) -> None:
        # Перемотка декодера — основа random access (клик по timeline).
        cap = self._assert_opened()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        self.current_frame = int(frame_id)

    def read_window(self, start_frame: int, window_size: int) -> torch.Tensor:
        """
        Читает подряд до window_size кадров, начиная с start_frame.

        Возвращает [T, 3, H, W], uint8, RGB — формат для core и web-слоя.
        """
        if window_size <= 0:
            return self._empty_batch()

        total_frames = self.total_frames
        if total_frames <= 0:
            return self._empty_batch()

        start = max(0, min(int(start_frame), total_frames - 1))
        self.seek(start)
        return self._read_window_from_current(int(window_size))

    def __getitem__(self, index: int) -> torch.Tensor:
        """Один кадр [3, H, W] — для превью на timeline."""
        frame_id = int(index)
        if frame_id < 0 or frame_id >= self.total_frames:
            raise IndexError(f"Кадр {frame_id} вне диапазона [0, {self.total_frames}).")

        window = self.read_window(frame_id, 1)
        if window.shape[0] == 0:
            raise RuntimeError(f"Не удалось прочитать кадр {frame_id}.")
        return window[0].clone()

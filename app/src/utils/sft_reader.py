from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Iterator

import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

_REQUIRED_KEYS = (
    "source_image_center",
    "source_image_sides",
    "previous_frame_center",
    "previous_frame_sides",
    "scene_logical_id",
    "frame_id",
)


def discover_sft_files(dataset_path: str | Path, *, recursive: bool = True) -> list[Path]:
    """Возвращает отсортированный список `.sft` в директории (по умолчанию — рекурсивно по подпапкам)."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Путь к датасету не найден: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"dataset_dir должен быть файлом или директорией: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    files = sorted(
        candidate
        for candidate in iterator
        if candidate.is_file() and candidate.suffix.lower() == ".sft"
    )
    if not files:
        scope = "рекурсивно" if recursive else "в корне папки"
        raise FileNotFoundError(f"Не найдено ни одного .sft файла ({scope}): {path}")
    return files


class SFTSceneDataset(IterableDataset[dict[str, Tensor]]):
    """Итеративный датасет SFT scene-полей.

    Читает по одному `.sft`-файлу и отдает сэмплы без загрузки всех частей набора в RAM.
    Разбиение на train/val/test выполняется стабильно по `scene_logical_id`, чтобы
    одна и та же logical scene всегда попадала в один split.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        recursive: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_dir)
        self.recursive = recursive
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Путь к датасету не найден: {self.dataset_path}")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split должен быть 'train', 'val' или 'test'.")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio должен быть в диапазоне (0, 1).")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio должен быть в диапазоне [0, 1).")
        if self.train_ratio + self.val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio должен быть <= 1.")

    def _resolve_all_files(self) -> list[Path]:
        return discover_sft_files(self.dataset_path, recursive=self.recursive)

    def _resolve_files_for_worker(self) -> list[Path]:
        files = self._resolve_all_files()
        worker = get_worker_info()
        if worker is None:
            return files
        return files[worker.id :: worker.num_workers]

    @staticmethod
    def _load_sft(path: Path) -> dict[str, Any]:
        try:
            from safetensors.torch import load_file

            data = load_file(str(path))
        except Exception:  # noqa: BLE001
            data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"Ожидался dict в {path}, получено: {type(data).__name__}")
        return data

    @staticmethod
    def _as_float_tensor(value: Any, *, path: Path, key: str, dims: tuple[int, ...]) -> Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path.name}:{key} должен быть torch.Tensor, получено: {type(value).__name__}")
        if value.ndim != len(dims):
            raise ValueError(
                f"{path.name}:{key} должен иметь {len(dims)} измерений, "
                f"получено: shape={tuple(value.shape)}"
            )
        if not torch.is_floating_point(value):
            raise ValueError(f"{path.name}:{key} должен быть float tensor, получено: {value.dtype}")
        for axis, expected in enumerate(dims):
            if expected > 0 and value.shape[axis] != expected:
                raise ValueError(
                    f"{path.name}:{key} axis={axis} имеет размер {value.shape[axis]}, ожидается {expected}"
                )
        return value

    @staticmethod
    def _as_int_vector(value: Any, *, path: Path, key: str) -> Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path.name}:{key} должен быть torch.Tensor, получено: {type(value).__name__}")
        if value.ndim != 1:
            raise ValueError(f"{path.name}:{key} должен быть 1D тензором, получено: {tuple(value.shape)}")
        if value.dtype not in {torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8}:
            raise ValueError(f"{path.name}:{key} должен быть int tensor, получено: {value.dtype}")
        return value.to(dtype=torch.int64)

    def _validate_payload(self, payload: dict[str, Any], path: Path) -> dict[str, Tensor]:
        has_legacy_source = "source_frame" in payload
        missing = [k for k in _REQUIRED_KEYS if k not in payload]
        if has_legacy_source:
            missing = [k for k in missing if k not in {"source_image_center", "source_image_sides"}]
        if missing:
            raise KeyError(f"{path.name} не содержит ключи: {missing}; есть: {sorted(payload.keys())}")

        if "source_image_center" in payload and "source_image_sides" in payload:
            source_center = self._as_float_tensor(
                payload["source_image_center"],
                path=path,
                key="source_image_center",
                dims=(-1, -1, -1, -1),
            )
            source_sides = self._as_float_tensor(
                payload["source_image_sides"],
                path=path,
                key="source_image_sides",
                dims=(source_center.shape[0], source_center.shape[1], source_center.shape[2], -1),
            )
        else:
            source_frame = self._as_float_tensor(
                payload["source_frame"],
                path=path,
                key="source_frame",
                dims=(-1, -1, -1, -1),
            )
            center_width = int(payload["source_image_center"].shape[-1])
            source_center = source_frame[..., (source_frame.shape[-1] - center_width) // 2 : (source_frame.shape[-1] + center_width) // 2]
            source_sides = torch.cat(
                (
                    source_frame[..., : (source_frame.shape[-1] - center_width) // 2],
                    source_frame[..., (source_frame.shape[-1] + center_width) // 2 :],
                ),
                dim=-1,
            )
        n_samples = source_center.shape[0]
        c = source_center.shape[1]
        h = source_center.shape[2]

        output: dict[str, Tensor] = {
            "source_image_center": source_center,
            "source_image_sides": source_sides,
            "previous_frame_center": self._as_float_tensor(
                payload["previous_frame_center"],
                path=path,
                key="previous_frame_center",
                dims=(n_samples, c, h, -1),
            ),
            "previous_frame_sides": self._as_float_tensor(
                payload["previous_frame_sides"],
                path=path,
                key="previous_frame_sides",
                dims=(n_samples, c, h, -1),
            ),
            "scene_logical_id": self._as_int_vector(payload["scene_logical_id"], path=path, key="scene_logical_id"),
            "frame_id": self._as_int_vector(payload["frame_id"], path=path, key="frame_id"),
        }

        for key, tensor in output.items():
            if tensor.shape[0] != n_samples:
                raise ValueError(
                    f"{path.name}:{key} имеет другой размер batch: {tensor.shape[0]} != {n_samples}"
                )
        return output

    def _scene_bucket(self, scene_id: int) -> float:
        scene_key = f"{scene_id}:{self.split_seed}"
        crc = zlib.crc32(scene_key.encode("utf-8")) & 0xFFFFFFFF
        return crc / 0xFFFFFFFF

    def _resolve_split_indices(self, scene_ids: Tensor) -> list[int]:
        selected: list[int] = []
        for index in range(scene_ids.shape[0]):
            bucket = self._scene_bucket(int(scene_ids[index].item()))
            is_train = bucket < self.train_ratio
            is_val = self.train_ratio <= bucket < (self.train_ratio + self.val_ratio)
            if self.split == "train" and is_train:
                selected.append(index)
            elif self.split == "val" and is_val:
                selected.append(index)
            elif self.split == "test" and (not is_train and not is_val):
                selected.append(index)
        return selected

    def count_samples(self) -> int:
        total = 0
        for file_path in self._resolve_all_files():
            payload = self._load_sft(file_path)
            data = self._validate_payload(payload, file_path)
            total += len(self._resolve_split_indices(data["scene_logical_id"]))
        return total

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        files = self._resolve_files_for_worker()
        worker = get_worker_info()
        worker_label = f"worker={worker.id}" if worker is not None else "worker=main"
        total_files = len(files)

        for file_index, file_path in enumerate(files, start=1):
            payload = self._load_sft(file_path)
            data = self._validate_payload(payload, file_path)
            split_indices = self._resolve_split_indices(data["scene_logical_id"])

            logger.info(
                "[sft:{}] {} loading {}/{}: {} (all={}, split={})",
                self.split,
                worker_label,
                file_index,
                total_files,
                file_path.name,
                int(data["source_image_center"].shape[0]),
                len(split_indices),
            )
            for sample_index in split_indices:
                yield {k: v[sample_index] for k, v in data.items()}


def create_sft_scene_dataloader(
    dataset_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    split: str = "train",
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> DataLoader[dict[str, Tensor]]:
    dataset = SFTSceneDataset(
        dataset_dir=dataset_dir,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

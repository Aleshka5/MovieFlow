from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from app.src.utils.sft_reader import SFTSceneDataset


class DiTSFTDataset(IterableDataset[dict[str, Tensor]]):
    """Baseline DiT: condition=source_image_center, target=source_image_sides (noise — на train-шаге)."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        condition_key: str,
        target_key: str,
        previous_sides_key: str | None,
        latent_channels: int,
        condition_height: int,
        condition_width: int,
        target_height: int,
        target_width: int,
        split: str = "train",
        train_ratio: float = 0.9,
        split_seed: int = 42,
        recursive: bool = True,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
    ) -> None:
        super().__init__()
        self._scene_dataset = SFTSceneDataset(
            dataset_dir=dataset_dir,
            split=split,
            train_ratio=train_ratio,
            val_ratio=1.0 - train_ratio,
            split_seed=split_seed,
            recursive=recursive,
        )
        self.condition_key = condition_key
        self.target_key = target_key
        self.previous_sides_key = previous_sides_key
        self.latent_channels = latent_channels
        self.condition_height = condition_height
        self.condition_width = condition_width
        self.target_height = target_height
        self.target_width = target_width
        self.distributed_rank = int(distributed_rank)
        self.distributed_world_size = int(distributed_world_size)
        if self.distributed_world_size < 1:
            raise ValueError("distributed_world_size должен быть >= 1.")
        if not 0 <= self.distributed_rank < self.distributed_world_size:
            raise ValueError(
                "distributed_rank должен быть в диапазоне [0, distributed_world_size). "
                f"Получено rank={self.distributed_rank}, world_size={self.distributed_world_size}."
            )

    @staticmethod
    def _load_sft(path: Path) -> dict[str, Any]:
        return SFTSceneDataset._load_sft(path)

    @staticmethod
    def _to_channels_first(tensor: Tensor, *, latent_channels: int, source: str) -> Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"{source}: ожидается [B, C, H, W] или [B, H, W, C], получено {tuple(tensor.shape)}")
        if tensor.shape[1] == latent_channels:
            return tensor
        if tensor.shape[-1] == latent_channels:
            return tensor.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"{source}: не удалось определить ось каналов для C={latent_channels}, shape={tuple(tensor.shape)}"
        )

    def _read_tensor(self, payload: dict[str, Any], key: str, path: Path) -> Tensor:
        if key not in payload:
            raise KeyError(
                f"{path.name}: отсутствует ключ {key!r}. "
                f"Доступные ключи: {sorted(payload.keys())}. "
                "Для DiT baseline нужны ключи source_image_center и source_image_sides (см. README)."
            )
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path.name}:{key} должен быть torch.Tensor, получено: {type(value).__name__}")
        if not torch.is_floating_point(value):
            value = value.to(torch.float32)
        return value

    def _validate_sample(self, condition: Tensor, target: Tensor, path: Path) -> tuple[Tensor, Tensor]:
        condition = self._to_channels_first(
            condition.unsqueeze(0) if condition.ndim == 3 else condition,
            latent_channels=self.latent_channels,
            source=f"{path.name}:{self.condition_key}",
        )[0]
        target = self._to_channels_first(
            target.unsqueeze(0) if target.ndim == 3 else target,
            latent_channels=self.latent_channels,
            source=f"{path.name}:{self.target_key}",
        )[0]
        if condition.shape != (self.latent_channels, self.condition_height, self.condition_width):
            raise ValueError(
                f"{path.name}:{self.condition_key} shape={tuple(condition.shape)}, "
                f"ожидается ({self.latent_channels}, {self.condition_height}, {self.condition_width})"
            )
        if target.shape != (self.latent_channels, self.target_height, self.target_width):
            raise ValueError(
                f"{path.name}:{self.target_key} shape={tuple(target.shape)}, "
                f"ожидается ({self.latent_channels}, {self.target_height}, {self.target_width})"
            )
        return condition, target

    def _validate_previous_sides(self, previous_sides: Tensor, path: Path) -> Tensor:
        previous_sides = self._to_channels_first(
            previous_sides.unsqueeze(0) if previous_sides.ndim == 3 else previous_sides,
            latent_channels=self.latent_channels,
            source=f"{path.name}:{self.previous_sides_key}",
        )[0]
        if previous_sides.shape != (self.latent_channels, self.target_height, self.target_width):
            raise ValueError(
                f"{path.name}:{self.previous_sides_key} shape={tuple(previous_sides.shape)}, "
                f"ожидается ({self.latent_channels}, {self.target_height}, {self.target_width})"
            )
        return previous_sides

    def __iter__(self):
        files = self._scene_dataset._resolve_files_for_worker()
        distributed_sample_index = 0
        for file_path in files:
            payload = self._load_sft(file_path)
            condition_all = self._read_tensor(payload, self.condition_key, file_path)
            target_all = self._read_tensor(payload, self.target_key, file_path)
            previous_sides_all = None
            if self.previous_sides_key is not None:
                previous_sides_all = self._read_tensor(payload, self.previous_sides_key, file_path)
            if condition_all.ndim == 3:
                condition_all = condition_all.unsqueeze(0)
            if target_all.ndim == 3:
                target_all = target_all.unsqueeze(0)
            if previous_sides_all is not None and previous_sides_all.ndim == 3:
                previous_sides_all = previous_sides_all.unsqueeze(0)
            if condition_all.shape[0] != target_all.shape[0]:
                raise ValueError(
                    f"{file_path.name}: batch size mismatch "
                    f"{self.condition_key}={condition_all.shape[0]} vs "
                    f"{self.target_key}={target_all.shape[0]}"
                )
            if previous_sides_all is not None and condition_all.shape[0] != previous_sides_all.shape[0]:
                raise ValueError(
                    f"{file_path.name}: batch size mismatch "
                    f"{self.condition_key}={condition_all.shape[0]} vs "
                    f"{self.previous_sides_key}={previous_sides_all.shape[0]}"
                )

            scene_ids = payload.get("scene_logical_id")
            if scene_ids is None:
                split_indices = list(range(condition_all.shape[0]))
            else:
                if not isinstance(scene_ids, torch.Tensor):
                    raise TypeError(f"{file_path.name}: scene_logical_id должен быть torch.Tensor")
                split_indices = self._scene_dataset._resolve_split_indices(scene_ids)

            for sample_index in split_indices:
                if self.distributed_world_size > 1:
                    should_take = (
                        distributed_sample_index % self.distributed_world_size
                        == self.distributed_rank
                    )
                    distributed_sample_index += 1
                    if not should_take:
                        continue
                condition, target = self._validate_sample(
                    condition_all[sample_index],
                    target_all[sample_index],
                    file_path,
                )
                sample = {"condition": condition, "target": target}
                if previous_sides_all is not None:
                    sample["previous_sides"] = self._validate_previous_sides(
                        previous_sides_all[sample_index],
                        file_path,
                    )
                yield sample


def create_sft_dataloader(
    *,
    dataset_dir: str | Path,
    condition_key: str,
    target_key: str,
    previous_sides_key: str | None,
    batch_size: int,
    num_workers: int,
    latent_channels: int,
    condition_height: int,
    condition_width: int,
    target_height: int,
    target_width: int,
    pin_memory: bool = False,
    split: str = "train",
    train_ratio: float = 0.9,
    split_seed: int = 42,
    recursive: bool = True,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    drop_last: bool = False,
) -> DataLoader[dict[str, Tensor]]:
    dataset = DiTSFTDataset(
        dataset_dir=dataset_dir,
        condition_key=condition_key,
        target_key=target_key,
        previous_sides_key=previous_sides_key,
        latent_channels=latent_channels,
        condition_height=condition_height,
        condition_width=condition_width,
        target_height=target_height,
        target_width=target_width,
        split=split,
        train_ratio=train_ratio,
        split_seed=split_seed,
        recursive=recursive,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )

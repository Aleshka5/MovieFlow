"""Генерирует минимальный encoded SFT для smoke-test DiT (не для продакшн-обучения)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    settings = get_settings()
    output_dir = args.output_dir or (settings.dataset_download_dir / "synthetic_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.samples
    payload = {
        settings.condition_key: torch.randn(
            n, settings.latent_channels, settings.condition_height, settings.condition_width
        ),
        settings.target_key: torch.randn(
            n, settings.latent_channels, settings.query_height, settings.query_width
        ),
        "scene_logical_id": torch.arange(n, dtype=torch.int32) % 8,
        "frame_id": torch.arange(n, dtype=torch.int32),
    }
    out_path = output_dir / "scene_dataset_part_00001.sft"
    save_file(payload, str(out_path))
    print(f"Wrote {out_path} with {n} samples")


if __name__ == "__main__":
    main()

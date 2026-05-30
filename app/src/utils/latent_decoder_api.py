from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
import torch


@dataclass(slots=True)
class LatentDecoderAPIClient:
    base_url: str
    timeout_sec: float = 120.0

    def readiness(self) -> requests.Response:
        return requests.get(f"{self.base_url.rstrip('/')}/readiness", timeout=self.timeout_sec)

    def decode_latents(self, latents: list[Any]) -> list[np.ndarray]:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/decode",
            json={"latents": latents},
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return _extract_rgb_frames(payload)

    def decode_single_latent(self, latent: Any) -> np.ndarray:
        frames = self.decode_latents([latent])
        if len(frames) != 1:
            raise ValueError(f"Expected exactly 1 decoded frame, got {len(frames)}")
        return frames[0]

    def decode_tensor_batch(self, latents_batch: torch.Tensor) -> np.ndarray:
        if latents_batch.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] latents, got shape={tuple(latents_batch.shape)}")
        latents = latents_batch.detach().cpu().to(torch.float32).tolist()
        frames = [self.decode_single_latent(latent) for latent in latents]
        return np.stack(frames, axis=0)

    def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("Expected non-empty list of RGB frames for encoding.")
        payload_frames: list[list[list[list[int]]]] = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"Frame must have shape [H, W, 3], got {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            payload_frames.append(arr.tolist())
        response = requests.post(
            f"{self.base_url.rstrip('/')}/encode",
            json={"frames": payload_frames},
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return _extract_latent_batch(payload)

    def encode_single_frame(self, frame: np.ndarray) -> np.ndarray:
        latents = self.encode_frames([frame])
        if latents.shape[0] != 1:
            raise ValueError(f"Expected exactly 1 encoded latent, got {latents.shape[0]}")
        return latents[0]

    def encode_tensor_batch(self, frames_batch: torch.Tensor) -> torch.Tensor:
        if frames_batch.ndim != 4 or frames_batch.shape[1] != 3:
            raise ValueError(
                f"Expected RGB batch [B, 3, H, W] for encoding, got shape={tuple(frames_batch.shape)}"
            )
        batch = frames_batch.detach().cpu()
        if torch.is_floating_point(batch):
            min_value = float(batch.min().item())
            max_value = float(batch.max().item())
            if min_value < 0.0 or max_value > 1.0:
                raise ValueError(
                    f"Float RGB batch for encoding must be in [0..1], got min={min_value}, max={max_value}"
                )
            batch = (batch * 255.0).round().clamp(0, 255).to(torch.uint8)
        elif batch.dtype != torch.uint8:
            raise ValueError(
                f"Unsupported dtype for RGB encoding batch: {batch.dtype}. Expected uint8 or float."
            )
        frames_hwc = batch.permute(0, 2, 3, 1).contiguous().numpy()
        encoded = self.encode_frames([frame for frame in frames_hwc])
        return torch.from_numpy(encoded).to(torch.float32)


def _extract_rgb_frames(payload: dict[str, Any]) -> list[np.ndarray]:
    candidate_keys = ("frames", "images", "rgb_frames", "decoded_frames")
    raw_frames = None
    for key in candidate_keys:
        if key in payload:
            raw_frames = payload[key]
            break
    if raw_frames is None:
        raise KeyError(f"Decoder response does not contain known frame keys: {candidate_keys}")
    if not isinstance(raw_frames, list):
        raise TypeError(f"Decoder frames payload must be a list, got: {type(raw_frames).__name__}")

    frames: list[np.ndarray] = []
    for item in raw_frames:
        arr = np.asarray(item)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Decoded frame must have shape [H, W, 3], got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.append(arr)
    return frames


def _extract_latent_batch(payload: dict[str, Any]) -> np.ndarray:
    candidate_keys = ("latents", "encoded_latents", "features")
    raw_latents = None
    for key in candidate_keys:
        if key in payload:
            raw_latents = payload[key]
            break
    if raw_latents is None:
        raise KeyError(f"Encode response does not contain known latent keys: {candidate_keys}")
    arr = np.asarray(raw_latents, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Encoded latents must have shape [B, C, H, W], got {arr.shape}")
    return arr

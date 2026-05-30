from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def crop_center_width(frame: Tensor, center_width: int) -> Tensor:
    full_width = frame.shape[-1]
    if center_width > full_width:
        raise ValueError(f"center_width={center_width} exceeds frame width={full_width}")
    if center_width == full_width:
        return frame
    start = (full_width - center_width) // 2
    end = start + center_width
    return frame[..., start:end]


def extract_source_sides(source_frame: Tensor, center_width: int) -> Tensor:
    full_width = source_frame.shape[-1]
    side_width_total = full_width - center_width
    if side_width_total <= 0 or side_width_total % 2 != 0:
        raise ValueError(
            f"Cannot derive sides from source_frame: full_width={full_width}, center_width={center_width}"
        )
    side_width = side_width_total // 2
    left = source_frame[..., :side_width]
    right = source_frame[..., full_width - side_width :]
    return torch.cat((left, right), dim=-1)


def build_scene_model_batch(
    batch: dict[str, Tensor],
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Собирает вход модели и target из batch SFT scene-dataset.

    Target — боковые полосы текущего кадра (`source_image_sides`).
    В high_res добавляется центр предыдущего кадра (`previous_frame_center`).
    """
    source_center = batch.get("source_image_center")
    source_sides = batch.get("source_image_sides")
    if source_center is None or source_sides is None:
        source_frame = batch["source_frame"]
        if "previous_frame_center" in batch:
            center_width = int(batch["previous_frame_center"].shape[-1])
        elif "source_image_center" in batch:
            center_width = int(batch["source_image_center"].shape[-1])
        else:
            raise KeyError("batch must contain source_image_center/sides or source_frame with previous_frame_center")
        source_center = crop_center_width(source_frame, center_width=center_width)
        source_sides = extract_source_sides(source_frame, center_width=center_width)
    else:
        center_width = int(source_center.shape[-1])

    previous_center = batch["previous_frame_center"]

    high_res = torch.cat(
        (
            source_center,
            previous_center,
            batch["opti_map_1"],
        ),
        dim=1,
    )
    low_res = torch.cat((batch["opti_map_2"], batch["depth_map"]), dim=1)
    camera = batch["camera_move_vector"]
    target = source_sides

    expected_high_hw = (target.shape[-2], center_width)
    expected_low_hw = (target.shape[-2] // 2, center_width // 2)
    if high_res.shape[-2:] != expected_high_hw:
        high_res = F.interpolate(high_res, size=expected_high_hw, mode="bilinear", align_corners=False)
    if low_res.shape[-2:] != expected_low_hw:
        low_res = F.interpolate(low_res, size=expected_low_hw, mode="bilinear", align_corners=False)

    model_input = {
        "high_res": high_res,
        "side": source_sides,
        "low_res": low_res,
        "camera": camera,
    }
    return model_input, target, source_sides

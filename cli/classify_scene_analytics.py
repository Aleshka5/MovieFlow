from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


def _safe_list(payload: dict[str, Any], key: str) -> list[float]:
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for value in raw:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def _safe_list_vec2(payload: dict[str, Any], key: str) -> list[tuple[float, float]]:
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            continue
        x, y = item
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            out.append((float(x), float(y)))
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _quantile(values: list[float], q: float, default: float) -> float:
    if not values:
        return default
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * q
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return float(sorted_values[low])
    alpha = idx - low
    return float(sorted_values[low] * (1.0 - alpha) + sorted_values[high] * alpha)


def _scene_metrics(scene_payload: dict[str, Any]) -> dict[str, float]:
    camera_speed = _safe_list(scene_payload, "camera_speed")
    camera_track = _safe_list_vec2(scene_payload, "camera_move_track")
    depth_mean = _safe_list(scene_payload, "depth_mean")
    depth_min = _safe_list(scene_payload, "depth_min")
    depth_max = _safe_list(scene_payload, "depth_max")
    brightness = _safe_list(scene_payload, "brightness_mean")
    diff_mean = _safe_list(scene_payload, "diff_mean")
    diff_noise = _safe_list(scene_payload, "diff_noise_ratio")
    diff_structure = _safe_list(scene_payload, "diff_structure_ratio")
    flow_mag = _safe_list(scene_payload, "optiflow_mean_magnitude")
    flow_edge_l = _safe_list(scene_payload, "optiflow_edge_mean_left")
    flow_edge_r = _safe_list(scene_payload, "optiflow_edge_mean_right")
    flow_coherence = _safe_list(scene_payload, "optiflow_coherence")

    depth_range = [
        max(0.0, mx - mn) for mx, mn in zip(depth_max, depth_min, strict=False)
    ] if depth_max and depth_min else []
    flow_edge = [0.5 * (l + r) for l, r in zip(flow_edge_l, flow_edge_r, strict=False)] if flow_edge_l and flow_edge_r else []

    if camera_track:
        end_x, end_y = camera_track[-1]
    else:
        end_x, end_y = 0.0, 0.0

    flow_mag_mean = _mean(flow_mag)
    flow_edge_mean = _mean(flow_edge)
    flow_edge_ratio = float(flow_edge_mean / (flow_mag_mean + 1e-6))

    return {
        "frame_count": float(int(scene_payload.get("frame_count", 0))),
        "camera_speed_mean": _mean(camera_speed),
        "camera_speed_std": _std(camera_speed),
        "camera_track_dx": float(end_x),
        "camera_track_dy": float(end_y),
        "camera_track_displacement": float(math.hypot(end_x, end_y)),
        "depth_mean_global": _mean(depth_mean),
        "depth_mean_std": _std(depth_mean),
        "depth_range_mean": _mean(depth_range),
        "brightness_mean_global": _mean(brightness),
        "brightness_std": _std(brightness),
        "diff_mean_global": _mean(diff_mean),
        "diff_noise_global": _mean(diff_noise),
        "diff_structure_global": _mean(diff_structure),
        "flow_mean_magnitude_global": flow_mag_mean,
        "flow_edge_mean_global": flow_edge_mean,
        "flow_edge_ratio": flow_edge_ratio,
        "flow_coherence_global": _mean(flow_coherence),
    }


def _build_thresholds(scene_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    def pick(metric: str, q: float, default: float) -> float:
        values = [m[metric] for m in scene_metrics.values()]
        return _quantile(values, q=q, default=default)

    return {
        "camera_active": pick("camera_speed_mean", 0.55, 0.08),
        "foreground_depth": pick("depth_mean_global", 0.5, 0.5),
        "depth_dynamic": pick("depth_mean_std", 0.55, 0.03),
        "bright_scene": pick("brightness_mean_global", 0.5, 96.0),
        "diff_near_zero": pick("diff_mean_global", 0.25, 0.01),
        "diff_noisy": pick("diff_noise_global", 0.65, 0.9),
        "diff_structured": pick("diff_structure_global", 0.5, 0.9),
        "edge_motion": pick("flow_edge_mean_global", 0.55, 0.1),
        "edge_motion_ratio": pick("flow_edge_ratio", 0.55, 0.8),
        "active_pixel_motion": pick("flow_mean_magnitude_global", 0.55, 0.15),
        "coherent_motion": pick("flow_coherence_global", 0.6, 0.25),
        "high_depth_contrast": pick("depth_range_mean", 0.55, 0.2),
        "dynamic_lighting": pick("brightness_std", 0.55, 8.0),
    }


def _classify_scene(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, str]:
    classes: dict[str, str] = {}

    classes["camera_motion"] = (
        "camera_active"
        if metrics["camera_speed_mean"] >= thresholds["camera_active"]
        else "camera_almost_static"
    )
    classes["depth_layout"] = (
        "foreground_dominant"
        if metrics["depth_mean_global"] >= thresholds["foreground_depth"]
        else "background_dominant"
    )
    classes["depth_dynamics"] = (
        "depth_dynamic"
        if metrics["depth_mean_std"] >= thresholds["depth_dynamic"]
        else "depth_stable"
    )
    classes["brightness"] = (
        "bright_scene"
        if metrics["brightness_mean_global"] >= thresholds["bright_scene"]
        else "dark_scene"
    )

    if metrics["diff_mean_global"] < thresholds["diff_near_zero"]:
        classes["diff_profile"] = "diff_near_zero"
    elif (
        metrics["diff_noise_global"] >= thresholds["diff_noisy"]
        and metrics["diff_structure_global"] < thresholds["diff_structured"]
    ):
        classes["diff_profile"] = "diff_noisy"
    else:
        classes["diff_profile"] = "diff_structured_changes"

    classes["edge_motion"] = (
        "motion_on_edges"
        if metrics["flow_edge_mean_global"] >= thresholds["edge_motion"]
        and metrics["flow_edge_ratio"] >= thresholds["edge_motion_ratio"]
        else "no_edge_motion"
    )
    classes["pixel_motion_intensity"] = (
        "active_pixel_motion"
        if metrics["flow_mean_magnitude_global"] >= thresholds["active_pixel_motion"]
        else "weak_pixel_motion"
    )

    # Дополнительные классы для оценки разнообразия.
    classes["motion_type"] = (
        "coherent_global_motion"
        if metrics["flow_coherence_global"] >= thresholds["coherent_motion"]
        else "chaotic_local_motion"
    )
    classes["depth_contrast"] = (
        "high_depth_contrast"
        if metrics["depth_range_mean"] >= thresholds["high_depth_contrast"]
        else "flat_depth_layout"
    )
    classes["lighting_dynamics"] = (
        "dynamic_lighting"
        if metrics["brightness_std"] >= thresholds["dynamic_lighting"]
        else "stable_lighting"
    )

    dx = abs(metrics["camera_track_dx"])
    dy = abs(metrics["camera_track_dy"])
    if dx <= 1e-6 and dy <= 1e-6:
        classes["camera_drift_direction"] = "no_drift"
    elif dx > dy * 1.3:
        classes["camera_drift_direction"] = "horizontal_drift"
    elif dy > dx * 1.3:
        classes["camera_drift_direction"] = "vertical_drift"
    else:
        classes["camera_drift_direction"] = "mixed_drift"

    return classes


def _save_class_distribution_plots(
    *,
    class_distribution: dict[str, Counter[str]],
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    saved_files: list[str] = []
    for family, counter in class_distribution.items():
        labels = list(counter.keys())
        values = [counter[label] for label in labels]
        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.8))
        bars = ax.bar(labels, values)
        ax.set_title(f"Class distribution: {family}")
        ax.set_ylabel("Scenes")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                bar.get_height() + 0.1,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        file_name = f"class_distribution_{family}.png"
        plot_path = output_dir / file_name
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        saved_files.append(file_name)
    return saved_files


def _save_overview_metric_plots(
    *,
    scene_classes: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    scene_names = list(scene_classes.keys())
    if not scene_names:
        return []

    camera_speed = [
        float(scene_classes[name]["metrics"]["camera_speed_mean"]) for name in scene_names
    ]
    flow_mag = [
        float(scene_classes[name]["metrics"]["flow_mean_magnitude_global"]) for name in scene_names
    ]
    brightness = [
        float(scene_classes[name]["metrics"]["brightness_mean_global"]) for name in scene_names
    ]
    depth_mean = [
        float(scene_classes[name]["metrics"]["depth_mean_global"]) for name in scene_names
    ]
    depth_std = [
        float(scene_classes[name]["metrics"]["depth_mean_std"]) for name in scene_names
    ]
    diff_noise = [
        float(scene_classes[name]["metrics"]["diff_noise_global"]) for name in scene_names
    ]

    saved_files: list[str] = []

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    scatter = ax.scatter(camera_speed, flow_mag, c=brightness, cmap="viridis", alpha=0.8)
    ax.set_title("Camera speed vs pixel flow intensity")
    ax.set_xlabel("camera_speed_mean")
    ax.set_ylabel("flow_mean_magnitude_global")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("brightness_mean_global")
    fig.tight_layout()
    file_name = "overview_camera_vs_flow.png"
    fig.savefig(output_dir / file_name, dpi=140)
    plt.close(fig)
    saved_files.append(file_name)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    scatter = ax.scatter(depth_mean, depth_std, c=diff_noise, cmap="plasma", alpha=0.8)
    ax.set_title("Depth layout vs depth dynamics")
    ax.set_xlabel("depth_mean_global")
    ax.set_ylabel("depth_mean_std")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("diff_noise_global")
    fig.tight_layout()
    file_name = "overview_depth_layout_vs_dynamics.png"
    fig.savefig(output_dir / file_name, dpi=140)
    plt.close(fig)
    saved_files.append(file_name)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.hist(brightness, bins=24)
    ax.set_title("Brightness distribution")
    ax.set_xlabel("brightness_mean_global")
    ax.set_ylabel("Scenes")
    fig.tight_layout()
    file_name = "overview_brightness_histogram.png"
    fig.savefig(output_dir / file_name, dpi=140)
    plt.close(fig)
    saved_files.append(file_name)

    return saved_files


def _build_visualizations(
    *,
    scene_classes: dict[str, dict[str, Any]],
    class_distribution: dict[str, Counter[str]],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    saved_files.extend(
        _save_class_distribution_plots(
            class_distribution=class_distribution,
            output_dir=output_dir,
        )
    )
    saved_files.extend(
        _save_overview_metric_plots(
            scene_classes=scene_classes,
            output_dir=output_dir,
        )
    )
    return saved_files


def run_classify_scene_analytics(
    *,
    analytics_json_path: Path,
    output_path: Path,
    viz_dir: Path | None,
) -> Path:
    if not analytics_json_path.is_file():
        raise SystemExit(f"Файл аналитики не найден: {analytics_json_path}")

    payload = json.loads(analytics_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Ожидался JSON-объект вида {video_name: {params...}}")
    if not payload:
        raise SystemExit("Входной JSON пуст, классифицировать нечего")

    scene_metrics: dict[str, dict[str, float]] = {}
    for scene_name, scene_payload in payload.items():
        if not isinstance(scene_payload, dict):
            continue
        scene_metrics[scene_name] = _scene_metrics(scene_payload)

    if not scene_metrics:
        raise SystemExit("Не удалось извлечь ни одной валидной сцены из JSON")

    thresholds = _build_thresholds(scene_metrics)

    scene_classes: dict[str, dict[str, Any]] = {}
    class_distribution: dict[str, Counter[str]] = defaultdict(Counter)

    for scene_name, metrics in scene_metrics.items():
        classes = _classify_scene(metrics, thresholds)
        scene_classes[scene_name] = {"metrics": metrics, "classes": classes}
        for family, label in classes.items():
            class_distribution[family][label] += 1

    distribution_json: dict[str, dict[str, int]] = {
        family: dict(counter) for family, counter in class_distribution.items()
    }
    visualization_files: list[str] = []
    if viz_dir is not None:
        try:
            visualization_files = _build_visualizations(
                scene_classes=scene_classes,
                class_distribution=class_distribution,
                output_dir=viz_dir,
            )
            logger.info("Сохранено графиков: {} -> {}", len(visualization_files), viz_dir)
        except Exception as exc:
            logger.warning("Не удалось построить визуализации: {}", exc)

    result = {
        "source_file": analytics_json_path.as_posix(),
        "scene_count": len(scene_classes),
        "thresholds": thresholds,
        "visualization_dir": viz_dir.as_posix() if viz_dir is not None else None,
        "visualization_files": visualization_files,
        "classes_description": {
            "camera_motion": "camera_active / camera_almost_static",
            "depth_layout": "foreground_dominant / background_dominant",
            "depth_dynamics": "depth_dynamic / depth_stable",
            "brightness": "bright_scene / dark_scene",
            "diff_profile": "diff_near_zero / diff_noisy / diff_structured_changes",
            "edge_motion": "motion_on_edges / no_edge_motion",
            "pixel_motion_intensity": "active_pixel_motion / weak_pixel_motion",
            "motion_type": "coherent_global_motion / chaotic_local_motion",
            "depth_contrast": "high_depth_contrast / flat_depth_layout",
            "lighting_dynamics": "dynamic_lighting / stable_lighting",
            "camera_drift_direction": "no_drift / horizontal_drift / vertical_drift / mixed_drift",
        },
        "distribution": distribution_json,
        "scenes": scene_classes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


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
            "Классифицировать сцены по сохраненной JSON-аналитике "
            "(camera/depth/brightness/diff/flow) для оценки разнообразия датасета."
        )
    )
    parser.add_argument(
        "--analytics-json",
        type=Path,
        required=True,
        help="Путь к JSON, который создан build_scene_analytics.py",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Путь к выходному JSON с классами по сценам и распределением",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Опциональная папка для PNG-визуализаций классов и метрик",
    )
    args = parser.parse_args()

    output = run_classify_scene_analytics(
        analytics_json_path=args.analytics_json,
        output_path=args.output_path,
        viz_dir=args.viz_dir,
    )
    logger.info("Готово: {}", output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

FLOWS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "fast": {
        # unsharp: размер матрицы должен быть нечётным (3, 5, 7, 9, ...), 10x10 недопустим.
        "vf": "atadenoise,hqdn3d=4,unsharp=9:9:0.5",
        "suffix": "fast_denoise",
        "label": "Быстрый: atadenoise+hqdn3d+unsharp",
        "required_filters": ("atadenoise", "hqdn3d", "unsharp"),
    },
    "strong_artifacts": {
        # deblock=1:1 из старых гайдов: в ffmpeg 6+ позиционные аргументы — block (4..512),
        # поэтому используем именованные: strong + block=16 (типичный размер макроблока H.264).
        "vf": "deblock=filter=strong:block=32,hqdn3d=1.5:1.5:6:6",
        "suffix": "strong_artifacts_denoise",
        "label": "Сильные артефакты: deblock+hqdn3d",
        "required_filters": ("deblock", "hqdn3d"),
    },
}


def _run_command(command: list[str], *, dry_run: bool) -> None:
    pretty = " ".join(shlex.quote(part) for part in command)
    logging.info("RUN: %s", pretty)
    if dry_run:
        return
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-40:])
        stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-20:])
        message = (
            f"Команда завершилась с кодом {completed.returncode}.\n"
            f"STDERR (tail):\n{stderr_tail or '<empty>'}\n"
            f"STDOUT (tail):\n{stdout_tail or '<empty>'}"
        )
        raise RuntimeError(message)


def _ffmpeg_has_filter(ffmpeg_bin: str, filter_name: str) -> bool:
    completed = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    lines = completed.stdout.splitlines()
    token = f" {filter_name} "
    return any(token in line for line in lines)


def _validate_filters(ffmpeg_bin: str, required_filters: tuple[str, ...]) -> None:
    missing = sorted(
        filter_name
        for filter_name in required_filters
        if not _ffmpeg_has_filter(ffmpeg_bin, filter_name)
    )
    if missing:
        missing_joined = ", ".join(missing)
        raise SystemExit(
            "В вашей сборке ffmpeg отсутствуют фильтры: "
            f"{missing_joined}. Проверьте `ffmpeg -filters` или используйте сборку с ними."
        )


def _build_encode_args(use_nvenc: bool) -> list[str]:
    if use_nvenc:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-cq",
            "19",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


def _render_video(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    vf_expr: str,
    use_nvenc: bool,
    dry_run: bool,
) -> None:
    # Фильтры (atadenoise/hqdn3d/unsharp/deblock) работают на CPU.
    # CUDA здесь = только NVENC-кодирование, без -hwaccel cuda (иначе ложные CUDA-fallback).
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        vf_expr,
        "-an",
        *_build_encode_args(use_nvenc),
        str(output_path),
    ]
    _run_command(command, dry_run=dry_run)


def _is_nvenc_failure(error: RuntimeError) -> bool:
    text = str(error).lower()
    nvenc_markers = (
        "h264_nvenc",
        "nvenc",
        "encoder",
        "cuda",
        "no capable devices",
        "out of memory",
    )
    return any(marker in text for marker in nvenc_markers)


def _render_with_fallback(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    vf_expr: str,
    dry_run: bool,
) -> bool:
    try:
        _render_video(
            ffmpeg_bin=ffmpeg_bin,
            input_path=input_path,
            output_path=output_path,
            vf_expr=vf_expr,
            use_nvenc=True,
            dry_run=dry_run,
        )
        return True
    except RuntimeError as error:
        if not _is_nvenc_failure(error):
            logging.error(
                "Ошибка фильтров/пайплайна для %s (NVENC/CPU fallback не поможет):\n%s",
                output_path.name,
                error,
            )
            raise

        logging.warning(
            "NVENC не сработал для %s. Пробую CPU-кодирование (libx264).\n%s",
            output_path.name,
            error,
        )
        _render_video(
            ffmpeg_bin=ffmpeg_bin,
            input_path=input_path,
            output_path=output_path,
            vf_expr=vf_expr,
            use_nvenc=False,
            dry_run=dry_run,
        )
        return False


def _collect_input_videos(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Путь не найден: {input_path}")

    iterator = input_path.rglob("*.mp4") if recursive else input_path.glob("*.mp4")
    videos = sorted(path.resolve() for path in iterator if path.is_file())
    if not videos:
        scope = "рекурсивно" if recursive else "в корне папки"
        raise SystemExit(f"Не найдено .mp4 файлов {scope}: {input_path}")
    return videos


def _build_output_path(
    *,
    source_video: Path,
    input_root: Path,
    output_arg: Path | None,
    flow_suffix: str,
    is_dir_mode: bool,
) -> Path:
    if not is_dir_mode:
        if output_arg is not None:
            return output_arg.resolve()
        return source_video.with_name(f"{source_video.stem}_{flow_suffix}.mp4")

    output_root = (
        output_arg.resolve() if output_arg is not None else input_root / f"{flow_suffix}_out"
    )
    relative = source_video.resolve().relative_to(input_root.resolve())
    output_rel = relative.with_name(f"{relative.stem}_{flow_suffix}.mp4")
    return output_root / output_rel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Убрать шум/артефакты кодека через ffmpeg (CPU-фильтры + NVENC, fallback на libx264)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Путь к входному видео ИЛИ к папке с .mp4 файлами.",
    )
    parser.add_argument(
        "--flow",
        type=str,
        default="fast",
        choices=tuple(FLOWS.keys()),
        help="Режим обработки: fast или strong_artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Для input-файла: путь к выходному видео (mp4). Для input-папки: путь к выходной папке."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Количество параллельных потоков обработки для режима папки (по умолчанию: 2).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Не сканировать подпапки при обработке папки (искать только в корне input).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Команда ffmpeg (по умолчанию: ffmpeg из PATH).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать команды ffmpeg без запуска.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()

    if shutil.which(args.ffmpeg_bin) is None:
        raise SystemExit(f"ffmpeg не найден: {args.ffmpeg_bin}")

    input_path = args.input.resolve()

    flow_cfg = FLOWS[args.flow]
    vf_expr = str(flow_cfg["vf"])
    flow_label = str(flow_cfg["label"])
    flow_suffix = str(flow_cfg["suffix"])
    required_filters_raw = flow_cfg["required_filters"]
    if not isinstance(required_filters_raw, tuple):
        raise SystemExit("Некорректная конфигурация required_filters.")
    required_filters = tuple(str(item) for item in required_filters_raw)

    _validate_filters(args.ffmpeg_bin, required_filters)

    if args.workers <= 0:
        raise SystemExit("--workers должен быть >= 1")

    videos = _collect_input_videos(input_path, recursive=not args.no_recursive)
    is_dir_mode = input_path.is_dir()

    if is_dir_mode and args.output is not None and args.output.suffix.lower() == ".mp4":
        raise SystemExit("Для режима папки --output должен указывать директорию, а не .mp4 файл.")

    workers = 1 if not is_dir_mode else min(args.workers, max(1, len(videos), os.cpu_count() or 1))

    logging.info("Вход: %s", input_path)
    logging.info("Найдено видео: %d", len(videos))
    logging.info("Параллельных потоков: %d", workers)
    logging.info("Режим: %s", flow_label)
    logging.info("Пайплайн: %s", vf_expr)
    if args.output is not None:
        logging.info("Базовый путь выхода: %s", args.output.resolve())

    cuda_count = 0
    cpu_fallback_count = 0
    failures: list[tuple[Path, str]] = []

    def _process_video(video_path: Path) -> tuple[Path, bool]:
        output_path = _build_output_path(
            source_video=video_path,
            input_root=input_path if is_dir_mode else video_path.parent,
            output_arg=args.output,
            flow_suffix=flow_suffix,
            is_dir_mode=is_dir_mode,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        used_cuda_local = _render_with_fallback(
            ffmpeg_bin=args.ffmpeg_bin,
            input_path=video_path,
            output_path=output_path,
            vf_expr=vf_expr,
            dry_run=args.dry_run,
        )
        return output_path, used_cuda_local

    if workers == 1:
        for video_path in videos:
            logging.info("Обработка: %s", video_path)
            try:
                output_path, used_cuda = _process_video(video_path)
                logging.info("Готово: %s", output_path)
                if used_cuda:
                    cuda_count += 1
                else:
                    cpu_fallback_count += 1
            except RuntimeError as error:
                failures.append((video_path, str(error)))
                logging.error("Ошибка обработки %s:\n%s", video_path, error)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video = {
                executor.submit(_process_video, video_path): video_path for video_path in videos
            }
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    output_path, used_cuda = future.result()
                    logging.info("Готово: %s", output_path)
                    if used_cuda:
                        cuda_count += 1
                    else:
                        cpu_fallback_count += 1
                except RuntimeError as error:
                    failures.append((video_path, str(error)))
                    logging.error("Ошибка обработки %s:\n%s", video_path, error)

    total_success = cuda_count + cpu_fallback_count
    logging.info("Успешно обработано: %d из %d", total_success, len(videos))
    logging.info("NVENC: %d, libx264 fallback: %d", cuda_count, cpu_fallback_count)
    if failures:
        logging.error("Ошибок: %d", len(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

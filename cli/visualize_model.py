from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.config import Settings, get_settings
from app.models.registry import build_model_from_registry
from app.models.u_net_baseline import UNetBaseline
from app.models.u_net_crossattention import UNetCrossAttention


def _shape_to_str(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        shape = "x".join(str(x) for x in value.shape)
        return f"{shape} ({value.dtype})"
    if isinstance(value, (tuple, list)):
        inner = ", ".join(_shape_to_str(v) for v in value)
        return f"[{inner}]"
    return str(type(value).__name__)


def _params_human(num: int) -> str:
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def _build_dummy_inputs(settings: Settings) -> dict[str, torch.Tensor]:
    return {
        "high_res": torch.randn(
            1,
            settings.model_high_res_channels,
            settings.model_high_res_height,
            settings.model_high_res_width,
        ),
        "side": torch.randn(
            1,
            settings.model_side_channels,
            settings.model_output_height,
            settings.model_output_width,
        ),
        "low_res": torch.randn(
            1,
            settings.model_low_res_channels,
            settings.model_low_res_height,
            settings.model_low_res_width,
        ),
        "camera": torch.randn(1, 2),
    }


def _run_forward(model: nn.Module, model_input: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**model_input)


def _collect_module_shapes(model: nn.Module, model_input: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    def _hook(name: str):
        def _capture(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                shapes[name] = tuple(output.shape)

        return _capture

    handles = [module.register_forward_hook(_hook(name)) for name, module in model.named_modules() if name]
    with torch.inference_mode():
        _ = _run_forward(model, model_input)
    for handle in handles:
        handle.remove()
    return shapes


def _module_summary(model: nn.Module, model_input: dict[str, torch.Tensor]) -> str:
    rows: list[str] = []
    hooks = []
    seen: set[int] = set()
    model.eval()

    def _hook(name: str):
        def _capture(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            module_id = id(module)
            if module_id in seen:
                return
            seen.add(module_id)
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in module.parameters())
            in_shape = _shape_to_str(args[0]) if args else "-"
            out_shape = _shape_to_str(output)
            rows.append(
                f"{name:<42} | {module.__class__.__name__:<20} | "
                f"{_params_human(total):>8} | {_params_human(trainable):>8} | "
                f"{in_shape:<24} -> {out_shape}"
            )

        return _capture

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(_hook(name)))

    with torch.inference_mode():
        _ = _run_forward(model, model_input)

    for handle in hooks:
        handle.remove()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    header = [
        "MODEL SUMMARY",
        "=" * 120,
        f"Model class: {model.__class__.__name__}",
        f"Total params: {total_params} ({_params_human(total_params)})",
        f"Trainable params: {trainable_params} ({_params_human(trainable_params)})",
        "Inputs:",
        f"  high_res: {list(model_input['high_res'].shape)}",
        f"  side: {list(model_input['side'].shape)}",
        f"  low_res: {list(model_input['low_res'].shape)}",
        f"  camera: {list(model_input['camera'].shape)}",
        "-" * 120,
        f"{'Layer name':<42} | {'Type':<20} | {'Params':>8} | {'Trainable':>8} | Input -> Output",
        "-" * 120,
    ]
    return "\n".join(header + rows) + "\n"


def _shape_label(name: str, shape: tuple[int, ...] | None, *, color: str, extra: str = "") -> str:
    shape_text = "x".join(str(x) for x in shape) if shape else "?"
    suffix = f"\\n{extra}" if extra else ""
    return (
        f'  "{name}" [label="{name}\\n{shape_text}{suffix}", '
        f'style="rounded,filled", fillcolor="{color}", color="#4D648D", fontname="Helvetica"];'
    )


def _build_unet_structural_dot(
    *,
    model_config: dict[str, Any],
    module_shapes: dict[str, tuple[int, ...]],
    input_shapes: dict[str, tuple[int, ...]],
) -> str:
    depth = int(model_config["depth"])
    lines = [
        "digraph SceneUNet {",
        '  rankdir=TB;',
        '  splines=ortho;',
        '  bgcolor="#FAFBFF";',
        '  node [shape=box, fontsize=11];',
        '  edge [color="#6B7FA6", penwidth=1.2];',
        "",
        _shape_label("high_res", input_shapes["high_res"], color="#DFF3E3", extra="center context"),
        _shape_label("side", input_shapes["side"], color="#DFF3E3", extra="source sides"),
        _shape_label("low_res", input_shapes["low_res"], color="#FFF1EA", extra="opti2+depth"),
        _shape_label("camera", input_shapes["camera"], color="#FFF8E7", extra="FiLM cond"),
        _shape_label("high_stem", module_shapes.get("high_stem"), color="#ECF3FF"),
        _shape_label("side_stem", module_shapes.get("side_stem"), color="#ECF3FF"),
        '  "high_res" -> "high_stem";',
        '  "side" -> "side_stem";',
        '  "camera" -> "high_stem" [style=dashed, label="FiLM"];',
        '  "camera" -> "side_stem" [style=dashed, label="FiLM"];',
    ]

    prev = "high_stem"
    for index in range(depth - 1):
        node = f"down_{index}"
        lines.append(_shape_label(node, module_shapes.get(f"down_blocks.{index}"), color="#ECF3FF"))
        lines.append(f'  "{prev}" -> "{node}";')
        if index == 1:
            lines.append(_shape_label("low_fuse", module_shapes.get("low_fuse"), color="#FFE8DC", extra="late fusion"))
            lines.append('  "low_res" -> "low_fuse" [color="#E07A5F"];')
            lines.append('  "side_stem" -> "low_fuse" [style=dashed, color="#E07A5F", label="side down"];')
            lines.append(f'  "{node}" -> "low_fuse";')
            lines.append('  "camera" -> "low_fuse" [style=dashed, label="FiLM"];')
            prev = "low_fuse"
        else:
            lines.append('  "camera" -> "' + node + '" [style=dashed, label="FiLM"];')
            prev = node

    decoder_prev = prev
    for up_index in range(depth - 1):
        up_node = f"up_{up_index}"
        block_node = f"up_block_{up_index}"
        lines.append(_shape_label(up_node, module_shapes.get(f"up_transpose.{up_index}"), color="#FBF0EE"))
        lines.append(_shape_label(block_node, module_shapes.get(f"up_blocks.{up_index}"), color="#FBF0EE"))
        lines.append(f'  "{decoder_prev}" -> "{up_node}";')
        lines.append(f'  "{up_node}" -> "{block_node}";')
        lines.append(f'  "high_stem" -> "{block_node}" [style=dashed, color="#E07A5F", label="skip"];')
        lines.append(f'  "camera" -> "{block_node}" [style=dashed, label="FiLM"];')
        decoder_prev = block_node

    lines.extend(
        [
            _shape_label("side_to_decoder", module_shapes.get("side_to_decoder"), color="#ECF3FF"),
            _shape_label("head", module_shapes.get("head"), color="#FCE8E8"),
            _shape_label(
                "output",
                (
                    1,
                    int(model_config["out_channels"]),
                    int(model_config["output_height"]),
                    int(model_config["output_width"]),
                ),
                color="#DFF3E3",
            ),
            f'  "{decoder_prev}" -> "side_to_decoder";',
            '  "side_stem" -> "side_to_decoder" [style=dashed, color="#E07A5F"];',
            '  "side_to_decoder" -> "head";',
            '  "head" -> "output";',
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_crossattention_structural_dot(
    *,
    module_shapes: dict[str, tuple[int, ...]],
    input_shapes: dict[str, tuple[int, ...]],
    model_config: dict[str, Any],
) -> str:
    lines = [
        "digraph UNetCrossAttention {",
        '  rankdir=TB;',
        '  splines=ortho;',
        '  bgcolor="#FAFBFF";',
        '  node [shape=box, fontsize=11];',
        '  edge [color="#6B7FA6", penwidth=1.2];',
        "",
        _shape_label("high_res", input_shapes["high_res"], color="#DFF3E3", extra="4 green streams"),
        _shape_label("side", input_shapes["side"], color="#FFE8CC", extra="orange query"),
        _shape_label("low_res", input_shapes["low_res"], color="#E8D4C4", extra="brown opti2+depth"),
        _shape_label("camera", input_shapes["camera"], color="#FFF8E7", extra="FiLM"),
        _shape_label("attn_stage1", module_shapes.get("attn_stage1"), color="#D6E8FF", extra="4x cross-attn"),
        _shape_label("attn_stage2", module_shapes.get("attn_stage2"), color="#D6E8FF", extra="5x cross-attn"),
        _shape_label("attn_stage3", module_shapes.get("attn_stage3"), color="#D6E8FF", extra="5x cross-attn"),
        _shape_label("fuse_a_down", module_shapes.get("fuse_a_down"), color="#ECF3FF"),
        _shape_label("fuse_ab_down", module_shapes.get("fuse_ab_down"), color="#ECF3FF"),
        _shape_label("bottleneck", module_shapes.get("film_bottleneck"), color="#F3E8FF", extra="856x8x4"),
        _shape_label("dec_block1", module_shapes.get("dec_block1"), color="#FBF0EE"),
        _shape_label("dec_block2", module_shapes.get("dec_block2"), color="#FBF0EE"),
        _shape_label(
            "output",
            (
                1,
                int(model_config["out_channels"]),
                int(model_config["output_height"]),
                int(model_config["output_width"]),
            ),
            color="#DFF3E3",
        ),
        '  "high_res" -> "attn_stage1" [color="#6BA368"];',
        '  "side" -> "attn_stage1" [color="#E07A5F", label="Q"];',
        '  "camera" -> "attn_stage1" [style=dashed, label="FiLM"];',
        '  "high_res" -> "attn_stage2" [color="#6BA368"];',
        '  "side" -> "attn_stage2" [color="#E07A5F", label="Q"];',
        '  "low_res" -> "attn_stage2" [color="#A67B5B"];',
        '  "camera" -> "attn_stage2" [style=dashed, label="FiLM"];',
        '  "high_res" -> "attn_stage3" [color="#6BA368"];',
        '  "side" -> "attn_stage3" [color="#E07A5F", label="Q"];',
        '  "low_res" -> "attn_stage3" [color="#A67B5B"];',
        '  "camera" -> "attn_stage3" [style=dashed, label="FiLM"];',
        '  "attn_stage1" -> "fuse_a_down";',
        '  "attn_stage2" -> "fuse_a_down" [label="concat"];',
        '  "fuse_a_down" -> "fuse_ab_down";',
        '  "attn_stage3" -> "fuse_ab_down" [label="concat"];',
        '  "fuse_ab_down" -> "bottleneck";',
        '  "camera" -> "bottleneck" [style=dashed, label="FiLM"];',
        '  "bottleneck" -> "dec_block1";',
        '  "attn_stage2" -> "dec_block1" [style=dashed, color="#E07A5F", label="skip B"];',
        '  "camera" -> "dec_block1" [style=dashed, label="FiLM"];',
        '  "dec_block1" -> "dec_block2";',
        '  "attn_stage1" -> "dec_block2" [style=dashed, color="#E07A5F", label="skip A"];',
        '  "side" -> "dec_block2" [style=dashed, color="#E07A5F"];',
        '  "camera" -> "dec_block2" [style=dashed, label="FiLM"];',
        '  "dec_block2" -> "output";',
        "}",
    ]
    return "\n".join(lines) + "\n"


def _build_model_dot(
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    module_shapes: dict[str, tuple[int, ...]],
    input_shapes: dict[str, tuple[int, ...]],
) -> str:
    if isinstance(model, UNetBaseline):
        return _build_unet_structural_dot(
            model_config=model_config,
            module_shapes=module_shapes,
            input_shapes=input_shapes,
        )
    if isinstance(model, UNetCrossAttention):
        return _build_crossattention_structural_dot(
            model_config=model_config,
            module_shapes=module_shapes,
            input_shapes=input_shapes,
        )
    return _build_unet_structural_dot(
        model_config=model_config,
        module_shapes=module_shapes,
        input_shapes=input_shapes,
    )


def _render_dot(dot_file: Path, output_format: str) -> Path | None:
    dot_exe = shutil.which("dot")
    if not dot_exe:
        return None
    out_path = dot_file.with_suffix(f".{output_format}")
    command = [dot_exe, f"-T{output_format}", str(dot_file), "-o", str(out_path)]
    subprocess.run(command, check=True)
    return out_path


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Visualize model architecture built from registry.")
    parser.add_argument("--architecture", type=str, default=settings.model_architecture_name)
    parser.add_argument("--model-config-json", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "model_viz",
    )
    parser.add_argument("--format", type=str, default="svg", choices=["svg", "png", "pdf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    model_config = settings.model_registry_config
    if args.model_config_json:
        model_config.update(json.loads(args.model_config_json))

    model, resolved_cfg = build_model_from_registry(
        architecture_name=args.architecture,
        model_config=model_config,
    )
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_input = _build_dummy_inputs(settings)
    input_shapes = {key: tuple(value.shape) for key, value in model_input.items()}

    module_shapes = _collect_module_shapes(model, model_input)
    summary_text = _module_summary(model, model_input)
    summary_path = args.output_dir / "model_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    dot_text = _build_model_dot(
        model,
        model_config=resolved_cfg,
        module_shapes=module_shapes,
        input_shapes=input_shapes,
    )
    dot_path = args.output_dir / "model_graph.dot"
    dot_path.write_text(dot_text, encoding="utf-8")

    rendered = _render_dot(dot_path, output_format=args.format)

    config_path = args.output_dir / "resolved_model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "architecture": args.architecture,
                "input_shapes": {k: list(v) for k, v in input_shapes.items()},
                "resolved_config": resolved_cfg,
                "module_shapes": {k: list(v) for k, v in module_shapes.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Summary: {summary_path}")
    print(f"DOT graph: {dot_path}")
    print(f"Config: {config_path}")
    if rendered is not None:
        print(f"Rendered graph: {rendered}")
    else:
        print("Graphviz 'dot' is not installed. DOT file is ready for manual rendering.")


if __name__ == "__main__":
    main()

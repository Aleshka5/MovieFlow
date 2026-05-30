from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchview import draw_graph

from app.config import Settings, get_settings

# from app.models.registry import build_model_from_registry
from app.models.model_archive import build_model_from_registry


class _KeywordModelAdapter(nn.Module):
    """Adapter to call keyword-only model with positional tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        high_res: torch.Tensor,
        side: torch.Tensor,
        low_res: torch.Tensor,
        camera: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(high_res=high_res, side=side, low_res=low_res, camera=camera)


def _build_dummy_inputs(
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    high_res = torch.randn(
        1,
        settings.model_high_res_channels,
        settings.model_high_res_height,
        settings.model_high_res_width,
    )
    side = torch.randn(
        1,
        settings.model_side_channels,
        settings.model_output_height,
        settings.model_output_width,
    )
    low_res = torch.randn(
        1,
        settings.model_low_res_channels,
        settings.model_low_res_height,
        settings.model_low_res_width,
    )
    camera = torch.randn(1, 2)
    return high_res, side, low_res, camera


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Visualize model architecture via torchview.")
    parser.add_argument("--architecture", type=str, default=settings.model_architecture_name)
    parser.add_argument("--model-config-json", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "model_viz")
    parser.add_argument("--filename", type=str, default="model_torchview")
    parser.add_argument("--depth", type=int, default=4, help="Recursion depth for torchview graph.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "svg", "pdf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    model_config: dict[str, Any] = dict(settings.model_registry_config)
    if args.model_config_json:
        model_config.update(json.loads(args.model_config_json))

    model, resolved_cfg = build_model_from_registry(
        architecture_name=args.architecture,
        model_config=model_config,
    )
    model.eval()
    adapter = _KeywordModelAdapter(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    high_res, side, low_res, camera = _build_dummy_inputs(settings)
    graph = draw_graph(
        adapter,
        input_data=[high_res, side, low_res, camera],
        device="cpu",
        expand_nested=True,
        depth=args.depth,
        save_graph=False,
        graph_name=f"{args.architecture}_torchview",
        hide_inner_tensors=False,
        hide_module_functions=False,
    )
    dot_path = args.output_dir / f"{args.filename}.dot"
    dot_path.write_text(graph.visual_graph.source, encoding="utf-8")

    rendered_path: Path | None = None
    if shutil.which("dot") is not None:
        rendered_path = Path(
            graph.visual_graph.render(
                filename=args.filename,
                directory=str(args.output_dir),
                format=args.format,
                cleanup=True,
            )
        )

    config_path = args.output_dir / f"{args.filename}.resolved_config.json"
    config_path.write_text(
        json.dumps(
            {
                "architecture": args.architecture,
                "resolved_config": resolved_cfg,
                "input_shapes": {
                    "high_res": list(high_res.shape),
                    "side": list(side.shape),
                    "low_res": list(low_res.shape),
                    "camera": list(camera.shape),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"DOT saved to: {dot_path.as_posix()}")
    if rendered_path is not None:
        print(f"Rendered graph: {rendered_path.as_posix()}")
    else:
        print("Rendered graph: skipped (install Graphviz 'dot' to export image)")
    print(f"Config saved to: {config_path.as_posix()}")
    print(f"Graph object nodes: {len(graph.visual_graph.body)}")


if __name__ == "__main__":
    main()

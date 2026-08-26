"""Export the fixed-resolution coarse FoundationStereo forward pass to ONNX."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# xFormers attention kernels are not exported to standard ONNX operators.
# Set this before importing FoundationStereo/DINOv2, matching the official exporter.
os.environ["XFORMERS_DISABLED"] = "1"


BATCH_SIZE = 1
CHANNELS = 3
HEIGHT = 1024
WIDTH = 1248
VALID_ITERS = 32
OPSET_VERSION = 16

REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_STEREO_ROOT = REPO_ROOT / "FoundationStereo"
CHECKPOINT_PATH = REPO_ROOT / "weight" / "23-51-11" / "model_best_bp2.pth"
CONFIG_PATH = CHECKPOINT_PATH.with_name("cfg.yaml")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "onnx" / "coarse_engine_1024x1248_iters32.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the fixed [1, 3, 1024, 1248], 32-iteration coarse "
            "FoundationStereo forward pass."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"ONNX output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def validate_onnx(output_path: Path, onnx: object) -> None:
    """Validate the exported graph and its fixed public tensor interface."""
    onnx.checker.check_model(str(output_path))
    graph = onnx.load(str(output_path), load_external_data=False).graph

    inputs = {value.name: value for value in graph.input}
    outputs = {value.name: value for value in graph.output}
    if set(inputs) != {"left", "right"}:
        raise RuntimeError(f"Unexpected ONNX inputs: {sorted(inputs)}")
    if set(outputs) != {"disp"}:
        raise RuntimeError(f"Unexpected ONNX outputs: {sorted(outputs)}")

    expected = {
        "left": [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH],
        "right": [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH],
        "disp": [BATCH_SIZE, 1, HEIGHT, WIDTH],
    }
    for name, value in {**inputs, **outputs}.items():
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT:
            actual_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            raise RuntimeError(f"{name} must be float32, got {actual_type}")

        dimensions = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                raise RuntimeError(f"{name} must have a fully static shape")
            dimensions.append(dimension.dim_value)
        if dimensions != expected[name]:
            raise RuntimeError(
                f"{name} shape must be {expected[name]}, got {dimensions}"
            )


def main() -> None:
    args = parse_args()

    require_file(CONFIG_PATH, "FoundationStereo config")
    require_file(CHECKPOINT_PATH, "FoundationStereo checkpoint")
    if not FOUNDATION_STEREO_ROOT.is_dir():
        raise FileNotFoundError(
            f"FoundationStereo source directory not found: {FOUNDATION_STEREO_ROOT}"
        )

    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Output path must end in .onnx: {output_path}")

    if str(FOUNDATION_STEREO_ROOT) not in sys.path:
        sys.path.insert(0, str(FOUNDATION_STEREO_ROOT))

    import onnx
    import torch
    from omegaconf import OmegaConf

    from core.foundation_stereo import FoundationStereo

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export the coarse FoundationStereo ONNX")

    class CoarseFoundationStereoOnnx(FoundationStereo):
        @torch.no_grad()
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=True):
                return FoundationStereo.forward(
                    self,
                    left,
                    right,
                    iters=VALID_ITERS,
                    test_mode=True,
                    low_memory=False,
                )

    config = OmegaConf.load(CONFIG_PATH)
    config.vit_size = config.get("vit_size", "vitl")
    config.height = HEIGHT
    config.width = WIDTH
    config.valid_iters = VALID_ITERS
    config.low_memory = 0

    model = CoarseFoundationStereoOnnx(config)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        weights_only=False,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    left = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, device=device) * 255.0
    right = torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, device=device) * 255.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Inputs: left/right float32 [{BATCH_SIZE}, {CHANNELS}, {HEIGHT}, {WIDTH}]")
    print(f"Iterations: {VALID_ITERS}")
    print(f"Exporting: {output_path}")

    torch.onnx.export(
        model,
        (left, right),
        output_path,
        input_names=["left", "right"],
        output_names=["disp"],
        opset_version=OPSET_VERSION,
        dynamo=False,
        external_data=True,
    )

    validate_onnx(output_path, onnx)
    print(f"Validated ONNX interface: disp float32 [{BATCH_SIZE}, 1, {HEIGHT}, {WIDTH}]")


if __name__ == "__main__":
    main()

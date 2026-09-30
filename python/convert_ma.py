#!/usr/bin/env python3
"""Export the RGB + metric-depth, two-view MapAnything notebook forward.

Run from the repository root using the mapanything Conda environment::

    conda activate mapanything
    python -m pip install onnx onnxscript
    python python/convert_ma.py

Validate the inference adaptation without ONNX dependencies::

    python python/convert_ma.py --validate-only

Optionally install onnxruntime (CPU) or onnxruntime-gpu and use --verify.
The ONNX graph has six FP32 inputs (img, unit rays, ray depth for each view).
Batch, image size, view count and available modalities are fixed at export.
Image/geometry preprocessing and infer() postprocessing stay outside the graph.
Keep the generated external weight file(s) together with the .onnx file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
from types import MethodType
from unittest.mock import patch

# Must precede loading the cached DINOv2 encoder.
os.environ["XFORMERS_DISABLED"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "map-anything"))

import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import apply_log_to_norm
from mapanything.utils.inference import preprocess_input_views_for_inference
from uniception.models.encoders import EncoderGlobalRepInput, ViTEncoderNonImageInput

from ma_utils import prepare_inputs


INPUT_KEYS = ("img", "ray_directions_cam", "depth_along_ray")
INPUT_NAMES = tuple(f"{name}{i}" for i in range(2) for name in ("img", "rays", "depth"))
OUTPUT_KEYS = (
    "pts3d", "pts3d_cam", "ray_directions", "depth_along_ray",
    "cam_trans", "cam_quats", "metric_scaling_factor", "conf",
    "non_ambiguous_mask", "non_ambiguous_mask_logits",
)
OUTPUT_NAMES = tuple(f"view{i}_{key}" for i in range(2) for key in OUTPUT_KEYS)


def fuse_geometry_for_export(self, views, encoder_features):
    """Equivalent to inference with rays + metric depths and no pose priors.

    Eliminate random dropout, data-dependent Python branches and boolean
    indexing. Preserve depth normalization, logarithmic encoding, metric scale
    encoding, addition order and fusion normalization from the original model.
    This specialization is intentionally NOT valid for other input modalities.
    """
    features = torch.cat(encoder_features, dim=0)
    rays = torch.cat([v["ray_directions_cam"] for v in views], dim=0)
    rays = rays.permute(0, 3, 1, 2).contiguous()
    ray_features = self.ray_dirs_encoder(ViTEncoderNonImageInput(data=rays)).features
    features = features + ray_features

    depth = torch.cat([v["depth_along_ray"] for v in views], dim=0)
    valid = depth > 0
    depth_sum = (depth * valid).sum(dim=(1, 2, 3))
    depth_count = valid.sum(dim=(1, 2, 3))
    norm_factor = (depth_sum / (depth_count + 1e-8)).clamp(min=1e-8)
    normalized_depth = depth / norm_factor[:, None, None, None]
    encoded_depth = apply_log_to_norm(normalized_depth).permute(0, 3, 1, 2).contiguous()
    depth_features = self.depth_encoder(
        ViTEncoderNonImageInput(data=encoded_depth)
    ).features
    scale_features = self.depth_scale_encoder(
        EncoderGlobalRepInput(data=torch.log(norm_factor + 1e-8).unsqueeze(-1))
    ).features
    features = features + depth_features + scale_features[:, :, None, None]

    # The notebook supplies no camera poses: all pose input contributions are
    # masked to zero in the original forward. Prediction heads remain intact.
    features = features.permute(0, 2, 3, 1).contiguous()
    features = self.fusion_norm_layer(features)
    features = features.permute(0, 3, 1, 2).contiguous()
    return features.chunk(len(views), dim=0)


class MapAnythingONNX(torch.nn.Module):
    def __init__(self, model, head_minibatch_size=1):
        super().__init__()
        self.model = model
        self.head_minibatch_size = head_minibatch_size

    def forward(self, img0, rays0, depth0, img1, rays1, depth1):
        views = [
            {
                "img": img,
                "ray_directions_cam": rays,
                "depth_along_ray": depth,
                "is_metric_scale": torch.ones(
                    img.shape[0], dtype=torch.bool, device=img.device
                ),
                "data_norm_type": [self.model.encoder.data_norm_type],
            }
            for img, rays, depth in (
                (img0, rays0, depth0), (img1, rays1, depth1)
            )
        ]
        predictions = self.model(
            views,
            memory_efficient_inference=self.head_minibatch_size > 0,
            minibatch_size=self.head_minibatch_size or None,
        )
        return tuple(pred[key] for pred in predictions for key in OUTPUT_KEYS)


@contextmanager
def export_adaptation(model):
    """Patch this instance only; restore everything on success or failure."""
    model_module = importlib.import_module("mapanything.models.mapanything.model")
    with (
        patch.object(
            model, "_encode_and_fuse_optional_geometric_inputs",
            MethodType(fuse_geometry_for_export, model),
        ),
        # Fixed head minibatches still call this allocator housekeeping helper.
        # It has no numerical effect and must not enter the exported graph.
        patch.object(model_module, "empty_cache", lambda device: None),
    ):
        yield


def check_dependencies(verify=False):
    required = ["onnx", "onnxscript"]
    if verify:
        required.append("onnxruntime")
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
            if name == "onnx":
                # The repository's onnx/ folder can be a namespace package
                # even when the actual ONNX package has not been installed.
                importlib.import_module("onnx.checker")
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"Missing packages in {sys.executable}: {', '.join(missing)}\n"
            "Activate conda environment mapanything and run:\n"
            f"  python -m pip install {' '.join(missing)}\n"
            "Or use --validate-only to check PyTorch forward without ONNX."
        )


def assert_outputs_close(actual, expected, *, rtol, atol, label):
    for name, result, reference in zip(OUTPUT_NAMES, actual, expected, strict=True):
        result, reference = result.detach().cpu(), reference.detach().cpu()
        torch.testing.assert_close(
            result, reference, rtol=rtol, atol=atol, msg=lambda msg: f"{label}/{name}: {msg}"
        )
        error = (result.float() - reference.float()).abs().max().item()
        print(f"  {name}: max_abs_error={error:.6g}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "onnx/MapAnything")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--captures", nargs=2, default=("capture1", "capture2"))
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=434)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "onnx/mapanything_2views_434x518.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--head-minibatch-size", type=int, choices=(0, 1, 2), default=1,
                        help="1 saves head memory; 0 disables head minibatching")
    parser.add_argument("--validate-only", action="store_true", help="Compare original and adapted PyTorch forward, then exit")
    parser.add_argument("--verify", action="store_true", help="Also compare ONNX Runtime with adapted PyTorch")
    parser.add_argument("--ort-provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"), default="CPUExecutionProvider")
    parser.add_argument("--rtol", type=float, default=1e-4, help="FP32 validation relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-4, help="FP32 validation absolute tolerance")
    args = parser.parse_args()
    if any(x <= 0 or x % 14 for x in (args.height, args.width)):
        parser.error("--height and --width must be positive multiples of 14")
    if args.opset < 18:
        parser.error("--opset must be at least 18")
    if args.rtol < 0 or args.atol < 0:
        parser.error("Validation tolerances must be nonnegative")
    if args.validate_only and args.verify:
        parser.error("--validate-only and --verify cannot be combined")
    if args.output.suffix != ".onnx":
        parser.error("--output must end in .onnx")
    return args


def main():
    args = parse_args()
    print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}\nDevice: {args.device}", flush=True)
    if not args.validate_only:
        check_dependencies(args.verify)
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}; choose another --output")
        if args.verify:
            import onnxruntime as ort
            if args.ort_provider not in ort.get_available_providers():
                raise RuntimeError(f"ORT provider unavailable: {args.ort_provider}; available: {ort.get_available_providers()}")
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {args.checkpoint / 'config.json'}")

    _, notebook_views, _ = prepare_inputs(
        args.data_root, args.captures, resize_mode="fixed_size",
        size=(args.width, args.height),
    )
    views = preprocess_input_views_for_inference(notebook_views)
    inputs = tuple(v[key].to(args.device, dtype=torch.float32).contiguous()
                   for v in views for key in INPUT_KEYS)
    for view in views:
        if view["img"].shape != (1, 3, args.height, args.width):
            raise ValueError("Preprocessed image shape differs from requested static shape")
        if not bool(view["is_metric_scale"].all()):
            raise ValueError("Only metric depth inputs are supported")
        if "camera_pose_quats" in view or "camera_pose_trans" in view:
            raise ValueError("This export does not support input pose priors")
    for name, tensor in zip(INPUT_NAMES, inputs, strict=True):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite input: {name}")
        print(f"  {name}: {tuple(tensor.shape)}, {tensor.dtype}", flush=True)

    print("Loading local checkpoint ...", flush=True)
    model = MapAnything.from_pretrained(str(args.checkpoint), local_files_only=True).float().to(args.device).eval()
    if model.scene_rep_type != "raydirs+depth+pose+confidence+mask":
        raise ValueError(f"Unsupported checkpoint scene representation: {model.scene_rep_type}")
    if any(v["data_norm_type"][0] != model.encoder.data_norm_type for v in views):
        raise ValueError("Input normalization does not match checkpoint encoder")
    model._configure_geometric_input_config(
        use_calibration=True, use_depth=True, use_pose=True,
        use_depth_scale=True, use_pose_scale=True,
    )
    # Compare both paths in FP32, without the notebook's BF16 autocast or TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wrapper = MapAnythingONNX(model, args.head_minibatch_size).eval()
    with torch.no_grad():
        print("Running original FP32 forward ...", flush=True)
        reference = tuple(t.detach().cpu() for t in wrapper(*inputs))
        with export_adaptation(model):
            print("Checking adapted FP32 forward ...", flush=True)
            expected = tuple(t.detach().cpu() for t in wrapper(*inputs))
            assert_outputs_close(expected, reference, rtol=args.rtol, atol=args.atol, label="adaptation")
            del reference
            if args.validate_only:
                print("Original/adapted forward comparison passed; no ONNX file written.", flush=True)
                return

            args.output.parent.mkdir(parents=True, exist_ok=True)
            reports = args.output.parent / f"{args.output.stem}_reports"
            print(f"Exporting static FP32 ONNX to {args.output} ...", flush=True)
            torch.onnx.export(
                wrapper, inputs, str(args.output),
                input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                opset_version=args.opset, dynamo=True, external_data=True,
                report=True, artifacts_dir=str(reports),
            )

    import onnx
    # Pass the path rather than a loaded protobuf: weights exceed 2 GB.
    onnx.checker.check_model(str(args.output))
    verification = "not requested"
    if args.verify:
        import onnxruntime as ort
        print(f"Checking ONNX Runtime ({args.ort_provider}) ...", flush=True)
        # Release PyTorch GPU weights before optionally loading ORT on the GPU.
        feed = {name: tensor.cpu().numpy() for name, tensor in zip(INPUT_NAMES, inputs, strict=True)}
        del wrapper, model, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        session = ort.InferenceSession(str(args.output), providers=[args.ort_provider])
        actual = tuple(torch.from_numpy(x) for x in session.run(list(OUTPUT_NAMES), feed))
        assert_outputs_close(actual, expected, rtol=args.rtol, atol=args.atol, label="onnxruntime")
        verification = "passed"

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "torch_version": torch.__version__, "opset": args.opset,
        "batch_size": 1, "num_views": 2, "height": args.height, "width": args.width,
        "captures": list(args.captures), "dtype": "float32",
        "inputs": [
            {"name": name, "shape": [1, 3, args.height, args.width] if name.startswith("img")
             else [1, args.height, args.width, 3 if name.startswith("rays") else 1]}
            for name in INPUT_NAMES
        ],
        "outputs": [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
                    for name, t in zip(OUTPUT_NAMES, expected, strict=True)],
        "preprocessing": "ma_utils.prepare_inputs, then preprocess_input_views_for_inference; RGB uses encoder normalization, rays are unit camera rays, depth is metric distance along rays (zero means invalid)",
        "postprocessing": "Group outputs by view into raw dictionaries; call postprocess_model_outputs_for_inference with corresponding input views to reproduce infer() masks, depth_z, intrinsics and camera_poses",
        "pytorch_adaptation_check": "passed", "onnx_checker": "passed",
        "onnxruntime_check": verification,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Export complete: {args.output}\nKeep external weight files alongside the model.\nONNX Runtime comparison: {verification}", flush=True)


if __name__ == "__main__":
    main()

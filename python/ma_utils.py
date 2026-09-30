"""Input preparation and previews for the MapAnything notebook.

Match the C++ capture export: rectify raw RGB, scale its intrinsics to the
saved metric Z-depth grid, then jointly preprocess RGB, depth and intrinsics.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np
import torch
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.inference import validate_input_views_for_inference

__all__ = [
    "repository_root", "prepare_capture_input", "prepare_inputs", "show_input_views",
    "prepare_model_for_inference", "inference_timer", "summarize_predictions",
    "merge_capture_meshes", "save_merged_mesh",
]


def repository_root() -> Path:
    """Locate the repository containing this helper module."""
    return Path(__file__).resolve().parents[1]


def _calibration_matrix(calibration, name):
    entry = calibration[name]
    return np.asarray(entry["data"], dtype=np.float64).reshape(entry["rows"], entry["cols"])


def prepare_capture_input(capture_dir: Path):
    """Return aligned RGB, pixel intrinsics, metric Z-depth and capture metadata."""
    capture_dir = Path(capture_dir)
    calibration = json.loads((capture_dir / "sentech_stereo_calibration.json").read_text())
    left_bgr = cv2.imread(str(capture_dir / "left.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture_dir / "depth.tiff"), cv2.IMREAD_UNCHANGED)
    if left_bgr is None or depth is None:
        raise FileNotFoundError(f"{capture_dir}: 无法读取 left.png 或 depth.tiff")
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"{capture_dir}: depth.tiff 应为单通道浮点米制深度")

    height, width = left_bgr.shape[:2]
    if (width, height) != (calibration["image_width"], calibration["image_height"]):
        raise ValueError(f"{capture_dir}: 原图尺寸与标定尺寸不一致")
    k1 = _calibration_matrix(calibration, "left_camera_matrix")
    k2 = _calibration_matrix(calibration, "right_camera_matrix")
    d1 = _calibration_matrix(calibration, "left_distortion")
    d2 = _calibration_matrix(calibration, "right_distortion")
    rotation = _calibration_matrix(calibration, "right_to_left_rotation")
    translation = _calibration_matrix(calibration, "right_to_left_translation")

    # 与 cpp/src/stereo/StereoFrame.cpp 完全相同的标定约定和校正参数。
    r1, _, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, (width, height), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        k1, d1, r1, p1, (width, height), cv2.CV_32FC1,
    )
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    rectified_left = cv2.remap(left_rgb, map_x, map_y, cv2.INTER_LINEAR)

    depth_h, depth_w = depth.shape
    image_rgb = cv2.resize(rectified_left, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
    # 复用生成 depth.tiff 时的内参缩放约定。
    intrinsics = p1[:, :3].copy()
    intrinsics[0, :] *= depth_w / width
    intrinsics[1, :] *= depth_h / height
    intrinsics = intrinsics.astype(np.float32)

    depth_m = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_m[~valid] = 0.0
    if not valid.any():
        raise ValueError(f"{capture_dir}: 没有有效深度")
    if not np.isfinite(intrinsics).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
        raise ValueError(f"{capture_dir}: 无效内参")

    view = {
        "img": image_rgb,
        "intrinsics": intrinsics,
        "depth_z": depth_m,
        "is_metric_scale": torch.tensor([True]),
    }
    info = {
        "capture": capture_dir.name,
        "baseline_m": abs(float(p2[0, 3] / p2[0, 0])),
        "valid_fraction": float(valid.mean()),
        "depth_p01_p50_p99_m": np.percentile(depth_m[valid], [1, 50, 99]),
    }
    return view, info


def prepare_inputs(
    data_root: Path,
    capture_names=("capture1", "capture2"),
    *,
    resize_mode="fixed_size",
    size=(518, 434),
):
    """Return raw views, CPU model inputs and metadata in capture order.

    ``size`` is (width, height); preprocessing aligns it to 14-pixel patches.
    Prints geometry and shape checks without loading or running the model.
    """
    data_root = Path(data_root)
    capture_names = tuple(capture_names)
    views, capture_info = [], []
    for capture_name in capture_names:
        view, info = prepare_capture_input(data_root / capture_name)
        views.append(view)
        capture_info.append(info)
        print(f"{capture_name}: RGB={view['img'].shape}, depth={view['depth_z'].shape}, "
              f"valid={info['valid_fraction']:.1%}, baseline={info['baseline_m']:.6f} m")
        print("  depth [p01, p50, p99] (m):", info["depth_p01_p50_p99_m"])
        print("  K on depth grid:\n", view["intrinsics"])

    # 联合处理 RGB、深度和内参；不要只缩放 RGB。
    resize_options = {"resize_mode": resize_mode}
    if resize_mode == "fixed_size":
        resize_options["size"] = size
    processed_views = preprocess_inputs(views, **resize_options)
    validate_input_views_for_inference(processed_views)

    for capture_name, view in zip(capture_names, processed_views):
        assert view["img"].shape[-2:] == view["depth_z"].shape[-2:]
        assert view["intrinsics"].shape == (1, 3, 3)
        assert torch.isfinite(view["depth_z"]).all()
        assert (view["depth_z"] >= 0).all()
        assert (view["depth_z"] > 0).any()
        assert all(n % 14 == 0 for n in view["img"].shape[-2:])
        print(f"{capture_name}: img={tuple(view['img'].shape)}, "
              f"depth_z={tuple(view['depth_z'].shape)}, "
              f"intrinsics={tuple(view['intrinsics'].shape)}")

    return views, processed_views, capture_info


def show_input_views(views, capture_names):
    """Show RGB, metric depth and their overlay on the original depth grid."""
    import matplotlib.pyplot as plt

    # 预处理前的 800×960 网格：核对校正后的 RGB 和已有深度是否对齐。
    valid_depths = np.concatenate([v["depth_z"][v["depth_z"] > 0] for v in views])
    vmin, vmax = np.percentile(valid_depths, [1, 99])
    fig, axes = plt.subplots(len(views), 3, figsize=(16, 5 * len(views)), squeeze=False)
    for row, (capture_name, view) in enumerate(zip(capture_names, views)):
        depth_display = np.ma.masked_where(view["depth_z"] <= 0, view["depth_z"])
        axes[row, 0].imshow(view["img"])
        axes[row, 0].set_title(f"{capture_name}: rectified left RGB")
        rendered = axes[row, 1].imshow(depth_display, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("Z-depth (m); invalid pixels masked")
        fig.colorbar(rendered, ax=axes[row, 1], shrink=0.75, label="m")
        axes[row, 2].imshow(view["img"])
        axes[row, 2].imshow(depth_display, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.5)
        axes[row, 2].set_title("RGB + depth overlay")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    plt.show()
    return fig


def prepare_model_for_inference(model, device):
    """Move the model to the selected device and enable evaluation mode."""
    device = torch.device(device)
    model = model.to(device).eval()
    label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Inference device: {device} ({label})")
    return model


@contextmanager
def inference_timer(device):
    """Measure one inference call, including transfers and postprocessing."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"Inference completed in {perf_counter() - started:.3f} s")


def summarize_predictions(predictions, capture_names, input_views):
    """Validate output dimensions and report masked Z-depth statistics in metres."""
    if len(predictions) != len(capture_names) or len(predictions) != len(input_views):
        raise ValueError("Prediction count must match captures and input views")
    summaries = []
    for name, pred, view in zip(capture_names, predictions, input_views):
        batch, _, height, width = view["img"].shape
        shapes = {
            "depth_z": (batch, height, width, 1),
            "pts3d": (batch, height, width, 3),
            "intrinsics": (batch, 3, 3),
            "camera_poses": (batch, 4, 4),
            "mask": (batch, height, width, 1),
        }
        for key, expected in shapes.items():
            if key not in pred or tuple(pred[key].shape) != expected:
                raise ValueError(f"{name}: expected {key} with shape {expected}")
        depth = pred["depth_z"]
        valid = pred["mask"].bool() & torch.isfinite(depth) & (depth > 0)
        selected = depth[valid].detach().float()
        summary = {
            "capture": name,
            "depth_shape": tuple(depth.shape),
            "device": str(depth.device),
            "valid_fraction": valid.float().mean().item(),
            "median_depth_m": selected.median().item() if selected.numel() else None,
        }
        summaries.append(summary)
        median = summary["median_depth_m"]
        depth_text = f"{median:.4f} m" if median is not None else "no valid depth"
        print(f"{name}: depth_z={summary['depth_shape']}, device={summary['device']}, "
              f"valid={summary['valid_fraction']:.1%}, median depth={depth_text}")
    return summaries


def merge_capture_meshes(data_root, predictions, capture_names=("capture1", "capture2")):
    """Combine two original meshes in the first rectified camera frame (metres).

    Prediction order must match capture_names. This concatenates mesh geometry;
    it does not fuse overlapping surfaces or run additional registration.
    """
    import open3d as o3d

    capture_names = tuple(capture_names)
    if len(capture_names) != 2 or len(predictions) != 2:
        raise ValueError("Mesh merging expects exactly two captures and predictions")
    poses, meshes = [], []
    for name, pred in zip(capture_names, predictions):
        pose = pred["camera_poses"]
        if tuple(pose.shape) != (1, 4, 4):
            raise ValueError(f"{name}: expected camera_poses shape (1, 4, 4)")
        pose = pose[0].detach().to(device="cpu", dtype=torch.float64).numpy()
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f"{name}: invalid camera pose")
        mesh_path = Path(data_root) / name / "mesh.ply"
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_vertices() or not mesh.has_triangles():
            raise ValueError(f"{mesh_path}: no triangle mesh geometry")
        poses.append(pose)
        meshes.append(mesh)

    # camera_poses are cam2world: camera 2 -> world -> camera 1.
    transform = np.linalg.inv(poses[0]) @ poses[1]
    meshes[1].transform(transform)
    combined_mesh = meshes[0] + meshes[1]
    combined_mesh.compute_vertex_normals()
    print(f"Merged {capture_names[1]} into {capture_names[0]} coordinates (metres): "
          f"{len(combined_mesh.vertices):,} vertices, {len(combined_mesh.triangles):,} triangles")
    return combined_mesh, transform


def save_merged_mesh(mesh, output_path):
    """Write a binary PLY with vertex colours and normals, returning its path."""
    import open3d as o3d

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".ply":
        raise ValueError("Merged mesh output must use the .ply extension")
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError("Cannot save an empty triangle mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(
        str(output_path), mesh, write_ascii=False,
        write_vertex_normals=True, write_vertex_colors=True,
    ):
        raise RuntimeError(f"Failed to write mesh: {output_path}")
    print(f"Saved PLY: {output_path}")
    return output_path

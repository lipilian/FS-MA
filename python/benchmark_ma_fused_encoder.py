"""Measure fixed-batch F_fused + CLS generation from the existing MA ONNX.

Extracts the unchanged ancestor graph before multi-view attention. Artifacts go
to a new directory; the GUI ONNX, external weights and engine are read-only.
The measured interval excludes input preparation/transfers and model loading.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import onnx
import torch

sys.path.append("/usr/lib/python3.12/dist-packages")
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "onnx/MapAnything/onnx/mapanything_dynamic_raw_bf16.onnx"
BOUNDARIES = {"F_fused": "clone_7", "CLS": "permute_51"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def extract(output, view_count):
    model = onnx.load(SOURCE, load_external_data=False)
    infos = {v.name: v for v in model.graph.value_info}
    expected_shapes = {"F_fused": ["num_views", 1536, 31, 37], "CLS": ["num_views", 1536, 1]}
    for label, name in BOUNDARIES.items():
        value = infos[name].type.tensor_type
        assert value.elem_type == onnx.TensorProto.FLOAT
        assert [d.dim_value or d.dim_param for d in value.shape.dim] == expected_shapes[label]
    # Verify the intended boundary, beyond matching its shape/name.
    producers = {s: n for n in model.graph.node for s in n.output}
    fused_identity = producers[BOUNDARIES["F_fused"]]
    assert fused_identity.op_type == "Identity"
    fused_transpose = producers[fused_identity.input[0]]
    assert fused_transpose.op_type == "Transpose"
    norm = producers[fused_transpose.input[0]]
    assert norm.op_type == "LayerNormalization" and norm.input[1] == "model.fusion_norm_layer.weight"
    subset = onnx.utils.Extractor(model).extract_model(["inputs"], list(BOUNDARIES.values()))
    before = {n.name: n.SerializeToString() for n in model.graph.node}
    assert all(n.SerializeToString() == before[n.name] for n in subset.graph.node)
    assert not any("info_sharing" in x.name or "dense_head" in x.name or "pose_head" in x.name for x in subset.graph.initializer)
    # Fix only the declared batch dimension. Runtime shape operations are left
    # to TensorRT; the export's per-view depth reductions also work with V=1.
    for value in list(subset.graph.input) + list(subset.graph.output) + list(subset.graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "num_views":
                dim.ClearField("dim_param")
                dim.dim_value = view_count
    for label, name in BOUNDARIES.items():
        subset.graph.node.append(onnx.helper.make_node("Identity", [name], [label], name=f"cache_output_{label}"))
    outputs = []
    for label, name in BOUNDARIES.items():
        value = copy.deepcopy(next(v for v in subset.graph.output if v.name == name))
        value.name = label
        outputs.append(value)
    del subset.graph.output[:]
    subset.graph.output.extend(outputs)
    external = set()
    for initializer in subset.graph.initializer:
        for field in initializer.external_data:
            if field.key == "location":
                external.add(field.value)
    # Copy rather than edit/relink the original weights. ONNX parsing therefore
    # has a self-contained, independent external-data file.
    sources = {str(SOURCE): digest(SOURCE), str(SOURCE.with_suffix(".engine")): digest(SOURCE.with_suffix(".engine"))}
    for location in external:
        source = (SOURCE.parent / location).resolve()
        destination = output / location
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sources[str(source)] = digest(source)
        assert digest(destination) == sources[str(source)]
    path = output / f"fused_encoder_v{view_count}.onnx"
    onnx.save(subset, path)
    onnx.checker.check_model(str(path))
    report = {"source_files_sha256": sources, "source_nodes": len(model.graph.node),
              "extracted_nodes": len(subset.graph.node), "ancestor_nodes_unchanged": True,
              "boundaries": BOUNDARIES, "output_shapes": {k: [view_count] + s[1:] for k, s in expected_shapes.items()},
              "onnx_sha256": digest(path)}
    write_json(output / "extraction.json", report)
    print("Extracted original DINO + geometry + fusion graph", flush=True)
    return path, report


def build(path, output):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    start = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")
    engine_path = path.with_suffix(".engine")
    engine_path.write_bytes(bytes(serialized))
    report = {"build_seconds": time.monotonic() - start, "engine_sha256": digest(engine_path),
              "builder_optimization_level": 3, "workspace_mib": 4096, "tf32": False}
    write_json(output / "build.json", report)
    print("Built independent fixed-batch engine", flush=True)
    return engine_path, report


def inputs(output, view_count, input_dir=None):
    labels = ("capture1", "capture2")
    if input_dir is not None:
        base = {label: torch.from_numpy(np.load(input_dir / f"{label}.input.npy")) for label in labels}
    else:
        sys.path.insert(0, str(ROOT / "map-anything"))
        from ma_utils import prepare_inputs
        from mapanything.utils.inference import preprocess_input_views_for_inference
        _, views, _ = prepare_inputs(ROOT / "data", labels, size=(518, 434))
        prepared = preprocess_input_views_for_inference(views)
        base = {label: torch.cat((view["img"], view["ray_directions_cam"].permute(0, 3, 1, 2),
                                 view["depth_along_ray"].permute(0, 3, 1, 2)), dim=1).float().contiguous().cpu()
                for label, view in zip(labels, prepared)}
    for value in base.values():
        assert value.shape == (1, 7, 434, 518) and torch.isfinite(value).all()
        assert value.dtype == torch.float32 and (value[:, 6] >= 0).all()
    result = {}
    for offset, label in enumerate(labels):
        rows = []
        for i in range(view_count):
            row = base[labels[(offset + i) % len(labels)]].clone()
            if i >= 2:
                row[:, :3] += 0.01 * i
                row[:, 6] *= 1 + 0.025 * i
            rows.append(row)
        value = torch.cat(rows, dim=0).contiguous()
        np.save(output / f"{label}.input.npy", value.numpy())
        result[label] = value
    return result


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.samples = defaultdict(list)

    def report_layer_time(self, name, ms):
        self.samples[name].append(float(ms))


def benchmark(path, output, view_count, input_dir=None):
    fixtures = inputs(output, view_count, input_dir)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    stream = torch.cuda.Stream()
    tensors = {}
    with torch.cuda.stream(stream):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            assert engine.get_tensor_dtype(name) == trt.float32
            tensors[name] = torch.empty(tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
            assert context.set_tensor_address(name, tensors[name].data_ptr())
    stream.synchronize()
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    (output / "layers.json").write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

    def run():
        assert context.execute_async_v3(stream.cuda_stream)

    def set_input(value):
        assert tuple(value.shape) == tuple(tensors["inputs"].shape)
        with torch.cuda.stream(stream):
            tensors["inputs"].copy_(value)
        stream.synchronize()

    set_input(fixtures["capture1"])
    # Sustained warmup, avoiding loading, initialization, or shape preparation
    # in the timing interval. Synchronize per run to match the timed cadence.
    for i in range(200):
        run()
        stream.synchronize()
        if (i + 1) % 50 == 0:
            print(f"warmup {i + 1}/200", flush=True)
    report = {"gpu": torch.cuda.get_device_name(), "tensorrt": trt.__version__, "torch": torch.__version__,
              "input_shape": [view_count, 7, 434, 518], "warmup": 200, "runs_per_pass": 50,
              "fixture_note": "Two real captures; batches above two views repeat captures with deterministic RGB/depth perturbations. Used for timing, not reconstruction-quality validation.",
              "input_sha256": {label: hashlib.sha256(value.numpy().tobytes()).hexdigest() for label, value in fixtures.items()},
              "method": "CUDA events; GPU-resident inputs/outputs; no CUDA Graph; no profiler during timing",
              "excluded": ["FS inference", "RGB/depth preprocessing", "H2D/D2H", "engine load", "initial allocation"],
              "engine_layers": engine.num_layers, "context_memory_bytes": engine.device_memory_size_v2,
              "outputs": {n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in tensors.items() if n != "inputs"},
              "passes": [], "validation": {}}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for label in ("capture1", "capture2", "capture2", "capture1"):
        set_input(fixtures[label])
        for _ in range(10):
            run()
        stream.synchronize()
        values = []
        for _ in range(50):
            start.record(stream)
            run()
            end.record(stream)
            end.synchronize()
            values.append(start.elapsed_time(end))
        report["passes"].append({"fixture": label, "gpu_ms": values, "median_ms": statistics.median(values)})
        print(f"{label}: {statistics.median(values):.3f} ms", flush=True)
        write_json(output / "benchmark.json", report)
    for label, value in fixtures.items():
        set_input(value)
        copies = []
        for _ in range(2):
            run()
            stream.synchronize()
            copies.append({n: t.cpu().numpy().copy() for n, t in tensors.items() if n != "inputs"})
        report["validation"][label] = {}
        for name in copies[0]:
            np.save(output / f"{label}.{name}.npy", copies[0][name])
            report["validation"][label][name] = {"finite": bool(np.isfinite(copies[0][name]).all()),
                                                 "repeat_exact": bool(np.array_equal(copies[0][name], copies[1][name]))}
    profiler = Profiler()
    context.profiler = profiler
    for _ in range(10):
        run()
        stream.synchronize()
    write_json(output / "profile.json", [{"name": n, "average_ms": statistics.mean(v), "samples": len(v)}
                                          for n, v in profiler.samples.items()])
    all_times = [v for p in report["passes"] for v in p["gpu_ms"]]
    report.update(median_ms=statistics.median(all_times), p05_ms=float(np.quantile(all_times, .05)),
                  p95_ms=float(np.quantile(all_times, .95)))
    write_json(output / "benchmark.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--views", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--input-dir", type=Path, help="Reuse capture1/2.input.npy from a prior single-view run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path, extraction = extract(args.output, args.views)
    engine_path, build_report = build(path, args.output)
    report = benchmark(engine_path, args.output, args.views, args.input_dir)
    for path, expected in extraction["source_files_sha256"].items():
        assert digest(Path(path)) == expected, f"Source changed: {path}"
    summary = {"scope": f"{args.views}-view batch DINO + ray/depth/scale encoders + fusion -> F_fused and CLS",
               "view_count": args.views, "fixture_note": report["fixture_note"],
               "median_ms": report["median_ms"], "p05_ms": report["p05_ms"], "p95_ms": report["p95_ms"],
               "pass_medians_ms": [p["median_ms"] for p in report["passes"]], "gpu": report["gpu"],
               "tensorrt": report["tensorrt"], "method": report["method"], "excluded": report["excluded"],
               "outputs": report["outputs"], "validation": report["validation"], "original_files_unchanged": True,
               "build": build_report,
               "limits": [f"New standalone fixed V={args.views} subgraph engine; not a segment measured inside the original dynamic V=2..5 engine.",
                          "Original ONNX ancestor operations and weights retained. This measurement does not establish numerical equality to full-engine intermediate outputs."]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

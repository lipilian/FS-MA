#!/usr/bin/env python3
"""Benchmark a split MA pose/scale engine on identical cached encoder features.

Measures only the suffix: GPU-resident F_fused/CLS -> raw poses/scale. No CUDA
Graph or profiler is active during timing. Reuse --features-dir when comparing
future engines. Features come from two real captures; V>2 includes repeated,
deterministically perturbed views, so this is not a reconstruction-quality test.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from benchmark_ma_fused_encoder import ROOT, Profiler, inputs, trt, write_json


FEATURES = {"F_fused": (1536, 31, 37), "CLS": (1536, 1)}
LABELS = ("capture1", "capture2")


def encode_features(encoder_path, directory, stream):
    directory.mkdir()
    fixtures = inputs(directory, 5)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(encoder_path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    for name, shape in {"inputs": (1, 7, 434, 518), **{n: (1, *s) for n, s in FEATURES.items()}}.items():
        assert tuple(engine.get_tensor_shape(name)) == shape
        assert engine.get_tensor_dtype(name) == trt.float32
    result = {}
    with torch.cuda.stream(stream):
        image = torch.empty((1, 7, 434, 518), device="cuda", dtype=torch.float32)
        assert context.set_tensor_address("inputs", image.data_ptr())
        for label, value in fixtures.items():
            result[label] = {name: torch.empty((5, *shape), device="cuda", dtype=torch.float32)
                             for name, shape in FEATURES.items()}
            for i in range(5):
                image.copy_(value[i:i + 1])
                for name, tensor in result[label].items():
                    assert context.set_tensor_address(name, tensor[i].data_ptr())
                assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            for name, tensor in result[label].items():
                array = tensor.cpu().numpy()
                assert np.isfinite(array).all()
                np.save(directory / f"{label}.{name}.npy", array)
    print("Cached features from the existing single-view encoder", flush=True)
    return result


def load_features(directory, stream):
    result = {}
    with torch.cuda.stream(stream):
        for label in LABELS:
            result[label] = {}
            for name, shape in FEATURES.items():
                array = np.load(directory / f"{label}.{name}.npy")
                assert array.shape == (5, *shape) and array.dtype == np.float32 and np.isfinite(array).all()
                result[label][name] = torch.from_numpy(array).to("cuda")
    stream.synchronize()
    return result


def benchmark(engine, fixtures, views, directory, stream, warmup, runs_per_pass):
    directory.mkdir()
    context = engine.create_execution_context()
    assert context is not None
    for name, shape in FEATURES.items():
        assert context.set_input_shape(name, (views, *shape))
    assert not context.infer_shapes()
    expected_outputs = {"poses": (views, 7), "scale": (1, 1, 1)}
    with torch.cuda.stream(stream):
        outputs = {}
        for name, shape in expected_outputs.items():
            assert tuple(context.get_tensor_shape(name)) == shape, (name, context.get_tensor_shape(name))
            outputs[name] = torch.empty(shape, device="cuda", dtype=torch.float32)
            assert context.set_tensor_address(name, outputs[name].data_ptr())
    stream.synchronize()

    def bind(features):
        for name, tensor in features.items():
            assert tensor.is_contiguous() and tuple(tensor.shape[1:]) == FEATURES[name]
            assert context.set_tensor_address(name, tensor.data_ptr())

    def run():
        assert context.execute_async_v3(stream.cuda_stream)

    def snapshot():
        run()
        stream.synchronize()
        return {name: tensor.cpu().numpy().copy() for name, tensor in outputs.items()}

    bind(fixtures[LABELS[0]])
    for _ in range(warmup):
        run()
        stream.synchronize()
    print(f"V={views}: warmup complete", flush=True)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    report = {"views": views, "warmup": warmup, "runs_per_pass": runs_per_pass,
              "outputs": {name: list(shape) for name, shape in expected_outputs.items()},
              "passes": [], "validation": {}}
    for label in ("capture1", "capture2", "capture2", "capture1"):
        bind(fixtures[label])
        for _ in range(5):
            run()
        stream.synchronize()
        values = []
        for _ in range(runs_per_pass):
            start.record(stream)
            run()
            end.record(stream)
            end.synchronize()
            values.append(start.elapsed_time(end))
        report["passes"].append({"fixture": label, "gpu_ms": values,
                                  "median_ms": statistics.median(values)})
        print(f"V={views} {label}: {statistics.median(values):.3f} ms", flush=True)

    for label in LABELS:
        bind(fixtures[label])
        first, second = snapshot(), snapshot()
        report["validation"][label] = {}
        for name, value in first.items():
            finite, repeat = bool(np.isfinite(value).all()), bool(np.array_equal(value, second[name]))
            assert finite and repeat, (views, label, name, finite, repeat)
            np.save(directory / f"{label}.{name}.npy", value)
            report["validation"][label][name] = {"finite": finite, "repeat_exact": repeat}
        assert np.all(np.linalg.norm(first["poses"][:, 3:], axis=1) > 0)
        assert np.isfinite(np.exp(first["scale"])).all()

    # Ensure a changed last view influences the first view or global scale.
    bind(fixtures["capture1"])
    reference = snapshot()
    with torch.cuda.stream(stream):
        changed = {name: value[:views].clone() for name, value in fixtures["capture1"].items()}
        for name in changed:
            changed[name][-1].copy_(fixtures["capture2"][name][views - 1])
    stream.synchronize()
    bind(changed)
    actual = snapshot()
    pose_delta = float(np.max(np.abs(actual["poses"][0] - reference["poses"][0])))
    scale_delta = float(np.max(np.abs(actual["scale"] - reference["scale"])))
    assert pose_delta > 0 or scale_delta > 0
    report["changed_last_view"] = {"reference_pose_max_abs_change": pose_delta,
                                   "scale_max_abs_change": scale_delta}

    bind(fixtures["capture1"])
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    (directory / "layers.json").write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    profiler = Profiler()
    context.profiler = profiler
    for _ in range(10):
        run()
        stream.synchronize()
    write_json(directory / "profile.json", [{"name": name, "average_ms": statistics.mean(values),
                                             "samples": len(values)} for name, values in profiler.samples.items()])
    times = [value for p in report["passes"] for value in p["gpu_ms"]]
    report.update(median_ms=statistics.median(times), p05_ms=float(np.quantile(times, .05)),
                  p95_ms=float(np.quantile(times, .95)))
    write_json(directory / "benchmark.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path,
                        default=ROOT / "onnx/MapAnything/updated_onnx/mapanything_fix_encoder_fmha.engine")
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs-per-pass", type=int, default=25)
    args = parser.parse_args()
    assert args.warmup > 0 and args.runs_per_pass > 0
    args.output.mkdir(parents=True, exist_ok=False)
    stream = torch.cuda.Stream()
    directory = args.features_dir or args.output / "features"
    with torch.inference_mode():
        fixtures = (load_features(directory, stream) if args.features_dir
                    else encode_features(args.encoder, directory, stream))
        gc.collect()
        torch.cuda.empty_cache()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
        assert engine is not None and engine.num_io_tensors == 4
        for i in range(engine.num_io_tensors):
            assert engine.get_tensor_dtype(engine.get_tensor_name(i)) == trt.float32
        summary = {
            "scope": "Token preparation + 16 Transformer blocks + raw pose/scale heads; no DPT or encoder",
            "engine": str(args.engine.resolve()), "feature_directory": str(directory.resolve()),
            "gpu": torch.cuda.get_device_name(), "tensorrt": trt.__version__, "torch": torch.__version__,
            "method": "CUDA events; GPU-resident buffers; no CUDA Graph or profiler during timing",
            "excluded": ["encoder", "FS", "preprocessing", "H2D/D2H", "load/allocation", "output decoding"],
            "fixture_note": "Two real captures; V>2 repeats and perturbs captures. Timing and wiring checks, not pose-accuracy validation.",
            "feature_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in directory.glob("*.npy") if any(p.name.endswith(f".{n}.npy") for n in FEATURES)},
            "engine_layers": engine.num_layers, "context_memory_bytes": engine.device_memory_size_v2,
            "results": {},
        }
        for views in (2, 3, 4, 5):
            summary["results"][str(views)] = benchmark(
                engine, fixtures, views, args.output / f"views_{views}", stream, args.warmup, args.runs_per_pass)
            write_json(args.output / "summary.json", summary)
        print(json.dumps({v: r["median_ms"] for v, r in summary["results"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()

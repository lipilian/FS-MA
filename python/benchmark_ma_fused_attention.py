"""Build and measure an independent MA encoder with native TensorRT attention.

Reads the existing fixed single-view ONNX and engine without modifying them.
Replaces the 24 exported SDPA subgraphs in the parsed TensorRT network, then
compares GPU timing and feature outputs against the existing engine.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import onnx

from benchmark_ma_fused_encoder import ROOT, SOURCE, benchmark, digest, trt, write_json


def attention_boundaries(path):
    model = onnx.load(path, load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    infos = {v.name: v.type.tensor_type for v in model.graph.value_info}

    def producer(name, op):
        node = producers[name]
        assert node.op_type == op, (name, node.op_type, op)
        return node

    def attributes(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    boundaries = []
    for output in model.graph.node:
        if not output.name.startswith("node_scaled_dot_product_attention"):
            continue
        assert output.op_type == "CastLike"
        av = producer(output.input[0], "MatMul")
        softmax = producer(av.input[0], "Softmax")
        assert attributes(softmax)["axis"] == -1
        scale_mul = producer(softmax.input[0], "Mul")
        qk = producer(scale_mul.input[0], "MatMul")
        scale_node = producer(scale_mul.input[1], "Constant")
        scale = float(onnx.numpy_helper.to_array(attributes(scale_node)["value"]).item())
        transpose = producer(qk.input[1], "Transpose")
        assert attributes(transpose)["perm"] == [0, 1, 3, 2]
        casts = [producer(name, "Cast") for name in (qk.input[0], transpose.input[0], av.input[1])]
        assert all(attributes(node)["to"] == onnx.TensorProto.FLOAT for node in casts)
        q, k, v = [node.input[0] for node in casts]
        assert output.input[1] == q
        for name in (q, k, v, output.output[0]):
            value = infos[name]
            assert value.elem_type == onnx.TensorProto.BFLOAT16
            assert [d.dim_value for d in value.shape.dim] == [1, 24, 1148, 64]
        assert scale == 1 / math.sqrt(64) == 0.125
        boundaries.append({"name": output.name, "q": q, "k": k, "v": v,
                           "output": output.output[0], "scale": scale})
    assert len(boundaries) == 24, len(boundaries)
    return boundaries


def build_fused(path, output):
    boundaries = attention_boundaries(path)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    original_layers = [network.get_layer(i) for i in range(network.num_layers)]
    tensors = {layer.get_output(i).name: layer.get_output(i)
               for layer in original_layers for i in range(layer.num_outputs)
               if layer.get_output(i) is not None}
    # Keep the numpy storage alive until the build finishes.
    weights = np.full((1, 1, 1, 1), 0.125, dtype=np.float32)
    constant = network.add_constant(weights.shape, weights)
    constant.name = "fmha_query_scale_fp32"
    cast = network.add_cast(constant.get_output(0), trt.bfloat16)
    cast.name = "fmha_query_scale_bf16"
    scale = cast.get_output(0)
    for i, boundary in enumerate(boundaries):
        q, k, v = [tensors[boundary[name]] for name in ("q", "k", "v")]
        assert all(t.dtype == trt.bfloat16 and tuple(t.shape) == (1, 24, 1148, 64) for t in (q, k, v))
        consumers = [(layer, j) for layer in original_layers for j in range(layer.num_inputs)
                     if layer.get_input(j) is not None and layer.get_input(j).name == boundary["output"]]
        assert consumers, boundary
        scaled = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD)
        scaled.name = f"dino_{i:02d}_query_scale"
        attention = network.add_attention_v2(scaled.get_output(0), k, v,
                                            trt.AttentionNormalizationOp.SOFTMAX,
                                            trt.CausalMaskKind.NONE)
        assert attention is not None
        attention.name = f"dino_{i:02d}_native_attention"
        attention.decomposable = False
        replacement = attention.get_output(0)
        replacement.name = boundary["output"] + "_fmha"
        assert replacement.dtype == tensors[boundary["output"]].dtype
        assert tuple(replacement.shape) == tuple(tensors[boundary["output"]].shape)
        boundary["rewired_consumers"] = []
        for layer, index in consumers:
            layer.set_input(index, replacement)
            assert layer.get_input(index).name == replacement.name
            boundary["rewired_consumers"].append({"layer": layer.name, "input": index})
    write_json(output / "replacement.json", {"source_onnx": str(path), "attention_count": len(boundaries),
                                             "decomposable": False, "boundaries": boundaries})
    print(f"Rewired {len(boundaries)} attention subgraphs", flush=True)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    start = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT native attention build failed")
    engine_path = output / "fused_encoder_v1_fmha.engine"
    engine_path.write_bytes(bytes(serialized))
    write_json(output / "build.json", {"build_seconds": time.monotonic() - start,
                                      "engine_sha256": digest(engine_path),
                                      "builder_optimization_level": 3, "workspace_mib": 4096, "tf32": False})
    return engine_path


def verify_fusion(directory, expected):
    layers = json.loads((directory / "layers.json").read_text())["Layers"]
    profile = {x["name"]: x["average_ms"] for x in json.loads((directory / "profile.json").read_text())}
    fused = [x for x in layers if "mha" in x.get("TacticName", "").lower()]
    softmax = [x for x in layers if "node_Softmax_" in x.get("Metadata", "")]
    qk = [x for x in layers if x["LayerType"] == "gemm" and
          any(t["Dimensions"] == [24, 1148, 1148] for t in x.get("Outputs", []))]
    av = [x for x in layers if x["LayerType"] == "gemm" and
          any(t["Dimensions"] == [24, 1148, 1148] for t in x.get("Inputs", []))]
    assert len(fused) == expected, len(fused)
    assert len(softmax) == len(qk) == len(av) == (0 if expected else 24)
    return {"fused_mha_count": len(fused), "separate_qk_count": len(qk),
            "separate_softmax_count": len(softmax), "separate_av_count": len(av),
            "attention_core_ms": sum(profile[x["Name"]] for x in fused + qk + softmax + av),
            "profile_total_ms": sum(profile.values()),
            "fused_tactics": sorted({x["TacticName"] for x in fused})}


def compare_outputs(reference, candidate):
    report = {}
    for label in ("capture1", "capture2"):
        report[label] = {}
        for name in ("F_fused", "CLS"):
            original = np.load(reference / f"{label}.{name}.npy")
            actual = np.load(candidate / f"{label}.{name}.npy")
            assert actual.shape == original.shape and actual.dtype == original.dtype == np.float32
            assert np.isfinite(actual).all() and np.isfinite(original).all()
            a, b = actual.astype(np.float64).ravel(), original.astype(np.float64).ravel()
            delta = np.abs(a - b)
            report[label][name] = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                                   "exact_equal": bool(np.array_equal(actual, original)),
                                   "mean_abs": float(delta.mean()), "p99_abs": float(np.quantile(delta, .99)),
                                   "max_abs": float(delta.max()),
                                   "rmse": float(np.sqrt(np.mean(delta**2))),
                                   "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                                   "cosine_similarity": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                                   "reference_rms": float(np.sqrt(np.mean(b**2)))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "result/ma_fused_encoder_20261104")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.baseline = args.baseline.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    original_paths = [SOURCE, SOURCE.with_suffix(".engine"), SOURCE.with_suffix(".onnx.data"),
                      args.baseline / "fused_encoder_v1.onnx", args.baseline / "fused_encoder_v1.engine",
                      args.baseline / "mapanything_dynamic_raw_bf16.onnx.data"]
    hashes = {str(p): digest(p) for p in original_paths}
    write_json(args.output / "original_hashes.json", hashes)
    candidate = build_fused(args.baseline / "fused_encoder_v1.onnx", args.output)
    timings, inspections = {}, {}
    for name, engine, count in [("baseline", args.baseline / "fused_encoder_v1.engine", 0),
                                ("fmha", candidate, 24)]:
        directory = args.output / name
        directory.mkdir()
        print(f"Benchmark {name}", flush=True)
        timings[name] = benchmark(engine, directory, 1, args.baseline)
        assert all(v["finite"] and v["repeat_exact"] for fixture in timings[name]["validation"].values()
                   for v in fixture.values())
        inspections[name] = verify_fusion(directory, count)
    comparison = compare_outputs(args.output / "baseline", args.output / "fmha")
    write_json(args.output / "output_comparison.json", comparison)
    old_repeat = compare_outputs(args.baseline, args.output / "baseline")
    write_json(args.output / "baseline_repeat_comparison.json", old_repeat)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, f"Original file changed: {path}"
    baseline_ms, fused_ms = [timings[k]["median_ms"] for k in ("baseline", "fmha")]
    summary = {"scope": "Single-view DINO + geometry encoders + fusion -> F_fused and CLS",
               "gpu": timings["fmha"]["gpu"], "tensorrt": trt.__version__,
               "candidate_engine": str(candidate), "baseline_ms": baseline_ms, "fmha_ms": fused_ms,
               "saved_ms": baseline_ms - fused_ms, "speedup": baseline_ms / fused_ms,
               "latency_reduction_percent": 100 * (1 - fused_ms / baseline_ms),
               "method": timings["fmha"]["method"], "excluded": timings["fmha"]["excluded"],
               "inspection": inspections, "comparison": comparison,
               "original_files_unchanged": True,
               "limits": ["Two real capture fixtures; does not validate downstream pose/scale or reconstruction quality.",
                          "Different TensorRT builds may select different kernels outside attention as well.",
                          "Native BF16 fused attention is not expected to be bitwise identical to the original FP32 attention graph."]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

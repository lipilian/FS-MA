#!/usr/bin/env python3
"""Extract MA's dynamic V=2..5 pose/scale suffix, without rebuilding weights.

Run in the mapanything environment:
  python python/extract_ma_pose_scale.py

The two FP32 inputs are F_fused [V,1536,31,37] and CLS [V,1536,1].
Outputs are RAW poses [V,7] and scale [1,1,1], before the existing adaptors.
The output shares the encoder's external weight file. Existing files are never
overwritten. No attention, precision, weight, or prediction-head optimization
is performed here; the DPT branch is removed by output reachability.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "onnx/MapAnything/onnx/mapanything_dynamic_raw_bf16.onnx"
DESTINATION = ROOT / "onnx/MapAnything/updated_onnx"
BOUNDARIES = {"clone_7": "F_fused", "permute_51": "CLS"}
SHAPES = {
    "F_fused": ["num_views", 1536, 31, 37],
    "CLS": ["num_views", 1536, 1],
    "poses": ["num_views", 7],
    "scale": [1, 1, 1],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    tensor = value.type.tensor_type
    return tensor.elem_type, [d.dim_param if d.dim_param else d.dim_value for d in tensor.shape.dim]


def check_shape_rewire(original, rewritten, opsets):
    """Numerically check the only changed operation, without running attention."""
    def evaluator(node, input_name, shape):
        graph = onnx.helper.make_graph(
            [node], "view_count_check",
            [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("val_0", onnx.TensorProto.INT64, [1])],
        )
        return ReferenceEvaluator(onnx.helper.make_model(graph, opset_imports=opsets))

    old = evaluator(original, "inputs", ["num_views", 7, 434, 518])
    new = evaluator(rewritten, "F_fused", SHAPES["F_fused"])
    checks = {}
    for views in range(2, 6):
        # Broadcast views avoid allocating image/feature data: Shape reads only dimensions.
        scalar = np.zeros((), dtype=np.float32)
        a = old.run(None, {"inputs": np.broadcast_to(scalar, (views, 7, 434, 518))})[0]
        b = new.run(None, {"F_fused": np.broadcast_to(scalar, (views, 1536, 31, 37))})[0]
        require(np.array_equal(a, b) and a.tolist() == [views], f"View count mismatch for V={views}")
        checks[str(views)] = {"original": a.tolist(), "extracted": b.tolist(), "exact_equal": True}
    return checks


def extract(source, output, weights):
    report_path = output.with_suffix(".extraction.json")
    for path in (output, report_path):
        require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite: {path}")
    require(weights.parent == output.parent, "Place the existing shared weight file beside the output ONNX")
    require(source.is_file() and weights.is_file(), "Source ONNX and shared weight file must exist")

    model = onnx.load(source, load_external_data=False)
    infos = {x.name: x for x in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    for old, new in BOUNDARIES.items():
        require(signature(infos[old]) == (onnx.TensorProto.FLOAT, SHAPES[new]), f"Unexpected boundary: {old}")
    for name in ("poses", "scale"):
        require(signature(infos[name]) == (onnx.TensorProto.FLOAT, SHAPES[name]), f"Unexpected output: {name}")

    # Keep inputs temporarily: the suffix reads only inputs.shape[0]. Checking
    # consumers prevents accidentally discarding a future dependency on pixels.
    subset = onnx.utils.Extractor(model).extract_model([*BOUNDARIES, "inputs"], ["poses", "scale"])
    consumers = [n for n in subset.graph.node if "inputs" in n.input]
    require(len(consumers) == 1, "Expected exactly one remaining consumer of inputs")
    shape = consumers[0]
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in shape.attribute}
    require(shape.name == "node_Shape_0" and shape.op_type == "Shape"
            and list(shape.output) == ["val_0"] and attrs == {"start": 0, "end": 1},
            "Expected inputs.shape[0], not an image-data dependency")
    original_shape = copy.deepcopy(shape)
    shape.input[0] = "clone_7"
    subset = onnx.utils.Extractor(subset).extract_model(list(BOUNDARIES), ["poses", "scale"])

    # Rename only the external interface, leaving the downstream graph intact.
    for node in subset.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = BOUNDARIES.get(name, name)
    for value in list(subset.graph.input) + list(subset.graph.output) + list(subset.graph.value_info):
        value.name = BOUNDARIES.get(value.name, value.name)
    require([x.name for x in subset.graph.input] == ["F_fused", "CLS"], "Unexpected extracted inputs")
    require([x.name for x in subset.graph.output] == ["poses", "scale"], "Unexpected extracted outputs")
    for value in list(subset.graph.input) + list(subset.graph.output):
        require(signature(value) == (onnx.TensorProto.FLOAT, SHAPES[value.name]), "Interface changed")

    originals = {n.name: n for n in model.graph.node}
    inverse = {new: old for old, new in BOUNDARIES.items()}
    for node in subset.graph.node:
        restored = copy.deepcopy(node)
        for i, name in enumerate(restored.input):
            restored.input[i] = inverse.get(name, name)
        if restored.name == original_shape.name:
            restored.input[0] = "inputs"
        require(restored.SerializeToString() == originals[node.name].SerializeToString(),
                f"Unexpected computation change: {node.name}")

    names = [x.name for x in subset.graph.initializer]
    prefixes = ("model.info_sharing.", "model.pose_head.", "model.scale_head.", "model.scale_token")
    require(all(name.startswith(prefixes) for name in names), "Unexpected encoder/DPT weights retained")
    blocks = {int(match.group(1)) for name in names
              if (match := re.match(r"model\.info_sharing\.self_attention_blocks\.(\d+)\.", name))}
    require(blocks == set(range(16)), "Expected all 16 Transformer blocks")
    require(all(any(name.startswith(prefix) for name in names) for prefix in prefixes), "Missing required weights")
    attention_count = sum(n.op_type == "Softmax" for n in subset.graph.node)
    require(attention_count == 16, "Expected 16 unchanged attention cores")

    external = [x for x in subset.graph.initializer if x.data_location == onnx.TensorProto.EXTERNAL]
    locations = {dict((f.key, f.value) for f in x.external_data)["location"] for x in external}
    require(len(locations) == 1, "Expected one original external weight file")
    source_weights = (source.parent / next(iter(locations))).resolve()
    tracked = {str(path): digest(path) for path in dict.fromkeys((source, source_weights, weights))}
    require(tracked[str(source_weights)] == tracked[str(weights)], "Shared weights differ from original data")
    originals_weights = {x.name: x for x in model.graph.initializer}
    for tensor in subset.graph.initializer:
        require(tensor.SerializeToString() == originals_weights[tensor.name].SerializeToString(),
                f"Weight changed during extraction: {tensor.name}")
        for field in tensor.external_data:
            if field.key == "location":
                field.value = weights.name

    rewritten_shape = next(n for n in subset.graph.node if n.name == original_shape.name)
    shape_checks = check_shape_rewire(original_shape, rewritten_shape, model.opset_import)
    subset.graph.name = "mapanything_dynamic_pose_scale"
    subset.doc_string = (
        "MA suffix: token preparation + 16 alternating-attention blocks + raw pose/scale heads. "
        "FP32 F_fused and CLS must share V=2..5 and view order; view 0 is the reference. "
        "No DPT, encoder, output adaptors, or fused-attention rewrite."
    )

    report = {
        "source": str(source), "output": str(output), "shared_weights": str(weights),
        "source_files_sha256": tracked, "source_nodes": len(model.graph.node),
        "extracted_nodes": len(subset.graph.node), "initializers": len(subset.graph.initializer),
        "external_initializers": len(external), "attention_blocks": attention_count,
        "inputs": {x.name: {"shape": SHAPES[x.name], "dtype": "float32"} for x in subset.graph.input},
        "outputs": {x.name: {"shape": SHAPES[x.name], "dtype": "float32"} for x in subset.graph.output},
        "supported_views": [2, 3, 4, 5],
        "view_constraints": "Both inputs must have the same V and view order; view 0 is the reference. Set TRT profiles to V=2..5.",
        "raw_outputs": True, "dpt_removed": True, "encoder_removed": True,
        "attention_fusion_applied": False,
        "retained_nodes_identical_after_undoing_interface_edits": True,
        "retained_weights_identical_except_external_location": True,
        "shape_rewire": {"node": original_shape.name, "old_input": "inputs", "new_input": "F_fused",
                         "start": 0, "end": 1, "numerical_checks": shape_checks},
        "full_forward_numerical_validation": "Not run; identical retained graph/weights and the sole Shape change checked separately.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the real weights. ONNX 1.23's checker rejects external-data
    # symlinks and multiply-linked files, including temporary aliases.
    with tempfile.NamedTemporaryFile(prefix=".ma_suffix_", suffix=".onnx", dir=output.parent) as model_file, \
            tempfile.NamedTemporaryFile(prefix=".ma_suffix_", suffix=".json", dir=output.parent) as report_file:
        staged = Path(model_file.name)
        onnx.save(subset, staged)
        onnx.checker.check_model(str(staged))
        report["onnx_checker_passed"] = True
        report["onnx_sha256"] = digest(staged)
        for name, expected in tracked.items():
            require(digest(Path(name)) == expected, f"Source changed: {name}")
        report["source_files_unchanged"] = True
        staged_report = Path(report_file.name)
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        # Same-filesystem hard links publish complete files without overwriting.
        report_path.hardlink_to(staged_report)
        try:
            output.hardlink_to(staged)
        except BaseException:
            report_path.unlink()
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DESTINATION / "mapanything_dynamic_pose_scale.onnx")
    parser.add_argument("--weights", type=Path, default=DESTINATION / "mapanything_dynamic_bf16.onnx.weight")
    args = parser.parse_args()
    # Resolve parents, but retain the output basename so dangling symlinks are
    # detected by the no-overwrite guard rather than followed to their targets.
    output = args.output.absolute().parent.resolve() / args.output.name
    report = extract(args.source.resolve(), output, args.weights.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

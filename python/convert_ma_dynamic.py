#!/usr/bin/env python3
"""Export dynamic V=2..5 MapAnything RAW heads, retaining FP32 or BF16 AMP.

From the repository root, in conda environment mapanything:
  python python/convert_ma_dynamic.py --validate-only --precision fp32
  python python/convert_ma_dynamic.py --precision fp32 --verify ort
  python python/convert_ma_dynamic.py --precision bf16 --verify tensorrt
  python python/convert_ma_dynamic.py --precision fp32 --verify-existing

ONNX needs onnx + onnxscript; ORT verification needs onnxruntime-gpu (or CPU
onnxruntime). TensorRT verification can use the local system Python bindings
via --trt-python-path. Both precisions keep FP32 external inputs and outputs.
No output activation or geometric reconstruction is part of this graph.
The .validation.pt companion holds test fixtures for --verify-existing; it is
not needed for deployment. Keep the .onnx and its external weight file together.
Current local validation: FP32 passes ORT; BF16 passes PyTorch parity but its
exported Dense output exceeds the 2e-2 tolerance in ORT and TensorRT 11.2.
Treat BF16 export as experimental until backend numerical validation passes.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys

os.environ["XFORMERS_DISABLED"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "map-anything"))

import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.inference import preprocess_input_views_for_inference
from ma_utils import prepare_inputs
from ma_dynamic import MapAnythingDynamicRaw, OUTPUT_NAMES


def pack_views(views):
    return torch.cat([
        torch.cat((v["img"], v["ray_directions_cam"].permute(0, 3, 1, 2),
                   v["depth_along_ray"].permute(0, 3, 1, 2)), dim=1)
        for v in views
    ], dim=0).float().contiguous()


def validate_inputs(x, height, width):
    if x.dtype != torch.float32 or x.ndim != 4 or tuple(x.shape[1:]) != (7, height, width):
        raise ValueError("Expected FP32 [V,7,H,W]")
    if not 2 <= x.shape[0] <= 5 or not torch.isfinite(x).all():
        raise ValueError("V must be 2..5 and input values must be finite")
    if (x[:, 6] < 0).any():
        raise ValueError("Ray depths must be nonnegative; zero denotes invalid depth")
    if not torch.allclose(torch.linalg.vector_norm(x[:, 3:6], dim=1),
                          torch.ones_like(x[:, 6]), rtol=1e-4, atol=1e-4):
        raise ValueError("Ray directions must be unit vectors")


def validation_cases(base):
    """Additional views are deterministic synthetic fixtures, NOT real captures."""
    for v in (2, 3, 4, 5):
        x = torch.cat([base[i % len(base):i % len(base) + 1].clone() for i in range(v)])
        for i in range(2, v):
            x[i, :3] += 0.01 * i
            x[i, 6] *= 1 + 0.025 * i
        yield f"views_{v}" + ("_real" if v == 2 else "_synthetic"), x
    x = base.clone()
    x[1, 6] = 0
    yield "zero_depth_view", x
    x = base.clone()
    x[:, 6, ::3, ::3] = 0
    yield "sparse_depth", x
    x = base.clone()
    x[1, :3] = x[1, :3].flip(-1) + 0.15
    x[1, 6] *= 1.2
    yield "changed_last_view", x


def raw_reference(model, x, precision, minibatch=1):
    """Capture original head outputs before adaptors; always remove hooks."""
    results = {k: [] for k in OUTPUT_NAMES}
    handles = []
    for name, head in zip(OUTPUT_NAMES, (model.dense_head, model.pose_head, model.scale_head)):
        def hook(module, args, output, key=name):
            results[key].append(output.decoded_channels.detach().float().cpu())
        handles.append(head.register_forward_hook(hook))
    old_config = dict(model.geometric_input_config)
    try:
        model._configure_geometric_input_config(True, True, True, True, True)
        views = [{
            "img": x[i:i+1, :3].contiguous(),
            "ray_directions_cam": x[i:i+1, 3:6].permute(0, 2, 3, 1).contiguous(),
            "depth_along_ray": x[i:i+1, 6:7].permute(0, 2, 3, 1).contiguous(),
            "is_metric_scale": torch.ones(1, device=x.device, dtype=torch.bool),
            "data_norm_type": [model.encoder.data_norm_type],
        } for i in range(x.shape[0])]
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            model(views, memory_efficient_inference=minibatch > 0, minibatch_size=minibatch or None)
        return {k: torch.cat(v, dim=0) for k, v in results.items()}
    finally:
        for handle in handles:
            handle.remove()
        model.geometric_input_config.clear()
        model.geometric_input_config.update(old_config)


class ValidationFailure(AssertionError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def compare(actual, reference, tolerance, label):
    stats = {}
    failures = []
    for name in OUTPUT_NAMES:
        a, b = actual[name].detach().float().cpu(), reference[name].detach().float().cpu()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError(f"{label}/{name}: non-finite output")
        error = (a - b).abs()
        stats[name] = {"max_abs": error.max().item(), "mean_abs": error.mean().item(), "shape": list(a.shape)}
        mismatched = error > (tolerance + tolerance * b.abs())
        stats[name]['mismatched_elements'] = int(mismatched.sum())
        stats[name]['total_elements'] = a.numel()
        try:
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
        except AssertionError as error:
            stats[name]['failed'] = str(error)
            failures.append(name)
    status = 'FAIL' if failures else 'PASS'
    print(f"{status} {label}: " + ", ".join(f"{k} max={v['max_abs']:.6g}" for k, v in stats.items()), flush=True)
    if failures:
        raise ValidationFailure(f"{label}: tolerance exceeded for {', '.join(failures)}", stats)
    return stats


def validate_forward(wrapper, base, tolerance, device):
    samples, report = [], {}
    for label, x_cpu in validation_cases(base):
        validate_inputs(x_cpu, wrapper.height, wrapper.width)
        x = x_cpu.to(device)
        reference = raw_reference(wrapper.model, x, wrapper.precision)
        with torch.no_grad():
            actual = {k: v.cpu() for k, v in wrapper(x).items()}
        v = x.shape[0]
        assert actual['depths'].shape == (v, 6, wrapper.height, wrapper.width)
        assert actual['poses'].shape == (v, 7)
        assert actual['scale'].shape == (1, 1, 1)
        report[label] = compare(actual, reference, tolerance, label)
        samples.append((label, x_cpu, actual))
    # Later views must influence the reference view, not just their own outputs.
    before, after = samples[0][2], samples[-1][2]
    if torch.equal(before['depths'][0], after['depths'][0]) and torch.equal(before['poses'][0], after['poses'][0]):
        raise AssertionError("Changing the last view did not affect reference-view predictions")
    return samples, report


def export_translations(precision):
    """Keep SDPA accumulation in FP32 rather than rounding every BF16 op.

    PyTorch fused SDPA accepts BF16 Q/K/V but uses higher-precision internal
    accumulation. The default opset-18 decomposition scales and materializes
    attention scores in BF16, introducing extra rounding. Cast only at the
    attention boundary; encoder/transformer projections remain BF16.
    """
    if precision != 'bf16':
        return None
    import inspect
    from onnxscript import opset18 as op
    from onnxscript.function_libs.torch_lib.ops.nn import aten_scaled_dot_product_attention

    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
             is_causal=False, scale=None, enable_gqa=False):
        if attn_mask is not None or dropout_p != 0.0 or is_causal or enable_gqa:
            raise ValueError('This export supports only the checkpoint unmasked inference SDPA')
        # Q/K/V have statically known head dimension in this checkpoint.
        factor = scale if scale is not None else float(query.shape[-1]) ** -0.5
        scores = op.MatMul(op.Cast(query, to=1),
                           op.Transpose(op.Cast(key, to=1), perm=[0, 1, 3, 2]))
        probabilities = op.Softmax(op.Mul(scores, factor), axis=-1)
        output = op.MatMul(probabilities, op.Cast(value, to=1))
        return op.CastLike(output, query)

    sdpa.__annotations__ = inspect.get_annotations(aten_scaled_dot_product_attention, eval_str=True)
    return {torch.ops.aten.scaled_dot_product_attention.default: sdpa}


def lower_fixed_sequences(path):
    """Replace fixed SwiGLU chunk sequences with exact slices for TensorRT.

    This deliberately does NOT run algebraic simplification (which removed
    nonzero epsilon constants in onnxscript 0.7.2). External weights are neither
    loaded nor rewritten. V is unaffected: these splits are on feature channels.
    """
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    producers = {name: node for node in nodes for name in node.output}

    def scalar(name):
        node = producers.get(name)
        if node is None or node.op_type != 'Constant':
            raise ValueError(f'Expected constant split size/index: {name}')
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        value = numpy_helper.to_array(attrs['value']) if 'value' in attrs else np.asarray(attrs['value_int'])
        if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f'Expected scalar integer: {name}')
        return int(value.reshape(-1)[0])

    splits = {n.output[0]: n for n in nodes if n.op_type == 'SplitToSequence'}
    if not splits:
        return 0
    for node in nodes:
        if any(name in splits for name in node.input) and node.op_type != 'SequenceAt':
            raise ValueError('Only constant-index SequenceAt consumers are supported')
    if any(o.name in splits for o in model.graph.output):
        raise ValueError('Sequence-valued public output is not supported')
    rewritten = []
    count = 0
    for node in nodes:
        if node.op_type == 'SplitToSequence':
            continue
        if node.op_type != 'SequenceAt' or node.input[0] not in splits:
            rewritten.append(node)
            continue
        split = splits[node.input[0]]
        attrs = {a.name: helper.get_attribute_value(a) for a in split.attribute}
        axis, keepdims = attrs.get('axis', 0), attrs.get('keepdims', 1)
        size = scalar(split.input[1])
        offset = scalar(node.input[1])
        if keepdims != 1 or size <= 0 or offset < 0 or axis != -1:
            raise ValueError('Unsupported sequence pattern: expected positive fixed feature-channel chunks')
        names = []
        for suffix, value in [('start', offset * size), ('end', (offset + 1) * size), ('axis', axis)]:
            name = f'{node.name}_fixed_{suffix}'
            names.append(name)
            rewritten.append(helper.make_node('Constant', [], [name], name=name,
                                               value=numpy_helper.from_array(np.array([value], dtype=np.int64))))
        rewritten.append(helper.make_node('Slice', [split.input[0], *names], list(node.output), name=node.name + '_slice'))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    keep_info = [v for v in model.graph.value_info if v.name not in splits]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_info)
    onnx.save_model(model, str(path))
    print(f'Lowered {count} fixed SequenceAt nodes to exact feature-channel slices.', flush=True)
    return count


def fold_small_constants(path):
    """Evaluate only small, fully constant subgraphs; never simplify x+eps."""
    import onnx
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
    model = onnx.load(str(path), load_external_data=False)
    values, nodes, folded = {}, [], 0
    # Do not read initializers: the large checkpoint remains in external data.
    supported = {'Reshape', 'Cast', 'Concat', 'Unsqueeze', 'Squeeze', 'Slice',
                 'Gather', 'Add', 'Sub', 'Mul', 'Div', 'Shape', 'Transpose'}
    opsets = {op.domain: op.version for op in model.opset_import}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if 'value' in attrs:
                value = numpy_helper.to_array(attrs['value'])
                if value.nbytes <= 8192:
                    values[node.output[0]] = value
            else:
                for key, dtype in [('value_int', np.int64), ('value_ints', np.int64),
                                   ('value_float', np.float32), ('value_floats', np.float32)]:
                    if key in attrs:
                        values[node.output[0]] = np.asarray(attrs[key], dtype=dtype)
                        break
            nodes.append(node)
        elif (node.op_type in supported and node.domain == ''
              and all(name in values for name in node.input)):
            outputs = ReferenceEvaluator(node, opsets=opsets).run(None, {n: values[n] for n in node.input})
            if any(x.nbytes > 8192 for x in outputs):
                nodes.append(node)
                continue
            for name, value in zip(node.output, outputs, strict=True):
                values[name] = value
                nodes.append(helper.make_node('Constant', [], [name], name=node.name + '_constant',
                                              value=numpy_helper.from_array(value)))
            folded += 1
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.save_model(model, str(path))
    print(f'Folded {folded} small constant nodes without algebraic rewrites.', flush=True)


def lower_bf16_convolutions(path):
    """Opset 18 Conv excludes BF16: retain BF16 boundaries, accumulate in FP32.

    Inputs/weights were already rounded to BF16 by autocast before these casts.
    This keeps a standard ONNX graph without upgrading its requested opset.
    """
    import onnx
    from onnx import helper
    model = onnx.load(str(path), load_external_data=False)
    types = {v.name: v.type.tensor_type.elem_type for v in
             list(model.graph.value_info) + list(model.graph.input)}
    nodes, count = [], 0
    for node in model.graph.node:
        if node.op_type != 'Conv' or types.get(node.input[0]) != onnx.TensorProto.BFLOAT16:
            nodes.append(node)
            continue
        for index, name in enumerate(list(node.input)):
            cast = f'{node.name}_fp32_input_{index}'
            nodes.append(helper.make_node('Cast', [name], [cast], name=cast, to=onnx.TensorProto.FLOAT))
            node.input[index] = cast
        original_output = node.output[0]
        node.output[0] = node.name + '_fp32_output'
        nodes.append(node)
        nodes.append(helper.make_node('Cast', [node.output[0]], [original_output],
                                      name=node.name + '_bf16_output', to=onnx.TensorProto.BFLOAT16))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.save_model(model, str(path))
    print(f'Lowered {count} BF16 Conv nodes to opset-18-compatible FP32 accumulation.', flush=True)


def preserve_bf16_fused_ops(path):
    """Preserve single rounding for PyTorch BF16 linear+bias and SiLU.

    Separate BF16 MatMul/Add or Sigmoid/Mul nodes introduce an extra rounding
    absent from the fused PyTorch operators. Use BF16 Gemm for biased linears
    and FP32 SiLU arithmetic with a BF16 output boundary.
    """
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    types = {v.name: v.type.tensor_type.elem_type for v in model.graph.value_info}
    producers = {o: n for n in nodes for o in n.output}
    uses = {}
    for n in nodes:
        for x in n.input:
            uses[x] = uses.get(x, 0) + 1
    replace, remove = {}, set()
    linear_count, silu_count = 0, 0
    for node in nodes:
        out = node.output[0]
        if types.get(out) != onnx.TensorProto.BFLOAT16:
            continue
        prefix = node.name + '_fused'
        new = []
        def constant(suffix, data):
            name = prefix + suffix
            new.append(helper.make_node('Constant', [], [name], name=name,
                                        value=numpy_helper.from_array(np.array(data, dtype=np.int64))))
            return name
        if out.startswith('linear') and node.op_type == 'Add':
            mm = producers.get(node.input[0])
            if mm is None or mm.op_type != 'MatMul' or uses[mm.output[0]] != 1:
                continue
            x, weight = mm.input
            bias = node.input[1]
            begin, end, last = constant('_begin', [0]), constant('_end', [-1]), constant('_last', [-1])
            flat, shape, leading = prefix + '_flat', prefix + '_shape', prefix + '_leading'
            ws, channels, target, gemm = (prefix + k for k in ('_ws', '_channels', '_target', '_gemm'))
            new.extend([
                helper.make_node('Flatten', [x], [flat], name=flat, axis=-1),
                helper.make_node('Shape', [x], [shape], name=shape),
                helper.make_node('Slice', [shape, begin, end], [leading], name=leading),
                helper.make_node('Shape', [weight], [ws], name=ws),
                helper.make_node('Gather', [ws, last], [channels], name=channels, axis=0),
                helper.make_node('Concat', [leading, channels], [target], name=target, axis=0),
                helper.make_node('Gemm', [flat, weight, bias], [gemm], name=gemm),
                helper.make_node('Reshape', [gemm, target], [out], name=prefix + '_reshape'),
            ])
            remove.add(mm.name)
            linear_count += 1
        elif out.startswith('silu') and node.op_type == 'Mul':
            sigmoid = producers.get(node.input[1])
            x = node.input[0]
            if sigmoid is None or sigmoid.op_type != 'Sigmoid' or list(sigmoid.input) != [x] or uses[sigmoid.output[0]] != 1:
                continue
            x32, sigmoid32, y32 = (prefix + k for k in ('_x32', '_sigmoid32', '_y32'))
            new.extend([
                helper.make_node('Cast', [x], [x32], name=x32, to=onnx.TensorProto.FLOAT),
                helper.make_node('Sigmoid', [x32], [sigmoid32], name=sigmoid32),
                helper.make_node('Mul', [x32, sigmoid32], [y32], name=y32),
                helper.make_node('Cast', [y32], [out], name=prefix + '_output', to=onnx.TensorProto.BFLOAT16),
            ])
            remove.add(sigmoid.name)
            silu_count += 1
        else:
            continue
        replace[node.name] = new
    rewritten = []
    for node in nodes:
        if node.name not in remove:
            rewritten.extend(replace.get(node.name, [node]))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    onnx.save_model(model, str(path))
    print(f'Preserved BF16 fused semantics: {linear_count} biased linears, {silu_count} SiLU operators.', flush=True)


def verify_ort(path, samples, tolerance):
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    providers = [('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    print('ORT providers:', session.get_providers(), flush=True)
    report = {}
    for label, x, reference in samples:
        values = session.run(list(OUTPUT_NAMES), {'inputs': x.numpy()})
        try:
            report[label] = compare(dict(zip(OUTPUT_NAMES, map(torch.from_numpy, values))), reference, tolerance, 'ORT/' + label)
        except ValidationFailure as error:
            report[label] = error.report
    if any('failed' in output for case in report.values() for output in case.values()):
        raise ValidationFailure('ORT numerical verification failed; see per-case report', report)
    return report


def verify_tensorrt(path, samples, tolerance, python_path, existing_engine=None):
    if python_path:
        sys.path.append(str(python_path))
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    if existing_engine is not None:
        serialized = existing_engine.read_bytes()
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(path)):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        profile = builder.create_optimization_profile()
        h, w = samples[0][1].shape[-2:]
        profile.set_shape('inputs', (2, 7, h, w), (3, 7, h, w), (5, 7, h, w))
        config.add_optimization_profile(profile)
        print('Building TensorRT validation engine (V min/opt/max=2/3/5) ...', flush=True)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT engine build failed')
        engine_path = path.with_suffix('.engine')
        engine_path.write_bytes(bytes(serialized))
        # Free parser/network weights before running the engine.
        del parser, network, builder, config
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('TensorRT engine deserialization failed')
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    report = {}
    for label, cpu_input, reference in samples:
        x = cpu_input.cuda().contiguous()
        if not context.set_input_shape('inputs', tuple(x.shape)):
            raise RuntimeError(f'TensorRT rejected shape {tuple(x.shape)}')
        outputs = {}
        context.set_tensor_address('inputs', x.data_ptr())
        for name in OUTPUT_NAMES:
            if engine.get_tensor_dtype(name) != trt.float32:
                raise TypeError(f'Expected FP32 TensorRT output: {name}')
            outputs[name] = torch.empty(tuple(context.get_tensor_shape(name)), device='cuda', dtype=torch.float32)
            context.set_tensor_address(name, outputs[name].data_ptr())
        stream.wait_stream(torch.cuda.current_stream())
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        try:
            report[label] = compare(outputs, reference, tolerance, 'TensorRT/' + label)
        except ValidationFailure as error:
            report[label] = error.report
    if any('failed' in output for case in report.values() for output in case.values()):
        raise ValidationFailure('TensorRT numerical verification failed; see per-case report', report)
    return report


def run_backend(args, samples, tolerance, metadata):
    try:
        if args.verify == 'ort':
            metadata['backend_validation'] = verify_ort(args.output, samples, tolerance)
        elif args.verify == 'tensorrt':
            metadata['backend_validation'] = verify_tensorrt(args.output, samples, tolerance, args.trt_python_path, args.trt_engine)
    except Exception as error:
        metadata['backend_validation'] = {'failed': str(error)}
        if isinstance(error, ValidationFailure):
            metadata['backend_validation']['cases'] = error.report
        raise
    finally:
        metadata.setdefault('backend_validations', {})[args.verify] = metadata['backend_validation']
        metadata['backend'] = args.verify
        args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'onnx/MapAnything')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--captures', nargs=2, default=('capture1', 'capture2'))
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--height', type=int, default=434)
    p.add_argument('--width', type=int, default=518)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--verify-existing', action='store_true', help='Validate an existing ONNX against its saved PyTorch fixtures, without exporting again')
    p.add_argument('--verify', choices=('auto', 'ort', 'tensorrt', 'none'), default='auto')
    p.add_argument('--trt-engine', type=Path, help='Explicitly validate an existing engine instead of rebuilding; must match this ONNX')
    p.add_argument('--trt-python-path', type=Path, default=Path('/usr/lib/python3.12/dist-packages'))
    args = p.parse_args()
    if any(x <= 0 or x % 14 for x in (args.height, args.width)):
        p.error('H/W must be positive multiples of 14')
    if args.validate_only and args.verify_existing:
        p.error('--validate-only and --verify-existing cannot be combined')
    if args.verify_existing and args.verify == 'none':
        p.error('--verify-existing requires a runtime backend')
    if args.precision == 'bf16' and not args.device.startswith('cuda'):
        p.error('BF16 validation/export targets CUDA autocast')
    args.output = args.output or ROOT / f'onnx/mapanything_dynamic_raw_{args.precision}.onnx'
    if args.output.suffix != '.onnx':
        p.error('--output must end in .onnx')
    if args.verify == 'auto':
        args.verify = 'ort' if args.precision == 'fp32' else 'tensorrt'
    if args.trt_engine is not None and (not args.verify_existing or args.verify != 'tensorrt'):
        p.error('--trt-engine requires --verify-existing --verify tensorrt')
    return args


def main():
    args = parse_args()
    tolerance = 1e-4 if args.precision == 'fp32' else 2e-2
    if not args.validate_only:
        for dependency in ('onnx.checker', 'onnxscript'):
            try:
                importlib.import_module(dependency)
            except ImportError as error:
                raise RuntimeError('Install export dependencies in this environment: python -m pip install onnx onnxscript') from error
        if args.verify == 'ort':
            try:
                importlib.import_module('onnxruntime')
            except ImportError as error:
                raise RuntimeError('ORT verification needs onnxruntime-gpu (CUDA) or onnxruntime (CPU); use --validate-only for PyTorch checks') from error
        elif args.verify == 'tensorrt':
            sys.path.append(str(args.trt_python_path))
            try:
                importlib.import_module('tensorrt')
            except ImportError as error:
                raise RuntimeError('TensorRT bindings unavailable; specify --trt-python-path or explicitly use --verify none for export only') from error
        if args.verify_existing and not args.output.is_file():
            raise FileNotFoundError(args.output)
        if args.output.exists() and not args.verify_existing:
            raise FileExistsError(f'{args.output} exists; choose another --output')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 support is required')
    print(f'Python: {sys.executable}\nTorch: {torch.__version__}\nPrecision: {args.precision}', flush=True)
    if args.verify_existing:
        metadata = json.loads(args.output.with_suffix('.json').read_text())
        if metadata['precision'] != args.precision or metadata['input']['shape'][2:] != [args.height, args.width]:
            raise ValueError('Existing artifact precision/shape differs from requested settings')
        samples = torch.load(args.output.with_suffix('.validation.pt'), map_location='cpu', weights_only=True)
        import onnx
        onnx.checker.check_model(str(args.output), full_check=True)
        run_backend(args, samples, metadata['validation_tolerance'], metadata)
        print('Existing ONNX runtime verification passed.', flush=True)
        return
    _, views, _ = prepare_inputs(args.data_root, args.captures, resize_mode='fixed_size', size=(args.width, args.height))
    base = pack_views(preprocess_input_views_for_inference(views))
    print('Loading checkpoint ...', flush=True)
    model = MapAnything.from_pretrained(str(args.checkpoint), local_files_only=True).float().to(args.device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wrapper = MapAnythingDynamicRaw(model, args.precision, args.height, args.width)
    samples, validation = validate_forward(wrapper, base, tolerance, args.device)
    if args.validate_only:
        print('All original/dynamic raw-head comparisons passed. No ONNX written.', flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.output.with_suffix('.validation.pt'))
    reports = args.output.parent / (args.output.stem + '_reports')
    # Export with an interior V value; the same graph must run at both boundaries.
    example = samples[1][1].to(args.device)
    with torch.no_grad():
        torch.onnx.export(wrapper, (example,), str(args.output),
                          input_names=['inputs'], output_names=list(OUTPUT_NAMES),
                          custom_translation_table=export_translations(args.precision),
                          dynamic_shapes={'inputs': {0: torch.export.Dim('num_views', min=2, max=5)}},
                          # onnxscript 0.7.2 optimization removes +1e-8 guards.
                          dynamo=True, opset_version=18, external_data=True, optimize=False,
                          report=True, artifacts_dir=str(reports))
    del example
    del wrapper, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import onnx
    lower_fixed_sequences(args.output)
    fold_small_constants(args.output)
    lower_bf16_convolutions(args.output)
    preserve_bf16_fused_ops(args.output)
    onnx.checker.check_model(str(args.output), full_check=True)
    graph = onnx.load(str(args.output), load_external_data=False)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    del graph
    metadata = {
        'precision': args.precision, 'external_dtype': 'float32',
        'input': {'name': 'inputs', 'shape': ['V', 7, args.height, args.width], 'views': [2, 5],
                  'channels': ['R_normalized', 'G_normalized', 'B_normalized', 'ray_x', 'ray_y', 'ray_z', 'depth_along_ray_m']},
        'outputs': {'depths': ['V', 6, args.height, args.width], 'poses': ['V', 7], 'scale': [1, 1, 1]},
        'raw_outputs': True, 'dense_channels': ['ray_x_raw', 'ray_y_raw', 'ray_z_raw', 'depth_raw', 'confidence_raw', 'mask_logit'],
        'pose_channels': ['tx_raw', 'ty_raw', 'tz_raw', 'qx_raw', 'qy_raw', 'qz_raw', 'qw_raw'],
        'scale_semantics': 'raw log scale before exp/clamp; no output adaptors applied',
        'reference_view': 0, 'checkpoint': str(args.checkpoint.resolve()),
        'torch_version': torch.__version__, 'opset': 18, 'exporter_optimize': False, 'fixed_sequences_lowered': True, 'small_constants_folded': True,
        'sdpa_accumulation': 'float32', 'conv_accumulation': 'float32', 'bf16_fused_ops_preserved': True,
        'validation_tolerance': tolerance, 'pytorch_validation': validation,
        'onnx_checker': 'passed', 'onnx_full_check': 'passed', 'backend': args.verify, 'backend_validation': 'not run',
        'fixtures': 'Two real captures; V>2 uses repeated views with deterministic perturbations, not real reconstruction validation.',
    }
    metadata_path = args.output.with_suffix('.json')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    run_backend(args, samples, tolerance, metadata)
    print(f'ONNX saved: {args.output}\nBackend verification: {args.verify}', flush=True)
    if args.verify == 'none':
        print('Backend verification was explicitly skipped; export is not runtime-validated.', flush=True)


if __name__ == '__main__':
    main()

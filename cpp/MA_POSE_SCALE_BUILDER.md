# MapAnything dynamic pose/scale builder

`ma_build_pose_scale` builds the extracted token preparation, 16 Transformer
blocks, and raw pose/scale heads. It replaces all eight global and eight per-view
attention cores with TensorRT `IAttention`. The encoder and DPT are excluded.

Build the C++ executable from the repository root:

```sh
cmake -S cpp -B cpp/build
cmake --build cpp/build --target ma_build_pose_scale -j4
```

For a new build directory without desktop/camera dependencies, configure with
`-DFS_BUILD_DESKTOP=OFF -DFS_BUILD_TESTS=OFF`. The executable itself links only
TensorRT, its ONNX parser, and CUDA Runtime; it needs no Python or FS plugin.

Build on the target GPU, using a new output name if an engine already exists:

```sh
cpp/build/ma_build_pose_scale \
  onnx/MapAnything/updated_onnx/mapanything_dynamic_pose_scale.onnx \
  onnx/MapAnything/updated_onnx/mapanything_dynamic_pose_scale_fmha.engine
```

The ONNX parser resolves `mapanything_dynamic_bf16.onnx.weight` relative to the
ONNX file. Neither source is modified. The engine contains its own weights.

| Tensor | Direction | Shape | Type |
| --- | --- | --- | --- |
| `F_fused` | Input | `[V,1536,31,37]` | FP32 |
| `CLS` | Input | `[V,1536,1]` | FP32 |
| `poses` | Output | `[V,7]` | FP32 |
| `scale` | Output | `[1,1,1]` | FP32 |

Both inputs must have the same V. One optimization profile covers V=2/3/5
(min/opt/max), including V=4. Defaults: 8192 MiB workspace, optimization level 3,
device 0, TF32 disabled, detailed layer information. Optional flags are
`--workspace-mb N`, `--optimization-level N` (0–5), and `--device N`.

The builder validates all 16 exported attention patterns before replacing any:
BF16 Q/K/V, FP32 casts, QK transpose, scalar scale 0.125, last-axis Softmax,
AV, and BF16 output. Global Q/K/V are `[1,24,1148*V+1,64]`; per-view Q/K/V are
`[V,24,1148,64]`. The parser's dynamic Softmax reshape is accepted only when
its shape input is proven to restore the Softmax input/output shape. Other
operations and weights are retained.

Each replacement scales BF16 Q by 0.125 and calls `addAttentionV2` with Softmax,
no causal mask, and `setDecomposable(false)`. An unsupported graph or fused
kernel configuration fails rather than falling back to separate attention
operations. Native BF16 attention is mathematically equivalent but need not be
bitwise identical to the original FP32 core.

Before saving, the builder deserializes the engine, verifies the input profile,
and checks the resolved output shapes for V=2,3,4,5. It writes the engine and
`<output.engine>.layers.json`, refusing to overwrite either. The layer report
comes from `IEngineInspector`; it is not required for inference. Kernel names
can vary by GPU and TensorRT version.

The outputs remain raw: `poses` holds tx,ty,tz,qx,qy,qz,qw; `scale` is one scene
log-scale. Existing postprocessing normalizes quaternions, computes
`max(exp(scale),1e-8)`, and multiplies translations by that value.

Benchmark with the cached baseline features and a new report directory:

```sh
python python/benchmark_ma_pose_scale.py \
  --engine onnx/MapAnything/updated_onnx/mapanything_dynamic_pose_scale_fmha.engine \
  --features-dir result/ma_pose_scale_baseline_20261109/benchmark/features \
  --output result/ma_pose_scale_fmha_check
```

This measures only the suffix with GPU-resident inputs. It excludes encoding,
preprocessing, transfers, allocation, output decoding, and CUDA Graph. Compare
saved pose/scale outputs as well as timings. The cached fixtures come from two
real captures, with repeated/perturbed views for V>2, and do not establish
reconstruction quality on independent multi-view scenes.

Local validation on 2026-11-09 (RTX PRO 4000 Blackwell, TensorRT 11.2.1.2):
all 16 cores compiled to `_gemm_mha_v2` kernels. Median suffix inference times
for V=2,3,4,5 were 31.29, 48.93, 68.38, 89.92 ms (baseline: 106.55, 146.12,
336.87, 501.39 ms). Context activation memory fell from 6203.19 to 235.45 MiB.
All view counts passed finite/repeatable-output and cross-view checks; the same
context also passed V=2,5,3,4,2 switching with exact agreement to fresh contexts.
On these fixtures, the maximum rotation difference from baseline was 0.0237
degrees and the maximum decoded-scale difference was 0.1501%. These comparisons
do not establish ground-truth pose accuracy. Existing engine/source hashes were
unchanged. Full reports are in `result/ma_pose_scale_fmha_20261109/` and the
updated model directory's README.

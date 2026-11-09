# MapAnything encoder builder

`ma_build_encoder` builds the extracted **single-view** encoder, including DINO,
geometry encoders, and feature fusion. It uses TensorRT 11's C++ API and ONNX
parser; it does not require Python, PyTorch, or the FoundationStereo plugin.

Build the executable from the repository root:

```sh
cmake -S cpp -B cpp/build
cmake --build cpp/build --target ma_build_encoder -j4
```

For a new build directory without the desktop/camera dependencies, configure with
`-DFS_BUILD_DESKTOP=OFF -DFS_BUILD_TESTS=OFF`. The existing CMake project still
requires the CUDA toolkit and OpenCV.

Build the engine on the target GPU:

```sh
cpp/build/ma_build_encoder \
  onnx/MapAnything/updated_onnx/mapanything_fix_encoder.onnx \
  onnx/MapAnything/updated_onnx/mapanything_fix_encoder_fmha.engine
```

The ONNX references `mapanything_dynamic_bf16.onnx.weight` in the same directory.
The parser resolves this name from the ONNX metadata. Both source files remain
unchanged. The resulting engine includes its weights and can be loaded directly
by the TensorRT runtime.

The default recipe matches the Python FMHA experiment: optimization level 3,
4096 MiB workspace, TF32 disabled, and detailed layer information. Optional flags
are `--workspace-mb N`, `--optimization-level N` (0–5), and `--device N`.
Use `--help` for usage. The tool refuses to overwrite either output file.

| Tensor | Shape | Type |
| --- | --- | --- |
| `inputs` | `[1,7,434,518]` | FP32 |
| `F_fused` | `[1,1536,31,37]` | FP32 |
| `CLS` | `[1,1536,1]` | FP32 |

Input channels are normalized RGB (3), unit camera rays (3), and distance along
the ray (1), as in the extracted model. Preprocessing is the caller's responsibility.

Before building, the tool checks this interface and matches all 24 exported SDPA
blocks. It verifies the casts, Q/K/V shapes `[1,24,1148,64]`, K transpose, Softmax
axis, and original scale `0.125`. It replaces the FP32 attention cores with native
BF16 `IAttention`, scaling Q once and disabling decomposition. Other model
operations and weights remain in the parsed graph. An unexpected graph or an
unsupported fused attention configuration fails the build instead of silently
selecting the old attention path.

After building, the tool deserializes the engine, checks its interface, and writes
`<output.engine>.layers.json` alongside the engine for inspecting fused kernels.
Kernel/tactic names can vary between GPU models and TensorRT versions. Build
engines on their target machines; using C++ does not make engines hardware-independent.

This command builds and inspects the engine; it does not measure inference or
validate feature accuracy. Native BF16 attention can differ numerically from the
original FP32 attention computation. Validate outputs and downstream pose/scale
on representative inputs before using a new build in the application.

Local validation on 2026-11-06 (RTX PRO 4000 Blackwell, TensorRT 11.2.1.2):
the engine contains 24 fused MHA kernels and no separate QK/Softmax/AV kernels.
Two real inputs passed finite-output and repeatability checks. Against the earlier
Python-built FMHA engine, CLS was bitwise identical and F_fused relative L2 error
was below `8e-8`. Median GPU inference time was 28.81 ms after 200 warmups and
200 measured runs, with GPU-resident buffers and no CUDA Graph. Preprocessing,
transfers, and downstream MA inference are excluded. Reports are saved under
`result/ma_cpp_encoder_20261106/` in the repository workspace.

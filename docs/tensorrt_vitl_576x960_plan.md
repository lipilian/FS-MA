# ViT-L 576×960、32 次迭代 ONNX/TensorRT 对照实验

## 结论与目标

NVIDIA 没有公开 `deployable_foundationstereo_small_576x960_v2.0.onnx` 的完整内部构建脚本。公开信息表明它是经过 TAO 适配的 ViT-Small 商业部署模型，并以不可训练的签名 ONNX 发布；TAO 的通用流程是 PyTorch checkpoint → ONNX export → TensorRT engine。[NGC 模型记录](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo/-/version-history)、[TAO 导出文档](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/depth_estimation/stereo_depth_estimation.html)。

可以使用相同类型的流程构建 ViT-L，但必须从 ViT-L checkpoint 重新导出，不能修改 small ONNX 得到 ViT-L。FoundationStereo 官方明确将 `23-51-11` 标为 ViT-L，并提供了对应的 ONNX/TensorRT 导出方法。[FoundationStereo 官方仓库](https://github.com/NVlabs/FoundationStereo)。

本轮目标改为：

- 使用本地 `23-51-11/model_best_bp2.pth`。
- 固定 batch 1、`576×960`、32 次 GRU、非 Hiera。
- 只构建 parity engine，不添加全局 `--fp16`。
- 与现有 PyTorch ViT-L baseline 比较完整耗时、四段耗时和整引擎显存。

## ONNX 导出

- 新增固定用途的 ViT-L exporter：

  - 在导入模型前设置 `XFORMERS_DISABLED=1`。
  - 显式设置 `vit_size=vitl`、`height=576`、`width=960`、`valid_iters=32`、`low_memory=0`。
  - 从 checkpoint 同目录加载 `cfg.yaml`，禁止使用不匹配的默认配置。
  - 使用 FP32 输入 `left/right [1,3,576,960]`，输出 `disp [1,1,576,960]`。
  - 使用模型已有 mixed-precision/autocast，ONNX opset 17、静态 batch/H/W、external-data 格式。
  - 不增加 debug intermediate outputs，避免改变 TensorRT fusion 和 tensor 生命周期。

- 导出后自动验证：

  - `onnx.checker` 通过。
  - 图中存在 DINO block 0–23。
  - 图中存在基础 update block 加 `_1`–`_31`，确认共 32 次 GRU。
  - 输入输出 shape、dtype 和名称完全符合预期。
  - 不允许 exporter 静默删除 GRU 或改为 initial disparity 输出。

## TensorRT 构建与正确性

- 使用 TensorRT 10.16 构建单一 parity engine：

  ```bash
  trtexec \
    --onnx=onnx/foundationstereo_vitl_576x960_iters32.onnx \
    --saveEngine=onnx/foundationstereo_vitl_576x960_iters32_parity.engine \
    --builderOptimizationLevel=3 \
    --profilingVerbosity=detailed \
    --skipInference
  ```

- 不使用 `--fp16`，保留 ONNX 已编码的 Cast 和 TensorRT 默认 TF32 行为。
- 保存完整 build log、ONNX/engine SHA256、TensorRT/driver/GPU 版本。
- 如果 build 失败，保留首个真实失败的 tactic/ForeignNode；不减少 GRU 次数、不删除 upsample、不偷偷改精度。该失败将作为直接 32 次路线的正式实验结果。
- 使用同一固定输入分别运行 PyTorch、ONNX Runtime CUDA 和 TensorRT：

  - 输出全部 finite。
  - TensorRT 相对 PyTorch 的平均绝对 disparity 误差不超过 `0.05 px`。
  - 误差大于 `1 px` 的像素比例不超过 `0.1%`。
  - 超出阈值时标记为非 parity，不发布速度提升结论。

## 性能与显存 Profiling

- 完整 inference：

  - GPU-resident input/output，不包含 H2D/D2H。
  - 5 秒 warmup、20 次测量、单 stream、3 个独立进程。
  - 报告 GPU Compute Time、Latency、Enqueue Time 的 mean、median、p90。
  - 普通执行为主结果；CUDA Graph 作为单独 enqueue 优化结果。

- 四段 layer-attributed time：

  - Feature/DINO。
  - Cost volume/3D aggregation。
  - 32 次 GRU refinement，同时报告总时间和 `/32` 单次均值。
  - Disparity upsample。
  - Context preparation、stem、跨阶段 fusion 和无法可靠归属的层进入 residual。
  - 四段加 residual 必须覆盖全部 TensorRT layer 且无重复。
  - 使用单独的 `--separateProfileRun`，按 profile 占比归一化到未插桩的 GPU Compute Time。

- 显存：

  - serialized engine 大小。
  - execution-context device memory。
  - 输入输出 buffer memory。
  - NVML 采集 TensorRT 进程峰值。
  - 用同一 NVML 方法重新测量 PyTorch 进程峰值，只有这两个进程级指标可以直接比较。
  - 保留现有 PyTorch `max_memory_allocated`，但不与 NVML/TensorRT managed memory 混算。
  - 不估算四段独立显存，因为 TensorRT 会提前分配 activation/workspace 并跨阶段复用。

## 页面与验收

- 将 `docs/index.html` 的 TensorRT 对照卡改为同一 ViT-L checkpoint、同一分辨率和同一 32 次迭代。
- NGC small 模型保留为独立部署参考，不参与 ViT-L 加速比。
- 三次 benchmark 的 GPU Compute Time 变异系数需不超过 3%。
- 页面必须同时显示四段、residual 和完整 forward，避免四段之和被误认为天然等于整引擎耗时。
- 明确说明 PyTorch 使用 xFormers，而 ONNX 导出必须禁用 xFormers；因此这是相同权重和模型拓扑的部署比较，但 attention kernel 实现不同。

# FS 桌面工作台 TODO

本文件保存已讨论的 UI 与 C++ 架构方案。按阶段逐步推进，完成并验证一项后再勾选；当前已完成阶段 1 的核心库构建拆分和代码目录整理，尚未开始 UI 功能实现。

## 已确定的方向

- 平台：仅 Linux 本地桌面应用。
- 技术路线：Qt 6 Widgets + VTK + 现有 C++ FoundationStereo TensorRT engine。
- 当前输入：导入 stereo pair 和已有标定文件。
- 未来输入：双目相机预览后拍摄一组，冻结图像，再重建和测量；目前没有相机硬件。
- 测量交互：在校正后的左图编辑 2D mask，联动 3D 点云、mesh 和曲面面积。
- 第一版默认支持一个无孔多边形区域；不包含自动分割、相机标定向导、连续实时重建和多视角融合。
- Python 保留作为算法实验和结果对照；实际应用以 C++ 为主。

## 当前代码基础

- `cpp/app/TSFS.cpp` 已完成文件输入、双目校正和 FS engine 推理的串联。
- `cpp/include/fs/inference/FS.hpp` 中的 `FS::inference()` 提交 GPU 工作，视差保留在内部 GPU 缓冲区，尚需补充结果读取接口。
- `StereoFrame` 当前通过文件路径加载图像和标定，后续需要支持内存输入。
- `python/fs_high_resolution_utils.py` 已有点云过滤、受约束 mesh 生成和面积计算，可作为 C++ 实现的参考。
- `docs/index.html` 继续作为开发文档；桌面应用单独建立入口。

## UI 工作台

保持在同一个窗口中操作，方便对照原图与三维结果，按阶段开放相关操作。

```text
┌──────────────────────────────────────────────────────────────┐
│ 导入图像组  相机预览  拍摄  开始重建  停止  保存结果            │
├─────────────┬───────────────────────────────┬────────────────┤
│ 输入与设置  │ 双目检查 / 区域测量 / 3D 浏览  │ 测量           │
│             │                               │                │
│ 当前图像组  │ ┌────────────┬──────────────┐ │ 多边形选区     │
│ 左右缩略图  │ │ 左图       │ 右图         │ │ 编辑 / 清空    │
│ 标定信息    │ │ 测量时叠加 │ 双目检查时   │ │                │
│             │ │ mask       │ 显示         │ │ 曲面面积 cm²   │
│ Engine 状态 │ └────────────┴──────────────┘ │ 有效点数       │
│             │ ┌───────────────────────────┐ │ 三角形数量     │
│ 高级参数    │ │ 3D：点云 / mesh / 线框     │ │                │
│ 深度范围    │ │ 旋转、缩放、平移、重置视角 │ │ 导出           │
│ 过滤阈值    │ └───────────────────────────┘ │                │
├─────────────┴───────────────────────────────┴────────────────┤
│ 当前阶段：重建 mesh…       阶段耗时       可展开日志           │
└──────────────────────────────────────────────────────────────┘
```

- 导入 capture 目录中的 `left.png`、`right.png`、`calibration.json`，展示原图、校正图和辅助水平线。
- 重建按校正、推理、点云生成、mesh 生成的顺序运行，完成后进入三维浏览。
- 在校正左图绘制多边形，通过拖动顶点修改区域，联动显示对应点云、mesh 和曲面面积。
- 3D 视图支持颜色、点大小、线框切换和视角重置；无选区时展示全图有效重建，有选区时突出目标区域。
- 相机流程为预览左右图、拍摄并冻结一组、进入共用重建流程；没有硬件时提供文件回放，真实相机入口显示“未连接”。

## 建议代码布局

```text
FS_Engine/
├── cpp/
│   ├── CMakeLists.txt
│   ├── app/
│   │   ├── TSFS.cpp                 # 保留命令行入口
│   │   ├── desktop_main.cpp         # Qt 桌面入口
│   │   └── build_single_engine.cpp
│   ├── include/
│   │   └── fs/
│   │       ├── core/               # 公共数据类型、配置
│   │       ├── capture/            # 输入源接口
│   │       ├── stereo/             # 标定、校正
│   │       ├── inference/          # FS engine
│   │       ├── geometry/           # 点云、mesh、面积
│   │       ├── pipeline/           # 全流程编排与缓存
│   │       └── io/                 # 结果保存
│   ├── src/                        # 与 include/fs 对应
│   │   ├── core/
│   │   ├── capture/
│   │   ├── stereo/
│   │   ├── inference/
│   │   ├── geometry/
│   │   ├── pipeline/
│   │   └── io/
│   ├── ui/
│   │   ├── MainWindow.hpp/.cpp
│   │   ├── PipelineController.hpp/.cpp
│   │   ├── PipelineWorker.hpp/.cpp
│   │   ├── widgets/
│   │   │   ├── StereoImageView.hpp/.cpp
│   │   │   ├── MaskEditor.hpp/.cpp
│   │   │   ├── SceneView3D.hpp/.cpp
│   │   │   └── MeasurementPanel.hpp/.cpp
│   │   └── resources/
│   └── tests/
├── python/                         # 实验和算法对照
├── docs/                           # 开发文档
└── result/                         # 本地输出，忽略生成文件
```

目录随功能逐步建立，不要求一次性创建全部空目录。

- FS 放入 inference，StereoFrame 放入 stereo，Logger 放入 core；GWC 插件及 CUDA kernel 放入 inference/plugins，保持独立编译。
- 新增 `fs_core` 库，供 CLI 和桌面应用共同链接。
- 桌面目标命名为 `fs_desktop`，通过 `FS_BUILD_DESKTOP` 开关构建；Qt/VTK 依赖仅加入桌面目标。
- `MainWindow` 组织控件，按钮调用 `PipelineController`，由 controller 提交后台任务。
- 推理、几何处理和流程编排放入核心库，避免窗口类承担计算逻辑。

## 核心接口与数据流

```text
FileStereoSource / ReplayStereoSource / 未来 CameraStereoSource
                            ↓
                   StereoCapture
              左右 RGB + 标定 + 帧标识
                            ↓
          校正 → FS 推理 → 缓存视差与相机参数
                            ↓
            点云过滤 → MeshBuilder → 面积
                            ↓
                 2D / 3D 展示与导出
```

| 接口或类型 | 职责 |
| --- | --- |
| `IStereoSource` | 打开、关闭、提供预览及捕获一组双目图像 |
| `StereoCapture` | 持有左右图、标定、帧标识和可选时间戳 |
| `StereoFrame` 内存构造接口 | 接收图像和标定，使相机输入无需先写文件 |
| `InferenceResult` | 明确视差尺寸、相机参数和结果所有权 |
| `ReconstructionPipeline` | 执行完整重建，以及根据新 mask 更新测量 |
| `ReconstructionResult` | 提供点云、mesh、像素对应关系和面积 |
| `ResultWriter` | 导出图像、标定、mask、PLY 和测量参数 |

- 第一版为 FS 增加同步后读取自有 CPU 视差副本的接口，明确结果生命周期；后台线程完成读取，UI 不接触 CUDA 指针。后续可在核心库内部继续优化 GPU 数据流。
- 计算放入独立 worker，通过 Qt queued signals 将结果交回主线程；Qt 控件和 VTK 场景更新统一在主线程执行。
- 使用 `QVTKOpenGLNativeWidget` 嵌入三维视图，并配合 `vtkGenericOpenGLRenderWindow`。

## 测量与更新规则

- mask 编辑以校正左图为基准，映射到当前 `800×960` 推理网格时使用最近邻采样；相机内参同步缩放，baseline 保持米。
- 修改选区不重新运行 FS 推理。缓存视差；由于现有 Python 去噪依赖 mask，重新执行与选区相关的过滤、mesh 和面积计算。
- 修改输入、标定或 engine 时，使对应下游结果失效并重算，避免旧面积与新图像混用。
- 结束一次选区编辑后自动提交计算，合并待处理修改，仅展示最新版本结果；旧面积标记为“待更新”。
- 面积为有效三角形的三维曲面面积，内部保存 m²，界面显示 cm²；仅代表当前可见重建区域，不推算遮挡面。
- 点云显示可以抽稀，面积始终来自测量 mesh。
- 缺少标定、engine 加载失败、空选区或没有有效三角形时显示原因；没有有效 mesh 时不显示误导性的 `0 cm²`。
- C++ 几何模块以现有 Python 为基准，保留受约束三角化、边长和深度跳变过滤的语义；VTK 负责显示。
- 停止请求在安全阶段边界生效，不强行中断运行中的 GPU 操作。

## 分阶段实施清单

### 阶段 1：拆出核心库，保持 CLI 可用

- [x] 将现有共用代码整理成 `fs_core`，保留 GWC 插件和 engine builder 的独立目标。
- [x] 按职责迁移 FS、StereoFrame 等代码并更新 include 和 CMake。
- [ ] 增加图像及标定的内存输入接口。
- [ ] 增加同步后的视差结果读取接口，明确数据所有权。
- [ ] 验证现有 CLI 输入、校正和推理行为保持正常。

核心库构建拆分和目录迁移均已验证：Release 配置及全部目标构建成功，
`data/Volunteer2_lower/0` 的普通运行与 `--measure` 十次推理均正常退出。
目录迁移后的 12 个 C++ 文件经比对仅改变位置和 include 路径，算法逻辑保持不变；
后续内存输入与视差结果接口完成后仍需重新验证 CLI。

### 阶段 2：打通桌面重建

- [ ] 增加可选的 `fs_desktop` 目标和 Qt 主窗口。
- [ ] 完成 capture 目录导入、左右图预览、标定状态及双目校正检查。
- [ ] 建立 controller、worker 和 pipeline，显示阶段进度、耗时及错误。
- [ ] 接入真实 FS engine 推理，保证界面保持响应。
- [ ] 补齐 C++ 点云过滤、受约束 mesh 生成和面积计算模块。
- [ ] 接入 VTK 点云、mesh、线框显示和基本视角操作。
- [ ] 对照 Python 验证固定输入的几何结果。

### 阶段 3：加入选区测量与导出

- [ ] 实现校正左图上的单个无孔多边形选区、顶点编辑和清空。
- [ ] 实现图像显示坐标、校正图坐标与推理网格之间的正确映射。
- [ ] 根据 mask 更新点云、mesh 和面积，复用已缓存的视差。
- [ ] 合并快速编辑产生的待处理任务，阻止旧结果覆盖新结果。
- [ ] 显示曲面面积、有效点数、三角形数量和结果更新状态。
- [ ] 导出输入图像、标定、mask、点云 PLY、mesh PLY 和测量参数及面积。

### 阶段 4：验证未来相机输入接口

- [ ] 实现统一的 `IStereoSource` 和文件输入适配器。
- [ ] 实现文件回放，模拟双目预览和拍摄一组图像。
- [ ] 验证拍摄冻结后可进入现有重建与测量流程。
- [ ] 在真实硬件到位后，根据相机 SDK 实现采集适配器，并验证左右帧配对、时间戳和标定匹配。

## 验收清单

- [ ] 同一输入的桌面与 CLI 推理结果一致。
- [ ] 固定视差、mask 和参数，对照 Python 的点云、mesh 和面积；差异定位到过滤或三角化阶段。
- [ ] 使用已知尺寸平面验证面积及 m²/cm² 单位换算。
- [ ] 使用缩放图像验证选区映射和内参缩放。
- [ ] 修改 mask 不触发 FS 推理，连续编辑不被旧任务结果覆盖。
- [ ] 推理过程中界面可操作，停止在安全阶段边界生效。
- [ ] 缺少标定、左右尺寸不匹配、engine 加载失败和无有效 mesh 时显示清楚的错误信息。
- [ ] 无 GPU 或无 engine 时仍可打开界面并检查输入，明确显示重建不可用原因。

## 参考

- [VTK：QVTKOpenGLNativeWidget](https://vtk.org/doc/nightly/html/classQVTKOpenGLNativeWidget.html)
- [Qt：Threads and QObjects](https://doc.qt.io/qt-6/threads-qobject.html)

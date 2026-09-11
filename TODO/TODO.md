# FS 桌面工作台 TODO

本文件保存已讨论的 UI 与 C++ 架构方案。按阶段逐步推进，完成并验证一项后再勾选；当前已完成阶段 1 的核心库构建拆分、代码目录整理、内存输入接口，以及 inference 后融合有效视差筛选、深度阈值和 XYZ 计算的 CUDA kernel，以及 FS/CLI 接入；已接入独立持有的 XYZ CPU 下载和 Qt Jet 深度图展示。已接入 Sentech 双目采集，独立预览工具已按用户要求移除，Qt 6 标定和 reconstruction window 已实现顺序切换；第二窗口已接入文件导入、相机预览/冻结、双目校正、真实 FS 推理和 GPU XYZ，以及校正左图 mask 编辑。已接入 SAM 2.1 Hiera Large 框/点交互选区；CPU Triangle mesh、面积和 Qt OpenGL 三维显示已接入。真实标定板精度验收待进行。

文档维护约定：后续统一维护 `docs/` 下的 HTML 和本 TODO；功能与操作记录放在 `docs/index.html`，构建、代码结构与接口说明放在 `docs/code_structure.html`。

## 已确定的方向

- 平台：仅 Linux 本地桌面应用。
- 技术路线：Qt 6 Widgets + Qt OpenGLWidgets + 现有 C++ FoundationStereo TensorRT engine。
- Qt 应用由两个独立的主窗口顺序组成：窗口 1 专门完成 calibration，正常完成后才能创建并进入窗口 2；窗口 2 为下文的重建、3D 浏览和面积测量工作台。
- 当前实现支持文件导入或 Sentech 双目相机采集，可从已有 JSON 读取标定；Qt 窗口 1 已新增 ChArUco 标定求解、独立检查和保存。
- 相机输入：按用户纠正，左右绑定改为左 STC-MCS500U3V(21LJ548)、右 STC-MCS500U3V(21LJ530)。旧顺序标定文件不再通过桌面的身份校验，需按新顺序重新标定；新绑定的实体画面仍待现场确认。此前已验证单次拍摄和关闭后重新连接。采用独立连续流，尚未实现硬件同步。
- 测量交互：在校正后的左图编辑 2D mask，联动 3D 点云、mesh 和曲面面积。
- 第一版支持二值 mask 选区及孔洞；包含专用标定窗口，已加入 SAM 2.1 Hiera Large 提示式辅助选区；不包含无提示全自动分割、连续实时重建和多视角融合。
- Python 保留作为算法实验和结果对照；实际应用以 C++ 为主。

## 当前代码基础

- `cpp/app/TSFS.cpp` 已完成文件输入、双目校正、FS engine 推理和融合 XYZ 计算的串联；`compute_xyz_map()` 内部同步检查 GPU 错误后返回。
- `FS::compute_xyz_map(min_depth_m=0, max_depth_m=1)` 只读原始 `disparity_output_device_`，使用缩放后的内参与 baseline，在同一 stream 写入独立的 `xyz_map_device_`。输出为米制 FP32 `[800][960][3]`，无效/超范围点为 `(0,0,0)`；无额外视差副本或 `valid_mask`。CPU 读取和选区 mask 上传仍待实现。
- `StereoFrame` 已支持文件路径和内存 RGB 图像 + `StereoCalibration` 两种输入，共用校验并复制持有数据；完整标定类型位于 `cpp/include/fs/stereo/StereoCalibration.hpp`。
- `python/fs_high_resolution_utils.py` 已有点云过滤、受约束 mesh 生成和面积计算，可作为 C++ 实现的参考。
- `docs/index.html` 继续作为开发文档；桌面应用单独建立入口。

## Qt 双窗口流程

两个主窗口属于同一个桌面应用，按顺序进入；不做同时运行的两个工作台。

```text
启动 fs_gui
       ↓
窗口 1：CalibrationWindow（专用标定）
自动加载已有标定 → 用户选择“跳过标定，继续”
或：连接双目 → 设置标定板 → 采集样本 → 计算标定 → 可选独立检查 → 保存
       ↓ 用户选择复用已有标定，或新标定完成条件满足
结束标定窗口及其检测/求解任务，交接已确认的标定和相机会话
       ↓
InferenceSplashWindow：后台创建 FS → 加载 engine/context → 分配推理缓冲区
       ↓ 初始化成功（失败留在 splash，可换 engine 重试或退出）
窗口 2：ReconstructionWindow（原计划工作台）
导入／拍摄 → 校正 → FS inference → 点云／mesh → 选区与面积
```

- 第二个窗口只在第一个窗口正常完成且 splash 初始化 FS 成功后创建并启动，不能通过启动参数或普通关闭操作跳过标定流程。
- 标定与重建主窗口在原生窗口首次 Expose 后执行一次最大化，适配本机 GNOME/X11 首次显示丢失最大化状态的问题；splash 保持小窗口。实际桌面验证两窗口具有横向/纵向最大化标志，窗口外框与屏幕可用区域一致。
- 标定窗口点击关闭或取消代表取消流程，不自动打开第二个窗口；释放后台任务和相机资源后退出。
- “计算标定”用于求解参数，“完成标定并进入工作台”用于窗口切换，避免把求解成功和完成整个标定流程混为一谈。
- 新做标定的完成条件：有效标定已保存、当前相机身份与图像尺寸已确认、求解 Stereo RMS ≤ 1.0 px 且没有正在进行的采样/求解任务。Check · new pose 为可选项，不参与 Finish 门限。复用已有文件使用下面的独立跳过入口。
- 按用户最新要求，启动时先尝试上次成功保存/加载的文件，再查找工作目录及程序目录下的 `sentech_stereo_calibration.json` / `calibration.json`。有效文件自动加载，显示路径和“Skip calibration · Continue”入口；用户点击后可复用已有结果，无需连接相机或重新检查。加载本身不自动关闭第一窗口，也不伪造本次检查通过。相机已连接时，需等当前图像尺寸与文件匹配后才能继续。
- 窗口 1 只负责标定，不启动 FS engine 或 3D 重建；窗口 2 接收第一窗口确认的结果。

### 窗口 1：CalibrationWindow

参考 `/home/liu4000/Desktop/FS` 的 ImGui calibration 逻辑，以 Qt 重做交互和任务组织。参考代码：

- `apps/sentech_stereo/main.cpp`：标定面板、候选样本选择、样本历史、计算与检查按钮。
- `include/ffs_viewer/calibration/` 与 `src/calibration/`：`LiveCharucoDetector`、`StereoCharucoCalibrator`、`StereoRectifier` 的职责和算法。

```text
┌────────────────────────────────────────────────────────────────┐
│ Calibration · 连接/停止相机  加载标定  保存标定                  │
├────────────────┬────────────────────────────┬──────────────────┤
│ ChArUco 参数   │ 左相机预览  │ 右相机预览   │ 样本历史         │
│ Squares X / Y  │ marker 边框/ID + corner 叠加    │ 样本列表与计数   │
│ Square length  │ 原图 / 校正图与水平辅助线  │ 上一组 / 下一组  │
│ Marker length  │                            │ 删除选中样本     │
│ Dictionary     │                            │                  │
│ 应用参数       │                            │ 左右角点/匹配数  │
├────────────────┴────────────────────────────┴──────────────────┤
│ 采集一组标定样本  计算标定  检查标定                             │
│ 左 RMS / 右 RMS / Stereo RMS / 检查误差 / 当前状态              │
│                                  完成标定并进入工作台          │
└────────────────────────────────────────────────────────────────┘
```

- 连接已有 `SentechStereoSource`，默认曝光为 100000 µs（100 ms），仅在未连接时可编辑；显示固定左右身份、图像尺寸和连续预览；检测叠加可开关；原图同时显示 ArUco marker 边框/ID 和 ChArUco 角点/ID，样本预览保留相同叠加；只有 marker 而无 ChArUco 角点时也显示已检测框。
- 设置并保存 ChArUco 的格子数量、方格边长、marker 边长、字典。用户已确认实际标定板：横向 10 格、纵向 8 格，方格边长 66.5 mm，marker 边长 50.5 mm；作为新窗口的默认尺寸。界面以 mm 输入和显示，内部及保存数据统一使用米（0.0665 m / 0.0505 m）。棋盘图案区域为 665×532 mm（不含外围留白），内部棋盘角点为 9×7＝63 个。字典已由用户确认为 `DICT_4X4_250`。旧工程 `build/sentech_stereo_charuco.json` 和 `build/sentech_stereo_calibration.json` 中曾保存 marker 为 55.5 mm；用户已明确本次使用 50.5 mm，新窗口默认值以本次确认为准。
- 参考“采集一组”先收集 5 组候选再选一组的交互，保留候选进度、左右 marker/角点数和共同角点数。选取时使用当前采集模块的主机到达时间差，保留设备时间戳作记录；不能照搬旧代码对独立设备时钟直接相减的同步假设。
- 参考历史容量 20 组，支持列表选择、前后浏览和删除；达到容量时提示先删除或重采，不静默覆盖。标定板参数、左右身份或图像网格变化后，旧样本不能继续混用，已有结果需重新标定。
- 参考左右单目标定后固定内参求解双目外参的流程，显示左 RMS、右 RMS、stereo RMS 和有效样本数。旧算法最低要求每侧 3 组有效检测、至少 3 组各有 4 个共同角点；这是求解下限，不代表覆盖充分或测量精度合格。
- “检查标定”使用新采集的独立样本，显示共同角点数、左右及 stereo 重投影 RMS；同时提供校正图和极线辅助检查。求解与独立检查的 Left / Right / Stereo RMS 按固定 1.0 px 分别着色：≤1 为绿色，>1 为红色。独立检查为可选诊断，结果保留绿/红显示；Finish 只检查求解 stereo RMS，不要求执行独立检查或检查误差通过。
- 保存与现有 `StereoCalibration::from_file()` 兼容的六个矩阵字段，并保存标定板参数、质量指标、左右相机身份和图像尺寸等元数据。旧 JSON 缺少身份/尺寸时，不自动认定匹配。
- 标定结果中的矩阵要通过当前 `StereoFrame` 校验；迁移旧算法时核对旋转/平移的实际方向与单位，避免仅凭字段名称再做一次反转。
- 采样有超时与取消，检测/求解在后台执行；保存失败、角点不足、求解失败或质量不合格时留在第一个窗口并展示原因，不启动第二个窗口。

### 窗口 2：ReconstructionWindow

沿用原 TODO 的工作台功能和下方布局，接收第一个窗口交接的标定结果。窗口内对照原图与三维结果，按阶段开放操作。

```text
┌──────────────────────────────────────────────────────────────┐
│ 导入图像组  相机预览  拍摄  开始重建  停止  保存结果            │
├─────────────┬───────────────────────────────┬────────────────┤
│ 输入与设置  │ 双目检查 / 区域测量 / 3D 浏览  │ 测量           │
│             │                               │                │
│ 当前图像组  │ ┌────────────┬──────────────┐ │ SAM / 刷子选区 │
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
- 在校正左图用 SAM 框/点和刷子修改 mask 区域，联动显示对应点云、mesh 和曲面面积。
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
│   │       ├── stereo/             # 标定数据类型、图像校正
│   │       ├── calibration/        # ChArUco 检测、样本、求解和检查
│   │       ├── inference/          # FS engine
│   │       ├── geometry/           # 点云、mesh、面积
│   │       ├── pipeline/           # 全流程编排与缓存
│   │       └── io/                 # 结果保存
│   ├── src/                        # 与 include/fs 对应
│   │   ├── core/
│   │   ├── capture/
│   │   ├── stereo/
│   │   ├── calibration/
│   │   ├── inference/
│   │   ├── geometry/
│   │   ├── pipeline/
│   │   └── io/
│   └── ui/
│       ├── DesktopController.hpp/.cpp      # 控制两窗口顺序与共享会话
│       ├── CalibrationWindow.hpp/.cpp      # 窗口 1：标定
│       ├── CalibrationController.hpp/.cpp
│       ├── CalibrationWorker.hpp/.cpp
│       ├── InferenceSplashWindow.hpp/.cpp  # 两窗口之间的模型准备与失败重试
│       ├── ReconstructionWindow.hpp/.cpp   # 窗口 2：重建与测量
│       ├── PipelineController.hpp/.cpp
│       ├── PipelineWorker.hpp/.cpp
│       ├── widgets/
│       │   ├── StereoImageView.hpp/.cpp
│       │   ├── CalibrationSampleList.hpp/.cpp
│       │   ├── CalibrationQualityPanel.hpp/.cpp
│       │   ├── MaskEditor.hpp/.cpp
│       │   ├── SceneView3D.hpp/.cpp
│       │   └── MeasurementPanel.hpp/.cpp
│       └── resources/
├── python/                         # 实验和算法对照
├── docs/                           # 开发文档
└── result/                         # 本地输出，忽略生成文件
```

目录随功能逐步建立，不要求一次性创建全部空目录。

- FS 放入 inference，StereoFrame 放入 stereo，Logger 放入 core；GWC 插件及 CUDA kernel 放入 inference/plugins，保持独立编译。
- 新增 `fs_core` 库，供 CLI 和桌面应用共同链接。
- 桌面目标命名为 `fs_gui`，默认构建，通过 `FS_BUILD_DESKTOP` 开关控制（默认 `ON`，可设为 `OFF` 关闭）；先搭建 Qt 标定阶段，窗口 2 使用 Qt OpenGLWidgets 显示三维结果。Qt/OpenGL 依赖不进入 `fs_core`，标定模块按需增加 OpenCV aruco 依赖。
- `DesktopController` 管理唯一的相机会话和当前已确认标定，启动时只显示 `CalibrationWindow`，收到成功完成信号后先显示 `InferenceSplashWindow`，后台完成 FS 初始化后才启动 `ReconstructionWindow`。
- 第一窗口由 `CalibrationController` / `CalibrationWorker` 组织检测、采样、求解和检查；第二窗口继续使用 `PipelineController` / `PipelineWorker`。两窗口只组织控件和显示状态。
- 切换时停止并等待标定专用后台任务，交接独立持有的标定结果；两个窗口不得同时打开同一组相机或重复创建采集线程。
- 推理、几何处理和流程编排放入核心库，避免窗口类承担计算逻辑。
- 在 `cpp/include/fs/inference/SamSegmenter.hpp` 与 `cpp/src/inference/SamSegmenter.cpp` 已实现独立的 `fs::SamSegmenter`，管理 SAM encoder/decoder 两个 TensorRT engine、GPU 缓冲区和图像特征缓存；与 `FS` 分开管理模型，供后台 worker 调用。

## 核心接口与数据流

```text
窗口 1 输出：已确认标定 + 相机身份/图像尺寸 + 共享相机会话
                            ↓
窗口 2 输入：FileStereoSource / ReplayStereoSource（待实现） / SentechStereoSource
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
| `DesktopController` | 控制标定窗口到重建窗口的唯一转换入口，管理共享相机会话 |
| `CalibrationSessionResult`（拟新增） | 独立持有 `StereoCalibration`、标定板配置、质量检查结果、相机身份、图像尺寸和保存路径 |
| `IStereoSource` | 打开、关闭、提供预览及捕获一组双目图像 |
| `StereoCapture` | 持有左右图、标定、帧标识和可选时间戳 |
| `StereoFrame` 内存构造接口 | 接收图像和标定，使相机输入无需先写文件 |
| `InferenceResult` | 明确视差尺寸、相机参数和结果所有权 |
| `ReconstructionPipeline` | 执行完整重建，以及根据新 mask 更新测量 |
| `ReconstructionResult` | 提供点云、mesh、像素对应关系和面积 |
| `ResultWriter` | 导出图像、标定、mask、PLY 和测量参数 |

- 第一版为 FS 增加同步后读取自有 CPU 视差副本的接口，明确结果生命周期；后台线程完成读取，UI 不接触 CUDA 指针。后续可在核心库内部继续优化 GPU 数据流。
- 标定检测/求解及重建计算分别放入后台 worker，通过 Qt queued signals 将结果交回主线程；Qt 控件和 OpenGL 场景更新统一在主线程执行。两个阶段之间不得继续使用已关闭窗口的回调。
- 当前使用 `QOpenGLWidget` 嵌入三维视图，不依赖 VTK。

## 测量与更新规则

- mask 编辑以校正左图为基准，映射到当前 `800×960` 推理网格时使用最近邻采样；相机内参同步缩放，baseline 保持米。
- 修改选区或深度范围时，复用保留的原始视差，重算 XYZ 及依赖 mask 的去噪、mesh 和面积，无需重新 inference。融合 kernel 不修改视差，扩大范围时可以恢复先前被排除的点。
- 修改输入、标定或 engine 时，使对应下游结果失效并重算，避免旧面积与新图像混用。窗口 2 导入文件时也需核对其标定与图像来源，不能静默覆盖第一窗口确认的标定。
- 结束一次选区编辑后自动提交计算，合并待处理修改，仅展示最新版本结果；旧面积标记为“待更新”。
- 面积为有效三角形的三维曲面面积，内部保存 m²，界面显示 cm²；仅代表当前可见重建区域，不推算遮挡面。
- 点云显示可以抽稀，面积始终来自测量 mesh。
- 缺少标定、engine 加载失败、空选区或没有有效三角形时显示原因；没有有效 mesh 时不显示误导性的 `0 cm²`。
- C++ 几何模块以现有 Python 为基准，保留受约束三角化、边长和深度跳变过滤的语义；Qt OpenGLWidgets 负责显示。
- 停止请求在安全阶段边界生效，不强行中断运行中的 GPU 操作。

## 分阶段实施清单

### 阶段 1：拆出核心库，保持 CLI 可用

- [x] 将现有共用代码整理成 `fs_core`，保留 GWC 插件和 engine builder 的独立目标。
- [x] 按职责迁移 FS、StereoFrame 等代码并更新 include 和 CMake。
- [x] 增加图像及标定的内存输入接口。
- [x] 增加 XYZ 的 CPU 读取接口，返回独立持有的 `CV_32FC3` 数据，并保证下载完成和数据生命周期。
- [ ] 增加原始视差的 CPU 读取接口，返回独立持有的 `CV_32FC1` 数据。
- [x] 从 XYZ 提取 Z 通道，使用本次运行预设的米制深度范围生成 Jet colormap，无效点显示黑色；已直接接入 Qt 展示。CLI 深度 PNG 导出和原始 disparity 伪彩图仍未实现。
- [x] 验证 CLI 普通及 `--measure` 十次推理后均可完成融合 XYZ 计算。

**当前里程碑：有效视差筛选、深度阈值和 XYZ 计算已融合完成，FS/CLI 已接入。** 参考 `python/fs_tensorrt800x960_gwc_plugin.ipynb` 及 `postprocess_disparity_gpu()`，保留原始 disparity，不再就地清零视差。XYZ CPU 下载、Qt Jet 深度图、选区 mask 上传与 CUDA 邻域去噪已完成；CPU Triangle mesh 和面积也已完成；原始视差下载仍待实现，阶段 1 尚未全部完成。

- [x] 在 `PostProcessing.hpp` / `PostProcessing.cu` 中使用 `compute_xyz_map()` 替代原 `filter_disparity()`，一个 kernel 完成筛选、深度判断和 XYZ 计算。
- [x] FS 分配并复用 `xyz_map_device_`：连续 FP32 `[800][960][3]`，按 X,Y,Z 交错排列，单位米，占 9,216,000 字节；原始视差不变，无额外筛选视差或有效掩码缓冲区。
- [x] 每线程处理一个像素，800×960 使用 3000 个 block × 256 个线程，保留边界检查，无跨步循环。
- [x] 有效条件为有限且正的 disparity、可选 selection mask，以及始终要求 `u ≥ d`；已移除可见性开关。未设置 mask 的 FS/CLI 处理全图，Qt 重建始终传入第二步确认的 mask。
- [x] 使用模型网格内参和米制 baseline，计算 `Z = fx × baseline / d`、`X = (u - cx) × Z / fx`、`Y = (v - cy) × Z / fy`。
- [x] 深度阈值可配置，默认 `0 < Z ≤ 1 m`；保留 `Z > 0` 且 `min_depth_m ≤ Z ≤ max_depth_m`，要求阈值有限且 `0 ≤ min_depth_m < max_depth_m`。无效、超范围或非有限 XYZ 全部写 `(0,0,0)`，通过 `Z > 0` 判断有效。
- [x] `FS::compute_xyz_map()` 使用 inference 同一 stream，内部同步并检查错误；CLI 无需额外同步，XYZ 计算不计入 inference 耗时。
- [x] 不维护输入/推理就绪标志，由调用方保证加载 engine→设置相机参数→准备图像→inference→XYZ 的顺序；检查必要参数和 CUDA 错误。
- [x] 对照 notebook 的 `denoise=False` 结果验证融合输出；更换深度范围可重算 XYZ，复用缓冲区和原始视差，无需重新 inference。
- [x] 接入 FS 选区 mask 上传；Qt 第二步在校正左图绘制的 mask 直接最近邻缩放到 960×800，用于限定邻域去噪区域，无需重复校正；XYZ 先对全图计算。

融合 kernel 验证：关闭 Qt 的 Release 构建通过；24 组合成 GPU 输入（可见性筛选始终开启）与 notebook 的 `postprocess_disparity_gpu(denoise=False)` 加近距阈值结果一致（`rtol=2e-6, atol=1e-7`），覆盖 NaN/Inf、零/负值、mask、可见性边界、深度阈值端点、非整块尺寸和模型网格；23 组非法参数被拒绝。检查了 XYZ 全零无效点、重复计算清除旧输出、非有限投影结果、原始视差/selection mask 不变及非默认 stream 顺序。

FS/CLI 验证：`data/Volunteer2_lower/0` 普通及 `--measure` 路径通过。临时 FS 诊断验证内参缩放、XYZ 布局、CPU 数值参考、原始 disparity 不变、缓冲区复用和返回前同步。深度 0–1 m 保留 430654/768000 个点，改为 0.2–0.3 m 后无有效点，再改回 0–1 m 恢复 430654 个点，全程只执行一次 inference。测试保存在 `/tmp`，未新增仓库测试目标。

历史说明：此前独立视差筛选及就地实现已完成验证；现已按新要求替换为只读视差、输出 XYZ 的融合实现。此前还验证过 engine 重新加载和带待完成任务析构。原始视差的 CPU 下载、XYZ/Z 下载及显示仍待实现。

核心库构建拆分和目录迁移均已验证：Release 配置及全部目标构建成功，
`data/Volunteer2_lower/0` 的普通运行与 `--measure` 十次推理均正常退出。
目录迁移后的 12 个 C++ 文件经比对仅改变位置和 include 路径，算法逻辑保持不变；
内存接口完成后，Release 构建、CTest 和真实样本的普通/计时推理已通过。
当时的测试覆盖文件/内存校正一致性、图像和标定独立持有、非连续缓冲区、
浮点标定及行/列向量、无效输入和缺失 JSON 字段。真实样本首次对比出现过一次
校正后右图不一致；补充原始左右图检查与最大差异诊断后，连续五次复测通过，
首次差异原因尚未定位，后续回归时继续关注。上述记录属于早期验证；筛选接入后的 CLI 验证见本节新增记录，后续增加下载接口时仍需验证。

按用户要求已删除两个自动化测试源文件及 CMake 测试目标；上面的测试结果保留为历史验证记录。

### 阶段 2A：Qt 基础与窗口 1 标定（初版已实现）

- [x] 增加可选 `fs_gui` 目标和 `DesktopController`，启动时只创建标定窗口。
- [x] 先搭建 `CalibrationWindow` 的双目预览、标定板参数、样本列表、质量结果及操作区。
- [x] 复用现有 Sentech 采集接口接入预览，确保左右绑定与 RGB 显示正确。
- [x] 参考旧工程迁移 ChArUco 检测、样本管理、单目/双目标定和独立样本检查逻辑，核对 RGB/BGR 边界及外参方向。
- [x] 实现板参数应用、候选样本采集、样本查看/删除、求解、原图/校正图切换和质量显示。
- [x] 实现标定及元数据保存/加载，保持现有 `StereoCalibration` JSON 兼容。
- [x] 确认实际标定板参数：10×8 格，方格 66.5 mm，marker 50.5 mm，字典 `DICT_4X4_250`。
- [x] 记住标定文件路径、启动自动加载有效文件，提供显式跳过入口复用已有标定；新标定仍要求求解通过并保存，独立检查可选。路径记忆按用户要求存入 `/tmp/FS_Engine-<uid>/fs_gui.conf`（当前用户为 `/tmp/FS_Engine-1000/fs_gui.conf`），随系统 `/tmp` 策略清理；标定 JSON 仍保存在用户选择的位置。
- [x] 固定质量门限为 1.0 px，移除左侧阈值设置，质量误差值按阈值显示绿/红文字。
- [ ] 确认第二窗口对离线输入与已有标定匹配的验收方式。
- [x] 实现“完成标定并进入工作台”的状态条件；关闭/取消不进入第二窗口。
- [ ] 验证异常和窗口生命周期：采样取消、保存失败、相机掉线、求解失败时不会误跳转或遗留采集线程。

当前初版运行方式：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target fs_gui --parallel
./cpp/build/fs_gui
```

- 本机使用 Qt 6.4.2 + OpenCV 4.6.0；Qt 6 是明确依赖，桌面目标不链接 highgui/Qt 5。
- 当前提供两条完成路径：“Finish calibration”要求求解 Stereo RMS ≤ 1.0 px 并保存，且相机连接和图像尺寸匹配；独立检查可选；“Skip calibration · Continue”允许显式复用已加载的有效标定。两者都发出独立持有的标定结果，并转移当前相机源；标定 worker 完全停止后，`DesktopController` 先显示 splash；FS 初始化成功后才创建第二窗口。普通关闭仍释放相机并退出。
- RMS 门限按用户要求固定为 1.0 px；阈值仅在质量面板显示，不再提供设置。当前仅允许加载带完整身份/尺寸元数据且板参数匹配的标定；加载后可选择跳过复用，也可重新检查；新检查开始后会撤销当前复用资格，避免失败检查沿用旧状态。旧矩阵文件仍可用于 CLI。样本只保存在本次进程内存中。
- 验证记录：Release 全部目标构建成功；临时合成数据检查检出 63 个角点，求解及独立重投影检查通过，0.12 m 合成基线尺度/方向正确，JSON 与旧读取接口兼容，交接矩阵独立持有。
- 可选独立检查验证：`fs_gui` 构建通过；临时 Qt 回放完成 8 组采样 → Compute → Save → 点击 Finish，全程未执行 Check（solve stereo RMS 约 0.213 px，交接结果保持 `checked=false`）；未求解提前完成和保存失败仍被拒绝。
- 临时回放源已验证完整窗口操作链：8 组样本采集 → 求解 → 独立检查 → 保存 → 完成（合成图像 stereo RMS 约 0.213 px，检查约 0.075 px）；提前完成被拒绝、保存失败不退出。
- 真实双目 Qt 预览为 2448×2048，左右绑定正确；采样取消、断开重连及关闭释放线程已验证。当前画面未检测到实体板，真实标定精度和物理掉线仍待验证。没有重新加入之前删除的仓库测试文件/CTest 目标。

- Capture 闪回旧样本修复：禁用当前焦点按钮时，Qt 自动转移焦点到历史列表并选择第一行，导致 `currentRowChanged` 意外提交旧样本预览。`refreshActions()` 现屏蔽列表信号并保留原选中行。临时 Qt 合成板回放：连续三次 Capture 的历史预览次数由 3 降为 0，手动点击历史、键盘上下选择及 Return to live 均正常；Release 构建通过。

### 阶段 2B：窗口 2 重建与三维浏览

- [x] 新建 `ReconstructionWindow`，仅由窗口 1 正常完成后启动，接收确认标定和共享相机会话。
- [x] 在两窗口之间增加 `InferenceSplashWindow`，后台创建 FS、加载 TensorRT engine/context，并分配 GPU 输入/视差/XYZ、pinned host 和模型尺寸 CPU RGB 缓冲区；初始化成功后才打开重建窗口。
- [x] 完成 capture 目录导入、左右图预览、标定状态及双目校正检查。
- [x] 建立 `PipelineController` / `PipelineWorker`，串联现有校正、FS、GPU XYZ，显示阶段、耗时、日志及错误。
- [ ] 将后续完整几何流程提取为核心库 `ReconstructionPipeline`；目前 Qt worker 仅编排现有核心接口。
- [x] 接入真实 FS engine 推理，保证界面保持响应。
- [x] 复用阶段 1 的基础深度/XYZ 后处理，接入 Python 对应的 3×3 邻域三维距离去噪、CPU Triangle 受约束 mesh 和面积计算。
- [x] 使用 Qt OpenGLWidgets 接入 mesh 顶点、彩色三角面、线框及旋转/缩放/平移/重置。
- [x] 固定真实 XYZ/mask：CPU Triangle 与 Python 的顶点和三角面连接完全一致，面积为 0.0786497924673 m²。

当前窗口 2 操作与边界（2026-09-11）：

- [x] 第三步 Reconstruct 旁新增 Next：重建成功且结果仍有效时启用，进入第四步；修改上游输入/参数或任务运行时禁用。日志分别记录 input preparation、inference、post processing（XYZ/去噪及下载/深度图准备分项）和第四步 CPU mesh build，同步后的墙钟耗时以毫秒保留三位小数，不包含主线程渲染。

- 由标定的 Finish / Skip 进入；先停止标定检测/求解线程，再交接同一个运行中的相机源，不重复打开设备。退出重建窗口时在安全阶段边界停止，等待后台任务并释放 GPU / SDK。
- `Rectified` 同时控制实时预览和冻结图像：勾选时后台按确认标定做全分辨率 stereo rectification，再缩小显示；取消时显示 raw RGB。校正映射跨帧缓存，实时校正视图也支持 `Epipolar guides`；切换时清空旧预览并过滤旧模式帧，捕获仍保留原图供重建使用。
- 实时校正验证：`fs_gui` 构建通过；临时 Qt 回放以 2448×2048 图像验证实时输出与 `StereoFrame::rectify()` 后缩小的左右图逐像素一致，并覆盖 raw/rectified 切换、实时辅助线、旧模式帧过滤、映射复用、拍摄冻结、恢复预览和关闭。实体相机的预览帧率待现场确认。
- 左栏 `Input and calibration` 默认折叠，点击标题展开/收起；状态更新和导入不会自动展开，内部选项在折叠后保留。移除 `Camera session` 与 `FoundationStereo engine` 信息块，顶部导入、连接/断开、预览、拍摄操作保留。
- 相机曝光从标定阶段随会话交接，重建窗口不再提供曝光设置；断开重连使用标定阶段的请求曝光。未连接相机直接 Skip 时也传递左侧曝光值，标定 JSON 格式保持不变。
- Import capture 默认读取同目录 `left.png`、`right.png`、`calibration.json`；支持 CLI 的旧矩阵标定，并明确提示未验证身份/尺寸元数据。有尺寸元数据时必须匹配。取消 `Use capture calibration` 则使用交接的内存标定，并要求左右图尺寸匹配；不重新读取可能已更改的标定文件。
- 相机输入始终使用窗口 1 的确认标定。预览为原始 RGB；Capture pair 冻结并校正一组。预览不会覆盖已冻结输入；缺少新帧、掉线或尺寸不符会显示原因。文件导入失败保留上一组有效输入。
- Stereo inspection 支持原图/校正图与水平辅助线；Region measurement 将选区操作合并到 SAM 面板，支持框/点、刷子补选和擦除、Finish draw 及 Clear mask；已移除多边形入口。切换输入会清空旧选区。
- 二值 mask 按完整校正左图坐标导出 PNG；缩放窗口不改变顶点坐标。已上传到推理网格并限定 GPU 邻域去噪范围，mask 外 XYZ 原样保留；点云显示及面积尚待接入。
- Engine 由 splash 自动尝试默认路径，失败时可选择其他文件重试或退出。创建 FS、加载 engine/context 与缓冲区分配都在 `PipelineWorker` 的 QThread 完成；GPU 同步检查成功后，controller 交付就绪状态再创建工作台。无 engine/GPU 时停留在 splash，不打开未就绪的 reconstruction window。
- 工作台移除 FoundationStereo engine 信息块，后续推理复用 splash 已准备的同一个 FS、context 和缓冲区，不再加载模型。图像校正、每组输入的相机参数、缩放/打包/上传仍随实际图像执行；新增 CPU 模型尺寸缓冲区预分配不执行虚拟推理。当前流程包括 GPU XYZ、CPU 下载和固定米制范围的 Jet 深度图，后续几何功能仍标明待接入。
- `Region measurement` 位于第二个 tab，`Depth map` 位于第三个。Start reconstruction 完成后自动显示校正左图和 Jet 深度图，均为 960×800（宽×高）；左图使用与模型输入一致的线性缩放。Jet 使用本次运行的 Minimum / Maximum（米），无效及超范围点为黑色，标签保留本次范围；修改设置在下次运行生效。重新计算或成功换图会清空旧深度结果，导入失败保留原结果。顶部移除 Stop；右上角仅 Stereo inspection 显示 Capture pair，Region measurement 显示 Finish draw，Depth map 显示 Reconstruct。四个 tab 按输入 → 确认 mask → 重建深度 → 3D 顺序解锁，标题右侧绿勾表示完成、黄点表示待完成。成功拍摄/导入自动进入 Region measurement；Finish draw 确认 mask 后进入第三步，设置深度范围并点击 Reconstruct。编辑 mask 会清除后续 UI 完成状态；Minimum / Maximum 修改也会使深度及后续步骤失效、清空旧结果，改回原值仍需重算；只切换 tab 不会重置完成状态；3D 后端未接入，暂不标完成。
- 深度图验证：Release 全目标构建通过；临时 Qt offscreen + 真实 `data/Volunteer2_lower/0` 推理验证两张图均为 960×800，左图与模型网格校正 RGB 逐像素一致，Jet 与独立 FP32 深度参考逐像素一致（0–1 m 内 430654 个有效点），0.2–0.3 m 的空范围全黑。覆盖计算完成自动切页、图例保留本次范围、重算/换图清空旧结果、失败导入保留结果、选区跳转及 CPU 下载独立持有。测试及界面截图保存在 `/tmp`。
- 3D browser 已接入 CPU mesh、线框、顶点及面积/点数/三角形显示；邻域过滤已启用。PLY/测量报告/结果包未实现，Export 面板保持移除。
- 验证：Release 全部目标构建通过；临时 Qt offscreen 检查实际 Desktop 的 Skip → 唯一重建窗口 → 关闭，以及普通关闭标定不跳转。真实 `data/Volunteer2_lower/0` 校正与 CLI 逐像素一致，engine 推理和 GPU XYZ 成功，事件循环保持响应；覆盖导入失败、engine 错误、深度输入、取消、换图失效、mask 坐标/导出。临时回放源验证标定 worker 发出同一个运行中的相机源、线程结束后交接、pipeline 不重复 start、预览、拍摄冻结与断开；真实推理中取消和关闭均等待安全边界完成。测试保留在 `/tmp`，未新增仓库测试目标；真实设备交接及物理拔插仍需现场验收。

- Splash 验证：Release 全部目标构建通过；实际 Qt 顺序切换、缺少/损坏 engine、无 GPU、失败后换路径重试、加载中取消、标定取消均通过临时 offscreen 检查。诊断确认 `loadEngine()` 返回前 GPU/pinned host/CPU resize 缓冲区存在，首次输入预处理复用地址；临时 engine 软链接在加载后移除，仍连续完成两次真实 FS/XYZ 推理，初始化日志只出现一次。回放相机保持同一源，预加载期间 GUI 事件循环响应正常；未新增仓库测试目标。

### 阶段 3：加入选区测量与导出

- [x] 实现校正左图 mask 编辑与清空；原多边形工具已替换为 SAM 和刷子。
- [x] 实现图像显示坐标与完整校正左图像素坐标映射，缩放窗口保留顶点位置。
- [ ] 接入校正图 mask 到推理网格的最近邻缩放与上传。
- [ ] 根据 mask 更新点云、mesh 和面积，复用已缓存的视差。
- [ ] 合并快速编辑产生的待处理任务，阻止旧结果覆盖新结果。
- [ ] 显示曲面面积、有效点数、三角形数量和结果更新状态。
- [x] 导出完整校正左图尺寸的二值 selection mask PNG。
- [ ] 导出输入图像、标定、点云 PLY、mesh PLY 和测量参数及面积的完整结果包。

#### SAM 2.1 Hiera Large 辅助选区

采用 Qt + C++ TensorRT，用前景点、背景点和矩形框生成并修正 selection mask。Python/AnyLabeling 仅用于结果对照，应用不依赖 Python 进程。

- [x] 使用工作区 `onnx/SAM2/sam2.1_hiera_large.encoder.onnx` / `.decoder.onnx` 对应的两个 engine：`onnx/sam2.1_hiera_large.encoder.engine` 和 `.decoder.engine`。Decoder 固定一个目标，支持 1–64 个提示点，框占两个点。
- [x] `fs::SamSegmenter` 独立持有 CUDA stream、两个 TensorRT context、GPU I/O、三组特征、pinned CPU staging 和模型尺寸图像缓冲区。Splash 完成加载、接口/profile 校验及分配后才允许打开重建窗口；支持分别选择 FS / SAM encoder / SAM decoder engine 并重试。
- [x] 继续使用 PipelineWorker 的后台 QThread 串行管理 FS 与 SAM；它们使用各自的 CUDA stream。第一条有效提示编码当前冻结校正左图一次；同图修改提示只跑 decoder，清空提示保留特征，成功换图清除特征。没有虚拟推理预热。
- [x] RGB 图像线性缩放到 1024×1024，按 mean/std 归一化；提示保持浮点坐标。Decoder 接收完整提示列表，`mask_input=0`、`has_mask_input=0`；选择三个候选中评分最高者，先把 logits 线性恢复到原图尺寸，再以 >0 阈值化。未照搬本机 AnyLabeling 的重复通道交换和整数坐标截断。
- [x] Region measurement 支持拖框、前景点、背景点、拖动点、删除提示、撤销提示、清空、mask 叠加及 Finish draw 确认。右键可删除提示，Backspace 撤销点，Escape 取消正在拖动的框/点。原 Polygon selection 面板已移除，Clear mask 清空提示、预测及手工修补。移除右侧面板，统一左栏按步骤显示：第一步 Input and calibration，第二步 Mask draw，第三步 Depth range / Geometry；Finish draw 在顶部替代原 Use mask 按钮，确认后进入第三步；SAM 2.1 auto draw / Manual draw 分隔模型提示与刷子工具。
- [x] Brush (+) / Eraser (−) 直接补选或擦除像素，大小为原图 1–100 px（默认 50 px），鼠标显示刷子轮廓，连续笔画不留断点。选中 Brush/Eraser 后在图内滚轮每格调整 1 px，同步大小输入框和轮廓。Mask draw 底部文字、Surface measurement 和 Export 面板已移除。修补层保留到后续 SAM 结果中，刷子不触发推理；Escape 取消本次笔画，修改后需重新 Finish draw。
- [x] 已确认的 mask 按完整校正左图坐标保留在内存；Export 面板及保存入口已按要求移除。SAM 保留原始二值区域及孔洞，不经过 AnyLabeling 的外轮廓简化/小区域过滤。
- [x] 使用图像编号及原子提示版本跳过过期排队任务、丢弃旧推理结果；修改 SAM 提示不触发 FS inference。关闭窗口等待后台阶段结束，再释放两模型及相机资源。
- [x] 实际 Large TensorRT encoder/decoder 推理通过；三组点/框提示与 ONNX Runtime mask IoU 为 0.99981、0.99980、0.99673（对齐 RGB 和浮点坐标，现有 engine 允许 TF32）。临时 Qt 测试通过实际标定 → splash → 重建窗口启动链、splash 失败重试、FS/SAM 同时加载、框/正负点/拖点/删点/撤销、确认与导出、缩放不改变 mask、清空/换图过滤旧结果，以及 FS 重建和关闭。
- [x] 确认后的 mask 最近邻缩放到 FS 宽 960 × 高 800 网格，disparity → XYZ 对全图计算；随后仅在 mask 内调用 FS 的 3×3 CUDA 邻域去噪，mask 外 XYZ 原样复制到最终缓冲区。默认 0.01 m / 内部 3 邻居 / 边缘最多 2 邻居，与 Python 一致。新增 GPU mask、pinned staging、CPU resize 和独立 XYZ scratch 均在 splash 分配，使用 FS stream。
- [x] 第三步 Geometry 启用去噪开关（默认开）和邻居距离设置；改动会重置后续完成状态。Jet 使用最终 Z；mask 外保留原深度，仅无效、超范围或被去噪剔除的点为黑色。
- [x] 验证：75 组 CUDA 去噪与 Python 逐像素一致；真实样例 mask 内有效点 110596 → 110362，keep mask 完全一致。真实 FS/Qt 覆盖空/全 mask、最近邻缩放、缓冲复用、重复计算、去噪开关和距离、mask 外 XYZ 完全不变及状态失效，临时测试保留在 /tmp。
- [x] 将选区及去噪后的 XYZ 接入原生 CPU Triangle：复用第三步已有下载，不增加 CUDA stream 或 mesh 显存缓冲；第四步 Generate mesh 后显示真实面积和数量。
- [x] 与 800×960 notebook 一致：全部有效选中点、外轮廓最近点吸附和硬约束、重心在 mask 内、最大边长 2 cm / 深度跳变 1 cm；内部孔洞不设约束，过滤掉的面不再次补洞。等距离最近点的选择可能与 SciPy 不同。
- [x] 实测 110359 顶点 / 218294 三角面，CPU 约 0.13 秒；逐顶点和逐面连接与 Python 一致。临时测试覆盖多区域、凹区、洞、空结果、取消、原始 XYZ 不变及 OpenGL 显示。
- [ ] 根据实际使用评估分割质量、长期显存稳定性，以及是否需要轮廓简化或单连通区域约束。

临时诊断文件位于 `/tmp/fs_sam_*`，未新增仓库测试目标。单次实测 encoder 包含输入处理约 86 ms；Qt 工作流首次 encoder+decoder 约 98–103 ms，同图缓存后的 decoder 和输出处理约 5–10 ms；这些是当前样例观测值，不是性能保证。无需引入视频记忆模块。

### 阶段 4：相机输入（按当前优先级提前推进）

- [x] 实现 `IStereoSource` 和 Sentech 采集适配器，配置左右身份，输出自有 RGB 图像及帧元数据。
- [ ] 实现文件输入适配器。
- [ ] 实现文件回放，模拟双目预览和拍摄一组图像。
- [ ] 验证拍摄冻结后可进入现有重建与测量流程。
- [x] 相机预览与标定采样统一通过 `fs_gui` 进入；原独立 `fs_sentech` CLI、构建目标和 VS Code 入口已按用户要求移除。
- [x] 使用实际双机验证身份绑定、采集、RGB 保存与帧元数据：左右均为 2448×2048，曝光按 SDK 调整为 49994.8 µs。
- [x] 自动化验证枚举顺序不影响左右角色、重复/缺失身份、双侧新帧、主机到达时间差、停止与错误传播。
- [ ] 使用该物理相机组合的正确标定验证校正和推理，检查极线对齐；不直接假定历史数据标定适用于当前安装。
- [ ] 验证预览窗口交互、长时间采集和真实拔插恢复；根据测量要求接入并验证曝光同步。

相机 SDK 默认 `/opt/sentech`，采集库随默认启用的 `FS_BUILD_DESKTOP` 构建；设为 `OFF` 可关闭 Qt 及相机 SDK 依赖。
参考项目为 `/home/liu4000/Desktop/FS`，沿用用户确认的左右相机绑定。
主机到达时间差门限仅用于软件配对，不代表两台相机同时曝光；原有校正偶发差异记录仍保留。

## 验收清单

以下验证通过实际运行和人工验收记录；当前不重新引入已按要求删除的测试文件。

- [x] 程序启动只出现标定窗口，未完成时无法进入重建与测量窗口。
- [x] 完成条件满足后只启动一个第二窗口；关闭/取消第一窗口不会启动第二窗口。
- [ ] 第一窗口的板参数、样本管理、计算、保存和独立样本检查行为与参考工程的预期流程一致。
- [ ] 采样、求解或检查失败时提示原因，先前的成功结果不会被误用于当前未通过状态。
- [ ] 两窗口交接后，左右相机绑定、RGB 图像网格、baseline 单位和标定内容保持一致，采集源不被重复打开。
- [ ] 更换板参数、相机或采集尺寸后，过期样本、结果及完成状态正确失效。
- [ ] 同一输入的桌面与 CLI 推理结果一致。
- [ ] 固定视差、mask 和参数，对照 Python 的点云、mesh 和面积；差异定位到过滤或三角化阶段。
- [ ] 使用已知尺寸平面验证面积及 m²/cm² 单位换算。
- [ ] 使用缩放图像验证选区映射和内参缩放。
- [ ] 修改 mask 不触发 FS 推理，连续编辑不被旧任务结果覆盖。
- [ ] 推理过程中界面可操作，停止在安全阶段边界生效。
- [ ] 缺少标定、左右尺寸不匹配、engine 加载失败和无有效 mesh 时显示清楚的错误信息。
- [x] 无 GPU、缺少/损坏 engine 时停留在 splash 显示失败原因，支持换文件重试或退出；不会创建未就绪的重建窗口。

## 参考

- `/home/liu4000/Desktop/FS/apps/sentech_stereo/main.cpp`：ImGui calibration 交互参考。
- `/home/liu4000/Desktop/FS/include/ffs_viewer/calibration/` 与 `src/calibration/`：ChArUco 检测、求解和检查参考。
- [VTK：QVTKOpenGLNativeWidget](https://vtk.org/doc/nightly/html/classQVTKOpenGLNativeWidget.html)
- [Qt：Threads and QObjects](https://doc.qt.io/qt-6/threads-qobject.html)
- [AnyLabeling：SAM ONNX 导出说明](https://anylabeling.nrl.ai/docs/samexporter)
- [SAM 2.1 ONNX 模型](https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models)
- [NVIDIA：SAM2 ONNX/TensorRT 参考](https://github.com/NVIDIA/DeepStream/blob/main/tools/sam2-onnx-tensorrt/README.md)（含视频流程；本项目仅需静态图像 encoder/decoder。）

# Stereo rectification C++ example

This CMake project reads `left.png`, `right.png`, and `calibration.json` from a
capture directory, then rectifies the stereo pair in memory. Its OpenCV
operations match `python/fs_high_resolution_utils.py`:

`stereoRectify` → `initUndistortRectifyMap` → `remap`.

Source layout:

Public headers live under `include/fs/`, with matching implementation
folders under `src/`:

- `core/`: logging (`Logger`).
- `stereo/`: stereo images, calibration, and rectification (`StereoFrame` and `StereoCalibration`).
- `inference/`: TensorRT inference (`FS`).
- `inference/plugins/`: the independent GWC plugin and CUDA kernel.
- `calibration/`: ChArUco detection, stereo calibration, independent checks and session JSON.
- `ui/`: Qt 6 calibration window, controller, worker and image widgets.
- `app/`: CLI, desktop and engine-builder entry points.

Include headers using their full path, for example
`#include "fs/inference/FS.hpp"` or `#include "fs/stereo/StereoFrame.hpp"`.
The CMake include root remains `cpp/include`.

Build targets:

- `fs_core`: static library containing `FS.cpp`, `StereoFrame.cpp`,
  `StereoCalibration.cpp`, and `Logger.cpp`, shared by the CLI and future desktop application. It exposes
  the project headers and OpenCV/CUDA dependencies needed by its public API;
  TensorRT and the GWC plugin are implementation dependencies.
- `TSFS`: command-line executable containing only `app/TSFS.cpp`, linked to
  `fs_core`.
- `GWCVolumePlugin` and `fs_build_single_engine`: independent plugin library
  and engine builder.

## Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
```

OpenCV 4 development packages are required.

## Qt 6 calibration window

Install Qt 6 Widgets development files (`qt6-base-dev` on Ubuntu) and OpenCV
including `aruco` (`libopencv-contrib-dev`). The Sentech SDK defaults to `/opt/sentech`.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target fs_gui --parallel
./cpp/build/fs_gui
```

For a fresh build directory on this machine, if CUDA is not on `PATH`, also
pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc`,
`-DCUDAToolkit_ROOT=/usr/local/cuda-13.3` and `-DCMAKE_CUDA_ARCHITECTURES=120`.
The existing `cpp/build` cache already has these settings.

`FS_BUILD_DESKTOP` defaults to ON. Set `-DFS_BUILD_DESKTOP=OFF` to disable the
desktop target and its camera SDK dependency for file-only builds. Existing
build directories retain their cached option values. The desktop target enables
the camera source library. The desktop
requires Qt 6 and does not link OpenCV highgui/Qt 5. VTK is not needed yet.
In VS Code, select **Run FS GUI**; its build task configures this target.

The first window provides:

1. **Connect cameras**: fixed left `21LJ530`, right `21LJ548`, editable exposure
   before connecting. Acquisition, corner detection and calibration run on a
   worker thread. RGB images stay independent of SDK buffers.
2. **Board settings**: 10 × 8 squares, 66.5 mm square side, 50.5 mm marker side,
   `DICT_4X4_250`. Inputs display mm; calibration JSON stores metres. Applying
   different parameters clears samples and calibration. Settings are included
   in saved calibration; this version starts with the confirmed defaults.
3. **Capture sample**: hold the board still. Collect five fresh candidates and
   choose the valid pair with the smallest host arrival gap. At least four
   shared corners are required; independent device timestamps are not compared.
   Capture times out after 15 seconds and can be cancelled. History holds up to
   20 pairs in memory, with selection, deletion and return to live preview.
4. **Compute**: solve per-camera intrinsics, then stereo extrinsics with fixed
   intrinsics. At least three valid pairs are required; capture additional varied
   positions and tilts for useful coverage. Adding/removing samples invalidates
   the old result. **Detection overlay** in the raw preview draws ArUco marker
   outlines and IDs together with ChArUco corners and IDs, including in saved
   sample previews. Marker outlines remain visible when no ChArUco corners are
   found. Rectified preview provides epilines.
5. **Check · new pose**: acquire a separate pair, estimate board pose in the left
   camera, and reproject into both cameras. The preview freezes on this check
   pair. The provisional stereo RMS limit is editable (default 1.0 px) and applies
   to both solving and checking; it is not a guarantee of measurement accuracy.
6. **Save calibration**: atomically write the six matrices accepted by
   `StereoCalibration::from_file()`, plus board, camera identities, image grid
   and RMS metadata. Loading requires matching board/camera/grid metadata;
   matrix-only legacy files remain supported by the CLI but are rejected here.
   Loading or reconnecting requires a fresh check. Check/threshold changes
   require saving again. Sample images are not exported in this version.
7. **Finish calibration**: enabled only when current cameras/grid match, solve
   and independent check pass, the result is saved, and no task is pending.
   This release then releases cameras and exits. The second reconstruction
   window will be added later at `DesktopController`'s successful-completion
   boundary. Ordinary close cancels the session and releases resources; it
   never signals successful completion. An in-progress OpenCV solve finishes
   before shutdown; the UI remains responsive while waiting.

Public algorithms live in `include/fs/calibration/CharucoCalibration.hpp` and
`src/calibration/CharucoCalibration.cpp` (`fs_calibration`). Qt code lives in
`ui/` (`fs_calibration_ui`), with `app/desktop_main.cpp` as the executable entry.
The calibration window does not instantiate an FS engine or run inference.
Extrinsic JSON field names retain the legacy `right_to_left_*` names; the actual
values are OpenCV's left-to-right transform, passed unchanged to rectification.

## Build one FoundationStereo TensorRT engine with the GWC plugin

First export the plugin-ready ONNX (it contains the
`foundation_stereo::GWCVolume` custom node), then build the C++ targets:

```bash
python python/make_onnx_liu_gwc_plugin.py
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target fs_build_single_engine --parallel
```

Build the fixed-shape single engine:

```bash
cpp/build/fs_build_single_engine \
  onnx/foundationstereo_800x960_gwc_plugin.onnx \
  onnx/foundationstereo_800x960_gwc_plugin.engine
```

The builder is intended for TensorRT 11.2 and the repository's `GWCVolume`
Plugin V3 implementation. It registers the plugin before parsing ONNX.
TensorRT 11 is strongly typed, so FP32/FP16 decisions come from the exported
ONNX tensor types rather than an `--fp32`/`--fp16` builder flag. Other useful
options are `--workspace-mb 8192`,
`--optimization-level 3`, and `--no-compilation-cache`.

The resulting engine must be deserialized by a program that loads and
registers `libGWCVolumePlugin.so` before creating the TensorRT runtime.

In VS Code, open the repository root and press `Ctrl+Shift+B` (or
`Cmd+Shift+B` on macOS) to run the same Release build using the default task.
Alternatively, install the recommended **CMake Tools** extension. Its **Build**
button in the VS Code status bar uses this project's Release configuration.

For the included sample run configuration, open **Run and Debug**, select
`Run TSFS (Volunteer2 lower/0, Release)`, and press `Ctrl+F5`. It
builds the Release binary first, then runs it with
`data/Volunteer2_lower/0`.

## Run

```bash
./cpp/build/TSFS \
  data/Volunteer3_lower/0
```

After `frame.rectify()`, use `frame.rectified_left()` and
`frame.rectified_right()` in `app/TSFS.cpp` for the next processing stage.

## In-memory stereo input

The file-path constructor remains supported. It loads RGB images and calibration,
then delegates to the same validated constructor used by memory input:

```cpp
#include "fs/stereo/StereoFrame.hpp"

// left_rgb and right_rgb are matching CV_8UC3 RGB images already in memory.
StereoCalibration calibration = StereoCalibration::from_file("calibration.json");
StereoFrame frame(left_rgb, right_rgb, calibration);
frame.rectify();

// Continue with the existing FS workflow using frame.rectified_left(),
// frame.rectified_right(), and frame.rectified_camera_parameters().
```

For fully in-memory input, populate `StereoCalibration` directly with
`left_camera_matrix`, `right_camera_matrix`, `left_distortion`,
`right_distortion`, `right_to_left_rotation`, and `right_to_left_translation`.
Its declaration is in `include/fs/stereo/StereoCalibration.hpp`.

- Images must be non-empty, matching 2D RGB `CV_8UC3` matrices. Convert camera
  SDK BGR output to RGB before calling the memory constructor. Non-contiguous
  image ROIs are supported.
- Calibration matrices must be finite, single-channel `CV_32F` or `CV_64F`.
  Intrinsics are 3×3 with positive focal lengths and last row `[0, 0, 1]`;
  distortion is a row/column vector of 4, 5, 8, 12 or 14 coefficients (use zeros
  for no distortion); rotation is a proper 3×3 rotation; translation is a nonzero
  3-element row/column vector in metres.
- Use calibration for the input image grid. Extrinsic field names and the
  existing JSON convention are preserved: rotation/translation pass unchanged
  to `stereoRectify`, without inversion or unit conversion.
- `StereoFrame` deep-copies both images and all calibration matrices, storing
  calibration as `CV_64F`. Callers can reuse or release their buffers after
  construction. Invalid image/calibration values throw `std::invalid_argument`.
  `from_file()` reads required matrices; the frame constructor validates their
  shapes and values. File loading errors remain exceptions.

## Sentech stereo cameras

The `fs_sentech_source` library implements `IStereoSource` with the Sentech StApi
SDK and is used by the Qt calibration window. Launch `fs_gui` and click
**Connect cameras**, or select **Run FS GUI** in VS Code.

Default identities, confirmed for this rig:

| Role | Display name | Serial |
| --- | --- | --- |
| Left | `STC-MCS500U3V(21LJ530)` | `21LJ530` |
| Right | `STC-MCS500U3V(21LJ548)` | `21LJ548` |

The source API accepts an exact display name, user-defined name, or serial.
Enumeration order never assigns a role; missing, ambiguous, or same-camera
matches fail. The current Qt window uses the fixed identities above.

The capture implementation follows the local reference project at
`/home/liu4000/Desktop/FS/src/io/sentech_stereo_source.cpp`: timed exposure defaults
to 50,000 µs (adjusted to supported increments), continuous white balance when
available, one acquisition worker per camera, and owned RGB8 snapshots.
Exposure can be changed in the Qt window before connecting.

These are **independent continuous streams, not hardware-synchronized exposures**.
A delivered pair contains a fresh frame from each side with host-arrival skew
at most `SentechStereoOptions::max_arrival_skew` (default 100 ms). Host arrival
time includes transport latency; camera-local device timestamps are retained
but never treated as a shared clock. Hold the board still during each capture.

Capture errors propagate to the Qt worker. Acquisition threads are joined before
SDK buffers/devices are released, and the cameras can be reopened. The Qt window
also releases the cameras on ordinary close; it waits for an active calibration
solve before exiting. Camera settings are not saved to device user sets.

SDK discovery defaults to `/opt/sentech`; override `FS_SENTECH_ROOT` if needed.
The source library links StApi TL/IP, GenICam, the SDK's turbojpeg, and OpenCV core.
It supplies the SDK runtime search paths and sets `GENICAM_GENTL64_PATH` only when
that variable is absent. Set `-DFS_BUILD_DESKTOP=OFF` for file-only builds without
Qt or SDK dependencies. The standalone OpenCV preview/capture CLI has been removed;
raw capture export and camera-to-inference UI will be added with the reconstruction
window.

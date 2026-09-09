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
- `app/`: CLI and engine-builder entry points.

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

The optional `fs_sentech_source` library implements `IStereoSource` with the
Sentech StApi SDK. `fs_sentech` provides a small OpenCV preview/capture tool and
an optional in-memory handoff to `StereoFrame` and FS inference. The full Qt/VTK
workbench remains a later step.

Default identities, confirmed for this rig:

| Role | Display name | Serial |
| --- | --- | --- |
| Left | `STC-MCS500U3V(21LJ530)` | `21LJ530` |
| Right | `STC-MCS500U3V(21LJ548)` | `21LJ548` |

Matching accepts an exact display name, user-defined name, or serial. Enumeration
order never assigns a role; missing, ambiguous, or same-camera matches fail.
Use `--left` and `--right` to override the defaults.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DFS_BUILD_SENTECH=ON
cmake --build cpp/build --parallel

./cpp/build/fs_sentech --list
./cpp/build/fs_sentech --preview

# Save one pair immediately into a NEW directory:
./cpp/build/fs_sentech --capture data/sentech_capture_001

# Preview, then press Space to save the displayed pair:
./cpp/build/fs_sentech --preview --capture data/sentech_capture_002
```

Preview shows LEFT and RIGHT side by side. Space captures one pair and exits;
Q/Esc exits without capturing. An existing output directory is never overwritten.
Saved files are `left.png`, `right.png`, and `capture.json` with identities,
frame IDs and timestamps. When `--calibration` is supplied, its JSON is copied
into the capture directory too. Raw captures can be saved without calibration.

To rectify and infer a captured pair, supply calibration for this physical rig,
left/right assignment, and acquisition resolution:

```bash
./cpp/build/fs_sentech --preview --capture data/sentech_capture_003 \
  --calibration /path/to/this_rig/calibration.json --infer
```

The source is stopped before processing the owned RGB snapshots. FS still uses
the existing 800×960 engine; `--engine PATH` overrides its path. Inference ends
with disparity in FS device memory; point clouds/mesh and public disparity
readback are not implemented by this tool. Calibration matching must be verified
on the actual rig; the existing JSON does not encode camera serials or image size.

The capture implementation follows the local reference project at
`/home/liu4000/Desktop/FS/src/io/sentech_stereo_source.cpp`: timed exposure defaults
to 50,000 µs (adjusted to supported increments), continuous white balance when
available, one acquisition worker per camera, and owned image snapshots. It
converts directly to RGB8 for the current `StereoFrame` contract.

These are **independent continuous streams, not hardware-synchronized exposures**.
A delivered pair contains a fresh frame from each side with host-arrival skew
at most `--max-arrival-skew-ms` (default 100). Host arrival time includes transport
latency; camera-local device timestamps are retained but never treated as a shared
clock. This pairing is suitable for connection checks and static-scene trials;
moving-scene measurement needs a verified synchronization arrangement.

`--exposure-us` changes requested exposure; `--timeout-ms` (default 5000) limits
waiting for valid pairs after streaming starts. Capture errors are propagated,
workers are joined before SDK buffers/devices are released, and subsequent captures
can reopen the cameras. Controls are changed in the active camera session; this
tool does not save camera user sets, change ROI, or perform calibration.

SDK discovery defaults to `/opt/sentech`; override `FS_SENTECH_ROOT` if needed.
The camera target links StApi TL/IP, GenICam and the SDK's turbojpeg, plus OpenCV
HighGUI. It supplies the SDK runtime search paths and sets `GENICAM_GENTL64_PATH`
only when that variable is absent. `FS_BUILD_SENTECH=OFF` keeps SDK/preview
requirements out of the file-only build (default).

In VS Code, select **Preview Sentech stereo (Left 21LJ530 / Right 21LJ548)** or
**List Sentech cameras** in Run and Debug; both build the camera target first.

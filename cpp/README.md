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

## Input regression tests

The default `BUILD_TESTING=ON` adds `stereo_frame_tests`. The tests create their
own small image/calibration fixtures and do not run GPU inference or require an
engine or capture dataset; building still uses this project's dependencies.

```bash
ctest --test-dir cpp/build --output-on-failure

# Optional equivalence check using an existing capture:
./cpp/build/stereo_frame_tests data/Volunteer2_lower/0
```

Coverage includes file/memory rectification equivalence, deep-copy ownership,
RGB preservation, strided inputs, float calibration, row/column vectors, invalid
inputs, missing JSON fields, and baseline units.

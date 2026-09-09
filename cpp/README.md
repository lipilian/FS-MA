# Stereo rectification C++ example

This CMake project reads `left.png`, `right.png`, and `calibration.json` from a
capture directory, then rectifies the stereo pair in memory. Its OpenCV
operations match `python/fs_high_resolution_utils.py`:

`stereoRectify` → `initUndistortRectifyMap` → `remap`.

Source layout:

- `include/StereoFrame.hpp`: stereo frame class interface.
- `src/StereoFrame.cpp`: stereo frame implementation.
- `app/TSFS.cpp`: command-line application entry point.

Build targets:

- `fs_core`: static library containing `FS.cpp`, `StereoFrame.cpp`, and
  `Logger.cpp`, shared by the CLI and future desktop application. It exposes
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

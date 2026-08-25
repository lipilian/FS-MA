# Stereo rectification C++ example

This CMake project reads `left.png`, `right.png`, and `calibration.json` from a
capture directory, then rectifies the stereo pair in memory. Its OpenCV
operations match `python/fs_high_resolution_utils.py`:

`stereoRectify` → `initUndistortRectifyMap` → `remap`.

Source layout:

- `include/StereoFrame.hpp`: stereo frame class interface.
- `src/StereoFrame.cpp`: stereo frame implementation.
- `src/main.cpp`: command-line application entry point.

## Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
```

OpenCV 4 development packages are required.

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
`frame.rectified_right()` in `src/main.cpp` for the next processing stage.

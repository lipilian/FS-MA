# Development Progress

Please see the [development progress](https://effective-robot-77r192w.pages.github.io/).

## Stereolabs ZED scanning

With the desktop dependencies and the Stereolabs ZED SDK installed, build and run:

```sh
cmake -S cpp -B cpp/build -DFS_BUILD_DESKTOP=ON
cmake --build cpp/build --target scan_gui -j6
./cpp/build/scan_gui
```

`scan_gui` skips the calibration screen, loads the existing FoundationStereo,
SAM 2.1 and MapAnything TensorRT engines from `onnx/`, then automatically connects
a Stereolabs ZED camera (including ZED Mini). The SDK captures synchronized
rectified RGB at 1920 × 1080 / 30 FPS. Each view is center-cropped to 1296 × 1080
(x = 312), then FoundationStereo uses the same OpenCV `INTER_LINEAR` resize to
960 × 800 as imported 2448 × 2048 captures. Intrinsics are shifted by the crop
and scaled with the images; the baseline remains in metres. FoundationStereo
computes depth; SDK depth is disabled. Use **Capture pair**
to reconstruct, then **Capture more** for up to five views. If the camera is not
connected at startup, attach it and click **Connect cameras**. A model load failure
keeps the loading window open so engine paths can be corrected and retried.

The first capture displays an FS triangle mesh. From the second capture onward,
MapAnything predicts camera poses and displays each view as a point cloud; MA
triangle meshes are not computed. MA results export as a world-space point-cloud
PLY without faces. FS point clouds remain available through the source selector.

The scan workspace uses fixed depth and geometry defaults and a full-width view.
Captures and imported pairs reconstruct automatically. In both camera and import
mode, click a captured camera or view label, then **Start draw** to edit its mask
directly in the scene's left preview and **Finish draw** to confirm it. Undrawn or
cleared FS geometry uses 30% opacity; selected regions are opaque. Mask edits retain
the existing FS/MA results and do not trigger reconstruction. Use **Save results…**
in the bottom-right corner to export.

Scan export preserves each mask's rectified-left image dimensions without resizing:
1296 × 1080 for ZED captures, or 2448 × 2048 for imports at that resolution.
Undrawn masks are all-zero images on the same grid. Exported RGB and float32 Z depth
use the 960 × 800 FoundationStereo input grid.

The existing `fs_gui` entry still starts with calibration. Imported captures in
`scan_gui` must include `left.png`, `right.png` and their own `calibration.json`
(or `sentech_stereo_calibration.json`).

With `FS_BUILD_TESTS=ON`, `ctest --test-dir cpp/build --output-on-failure` includes
the calibration-free startup regression. An optional hardware smoke test loads
the models, connects/reconnects a ZED, captures a pair and checks OpenGL display:

```sh
./cpp/build/fs_scan_startup_test --live
```

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

The first scan capture displays FS points. Confirm a mask with **Finish draw** to
automatically build a GPU triangle mesh using only valid points inside that mask
and calculate its surface area. The mask is sampled on the 960 × 800 FS grid with
nearest-neighbour resizing. Editing and confirming the mask rebuilds this mesh;
clearing the mask removes the mesh and its area. From the second capture onward,
MapAnything predicts camera poses and displays each view as a point cloud; MA
triangle meshes are not computed. MA results export as a world-space point-cloud
PLY without faces. FS point clouds remain available through the source selector.

With geometry filtering enabled, a single view uses only the 5 × 5 neighbour
support/distance filter. Before MA runs with 2–5 views, every view also receives
the normal/view-angle check (< 60°). The first view is updated from its cached
pre-denoising XYZ without rerunning FS; its depth, point cloud and MA input are
updated together. Completed views are not filtered repeatedly as more are added.

The scan workspace uses fixed depth and geometry defaults and a full-width view.
Captures and imported pairs reconstruct automatically. In both camera and import
mode, click a captured camera or view label, then **Start draw** to edit its mask
directly in the scene's left preview and **Finish draw** to confirm it. Undrawn or
cleared FS geometry uses 30% opacity; selected regions are opaque. Mask edits retain
the existing FS depth and MA results without rerunning inference. Only single-view
mask confirmation triggers GPU meshing and area calculation. Use **Save results…**
in the bottom-right corner to export.

Scan export resizes each rectified-left mask to 960 × 800 (width × height) using
nearest-neighbour sampling and saves binary 0/255 pixels aligned with the FS images
and depth. Undrawn masks are all-zero images on the same grid. Each view exports `left01.png`,
`right01.png`, `mask01.png` and `depth01.tiff` (then `02`, `03`, etc.). Left and right
RGB images are the exact rectified, resized FoundationStereo inputs at 960 × 800;
float32 Z depth uses the same grid. A shared `cameras.json` contains the FS intrinsics
and original MA camera-to-world poses in metres (identity for a single view).

The existing `fs_gui` entry still starts with calibration. Imported captures in
`scan_gui` must include `left.png`, `right.png` and their own `calibration.json`
(or `sentech_stereo_calibration.json`).

With `FS_BUILD_TESTS=ON`, `ctest --test-dir cpp/build --output-on-failure` includes
the calibration-free startup regression. An optional hardware smoke test loads
the models, connects/reconnects a ZED, captures a pair and checks OpenGL display:

```sh
./cpp/build/fs_scan_startup_test --live
```

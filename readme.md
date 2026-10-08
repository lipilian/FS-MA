# Development Progress

Please see the [development progress](https://effective-robot-77r192w.pages.github.io/).

## RealSense scanning

With the desktop dependencies and `librealsense2-dev` installed, build and run:

```sh
cmake -S cpp -B cpp/build -DFS_BUILD_DESKTOP=ON
cmake --build cpp/build --target scan_gui -j6
./cpp/build/scan_gui
```

`scan_gui` skips the calibration screen, loads the existing FoundationStereo,
SAM 2.1 and MapAnything TensorRT engines from `onnx/`, then automatically connects
an Intel RealSense D435. The camera's factory-calibrated left/right IR images are
center-cropped to 960 × 800; FoundationStereo computes depth. Use **Capture pair**
to reconstruct, then **Capture more** for up to five views. If the camera is not
connected at startup, attach it and click **Connect cameras**. A model load failure
keeps the loading window open so engine paths can be corrected and retried.

The existing `fs_gui` entry still starts with calibration. Imported captures in
`scan_gui` must include `left.png`, `right.png` and their own `calibration.json`
(or `sentech_stereo_calibration.json`).

With `FS_BUILD_TESTS=ON`, `ctest --test-dir cpp/build --output-on-failure` includes
the calibration-free startup regression. An optional hardware smoke test loads
the models, connects/reconnects a D435, captures a pair and checks OpenGL display:

```sh
./cpp/build/fs_scan_startup_test --live
```

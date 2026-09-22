# syntax=docker/dockerfile:1
# Build from the repository root (requires Docker Buildx):
# docker buildx build --load --build-context sentech-sdk=/opt/sentech \
#     -t fs-engine:cuda12.9-trt10.14 .
# The named context is an installed Linux x86_64 Sentech SDK, including
# include/, lib/, and lib/GenICam/. It is copied into the image, not into Git.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG TENSORRT_VERSION=10.14.1.48-1+cuda12.9

# C/C++17 + CUDA build tools; OpenCV includes the contrib ArUco module.
# Qt 6 is used by fs_gui. Qt 5 runtime libraries support Sentech's own tools.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        gdb \
        ca-certificates \
        curl \
        libopencv-dev \
        libopencv-contrib-dev \
        qt6-base-dev \
        qt6-base-dev-tools \
        libqt6opengl6-dev \
        qt6-qpa-plugins \
        qt6-wayland \
        libqt5widgets5t64 \
        libqt5opengl5t64 \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libxcb-xinerama0 \
        fonts-dejavu-core \
        xauth \
        mesa-utils \
        libusb-1.0-0 \
        libudev1 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# The CUDA image already configures NVIDIA's signed apt repository.
# Pin both the TensorRT release and CUDA variant, including transitive packages.
# Install C++ development headers/libraries, ONNX parser, plugins and trtexec.
RUN printf 'Package: libnvinfer* libnvonnxparsers*\nPin: version %s\nPin-Priority: 1001\n' \
        "${TENSORRT_VERSION}" > /etc/apt/preferences.d/tensorrt \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "libnvinfer10=${TENSORRT_VERSION}" \
        "libnvinfer-headers-dev=${TENSORRT_VERSION}" \
        "libnvinfer-dev=${TENSORRT_VERSION}" \
        "libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}" \
        "libnvinfer-plugin10=${TENSORRT_VERSION}" \
        "libnvinfer-plugin-dev=${TENSORRT_VERSION}" \
        "libnvonnxparsers10=${TENSORRT_VERSION}" \
        "libnvonnxparsers-dev=${TENSORRT_VERSION}" \
        "libnvinfer-bin=${TENSORRT_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# cpp/cmake/Sentech.cmake requires the SDK even for file-only GUI sessions.
COPY --from=sentech-sdk / /opt/sentech/
RUN test -f /opt/sentech/include/StApi/StApi_TL.h \
    && test -f /opt/sentech/include/GenICam/GenApi/GenApi.h \
    && test -f /opt/sentech/lib/libStApi_TL.so \
    && test -f /opt/sentech/lib/libStApi_IP.so \
    && test -f /opt/sentech/lib/libturbojpeg.so \
    && test -f /opt/sentech/lib/GenICam/libGCBase.so \
    && test -f /opt/sentech/lib/GenICam/libGenApi.so \
    && printf '/opt/sentech/lib\n/opt/sentech/lib/GenICam\n' \
        > /etc/ld.so.conf.d/sentech.conf \
    && ldconfig

ENV STAPI_ROOT_PATH=/opt/sentech \
    GENICAM_GENTL64_PATH=/opt/sentech/lib \
    LD_LIBRARY_PATH=/opt/sentech/lib:/opt/sentech/lib/GenICam:${LD_LIBRARY_PATH} \
    PATH=/usr/src/tensorrt/bin:${PATH} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    QT_QPA_PLATFORM=xcb

# Mount this repository here at runtime. Keep the container build separate
# from cpp/build, which may contain the host's CUDA 13 / TensorRT 11 artifacts:
# cmake -S cpp -B /tmp/fs-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     -DFS_BUILD_DESKTOP=ON -DCMAKE_CUDA_ARCHITECTURES=120
# cmake --build /tmp/fs-build --parallel
# GPU access: --device=nvidia.com/gpu=all (CDI), or --gpus all with the NVIDIA runtime.
# GUI use also requires DISPLAY, the X11 socket and Xauthority.
# Physical cameras require their USB devices or GigE network access at runtime.
WORKDIR /workspace/FS_Engine

# Notebook environment: Python 3.12 + pip, independent of Conda and system Python.
# Keep these layers after the C++ dependencies so Python changes reuse that cache.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ARG PYTORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
# NVIDIA republished the Python 3.12 bindings as post1; runtime release is 10.14.1.48.
ARG TENSORRT_PYTHON_VERSION=10.14.1.48.post1
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XFORMERS_DISABLED=1

# Match the C++ TensorRT release and keep later pip installs on the cu129 wheels.
RUN printf 'torch==%s+cu129\ntorchvision==%s+cu129\ntensorrt-cu12==%s\n' \
        "${PYTORCH_VERSION}" "${TORCHVISION_VERSION}" "${TENSORRT_PYTHON_VERSION}" \
        > /opt/python-constraints.txt \
    && python -m pip install --upgrade pip setuptools wheel
ENV PIP_CONSTRAINT=/opt/python-constraints.txt

RUN python -m pip install \
        "torch==${PYTORCH_VERSION}+cu129" \
        "torchvision==${TORCHVISION_VERSION}+cu129" \
        --index-url https://download.pytorch.org/whl/cu129

# Python bindings and runtime use the same 10.14 release as libnvinfer above.
RUN python -m pip install \
        "tensorrt-cu12==${TENSORRT_PYTHON_VERSION}" \
        --extra-index-url https://pypi.nvidia.com

# Versions follow the working host fs environment. Headless OpenCV avoids
# Python wheels injecting their own Qt plugins into the C++ GUI environment.
# triangle is the Python mesh dependency; the bundled C++ Triangle is separate.
# ONNX packages also support python/make_onnx_liu_gwc_plugin.py.
RUN python -m pip install \
        numpy==2.4.4 \
        scipy==1.18.0 \
        matplotlib==3.11.1 \
        imageio==2.37.4 \
        pillow==12.2.0 \
        opencv-contrib-python-headless==5.0.0.93 \
        omegaconf==2.3.1 \
        timm==1.0.28 \
        einops==0.8.2 \
        huggingface-hub==1.25.1 \
        open3d==0.19.0 \
        triangle==20250106 \
        trimesh==4.12.2 \
        pandas==3.0.5 \
        joblib==1.5.3 \
        jupyterlab==4.6.2 \
        ipykernel==7.3.0 \
        onnx==1.22.0 \
        onnxscript==0.7.1 \
    && python -m pip check \
    && python -m ipykernel install --sys-prefix --name fs \
        --display-name "FS Engine (CUDA 12.9 / TensorRT 10.14)"

# Start Jupyter with authentication enabled (the startup log prints its token):
# docker run --rm -it --device=nvidia.com/gpu=all -p 127.0.0.1:8888:8888 \
#     --mount type=bind,src="$PWD",dst=/workspace/FS_Engine \
#     fs-engine:cuda12.9-trt10.14 \
#     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Select the fs kernel. Mount weights/data and use a container-built plugin and
# TensorRT 10.14 engine; the notebook currently points at cpp/build and onnx/.
EXPOSE 8888
CMD ["bash"]

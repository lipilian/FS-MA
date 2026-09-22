# Only to test if the Engine work with TENSORRT_VERSION=10.14.1.48-1+cuda12.9

# 1. Build the docker image
```bash
docker buildx build --load \
  --build-context sentech-sdk=/opt/sentech \
  -t fs-engine:cuda12.9-trt10.14 \
  .
```

# 2. Run the docker container
```bash
docker run -dit \
  --device=nvidia.com/gpu=all \
  --mount type=bind,src="$PWD",dst=/workspace/FS_Engine \
  --name fs-engine \
  fs-engine:cuda12.9-trt10.14
```
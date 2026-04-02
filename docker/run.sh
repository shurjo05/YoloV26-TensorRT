#!/bin/bash
# Run the aphid-inference container.
# Call from anywhere — script resolves the repo root automatically.
# Usage: bash docker/run.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker run -it --rm \
    --runtime nvidia \
    --privileged \
    --network host \
    -v "$REPO_ROOT":/workspace \
    -v /usr/src/tensorrt:/usr/src/tensorrt:ro \
    -v /usr/lib/aarch64-linux-gnu:/host-trt-lib:ro \
    -v /usr/include/aarch64-linux-gnu:/host-trt-include:ro \
    aphid

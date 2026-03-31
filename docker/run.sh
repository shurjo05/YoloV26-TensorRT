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
    aphid

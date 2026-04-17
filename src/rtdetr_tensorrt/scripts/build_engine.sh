#!/bin/bash
#
# Build a TensorRT engine from the RT-DETR ONNX file using trtexec.
# Must be run on the target Jetson — engines are device-specific.
#
# We recommend running this OUTSIDE of Docker (on the host shell) because inside
# the container we've occasionally seen `double free or corruption` exits during
# engine build (see CLAUDE.md Issue 21 for context on TRT version handling).
#
# Usage:
#   ./scripts/build_engine.sh [onnx_file] [--fp32]
#
# Default ONNX: models/rtdetrsp3.onnx (single-class aphid model).
# Add --fp32 to skip the --fp16 flag — useful when debugging FP16 precision issues.
# Engine is written next to the ONNX file with a .engine extension.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

ONNX_FILE="${1:-$REPO_DIR/models/rtdetrsp3.onnx}"
PRECISION_FLAG="--fp16"
ENGINE_SUFFIX=".engine"

for arg in "$@"; do
    if [ "$arg" == "--fp32" ]; then
        PRECISION_FLAG=""
        ENGINE_SUFFIX="_fp32.engine"
    fi
done

ENGINE_FILE="${ONNX_FILE%.onnx}${ENGINE_SUFFIX}"

if [ ! -f "$ONNX_FILE" ]; then
    echo "ERROR: ONNX file not found: $ONNX_FILE"
    exit 1
fi

if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    echo "       Make sure JetPack/TensorRT is installed."
    exit 1
fi

echo "Building TensorRT engine (RT-DETR)..."
echo "  ONNX:      $ONNX_FILE"
echo "  Engine:    $ENGINE_FILE"
echo "  Precision: ${PRECISION_FLAG:-fp32}"
echo "  (First run takes 5-15 minutes due to layer profiling)"
echo ""

"$TRTEXEC" \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    $PRECISION_FLAG

echo ""
echo "Done. Engine saved to: $ENGINE_FILE"
echo ""
echo "Next step: update engine_file_path in config/params.yaml to:"
echo "  $ENGINE_FILE"

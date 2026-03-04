#!/bin/bash
#
# Builds a TensorRT engine from an ONNX file using trtexec.
# Must be run on the target Jetson device — engines are device-specific.
#
# Usage:
#   ./scripts/build_engine.sh [onnx_file]
#
# If no argument given, defaults to models/yolo26n-aphid.onnx.
# Engine is saved alongside the ONNX file with a .engine extension.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

ONNX_FILE="${1:-$REPO_DIR/models/yolo26n-aphid.onnx}"
ENGINE_FILE="${ONNX_FILE%.onnx}.engine"

if [ ! -f "$ONNX_FILE" ]; then
    echo "ERROR: ONNX file not found: $ONNX_FILE"
    exit 1
fi

if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    echo "       Make sure JetPack/TensorRT is installed."
    exit 1
fi

echo "Building TensorRT engine..."
echo "  ONNX:   $ONNX_FILE"
echo "  Engine: $ENGINE_FILE"
echo "  (First run takes 5-15 minutes due to layer profiling)"
echo ""

"$TRTEXEC" \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16

echo ""
echo "Done. Engine saved to: $ENGINE_FILE"
echo ""
echo "Next step: update engine_file_path in config/params.yaml to:"
echo "  $ENGINE_FILE"

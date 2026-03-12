#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_flow_matching.py"
CONFIG_PATH="$PROJECT_ROOT/configs/cifar10_flow_matching_sizes.yaml"
PYTHON_BIN="${PYTHON_BIN:-/tmp/.venvs/gcnn/bin/python}"
EXTRA_ARGS=("$@")

require_file() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
}

require_file "$TRAIN_SCRIPT"
require_file "$CONFIG_PATH"

"$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --config "$CONFIG_PATH" \
    --model p4munet \
    --model-size large \
    --epochs 500 \
    --batch-size 256 \
    --eval-batch-size 256 \
    --num-workers 4 \
    --log-interval 25 \
    --sample-every-epochs 5 \
    --num-eval-samples 16 \
    --num-final-samples 16 \
    --sample-batch-size 8 \
    --num-sample-steps 30 \
    --wandb \
    --wandb-name cifar10-p4munet-large \
    --save-last-checkpoint "$PROJECT_ROOT/checkpoints/cifar10_flow_matching/cifar10_p4munet_large_last.pt" \
    --save-best-checkpoint "$PROJECT_ROOT/checkpoints/cifar10_flow_matching/cifar10_p4munet_large_best.pt" \
    --samples-dir "$PROJECT_ROOT/samples/flow_matching/cifar10_p4munet_large" \
    "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train.py"
CIFAR_CONFIG_PATH="$PROJECT_ROOT/configs/cifar10_sizes.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
CIFAR_EPOCHS="${CIFAR_EPOCHS:-300}"
CIFAR_MODEL_SIZE="${CIFAR_MODEL_SIZE:-base}"
CIFAR_SEED="${CIFAR_SEED:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/cifar10_avgpool}"
MODELS=("resnet" "p4resnet" "p4mresnet")
EXTRA_ARGS=("$@")

require_file() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
}

run_train() {
    local model="$1"
    local checkpoint_dir="$2"
    local run_name="$3"

    mkdir -p "$checkpoint_dir"

    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --config "$CIFAR_CONFIG_PATH" \
        --model "$model" \
        --model-size "$CIFAR_MODEL_SIZE" \
        --downsample-mode avgpool \
        --epochs "$CIFAR_EPOCHS" \
        --seed "$CIFAR_SEED" \
        --wandb \
        --wandb-name "$run_name" \
        --save-last-checkpoint "$checkpoint_dir/last.pt" \
        --save-best-checkpoint "$checkpoint_dir/best.pt" \
        "${EXTRA_ARGS[@]}"
}

require_file "$TRAIN_SCRIPT"
require_file "$CIFAR_CONFIG_PATH"

for model in "${MODELS[@]}"; do
    run_name="cifar10_${model}_${CIFAR_MODEL_SIZE}_avgpool_seed${CIFAR_SEED}"
    checkpoint_dir="$CHECKPOINT_ROOT/${model}/${CIFAR_MODEL_SIZE}/seed_${CIFAR_SEED}"

    echo "=== cifar10 model=$model size=$CIFAR_MODEL_SIZE downsample=avgpool seed=$CIFAR_SEED ==="

    run_train "$model" "$checkpoint_dir" "$run_name"
done

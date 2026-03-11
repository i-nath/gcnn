#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train.py"
MNIST_CONFIG_PATH="$PROJECT_ROOT/configs/mnist_rot_sizes.yaml"
CIFAR_CONFIG_PATH="$PROJECT_ROOT/configs/cifar10_sizes.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
MNIST_EPOCHS="${MNIST_EPOCHS:-100}"
CIFAR_EPOCHS="${CIFAR_EPOCHS:-300}"
NUM_RUNS="${NUM_RUNS:-10}"
MNIST_MODEL_SIZE="${MNIST_MODEL_SIZE:-base}"
CIFAR_MODEL_SIZE="${CIFAR_MODEL_SIZE:-base}"
CIFAR_SEED="${CIFAR_SEED:-0}"
MODELS=("resnet" "p4resnet" "p4mresnet")
EXTRA_ARGS=("$@")

require_file() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
}

generate_seed() {
    od -An -N4 -tu4 /dev/urandom | tr -d ' '
}

run_train() {
    local config_path="$1"
    local model="$2"
    local model_size="$3"
    local checkpoint_dir="$4"
    local run_name="$5"
    shift 5

    mkdir -p "$checkpoint_dir"

    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --config "$config_path" \
        --model "$model" \
        --model-size "$model_size" \
        --wandb \
        --wandb-name "$run_name" \
        --save-last-checkpoint "$checkpoint_dir/last.pt" \
        --save-best-checkpoint "$checkpoint_dir/best.pt" \
        "${EXTRA_ARGS[@]}" \
        "$@"
}

require_file "$TRAIN_SCRIPT"
require_file "$MNIST_CONFIG_PATH"
require_file "$CIFAR_CONFIG_PATH"

for model in "${MODELS[@]}"; do
    for ((run_idx = 1; run_idx <= NUM_RUNS; run_idx++)); do
        seed="$(generate_seed)"
        run_name="mnist-rot_${model}_${MNIST_MODEL_SIZE}_seed${seed}"
        checkpoint_dir="$PROJECT_ROOT/checkpoints/mnist_rot_sizes/${model}/${MNIST_MODEL_SIZE}/seed_${seed}"

        echo "=== mnist-rot model=$model size=$MNIST_MODEL_SIZE run=$run_idx/$NUM_RUNS seed=$seed ==="

        run_train \
            "$MNIST_CONFIG_PATH" \
            "$model" \
            "$MNIST_MODEL_SIZE" \
            "$checkpoint_dir" \
            "$run_name" \
            --epochs "$MNIST_EPOCHS" \
            --seed "$seed"
    done
done

for model in "${MODELS[@]}"; do
    run_name="cifar10_${model}_${CIFAR_MODEL_SIZE}_seed${CIFAR_SEED}"
    checkpoint_dir="$PROJECT_ROOT/checkpoints/cifar10_sizes/${model}/${CIFAR_MODEL_SIZE}/seed_${CIFAR_SEED}"

    echo "=== cifar10 model=$model size=$CIFAR_MODEL_SIZE seed=$CIFAR_SEED ==="

    run_train \
        "$CIFAR_CONFIG_PATH" \
        "$model" \
        "$CIFAR_MODEL_SIZE" \
        "$checkpoint_dir" \
        "$run_name" \
        --epochs "$CIFAR_EPOCHS" \
        --seed "$CIFAR_SEED"
done

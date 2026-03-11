#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/workspace/gcnn"
VENV_DIR="/tmp/.venvs/gcnn"

mkdir -p "$(dirname "$VENV_DIR")"

uv venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

uv sync --project "$PROJECT_ROOT" --python "$VENV_DIR/bin/python"

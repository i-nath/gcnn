#!/usr/bin/env bash

GCNN_ROOT="/workspace/gcnn"
GCNN_VENV="/tmp/.venvs/gcnn"
GCNN_ACTIVATE="$GCNN_VENV/bin/activate"

if [[ -f "$GCNN_ACTIVATE" ]]; then
    if [[ "${VIRTUAL_ENV:-}" != "$GCNN_VENV" ]]; then
        # shellcheck disable=SC1090
        source "$GCNN_ACTIVATE"
    fi
fi

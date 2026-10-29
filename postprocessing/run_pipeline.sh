#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Error: project virtual environment not found at ${PYTHON}" >&2
  echo "Create it with: python3.13 -m venv ${PROJECT_ROOT}/.venv" >&2
  exit 1
fi

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:${DYLD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pogotrack-mpl}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON}" "${SCRIPT_DIR}/run_pipeline.py" "$@"

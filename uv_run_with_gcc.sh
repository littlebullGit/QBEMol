#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOMEBREW_GCC_PREFIX="$(brew --prefix gcc)"
export CC="${HOMEBREW_GCC_PREFIX}/bin/gcc-15"
export CXX="${HOMEBREW_GCC_PREFIX}/bin/g++-15"
export FC="${HOMEBREW_GCC_PREFIX}/bin/gfortran-15"
export PATH="${HOMEBREW_GCC_PREFIX}/bin:${PATH}"

# Ensure Python headers are locatable when building C extensions under GCC
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CPATH="${SDKROOT}/System/Library/Frameworks/Python.framework/Versions/Current/include/python3.12:${CPATH:-}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

if [ "${QBEMOL_SKIP_SYNC:-0}" != "1" ]; then
    echo "[uv] syncing environment (GCC/OpenMP toolchain)"
    uv sync --project "${SCRIPT_DIR}"
else
    echo "[uv] skipping environment sync because QBEMOL_SKIP_SYNC=1"
fi

echo "[uv] rebuilding quemb with GCC/OpenMP"
uv pip install --project "${SCRIPT_DIR}" --force-reinstall --no-deps "${SCRIPT_DIR}/quemb"

echo "[uv] running command with GCC toolchain"
uv run --no-sync "$@"

#!/usr/bin/env bash
# Run the full TomoImageStitcher test suite.
# CPU-only tests run anywhere; the GPU tests are skipped automatically when
# no CUDA device is visible to CuPy.
set -euo pipefail

cd "$(dirname "$0")/.."

# Install the dev extras if not already present.
pip install -e ".[dev]"

# CPU tests (no GPU required)
pytest tests/test_utilities.py tests/test_transform.py -v

# GPU tests. These skip cleanly when no CUDA device is visible (see
# tests/conftest.py), so a non-zero exit here means a genuine failure on a host
# that does have a GPU -- it must not be swallowed, or real regressions on GPU
# machines go unnoticed.
pytest tests/ -v -m gpu

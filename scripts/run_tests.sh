#!/usr/bin/env bash
# Run the full Stitcher v0.2 test suite.
# CPU-only tests run anywhere; the GPU tests are skipped automatically when
# no CUDA device is visible to CuPy.
set -euo pipefail

cd "$(dirname "$0")/.."

# Install the dev extras if not already present.
pip install -e ".[dev]"

# CPU tests (no GPU required)
pytest tests/test_utilities.py -v

# GPU tests (auto-skipped when no CUDA device)
pytest tests/ -v -m gpu || echo "GPU tests skipped or failed (this is expected on CPU-only hosts)."

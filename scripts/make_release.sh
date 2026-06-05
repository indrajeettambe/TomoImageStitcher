#!/usr/bin/env bash
# Build and publish a new release to PyPI.
# Run from the project root, after bumping the version in pyproject.toml.
set -euo pipefail

cd "$(dirname "$0")/.."

# Clean previous builds
rm -rf build/ dist/ *.egg-info src/*.egg-info

# Build
python -m pip install --upgrade build twine
python -m build

# Check
python -m twine check dist/*

# Upload to PyPI (you will be prompted for credentials; consider using a token)
python -m twine upload dist/*

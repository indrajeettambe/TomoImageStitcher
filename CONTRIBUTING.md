# Contributing

Thanks for your interest in improving Stitcher v0.2!  Contributions of all
sizes are welcome — bug reports, documentation fixes, new examples, new
features and refactorings.

## Development setup

1. Fork and clone the repository.
2. Create a fresh environment and install the package in editable mode with
   the `dev` extra:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   pip install cupy-cuda12x        # or whichever wheel matches your CUDA
   ```

3. Verify the install:

   ```bash
   pytest tests/test_utilities.py -v
   ```

## Code style

The project uses:

* **black** with `line-length = 100` for formatting
* **isort** in the `black` profile
* **flake8** for linting (with `E501, W503` ignored — black handles them)

Run all three locally before pushing:

```bash
black src/ tests/ examples/
isort src/ tests/ examples/
flake8 src/ tests/ examples/
```

## Testing

* The CPU-only test suite is in `tests/test_utilities.py` and runs anywhere.
* The GPU test suite is in `tests/test_transform.py` and
  `tests/test_stitcher.py` and is automatically skipped on machines without
  a CUDA device.

Run the full suite:

```bash
pytest tests/ -v
```

## Pull request workflow

1. Open an issue describing the bug or feature (use the templates under
   `.github/ISSUE_TEMPLATE/`).
2. Fork the repository and create a feature branch
   (`git checkout -b my-feature`).
3. Make your change. Add tests for any new functionality.
4. Make sure all tests pass and the linters are happy.
5. Push the branch and open a pull request. Reference the issue in the
   description.

By contributing, you agree that your contributions will be licensed under
the project's MIT license.

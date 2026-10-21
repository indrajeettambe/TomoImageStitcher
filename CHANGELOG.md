# Changelog

All notable changes to TomoImageStitcher are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

* `stitch_volumes_blend_equalize`: the `use_existing_equalize` branch applied the
  measured factors as `(img + intercept) * slope`, which is not the transform the
  fit produces. The fit is `linregress(tile, mosaic)`, i.e.
  `mosaic ~ slope * tile + intercept`, and that is the form used when the factors
  are first computed — but a different one when `stitch_layers` reuses them on the
  full volume. The two agree only when the intercept is 0 or the slope is 1, so
  intensity equalization measured on the representative slices was applied
  incorrectly to the rest of the volume, leaving visible steps at tile boundaries.
  Both branches now use `img * slope + intercept`.
  *(Indrajeet Tambe, 2026-10-19)*

* `stitch_volumes_blend_equalize`: the equalization fit now rejects untrustworthy
  results instead of applying them blindly. Previously any `linregress` output was
  used, including fits driven by the zeros the circular mask leaves inside the
  blend region; on a real open-cell foam series this produced slopes ranging from
  0.007 to 1.6 and correlation coefficients down to 0.014, which were then applied
  to the whole volume. The fit now uses only mutually non-zero samples, and falls
  back to identity when there are fewer than 256 common voxels, when the slope or
  intercept is not finite, when the slope falls outside a generous physical window
  (0.05–20), or when the fit does not actually correlate (`|r| < 0.3`).
  `use_existing_equalize` applies the same validity check to the factors it reads.
  *(Indrajeet Tambe, 2026-10-19)*

* danmax: the optional-import guards caught `BaseException` rather than
  `ImportError`. `BaseException` also covers `KeyboardInterrupt` and
  `SystemExit`, so interrupting a slow import (pyFAI is large, and `lib.misc`
  may sit on a network filesystem at the beamline) was swallowed and reported as
  an unavailable dependency; execution then continued with the helper names
  undefined and failed later as a `NameError` somewhere unrelated. Interrupts
  now abort, and a module that exists but is broken raises its real error.
  *(Indrajeet Tambe, 2026-10-20)*

* `stitch_layers`: each layer's output dataset was sized with that layer's
  padding but with the tile extents of layer 0, image 0. With homogeneous tiles
  the two values coincide, so nothing changed in practice; with tiles that
  differ in y or x between layers, every layer after the first was silently
  cropped or padded to the first one's geometry. The size is now computed once
  per layer, from that layer's own extents, by `mosaic_plane_shape`.
  *(Indrajeet Tambe, 2026-10-21)*

* docs/installation: the documented CPU fallback
  `pip install cupy-cuda11x==13.0 --install-option="--no-cuda"` no longer works.
  pip removed `--install-option` in 23.1 in favour of `--config-settings`, so the
  command now fails with `no such option: --install-option` (verified on
  pip 26.1.2). Replaced with a supported command, and noted that a CPU-only
  CuPy is not equivalent to a CUDA build for this project because the
  correlation engine dispatches to CuPy kernels.
  *(Indrajeet Tambe, 2026-10-19)*

* docs/troubleshooting: added a "Visible intensity steps between tiles" entry.
  Registration that reports a good NCC but a mosaic with seams along tile
  boundaries is usually an equalisation problem rather than a registration one;
  the entry gives the order to check it in, and states plainly that
  equalisation cannot remove a genuine per-tile grey-scale difference, because
  each tile is only ever matched to the part of the mosaic it overlaps.
  *(Indrajeet Tambe, 2026-10-19)*

* tests: the GPU-marked tests now actually skip on a host with no CUDA device.
  `pytest.importorskip("cupy")` only skips when CuPy is *not installed*; on a
  host that has CuPy but no usable GPU the import succeeded and the test then
  died with `CUDARuntimeError: cudaErrorNoDevice`, contradicting its own
  docstring. Verified by running the suite with `CUDA_VISIBLE_DEVICES=""`,
  which failed before this change and skips cleanly after. Adds
  `tests/conftest.py`, which skips `gpu`-marked tests when no device can be
  enumerated. *(Indrajeet Tambe, 2026-10-19)*

* scripts/run_tests.sh: stopped swallowing GPU test failures. The GPU step ended
  in `|| echo "...failed (this is expected on CPU-only hosts)"`, so a genuine
  regression on a real GPU host was reported as success and the script still
  exited 0. Skipping on CPU-only hosts is now handled properly by the conftest
  hook above, so the failure can surface. *(Indrajeet Tambe, 2026-10-19)*

* notebooks/03_stitching_with_rotation: fixed a bug that stopped the notebook
  running at all. A previous edit had merged an import into the end of a
  comment -- `# (removed old sys.path manipulation...)import numpy as np` --
  so NumPy was never imported, while `np` is used ten times in that notebook,
  including in a function-signature default (`data_type=np.uint8`), which
  raises `NameError` at definition time. *(Indrajeet Tambe, 2026-10-19)*

* notebooks/03_stitching_with_rotation: the five absolute DanMAX paths are now
  defined once in an editable block at the top instead of being spread through
  the notebook, where they were neither runnable elsewhere nor obvious to
  change. Also cleared the committed outputs (about 300 kB of base64 images and
  beamline logs that still contained the proposal number in plain text) and
  dropped a trailing empty cell; the notebook shrinks from 360 kB to 37 kB.
  *(Indrajeet Tambe, 2026-10-19)*

* notebooks: added the cell `id` field required by nbformat 4.5+ to
  `01_quickstart` and `02_full_pipeline`, and bumped `01_quickstart` from
  nbformat 4.4 to 4.5. nbformat currently fixes the missing ids silently on
  read but warns that it will become a hard error. All three notebooks now
  validate cleanly with no warnings. *(Indrajeet Tambe, 2026-10-19)*

* packaging: dropped `scikit-image` from the dependencies and `pandas` from
  requirements.txt. Neither is imported anywhere in the package, tests,
  examples or notebooks. The CI dependency list had drifted to match, and has
  been corrected. *(Indrajeet Tambe, 2026-10-19)*

* packaging: the version is now read from `tomo_image_stitcher.__version__`
  via `[tool.setuptools.dynamic]` instead of being written out again in
  pyproject.toml, so bumping one cannot leave the other stale. *(Indrajeet
  Tambe, 2026-10-19)*

* danmax: the guarded imports now report what is actually missing instead of
  printing "Change to HDF5 server to load all modules", which was unrelated to
  the cause. The module still imports without the optional dependencies; only
  the message changed. *(Indrajeet Tambe, 2026-10-19)*

## [0.2.0] - 2025-01-01

Initial public release. Includes:

* 3D ZNCC pixel search on the GPU (multi-stage, mask-aware)
* IC-GN Lucas–Kanade refinement with optional affine / rigid warp
* Chunk-by-chunk affine transform with mask-aware interpolation
* Translation-only stitching path via SimpleITK
* Per-layer batching and rotation-stage support
* Distance-map blending with optional intensity equalization
* Optional DanMAX beamline utilities

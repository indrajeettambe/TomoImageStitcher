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

## [0.2.0] - 2025-01-01

Initial public release. Includes:

* 3D ZNCC pixel search on the GPU (multi-stage, mask-aware)
* IC-GN Lucas–Kanade refinement with optional affine / rigid warp
* Chunk-by-chunk affine transform with mask-aware interpolation
* Translation-only stitching path via SimpleITK
* Per-layer batching and rotation-stage support
* Distance-map blending with optional intensity equalization
* Optional DanMAX beamline utilities

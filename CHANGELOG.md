# Changelog

All notable changes to TomoImageStitcher are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-01

Initial public release. Includes:

* 3D ZNCC pixel search on the GPU (multi-stage, mask-aware)
* IC-GN Lucas–Kanade refinement with optional affine / rigid warp
* Chunk-by-chunk affine transform with mask-aware interpolation
* Translation-only stitching path via SimpleITK
* Per-layer batching and rotation-stage support
* Distance-map blending with optional intensity equalization
* Optional DanMAX beamline utilities

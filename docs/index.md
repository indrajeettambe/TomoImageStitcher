# TomoImageStitcher — Documentation

GPU-accelerated, sub-pixel accurate 3D volumetric stitcher for tomographic
and large-volume microscopy datasets. Originally developed for stitching
local X-ray tomography volumes for the Experiment at the **DanMAX**
beamline, Sweden.

## Start here

The best way to learn the pipeline is the **detailed walkthrough
notebook** — it covers all six stages with explanations, parameter
choices, and visualisations on a synthetic dataset:

- [`../notebooks/02_full_pipeline.ipynb`](../notebooks/02_full_pipeline.ipynb)

The mathematical details of the registration (ZNCC, IC-GN Lucas–Kanade,
mask weighting) and blending (distance-map weighting) are described in
[PUBLICATION].

## Contents

| Page                                          | Description |
|-----------------------------------------------|-------------|
| [Installation](installation.md)               | Step-by-step install: drivers, CuPy, the package itself. |
| [Quickstart](quickstart.md)                   | Copy-paste recipes for the most common workflows. |
| [Architecture](architecture.md)               | Data structures, pipeline stages, coordinate conventions. |
| [API reference](api.md)                       | Public classes, methods, and parameters. |
| [Troubleshooting](troubleshooting.md)         | Common errors and how to recover. |

## Quick links

- [GitHub repository](https://github.com/indrajeettambe/TomoImageStitcher)
- [Project layout](../README.md#project-layout)
- [Citation](../README.md#citation)
- [Detailed walkthrough notebook](../notebooks/02_full_pipeline.ipynb)

## Why TomoImageStitcher?

Most off-the-shelf stitching tools (ImageJ/Fiji Grid/Collection stitching,
BigStitcher, etc.) are designed for 2D tiles. TomoImageStitcher is built
specifically for **3D sub-volumes** with the following goals in mind:

- Sub-pixel registration via a ZNCC pixel search followed by an
  inverse-compositional Gauss–Newton (IC-GN) Lucas–Kanade refinement.
- GPU acceleration of every heavy step (correlate, Lucas–Kanade, Gaussian
  filtering, affine transform) through CuPy.
- Mask-aware interpolation so that background (zero) pixels never bleed
  into the foreground when blending.
- Affine or rigid transformations per pair, with optional extraction of
  the rigid component.
- Per-layer batching to deal with stage-z (height) stratification and
  rotation stages.
- Intensity equalisation through joint histograms of overlapping regions.

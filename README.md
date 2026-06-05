# TomoImageStitcher

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CuPy](https://img.shields.io/badge/GPU-CuPy-76B900.svg)](https://cupy.dev/)

A GPU-accelerated, sub-pixel accurate 3D volumetric stitcher for tomographic and
large-volume microscopy datasets. TomoImageStitcher registers overlapping 3D
sub-volumes acquired on a translation (and optionally rotation) stage and
produces a single seamless volume with mask-aware blending and optional
intensity equalization.

> Originally developed at the **DanMAX** beamline (MAX IV Laboratory, Sweden)
> for stitching of X-ray tomography reconstructions and projection volumes.

---

## Table of contents

- [Why TomoImageStitcher?](#why-tomoimagestitcher)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Citation](#citation)
- [Contributors](#contributors)
- [License](#license)

---

## Why TomoImageStitcher?

Most off-the-shelf stitching tools (ImageJ/Fiji Grid/Collection stitching,
BigStitcher, etc.) are designed for 2D tiles. TomoImageStitcher is built
specifically for **3D sub-volumes** with the following goals in mind:

- **Sub-pixel registration** via a ZNCC pixel search followed by an
  Inverse-Compositional Gauss–Newton (IC-GN) Lucas–Kanade refinement.
- **GPU acceleration** of every heavy step (correlate, Lucas–Kanade, Gaussian
  filtering, affine transform) through CuPy.
- **Mask-aware interpolation** so that background (zero) pixels never bleed
  into the foreground when blending.
- **Affine or rigid** transformations per pair, with optional extraction of
  the rigid component.
- **Per-layer batching** to deal with stage-z (height) stratification and
  rotation stages.
- **Intensity equalization** through joint histograms of overlapping regions.

---

## Key features

| Feature | Description |
|---|---|
| 3D ZNCC pixel search | Multi-stage downscaling correlation on overlapping intersections |
| Lucas–Kanade refinement | IC-GN optimiser with optional affine or rigid warp |
| Mask-aware correlation | Eroded binary mask removes interpolation artefacts at the borders |
| Affine transform | Large-volume affine warp chunk-by-chunk on the GPU |
| Translation-only path | SimpleITK-based shift for fast, memory-cheap stitching |
| Layered stitching | Classifies sub-volumes into `z`-layers automatically |
| Intensity equalization | Linear histogram matching in the overlap region |
| Distance-map blending | Smooth radial / squared / directional blend with `alpha` exponent |
| HDF5 I/O | Reads NeXus-style, DanMAX-style and generic h5 layouts |
| Save intermediate data | Registration results, layer metadata, etc. |

---

## How it works

The pipeline has four main stages:

```
                    ┌────────────────────┐
   list of .h5  ──▶ │  1. Organize       │  classify into z-layers,
   files + motor     │     sub-volumes    │  compute global pad, find
   coordinates       └─────────┬──────────┘  intersections
                                │
                                ▼
                    ┌────────────────────┐
                    │  2. Correlate      │  ZNCC pixel search + LK
                    │     intersections  │  for every neighbour pair
                    └─────────┬──────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │  3. Accumulate     │  build displacement pyramid,
                    │     displacements  │  accumulate, prune by NCC
                    └─────────┬──────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │  4. Stitch & blend │  distance-map blending,
                    │     the volume     │  optional equalization
                    └────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full description of
the data structures used between steps.

---

## Installation

### Requirements

* Python **3.9+**
* An **NVIDIA GPU** with CUDA 11+ (CuPy 13+)
* 16 GB+ of GPU memory recommended for large overlaps

### 1. Clone the repository

```bash
git clone https://github.com/indrajeettambe/TomoImageStitcher.git
cd TomoImageStitcher
```

### 2. (Optional) create a clean environment

```bash
conda create -n tomo-image-stitcher python=3.10
conda activate tomo-image-stitcher
```

### 3. Install the package

```bash
pip install -e .
```

This installs both the Python package `tomo_image_stitcher` and the optional
beamline utilities (`tomo_image_stitcher.danmax`).

### 4. Install CuPy matching your CUDA version

`pip install cupy-cuda12x` is the most common choice — pick the wheel that
matches your CUDA toolkit, see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

Verify the install:

```python
import tomo_image_stitcher, cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # should be > 0
```

---

## Quick start

The example below assumes you have two overlapping `.h5` volumes and the
motor coordinates of their centres. The example with rotation is in
`notebooks/03_stitching_with_rotation.ipynb`.

```python
import numpy as np
from tomo_image_stitcher import Stitcher

# 1) Where are the volumes?
file_path_list = [
    "/data/experiment/scan-0001_recon.h5",
    "/data/experiment/scan-0002_recon.h5",
]

# 2) Motor positions in mm (one row per file)
motor_positions = np.array([
    [ 0.0,  0.0, 0.0],
    [ 0.5,  0.0, 0.0],
])

# 3) Spatial calibration (mm / voxel)
mm_per_voxel = 0.00065

# 4) Map motor axes → image axes (1=x, 2=y, 3=z, sign is allowed)
x_y_z_correspondance = (1, 2, 3)

# 5) Build the stitcher
st = Stitcher(
    file_path_list=file_path_list,
    physical_coordinates=motor_positions,
    mm_per_voxel=mm_per_voxel,
    x_y_z_correspondance=x_y_z_correspondance,
    saving_path="/data/experiment/stitching",
)

# 6) Run the pipeline
st.get_layers_in_z(tolerance_mm=2)     # classify by z
st.get_padding()                      # compute global pad
st.get_intersections()                # find overlap regions
st.compute_shift_in_layers(           # register every pair
    start_slice=st.img_depth // 2 - 2,
    end_slice=st.img_depth // 2 + 2,
    mask=True, mask_radius=500,
    downscale=2, downscale_stages=2,
    apply_affine_warp=True, keep_rigid_only=True,
)
st.get_displacement_pyramid(starting_coord=(0, 0, 0))
st.accumulate_displacement(exclude_NCC=50, weighted_avg=False, affine_operator=True)
st.compose_final_displacements()
st.push_stitch_parameters()
st.stitch_layers(chunk_size_series=40, chunk_size_parallel=5, n_cores=8)
```

The final stitched volume is written to
`<saving_path>/Stitched_layers/Layer_<i>.h5`.

For a full walk-through, open `notebooks/01_quickstart.ipynb`.

---

## Documentation

* [`docs/installation.md`](docs/installation.md) — detailed installation
  instructions including GPU setup.
* [`docs/quickstart.md`](docs/quickstart.md) — copy-paste recipes.
* [`docs/architecture.md`](docs/architecture.md) — design notes & data flow.
* [`docs/api.md`](docs/api.md) — auto-generated reference of public classes.
* [`docs/troubleshooting.md`](docs/troubleshooting.md) — common errors.

---

## Project layout

```
github-repo/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .gitignore
├── src/
│   └── tomo_image_stitcher/
│       ├── __init__.py
│       ├── stitcher.py        # main Stitcher class
│       ├── registration.py    # RegistrationKIT (ZNCC + IC-GN Lucas–Kanade)
│       ├── transform.py       # chunk-wise affine transform on GPU
│       └── danmax.py          # DanMAX beamline utilities (optional)
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_full_pipeline.ipynb
│   └── 03_stitching_with_rotation.ipynb
├── examples/
│   ├── example_2d_projection.py
│   └── example_with_rotation.py
├── tests/
│   ├── test_stitcher.py
│   ├── test_transform.py
│   └── test_utilities.py
├── docs/
│   ├── installation.md
│   ├── quickstart.md
│   ├── architecture.md
│   ├── api.md
│   └── troubleshooting.md
└── .github/
    └── workflows/
        └── tests.yml
```

---

## Citation

A formal citation for TomoImageStitcher is not yet available — the
associated publication is in preparation. Citation details (BibTeX entry)
will be added here once the paper is accepted.

If you use this software in the meantime, please acknowledge the GitHub
repository and the authors listed below.

---

## Contributors

* **Endri Lacaj**
* **Indrajeet Tambe**

Contributions are welcome — please open an issue or pull request.

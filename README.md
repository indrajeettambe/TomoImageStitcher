# TomoImageStitcher

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CuPy](https://img.shields.io/badge/GPU-CuPy-76B900.svg)](https://cupy.dev/)

A GPU-accelerated, sub-pixel accurate **3D volumetric stitcher** for tomographic and
large-volume microscopy datasets. TomoImageStitcher registers overlapping 3D
sub-volumes acquired on a translation (and optionally rotation) stage and produces
a single seamless volume with mask-aware blending and optional intensity
equalisation.

> Originally developed for stitching local X-ray tomography volumes for the Experiment at the **DanMAX** beamline, Sweden.

For a **detailed step-by-step walkthrough** of every pipeline stage with synthetic
data, see [`notebooks/02_full_pipeline.ipynb`](notebooks/02_full_pipeline.ipynb).
For the mathematical details of the registration, see **[PUBLICATION]**.

---

## What is it

TomoImageStitcher stitches together a set of **overlapping 3D sub-volumes**
(typically reconstructed tomography volumes) into **one seamless volume**.
Each stage is a Python call on the `Stitcher` object, so you can inspect and
re-run any stage on its own.

The pipeline runs in **six stages**:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Organise sub-volumes** | Classify into z-layers, compute global padding, find intersections. |
| 2 | **Registration** | ZNCC pixel search + IC-GN Lucas–Kanade refinement per pair. |
| 3 | **Accumulate displacements** | Chain per-pair shifts into a global warp graph (BFS). |
| 4 | **Equalisation** | Match intensities across overlaps via joint histograms. |
| 5 | **Blending** | Distance-map blending onto the global canvas, on the GPU. |
| 6 | **Save and inspect** | Write per-layer `.h5` files with full pipeline metadata. |

The full per-stage walk-through with the code for every step lives in the
**detailed notebook** linked above. The math behind the registration and
blending lives in [PUBLICATION].

---

## Where you can use it

- Stitching **3D X-ray tomography reconstructions** from a multi-tile
  translation scan.
- Stitching **raw projection volumes** (radiographs) before reconstruction.
- Stitching **3D microscopy datasets** (light-sheet, confocal) where individual
  tiles are too large to fit into memory.
- **Multi-scan** stitching where you have several scans with overlapping
  lateral extent.
- Stitching on a **rotation stage** (helical or tomographic) — see
  `notebooks/03_stitching_with_rotation.ipynb`.

---

## Install

TomoImageStitcher is on PyPI. A CUDA-capable GPU with the matching CuPy wheel
is required for the GPU stages; everything else is plain Python.

```bash
# 1. (Recommended) a clean environment
python -m venv .venv && source .venv/bin/activate

# 2. Install with the notebook extras
pip install -U pip
pip install "tomo-image-stitcher[notebook,danmax]"

# 3. Install CuPy matching your CUDA version (CUDA 12.x shown)
pip install cupy-cuda12x
```

If you cannot install `git`, or you are behind a proxy that blocks it:

```bash
pip install https://github.com/indrajeettambe/TomoImageStitcher/archive/main.zip
```

For full instructions (drivers, conda env, troubleshooting) see
[`docs/installation.md`](docs/installation.md).

---

## Quick start

A minimal end-to-end example on synthetic data. The full version with
explanations and intermediate visualisations is in
[`notebooks/02_full_pipeline.ipynb`](notebooks/02_full_pipeline.ipynb).

```python
import numpy as np
from tomo_image_stitcher import Stitcher

# 1. List of .h5 files and their motor positions in millimetres
file_paths    = ["scan_001.h5", "scan_002.h5", "scan_003.h5"]
motor_coords  = np.array([[ 0.0,  0.0,  0.0],
                          [ 0.8,  0.0,  0.0],
                          [ 1.6,  0.0,  0.0]])
mm_per_voxel  = 0.0022                       # 2.2 µm voxels

# 2. Initialise the stitcher
st = Stitcher(file_paths, motor_coords, mm_per_voxel,
              x_y_z_correspondance=(-1, 3, 2))

# 3. Run the six stages
st.get_layers_in_z(tolerance_mm=4)           # (1) Organise
st.get_padding()
st.get_intersections(check=True)
st.compute_shift_in_layers(downscale=4, downscale_stages=4,   # (2) Registration
                           downscale_LC=True, mask=True, mask_radius=300)
st.get_displacement_pyramid(check=False)
st.accumulate_displacement(exclude_NCC=50)   # (3) Accumulate
st.compose_final_displacements()
st.stitch_volumes_blend_equalize(...)        # (4) + (5) Equalise + Blend
st.stitch_layers(path_save="output/")        # (6) Save

# 4. Read the stitched volume
import h5py
with h5py.File("output/Stitched_layers/Layer_0.h5", "r") as f:
    volume = f["stitched_data/stitched_image"][:]
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [`notebooks/02_full_pipeline.ipynb`](notebooks/02_full_pipeline.ipynb) | **Detailed step-by-step walkthrough** of the full pipeline on synthetic data. Start here. |
| [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Minimal 5-line end-to-end example. |
| [`notebooks/03_stitching_with_rotation.ipynb`](notebooks/03_stitching_with_rotation.ipynb) | Original DanMAX rotation-stage example (update paths before running). |
| [`docs/architecture.md`](docs/architecture.md) | Data structures used between pipeline stages. |
| [`docs/api.md`](docs/api.md) | Public classes, methods, and parameters. |
| [`docs/quickstart.md`](docs/quickstart.md) | More copy-paste recipes. |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common errors and how to recover. |

The mathematical details of the registration (ZNCC, IC-GN Lucas–Kanade, mask
weighting) and blending (distance-map weighting) are described in
**[PUBLICATION]**.

---

## Project layout

```
TomoImageStitcher/
├── src/tomo_image_stitcher/    Package source (Stitcher, RegistrationKIT, …)
├── notebooks/                  Jupyter tutorials (start with 02_full_pipeline)
├── examples/                   Standalone Python scripts
├── tests/                      pytest test-suite
├── docs/                       Architecture, API, troubleshooting
├── pyproject.toml              Build & dependency metadata
├── LICENSE                     MIT
└── CITATION.cff                Software citation
```

---

## Citation

If you use TomoImageStitcher in your research, please cite it using the
metadata in [`CITATION.cff`](CITATION.cff). A publication describing the
algorithm is in preparation and will be linked here when available
([PUBLICATION]).

---

## License

MIT — see [`LICENSE`](LICENSE).

## Contributors

TomoImageStitcher was originally developed at the **DanMAX** beamline
(MAX IV Laboratory, Sweden). See the git log for the full list of
contributors.

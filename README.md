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
- [Where you can use it](#where-you-can-use-it)
- [How it works](#how-it-works)
- [Pipeline stages in detail](#pipeline-stages-in-detail)
- [Install without cloning](#install-without-cloning)
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

## Where you can use it

TomoImageStitcher was built for stitching 3D sub-volumes.
The core is a sub-pixel registration engine.
It runs on the GPU.
The use cases below share the same need.
A single scan is too small.
A single scan is too low resolution.
Many scans tile the full region.
TomoImageStitcher joins them.

### Synchrotron X-ray tomography and microtomography

Modern synchrotrons produce 3D volumes at sub-micron resolution.
A single field of view rarely covers a full sample.
The motor stage moves the sample in small steps.
Each step records a sub-volume.
Sub-volumes overlap by a small amount.
TomoImageStitcher registers each overlap.
It builds the full 3D image from the tiles.

TomoImageStitcher was developed at the DanMAX beamline.
DanMAX is at the MAX IV Laboratory in Sweden.
The pipeline runs on the beamline compute nodes.
It handles translation and rotation acquisitions.
Mask-aware blending avoids background bleed.
The output is a single seamless volume.
The pipeline works with both transmission and fluorescence data.

### Battery research

Li-ion batteries need non-destructive 3D imaging.
Researchers study electrodes, separators, and current collectors.
Synchrotron microtomography gives the resolution needed.
Multiple sub-volumes tile the full cell.
TomoImageStitcher reconstructs the full cell.
It supports coin, pouch, and cylindrical formats.
The output is ready for segmentation and analysis.
Internal short circuits and dendrite growth can be studied.

### Materials science and metallurgy

Engineered materials have microstructure at the micron scale.
A single scan may not capture a representative volume.
Stitcher tiles sub-volumes for a larger region.
This gives statistically meaningful data.
It is useful for composites, alloys, and porous media.
The pipeline works for fibers and laminates.
Additive manufacturing parts can be inspected.
The output feeds into digital twins and FEM models.

### Semiconductor and electronics inspection

Modern chips contain hybrid bonds and through-silicon vias.
These features sit at the sub-micron scale.
Nano-CT imaging captures the internal structure.
Multiple scans tile the area of interest.
TomoImageStitcher combines them into one 3D model.
The output supports failure analysis.
It also supports process control.
Bond integrity and TSV alignment can be verified.

### Geology, petrophysics, and geo-energy

Rock cores contain pores, fractures, and inclusions.
Synchrotron tomography reveals the internal structure.
Large cores need many overlapping scans.
Stitcher combines them into a single volume.
The output helps with reservoir characterisation.
It supports CO₂ storage research.
It also supports oil and gas extraction studies.
Paleontology and soil science benefit too.

### Life science and cleared-tissue microscopy

The pipeline generalises to other 3D imaging modalities.
Light-sheet microscopy produces terabyte-sized volumes.
Many tiles are acquired across a cleared sample.
TomoImageStitcher assembles them into one 3D image.
This is useful for whole-organ imaging.
It also works for cleared tissue and expanded samples.
Mask-aware blending handles low-signal regions.
The output feeds into neuron tracing and cell counting.

### In vivo skin and dermatology

Reflectance confocal microscopy produces 3D stacks.
Multiphoton microscopy adds molecular contrast.
A single stack rarely covers a clinical lesion.
TomoImageStitcher combines adjacent stacks.
Sub-pixel registration matters for diagnosis.
The pipeline supports motion-tolerant reconstruction.
It enables histology-like views of larger areas.
Melanoma, vitiligo, and other conditions can be studied.

### Optical coherence tomography (OCT)

OCT produces high-speed 3D volumes of tissue.
The lateral field of view is limited by the optics.
Multiple volumes can be stitched to image larger areas.
The same ZNCC plus Lucas–Kanade engine is used.
This works for retinal and dermal OCT.
It also works for coronary and endoscopic OCT.
The output is a panoramic 3D view.

### Industrial quality control and non-destructive testing

Manufactured parts may have internal defects.
TomoImageStitcher works for non-destructive testing.
The pipeline runs on laboratory nano-CT systems.
It supports offline and online workflows.
The output is suitable for metrology.
It also supports defect reporting.
Aerospace, automotive, and additive parts can be inspected.

### Cultural heritage and palaeontology

Museum objects and fossils are often fragile.
They cannot be cut or sectioned.
Synchrotron tomography reveals their internal structure.
Stitcher joins many sub-volumes.
The output is a digital twin.
It supports conservation and study.
Paintings, mummies, and fossils can be imaged.

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

## Pipeline stages in detail

Each stage below is a Python call.
Each stage is independent and can be inspected.
The full reference lives in [`docs/api.md`](docs/api.md).

### 1. Organise sub-volumes

The pipeline starts with a list of `.h5` files.
Each file holds one 3D sub-volume.
Each file also has motor coordinates.
The motor positions are in millimetres.
The pipeline reads the volumes into memory.
It reads only the slices it needs.

The pipeline classifies sub-volumes into z-layers.
Sub-volumes at the same height belong to the same layer.
Layers are processed one at a time.
This avoids wasted work on non-overlapping regions.
It also matches how beamline scans are acquired.

The pipeline computes the global padding.
It finds the bounding box of all sub-volumes.
It finds the intersections between neighbours.
Each intersection is a small overlapping 3D block.
These intersections are the units of registration.

### 2. Registration

Registration finds the displacement between two sub-volumes.
It runs once per neighbour pair.
Each call uses the ZNCC pixel search first.
ZNCC stands for Zero-mean Normalized Cross-Correlation.
ZNCC is robust to intensity offsets.
ZNCC is also robust to global scaling.
The search is performed in Fourier space on the GPU.

ZNCC runs at multiple scales.
It starts at the coarsest scale.
It steps down to the finest scale.
Each step refines the result of the previous step.
This is called a coarse-to-fine search.
The coarse stage handles large displacements.
The fine stage handles small displacements.
Sub-pixel accuracy is the final result.

The Lucas–Kanade step refines the result further.
The Lucas–Kanade variant used is IC-GN.
IC-GN stands for Inverse-Compositional Gauss–Newton.
The Hessian is computed once on the template.
The template is warped at every iteration.
Each iteration updates the warp parameters.
The loop runs until convergence.

The warp can be a translation.
The warp can also be an affine.
An affine has 12 parameters.
A rigid transform has 6 parameters.
The `keep_rigid_only` flag extracts the rigid part.
The rigid part is computed by polar decomposition.
Affine and rigid warps both work on the GPU.

Mask-aware correlation is a key feature.
A circular mask can be applied before correlating.
The mask excludes background pixels.
This avoids spurious correlations on empty regions.
The mask can be eroded with a structuring element.
Erosion removes pixels close to the object border.

Each registration call returns:
* a 3D displacement `(dx, dy, dz)` in voxels
* an optional 4×4 affine operator
* a final Normalized Cross-Correlation value
* a count of valid correlation samples

The NCC value is a quality metric.
It is between -1 and 1.
1 means perfect agreement.
0 means no agreement.
-1 means inverted contrast.
The pipeline can drop low-NCC pairs.
The threshold is set by `exclude_NCC`.

### 3. Accumulate displacements

Every neighbour pair has a displacement.
The pipeline needs global displacements per sub-volume.
A breadth-first search builds a graph.
One sub-volume is the seed.
Each connected sub-volume gets a global shift.
The shifts are accumulated along the path.

A weighted average can be used.
Weights come from the NCC values.
Better registrations contribute more.
A switch `weighted_avg=False` uses the best NCC instead.

Affine operators can also be chained.
A 4×4 matrix is composed at each step.
The composition is on the GPU.
This gives a full 6-DoF pose per sub-volume.
The pose can be used for affine stitching.

Bad correlations are pruned.
Pyramid sub-layers can be discarded.
The `exclude_NCC` threshold filters low-quality pairs.
The remaining displacements are trusted.

### 4. Equalisation

Adjacent scans often have different intensities.
The X-ray flux can change between scans.
The detector dark current can drift.
The sample can absorb differently in different regions.
This creates visible seams in the overlap.

Equalisation removes these seams.
It runs in the overlap region of every neighbour pair.
It builds a joint histogram of the two intensities.
It fits a linear map between the two scales.
It applies the map to one side of the overlap.
The result is a smooth intensity transition.

Equalisation is optional.
It is enabled with `use_equalize=True`.
It can also reuse a previous fit.
`use_existing_equalize=True` skips the fit step.
This is useful for time-lapse datasets.
It is also useful for re-running a failed run.

The equalised data feeds into blending.
Equalisation is one of two steps.
Blending is the other.
Equalisation and blending work together.
Equalisation removes the intensity offset.
Blending removes the spatial seam.

### 5. Blending

Blending combines the sub-volumes into one volume.
Each sub-volume has a final global shift.
The shifts come from the accumulation stage.
The pipeline writes the sub-volumes into one big canvas.
The canvas is the size of the global padding box.
The sub-volumes sit inside this canvas.

A distance map is built for each sub-volume.
The map is large near the centre of the sub-volume.
The map is small near the borders.
Pixels far from the border get a high weight.
Pixels close to the border get a low weight.
The weight controls how much each pixel contributes.

The final value at each voxel is a weighted sum.
Weights come from the distance maps of all contributors.
The result is a smooth transition between sub-volumes.
No visible seams remain.
The transition width is controlled by `alpha`.
Larger `alpha` gives sharper transitions.
Smaller `alpha` gives smoother transitions.

The distance function is configurable.
A radial function gives a circular falloff.
A squared function gives a Chebyshev falloff.
The `prop_x_y` parameter controls direction.
`(0, 0)` means no direction preference.
`(1, 0)` propagates along the x-axis.
`(0, 1)` propagates along the y-axis.
`(1, 1)` uses both axes equally.

Blending is mask-aware.
Background pixels (value 0) never bleed in.
The mask is resampled with the same interpolator.
A separate mask distance map is used.
The output honours the original object silhouette.

The pipeline supports two blending paths.
The translation path is fast and memory-cheap.
It uses SimpleITK for warping.
It does not use the GPU for the warp.
It works well for small overlaps.

The affine path handles rotation and shear.
It uses a chunk-by-chunk GPU affine warp.
Each chunk is a slab in the z-direction.
Chunk size is controlled by `chunk_size_series`.
The number of parallel chunks is `chunk_size_parallel`.
This is where the GPU shines.
A typical run is 10× to 100× faster than CPU.
The output is one `.h5` per layer.

### 6. Save and inspect

The pipeline writes intermediate results.
You can inspect them after each stage.
`check_padding` shows a 2D sanity view.
`check_intersection` shows the overlap region.
`save_reg=True` writes the registered slices.
These help you debug bad correlations.

The final output is one `.h5` per layer.
The path is `<saving_path>/Stitched_layers/Layer_<i>.h5`.
The file holds the stitched volume.
The file also holds the pipeline metadata.
The metadata includes all shifts and operators.
You can re-run blending from the metadata alone.

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

## Install without cloning

You do not need to clone the repo to use TomoImageStitcher.
Pick the option that matches your environment.

### Option 1: Install from PyPI (recommended for users)

```bash
pip install tomo-image-stitcher
```

This pulls the latest release from PyPI.
It installs the `tomo_image_stitcher` Python package.
It also installs the `tomo_image_stitcher.danmax` beamline helpers.
No `git` is needed.
No manual download is needed.

### Option 2: Install directly from GitHub (no clone)

If the package is not on PyPI yet, or you want the latest commit on `main`:

```bash
pip install git+https://github.com/indrajeettambe/TomoImageStitcher.git
```

`git` must be installed on the machine.
pip runs `git clone` internally; you do not run it.
You can pin to a tag, a branch, or a commit:

```bash
# a specific tag
pip install git+https://github.com/indrajeettambe/TomoImageStitcher.git@v0.2.0
# a specific branch
pip install git+https://github.com/indrajeettambe/TomoImageStitcher.git@main
# a specific commit (reproducible)
pip install git+https://github.com/indrajeettambe/TomoImageStitcher.git@<commit-sha>
```

You can pull the optional extras the same way:

```bash
pip install "tomo-image-stitcher[notebook,danmax] @ git+https://github.com/indrajeettambe/TomoImageStitcher.git"
```

### Option 3: Install from a ZIP archive (no git at all)

If you cannot install `git`, or you are behind a proxy that blocks it:

```bash
pip install https://github.com/indrajeettambe/TomoImageStitcher/archive/main.zip
```

pip downloads the source tarball directly.
It runs the build and install in one step.
This works on any system with Python and pip.

### Option 4: Use the Docker image (handles CUDA + deps)

For beamline or HPC deployments where the CUDA stack is fixed:

```bash
docker pull ghcr.io/indrajeettambe/tomo-image-stitcher:latest
docker run --gpus all -it ghcr.io/indrajeettambe/tomo-image-stitcher:latest
```

The image is based on `nvidia/cuda:12.x-runtime`.
It includes Python, CuPy, SimpleITK, and the package.
It needs the NVIDIA Container Toolkit on the host.
Tagged releases (`v0.2.0`, `v0.2`, `latest`) and `main` are published.
See `docker/` in the repo for the build files.

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

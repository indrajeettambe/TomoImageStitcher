# Installation

This page covers everything you need to install TomoImageStitcher, including the
GPU drivers and CuPy wheel that match your CUDA toolkit.

## 1. System requirements

| Item            | Minimum                | Recommended                  |
|-----------------|------------------------|------------------------------|
| Python          | 3.9                    | 3.10 or 3.11                 |
| OS              | Linux x86_64           | Linux x86_64 (Ubuntu 20.04+) |
| GPU             | NVIDIA, 6 GB VRAM      | NVIDIA, 16 GB+ VRAM          |
| CUDA            | 11.x                   | 12.x                         |
| cuDNN           | 8.x                    | 8.6+                         |
| Free disk space | 500 MB                 | 5 GB+ (for very large runs)  |

> **macOS / Windows** — the code is portable, but only the Linux build of
> CuPy is officially supported. On Windows you can install the CPU-only
> dependencies and the registration engine will fall back to CuPy-as-numpy.

## 2. Create a clean Python environment

We strongly recommend isolating TomoImageStitcher in its own environment to avoid
clashes with the rest of your Python installation.

```bash
# Option A — conda
conda create -n tomo-image-stitcher python=3.10
conda activate tomo-image-stitcher

# Option B — venv
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install CUDA drivers and toolkit

If you already have `nvidia-smi` working on your machine, you can skip this
step. Otherwise follow NVIDIA's [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Check what is already available:

```bash
nvidia-smi
nvcc --version
```

The CUDA version reported by `nvidia-smi` is the **runtime** version — it is
the version you should match when installing CuPy.

## 4. Install the TomoImageStitcher package

Clone the repository and install in editable mode, choosing the CuPy extra that
matches the CUDA runtime reported by `nvidia-smi`:

```bash
git clone https://github.com/indrajeettambe/TomoImageStitcher.git
cd TomoImageStitcher

# CUDA 12.x
pip install -e ".[gpu12]"

# or CUDA 11.x
pip install -e ".[gpu11]"
```

CuPy is an extra rather than a normal dependency on purpose: each wheel is built
against one CUDA version, so a single default would install a wheel that cannot
work for anyone on a different driver. Importing it would succeed and the first
kernel launch would fail with `cudaErrorInsufficientDriver`.

`pip install -e .` on its own installs the package without CuPy. That is enough
for the CPU-only test suite, but the GPU stages will not run.

If you plan to use the DanMAX beamline utilities or the example notebooks, add
those extras too:

```bash
pip install -e ".[gpu12,notebook,danmax]"
```

## 5. CuPy wheel reference

Step 4 already installs the right wheel through the `gpu11` or `gpu12` extra.
If you prefer to install CuPy yourself, pick the wheel that matches the runtime
CUDA version:

| CUDA runtime | Via the extra                  | Manual install                  |
|--------------|--------------------------------|---------------------------------|
| 12.x         | `pip install -e ".[gpu12]"`    | `pip install cupy-cuda12x`     |
| 11.x         | `pip install -e ".[gpu11]"`    | `pip install cupy-cuda11x`     |
| CPU fallback | —                              | `conda install -c conda-forge cupy` (see note below) |

Note on the CPU fallback: earlier versions of this page recommended
`pip install cupy-cuda11x==13.0 --install-option="--no-cuda"`. That no longer
works — pip removed `--install-option` in 23.1 in favour of `--config-settings`,
so the command now fails with `no such option: --install-option`. The replacement
above is the supported route. Note also that a CPU-only CuPy is not equivalent
to a CUDA build for this project: the correlation engine dispatches to CuPy
kernels, so the GPU paths still require a CUDA-capable device. Use the CPU
fallback only for import-time checks and for running the non-GPU test suite.

See the [CuPy installation page](https://docs.cupy.dev/en/stable/install.html)
for the full table of supported combinations.

## 6. Verify the installation

```python
>>> import tomo_image_stitcher
>>> tomo_image_stitcher.__version__
'0.2.0'
>>> import cupy as cp
>>> cp.cuda.runtime.getDeviceCount()  # should be >= 1
1
```

If the second command returns `0`, the GPU is not visible to CuPy — check
your CUDA driver and make sure you picked the right wheel.

## Troubleshooting

* **"libcudart.so not found"** — the CuPy wheel does not match your CUDA
  runtime. Re-install the right `cupy-cudaXXx` package.
* **`Stitcher(...)` raises `ImportError: No module named cupy`** — CuPy is an
  extra, not a normal dependency, so a bare `pip install -e .` leaves it out.
  Run `pip install -e ".[gpu12]"`, or `.[gpu11]` if `nvidia-smi` reports CUDA
  11.x.
* **Out of memory when registering large overlaps** — lower
  `equal_crop_xy` or `crop_x/crop_y`, or split the registration in
  sub-volumes along `z`.

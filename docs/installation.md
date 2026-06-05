# Installation

This page covers everything you need to install Stitcher v0.2, including the
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

We strongly recommend isolating Stitcher in its own environment to avoid
clashes with the rest of your Python installation.

```bash
# Option A — conda
conda create -n stitcher python=3.10
conda activate stitcher

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

## 4. Install the Stitcher package

Clone the repository and install in editable mode:

```bash
git clone https://github.com/indrajeettambe/volume-stitcher.git
cd volume-stitcher
pip install -e .
```

This installs the core `stitcher` package.

If you plan to use the DanMAX beamline utilities or the example notebooks,
install the optional extras:

```bash
pip install -e ".[notebook,danmax]"
```

## 5. Install CuPy matching your CUDA version

Pick the wheel that matches the runtime CUDA version reported by
`nvidia-smi`:

| CUDA runtime | Install command                |
|--------------|--------------------------------|
| 12.x         | `pip install cupy-cuda12x`     |
| 11.x         | `pip install cupy-cuda11x`     |
| CPU fallback | `pip install cupy-cuda11x==13.0 --install-option="--no-cuda"` |

See the [CuPy installation page](https://docs.cupy.dev/en/stable/install.html)
for the full table of supported combinations.

## 6. Verify the installation

```python
>>> import stitcher
>>> stitcher.__version__
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
* **`Stitcher(...)` raises `ImportError: No module named cupy`** — the
  optional dependency was not installed. Run
  `pip install cupy-cuda12x` (or matching wheel).
* **Out of memory when registering large overlaps** — lower
  `equal_crop_xy` or `crop_x/crop_y`, or split the registration in
  sub-volumes along `z`.

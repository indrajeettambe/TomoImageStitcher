# Troubleshooting

Common pitfalls and how to recover from them.

## GPU is not detected

```
cupy.cuda.runtime.CUDARuntimeError: cudaErrorNoDevice: no CUDA-capable device is detected
```

* Run `nvidia-smi` and verify the GPU is visible to the OS.
* Re-install the CuPy wheel that matches the CUDA runtime version reported
  by `nvidia-smi` (e.g. `cupy-cuda12x` for CUDA 12).
* Inside a Docker container, make sure to pass `--gpus all` and the NVIDIA
  Container Toolkit is installed.

## "All the stacks must have the same number of slices in z."

`get_image_space_coordinates` enforces that all input volumes have the same
depth. Pad the shorter ones with zeros (and update the motor coordinates
accordingly) before running the pipeline.

## "The image must be a 3D numpy array!" / H5 layout not recognised

The `Utilities.H5MaxIV.reader` tries a list of common layouts:

* `exchange/data` — ForMAX / HZB
* `image` — generic
* `stitched_data/stitched_image` — outputs of a previous Stitcher run
* `entry/instrument/zyla/data` — a NeXus layout

If your data uses a different layout, either:

1. Edit `Utilities.H5MaxIV.reader` to add your path, or
2. Pre-process your data to one of the supported layouts.

## ZNCC never exceeds the `exclude_NCC` threshold

Possible causes:

* The motors were mis-calibrated — check `check_padding` for obvious
  misalignments.
* The mask radius is too small and only background is being correlated.
* The downscale is too aggressive — try `downscale=0.5` with
  `downscale_stages=2` instead of `downscale=0.25, downscale_stages=1`.
* The features in the overlap are too fine — try
  `apply_mean_filter_zyx=(3, 3, 3)`.

## Out-of-memory during registration

Lower the per-pair data volume sent to the GPU:

* Increase `mask_radius` to exclude border regions.
* Use `equal_crop_xy=N` to send only a centered square of side `N`.
* Crop in z with `start_slice=` and `end_slice=`.
* Disable the affine step (`apply_affine_warp=False`) and use
  translation-only registration.

## Stitched volume has hard edges

Try:

* `alpha=1` for a smooth distance-map blend, or higher for sharper edges.
* `use_equalize=True` to apply linear intensity matching in the overlap.
* `prop_x_y=(0, 0)` (Chebyshev) for "square" distance propagation, useful
  when the slices are arranged on a regular grid.

## Stitcher hangs at `stitch_volumes_blend_equalize_parallel`

`multiprocessing.Pool` is not compatible with CuPy on some setups — the
code uses a `ThreadPool` instead, but you may still hit the global
interpreter lock. Lower `n_cores` or use a single process.

## Rotation-stage stitching: angles are off

Verify that the motor coordinates you pass to `Stitcher` are in millimetres
*and* degrees (or both, in a consistent unit). The pipeline does not
auto-detect; you control the `x_y_z_correspondance` mapping.

## I see "float16 not supported on this device"

Force `float32` everywhere by setting:

```python
os.environ["CUPY_DTYPE_POLICY"] = "strict"
```

before importing CuPy.

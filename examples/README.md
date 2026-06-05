# Examples

This directory contains runnable Python scripts that demonstrate the main
features of Stitcher v0.2 on **synthetic** data. The point is to verify the
install and show the API surface — the original beamline data is too large
to ship here.

| File | Description |
|---|---|
| `example_2d_projection.py`    | Stitch two translation-stage sub-volumes. |
| `example_with_rotation.py`   | Stitch a small 2D grid of sub-volumes.   |

## Running an example

```bash
# 1) Activate your environment
conda activate stitcher

# 2) Run
python examples/example_2d_projection.py
```

You should see something like:

```
Stitched volume shape: (16, 256, 624), dtype: float32
```

## Adapting to your own data

1. Replace the `write_synthetic_volume` calls with paths to your real `.h5`
   files.
2. Set `motor_positions` to a `(N, 3)` array of your motor coordinates in mm.
3. Set `mm_per_voxel` to the spatial calibration of your detector.
4. Set `x_y_z_correspondance` to map motor axes to image axes (and to flip
   them if needed).
5. Run!

See `docs/quickstart.md` for the full list of knobs and `docs/api.md` for
the API reference.

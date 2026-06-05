# Quickstart

This page is a copy-paste tour of the most common Stitcher v0.2 workflows.
Every snippet below uses synthetic data so you can verify the install without
needing real `.h5` files.

## 0. Generate fake data (only if you don't have any)

```python
import h5py
import numpy as np

def fake_h5(path, shape, value_offset=0):
    """Write a constant-filled 3D volume to ``path`` and return the dataset."""
    with h5py.File(path, "w") as fh:
        return fh.create_dataset("image", data=np.full(shape, value_offset, dtype=np.uint16))
```

## 1. Two-overlap translation stitching

The simplest case: two 3D volumes acquired on a translation stage with a
small overlap. This is what the `01_quickstart` notebook walks through.

```python
import numpy as np
import cupy as cp
from stitcher import Stitcher

# Generate two fake volumes with a known translation
shape = (16, 256, 256)
overlap = 50
fake_h5("left.h5",  shape, value_offset=100)
fake_h5("right.h5", shape, value_offset=200)

# Motor coordinates in mm — right is shifted +0.5 mm in x
motor_positions = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
])

st = Stitcher(
    file_path_list=["left.h5", "right.h5"],
    physical_coordinates=motor_positions,
    mm_per_voxel=0.01,                 # 10 µm / voxel
    x_y_z_correspondance=(1, 2, 3),    # (x_motor, y_motor, z_motor)
    saving_path="out_run",
)

st.get_layers_in_z(tolerance_mm=2)
st.get_padding()
st.get_intersections()
st.check_padding(layer_index=0)        # visual sanity check

# 1) Register
st.compute_shift_in_layers(
    start_slice=st.img_depth // 2 - 1,
    end_slice=st.img_depth // 2 + 1,
    mask=True, mask_radius=100,
    downscale=2, downscale_stages=2,
    apply_affine_warp=True,
    keep_rigid_only=True,
)

# 2) Build the displacement pyramid
st.get_displacement_pyramid(starting_coord=(0, 0, 0))

# 3) Accumulate, accounting for bad correlations
st.accumulate_displacement(exclude_NCC=50, weighted_avg=False, affine_operator=True)

# 4) Compose and stitch
st.compose_final_displacements()
st.push_stitch_parameters()
st.stitch_layers(chunk_size_series=8, chunk_size_parallel=2, n_cores=4)
```

The result lands in `out_run/Stitched_layers/Layer_0.h5`.

## 2. Translation + rotation stage

When the sub-volumes were acquired on a rotation stage, pass the rotation
centre (or simply rely on the motor positions) and Stitcher will
automatically detect the rotation. See
`notebooks/02_stitching_with_rotation.ipynb` for the full workflow.

```python
st = Stitcher(
    file_path_list=path_list,
    physical_coordinates=motor_positions,   # (x, y, theta) in this case
    mm_per_voxel=0.0065,
    x_y_z_correspondance=(1, 2, 3),
)
# ... the rest of the pipeline is identical.
```

## 3. Projection stitching (xy only)

For 2D-style stitching of projection images, set
`st.projection_xy_stitching = True` so the registration ignores the `z`
component.

```python
st.projection_xy_stitching = True
```

## 4. Tweaking the correlation

Useful knobs:

| Parameter            | Effect                                                  |
|----------------------|---------------------------------------------------------|
| `downscale`          | Scale factor per stage; < 1 means down-sample, > 1 up.  |
| `downscale_stages`   | Number of down/up-sample stages.                        |
| `mask`               | Apply a circular mask before correlating.               |
| `mask_radius`        | Radius (voxels) of the circular mask.                   |
| `apply_affine_warp`  | Refine with an affine or with translation only.         |
| `keep_rigid_only`    | Project the affine back to a rigid transform.           |
| `apply_mean_filter_zyx` | Local mean filter to help fine-feature correlation.  |
| `apply_detrend_filter_yx` | Detrend with a Gaussian kernel before correlating. |

## 5. Blending

After registration, the stitching is performed in
:func:`Stitcher.stitch_volumes_blend_equalize` which exposes:

* `alpha` — exponent of the distance map (higher = sharper transitions).
* `square_dist` — use a Chebyshev (square) distance function.
* `prop_x_y` — propagate the blend along `x`, `y` or both.
* `use_equalize` — apply linear intensity matching on the overlap.
* `exclude_NCC` — drop any volume whose final NCC is below this threshold.

Push the parameters to the full-stitcher with
:func:`Stitcher.push_stitch_parameters` and then call
:func:`Stitcher.stitch_layers` to write the result.

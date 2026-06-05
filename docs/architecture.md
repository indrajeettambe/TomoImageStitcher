# Architecture

This document describes the data structures and processing stages of Stitcher
v0.2. It is the reference you should read when contributing or when the
default pipeline does not quite fit your data.

## High-level pipeline

```
list of .h5 files + motor positions
        │
        ▼
┌────────────────────────────────────────┐
│  1. Organize sub-volumes                │   Stitcher.__init__
│     ─ physical → image coordinates      │   get_image_space_coordinates
│     ─ classify into z-layers            │   get_layers_in_z
│     ─ compute the global padding        │   get_padding
│     ─ find overlapping regions          │   get_intersections
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  2. Correlate every pair                │   compute_shift_in_layers
│     ─ ZNCC pixel search (GPU)           │     → RegistrationKIT.correlate_NCC
│     ─ IC-GN Lucas–Kanade refinement     │     → RegistrationKIT.lucas_kanade_3D_inv_mask
│     ─ mask-aware error metric           │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  3. Build displacement pyramid          │   get_displacement_pyramid
│     ─ BFS from a seed volume            │
│     ─ discard bad correlations (NCC)    │   accumulate_displacement
│     ─ weighted average (or best NCC)    │
│     ─ optional affine operator chain    │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  4. Stitch and blend                    │   compose_final_displacements
│     ─ chunk-by-chunk transformation     │   stitch_volumes_blend_equalize
│     ─ distance-map blending             │   stitch_layers
│     ─ optional intensity equalization   │
└────────────────────────────────────────┘
```

## Class layout

* **`Stitcher`** — drives the pipeline. One instance per stitching run.
  Holds *all* intermediate data in attributes (see the next section).
* **`RegistrationKIT`** — static-method class with the two registration
  engines. Stateless by design so it can be reused across calls.
* **`affine_transform_large_data`** — wraps a chunked GPU affine warp. Used
  by the final stitching step.
* **`Utilities`** — nested static class with H5 I/O, masks, distance
  functions and SimpleITK-based translation helpers.

## Important class attributes on `Stitcher`

| Attribute                              | Type                                    | Description |
|----------------------------------------|-----------------------------------------|-------------|
| `file_paths`                           | `list[str]`                             | Input `.h5` files. |
| `physical_coordinates_mm`             | `(N, 3) ndarray`                        | Motor positions in mm. |
| `mm_per_voxel`                         | `float`                                 | Spatial calibration. |
| `x_y_z_correspondance`                 | `(3,) tuple[int]`                       | Motor → image axis map. |
| `img_depth`, `img_height`, `img_width` | `int`                                   | Dimensions of the first image. |
| `layers_coordinates`                   | `list[(n_i, 3) ndarray]`                | Per-layer pixel coordinates. |
| `layers_paths`                         | `list[list[str]]`                       | Per-layer file paths. |
| `layers_intersecting_images_bb_x_y`    | `list[... , 2][(x0,y0),(x1,y1)]`        | Bounding boxes of overlaps. |
| `layers_reg_disp_data`                 | `list[list[(dx,dy,dz)]]`                | Per-pair rigid displacements. |
| `layers_reg_oper_data`                 | `list[list[4x4 ndarray]]`               | Per-pair affine operators. |
| `layers_reg_ncc_data`                  | `list[list[int]]`                       | Per-pair final NCC. |
| `cumulative_displacements_pyramid_sub_layer` | per-layer, per-step, per-subvolume | accumulated displacements |
| `cumulative_global_operators_pyramid_sub_layer` | per-layer, per-step, per-subvolume | accumulated affine operators |
| `layer_final_displacements`            | `list[list[(dx,dy,dz)]]`                | Final per-volume shift. |
| `layer_final_operators`                | `list[list[4x4 ndarray]]`               | Final per-volume operator. |
| `layer_final_NCC`                      | `list[list[float]]`                     | Final per-volume NCC. |

## Coordinate conventions

The pipeline uses two different coordinate systems:

1. **Motor coordinates** `(x_motor, y_motor, z_motor)` in millimetres — what
   the experimentalist sees.
2. **Image coordinates** `(x_img, y_img, z_img)` in voxels — what the
   registration engine works with.

The mapping is:

```
x_img = x_motor / mm_per_voxel * sign(a)
y_img = y_motor / mm_per_voxel * sign(b)
z_img = z_motor / mm_per_voxel * sign(c)
```

where `(a, b, c)` is the `x_y_z_correspondance` tuple passed to the
`Stitcher` constructor. Using negative values for `a`, `b` or `c` flips the
corresponding axis in image space, which is useful when the detector
orientation does not match the motor one.

## Mask convention

Throughout the code, **0 is reserved for "background" / masked pixels**.
This lets us:

* multiply an image by its mask (`image *= (image != 0)`)
* build mask-aware correlators that ignore background contributions
* use a same-interpolator resample of `(image != 0)` to avoid bleeding
  background into foreground when warping

If your data legitimately contains zeros, set
`stitcher.add_value_for_mask = 1` so the data is shifted by one when loaded
and shifted back when written.

## Registration engines

### ZNCC (`RegistrationKIT.correlate_NCC`)

* operates in Fourier space via `cupyx.scipy.ndimage.correlate`
* supports multi-stage downscaling (`downscale < 1`) or binning
  (`downscale > 1`)
* mask-aware — the correlation is normalised only over the masked voxels

### Lucas–Kanade (`RegistrationKIT.lucas_kanade_3D_inv_mask`)

* IC-GN variant — Hessian is computed once, template is warped at every step
* supports rigid, affine, and 2D-only `xy_reg` warps
* mask is re-evaluated at every step and can be eroded
* can save intermediate warped templates (`save=True`)

The "rigid" warp is extracted from the affine via polar decomposition
(inside `_extract_rigid_transform`).

## Saving conventions

`stitcher.compute_shift_in_layers(..., save_reg=True, save_path=...)` writes
two `.h5` files per intersection:

* `registration_<layer>_<image>_<k>.h5` — operator, NCC, count, etc.
* `registration_NCC_<layer>_<image>_<k>.h5` — registered slices used to
  measure the NCC.

`stitcher.stitch_layers(...)` writes one file per layer under
`<saving_path>/Stitched_layers/Layer_<i>.h5`. The group `/stitched_data`
contains:

* `stitched_image` — the actual volume.
* `layer_coordinates`, `layer_paths`, `layer_final_displacements`, …
  — pipeline metadata.

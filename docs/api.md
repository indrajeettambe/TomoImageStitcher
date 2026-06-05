# API reference

The full reference is auto-generated from the docstrings. The most useful
entry points are listed below.

## `stitcher.Stitcher`

```python
class stitcher.Stitcher(
    file_path_list,
    physical_coordinates,
    mm_per_voxel,
    x_y_z_correspondance=(1, 2, 3),
    saving_path=None,
)
```

Drives the entire stitching pipeline. The order of operations is:

1. `get_layers_in_z(tolerance_mm)`
2. `get_padding()`
3. `get_intersections(check=False, radius=None)`
4. `compute_shift_in_layers(...)`
5. `get_displacement_pyramid(check=False, starting_coord=None)`
6. `accumulate_displacement(exclude_NCC=80, weighted_avg=True, affine_operator=False)`
7. `compose_final_displacements(verbose=False)`
8. `push_stitch_parameters()`
9. `stitch_layers(chunk_size_series=200, chunk_size_parallel=10, n_cores=10, ...)`

### Key methods

* `extract_intersection(start_slice, end_slice, layer, image, intersection, mask=False, mask_radius=None)`
* `correlate_intersection(start_slice, end_slice, layer, image, intersection, ...)`
* `get_transformed_slices(layer, image, start_slice, end_slice, mask=False, mask_radius=None)`
* `get_transformed_slices_affine(layer, image, start_slice, end_slice, mask=False, mask_radius=None, chunk_size=None)`
* `stitch_volumes_blend_equalize(stitch_layer=None, start_slice=None, end_slice=None, mask=False, mask_radius=None, alpha=1, use_equalize=False, use_existing_equalize=False, normalize_dist_radially=True, square_dist=False, crop_x=(0,0), crop_y=(0,0), exclude_indexes=[], exclude_NCC=True, show_progress_bar=True)`

### Key class attributes

| Attribute                          | Default                                | Description |
|------------------------------------|----------------------------------------|-------------|
| `add_value_for_mask`               | `0`                                    | Shift added to all values when reading (reserve 0 for mask). |
| `erosion_mask_LC_xyz`              | `(11, 11, 11)`                         | Erosion applied to the registration mask. |
| `GPU_chunk_size`                   | `1`                                    | Z-chunk size for the affine warp. |
| `sitk_interpolator`                | `sitk.sitkLinear`                      | Interpolation used by `get_transformed_slices`. |
| `affine_interpolator_order`        | `1`                                    | Spline order for the affine transform. |
| `force_rigid_warp`                 | `False`                                | If True, only the rigid part of the affine is used. |
| `projection_xy_stitching`          | `False`                                | If True, ignore z in the registration. |
| `affine_warp`                      | `None`                                 | Whether to use affine (vs shift) for stitching. |
| `prop_x_y`                         | `(0, 0)`                               | Direction for the blending distance field. |

## `stitcher.RegistrationKIT`

Static-method class. Useful methods:

* `RegistrationKIT.correlate_NCC(search, template, downscale=1, downscale_stages=1, use_spline=False, use_mask_template=False, use_mask_search=False, use_minimun_count=False, mask_threshold=(-1E-10, 1E-10), minimum_count=1E6, apply_gaussian_img_x_y_z=(0,0,0), apply_gaussian_NCC_x_y_z=(0,0,0))`
* `RegistrationKIT.lucas_kanade_3D_inv_mask(template_img, moving_img, derivatives="gaussian", sigma_z_y_x=(1, 1, 1), margins_xyz=(20, 20, 20), max_iter=20, convergence_criteria=0.001, mask=False, erodeMask=False, erosionElement=np.ones((1, 1, 1)), initial_guess=None, interp_order=1, regulate=False, slice_extract=None, save=False, affine_warp=False, affine_guess=True, rigid_warp=False, xy_reg=False)`

## `stitcher.affine_transform_large_data`

```python
class stitcher.affine_transform_large_data(img_h5_pointer, chunk_size=1, add_value_for_mask=False)
```

* `get_chunks_position(start_slice, end_slice)`
* `set_affine_transform_operator(operator)` — `4x4 ndarray`
* `set_affine_transform_center((x, y, z))`
* `scale_affine_transform(s)` — multiply `s` into the homogeneous coord
* `set_spline_order(order)` — 0, 1, 3 or 5
* `transform_chunk(chunk_index, use_mask=False)`
* `chunk_by_chunk_transform(path=None, file_name=None, overwrite=False)`

## `stitcher.Utilities`

Static helpers (no instance needed):

* `Utilities.H5MaxIV.reader(file_path)`, `get_shape(...)`, `get_bit_depth(...)`, `get_slices(...)`
* `Utilities.GeneralTiff.reader(file_path)`, `get_shape(...)`, `save(...)`, `get_slices(...)`
* `Utilities.circular_mask(img, radius=None, center=(0, 0))`
* `Utilities.dist_function(img, center=(0, 0))`
* `Utilities.dist_function_sq(img, center=(0, 0), prop_x_y=(0, 0))`
* `Utilities.normalize_distance_map_radially(distance_map, center=None)`
* `Utilities.normalize_with_masked_gaussian_filter_cupy_2D(image_np, sigma_np_xy, divide_by_norm=False, use_mask=False, notification_frequency=50, value_return_mask=cp.array(0))`
* `Utilities.translate_itk(img, d_x_y_z=(0, 0, 0), sitk_interpolator=sitk.sitkLinear)`
* `Utilities.translate_itk_masked(img, d_x_y_z=(0, 0, 0), sitk_interpolator=sitk.sitkLinear, use_mask=True)`
* `Utilities.mean_filter_masked(img_np, element_shape=(1, 1, 1))`
* `Utilities.convert(img, maxValue=None, minImg=None, maxImg=None, data_type=np.uint16)`
* `Utilities.plot_slice(image, title="Slice")`

## `stitcher.danmax` (optional)

Beamline-specific helpers, only useful on the DanMAX Jupyter environment:

* `getCurrentProposal(...)`, `getCurrentProposalType(...)`
* `findAllScans(...)`, `findScan(...)`, `getLatestScan(...)`
* `getMetaData(...)`, `getMetaDic(...)`, `getAzintData(...)`
* `appendScans(...)`, `getMotorSteps(...)`, `getPixelCoords(...)`
* `averageLargeScan(...)`, `getAverageImage(...)`, `getHottestPixel(...)`
* `InteractiveMask(images, reduction_mode='std', mask_alpha=0.3)`
* `darkMode(use=True, style_dic=...)`, `lightMode(use=True, style_dic=...)`
* `interactiveImageHist(im, ignore_zero=False)`

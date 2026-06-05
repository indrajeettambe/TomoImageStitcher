"""End-to-end example: stitching two overlapping 3D volumes.

This example generates synthetic data so it can be run on any machine with a
CUDA-capable GPU. It mirrors the workflow described in
``notebooks/01_quickstart.ipynb``.

Usage:
    python examples/example_2d_projection.py
"""
import os
import tempfile

import h5py
import numpy as np
import cupy as cp

from tomo_image_stitcher import Stitcher


def write_synthetic_volume(
    path: str,
    shape: tuple[int, int, int] = (16, 128, 128),
    value: int = 100,
) -> None:
    """Write a constant-filled 3D volume to an HDF5 file."""
    with h5py.File(path, "w") as fh:
        fh.create_dataset("image", data=np.full(shape, value, dtype=np.uint16))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        left_path = os.path.join(tmp, "left.h5")
        right_path = os.path.join(tmp, "right.h5")
        write_synthetic_volume(left_path)
        write_synthetic_volume(right_path)

        st = Stitcher(
            file_path_list=[left_path, right_path],
            physical_coordinates=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            mm_per_voxel=0.01,
            x_y_z_correspondance=(1, 2, 3),
            saving_path=os.path.join(tmp, "out"),
        )

        st.get_layers_in_z(tolerance_mm=2)
        st.get_padding()
        st.get_intersections()

        # --- Registration ---
        st.compute_shift_in_layers(
            start_slice=st.img_depth // 2 - 2,
            end_slice=st.img_depth // 2 + 2,
            mask=True,
            mask_radius=50,
            downscale=2,
            downscale_stages=2,
            apply_affine_warp=True,
            keep_rigid_only=True,
        )

        # --- Displacement accumulation ---
        st.get_displacement_pyramid(starting_coord=(0, 0, 0))
        st.accumulate_displacement(
            exclude_NCC=50, weighted_avg=False, affine_operator=True
        )
        st.compose_final_displacements()

        # --- Stitching ---
        st.push_stitch_parameters()
        st.stitch_layers(
            chunk_size_series=8, chunk_size_parallel=2, n_cores=4
        )

        out_file = os.path.join(tmp, "out", "Stitched_layers", "Layer_0.h5")
        with h5py.File(out_file, "r") as fh:
            ds = fh["/stitched_data/stitched_image"]
            print(f"Stitched volume shape: {ds.shape}, dtype: {ds.dtype}")

    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

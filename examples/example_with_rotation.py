"""Example: stitching volumes acquired on a rotation stage.

The motor positions of each sub-volume are passed as ``(x, y, theta)``; the
``x_y_z_correspondance`` argument tells Stitcher which image axis corresponds
to which motor axis. The rest of the pipeline is identical to the
translation-only case.

Usage:
    python examples/example_with_rotation.py
"""
import os
import tempfile

import h5py
import numpy as np
import cupy as cp

from stitcher import Stitcher


def write_synthetic_volume(
    path: str,
    shape: tuple[int, int, int] = (8, 64, 64),
    value: int = 100,
) -> None:
    with h5py.File(path, "w") as fh:
        fh.create_dataset("image", data=np.full(shape, value, dtype=np.uint16))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # 11 fake volumes on a 5x5 grid rotated by a varying angle.
        # The exact geometry is irrelevant for the smoke test.
        paths = []
        positions = []
        for k in range(11):
            p = os.path.join(tmp, f"vol_{k:02d}.h5")
            write_synthetic_volume(p)
            paths.append(p)
            positions.append([0.1 * k, 0.0, 0.0])
        positions = np.array(positions)

        st = Stitcher(
            file_path_list=paths,
            physical_coordinates=positions,
            mm_per_voxel=0.01,
            x_y_z_correspondance=(1, 2, 3),
            saving_path=os.path.join(tmp, "out"),
        )

        st.get_layers_in_z(tolerance_mm=2)
        st.get_padding()
        st.get_intersections()

        st.compute_shift_in_layers(
            start_slice=st.img_depth // 2 - 1,
            end_slice=st.img_depth // 2 + 1,
            mask=True,
            mask_radius=20,
            downscale=2,
            downscale_stages=2,
            apply_affine_warp=True,
            keep_rigid_only=True,
        )

        st.get_displacement_pyramid(starting_coord=(0, 0, 0))
        st.accumulate_displacement(
            exclude_NCC=0, weighted_avg=False, affine_operator=True
        )
        st.compose_final_displacements()
        st.push_stitch_parameters()
        st.stitch_layers(
            chunk_size_series=4, chunk_size_parallel=2, n_cores=2
        )

        out_file = os.path.join(tmp, "out", "Stitched_layers", "Layer_0.h5")
        with h5py.File(out_file, "r") as fh:
            ds = fh["/stitched_data/stitched_image"]
            print(f"Stitched volume shape: {ds.shape}, dtype: {ds.dtype}")

    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

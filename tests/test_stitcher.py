"""End-to-end smoke test of the TomoImageStitcher pipeline on synthetic data.

This is intentionally minimal: it does not check the quality of the
registration, only that the full pipeline runs without errors. The test is
skipped on machines without a CUDA-capable GPU.
"""
import pytest

from tomo_image_stitcher import Stitcher

import h5py
import numpy as np

cupy = pytest.importorskip("cupy")  # noqa: F811  (also re-selected by -m gpu)


def _write_fake_h5(path, shape=(8, 32, 32), value=100):
    with h5py.File(path, "w") as fh:
        fh.create_dataset("image", data=np.full(shape, value, dtype=np.uint16))


@pytest.mark.gpu
def test_full_pipeline_runs(tmp_path):
    left = tmp_path / "left.h5"
    right = tmp_path / "right.h5"
    _write_fake_h5(str(left))
    _write_fake_h5(str(right))

    st = Stitcher(
        file_path_list=[str(left), str(right)],
        physical_coordinates=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        mm_per_voxel=0.01,
        x_y_z_correspondance=(1, 2, 3),
        saving_path=str(tmp_path / "out"),
    )
    st.get_layers_in_z(tolerance_mm=2)
    st.get_padding()
    st.get_intersections()

    # Registration
    st.compute_shift_in_layers(
        start_slice=st.img_depth // 2 - 1,
        end_slice=st.img_depth // 2 + 1,
        mask=True, mask_radius=10,
        downscale=2, downscale_stages=2,
        apply_affine_warp=False,
    )
    # Pyramid + accumulation
    st.get_displacement_pyramid(starting_coord=(0, 0, 0))
    st.accumulate_displacement(exclude_NCC=0, weighted_avg=False, affine_operator=False)
    st.compose_final_displacements()
    st.push_stitch_parameters()
    st.stitch_layers(chunk_size_series=4, chunk_size_parallel=2, n_cores=2)

    # The output file should exist
    out = tmp_path / "out" / "Stitched_layers" / "Layer_0.h5"
    assert out.exists()
    with h5py.File(out, "r") as fh:
        ds = fh["/stitched_data/stitched_image"]
        assert ds.ndim == 3

"""Smoke tests for the affine transform on small synthetic volumes.

These tests are skipped automatically if CuPy cannot find a GPU.
"""
import pytest

from tomo_image_stitcher import affine_transform_large_data

import tempfile

import h5py
import numpy as np

cupy = pytest.importorskip("cupy")


@pytest.mark.gpu
def test_identity_affine_preserves_values():
    arr = np.random.randint(0, 65535, size=(4, 16, 16), dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as fh:
            fh.create_dataset("image", data=arr)

        with h5py.File(tmp.name, "r") as fh:
            tr = affine_transform_large_data(fh["image"], chunk_size=1)
            tr.set_affine_transform_operator(np.eye(4))
            tr.set_spline_order(1)
            tr.set_affine_transform_center(
                (arr.shape[2] / 2, arr.shape[1] / 2, arr.shape[0] / 2)
            )
            tr.get_chunks_position(0, 2)
            out = tr.transform_chunk(0)

        # Identity transform with linear interpolation: result close to input.
        np.testing.assert_allclose(out[0].astype(np.float32), arr[0:1].astype(np.float32), atol=2)

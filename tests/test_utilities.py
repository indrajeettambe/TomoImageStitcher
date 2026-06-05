"""Smoke tests that do not require a GPU.

These tests are designed to run on any machine with NumPy + SciPy. They verify
the I/O helpers and the pure-Python parts of the package.
"""
import os
import tempfile

import h5py
import numpy as np
import pytest

from stitcher import Utilities


def _fake_h5(path, shape=(8, 32, 32), value=100):
    with h5py.File(path, "w") as fh:
        fh.create_dataset("image", data=np.full(shape, value, dtype=np.uint16))
    return path


def test_h5_reader_layout():
    with tempfile.TemporaryDirectory() as d:
        path = _fake_h5(os.path.join(d, "x.h5"))
        ds = Utilities.H5MaxIV.reader(path)
        assert ds.shape == (8, 32, 32)
        assert ds.dtype == np.uint16


def test_h5_get_slices_returns_correct_window():
    with tempfile.TemporaryDirectory() as d:
        path = _fake_h5(os.path.join(d, "x.h5"))
        s = Utilities.H5MaxIV.get_slices(path, start_slice=2, end_slice=5)
        assert s.shape == (3, 32, 32)
        # The synthetic data is uniform; verify the shift-mask logic.
        s_mask = Utilities.H5MaxIV.get_slices(
            path, start_slice=2, end_slice=5, add_value_for_mask=7
        )
        assert np.all(s_mask == 107)


def test_circular_mask_is_correct_shape():
    img = np.zeros((4, 16, 16), dtype=np.float32)
    mask = Utilities.circular_mask(img, radius=6)
    assert mask.shape == img.shape
    # The centre of the central slice should be inside the mask.
    assert mask[2, 8, 8] == 1
    # The far corner should be outside.
    assert mask[2, 0, 0] == 0


def test_dist_function_monotonic_from_centre():
    img = np.zeros((4, 16, 16), dtype=np.float32)
    d = Utilities.dist_function(img, center=(0, 0))
    centre = d[2, 8, 8]
    corner = d[2, 0, 0]
    assert centre < corner


def test_dist_function_sq_propagation_modes():
    img = np.zeros((4, 16, 16), dtype=np.float32)
    d_xy = Utilities.dist_function_sq(img, center=(0, 0), prop_x_y=(0, 0))
    d_x = Utilities.dist_function_sq(img, center=(0, 0), prop_x_y=(1, 0))
    d_y = Utilities.dist_function_sq(img, center=(0, 0), prop_x_y=(0, 1))
    # Chebyshev distance should be >= each of the two axis distances.
    assert np.all(d_xy >= d_x)
    assert np.all(d_xy >= d_y)
    # The two axis distances should be different (they measure different axes).
    assert not np.allclose(d_x, d_y)


def test_convert_scales_to_target_dtype():
    img = np.linspace(0, 1, 50, dtype=np.float32).reshape(5, 10)
    out = Utilities.convert(img, maxValue=65535, minImg=0, maxImg=1, data_type=np.uint16)
    assert out.dtype == np.uint16
    assert out.max() == 65535
    assert out.min() == 0


def test_translate_itk_shifts_image():
    """`translate_itk` should shift a 2D/3D image by the given (dx, dy, dz)."""
    arr = np.zeros((4, 32, 32), dtype=np.float32)
    arr[:, 10:20, 10:20] = 1.0
    out = Utilities.translate_itk(arr, d_x_y_z=(5, 5, 0))
    # The shifted blob's centre should be at (15, 15) instead of (14, 14).
    assert out[:, 15, 15] > 0
    assert out[:, 10, 10] == 0


def test_translate_itk_masked_preserves_zero_background():
    arr = np.zeros((4, 32, 32), dtype=np.float32)
    arr[:, 10:20, 10:20] = 1.0
    out = Utilities.translate_itk_masked(arr, d_x_y_z=(5, 5, 0))
    # Background should remain background — no bleeding of 0 into the blob.
    blob = out[:, 10:20, 10:20]
    assert np.all(blob[blob > 0] > 0.5)

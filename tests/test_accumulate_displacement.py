"""Regression coverage for confidence-weighted displacement accumulation."""
import numpy as np
import pytest

from tomo_image_stitcher import Stitcher


def _two_path_stitcher(displacements, ncc):
    """Two first-ring nodes both lead to the same second-ring tile."""
    st = Stitcher.__new__(Stitcher)
    st.intersections_pyramid_sub_layer = [[np.array([0]), np.array([1, 2]), np.array([3, 3])]]
    st.central_neighbor_pyramid_sub_layer = [[np.array([0, 0]), np.array([1, 2])]]
    st.displacements_pyramid_sub_layer = [[np.asarray(displacements, dtype=float), np.zeros((2, 3))]]
    st.displacements_ncc_pyramid_sub_layer = [[np.array([100., 100.]), np.asarray(ncc, dtype=float)]]
    identity = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    st.operator_pyramid_sub_layer = [[identity.copy(), identity.copy()]]
    st.reg_cent_pyramid_sub_layer = [[np.zeros((2, 3)), np.zeros((2, 3))]]
    st.layers_intersecting_images_indices = [[np.array([]), np.array([]), np.array([]), np.array([1, 2])]]
    st.layers_reg_ncc_data = [[np.array([]), np.array([]), np.array([]), np.asarray(ncc, dtype=float)]]
    return st


@pytest.mark.parametrize(
    ("paths", "ncc", "threshold", "expected"),
    [
        ([[10, 0, 0], [10, 0, 0]], [80, 100], 50, [10, 0, 0]),
        ([[10, 0, 0], [20, 0, 0]], [80, 100], 50, [2800 / 180, 0, 0]),
        ([[10, 0, 0], [20, 0, 0]], [40, 100], 50, [20, 0, 0]),
    ],
)
def test_accumulation_uses_sum_of_accepted_ncc_weights(paths, ncc, threshold, expected):
    st = _two_path_stitcher(paths, ncc)
    st.accumulate_displacement(exclude_NCC=threshold, weighted_avg=True)
    np.testing.assert_allclose(st.cumulative_displacements_pyramid_sub_layer[0][1], [expected, expected])


def test_unweighted_accumulation_uses_best_accepted_path():
    st = _two_path_stitcher([[10, 0, 0], [20, 0, 0]], [80, 100])
    st.accumulate_displacement(exclude_NCC=50, weighted_avg=False)
    np.testing.assert_allclose(st.cumulative_displacements_pyramid_sub_layer[0][1], [[20, 0, 0], [20, 0, 0]])


def test_no_accepted_weights_produces_zero_displacement():
    st = _two_path_stitcher([[10, 0, 0], [20, 0, 0]], [10, 20])
    st.accumulate_displacement(exclude_NCC=50, weighted_avg=True)
    np.testing.assert_allclose(st.cumulative_displacements_pyramid_sub_layer[0][1], np.zeros((2, 3)))


def test_cancelling_weights_produce_zero_displacement():
    st = _two_path_stitcher([[10, 0, 0], [20, 0, 0]], [-10, 10])
    st.accumulate_displacement(exclude_NCC=-20, weighted_avg=True)
    np.testing.assert_allclose(st.cumulative_displacements_pyramid_sub_layer[0][1], np.zeros((2, 3)))

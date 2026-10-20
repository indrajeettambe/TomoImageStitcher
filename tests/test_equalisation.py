"""Tests for the intensity-equalisation transform.

These exist because the transform was once written out twice -- once where the
factors were fitted, once where they were reused -- and the two copies drifted
apart. Both call sites now go through ``apply_equalisation``, so these tests pin
the transform itself rather than either of its callers.
"""
import numpy as np
import pytest

from tomo_image_stitcher.stitcher import (
    MAX_EQUALISATION_SLOPE,
    MIN_EQUALISATION_SLOPE,
    apply_equalisation,
    valid_equalisation_factors,
)


def test_identity_leaves_data_untouched():
    img = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]])
    assert np.array_equal(apply_equalisation(img, 1.0, 0.0), img)


def test_applies_the_form_that_was_fitted():
    # linregress(tile, mosaic) gives mosaic ~ slope * tile + intercept, so that
    # is the map the stored factors have to describe.
    img = np.array([[0.0, 4.0], [8.0, 12.0]])
    got = apply_equalisation(img, 0.82, 0.35)
    assert np.allclose(got, np.array([[0.0, 3.63], [6.91, 10.19]]))


def test_zeros_stay_zero():
    # The circular mask writes zeros inside the blend region; scaling them would
    # turn background into a signal.
    img = np.array([[0.0, 5.0]])
    out = apply_equalisation(img, 2.0, 3.0)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 13.0


def test_the_old_and_new_forms_are_not_the_same_map():
    # The bug: the reuse path applied (img + intercept) * slope instead.
    img = np.array([10.0])
    correct = apply_equalisation(img, 0.82, 0.35)[0]
    old = (img + 0.35) * 0.82
    # They differ by intercept * (slope - 1), which is -0.063 at these factors.
    assert old == pytest.approx(correct - 0.063)


@pytest.mark.parametrize(
    "slope,intercept",
    [(1.0, 0.0), (0.5, 3.0), (MAX_EQUALISATION_SLOPE, -1.0), (0.051, 0.2)],
)
def test_usable_factors_pass_through(slope, intercept):
    assert valid_equalisation_factors(slope, intercept) == (slope, intercept)


@pytest.mark.parametrize(
    "slope,intercept",
    [
        (MIN_EQUALISATION_SLOPE, 0.0),  # the lower bound is itself rejected
        (MAX_EQUALISATION_SLOPE + 1e-9, 0.0),
        (0.0, 0.0),
        (np.nan, 0.0),
        (1.0, np.inf),
        (None, 0.0),
    ],
)
def test_unusable_factors_fall_back_to_identity(slope, intercept):
    assert valid_equalisation_factors(slope, intercept) == (1.0, 0.0)


def test_apply_ignores_unusable_factors():
    img = np.array([[0.0, 5.0]])
    assert np.array_equal(apply_equalisation(img, 0.0, 7.0), img)

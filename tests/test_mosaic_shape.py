"""The mosaic size must follow the layer being written, not layer 0.

``stitch_layers`` sized every layer's output with that layer's padding but with
the tile extents of layer 0, image 0. With homogeneous tiles the two coincide,
which is why it went unnoticed; with tiles that differ in y or x between layers,
every layer after the first was silently cropped or padded to the first one's
geometry.
"""
import numpy as np

from tomo_image_stitcher import Stitcher


def _stitcher(padding_neg_y, padding_pos_y, padding_neg_x, padding_pos_x,
              left, right, crop_y=(0, 0), crop_x=(0, 0)):
    """A Stitcher without running __init__: only what the size depends on."""
    st = object.__new__(Stitcher)
    st.img_depth = 4
    st.params_final = {"crop_y": crop_y, "crop_x": crop_x}
    st.padding_neg_y = [np.array([v]) for v in padding_neg_y]
    st.padding_pos_y = [np.array([v]) for v in padding_pos_y]
    st.padding_neg_x = [np.array([v]) for v in padding_neg_x]
    st.padding_pos_x = [np.array([v]) for v in padding_pos_x]
    st.layers_outer_most_left_xyz = [np.array([v]) for v in left]
    st.layers_outer_most_right_xyz = [np.array([v]) for v in right]
    return st


def test_each_layer_is_sized_from_its_own_tiles():
    st = _stitcher(
        padding_neg_y=[10, 20], padding_pos_y=[30, 40],
        padding_neg_x=[1, 2], padding_pos_x=[3, 4],
        left=[[5, 6, 0], [7, 8, 0]],
        right=[[9, 10, 0], [11, 12, 0]],
    )
    # height = neg_y + pos_y + right_y + left_y - crop_y
    # width  = neg_x + pos_x + right_x + left_x - crop_x
    assert st.mosaic_plane_shape(0) == (10 + 30 + 10 + 6, 1 + 3 + 9 + 5)
    assert st.mosaic_plane_shape(1) == (20 + 40 + 12 + 8, 2 + 4 + 11 + 7)


def test_layer_one_is_not_sized_from_layer_zero():
    # The shape this test rejects is the old behaviour: layer 1's padding with
    # layer 0's right_y/left_y and right_x/left_x.
    st = _stitcher(
        padding_neg_y=[10, 20], padding_pos_y=[30, 40],
        padding_neg_x=[1, 2], padding_pos_x=[3, 4],
        left=[[5, 6, 0], [7, 8, 0]],
        right=[[9, 10, 0], [11, 12, 0]],
    )
    assert st.mosaic_plane_shape(1) != (20 + 40 + 10 + 6, 2 + 4 + 9 + 5)


def test_homogeneous_tiles_give_every_layer_the_same_size():
    st = _stitcher(
        padding_neg_y=[10, 10, 10], padding_pos_y=[30, 30, 30],
        padding_neg_x=[1, 1, 1], padding_pos_x=[3, 3, 3],
        left=[[5, 6, 0]] * 3, right=[[9, 10, 0]] * 3,
    )
    assert len({st.mosaic_plane_shape(i) for i in range(3)}) == 1


def test_crops_are_subtracted():
    st = _stitcher(
        padding_neg_y=[10], padding_pos_y=[30],
        padding_neg_x=[1], padding_pos_x=[3],
        left=[[5, 6, 0]], right=[[9, 10, 0]],
        crop_y=(2, 2), crop_x=(3, 1),
    )
    assert st.mosaic_plane_shape(0) == (10 + 30 + 10 + 6 - 4, 1 + 3 + 9 + 5 - 4)

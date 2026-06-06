"""
TomoImageStitcher — main Stitcher class.

This module exposes the high-level :class:`Stitcher` class that drives the
full stitching pipeline, plus the small :class:`Utilities` helper namespace
used throughout the package.
"""
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from .registration import RegistrationKIT
import tifffile as tif
from copy import deepcopy
import os
import SimpleITK as sitk
from scipy.stats import linregress
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from .transform import affine_transform_large_data as tr
from scipy.linalg import polar
from time import time as timing
import scipy.ndimage as sci

try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve
    from cupyx.scipy.ndimage import gaussian_filter

    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - allows CPU-only environments
    cp = None  # type: ignore[assignment]
    convolve = None  # type: ignore[assignment]
    gaussian_filter = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


class Stitcher:
    def __init__(self, file_path_list, physical_coordinates, mm_per_voxel, x_y_z_correspondance=(1, 2, 3), saving_path=None):
        """
        Initialize the Stitcher class with the required parameters.

        Parameters
        ----------

        file_path_list: list of str
            A list containing the paths to the images to be stitched.

        physical_coordinates: nx3 ndarray
            Motor coordinates in mm for each image corresponding to the file_path_list.

        mm_per_voxel: spatial resolution. e.g. 0.0065 for a 6.5 microns/voxel

        x_y_z_correspondance: list of int --> e.g. (a, b, c)
            Set the correspondance of the motor axes to image coordinate system (x,y,z):
                (a,b,c) -> x_image = physical_coordinates[:, :, a - 1] * mm_per_voxel * np.sign(a)
                (a,b,c) -> y_image = physical_coordinates[:, :, b - 1] * mm_per_voxel * np.sign(b)
                (a,b,c) -> z_image = physical_coordinates[:, :, c - 1] * mm_per_voxel * np.sign(c)

        saving_path: str
                A path to save the data generated during the procedure. If None, os.getcwd() will be used. Default None.

        Important
        -------
        This tool only accepts images that have the same depth (z slices). Pad the images beforehand if necessary.

        """
        # Path and coordinates for the volumetric images
        self.file_paths = file_path_list  # The list containing the paths of the images to be stitched
        self.physical_coordinates_mm = physical_coordinates  # The coordinates on the physical coordinate system: motor stage
        self.x_y_z_correspondance = x_y_z_correspondance  # The x, y and z image axes
        self.mm_per_voxel = mm_per_voxel  # mm/voxel
        # Saving data path
        self.saving_path = saving_path
        # verify the inputs
        self._check_input()
        # Image dimensions using the first path in the list
        self.img_depth, self.img_height, self.img_width = Utilities.H5MaxIV.get_shape(self.file_paths[0])
        # Check the dimensions
        # if self.img_height != self.img_width:
        #   sys.exit("The script works only with images having similar x and y dimensions! Padd the images and modify the physical coordinates accordingly beforehand.")
        # Image radius
        self.radius = np.floor(self.img_height / 2).astype(int)
        # Attribute to keep the coordinates of every image center in the pixel space
        self.global_coord_x_y_z = None
        # Attribute to keep the extents from the center of outermost left and right edges of the images
        self.outer_most_left_xyz = None
        self.outer_most_right_xyz = None
        # Attributes to keep the image shape, file paths and their global positions in the pixel space for every layer
        self.layers_img_shapes_xyz = None
        self.layers_coordinates = None  # will contain the global coordinates in the pixel space of every image center on every layer. Layers are in an increasing order in terms of height.
        self.layers_paths = None  # will contain the global coordinates in the pixel space of every image center on every layer. Layers are in an increasing order in terms of height.
        self.layers_outer_most_left_xyz = None  # Attribute to keep the extents from the center of outermost left and right edges of the images for every layer.
        self.layers_outer_most_right_xyz = None  # Attribute to keep the extents from the center of outermost left and right edges of the images for every layer
        # Attributes to keep the padding required for each volume in order to stitch the slices together
        self.padding_neg_x = None
        self.padding_neg_y = None
        self.padding_pos_x = None
        self.padding_pos_y = None
        # Attributes related to the intersections between images in every layer
        self.layers_intersecting_images_bb_x_y = None
        self.layers_intersecting_images_mask_c = None
        self.layers_intersecting_images_indices = None
        # Attributes related to the registration data
        self.layers_reg_disp_data = None
        self.layers_reg_cent_data = None  # The main list containing the organized registration data
        self.layers_reg_oper_data = None
        self.layers_reg_ncc_data = None
        self.layers_reg_diff_data = None
        self.affine_warp = None
        self.force_rigid_warp = False  # forces the rigid warp during the correlation procedure and as an extraction from the affine operator after
        # Parameters related to the ele
        self.start_slice_reg = None
        self.end_slice_reg = None
        self.crop_x_reg = None
        self.crop_y_reg = None
        self.reduce_slices_LC_reg = None
        # Attributes related to the displacement pyramid
        self.intersections_pyramid_sub_layer = None  # A global array to contain the intersections_pyramids of every layer
        self.operator_pyramid_sub_layer = None  # A global array to contain the affine operators of every layer
        self.reg_cent_pyramid_sub_layer = None  # A global array to contain the registration center of every layer
        self.displacements_pyramid_sub_layer = None  # A global array to contain the displacement_pyramids of every layer
        self.displacements_ncc_pyramid_sub_layer = None  # A global array to contain the normalized cross correlation coefficient for the warped displacements of every expansion layer of intersections
        self.gray_level_diff_pyramid_sub_layer = None  # A global array to contain the gray level difference statistics between the templates and the intersecting images after registration of every expansion layer of intersections
        self.central_neighbor_pyramid_sub_layer = None
        # Attributes related to the accumulated displacement and operator
        self.cumulative_displacements_pyramid_sub_layer = None
        self.cumulative_global_operators_pyramid_sub_layer = None
        self.cumulative_global_cent_pyramid_sub_layer = None  # A global array to gather the cumulative_cent_pyramid_sub of every layer
        # Composed final displacements and operators
        self.layer_final_displacements = None
        self.layer_final_global_coordinates = None
        self.layer_final_operators = None
        self.layer_final_centers_xyz = None
        self.layer_final_NCC = None  # A global array to gather the NCC of every layer
        self.exclude_NCC = None  # The NCC in percentage used to discriminate bad volumes from stitching when set to true in the stitching method.
        # Equalizing parameters
        self.layer_equalize_slope_intercept = None
        # General stitching parameter space
        self.params_final = None
        self.params_temp = None  # (start_slice, end_slice, mask, mask_radius, alpha, use_equalize, square_dist, crop_x, crop_y)
        # Other tools for selection out of a function
        self.prop_x_y = (0, 0)  # Sets the mode of propagation in a case of a square distance field
        self.erosion_mask_LC_xyz = (11, 11, 11)  # The erosion of the mask to avoid bad interpolation at the borders during the correlation.
        self.GPU_chunk_size = 1  # The chunk size by default when using the affine transform in GPU
        self.add_value_for_mask = 0  # This parameter makes sure that when the images are read a the self.add_value_for_mask values is added to skip the zeros.
        # Interpolator for the itk translation
        self.mask_interpolator = True  # This defines if the image interpolation will avoid interpolationg anything that overlaps with regions containing 0
        self.sitk_interpolator = sitk.sitkLinear  # Options for the interpolation in case of a translation only: sitk.Linear, sitk.sitkBSpline5, sitk.sitkBSpline or other options as in https://simpleitk.org/doxygen/latest/html/sitkInterpolator_8h.html
        self.affine_interpolator_order = 1  # Options for the interpolation in case of affine transform. 0, 1, 3, and 5 similar to scipy.ndimage.map_coordinates
        # Variable that allows only an xy registration (for example in projection stitching)
        self.projection_xy_stitching = False

    # Check methods for inputs
    def _check_input(self):
        """
        Internal function to check if the initial inputs are correct.
        """
        # Check if the coordinates are homogeneous
        try:
            self.physical_coordinates_mm = np.array(self.physical_coordinates_mm)
            # Check the shape
            if self.physical_coordinates_mm.shape[0] == 1:
                sys.exit("The list of coordinates must contain more than one coordinate!")
            elif self.physical_coordinates_mm[0].size != 3:
                sys.exit("3D coordinates required (x, y, z)!")
        except BaseException:
            sys.exit("The coordinates must be in a form of a list [[x_0, y_0, z_0], ..., [x_1, y_1, z_1]]!")

    # Method to get the coordinates in the image space from the axes correspondance and the given coordinates
    def get_image_space_coordinates(self):
        """
        Internal function to get the center position of the images in the pixel space from the provided motor coordinates.

        It signs the converted global coordinates as class attributes at self.global_coord_x_y_z (nx3 ndarray)
        """
        # Get the global coordinates in the image space, of each image center given the voxel size and the x, y, z correspondance and flip them if requested
        x_global = self.physical_coordinates_mm[:, np.abs(self.x_y_z_correspondance[0]) - 1] / self.mm_per_voxel * np.sign(self.x_y_z_correspondance[0])
        y_global = self.physical_coordinates_mm[:, np.abs(self.x_y_z_correspondance[1]) - 1] / self.mm_per_voxel * np.sign(self.x_y_z_correspondance[1])
        z_global = self.physical_coordinates_mm[:, np.abs(self.x_y_z_correspondance[2]) - 1] / self.mm_per_voxel * np.sign(self.x_y_z_correspondance[2])
        # Get the shape of the images of every layer
        self.img_shapes_xyz = np.array([Utilities.H5MaxIV.get_shape(img_path)[::-1] for img_path in self.file_paths])
        # Check if the z is the same for all the images. Otherwise exit and notify that this case has not been accounted for. The images should be padded beforehand.
        if np.unique(self.img_shapes_xyz[:, 2]).size > 1:
            sys.exit("All the stacks must have the same number of slices in z. Make sure to pad them with zeros beforehand!")
        # Get the outermost left and right and bottom edge of every image
        self.outer_most_left_xyz = np.divide(self.img_shapes_xyz, 2).astype(int)
        self.outer_most_right_xyz = self.img_shapes_xyz - np.divide(self.img_shapes_xyz, 2).astype(int)

        # Set the global coordinates as all positive and starting from 0 coordinates in the class attribute
        self.global_coord_x_y_z = np.column_stack((x_global - x_global.min(), y_global - y_global.min(), z_global - z_global.min())).astype(int) + np.max(self.outer_most_left_xyz, axis=0)

    def get_layers_in_z(self, tolerance_mm):
        """

        This method classifies the images into layers and puts them as attributes at:
                self.layers_coordinates --> list of nx3 ndarray where n is the number of images per layer.
                self.layers_paths --> list of strings where n is the number of images per layer.
                self.layers_img_shapes_xyz --> the shapes of the images (width, height, depth)
                self.layers_outer_most_left_xyz --> the number of pixels on the left of the image from the center (center = img.dimensions // 2)
                self.layers_outer_most_right_xyz --> the number of pixels on the right of the image from the center

        Parameters
        ----------
        tolerance_mm : int
            The tolerance is used to classify the layers along the z-image-axis based on the motor positions.
            A layer will contain all images having an equal: np.round(x.xxxx, tol=tolerance_mm).
            If the tolerance is expected to be in the order of mm, manually update the self.layers_coordinates and self.layers_paths.

        Returns
        -------
        dict
            A dictionary informing about the number of layers and their height in mm.

        """
        # Check if the global_coord_x_y_z has been initialized and do so if not
        if isinstance(self.global_coord_x_y_z, type(None)):
            self.get_image_space_coordinates()
        # Get the layers
        rounded_z_layers = np.round(self.global_coord_x_y_z[:, 2] * self.mm_per_voxel, tolerance_mm)
        z_layers = np.unique(rounded_z_layers)
        # Get the x and y positions within each rounded layer and the respective paths organized in the same way
        layers_coordinates = []
        layers_paths = []
        layers_img_shapes_xyz = []
        layers_outer_most_left_xyz = []
        layers_outer_most_right_xyz = []
        for i, height in enumerate(z_layers):
            mask_height = (rounded_z_layers == height)
            layers_coordinates.append(self.global_coord_x_y_z[mask_height])
            layers_paths.append([self.file_paths[j] for j in np.where(mask_height)[0]])
            layers_img_shapes_xyz.append(self.img_shapes_xyz[mask_height])
            layers_outer_most_left_xyz.append(self.outer_most_left_xyz[mask_height])
            layers_outer_most_right_xyz.append(self.outer_most_right_xyz[mask_height])
        # Update the gathered layer properties into the class
            # Set the coordinates and the paths
        self.layers_coordinates = [np.array(layer_coord) for layer_coord in layers_coordinates]
        self.layers_paths = layers_paths
        self.layers_img_shapes_xyz = layers_img_shapes_xyz
        self.layers_outer_most_left_xyz = layers_outer_most_left_xyz
        self.layers_outer_most_right_xyz = layers_outer_most_right_xyz

        return {"Number of layers": z_layers.shape[0], "Height in mm": z_layers}

    def get_padding(self):
        """
        Method that computes the padding required to form the "mosaic" of the stitch for every image based on their dimensions and the global positions in the pixel space.

        """
        # Compute the image padding to compose for each of them the same bounding box
        self.padding_neg_x = [layer_coord[:, 0] - layer_left[:, 0] for layer_coord, layer_left in zip(self.layers_coordinates, self.layers_outer_most_left_xyz)]  # Padding dimensions in the negative X direction grouped on layers (shape[0])
        self.padding_neg_y = [layer_coord[:, 1] - layer_left[:, 1] for layer_coord, layer_left in zip(self.layers_coordinates, self.layers_outer_most_left_xyz)]  # Padding dimensions in the negative Z direction grouped on layers (shape[0])

        self.padding_pos_x = [(self.global_coord_x_y_z[:, 0].max() + np.max(self.outer_most_right_xyz[:, 0], axis=0)) - (layer_coord[:, 0] + layer_right[:, 0]) for layer_coord, layer_right in zip(self.layers_coordinates, self.layers_outer_most_right_xyz)]  # Padding dimensions in the positive X direction grouped on layers (shape[0])
        self.padding_pos_y = [(self.global_coord_x_y_z[:, 1].max() + np.max(self.outer_most_right_xyz[:, 1], axis=0)) - (layer_coord[:, 1] + layer_right[:, 1]) for layer_coord, layer_right in zip(self.layers_coordinates, self.layers_outer_most_right_xyz)]  # Padding dimensions in the positive Z direction grouped on layers (shape[0])

    def check_padding(self, layer_index=0, circular_mask=False, radius_mask=None):
        """
        Method for checking how the "mosaic" of the stitch looks like. This could be useful to define the right correspondece between axes.

        Parameters
        ----------
        layer_index : INT, optional
            The index of the layer to be checked. The default is 0.
        circular_mask : TYPE, optional
            The circular mask acts on each image from the center. It will keep only the values within the radius_mask and use only those for the blending. The default is False.
        radius_mask : float, optional
            The radius of the circular mask from the center of each image. The default is None and uses half of the smallest dimension as radius.

        Returns
        -------
        layer_slices : ndarray
            A stitched slice of the layer.

        """
        # Padding the images: the img_slices_layers list contains the list of every slice grouped into layers
        layer_slices = []  # Temporary list to gather the padded slices of the layer

        i = layer_index  # point i to the layer index for shorter notation below
        for j, file_path in enumerate(self.layers_paths[i]):  # the iterator j is used to acces the padding values on each slice
            # Extract the middle slice
            extracted_slice = Utilities.H5MaxIV.get_slices(file_path, start_slice=self.img_depth // 2, end_slice=self.img_depth // 2, add_value_for_mask=self.add_value_for_mask)
            # Apply a mask if requested
            if circular_mask:
                extracted_slice = np.where(Utilities.circular_mask(extracted_slice, radius=radius_mask, center=(0, 0)), extracted_slice, 0)
            # Padd the middle slice
            pad_x_y = np.pad(extracted_slice, ((self.padding_neg_y[i][j], self.padding_pos_y[i][j]), (self.padding_neg_x[i][j], self.padding_pos_x[i][j])))
            # Append the padded slice to the layer slices list
            layer_slices.append(pad_x_y)

        # get the maximum values
        layer_slices = np.max(layer_slices, axis=0)

        # plot the image
        Utilities.plot_slice(layer_slices, title=f"Layer {layer_index}")

        return layer_slices

    def get_intersections(self, check=False, radius=None):
        '''
        Description: Separation of the intersection regions

        check --> It shows some results for the first volume and its intersections. Set it to true for debugging.
        '''
        # Get the intersection between images using the centers on the global coordinate system: self.layers_coordinates
        layers_intersecting_images_bb_x_y = []  # This array will keep the bounding boxes of the intersections j of a volume i corresponding to the order in the self.layers_coordinates or self.layers_paths. The bounding box is [(x_0, y_0), (x_1, y_1)]. The first entry is the reference volume itself as the distance is 0.
        layers_intersecting_images_mask_c = []  # This array will keep the mask centers of the intersections j of a volume i corresponding to the order in the self.layers_coordinates or self.layers_paths
        layers_intersecting_images_indices = []  # This array will keep the indices of the intersections j of a volume i corresponding to the order in the self.layers_coordinates or self.layers_paths
        # Iterate over every layer of coordinates
        for i, layer_coord in enumerate(self.layers_coordinates):
            # Initiate neighbors
            nbrs = NearestNeighbors(n_neighbors=layer_coord.shape[0], algorithm='ball_tree').fit(layer_coord)
            # Get distances from neighbors and their order
            distances, indices = nbrs.kneighbors(layer_coord)  # These arrays keep the distances and the indices of every intersecting slice for every slice in the current loop layer

            # the distances could be used then to get the intersections (distances < 2*radius or diameter)
            if type(radius) is None:
                intersecting_condition = (distances > 0) & (distances < 2 * self.radius)  # The (distances > 0) is added as the first index belongs to the slice itself
            else:
                # Compute deltas in x, y and z from the global positions of the image centers
                space_deltas_xyz = layer_coord[indices] - layer_coord[:, np.newaxis, :]  # shape: (n_samples, n_neighbors, 2)

                # Split into dx and dy (both shape: n_samples x n_neighbors)
                dx = np.abs(space_deltas_xyz[:, :, 0])
                dy = np.abs(space_deltas_xyz[:, :, 1])

                # Compute the distance that is covered by the image dimensions
                img_deltas_xyz = self.layers_img_shapes_xyz[i][indices] // 2 + self.layers_img_shapes_xyz[i][:, np.newaxis, :] // 2

                # Split into dx and dy (both shape: n_samples x n_neighbors)
                img_dx = img_deltas_xyz[:, :, 0]
                img_dy = img_deltas_xyz[:, :, 1]

                intersecting_condition = (distances > 0) & ((dx < img_dx) & (dy < img_dy))

            # a demonstration of how to get the intersecting volumes for every sub-volume in a layer: the indexes
            # intersecting_volumes_subv_0 = indices[0][intersecting_condition[0]]
            # print(intersecting_volumes_subv_0)

            intersecting_images_bb_x_y = []  # A temporary list to keep the intersecting bounding boxes of the  of every layer
            intersecting_images_mask_c = []  # A temporary list to keep the mask centers of the central volume for the intersecting volume
            intersecting_images_indices = []  # A temporary list to keep the intersecting indices of every layer

            for j, img_coord in enumerate(layer_coord):  # j keeps the layer index 0, 1 ... used to get other attributes of an image in their respective arrays
                # get the indexes of the intersecting subvolumes for the current subvolume in the loop
                intersecting_images = indices[j][intersecting_condition[j]]
                # getting the indices for this subvolume of the layer
                intersecting_images_indices.append(intersecting_images)

                # Extract the coordinates of the bounding box that defines the intersection with the current subvolume:
                # The global coordinates of subvolume in the full image
                x_0 = img_coord[0]
                y_0 = img_coord[1]

                x_0_left = self.layers_outer_most_left_xyz[i][j][0]
                y_0_left = self.layers_outer_most_left_xyz[i][j][1]

                x_0_right = self.layers_outer_most_right_xyz[i][j][0]
                y_0_right = self.layers_outer_most_right_xyz[i][j][1]

                x_0_dict = {"-1": x_0_left, "1": x_0_right, "0": 0}
                y_0_dict = {"-1": y_0_left, "1": y_0_right, "0": 0}

                # Check if it is working OK
                if (j == 0) and (i == 0) and (check):
                    # Check if it is working OK
                    print("Intersecting subvolumes full indexes:\n", indices)
                    print("\nIntersecting subvolume indexes for subvolume 0:", intersecting_images)
                    print("Subvolume 0 coord:")
                    print(x_0, y_0)

                # Iterate through each intersecting slice to get the points (p_0_x, p_0_y) and (p_1_x, p_1_y) defining the bounding box of the intersection between the current slice and the one in the following loop
                intersecting_images_bb_x_y_ = []  # A temporary array to keep the bounding boxes of the intersecting volumes of every image in the layer
                intersecting_images_mask_c_ = []  # A temporary array to keep the bounding boxes of the intersecting volumes of every image in the layer

                for k, intersection in enumerate(intersecting_images):
                    # The global coordinates of each intersecting slice in the full image
                    x_1 = layer_coord[intersection][0]
                    y_1 = layer_coord[intersection][1]

                    x_1_left = self.layers_outer_most_left_xyz[i][intersection][0]
                    y_1_left = self.layers_outer_most_left_xyz[i][intersection][1]

                    x_1_right = self.layers_outer_most_right_xyz[i][intersection][0]
                    y_1_right = self.layers_outer_most_right_xyz[i][intersection][1]

                    x_1_dict = {"-1": x_1_left, "1": x_1_right, "0": 0}
                    y_1_dict = {"-1": y_1_left, "1": y_1_right, "0": 0}

                    # Compute the bounding box for every intersection having the coordinates (p_0_x, p_0_y) and (p_3_x, p_3_y) on two of its opposite corners
                    # p_0_3_x = np.array((x_0 + np.sign(x_1 - x_0)*x_0_dict[str(np.sign(x_1 - x_0))], x_1 + np.sign(x_0 - x_1)*x_1_dict[str(np.sign(x_0 - x_1))]))
                    # p_0_3_y = np.array((y_0 + np.sign(y_1 - y_0)*y_0_dict[str(np.sign(y_1 - y_0))], y_1 + np.sign(y_0 - y_1)*y_1_dict[str(np.sign(y_0 - y_1))]))

                    p_0_3_x = np.array((x_0 + np.sign(x_1 - x_0) * x_0_dict[str(np.sign(x_1 - x_0))], x_1 + np.sign(x_1 - x_0) * x_1_dict[str(np.sign(x_1 - x_0))],
                                        x_0 + np.sign(x_0 - x_1) * x_0_dict[str(np.sign(x_0 - x_1))], x_1 + np.sign(x_0 - x_1) * x_1_dict[str(np.sign(x_0 - x_1))]))

                    p_0_3_y = np.array((y_0 + np.sign(y_1 - y_0) * y_0_dict[str(np.sign(y_1 - y_0))], y_1 + np.sign(y_1 - y_0) * y_1_dict[str(np.sign(y_1 - y_0))],
                                        y_0 + np.sign(y_0 - y_1) * y_0_dict[str(np.sign(y_0 - y_1))], y_1 + np.sign(y_0 - y_1) * y_1_dict[str(np.sign(y_0 - y_1))]))

                    # Compute the bounding box for every intersection having the coordinates (p_0_x, p_0_y) and (p_3_x, p_3_y) on two of its opposite corners
                    # p_0_3_x = np.array((x_0 + np.sign(x_1 - x_0)*self.radius, x_1 + np.sign(x_0 - x_1)*self.radius))
                    # p_0_3_y = np.array((y_0 + np.sign(y_1 - y_0)*self.radius, y_1 + np.sign(y_0 - y_1)*self.radius))

                    p_0_3_x = np.sort(p_0_3_x)[[1, 2]]
                    p_0_3_y = np.sort(p_0_3_y)[[1, 2]]

                    # In the previous computation the np.sign(x_0 - x_1) breaks for x_0 = x_1 or on z, which is when the volumes share one dimension entirely.
                    # The correction below makes sure to fix this singularity
                    if p_0_3_x[0] == p_0_3_x[1]:
                        p_0_3_x[0] = x_0 - x_0_left
                        p_0_3_x[1] = x_0 + x_0_right

                    if p_0_3_y[0] == p_0_3_y[1]:
                        p_0_3_y[0] = y_0 - y_0_left
                        p_0_3_y[1] = y_0 + y_0_right

                    # get the cropped image for of the subvolume at the intersection with the intersection
                    min_bb_x = p_0_3_x.min() - (x_0 - x_0_left)
                    min_bb_y = p_0_3_y.min() - (y_0 - y_0_left)
                    max_bb_x = p_0_3_x.max() - (x_0 - x_0_left)
                    max_bb_y = p_0_3_y.max() - (y_0 - y_0_left)

                    # cropped_intersection_unmasked = img_subvolumes_layers[i][j][:, min_bb_y:max_bb_y, min_bb_x:max_bb_x]
                    # img_subvolumes_layers[i][j][:, :, :]

                    # Check if it is working OK
                    if (i == 0) and (j == 0) and (check):
                        print(p_0_3_x, p_0_3_y)
                        print(f"Intersecting slice {intersection} and bounding box coords:")
                        print((x_1, y_1), p_0_3_x, p_0_3_y)
                        print("Command to check in Fiji: ")
                        print(f"makeRectangle({p_0_3_x.min()}, {p_0_3_y.min()}, {p_0_3_x.max() - p_0_3_x.min()}, {p_0_3_y.max() - p_0_3_y.min()})")
                        # plt.imshow(padded_images[i][j][int(p_0_3_x[0]):int(p_0_3_x[1]), int(p_0_3_y[0]):int(p_0_3_y[1])])
                        if k == 0:
                            first_intersection = Utilities.H5MaxIV.get_slices(self.layers_paths[i][j], start_slice=self.img_depth // 2, end_slice=self.img_depth // 2, add_value_for_mask=self.add_value_for_mask)
                            Utilities.plot_slice(first_intersection[min_bb_y:max_bb_y, min_bb_x:max_bb_x], title="First intersection taken during computation")

                    # Set the cropped and masked intersections into the gatherings of intersections for the subvolume
                    # dx = (p_0_3_x.min() - x_1) + (max_bb_x - min_bb_x) / 2 # The second term is added because the function circular_mask() automatically sends the center to the center of the image
                    # dy = (p_0_3_y.min() - y_1) + (max_bb_y - min_bb_y) / 2 # The second term is added because the function circular_mask() automatically sends the center to the center of the image
                    dx = x_1 - (p_0_3_x.min() + (max_bb_x - min_bb_x) / 2)  # The second term is added because the function circular_mask() automatically sends the center to the center of the image
                    dy = y_1 - (p_0_3_y.min() + (max_bb_y - min_bb_y) / 2)  # The second term is added because the function circular_mask() automatically sends the center to the center of the image
                    # Append the center to the gathering list of the image in the loop
                    intersecting_images_bb_x_y_.append(((min_bb_x, min_bb_y), (max_bb_x, max_bb_y)))
                    intersecting_images_mask_c_.append((dx, dy))  # It has to be provided as center(b,a) in the circular_mask() method
                    # intersecting_volume_slices.append(cropped_intersection_unmasked*circular_mask(cropped_intersection_unmasked, center=(b,a), radius=radius))

                # Append into the current layer group
                intersecting_images_bb_x_y.append(intersecting_images_bb_x_y_)
                intersecting_images_mask_c.append(intersecting_images_mask_c_)

            # Append into the global layer list
            layers_intersecting_images_bb_x_y.append(intersecting_images_bb_x_y)
            layers_intersecting_images_mask_c.append(intersecting_images_mask_c)
            # Composition for slicing purposes: intersecting_volumes_subv[layer, slice, intersection]
            layers_intersecting_images_indices.append(intersecting_images_indices)
        # Update the class attributes related to the intersections
        self.layers_intersecting_images_bb_x_y = layers_intersecting_images_bb_x_y
        self.layers_intersecting_images_mask_c = layers_intersecting_images_mask_c
        self.layers_intersecting_images_indices = layers_intersecting_images_indices

    def check_intersection(self, layer=0, image=0, intersection=0, mask=False, mask_radius=None):
        """
        Function to plot any intersection by defining the layer, image and intersecting indexes.

        Parameters
        ----------
        layer : INT, optional
            The layer index. The default is 0.
        image : INT, optional
            The image index. The default is 0.
        intersection : INT, optional
            The intersection index. The default is 0.
        mask : boolean, optional
            Apply a circular mask of radius mask_radius. The default is False.
        mask_radius : float, optional
            The radius for the mask. The default is None.

        """
        Utilities.plot_slice(self.extract_intersection(start_slice=self.img_depth // 2, end_slice=self.img_depth // 2, layer=layer, image=image, intersection=intersection, mask=mask, mask_radius=mask_radius), title="Extracted intersection")

    def extract_intersection(self, start_slice=None, end_slice=None, layer=0, image=0, intersection=0, mask=False, mask_radius=None):
        """
        Method used to extract a stack of slices from any intersection image of an image with one of its intersecting neighbors.

        Parameters
        ----------
        start_slice : INT, optional
            The start height to extract. The default is None.
        end_slice : INT, optional
            The end height to exctact. The default is None.
        layer : TYPE, optional
            DESCRIPTION. The default is 0.
        image : INT, optional
            The image where the intersection will be extracted. The default is 0.
        intersection : INT, optional
            The index of the intersecting image with the image. The default is 0.
        mask : boolean, optional
            Apply a circular mask. The default is False.
        mask_radius : float, optional
            The radius of the mask. The default is None.

        Returns
        -------
        intersection_bb : ndarray
            The stack of the slices from the intersection of image with the intersection.

        """
        if isinstance(self.layers_intersecting_images_bb_x_y, type(None)):
            sys.exit("Make sure to have run the get_intersections() method first")
        if isinstance(mask_radius, type(None)):
            mask_radius = self.radius
        # Get the bounding box for the specified layer-image-intersection
        x_0, y_0 = self.layers_intersecting_images_bb_x_y[layer][image][intersection][0]
        x_1, y_1 = self.layers_intersecting_images_bb_x_y[layer][image][intersection][1]
        # Extract the require image from the required layer
        intersection_slice = Utilities.H5MaxIV.get_slices(self.layers_paths[layer][image], start_slice=start_slice, end_slice=end_slice, add_value_for_mask=self.add_value_for_mask)
        # Apply a mask if required
        if mask:
            # Apply the mask for the first volume from its center
            intersection_slice = np.where(Utilities.circular_mask(intersection_slice, radius=mask_radius, center=(0, 0)), intersection_slice, 0)
            # Exctract the bounding box
            if start_slice == end_slice:
                intersection_bb = intersection_slice[y_0:y_1, x_0:x_1]
            else:
                intersection_bb = intersection_slice[:, y_0:y_1, x_0:x_1]
            # Apply the mask for the first volume from its first neighbor
            shift_columns, shift_rows = self.layers_intersecting_images_mask_c[layer][image][intersection]
            intersection_bb = np.where(Utilities.circular_mask(intersection_bb, radius=mask_radius, center=(shift_rows, shift_columns)), intersection_bb, 0)
        else:
            if start_slice == end_slice:
                intersection_bb = intersection_slice[y_0:y_1, x_0:x_1]
            else:
                intersection_bb = intersection_slice[:, y_0:y_1, x_0:x_1]
        # return the image
        return intersection_bb

    def correlate_intersection(self, start_slice=None, end_slice=None, crop_x=(0, 0), crop_y=(0, 0), equal_crop_xy=None, reduce_slices_LC=None, layer=0, image=0, intersection=0, mask=False, mask_radius=None, verbose=False, downscale=0.5, downscale_stages=2, downscale_LC=False, spline_interp=False, apply_mean_filter_zyx=(0, 0, 0), apply_detrend_filter_yx=(0, 0), apply_affine_warp=False, keep_rigid_only=True):
        """
        This function uses the correlation engine to perform a registration of the intersections extracted from the self.exctact_intersection. It operates in two steps:
            1 - A pixel search using the ZNCC to find the best match on downscaled images
            2 - A refined search using a lucas-kanade algorithm for any sub-pixel shift or affine transform
        The correlation engine used CUPY (CUDA implementation) and is therefore limited in amount of data that can process. A start and end slice as well as a crop are defined in this function to
        reduce the amount of data sent to the GPU. This makes the code flexible and fast. Be careful if in the overlap if there is not much information to correlate!

        The mask is set to False by default but consider setting it always to True to take into account the masking. Other ways of correlating are possible in the correlation engine

        The lukas-kanade step uses an erosion of the mask where the images overlap in order to avoid the interpolation artefacts at the borders (interpolation with the surrounding zeros!).
        An erosion of (11,11,11) is set by default. It can be adjusted from the class attributes.

        Other parameters could be used for finding the best mapping. Look at the correlation engines and modify the current method for more specific needs.

        Parameters
        ----------
        start_slice : INT, optional
            The starting slice of the intersection. The default is None.
        end_slice : TYPE, optional
            The ending slice of the intersection. The default is None.
        crop_x : INT, optional
            The crop in x_left and x_right defining the cropped intersection. The crop is intersection[:, :, crop_x[0]:crop_x[1]]. The default is (0,0).
        crop_y : INT, optional
            The crop in x_left and x_right defining the cropped intersection. The crop is intersection[:, crop_y[0]:crop_y[1], :]. The default is (0,0).
        equal_crop_xy: INT, optional
            When set to a value, a square region of dimensions (equal_crop_xy x equal_crop_xy) in the middle of the intersection will be selected for correlation. It limits the amount of data sent to the GPU considerably. The default is None.
            When active, the crop_x_y is not taken into account.
        reduce_slices_LC : TYPE, optional
            Reduce the number of slices for the Lucas-kanade refinement as this is not downscaled and might be too much for the GPU.
            It operates as following, intersection[reduce_slices_LC:-reduce_slices_LC, :, :]. The default is None.
        layer : TYPE, optional
            DESCRIPTION. The default is 0.
        image : INT, optional
            The image where the intersection will be extracted. The default is 0.
        intersection : INT, optional
            The index of the intersecting image with the image. The default is 0.
        mask : boolean, optional
            Apply a circular mask. The default is False.
        mask_radius : float, optional
            The radius of the mask. The default is None.
        verbose : boolean, optional
            Provides information about the correaltion process. The default is False.
        downscale : float, optional
            The downscale to be applied on downscale_stages stages. If smaller than one interpolation is used. If greater binning is used. The default is 0.5.
        downscale_stages : INT, optional
            The number of downscales to be performed. The final image will have intersection.shape*(downscale**downscale_stages). The default is 2.
        downscale_LC: boolean, optional
            When True, the Lucas-Kanade will perform on the downscaled images from the initial guess.
        spline_interp : boolean, optional
            If True, during the downscale the interpolation is a tri-cubic spline function. The default is False.
        apply_mean_filter_zyx : list(INT, INT, INT), optional
            A masked mean filter is applied. It is necessary if the features represent high frequency and are hard to correlate. The default is (0,0,0) and does not filter the images.
        apply_detrend_filter_yx: list(float, float), optional
            A masked detrend filter with gaussian kernel is applied. It is necessary if there is a lot of variation of grey-levels. The default is (0,0) and does not filter the images.
        apply_affine_warp : boolean, optional
            If False rigid registration is performed in the lucas-kanade algorithm. The default is False.
        keep_rigid_only : boolean, optional
            If True only the rigid part of the transformation is kept from the affine registration. The default is True.

        Returns
        -------
        disp_x : float
            The displacement in x taking the reference image to the intersection one.
        disp_y : float
            The displacement in x taking the reference image to the intersection one.
        disp_z : float
            The displacement in x taking the reference image to the intersection one.
        count : INT
            The number of iterations to convergence.
        T : 4x4 ndarray
            The transform operator taking the reference to the intersection.
        last_img : 3D ndarray
            Stack of images taken durine each registration step on the convergence process.
        ncc : float
            The final ZNCC in percentage.
        stats : (INT, INT)
            It does not do anything in this configuration.
        registered_slices : ndarray
            The slices at the reference and intersection after the best match from the pixel search.
        x_0 : float
            The x center of transformation on the global system of coordinates.
        y_0 : float
            The y center of transformation on the global system of coordinates.
        z_0 : float
            The z center of transformation on the global system of coordinates.
        local_T : 4x4 ndarray
            The transform operator taking the reference to the intersection on the local coordinate system of the intersection.

        """
        t_0 = timing()
        # Get a copy of the intersection as it will be replaced below!
        intersection_index = deepcopy(intersection)
        # Extract the intersections
        reference = self.extract_intersection(start_slice=start_slice, end_slice=end_slice, layer=layer, image=image, intersection=intersection, mask=mask, mask_radius=mask_radius)
        # Get the position of the intersection between the reference image and the intersecting image on the latter
        inter_image = self.layers_intersecting_images_indices[layer][image][intersection]
        inter_to_ref_ind = np.where(self.layers_intersecting_images_indices[layer][inter_image] == image)[0][0]
        # Extract the intersection image
        intersection = self.extract_intersection(start_slice=start_slice, end_slice=end_slice, layer=layer, image=inter_image, intersection=inter_to_ref_ind, mask=mask, mask_radius=mask_radius)
        # Check the equal crop
        if not isinstance(equal_crop_xy, type(None)):
            if (equal_crop_xy > np.min(reference.shape[1:])) or (equal_crop_xy <= 0):
                sys.exit("The shrink factor must be bigger than 0 and smaller than the smallest image dimension in x and y!")
            # Compute the crop in x and y required to reduce the size of the images
            half_depth, half_height, half_width = (np.array(reference.shape) / 2).astype(int)
            # In x
            crop_x_left = half_width - (equal_crop_xy // 2)
            crop_x_right = crop_x_left + equal_crop_xy
            # In y
            crop_y_left = half_height - (equal_crop_xy // 2)
            crop_y_right = crop_y_left + equal_crop_xy
            # Set the crop_x and crop_y
            crop_x = (crop_x_left, crop_x_right)
            crop_y = (crop_y_left, crop_y_right)
        # Get the cropping in x and y
        if crop_x != (0, 0):
            reference = reference[:, :, crop_x[0]:crop_x[1]]
            intersection = intersection[:, :, crop_x[0]:crop_x[1]]
        if crop_y != (0, 0):
            reference = reference[:, crop_y[0]:crop_y[1], :]
            intersection = intersection[:, crop_y[0]:crop_y[1], :]

        if verbose:
            print(f"To prepare data took {np.round(timing() - t_0, 1)} seconds")

        # Get an initial guess using a pixel search
        total_count = np.sum((reference != 0) & (intersection != 0))

        # A downscale factor for the initial correlation stage
        downscale = downscale
        downscale_stages = downscale_stages
        # When the mask is requested only regions with over 50% of the total count are accepted for the correlation results
        if downscale <= 1:
            min_count = 0.5 * total_count * ((downscale**3)**downscale_stages)
        else:
            min_count = 0.5 * total_count / (downscale**3)

        if apply_detrend_filter_yx != (0, 0):
            reference = Utilities.normalize_with_masked_gaussian_filter_cupy_2D(reference, sigma_np_xy=np.array(apply_detrend_filter_yx)[::-1], divide_by_norm=False, use_mask=True, notification_frequency=1000, value_return_mask=cp.array(0))
            intersection = Utilities.normalize_with_masked_gaussian_filter_cupy_2D(intersection, sigma_np_xy=np.array(apply_detrend_filter_yx)[::-1], divide_by_norm=False, use_mask=True, notification_frequency=1000, value_return_mask=cp.array(0))

        if apply_mean_filter_zyx != (0, 0, 0):
            reference = Utilities.mean_filter_masked(reference, element_shape=apply_mean_filter_zyx)
            intersection = Utilities.mean_filter_masked(intersection, element_shape=apply_mean_filter_zyx)

        # Run the correaltion
        pos_opt, NCC_opt, N_opt, registered_slices, intersection_ncc, reference_ncc = RegistrationKIT.correlate_NCC(reference,
                                                                                                                    intersection,
                                                                                                                    downscale=downscale,
                                                                                                                    downscale_stages=downscale_stages,
                                                                                                                    use_spline=spline_interp,
                                                                                                                    use_mask_template=mask,
                                                                                                                    use_mask_search=mask,
                                                                                                                    use_minimun_count=mask,
                                                                                                                    mask_threshold=(-1E-10, 1E-10),
                                                                                                                    minimum_count=min_count,
                                                                                                                    apply_gaussian_img_x_y_z=(0, 0, 0),
                                                                                                                    apply_gaussian_NCC_x_y_z=(0, 0, 0))

        if verbose:
            print(f"To NCC correlation took {np.round(timing() - t_0, 1)} seconds")

        if not downscale_LC:
            # Get the center of the full images
            c_0, c_1, c_2 = np.array(reference.shape) // 2
            z, y, x = (c_0 - pos_opt[0], c_1 - pos_opt[1], c_2 - pos_opt[2])
        else:
            # Sign the new images
            reference = reference_ncc
            intersection = intersection_ncc
            # Get the center of the downscaled images
            c_0, c_1, c_2 = np.array(reference.shape) // 2
            # Get the z, y and x in the downscaled version
            if downscale < 1:
                downscale = 1 / (downscale ** downscale_stages)
            # Inverse the operation from the NCC correlation
            z, y, x = (c_0 - (pos_opt[0] / downscale),
                       c_1 - (pos_opt[1] / downscale),
                       c_2 - (pos_opt[2] / downscale))

        if verbose:
            print(f"Initial correlation x = {x}, y = {y}, z = {z}, NCC = {NCC_opt}")

        # In case of projection stitching in xy only set the displacement in z to 0
        if self.projection_xy_stitching:
            z = 0

        # Runnning the Lucas-Kanade optimization step
        # Compose the guess_transform
        guess_transform = np.array([[1, 0, 0, z],
                                    [0, 1, 0, y],
                                    [0, 0, 1, x],
                                    [0, 0, 0, 1]])

        # set the sigmas of the gaussian derivatives
        sigma_multiscale = ((1, 1, 1),)

        # the erosion to be applied at each scale. Tip: use at least double the gaussian sigma as a thumb of rule or as much as your maximal displacement.
        erosion_multiscale = ((self.erosion_mask_LC_xyz[2],
                               self.erosion_mask_LC_xyz[1],
                               self.erosion_mask_LC_xyz[0]),)

        # the convergence criteria to be used at each scale. If regulate is set to True make sure to decrease further this convergence
        conv_criteria_multiscale = (0.01,)

        # the type of warps to be applied at each scale
        apply_affine_warps = (apply_affine_warp,)

        # The type of derivatives to be used at each scale
        der_types = ("else",)

        if (not isinstance(reduce_slices_LC, type(None))) and (reduce_slices_LC != 0):
            # Extract the intersections
            reference = reference[reduce_slices_LC:-reduce_slices_LC, :, :]
            intersection = intersection[reduce_slices_LC:-reduce_slices_LC, :, :]

        # Correlate
        for sigma, erosion_element, conv_criteria, affine_warp, der_type in zip(sigma_multiscale, erosion_multiscale, conv_criteria_multiscale, apply_affine_warps, der_types):

            if NCC_opt != -100:
                disp_x, disp_y, disp_z, count, T, last_img, ncc, stats = RegistrationKIT.lucas_kanade_3D_inv_mask(reference.astype(np.float32),
                                                                                                                  intersection.astype(np.float32),
                                                                                                                  derivatives=der_type,
                                                                                                                  sigma_z_y_x=sigma,
                                                                                                                  mask=mask,
                                                                                                                  erodeMask=mask,
                                                                                                                  erosionElement=np.ones((erosion_element[0],
                                                                                                                                          erosion_element[1],
                                                                                                                                          erosion_element[2])),
                                                                                                                  convergence_criteria=conv_criteria,
                                                                                                                  initial_guess=guess_transform,
                                                                                                                  max_iter=50,
                                                                                                                  interp_order=1,
                                                                                                                  regulate=False,
                                                                                                                  slice_extract=reference.shape[0] // 2,
                                                                                                                  affine_warp=affine_warp,
                                                                                                                  rigid_warp=self.force_rigid_warp,
                                                                                                                  xy_reg=self.projection_xy_stitching)
                if verbose:
                    print(f"To Lucas-Kanade took {np.round(timing() - t_0, 1)} seconds")
                # Make sure to scale the rigid displacements in case of a downscaled image
                if (downscale_LC) and (downscale != 1):
                    disp_x *= downscale
                    disp_y *= downscale
                    disp_z *= downscale
                    T[0][-1] *= downscale
                    T[1][-1] *= downscale
                    T[2][-1] *= downscale
            else:
                disp_x, disp_y, disp_z, count, T, last_img, ncc, stats = (0.0, 0.0, 0.0, 0, np.eye(4), np.array(0), [np.array(0), np.array(0)], [0, 0])

        # In case of an affine warp, the rigid displacement is taken from the middle point of the correlating volumes
        # Get the vector of the local coordinate system where the correlation acts, (x_0, y_0, z_0)
        # x_0 and y_0
        x_0, y_0 = np.add(self.layers_intersecting_images_bb_x_y[layer][image][intersection_index][0], (self.padding_neg_x[layer][image] + crop_x[0], self.padding_neg_y[layer][image] + crop_y[0]))
        # z_0
        if not isinstance(reduce_slices_LC, type(None)):
            z_0 = start_slice + reduce_slices_LC
        else:
            z_0 = start_slice

        if apply_affine_warp:
            # keep only the best rigid transform if necessary
            if keep_rigid_only:
                # Extract the rigid transform only
                # Separate R  --> rotation tensor, U --> stretch tensor by polar decomposition
                R, U = polar(T[:-1, :-1])
                # Get the rigid transform
                T = np.pad(R, ((0, 1), (0, 1))) + np.pad(T[:, -1][..., np.newaxis], ((0, 0), (3, 0)))

            # Compute global shift required to form the global operator, a transformation matrix that can be applied from the origin of the stitching mosaic.
            global_shift_zyx = T[:-1, :-1] @ np.array([z_0, y_0, x_0]) - np.array([z_0, y_0, x_0])
            # global_shift_zyx = ((T-np.eye(4)) @ np.array([z_0, y_0, x_0, 1]))[:-1]

            # Compose the global operator in zyx format
            T_1 = deepcopy(T)
            global_operator_T_zyx = np.array([[T_1[0][0], T_1[0][1], T_1[0][2], T_1[0][3] - global_shift_zyx[0]],
                                              [T_1[1][0], T_1[1][1], T_1[1][2], T_1[1][3] - global_shift_zyx[1]],
                                              [T_1[2][0], T_1[2][1], T_1[2][2], T_1[2][3] - global_shift_zyx[2]],
                                              [0, 0, 0, 1]])
            # Compute the shift at the center of the image
            disp_x, disp_y, disp_z = ((global_operator_T_zyx - np.eye(4)) @ np.append(np.add((z_0, y_0, x_0), self.layers_img_shapes_xyz[layer][image][::-1] / 2), 1))[:-1][::-1]

            # Replace the T by the global operator in xyz format
            # Global operator
            T = np.array([[T_1[2][2], T_1[2][1], T_1[2][0], T_1[2][3] - global_shift_zyx[2]],
                          [T_1[1][2], T_1[1][1], T_1[1][0], T_1[1][3] - global_shift_zyx[1]],
                          [T_1[0][2], T_1[0][1], T_1[0][0], T_1[0][3] - global_shift_zyx[0]],
                          [0, 0, 0, 1]])
            # Local operator
            local_T = np.array([[T_1[2][2], T_1[2][1], T_1[2][0], T_1[2][3]],
                                [T_1[1][2], T_1[1][1], T_1[1][0], T_1[1][3]],
                                [T_1[0][2], T_1[0][1], T_1[0][0], T_1[0][3]],
                                [0, 0, 0, 1]])
        else:
            local_T = T
        return disp_x, disp_y, disp_z, count, T, last_img, ncc, stats, registered_slices, x_0, y_0, z_0, local_T

    def compute_shift_in_layers(self, start_slice=None, end_slice=None, crop_x=(0, 0), crop_y=(0, 0), equal_crop_xy=None, reduce_slices_LC=None, mask=False, mask_radius=None, verbose=False, save_reg=False, save_path=None, downscale=0.5, downscale_stages=2, downscale_LC=False, spline_interp=False, apply_mean_filter_zyx=(0, 0, 0), apply_detrend_filter_yx=(0, 0), apply_affine_warp=False, keep_rigid_only=True):
        """
        The main function calling the registration of intersections for every layer.

        Parameters (Look at the self.correlate_intersection() for the rest of the parameters)
        ----------

        save_reg : boolean, optional
            It True the registration files are saved in a h5 format with most of the data integrated in a single file. The default is False.
        save_path : str, optional
            The path to saving the registration files. If None the os.getcwd() will be used. The default is None.

        """
        # Update the class parameters
        self.start_slice_reg = start_slice
        self.end_slice_reg = end_slice
        self.crop_x_reg = crop_x
        self.crop_y_reg = crop_y
        self.reduce_slices_LC_reg = reduce_slices_LC
        self.affine_warp = apply_affine_warp

        # Check if saving is requested
        if save_reg:
            if (isinstance(save_path, type(None))) and (isinstance(self.saving_path, type(None))):  # Better way to check for None
                registration_saving_path = os.path.join(os.getcwd(), "Correlation_process")  # Remove leading "/"
            else:
                if (not isinstance(save_path, type(None))):
                    registration_saving_path = os.path.join(save_path, "Correlation_process")  # Remove leading "/"
                else:
                    registration_saving_path = os.path.join(self.saving_path, "Correlation_process")  # Remove leading "/"

            # Create the directory and notify the user
            os.makedirs(registration_saving_path, exist_ok=True)
            print("Registration data will be saved at:", registration_saving_path)

        # Get the number of correlations to be performed
        correlations_left = 0
        for i, layer_indices in enumerate(self.layers_intersecting_images_indices):
            for j, reference_image in enumerate(layer_indices):
                for k, reference_intersection in enumerate(reference_image):
                    correlations_left += 1

        self.layers_reg_disp_data = []  # The main list containing the organized registration data
        self.layers_reg_cent_data = []  # The main list containing the organized registration data
        self.layers_reg_oper_data = []  # The main list containing the organized registration data in a form of transformation matrix
        self.layers_reg_ncc_data = []  # The main list containing the organized correlation coefficient data
        self.layers_reg_diff_data = []  # The main list containing the organized gray level difference stats of the template and registered images

        for i, layer_indices in enumerate(self.layers_intersecting_images_indices):
            # Make sure to avoid duplicating the computations
            correlated_volumes = []  # This list will contain the sets of correlated volumes in order to avoid duplicates
            correlation_data = []  # This list will contain the correlation results from the already correlated volumes in order to fill-in the duplica in the ordered lists

            # Iterate over the subvolumes of each layer and correlate their respective intersections
            registration_subvolumes = []  # A temporary list to collect the registrations of every layer
            transformation_center = []  # A temporary list to collect the origin of the registrations for every layer
            operator_subvolumes = []  # A temporary list to collect the transform operator of every layer
            ncc_subvolumes = []  # A temporary list to collect the normalized correlation coefficient of every layers registration
            gray_level_diff_subvolumes = []  # A temporary list to collect the gray level difference stats of every layers registration

            for j, reference_subvolumes in enumerate(layer_indices):
                # the index j is equal to that of the reference subvolume: the extension _slices refers to volumes in this case but the script was adapted from the 2D version
                registration_subvolumes_ = []  # A temporary list to collect the registrations of every intersection with the main subvolume
                transformation_center_ = []  # A temporary list to collect the origin of the registrations of every intersection with the main subvolume in the global coordinate system
                operator_subvolumes_ = []  # A temporary list to collect the transform operator of every intersection with the main subvolume
                ncc_subvolumes_ = []  # A temporary list to collect the ncc of every intersection with the main subvolume
                gray_level_diff_subvolumes_ = []  # A temporary list to collect the gray level difference stats of every intersection with the main subvolume

                current_jobs_status = []  # a dummy list to carry out the information about the jobs that were passed or not for every reference subvolume
                results = []  # a dummy list to contain the results. this was added after the new cuda implementation to avoid the multiprocessing pool below

                reference_img_title = self.file_paths[j]

                for k, reference_intersection in enumerate(reference_subvolumes):
                    current_pair = sorted([j, reference_intersection])  # sort the arrays for simple comparisons
                    intersection_img_title = self.layers_paths[i][reference_intersection]  # the path of the intersecting image
                    if current_pair not in correlated_volumes:
                        result_current_pair = self.correlate_intersection(start_slice=start_slice,
                                                                          end_slice=end_slice,
                                                                          crop_x=crop_x,
                                                                          crop_y=crop_y,
                                                                          equal_crop_xy=equal_crop_xy,
                                                                          reduce_slices_LC=reduce_slices_LC,
                                                                          layer=i,
                                                                          image=j,
                                                                          intersection=k,
                                                                          mask=mask,
                                                                          mask_radius=mask_radius,
                                                                          verbose=verbose,
                                                                          downscale=downscale,
                                                                          downscale_stages=downscale_stages,
                                                                          downscale_LC=downscale_LC,
                                                                          spline_interp=spline_interp,
                                                                          apply_mean_filter_zyx=apply_mean_filter_zyx,
                                                                          apply_detrend_filter_yx=apply_detrend_filter_yx,
                                                                          apply_affine_warp=apply_affine_warp,
                                                                          keep_rigid_only=keep_rigid_only)

                        # Append to the results list
                        results.append(result_current_pair)
                        # Update the job status and already correlated volumes
                        current_jobs_status.append(1)  # the current job passed
                        correlated_volumes.append(current_pair)  # the current image pair passed
                    else:
                        current_jobs_status.append(0)  # a similar job has already been done

                # Unwrap the results depending on the job status
                trace_results = 0
                for job_idx, job_status in enumerate(current_jobs_status):
                    moving_intersection_indice = reference_subvolumes[job_idx]  # get the row to be synchronize the current loop to the previous one
                    if job_status == 1:
                        # Gather computation data
                        disp_x, disp_y, disp_z, count, T, last_img, ncc, stats, registered_slices, x_0, y_0, z_0, local_T = results[trace_results]
                        # Check for nan and if so return default values
                        if not np.isnan(ncc[1]):
                            registration_subvolumes_.append((disp_x, disp_y, disp_z))
                            transformation_center_.append((x_0, y_0, z_0))
                            operator_subvolumes_.append(T)
                            ncc_subvolumes_.append(int(ncc[1]))
                            gray_level_diff_subvolumes_.append(stats)
                        else:
                            registration_subvolumes_.append((0, 0, 0))
                            transformation_center_.append((0, 0, 0))
                            operator_subvolumes_.append(np.eye(4))
                            ncc_subvolumes_.append(-100)
                            gray_level_diff_subvolumes_.append((0, 0))

                        # Save registration data if required
                        if save_reg:
                            # reduce_slices_LC=None, mask=False, mask_radius=None, verbose=False, save_reg=False, save_path=None, downscale=0.5, downscale_stages=2, spline_interp=False, apply_mean_filter_zyx=(0,0,0)
                            data_set_title = ("image", "disp_xyz", "global_operator_xyz1", "affine_warp", "iterations", "ZNCC", "Start_end_slice", "crop_x_y", "equal_crop_xy", "reduce_LC", "mask", "mask_radius", "downscale", "downscale_stages", "spline", "mean_filter", "reference_intersection", "local_operator_xyz")
                            data_set_data = (np.array(last_img), np.array([disp_x, disp_y, disp_z]), T, apply_affine_warp, count, float(ncc[1]), np.array((start_slice, end_slice)), np.array((crop_x, crop_y)), str(equal_crop_xy), str(reduce_slices_LC), mask, str(mask_radius), downscale, downscale_stages, spline_interp, apply_mean_filter_zyx, (reference_img_title, intersection_img_title), local_T)

                            with h5py.File(registration_saving_path + f"/registration_{i}_{j}_{job_idx}.h5", "w") as h5f:
                                h5_file = h5f.create_group('registration_data', track_order=True)
                                for title, data_set in zip(data_set_title, data_set_data):
                                    h5_file.create_dataset(title, data=data_set)

                            with h5py.File(registration_saving_path + f"/registration_NCC_{i}_{j}_{job_idx}.h5", "w") as h5f:
                                h5_file = h5f.create_dataset("image", data=np.array(registered_slices))

                        # Fill-in the computation data of the pair
                        if not np.isnan(ncc[1]):
                            correlation_data.append((disp_x, disp_y, disp_z, count, T, last_img, ncc, stats, registered_slices, x_0, y_0, z_0))
                        else:
                            correlation_data.append((0, 0, 0, -1, np.eye(4), 0, (-100, -100), (0, 0), 0, 0, 0, 0))
                        trace_results += 1
                    else:
                        if verbose:
                            print(sorted([j, moving_intersection_indice]), "--> already done")
                        # Gather data from previous computations: The displacements is inversed before appending the data.
                        result_indice = correlated_volumes.index(sorted([j, moving_intersection_indice]))
                        disp_x, disp_y, disp_z, count, T, last_img, ncc, stats, registered_slices, x_0, y_0, z_0 = correlation_data[result_indice]
                        # Compose subvolume data together into a single list
                        registration_subvolumes_.append((-disp_x, -disp_y, -disp_z))
                        transformation_center_.append((x_0, y_0, z_0))
                        operator_subvolumes_.append(np.linalg.inv(T))
                        ncc_subvolumes_.append(int(ncc[1]))
                        gray_level_diff_subvolumes_.append((-stats[0], stats[1]))  # When obtained from the other way around the mean value of the difference changes sign

                    # Compute the number of correlations left
                    correlations_left = correlations_left - 1

                    # Check if the images are being selected correctly
                    if verbose:
                        print(f"Layer_{i}_subvolume_{j}_intersection_{moving_intersection_indice} --> dx, dy, dz: {disp_x:.2f}, {disp_y:.2f}, {disp_z:.2f} (NCC = {ncc[1]:.1f}, Difference stats = {np.round(stats, 4)}) ")

            # Append the data to every set of the reg_disp_data list
                registration_subvolumes.append(np.array(registration_subvolumes_))
                transformation_center.append(np.array(transformation_center_))
                operator_subvolumes.append(operator_subvolumes_)
                ncc_subvolumes.append(ncc_subvolumes_)
                gray_level_diff_subvolumes.append(gray_level_diff_subvolumes_)
            self.layers_reg_disp_data.append(registration_subvolumes)
            self.layers_reg_cent_data.append(transformation_center)
            self.layers_reg_oper_data.append(operator_subvolumes)
            self.layers_reg_ncc_data.append(ncc_subvolumes)
            self.layers_reg_diff_data.append(gray_level_diff_subvolumes)

    def get_displacement_pyramid(self, check=False, starting_coord=None):
        """
        This function computes the so-called displacement pyramid. It starts from an image whose center is closest to the starting_coord and propagates across the neighbors accumulating the
        displacement, affine operators and other information about the stitching order.

        Parameters
        ----------
        check: boolean, optional
            IT SHOWES SOME INFO ABOUT THE PYRAMID IF TRUE. The default is False.
        starting_coord: (INT, INT, INT), optional
            THE COORDINATE WHICH WILL BE USED TO GET THE REFERENCE VOLUME (THE ONE HAVING THE CLOSEST CENTER TO THIS ONE BASED ON PIXEL POSITIONS OBTAINED FROM THE MOTOR ONES). The default is (0,0,0)
        """

        # Only one layer will be used to define the stitching order. Set it down below:
        # starting_layer = 0

        # To keep the same subvolume fixed for every layer after the first cycle, in the other layers you must look for the closest point in xy (image coordinates) or in xz (rotation stage coordinates)
        self.intersections_pyramid_sub_layer = []  # A global array to contain the intersections_pyramids of every layer
        self.displacements_pyramid_sub_layer = []  # A global array to contain the displacement_pyramids of every layer
        self.operator_pyramid_sub_layer = []  # A global array to contain the operator_pyramid of every layer
        self.reg_cent_pyramid_sub_layer = []  # A global array to contain the registration center of every layer
        self.displacements_ncc_pyramid_sub_layer = []  # A global array to contain the normalized cross correlation coefficient for the warped displacements of every expansion layer of intersections
        self.gray_level_diff_pyramid_sub_layer = []  # A global array to contain the gray level difference statistics between the templates and the intersecting images after registration of every expansion layer of intersections
        self.central_neighbor_pyramid_sub_layer = []  # A global array to contain the previous or originating neighbors of every step so that the full displacement vector can be computed later. The first one is set to none as the fixed volume does not have any previous neighbor.

        for starting_layer, i in enumerate(self.layers_intersecting_images_indices):
            if (starting_layer == 0) and (isinstance(starting_coord, type(None))):
                # Select the starting volume: with most neighbors
                no_neigh_st_layer = np.array([len(img_index) for img_index in self.layers_intersecting_images_indices[starting_layer]])
                # Select the first one on the array: there might be a couple with the same number of neighbors
                starting_img = no_neigh_st_layer.argmax()
                # Coordinates of the starting point
                starting_coord = self.layers_coordinates[starting_layer][starting_img]
            else:
                # Here to add something that makes sure to select the same starting point (in terms of xy approximity) for every img_index. The elements taken from coordinates are (True, False, True) as they are based on the motor positions
                starting_coord = np.array(starting_coord)
                distances = np.array([np.round(np.sqrt(np.sum((point[np.array((True, True, False))] - starting_coord[np.array((True, True, False))])**2)), 0) for point in self.layers_coordinates[starting_layer]])
                starting_img = distances.argmin()

            # Select step by step the other ones
            all_indices = []  # This array will keep track of the volumes (indices) that will have already be placed into the mesh
            for img_index in self.layers_intersecting_images_indices[starting_layer]:
                all_indices.extend(img_index)
            all_indices = np.unique(np.array(all_indices))
            all_indices = np.delete(all_indices, all_indices == starting_img)  # Delete since we are starting directly with this one in the search loop below --> current_layer list

            # Get all the layers and the respective displacements over every intersection
            current_layer = [starting_img]  # This array will keep the elements of the current expansion layer through which all the intersections will be gathered in a loop
            intersections_pyramid_sub = [[starting_img]]  # The list that will contain the slices of every expansion layer of intersections
            displacements_pyramid_sub = []  # The list that will contain the displacements of every expansion layer of intersections
            operator_pyramid_sub = []  # The list that will contain the transform operator of every expansion layer of intersections
            reg_cent_pyramid_sub = []  # The list that will contain the registration cener of every expansion layer of intersections
            displacements_ncc_pyramid_sub = []  # The list that will contain the normalized cross correlation coefficient for the warped displacements of every expansion layer of intersections
            gray_level_diff_pyramid_sub = []  # The list that will contain the gray level difference statistics between the templates and the intersecting images after registration of every expansion layer of intersections
            central_neighbor_pyramid_sub = []  # This list will contain the previous or originating neighbors of every step so that the full displacement vector can be computed later. The first one is set to none as the fixed volume does not have any previous neighbor.
            count = 0  # A dummy count to control the while loop
            while all_indices.shape[0] != 0:
                intersections_slices = []  # Temporary array to keep the intersections in a current expansion layer
                intersections_displacements = []  # Temporary array to keep the displacements in a current expansion layer
                intersections_operators = []  # Temporary array to keep the transform operator in a current expansion layer
                intersections_reg_centers = []  # Temporary array to keep the registration center in a current expansion layer
                intersections_ncc = []  # Temporary array to keep the ncc for the warped displacements in a current expansion layer
                intersections_gray_level_diff = []  # Temporary array to keep the gray level difference statistics in a current expansion layer
                intersections_central_neighbor = []  # Temporary array to keep the originating central neighbor of the current expansion layer
                for img_index in current_layer:
                    # Full intersections of the img_index:
                    intersections_slice = self.layers_intersecting_images_indices[starting_layer][img_index]
                    # Full displacements of the img_index:
                    displacements_slice = self.layers_reg_disp_data[starting_layer][img_index]
                    # Full transform operator of the img_index:
                    operator_slice = self.layers_reg_oper_data[starting_layer][img_index]
                    # Full center of transform of the img_index:
                    intersections_reg_center = self.layers_reg_cent_data[starting_layer][img_index]
                    # Ncc of the img_index
                    displacement_ncc_slice = self.layers_reg_ncc_data[starting_layer][img_index]
                    # Grey level difference stats of the img_index
                    gray_level_diff_slice = self.layers_reg_diff_data[starting_layer][img_index]
                    # Get a mask that contains the positions of the intersections remaining in all_indices but also not containing the current_layer:
                    mask_intersections = (np.isin(intersections_slice, all_indices) & ~np.isin(intersections_slice, current_layer))
                    # Extend the new intersections to the intersections_slices gathering the intersections per img_index:
                    intersections_slices.extend(intersections_slice[mask_intersections])
                    # Extend the new displacements to the intersections_displacements gathering the intersections per img_index:
                    intersections_displacements.extend(displacements_slice[mask_intersections])
                    # Extend the new transform operator to the intersections_displacements gathering the intersections per img_index:
                    intersections_operators.extend(np.array(operator_slice)[mask_intersections])
                    # Extend the new transformation center to the intersections_displacements gathering the intersections per img_index:
                    intersections_reg_centers.extend(np.array(intersections_reg_center)[mask_intersections])
                    # Extend the new ncc of the displacements to the intersections_ncc gathering the ncc per img_index:
                    intersections_ncc.extend(np.array(displacement_ncc_slice)[mask_intersections])
                    # Extend the new gray_level_diff_volumes_sub of the displacements to the intersections_ncc gathering the ncc per img_index:
                    intersections_gray_level_diff.extend(np.array(gray_level_diff_slice)[mask_intersections])
                    # Extend the central neighbors to the intersections
                    intersections_central_neighbor.extend(np.ones(np.sum(mask_intersections)).flatten().astype(int) * img_index)
                # Add the remaining intersections and displacements to the pyramids:
                intersections_pyramid_sub.append(intersections_slices)
                displacements_pyramid_sub.append(intersections_displacements)
                operator_pyramid_sub.append(intersections_operators)
                reg_cent_pyramid_sub.append(intersections_reg_centers)
                displacements_ncc_pyramid_sub.append(intersections_ncc)
                gray_level_diff_pyramid_sub.append(intersections_gray_level_diff)
                central_neighbor_pyramid_sub.append(intersections_central_neighbor)
                # Update the current layer:
                current_layer = np.array(intersections_slices).copy()
                # Remove the added intersections and displacements from the all_indices:
                all_indices = np.delete(all_indices, np.isin(all_indices, intersections_slices))
                # Print something to check if it works correctly
                if check:
                    if count == 0:
                        print(f"Unique labels for the layer {count} and originating layer {[None]}:", intersections_pyramid_sub[0])
                    print(f"Unique labels for the layer {count + 1} and originating layer {np.unique(intersections_central_neighbor)}:", np.unique(intersections_slices))
                count += 1

            # Fill in the global lists
            self.intersections_pyramid_sub_layer.append(deepcopy(intersections_pyramid_sub))  # Collecting the intersections_pyramids of every layer
            self.displacements_pyramid_sub_layer.append(deepcopy(displacements_pyramid_sub))  # Collecting the displacement_pyramids of every layer
            self.operator_pyramid_sub_layer.append(deepcopy(operator_pyramid_sub))
            self.reg_cent_pyramid_sub_layer.append(deepcopy(reg_cent_pyramid_sub))
            self.displacements_ncc_pyramid_sub_layer.append(deepcopy(displacements_ncc_pyramid_sub))  # Collecting the normalized cross correlation coefficient for the warped displacements of every expansion layer of intersections
            self.gray_level_diff_pyramid_sub_layer.append(deepcopy(gray_level_diff_pyramid_sub))  # Collecting the grey level differences for every expansion layer of intersections
            self.central_neighbor_pyramid_sub_layer.append(deepcopy(central_neighbor_pyramid_sub))  # Collecting the previous or originating neighbors of every step.

        # scatter the points to verify the point alignment. The z is inversed to make for a similar view as in image xy coordinate system
        if check:
            for starting_layer, i in enumerate(self.intersections_pyramid_sub_layer):
                fig, ax = plt.subplots()
                count = 0
                for intersections_slices in self.intersections_pyramid_sub_layer[starting_layer]:
                    for point in np.unique(intersections_slices):
                        if (count % 2 == 1):
                            ax.scatter(self.layers_coordinates[starting_layer][point, 0], -self.layers_coordinates[starting_layer][point, 1], color='b')
                            ax.annotate(point, (self.layers_coordinates[starting_layer][point, 0], -self.layers_coordinates[starting_layer][point, 1]))
                        else:
                            ax.scatter(self.layers_coordinates[starting_layer][point, 0], -self.layers_coordinates[starting_layer][point, 1], color='r')
                            ax.annotate(point, (self.layers_coordinates[starting_layer][point, 0], -self.layers_coordinates[starting_layer][point, 1]))
                    count += 1
                # Add circle patches
                plt.xlabel("X (voxels)")
                plt.ylabel("-Y (voxels)")
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                plt.title(f"Layer_{starting_layer}")
                print(f"Layer_{starting_layer} --> Intersections, Central Neighbor, NCC:")
                print(self.intersections_pyramid_sub_layer[starting_layer])
                print(self.central_neighbor_pyramid_sub_layer[starting_layer])
                # for expansion in gray_level_diff_pyramid_sub_layer[starting_layer]:
                #    print(np.round(expansion, 2))
                print(self.displacements_ncc_pyramid_sub_layer[starting_layer])

    def accumulate_displacement(self, exclude_NCC=80, verbose=False, weighted_avg=True, affine_operator=False):
        """
        This function accumulates the displacements, the affine operators and the transformation centers based on the order defined on the previous pyramid.

        Parameters
        ----------
        exclude_NCC: INT, optional
            THE REGISTRATION DATA HAVING A LOWER ZNCC THAN THIS PARAMETER WILL NOT BE CONSIDERED. The default is 80.
        verbose: boolean, optional
            IT PRINTS OUT INFORMATION ABOUT THE PROCESS. The default is False.
        weighted_avg: boolean, optional
            WHEN TRUE THE DISPLACEMENT OF A CERTAIN VOLUME IN THE PYRAMID IS AN WEIGHTED AVERAGE FROM ALL ITS PREVIOUS INTERSECTIONS LEADING IT TO THE FIXED VOLUME.
            WHEN FALSE, THE DISPLACEMENT IS OBTAINED FROM THE SINGLE VOLUME ON THE PREVIOUS LAYER HAVING THE HIGHEST ZNCC. The default is True.
        affine_operator: boolean, optional
            WHEN THE AFFINE OPERATOR IS SET TO TRUE THE AFFINE OPERATOR IS ALSO ACCUMULATED. IN THIS CASE THE ONE RESULTING FROM THE MAXIMAL ZNCC FROM THE PREVIOUS NEIGHBORS IS KEPT. The default is False.
        """

        self.exclude_NCC = exclude_NCC  # The value in percentage to exclude the results of a given correlation when averiging the displacements based on the neighborhood

        # The affine operator has no weighted average implemented
        if affine_operator:
            weighted_avg = False
            print("The weighted average on accumulate_displacement is set to False when affine operator")

        self.cumulative_displacements_pyramid_sub_layer = []  # A global array to gather the cumulative_displacements_pyramid_sub of every layer
        self.cumulative_global_operators_pyramid_sub_layer = []  # A global array to gather the cumulative_displacements_pyramid_sub of every layer
        self.cumulative_global_cent_pyramid_sub_layer = []  # A global array to gather the cumulative_cent_pyramid_sub of every layer
        self.cumulative_NCC_pyramid_sub_layer = []  # A global array to gather the NCC of every layer

        for j, displacements_pyramid_sub in enumerate(self.displacements_pyramid_sub_layer):

            # cumulative_displacements_pyramid_sub = displacements_pyramid_sub.copy()
            cumulative_displacements_pyramid_sub = deepcopy(self.displacements_pyramid_sub_layer[j])
            cumulative_operator_pyramid_sub = deepcopy(self.operator_pyramid_sub_layer[j])
            cumulative_cent_pyramid_sub = deepcopy(self.reg_cent_pyramid_sub_layer[j])
            cumulative_NCC_pyramid_sub = deepcopy(self.displacements_ncc_pyramid_sub_layer[j])

            # For every layer if the NCC is smaller than the threshold do not take it into account. This is necessary as it is not checked within the loop below.
            for iloop, ncc_ in enumerate(self.displacements_ncc_pyramid_sub_layer[j][0]):
                if ncc_ < exclude_NCC:
                    cumulative_displacements_pyramid_sub[0][iloop] = np.array([0, 0, 0])
                    cumulative_NCC_pyramid_sub[0][iloop] = 0

            if verbose:
                print(f"Layer: {j}")

            unique_volumes_index = []  # An array to keep the indexes of the volumes that have already been computed so that the computation does not repeat in subvolumes having multiple neighbors
            unique_volumes_disp_vector = []  # An array to keep the cumulated displacement of the first computation
            unique_volumes_global_operator = []  # An array to keep the cumulated transform operator of the first computation
            unique_volumes_center = []  # An array to keep the cumulated transform center of the first computation
            unique_NCC_pyramid_sub = []  # An array to keep the best NCC of the first computation
            for expansion_layer in range(0, len(self.intersections_pyramid_sub_layer[j])):
                for intersection_in_layer in range(len(self.intersections_pyramid_sub_layer[j][expansion_layer])):
                    if expansion_layer > 1:  # Starting from 1 as the fixed subvolume does not have neighbors nor displacement nor ncc.
                        # Get the current subvolume
                        current_subvolume = self.intersections_pyramid_sub_layer[j][expansion_layer][intersection_in_layer]  # [expansion_layer - 1] must remove 1 as the first layer was skipped since the lists are not filled for the fixed subvolume contrary to the intersections_pyramid_copy of this loop
                        # Check if already done
                        if current_subvolume not in unique_volumes_index:
                            # Set in the unique volumes index
                            unique_volumes_index.append(current_subvolume)
                            # Get the positions of the current subvolumes in the current expansion layer (if more than one means that the current_subvolume intersects multiple volumes from the previous layer)
                            current_subvolumes_in_layer = (self.intersections_pyramid_sub_layer[j][expansion_layer] == current_subvolume)
                            # Get the central subvolumes of the current subvolume from the previous expansion layer
                            previous_neighbors = np.array(self.central_neighbor_pyramid_sub_layer[j][expansion_layer - 1])[current_subvolumes_in_layer]
                            # Get the displacement vectors of the current subvolume relative to its central subvolumes from the cumulative_displacements_pyramid_sub
                            displacement_vectors_v = np.array(cumulative_displacements_pyramid_sub[expansion_layer - 1])[current_subvolumes_in_layer]
                            # Get the transform operator of the current subvolume relative to its central subvolumes from the cumulative_operator_pyramid_sub_layer
                            global_operator_matrix_v = np.array(cumulative_operator_pyramid_sub[expansion_layer - 1])[current_subvolumes_in_layer]
                            # Get the center of the transform operator of the current subvolume relative to its central subvolumes from the cumulative_operator_pyramid_sub_layer
                            transformation_center_v = np.array(cumulative_cent_pyramid_sub[expansion_layer - 1])[current_subvolumes_in_layer]

                            # Get the correlation coefficients between the current_subvolume and the neighbors from ncc_volumes_sub[layer, subvolume, intersection] qs well as the displacement vectors
                            correlation_coeff = []  # array keeping the NCC of the correlation between the current subvolume and its central ones
                            displacement_vectors_u = []  # array keeping the accumulated displacements of the central subvolumes of the current one
                            weighted_sum_ncc = 0
                            no_accounted_neighbors = 0
                            averaged_disp_current = 0
                            max_affine_global_operator = np.eye(4)
                            max_transformation_center = np.zeros(3)
                            NCC = 0.0

                            max_NCC = deepcopy(exclude_NCC)

                            for i, neighbor in enumerate(previous_neighbors):
                                # Get the position in the correlation coefficients list and fill in the current one
                                correl_coeff_pos = (self.layers_intersecting_images_indices[j][current_subvolume] == neighbor)
                                correl_coeff_value = np.array(self.layers_reg_ncc_data[j][current_subvolume])[correl_coeff_pos]
                                correlation_coeff.append(correl_coeff_value[0])

                                # Get the displacement of the previous expansion layer
                                displacement_pos = (self.intersections_pyramid_sub_layer[j][expansion_layer - 1] == neighbor)
                                displacement_vector = np.array(cumulative_displacements_pyramid_sub[expansion_layer - 2])[displacement_pos]
                                displacement_vectors_u.append(displacement_vector[0])

                                # Get the transform operator of the previous layer
                                operator_matrix = np.array(cumulative_operator_pyramid_sub[expansion_layer - 2])[displacement_pos]

                                # Get the sum of displacements (U_1 + V_1) weighted by the correlation_coef: [NCC_1*(U_1 + V_1) + ...... + NCC_2*(U_2 + V_2)] / len(previous_neighbors) / max(NCC)
                                if (correl_coeff_value[0] > exclude_NCC) and weighted_avg:
                                    weighted_sum_ncc += ((displacement_vector[0] + displacement_vectors_v[i]) * correl_coeff_value[0])
                                    no_accounted_neighbors += 1

                                elif (correl_coeff_value[0] > max_NCC):
                                    # Get the displacement
                                    weighted_sum_ncc = displacement_vector[0] + displacement_vectors_v[i]
                                    max_NCC = deepcopy(correl_coeff_value[0])
                                    no_accounted_neighbors = 1
                                    # Get the affine operator
                                    max_affine_global_operator = global_operator_matrix_v[i] @ operator_matrix[0]
                                    # Get the global transformation center
                                    max_transformation_center = transformation_center_v[i]

                            # Get the weighted average
                            if no_accounted_neighbors != 0:
                                if weighted_avg:
                                    averaged_disp_current = weighted_sum_ncc / no_accounted_neighbors / np.max(correlation_coeff)
                                    NCC = np.mean(correlation_coeff)
                                else:
                                    averaged_disp_current = weighted_sum_ncc
                                    NCC = np.max(correlation_coeff)
                            else:
                                averaged_disp_current = np.array([0, 0, 0])
                                NCC = 0.0

                            # Check the results
                            if verbose:
                                print(f"Current_subvolume_{current_subvolume}_central_subvolumes_{previous_neighbors}_NCC_{correlation_coeff} ---> Averaged displacement vector = {averaged_disp_current}")
                                for i, disp in enumerate(displacement_vectors_u):
                                    print(f"Previous_displacements_{np.round(disp, 2)}. Current_displacement_{np.round(displacement_vectors_v[i], 2)}")

                            # Update the entries in cumulative_displacements_pyramid_sub and operator
                            cumulative_displacements_pyramid_sub[expansion_layer - 1][intersection_in_layer] = deepcopy(averaged_disp_current)
                            cumulative_operator_pyramid_sub[expansion_layer - 1][intersection_in_layer] = deepcopy(max_affine_global_operator)
                            cumulative_cent_pyramid_sub[expansion_layer - 1][intersection_in_layer] = deepcopy(max_transformation_center)
                            cumulative_NCC_pyramid_sub[expansion_layer - 1][intersection_in_layer] = deepcopy(NCC)

                            # Keep this value for repeated ones
                            unique_volumes_disp_vector.append(deepcopy(averaged_disp_current))
                            unique_volumes_global_operator.append(deepcopy(max_affine_global_operator))
                            unique_volumes_center.append(deepcopy(max_transformation_center))
                            unique_NCC_pyramid_sub.append(deepcopy(NCC))
                        else:
                            cumulative_displacements_pyramid_sub[expansion_layer - 1][intersection_in_layer] = unique_volumes_disp_vector[unique_volumes_index.index(current_subvolume)]
                            cumulative_operator_pyramid_sub[expansion_layer - 1][intersection_in_layer] = unique_volumes_global_operator[unique_volumes_index.index(current_subvolume)]
                            cumulative_cent_pyramid_sub[expansion_layer - 1][intersection_in_layer] = unique_volumes_center[unique_volumes_index.index(current_subvolume)]
                            cumulative_NCC_pyramid_sub[expansion_layer - 1][intersection_in_layer] = unique_NCC_pyramid_sub[unique_volumes_index.index(current_subvolume)]
            # Append to the global array
            self.cumulative_displacements_pyramid_sub_layer.append(deepcopy(cumulative_displacements_pyramid_sub))
            self.cumulative_global_operators_pyramid_sub_layer.append(deepcopy(cumulative_operator_pyramid_sub))
            self.cumulative_global_cent_pyramid_sub_layer.append(deepcopy(cumulative_cent_pyramid_sub))
            self.cumulative_NCC_pyramid_sub_layer.append(deepcopy(cumulative_NCC_pyramid_sub))
        # === IMPORTANT: see notes below ===

        # The displacement in the cumulative_displacement_pyramid_sub is from the central neighbor to the current subvolume of the layer so when moving the inages the displacement vector must be inversed.
        # Also the cumulative_displacement_pyramid_sub contains repeated elements in the expanding layers due to one or more subvolumes intersecting the same central neighbor. When looking for a displacement.
        # you must get one of the correspondences as the average values of the displacement are normally set to be the same.

        # === end IMPORTANT ===
    def check_accumulated_displacement(self, verbose=False):
        """
        Description: Plot the displacement at every step to see if it is the correct way or not.
        verbose: Set to True to see more info during the process.
        """
        # scatter the points to verify the point alignment. The z is inversed to make for a similar view as in image xy coordinate system
        for starting_layer, nothing in enumerate(self.intersections_pyramid_sub_layer):
            if verbose:
                print(f"Layer_{starting_layer}")
            fig, ax = plt.subplots()
            count = 0
            for i, intersections_slices in enumerate(nothing):
                for j, point in enumerate(intersections_slices):
                    # Old coordinates
                    old_x = self.layers_coordinates[starting_layer][point, 0]
                    old_y = -self.layers_coordinates[starting_layer][point, 1]
                    ax.scatter(old_x, old_y, color='b')
                    ax.annotate(point, (old_x, old_y))
                    if i > 0:
                        # New coordinates
                        if verbose:
                            print(f"Point: {point} -->{-np.round(self.cumulative_displacements_pyramid_sub_layer[starting_layer][i - 1][j], 1)}")
                        new_x = self.layers_coordinates[starting_layer][point, 0] - self.cumulative_displacements_pyramid_sub_layer[starting_layer][i - 1][j][0]  # The minus is required due to the way the lists are built
                        new_y = -self.layers_coordinates[starting_layer][point, 1] + self.cumulative_displacements_pyramid_sub_layer[starting_layer][i - 1][j][1]  # The plus is required because the z axis is flipped to make for a similar look to an image.
                        ax.scatter(new_x, new_y, color='r')
                        ax.quiver([old_x], [old_y], [new_x - old_x], [new_y - old_y], angles='xy', scale_units='xy', scale=0.025, color='g', headwidth=3)
                count += 1
            # Add circle patches
            plt.xlabel("X (voxels)")
            plt.ylabel("-Y (voxels)")
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

    def compose_final_displacements(self, verbose=False):
        """

        This function gets the transformation of every volume and pack them into an array in a similar order as in >>> self.layers_intersecting_images_indices = [] <<<,
        where the positional indexes of every entry in a given layer correspond to the the indices found in the intersections_pyramid_sub_layer. This is used for the final
        transformation and the stitching order.

        Set verbose to True to see more info during the process.

        """
        # Create a list that will contain the displacement values
        self.layer_final_displacements = []
        self.layer_final_operators = []
        self.layer_final_NCC = []
        self.layer_final_centers_xyz = []
        self.layer_final_global_coordinates = []

        # Create dummy list to keep track of displacements
        layer_track_indexes = []
        for i, layer in enumerate(self.layers_intersecting_images_indices):
            gather_layer_indexes = []  # An array to gather the subvolume indexes per layer
            gather_layer_displacements = []  # An array to gather the subvolume indexes per layer
            gather_layer_operators = []  # An array to gather the subvolume transform operators per layer
            gather_layer_centers = []  # An array to gather the subvolume transform operators per layer
            gather_layer_global_coordinates = []  # An array to gather the global coordinates per layer
            gather_layer_final_NCC = []  # An array to gather the final NCC per layer
            for j, subvolume in enumerate(layer):
                gather_layer_indexes.append(j)  # append the indexes which will be transformed into -1 as we fill in the displacements
                gather_layer_displacements.append(np.array([0.0, 0.0, 0.0]))  # append something stupidly large
                gather_layer_operators.append(np.eye(4))
                gather_layer_centers.append(np.zeros(3))
                gather_layer_global_coordinates.append(np.array([0.0, 0.0, 0.0]))
                gather_layer_final_NCC.append(100.0)
            # Fill in the global lists
            layer_track_indexes.append(gather_layer_indexes)
            self.layer_final_displacements.append(gather_layer_displacements)
            self.layer_final_operators.append(gather_layer_operators)
            self.layer_final_centers_xyz.append(gather_layer_centers)
            self.layer_final_global_coordinates.append(gather_layer_global_coordinates)
            self.layer_final_NCC.append(gather_layer_final_NCC)

        # Fill in the arrays in layer_final_displacements by checking the remaining indexes in layer_track_indexes
        for i, layer_intersections in enumerate(self.intersections_pyramid_sub_layer):
            if verbose:
                print(f"Layer_{i}:")
            for j, layer_subvolumes in enumerate(layer_intersections):
                for k, subvolume in enumerate(layer_subvolumes):
                    # Start by filling-in the fixed volumes position with a 0 value as this one is not found in the displacements_pyramid_sub_layer
                    if j == 0:
                        # Get the final x,y,z global coordinates
                        self.layer_final_global_coordinates[i][layer_track_indexes[i].index(subvolume)] = self.layers_coordinates[i][layer_track_indexes[i].index(subvolume)]
                        # Print to check
                        if verbose:
                            print(f"Subvolume_{subvolume}_{self.layer_final_displacements[i][layer_track_indexes[i].index(subvolume)]}. New global coord: ~{np.round(self.layer_final_global_coordinates[i][layer_track_indexes[i].index(subvolume)])}")
                        # Set the index as done
                        layer_track_indexes[i][layer_track_indexes[i].index(subvolume)] = -1
                    else:  # Continue with the rest / normally the first list of the intersections_pyramid_sub_layer contains a single element so it should not pass to j>0
                        if subvolume in layer_track_indexes[i]:
                            # Set the displacement
                            self.layer_final_displacements[i][layer_track_indexes[i].index(subvolume)] = -self.cumulative_displacements_pyramid_sub_layer[i][j - 1][k]
                            self.layer_final_operators[i][layer_track_indexes[i].index(subvolume)] = np.linalg.inv(self.cumulative_global_operators_pyramid_sub_layer[i][j - 1][k])
                            self.layer_final_centers_xyz[i][layer_track_indexes[i].index(subvolume)] = self.cumulative_global_cent_pyramid_sub_layer[i][j - 1][k]
                            self.layer_final_NCC[i][layer_track_indexes[i].index(subvolume)] = self.cumulative_NCC_pyramid_sub_layer[i][j - 1][k]
                            # Get the final x and z global coordinates
                            old_xyz = self.layers_coordinates[i][layer_track_indexes[i].index(subvolume)]
                            new_xyz = old_xyz + self.layer_final_displacements[i][layer_track_indexes[i].index(subvolume)]
                            self.layer_final_global_coordinates[i][layer_track_indexes[i].index(subvolume)] = new_xyz
                            # Print to check
                            if verbose:
                                print(f"Subvolume_{subvolume}_{-np.round(self.cumulative_displacements_pyramid_sub_layer[i][j - 1][k], 2)}. New global coord: ~{np.round(self.layer_final_global_coordinates[i][layer_track_indexes[i].index(subvolume)])}")
                            # Set the index as done
                            layer_track_indexes[i][layer_track_indexes[i].index(subvolume)] = -1

    def stitch_volumes_blend_equalize(self,
                                      stitch_layer=None,
                                      start_slice=None,
                                      end_slice=None,
                                      mask=False,
                                      mask_radius=None,
                                      alpha=1,
                                      use_equalize=False,
                                      use_existing_equalize=False,
                                      normalize_dist_radially=True,
                                      square_dist=False,
                                      crop_x=(0, 0),
                                      crop_y=(0, 0),
                                      exclude_indexes=[],
                                      exclude_NCC=True,
                                      show_progress_bar=True):
        """
        Description:
        Stitches and blends the slices based on the global coordinates (computed from padding and image dimensions). This is done step by step based on the given stitching_index_order specifying the order of the slices in the img_slices.
        pad_x_y_j
        Attention: The mask is extracted based on 0 values, so consider using different grey values for the image if this becomes an issue.
        Whenever this function is called, a list of temporary attributes keeping these configurations for the stitching is generated within the class. Using the push_parameters these could then be used for the full stitching. This includes
        most of parameters but not all. The equalizing parameters are kept separately in the class whenever the use_equalize is set to True.

        Parameters
        ----------
        stitch_layer: INT. optional
            THE LAYER TO BE STITCHED. IF NONE ALL THE LAYERS ARE DONE IN SERIES; THIS ALLOWS TO COLLECT THE EQUALIZE PARAMETERS FOR INSTANCE FOR THE FULL STITCHING LATER. The default is None.
        start_slice : INT, optional
            The starting slice of the intersection. The default is None.
        end_slice : TYPE, optional
            The ending slice of the intersection. The default is None.
        mask : boolean, optional
            Apply a circular mask. The default is False.
        mask_radius : float, optional
            The radius of the mask. The default is None.
        alpha : float, optional.
            The blending parameter: img_slices_j_1*(1-distance_map_red**(alpha)) + intersection_j_1*(distance_map_red**(alpha)) where the img_slices_j_1 is the image being set in the mosaic and intersection_j_1 are the already
            existing images in the mosaic. The default is 1.
        use_equalize : boolean, optional.
            Equalizes the image being put on the mosaic using the joint histogram in the intersection. The default is False.
            When True, the equalizing parameters are kept in the class attributes for the full stitch later if requested.
        use_existing_equalize : boolean, optional
            When True, if a previous call of the function was performed with the use_equalize option the latter are used as parameters. This is useful for the full stitching later to separate the equalizing term from the rest
            of the parameters. The default is False.
        crop_x : (INT, INT), optional
            The crop in x_left and x_right defining the cropped mosaic. In this case the crop is defined as mosaic[:, :, crop_x[0]:mosaic.shape[2] - crop_x[1]]. The default is (0,0).
        crop_y : INT, optional
            The crop in y_left and y_right defining the cropped mosaic. In this case the crop is defined as mosaic[:, crop_y[0]:mosaic.shape[1] - crop_y[1], :]. The default is (0,0).
        exclude_NCC : boolean. optional.
            `When True it does not stitch the volumes whoose final NCC is below the NCC defined in the accumulate_displacement method. This NCC can be changed manually by the self.exlude_NCC
        exclude_indexes : [INT, ]. optional.
            A list of indexes on the layer to be avoided during the stitching.
        show_progress_bar: boolean, optional.
            If True a progress bar is shown during the process.
        """
        # Re-initialize the layer_equalize_slope_intercept if use_equalize=True to save the slope and intercept
        if use_equalize:
            self.layer_equalize_slope_intercept = []

        # Keep the passed parameters in the temporary space
        self.params_temp = {"start_slice": start_slice, "end_slice": end_slice, "mask": mask, "mask_radius": mask_radius, "alpha": alpha, "use_equalize": use_equalize, "normalize_dist_radially": normalize_dist_radially, "square_dist": square_dist, "crop_x": crop_x, "crop_y": crop_y, "exclude_NCC": exclude_NCC}

        if isinstance(stitch_layer, type(None)):
            iterator = [i for i, layer_paths in enumerate(self.layers_paths)]
        else:
            iterator = (stitch_layer,)

        # In case all the layers are done at once the collected_slices collects the padded and stitched slices of every layer and then returns it
        if len(iterator) > 1:
            collected_slices = []

        for _, stitch_layer_ in enumerate(iterator):
            # Get the stitching order
            stitching_index_order = [subvolume_index for layer in self.intersections_pyramid_sub_layer[stitch_layer_] for subvolume_index in layer]

            # Exclde any index if required
            for index_to_exclude in exclude_indexes:
                stitching_index_order.remove(index_to_exclude)

            # Get the check lists to avoid any double computation
            already_passed_index = [None for elem in np.unique(stitching_index_order)]  # None for passed indexes: will be turned to index values during the loop

            if show_progress_bar:
                # Initialize tqdm inside the loop, so it refreshes per `stitch_layer_`
                progress_bar = tqdm(total=len(stitching_index_order) - 2, desc=f"Stitching Layer {stitch_layer_}", ncols=100)

            # An element to keep the padded stitched slices while looping
            pad_x_y_j = None

            if use_equalize:
                # An array to keep the equalization parameters for every layer
                layer_equalize_slope_intercept_ = []

            for i, slice_index in enumerate(stitching_index_order[:-1]):  # the iterator i is used to acces the padding values on each slice
                # Index for the consecutive slices
                j = stitching_index_order[i]
                j_1 = stitching_index_order[i + 1]
                # Check if the index has been passed
                if already_passed_index[j_1] is None:
                    # Check in case of a bad correlation
                    if (self.layer_final_NCC[stitch_layer_][j_1] > self.exclude_NCC) or not exclude_NCC:
                        if isinstance(self.affine_warp, type(None)) or not self.affine_warp:
                            # Extract a bunch of slices to account for the translation
                            img_slices_j = self.get_transformed_slices(layer=stitch_layer_, image=j, start_slice=start_slice, end_slice=end_slice, mask=mask, mask_radius=mask_radius)
                            img_slices_j_1 = self.get_transformed_slices(layer=stitch_layer_, image=j_1, start_slice=start_slice, end_slice=end_slice, mask=mask, mask_radius=mask_radius)
                        else:
                            img_slices_j = self.get_transformed_slices_affine(layer=stitch_layer_, image=j, start_slice=start_slice, end_slice=end_slice, mask=mask, mask_radius=mask_radius, chunk_size=1)
                            img_slices_j_1 = self.get_transformed_slices_affine(layer=stitch_layer_, image=j_1, start_slice=start_slice, end_slice=end_slice, mask=mask, mask_radius=mask_radius, chunk_size=1)
                        # Initialize the pad_x_y_j with the fixed slice
                        if i == 0:
                            pad_x_y_j = np.pad(img_slices_j,
                                               ((0, 0),
                                                (self.padding_neg_y[stitch_layer_][j], self.padding_pos_y[stitch_layer_][j]),
                                                (self.padding_neg_x[stitch_layer_][j], self.padding_pos_x[stitch_layer_][j])),
                                               mode='constant',
                                               constant_values=0)

                        # For debugging only
                        # with h5py.File(f"/data/visitors/danmax/20240666/2024101508/process/trial_stitch_endri/test_stitch_SiNMC_1_overview1_offlineServer/pad_x_y_j_{i}.h5", "w") as h5f: h5f.create_dataset("image", data=pad_x_y_j)

                        # Get the distance function for the fixed image
                        if not square_dist:
                            distance_map = Utilities.dist_function(img_slices_j_1, center=(0, 0))
                        else:
                            distance_map = Utilities.dist_function_sq(img_slices_j_1, center=(0, 0), prop_x_y=self.prop_x_y)
                        # Extract the region that will be covered by the slice_j_1 on the pad_x_y_j image
                        intersection_j_1 = pad_x_y_j[:, self.padding_neg_y[stitch_layer_][j_1]:self.padding_neg_y[stitch_layer_][j_1] + img_slices_j_1.shape[1], self.padding_neg_x[stitch_layer_][j_1]:self.padding_neg_x[stitch_layer_][j_1] + img_slices_j_1.shape[2]]
                        # Get the reduced distance map within the intersection region where the blending will be applied
                        distance_map_red = np.where((intersection_j_1 != 0) & (img_slices_j_1 != 0), distance_map, 0)
                        # Make sure that the if belo works if the distance_map_red returns None (no intersection or slice beyond boundaries)
                        if distance_map_red.size == 0:
                            distance_map_red = np.array((0))
                        # Get a mask where the distance_map_red is different from zero for further processing in the overlap region (different from 0)
                        distance_map_red_mask = (distance_map_red != 0)
                        # Continue with further processing only if an overlap exists:
                        if distance_map_red.max() != 0:
                            if not normalize_dist_radially:
                                # Get the minimum and maximum values within the distance_map_red on the overlap region (different from 0)
                                distance_map_max = distance_map_red[distance_map_red_mask].max()
                                distance_map_min = distance_map_red[distance_map_red_mask].min()
                                # Redistribute the distance map between 0 and 1
                                if (distance_map_max - distance_map_min) != 0:
                                    distance_map_red = (distance_map_red - distance_map_min) / (distance_map_max - distance_map_min)
                                else:
                                    print("Could not normalize the distance field due to overlap issues (division by 0)")
                                # Convert all the other values outside of the overlap to 0
                                distance_map_red = np.where(distance_map_red_mask, distance_map_red, 0)
                            else:
                                distance_map_red = Utilities.normalize_distance_map_radially(distance_map_red, center=None)

                            # Convert all the values that could be part of the pad_x_y_j image to 1
                            distance_map_red = np.where((intersection_j_1 != 0) & (~distance_map_red_mask), 1, distance_map_red)

                            if use_equalize:
                                # Get the slope and intercept of the joint histogram for equalizing the slices
                                slope, intercept, r, p, se = linregress(img_slices_j_1[distance_map_red_mask], intersection_j_1[distance_map_red_mask])
                                # Equalize the gray levels on the j_1 image
                                # img_slices_j_1 = np.where(img_slices_j_1 != 0, (img_slices_j_1 + intercept) * slope, 0)
                                img_slices_j_1 = np.where(img_slices_j_1 != 0, (img_slices_j_1 * slope + intercept), 0)
                                # Append to the global collector
                                layer_equalize_slope_intercept_.append((slope, intercept))

                            elif use_existing_equalize:
                                img_slices_j_1 = np.where(img_slices_j_1 != 0, (img_slices_j_1 + self.layer_equalize_slope_intercept[stitch_layer_][i][1]) * self.layer_equalize_slope_intercept[stitch_layer_][i][0], 0)

                            # for any debugging
                            # with h5py.File(f"/data/visitors/danmax/20240666/2024101508/process/trial_stitch_endri/test_stitch_SiNMC_1_overview1_offlineServer/distance_map_red_{i}.h5", "w") as h5f: h5f.create_dataset("image", data=distance_map_red)
                            # with h5py.File(f"/data/visitors/danmax/20240666/2024101508/process/trial_stitch_endri/test_stitch_SiNMC_1_overview1_offlineServer/img_slices_j_1_{i}.h5", "w") as h5f: h5f.create_dataset("image", data=img_slices_j_1)
                            # with h5py.File(f"/data/visitors/danmax/20240666/2024101508/process/trial_stitch_endri/test_stitch_SiNMC_1_overview1_offlineServer/intersection_j_1_{i}.h5", "w") as h5f: h5f.create_dataset("image", data=intersection_j_1)
                            # with h5py.File(f"/data/visitors/danmax/20240666/2024101508/process/trial_stitch_endri/test_stitch_SiNMC_1_overview1_offlineServer/img_slices_j_{i}.h5", "w") as h5f: h5f.create_dataset("image", data=img_slices_j)

                            # Blend the current image to the pad_x_y_j
                            if np.isnan(alpha):
                                slice_j_1 = np.where((intersection_j_1 != 0) & (img_slices_j_1 != 0), 0, img_slices_j_1 + intersection_j_1)
                            else:
                                slice_j_1 = img_slices_j_1 * (1 - distance_map_red**(alpha)) + intersection_j_1 * (distance_map_red**(alpha))
                        else:
                            # If no overlap is found set the original slice as it is (normally only zeros)
                            slice_j_1 = img_slices_j_1
                            # Append a 0,0 in case of a use equalize
                            if use_equalize:
                                layer_equalize_slope_intercept_.append((0, 0))
                        # Set the current slice to the pad_x_y_j image
                        pad_x_y_j[:, self.padding_neg_y[stitch_layer_][j_1]:self.padding_neg_y[stitch_layer_][j_1] + img_slices_j_1.shape[1], self.padding_neg_x[stitch_layer_][j_1]:self.padding_neg_x[stitch_layer_][j_1] + img_slices_j_1.shape[2]] = deepcopy(slice_j_1)
                    else:
                        # Append a 0,0 in case of a use equalize
                        if use_equalize:
                            layer_equalize_slope_intercept_.append((0, 0))
                        # Initialize the pad_x_y_j with the fixed slice in case the first neighbor is not compliant
                        if i == 0:
                            # Get the first image
                            img_slices_j = self.get_transformed_slices(layer=stitch_layer_, image=j, start_slice=start_slice, end_slice=end_slice, mask=mask, mask_radius=mask_radius)
                            # Generate the stitch mesh
                            pad_x_y_j = np.pad(img_slices_j,
                                               ((0, 0),
                                                (self.padding_neg_y[stitch_layer_][j], self.padding_pos_y[stitch_layer_][j]),
                                                (self.padding_neg_x[stitch_layer_][j], self.padding_pos_x[stitch_layer_][j])),
                                               mode='constant',
                                               constant_values=0)

                # Fill in the equalizing parameters to make sure it keeps the same format as the layer properties. (-1, -1 will be however used)
                elif use_equalize:
                    layer_equalize_slope_intercept_.append((-1, -1))

                # Fill in the check lists
                already_passed_index[j_1] = j_1

                if show_progress_bar:
                    progress_bar.update(1)  # Update the progress bar on every iteration

            if use_equalize:
                self.layer_equalize_slope_intercept.append(deepcopy(layer_equalize_slope_intercept_))

            if show_progress_bar:
                progress_bar.close()  # Close the progress bar after finishing all iterations

            if len(iterator) > 1:
                collected_slices.append(deepcopy(pad_x_y_j[:, crop_y[0]:pad_x_y_j.shape[1] - crop_y[1], crop_x[0]:pad_x_y_j.shape[2] - crop_x[1]]))
        if len(iterator) == 1:
            return pad_x_y_j[:, crop_y[0]:pad_x_y_j.shape[1] - crop_y[1], crop_x[0]:pad_x_y_j.shape[2] - crop_x[1]]
        else:
            return collected_slices

    def push_stitch_parameters(self):
        """
        This function sets the current temporary stitch parameters in the final ones that can then be used for the full stitching.
        """
        if self.params_temp is None:
            # No test stitch was performed (e.g. caller skipped the equalization
            # learning step). Populate sensible defaults so that stitch_layers()
            # can still read crop_x / crop_y / mask / etc. without crashing.
            self.params_final = {
                "start_slice": None,
                "end_slice": None,
                "mask": True,
                "mask_radius": None,
                "alpha": 1.0,
                "use_equalize": False,
                "normalize_dist_radially": False,
                "square_dist": False,
                "crop_x": (0, 0),
                "crop_y": (0, 0),
                "exclude_NCC": True,
            }
        else:
            self.params_final = deepcopy(self.params_temp)
        return print("Parameters ready!")

    def get_transformed_slices(self, layer=0, image=0, start_slice=None, end_slice=None, mask=False, mask_radius=None):
        """
        Function performing a transformation of images (shift only) based on the SimpleITK workflows. A linear interpolation is used. Modify the interpolant if needed in the function directly.
        The start slice and end slice are mandatory otherwise the function is not excecuted.

        Parameters
        ----------
        layer : INT, optional
            The layer where the slices of the image are to be extracted and transformed. The default is 0.
        image : INT, optional
            The image on the layer whose slices are to be extracted and transformed. The default is 0.
        start_slice : INT, optional
            The starting slice of the extraction. The default is None.
        end_slice : INT, optional
            The ending slice of the extraction. The default is None.
        mask : boolean, optional
            Apply a circular mask. The default is False.
        mask_radius : TYPE, optional
            The radius of the circular mask. If you do not want any masked region set it to a very high value. The default is None.

        Returns
        -------
        ndarray
            The transformed slices.

        """
        if ((start_slice == end_slice)) and (not isinstance(start_slice, type(None))):
            sys.exit("For single slices use start_slice + 1 as end_slice!")
        # Get the range of slices to extract
        # reference_slice = np.floor(start_slice - self.layer_final_displacements[layer][image][2]).astype(int)
        reference_slice = start_slice - np.round(self.layer_final_displacements[layer][image][2]).astype(int)
        # Compute the slices required for extraction
        lower_boundary = reference_slice - 2
        upper_boundary = reference_slice + (end_slice - start_slice) + 2
        # Make sure first that the slices remain within the volume:
        if lower_boundary < 0:
            lower_boundary = 0
        if lower_boundary > self.img_depth:
            lower_boundary = self.img_depth
        if upper_boundary < 0:
            upper_boundary = 0
        if upper_boundary > self.img_depth:
            upper_boundary = self.img_depth
        # Extract the slices only if at least one slice is contained
        if upper_boundary - 1 >= lower_boundary:
            img_slices = Utilities.H5MaxIV.get_slices(self.layers_paths[layer][image], start_slice=lower_boundary, end_slice=upper_boundary, add_value_for_mask=self.add_value_for_mask)
            # Append zeros if the array does not have the excpected shape
            if (img_slices.shape[0] != (end_slice - start_slice) + 4) and (reference_slice - 2 >= 0):
                img_slices = np.pad(img_slices, ((0, (end_slice - start_slice) + 4 - img_slices.shape[0]), (0, 0), (0, 0)))
            if (img_slices.shape[0] != (end_slice - start_slice) + 4) and (reference_slice - 2 < 0):
                img_slices = np.pad(img_slices, (((end_slice - start_slice) + 4 - img_slices.shape[0], 0), (0, 0), (0, 0)))
            # Transform the slice
            if (self.layer_final_displacements[layer][image][0] != 0) or (self.layer_final_displacements[layer][image][1] != 0) or (self.layer_final_displacements[layer][image][2] != 0):
                img_slices = Utilities.translate_itk_masked(img_slices, d_x_y_z=(-self.layer_final_displacements[layer][image][0],
                                                                                 -self.layer_final_displacements[layer][image][1],
                                                                                 -(self.layer_final_displacements[layer][image][2] - np.round(self.layer_final_displacements[layer][image][2]).astype(int))),
                                                            sitk_interpolator=self.sitk_interpolator, use_mask=self.mask_interpolator)
            # Apply a mask if requested
            if mask:
                img_slices = np.where(Utilities.circular_mask(img_slices,
                                                              radius=mask_radius,
                                                              center=(self.layer_final_displacements[layer][image][1],
                                                                      self.layer_final_displacements[layer][image][0])),
                                      img_slices,
                                      0)
            # Return the transformed slice
            return img_slices[2:-2, :, :]

        else:
            if end_slice - start_slice > 0:
                img_slices = np.zeros((end_slice - start_slice, self.img_height, self.img_width))
            else:
                img_slices = np.zeros((1, self.img_height, self.img_width))
        # Return the transformed slice
        return img_slices

    def get_transformed_slices_affine(self, layer=0, image=0, start_slice=None, end_slice=None, mask=False, mask_radius=None, chunk_size=None):
        """
        Similar to the get_transformed_slices but an affine transform is performed. This implementation uses cupy.ndimage.map_coordinates and is performed slice by slice.
        The chunk_size can allow for eventually send more than one slice in the GPU but is set to 1 to avoid saturatinf the memory.

        By default a linear interpolation is used for pixel mapping. Set higher spline order in the function itself if needed.

        Check the script used for the transfomation for more adjustement and understanding.

        """
        # t_0 = timing()
        # Set the chunk size
        chunk_size = self.GPU_chunk_size
        # Initialize the affine transform class
        transform = tr(Utilities.H5MaxIV.reader(self.layers_paths[layer][image]), chunk_size=chunk_size, add_value_for_mask=self.add_value_for_mask)
        # Set the chunk positions
        transform.get_chunks_position(start_slice, end_slice)
        # Get the affine transform
        affine_transform_x_y_z_1 = self.layer_final_operators[layer][image]
        # Sign the transform
        transform.set_affine_transform_operator(affine_transform_x_y_z_1)
        # Sign the interpolation order
        transform.set_spline_order(self.affine_interpolator_order)
        # Set the transform center
        # transform.set_affine_transform_center(self.layer_final_centers_xyz[layer][image].tolist()) --> this would not work for the images on the left of the reference
        top_left_xyz = self.layers_coordinates[layer][image] - self.layers_outer_most_left_xyz[layer][image]
        transform.set_affine_transform_center(top_left_xyz.tolist())
        # Collect transformed slices
        img_slices = []
        for i, chunk_position in enumerate(transform.chunk_positions):
            if chunk_size == 1:
                img_slices.append(transform.transform_chunk(i, use_mask=self.mask_interpolator))
            else:
                stack_of_slices = transform.transform_chunk(i, use_mask=self.mask_interpolator)
                for j in stack_of_slices:
                    img_slices.append(j)
        # Convert the collected slices to a numpy array if more than one slice
        if len(img_slices) != 1:
            img_slices = np.squeeze(img_slices)
        else:
            img_slices = img_slices[0]
        # Check for any mask
        if mask:
            img_slices = np.where(Utilities.circular_mask(img_slices,
                                                          radius=mask_radius,
                                                          center=(self.layer_final_displacements[layer][image][1],
                                                                  self.layer_final_displacements[layer][image][0])),
                                  img_slices,
                                  0)
        # Return the transformed/masked slices
        # print(f"It took {np.round(timing() - t_0, 1)} seconds for {chunk_size} slices.")
        return img_slices

    def stitch_volumes_blend_equalize_parallel(self,
                                               stitch_layer=None,
                                               start_slice=None,
                                               end_slice=None,
                                               use_existing_equalize=None,
                                               exclude_indexes=[],
                                               chunk_size=50,
                                               ncores=5):
        """
        Excecutes in parallel processes the stitching of the layers.

        Parameters
        ----------
        chunk_size: INT, optional
            The size of the chunk per process in terms of number of slices. The default is 50.
        ncores
            The number of parallel processes each treating a chunk. The default is 5.

        Look at the stitch_volumes_blend_equalize for more info on the other parameters.

        Attention
        ---------
        Be careful when using this function with the affine transform to not block the GPU.

        """
        if isinstance(stitch_layer, type(None)):
            sys.exit("You must specify the stiching layer!")
        if (isinstance(self.layer_equalize_slope_intercept, type(None))) or (isinstance(use_existing_equalize, type(None))):
            use_existing_equalize = False
        # Get the start and end slices for every chunk
        start_end_slices = np.append(np.arange(start_slice, end_slice, chunk_size), end_slice)
        # Create working chunks
        # (start_slice, end_slice, mask, mask_radius, alpha, use_equalize, square_dist, crop_x, crop_y)
        tasks = [((stitch_layer, start_end_slices[i], start_end_slices[i + 1], self.params_final["mask"], self.params_final["mask_radius"], self.params_final["alpha"], False, use_existing_equalize, self.params_final["normalize_dist_radially"], self.params_final["square_dist"], self.params_final["crop_x"], self.params_final["crop_y"], exclude_indexes, self.params_final["exclude_NCC"], False)) for i in range(len(start_end_slices) - 1)]
        # Create a Pool of workers
        """
        # The following was replaced by the ThreadPool as the multiprocessing is not compatible with the GPU (in some cases at least!)
        with multiprocessing.Pool(processes=ncores) as pool:
            results = pool.starmap(self.stitch_volumes_blend_equalize, tasks)
        """
        with ThreadPool(processes=ncores) as pool:
            results = pool.starmap(self.stitch_volumes_blend_equalize, tasks)

        # Make sure to clear the gpu memory if affine transform is used
        if self.affine_warp:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        return results

    def stitch_layers(self, chunk_size_series=200, chunk_size_parallel=10, n_cores=10, use_existing_equalize=None, path_save=None, check=False):
        """
        This method will stitch the layers individually and save them as h5 files.

        Parameters
        ----------
        chunk_size_series : INT, optional
            THE NUMBER OF SLICES TO BE SENT FOR PARALLEL STITCHING. THIS AMOUNT OF SLICES WILL BE HELD IN MEMORY. The default is 200.
        chunk_size_parallel : INT, optional
            THE CHUNK SIZE FOR PARALLEL STITCHING OF THE SLICES FROM EACH CHUNK_SIZE_SERIES. The default is 10.
        n_cores: INT, optional
            THE NUMBER OF THREAD CORES TO BE USED FOR PARALLEL STITCHING. The default is 10.
        use_equalize: boolean, optional
            SET TO TRUE TO USE ANY EQUALIZATION FROM THE PREVIOUS STITCHING SETTINGS. The default is None.
        path_save : str, optional
            The saving path. The default is None.
        check : boolean, optional
            Plots the last stitched slice from every block. The default is False.
        Returns
        -------
        None.

        """
        if (isinstance(path_save, type(None))) and (isinstance(self.saving_path, type(None))):
            layers_saving_path = os.path.join(os.getcwd(), "Stitched_layers")
        else:
            if (not isinstance(path_save, type(None))):
                layers_saving_path = os.path.join(path_save, "Stitched_layers")
            else:
                layers_saving_path = os.path.join(self.saving_path, "Stitched_layers")

        # Create the directory and notify the user
        os.makedirs(layers_saving_path, exist_ok=True)

        # Run the parallel stitching for every chunk_size_series
        for layer_inx, layer_paths in enumerate(self.layers_paths):
            # Get the start and end slices for every chunk

            start_end_slices = np.append(np.arange(0, self.img_depth, chunk_size_series), self.img_depth)

            # Open the HDF5 file for writing
            with h5py.File(layers_saving_path + f"/Layer_{layer_inx}.h5", "w") as h5f:

                # Initialize tqdm inside the loop, so it refreshes per `stitch_layer`
                progress_bar = tqdm(total=len(start_end_slices) - 1, desc=f"Stitching layer {layer_inx}", ncols=100)

                # Create a dataset to store the image slices
                # The initial shape is (0, stitcher.img_height, stitcher.img_width) since we will resize it dynamically.

                h5_file = h5f.create_group('stitched_data', track_order=True)

                """
                # Replaced by a chunked but flexible array
                img_dataset = h5_file.create_dataset(
                    "stitched_image",
                    shape=(self.img_depth,
                           self.padding_neg_y[layer_inx][0] + self.padding_pos_y[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][1] + self.layers_outer_most_left_xyz[0][0][1]) - np.sum(self.params_final["crop_y"]),
                           self.padding_neg_x[layer_inx][0] + self.padding_pos_x[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][0] + self.layers_outer_most_left_xyz[0][0][0]) - np.sum(self.params_final["crop_x"])),
                    dtype='float32'
                    )
                """

                img_dataset = h5_file.create_dataset(
                    "stitched_image",
                    shape=(self.img_depth,
                           self.padding_neg_y[layer_inx][0] + self.padding_pos_y[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][1] + self.layers_outer_most_left_xyz[0][0][1]) - np.sum(self.params_final["crop_y"]),
                           self.padding_neg_x[layer_inx][0] + self.padding_pos_x[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][0] + self.layers_outer_most_left_xyz[0][0][0]) - np.sum(self.params_final["crop_x"])),
                    maxshape=(None,
                              self.padding_neg_y[layer_inx][0] + self.padding_pos_y[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][1] + self.layers_outer_most_left_xyz[0][0][1]) - np.sum(self.params_final["crop_y"]),
                              self.padding_neg_x[layer_inx][0] + self.padding_pos_x[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][0] + self.layers_outer_most_left_xyz[0][0][0]) - np.sum(self.params_final["crop_x"])),
                    chunks=(1,
                            self.padding_neg_y[layer_inx][0] + self.padding_pos_y[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][1] + self.layers_outer_most_left_xyz[0][0][1]) - np.sum(self.params_final["crop_y"]),
                            self.padding_neg_x[layer_inx][0] + self.padding_pos_x[layer_inx][0] + (self.layers_outer_most_right_xyz[0][0][0] + self.layers_outer_most_left_xyz[0][0][0]) - np.sum(self.params_final["crop_x"])),
                    dtype='float32'
                    )

                # Gather the information about the layer into a dictionary
                layer_info_dict = {"layer_coordinates": np.squeeze(self.layers_coordinates[layer_inx]),
                                   "layer_paths": self.layers_paths[layer_inx],
                                   "intersections_pyramid_sub_layer": [str(layer__) for layer__ in self.intersections_pyramid_sub_layer[layer_inx]],
                                   "displacements_pyramid_sub_layer": [str(layer__) for layer__ in self.displacements_pyramid_sub_layer[layer_inx]],
                                   "displacements_ncc_pyramid_sub_layer": [str(layer__) for layer__ in self.displacements_ncc_pyramid_sub_layer[layer_inx]],
                                   "layer_final_displacements": self.layer_final_displacements[layer_inx],
                                   "layer_final_global_coordinates": self.layer_final_global_coordinates[layer_inx],
                                   "layer_equalize_slope_intercept": "None" if isinstance(self.layer_equalize_slope_intercept, type(None)) else self.layer_equalize_slope_intercept[layer_inx]}

                # Distribute the layer info into the h5 file
                for key in layer_info_dict.keys():
                    h5_file.create_dataset(key, data=layer_info_dict[key])
                # Distribute the final parameters
                for key in self.params_final.keys():
                    value = self.params_final[key]
                    if isinstance(value, type(None)):
                        h5_file.create_dataset(key, data="None")
                    elif isinstance(value, (tuple, list)):
                        h5_file.create_dataset(key, data=np.asarray(value))
                    else:
                        h5_file.create_dataset(key, data=value)

                # An index to keep track of the appended slices on the h5 file
                j = 0

                # Iterate through each slice from start_slice to end_slice
                for slice_idx in range(len(start_end_slices) - 1):

                    stitch_block = self.stitch_volumes_blend_equalize_parallel(stitch_layer=layer_inx,
                                                                               start_slice=start_end_slices[slice_idx],
                                                                               end_slice=start_end_slices[slice_idx + 1],
                                                                               use_existing_equalize=use_existing_equalize,
                                                                               exclude_indexes=[],
                                                                               chunk_size=chunk_size_parallel,
                                                                               ncores=n_cores)

                    if (slice_idx == 0) and (check):
                        plt.ion
                        # Optionally, visualize the current slice (if you want to monitor progress)
                        plt.figure(figsize=(20, 20))
                        plt.imshow(stitch_block[0][-1])
                        plt.show(block=False)

                    for slices_img in stitch_block:
                        for slice_img in slices_img:
                            # Resize the dataset to append the current slice
                            img_dataset[j] = slice_img  # Assign the slice to the last index
                            j += 1

                    progress_bar.update(1)  # Update the progress bar on every iteration

        print("Stitched layers will be saved at:", layers_saving_path)


class Utilities:
    class H5MaxIV:

        @staticmethod
        def reader(file_path):
            """
            Contains some of the H5 formats, specifically ForMax, DanMax or the ones from the stitcher.

            Parameters
            ----------
            file_path : TYPE
                DESCRIPTION.

            Returns
            -------
            TYPE
                DESCRIPTION.

            """
            try:
                """Returns a reference (pointer) to the h5 dataset without loading it into memory."""
                f = h5py.File(file_path, 'r')  # Open file (user must close it)
                return f['exchange']['data']  # Returns a reference to the dataset
            except BaseException:
                try:
                    """General reader (My way of naming files)."""
                    f = h5py.File(file_path, 'r')  # Open file (user must close it)
                    return f['image']  # Returns a reference to the dataset
                except BaseException:
                    try:
                        """General reader for stitched images (My way of naming files)."""
                        f = h5py.File(file_path, 'r')  # Open file (user must close it)
                        return f['/stitched_data/stitched_image/']  # Returns a reference to the dataset
                    except BaseException:
                        try:
                            """Returns a reference (pointer) to the h5 dataset without loading it into memory."""
                            f = h5py.File(file_path, 'r')  # Open file (user must close it)
                            return f['entry']['instrument']['zyla']['data']  # Returns a reference to the dataset
                        except BaseException:
                            sys.exit("Unrecognised H5 layout. See `docs/troubleshooting.md` for the list of supported paths.")

        @staticmethod
        def get_shape(file_path):
            """Returns the shape of the h5 data."""
            return Utilities.H5MaxIV.reader(file_path).shape

        @staticmethod
        def get_bit_depth(file_path):
            """Returns the byte depth of the h5 data as a numpy dtype."""
            return Utilities.H5MaxIV.reader(file_path).dtype

        @staticmethod
        def get_slices(file_path, start_slice=None, end_slice=None, add_value_for_mask=0):
            """
            Reads slices from an h5 file given a start and end slice.
            If start and end slices are identical, a single slice is returned.

            Parameters:
            - file_path (str): Path of the h5 file.
            - start_slice (int, optional): Starting slice. Default is None (first slice).
            - end_slice (int, optional): Ending slice. Default is None (last slice).

            Returns:
            - A single slice or a sequence of slices.
            """
            # Read the file
            h5_data = Utilities.H5MaxIV.reader(file_path)

            # Set default slice values
            if start_slice is None:
                start_slice = 0
            if end_slice is None:
                end_slice = h5_data.shape[0]
            # Return the requested slices
            if add_value_for_mask == 0:
                return h5_data[start_slice] if start_slice == end_slice else h5_data[start_slice:end_slice]
            else:
                if start_slice == end_slice:
                    return np.add(h5_data[start_slice], add_value_for_mask, dtype=h5_data.dtype)
                else:
                    return np.add(h5_data[start_slice:end_slice], add_value_for_mask, dtype=h5_data.dtype)

    class GeneralTiff:

        @staticmethod
        def reader(file_path):
            """
            Internal function that reads slices of tiff files provided a filepath.
            file_path: str
                the path of the .tiff file
            start_slice: int
                The starting slice. The default is None.
            end_slice: int
                The end slice. The default is None.
            """
            return tif.TiffFile(file_path).series[0]

        """
        with tif.TiffFile(file_path) as volume_i_1:
            #subvolume_i_1 = volume_i_1.series[0].asarray()[z_i_1_min:z_i_1_max, y_i_1_min:y_i_1_max, x_i_1_min:x_i_1_max]
            series_i_1 = volume_i_1.series[0]

            # Access the specified slices as a list of arrays
            subvolume_i_1 = [series_i_1.pages[k].asarray() for k in range(start_slice, end_slice)]

            # Stack slices into a 3D numpy array
            subvolume_i_1 = np.stack(subvolume_i_1, axis=0)
        return subvolume_i_1
        """

        @staticmethod
        def get_shape(file_path):
            """Returns the shape of the h5 data."""
            return Utilities.General.reader(file_path).shape

        @staticmethod
        def save(file_path, data):
            # Ensure the directory exists before saving the file
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesn't exist

            # Proceed with saving the data
            tif.imwrite(file_path, data)

        @staticmethod
        def get_bit_depth(file_path):
            """Returns the byte depth of the h5 data as a numpy dtype."""
            return Utilities.General.reader(file_path).dtype

        @staticmethod
        def get_slices(file_path, start_slice=None, end_slice=None):
            """
            Reads slices from an tiff file given a start and end slice.
            If start and end slices are identical, a single slice is returned.

            Parameters:
            - file_path (str): Path of the tiff file.
            - start_slice (int, optional): Starting slice. Default is None (first slice).
            - end_slice (int, optional): Ending slice. Default is None (last slice).

            Returns:
            - A single slice or a sequence of slices.
            """
            # Read the file
            tiff_data = Utilities.General.reader(file_path)

            # Set default slice values
            if start_slice is None:
                start_slice = 0
            if end_slice is None:
                end_slice = tiff_data.shape[0]

            # Return the requested slices
            return tiff_data.pages[start_slice].asarray() if start_slice == end_slice else np.stack([tiff_data.pages[k].asarray() for k in range(start_slice, end_slice)], axis=0)

    @staticmethod
    def mean_filter_masked(img_np, element_shape=(1, 1, 1)):
        # Convert the image
        img_cp = cp.asarray(img_np, dtype=cp.float32)
        # Get the filtering element
        filter_2d = cp.ones(element_shape, dtype=cp.float32)
        # get the mask
        mask = (img_cp != 0).astype(cp.float32)
        # get the counts
        the_count = convolve(mask, filter_2d)
        # get the sum
        the_sum = convolve(img_cp, filter_2d)
        # get the average
        the_avg = cp.where(mask, the_sum / the_count, 0)
        # return the normalized image
        return the_avg.get()

    @staticmethod
    def circular_mask(img, radius=None, center=(0, 0)):
        """
        Get a circular mask to remove the outliers of the reconstruction on extracted subvolumes. Returns an array with 1s and 0s. The default radius
        is computed based on the minimum dimension in x or y. Define another value if necessary. Default is None.
        The center is used to shift the center of the image with (n_rows, n_columns).
        """
        if len(img.shape) == 3:
            im_dimension = np.minimum(img.shape[1], img.shape[2])  # The radius is defined on the minimum dimension if not specified
            # Use mgrid to generate a grid of coordinates in a 2D slice
            y, x = np.mgrid[0:img.shape[1], 0:img.shape[2]].astype(np.float32)
            # Compute the radius
            img_radius = np.sqrt((x - (img.shape[2] - 1) / 2 - center[1])**2 + (y - (img.shape[1] - 1) / 2 - center[0])**2)
        else:
            im_dimension = np.minimum(img.shape[0], img.shape[1])  # The radius is defined on the minimum dimension if not specified
            # Use mgrid to generate a grid of coordinates in a 2D slice
            y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
            # Compute the radius
            img_radius = np.sqrt((x - (img.shape[1] - 1) / 2 - center[1])**2 + (y - (img.shape[0] - 1) / 2 - center[0])**2)

        # Turn the img_radius into an array containing 1 and 0 for masking purposes
        if radius is None:
            circular_mask = (img_radius <= im_dimension / 2)
        else:
            circular_mask = (img_radius <= radius)

        # Add a new axis to pad values along the z axis of the image if we are dealing with a 3D image
        if len(img.shape) == 3:
            circular_mask = circular_mask[np.newaxis, ...]
            # Extend the z axis to comply with the shape of the input image
            return np.pad(circular_mask, ((0, img.shape[0] - 1), (0, 0), (0, 0)), mode="edge")
        else:
            return circular_mask

    @staticmethod
    def convert(img, maxValue=None, minImg=None, maxImg=None, data_type=np.uint16):
        """
        Converts an image (img) of a given type to another specified by the data_type input.
        The maxValue is the maxValue to be signed to the image after conversion. This is used because later one or more values could be added for masking purposes.
        minImg and maxImg are the minimum and maximum values that will be converted. Every value falling lower or higher are converted to the minImg and maxImg.
        """
        if isinstance(minImg, type(None)):
            minImg = img.min()
        if isinstance(maxImg, type(None)):
            maxImg = img.max()
        if isinstance(maxValue, type(None)):
            maxValue = 65535

        # Step 1: Set any values lower than minImg to 0 and higher than maxImg to maxValue
        img = np.clip(img, a_min=minImg, a_max=maxImg)

        # Step 2: Normalize values within [minImg, maxImg] to the range [0, 1]
        img = (img - minImg) / (maxImg - minImg)

        # Step 3: Scale normalized values to [0, maxValue]
        img = (img * maxValue).astype(data_type)

        return img

    @staticmethod
    def plot_slice(image, title="Slice"):
        # Plot the image
        fig, ax = plt.subplots()
        cax = ax.imshow(image, cmap='gray')  # Use colormap for grayscale images

        # Add a colorbar
        colorbar = plt.colorbar(cax, ax=ax)
        colorbar.set_label('Intensity')

        # Add a title
        ax.set_title(title)

        # Show the plot
        plt.show(block=False)

    @staticmethod
    def translate_itk(img, d_x_y_z=(0, 0, 0), sitk_interpolator=sitk.sitkLinear):
        """
        Applies a 3D translation to a NumPy array using SimpleITK.
        The applied translation will be the inverse of d_x_y_z due to the inverse mapping technique.

        - Faster than `scipy.ndimage.shift`
        - Slightly slower than `cupy.ndimage.shift` (including numpy-ITK conversions)

        Parameters:
            img (numpy.ndarray): Input 3D image.
            d_x_y_z (tuple): Translation offsets (dx, dy, dz).

        Returns:
            numpy.ndarray: Translated image.
        """

        # Convert the NumPy array to a SimpleITK image
        sitk_image = sitk.GetImageFromArray(img)

        # Preserve spacing if available (avoid hardcoded 1.0 values)
        spacing = sitk_image.GetSpacing() if sitk_image.HasMetaDataKey("spacing") else (1.0, 1.0, 1.0)
        sitk_image.SetSpacing(spacing)

        # Define the translation transform
        translation = sitk.TranslationTransform(3)  # 3D translation
        translation.SetOffset((d_x_y_z[0], d_x_y_z[1], d_x_y_z[2]))  # Negate for correct direction

        # Apply resampling (simpler, avoids creating an extra filter)
        translated_sitk_image = sitk.Resample(
            sitk_image, sitk_image, translation, sitk_interpolator, 0.0, sitk_image.GetPixelID()
        )
        # Convert back to NumPy array
        return sitk.GetArrayFromImage(translated_sitk_image)

    @staticmethod
    def translate_itk_masked(img, d_x_y_z=(0, 0, 0), sitk_interpolator=sitk.sitkLinear, use_mask=True):
        """
        Applies a 3D translation to a NumPy array using SimpleITK,
        avoiding interpolation between background (0) and foreground.

        The applied translation will be the inverse of d_x_y_z due to
        inverse mapping used by SimpleITK.

        Parameters:
            img (numpy.ndarray): Input 3D image.
            d_x_y_z (tuple): Translation offsets (dx, dy, dz).
            sitk_interpolator: SimpleITK interpolator (e.g. sitkLinear).

        Returns:
            numpy.ndarray: Translated image.
        """

        # Convert NumPy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(img)

        # Define translation transform
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(d_x_y_z)

        # Create foreground mask (1 where data exists, 0 where background)
        mask = sitk_image != 0

        # Resample image (allows smooth interpolation inside foreground)
        resampled_img = sitk.Resample(
            sitk_image, sitk_image, translation, sitk_interpolator, 0.0, sitk_image.GetPixelID(),
        )
        if use_mask:
            # Resample mask with SAME interpolator
            resampled_mask = sitk.Resample(
                mask, mask, translation, sitk_interpolator, 0.0, sitk.sitkFloat32,
            )

            # Enforce binary mask: anything < 0.98 → 0 (other values were tested and restricting to above 0.999 yields some missing points on the edges for higher interpolants)
            resampled_mask = sitk.BinaryThreshold(
                resampled_mask, lowerThreshold=0.98, upperThreshold=1e9, insideValue=1, outsideValue=0,
            )

            # Apply mask
            resampled_img = sitk.Mask(resampled_img, resampled_mask)
        # Convert back to NumPy array
        return sitk.GetArrayFromImage(resampled_img)

    @staticmethod
    def dist_function(img, center=(0, 0)):
        """
        Get the exact distance function from the center of the image.
        The center parameter is used to shift the center of the image with (n_rows, n_columns).
        """
        if len(img.shape) == 3:
            # Use mgrid to generate a grid of coordinates in a 2D slice
            y, x = np.mgrid[0:img.shape[1], 0:img.shape[2]].astype(np.float32)
            # Compute the radius
            img_radius = np.sqrt((x - (img.shape[2] - 1) / 2 - center[1])**2 + (y - (img.shape[1] - 1) / 2 - center[0])**2).astype(np.float32)
        else:
            # Use mgrid to generate a grid of coordinates in a 2D slice
            y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
            # Compute the radius
            img_radius = np.sqrt((x - (img.shape[1] - 1) / 2 - center[1])**2 + (y - (img.shape[0] - 1) / 2 - center[0])**2).astype(np.float32)

        # Add a new axis to pad values along the z axis of the image if we are dealing with a 3D image
        if len(img.shape) == 3:
            img_radius = img_radius[np.newaxis, ...]
            # Extend the z axis to comply with the shape of the input image
            return np.pad(img_radius, ((0, img.shape[0] - 1), (0, 0), (0, 0)), mode="edge")
        else:
            return img_radius

    @staticmethod
    def dist_function_sq(img, center=(0, 0), prop_x_y=(0, 0)):
        """
        Get the exact distance function from the center of the image.
        The center parameter is used to shift the center of the image with (n_rows, n_columns).
        prop_x_y is used to set only one prefered direction:
            (0,1) for a propagation along y only
            (1,0) for a propagation along x only
        """
        if len(img.shape) == 3:
            # Use mgrid to generate a grid of coordinates in a 2D slice
            y, x = np.mgrid[0:img.shape[1], 0:img.shape[2]].astype(np.float32)
            # Compute the radius with the requested propagation mode
            if prop_x_y == (0, 1):
                img_radius = np.abs(y - (img.shape[1] - 1) / 2 - center[0]).astype(np.float32)
            if prop_x_y == (1, 0):
                img_radius = np.abs(x - (img.shape[2] - 1) / 2 - center[1]).astype(np.float32)
            if prop_x_y == (0, 0):
                img_radius = np.maximum(np.abs(x - (img.shape[2] - 1) / 2 - center[1]), np.abs(y - (img.shape[1] - 1) / 2 - center[0])).astype(np.float32)
        else:
            # Use mgrid to generate a grid of coordinates in a 2D slice
            y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
            # Compute the radius with the requested propagation mode
            if prop_x_y == (0, 1):
                img_radius = np.abs(y - (img.shape[0] - 1) / 2 - center[0]).astype(np.float32)
            if prop_x_y == (1, 0):
                img_radius = np.abs(x - (img.shape[1] - 1) / 2 - center[1]).astype(np.float32)
            if prop_x_y == (0, 0):
                img_radius = np.maximum(np.abs(x - (img.shape[1] - 1) / 2 - center[1]), np.abs(y - (img.shape[0] - 1) / 2 - center[0])).astype(np.float32)
        # Add a new axis to pad values along the z axis of the image if we are dealing with a 3D image
        if len(img.shape) == 3:
            img_radius = img_radius[np.newaxis, ...]
            # Extend the z axis to comply with the shape of the input image
            return np.pad(img_radius, ((0, img.shape[0] - 1), (0, 0), (0, 0)), mode="edge")
        else:
            return img_radius

    @staticmethod
    def normalize_distance_map_radially(distance_map, center=None):
        """
        Given the exact distance function from the center of the image this function normalizes the distance radially from min to max value.
        The center parameter is the center of the image with (n_rows, n_columns), e.g. (0,0)
        If None, the center is automatically set to the middle point of the image
        """
        # Check the initial shape
        if len(distance_map.shape) == 3:
            depth = distance_map.shape[0]
            if depth == 1:
                distance_map = distance_map[0]
            else:
                distance_map = distance_map[depth // 2]
        else:
            depth = 0

        # Check the initial datatype and make sure to work with float 32
        if distance_map.dtype != np.float32:
            distance_map = distance_map.astype(np.float32)

        # Get the center coordinates
        if isinstance(center, type(None)):
            c_y, c_x = np.subtract(distance_map.shape, (1, 1)) / 2
        else:
            c_y, c_x = center

        # Convert it to polar coordinates
        # Polar coordinates: 0 to 360 degree and 0 to the maximum center
        polar_theta, polar_r = np.mgrid[0:361,
                                        0:int(np.max(np.divide(distance_map.shape, 2))) + 1]
        # To map coordinates from the cartesian grid
        to_map_x_polar = (polar_r * np.cos(np.deg2rad(polar_theta)) + c_x).astype(np.float32)
        to_map_y_polar = (polar_r * np.sin(np.deg2rad(polar_theta)) + c_y).astype(np.float32)
        # Stack the coordinates to map
        coord_to_map = np.vstack((to_map_y_polar.flatten(),
                                  to_map_x_polar.flatten()))
        # Map coordinates
        polar_map = sci.map_coordinates(distance_map,
                                        coord_to_map,
                                        order=0).reshape(polar_theta.shape)
        # Get max/min radius and their difference
        max_radius = np.pad(np.max(polar_map, axis=1)[..., np.newaxis],
                            ((0, 0),
                             (0, polar_r.shape[1] - 1)),
                            mode="edge")
        min_radius = np.pad(np.min(polar_map, axis=1,
                                   where=polar_map != 0,
                                   initial=np.inf)[..., np.newaxis],
                            ((0, 0),
                             (0, polar_r.shape[1] - 1)),
                            mode="edge")
        max_min_difference = np.subtract(max_radius, min_radius)

        # Get the shape function in polar coordinates
        shape_function = np.divide(max_radius - polar_r, max_min_difference, where=max_min_difference != 0, dtype=np.float32)
        # shape_function = np.where((shape_function >= 0) & (shape_function <= 1), shape_function, 0)

        # Get the shape function in cartesian coordinates
        # Get the coordinates for the angles and radius
        y, x = np.mgrid[0:distance_map.shape[0], 0:distance_map.shape[1]]

        # Get the radius and angles to be found on the polar grid
        to_map_radius_cartesian = np.sqrt((x - c_x)**2 + (y - c_y)**2, dtype=np.float32)
        to_map_theta_cartesian = np.rad2deg((np.unwrap(np.arctan2(y - c_y, x - c_x)) + 2 * np.pi) % (2 * np.pi), dtype=np.float32)
        # Map back the coordinates from the polar form
        coord_to_map_cart = np.vstack((to_map_theta_cartesian.flatten(),
                                       to_map_radius_cartesian.flatten()))
        # Map coordinates
        cartesian_map = sci.map_coordinates(shape_function,
                                            coord_to_map_cart,
                                            order=0).reshape(to_map_radius_cartesian.shape)
        # Flip the order from 0 to 1
        cartesian_map = np.abs(np.subtract(cartesian_map, 1))

        # Retain only the region within the initial distance field
        cartesian_map = np.where(distance_map != 0, cartesian_map, 0.0)

        # Clip values below or above 0
        cartesian_map = np.clip(cartesian_map, 0, 1)

        # Make sure to bring back the initial shape
        if depth != 0:
            if depth == 1:
                cartesian_map = cartesian_map[np.newaxis, ...]
            else:
                cartesian_map = np.pad(cartesian_map[np.newaxis, ...], ((0, depth - 1), (0, 0), (0, 0)), mode='edge')
        # return the map
        return cartesian_map

    @staticmethod
    def normalize_with_masked_gaussian_filter_cupy_2D(image_np,
                                                      sigma_np_xy,
                                                      divide_by_norm=False,
                                                      use_mask=False,
                                                      notification_frequency=50,
                                                      value_return_mask=None):
        # check the input
        assert (isinstance(image_np, np.ndarray))
        assert (isinstance(sigma_np_xy, np.ndarray))
        assert (sigma_np_xy.shape == (2,))
        # Create an array to keep the normalized image
        normalized_image = np.zeros_like(image_np, dtype=np.float32)
        # Run through every slice and apply the gaussian filter
        sigma_cp_yx = cp.asarray(sigma_np_xy[::-1], dtype=cp.float32)
        for i, img_slice in enumerate(image_np):
            if i % notification_frequency == 0:
                print(f"At index {i} of {image_np.shape[0]}")
            # transfer the array to GPU and filter it
            img_slice_cp = cp.asarray(img_slice, dtype=cp.float32)
            img_slice_cp_filtered = gaussian_filter(img_slice_cp, sigma_cp_yx)
            # get the mask if needed
            if use_mask:
                # zero-normalize around the mean
                img_slice_cp_mask = ~cp.isclose(img_slice_cp, 0.0)
                img_slice_cp_mask_filtered = gaussian_filter(img_slice_cp_mask.astype(cp.float32), sigma_cp_yx)
                img_slice_cp_filtered = cp.where(img_slice_cp_mask, img_slice_cp - img_slice_cp_filtered / img_slice_cp_mask_filtered, value_return_mask)
                # divide by the norm
                if divide_by_norm:
                    img_slice_cp_filtered_norm = cp.sqrt(gaussian_filter(img_slice_cp_filtered**2, sigma_cp_yx) / img_slice_cp_mask_filtered)
                    img_slice_cp_filtered = cp.where(img_slice_cp_mask, img_slice_cp_filtered / img_slice_cp_filtered_norm, value_return_mask)
            normalized_image[i] = img_slice_cp_filtered.get()
        return normalized_image

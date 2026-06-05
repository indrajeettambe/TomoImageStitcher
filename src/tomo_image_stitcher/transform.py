# -*- coding: utf-8 -*-
"""
GPU-accelerated, chunk-by-chunk affine transform of large 3D volumes.

The :class:`affine_transform_large_data` class reads a 3D volume lazily from
an ``h5py`` dataset (so the whole volume does not need to fit in CPU memory)
and applies a 4x4 affine transform one chunk at a time using CuPy.
"""
import tifffile as tif
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates
import os
import sys
from time import time
import csv
import h5py

class affine_transform_large_data:
    """
    This class transforms large volumes of data by chunking it out in smaller pieces.
    
    The main idea:
    # Iterate over the chunks of an array having the same shape as img_np
        # Get the global coordinates on the current chunk using cp.mgrid
        # Compute by inverse mapping the coordinates whose grey-levels are to be extracted on the original image (self.img_np)
            # Account for any scaling factor or a different center of affine transform
        # Extract the image from where the interpolated values will be computed
            # Extend the image borders (on z only) depending on the order of interpolation to be used
            # Map the coordinates computed previously
                # Account for a change in coordinates based on the local coordinate system of the extracted image
                # Make sure that out of boundaries values do not represent a risk
                
    """
    def __init__(self, img_h5_pointer, chunk_size=1, add_value_for_mask=False):
        """
        The image to be transformed as a numpy array.

        Parameters
        ----------
        img_h5_pointer : 3d ndarray
            A h5 pointer to the image to be transformed. The data will be read from there.
        chunk_size : INT, optional
            The number of slices that will constitute a chunk. The default is 1.
        add_one_for_mask : boolean, optional
            This adds a value of 1 to the grey levels to liberate the 0 for masking purposes later. This is required for the stitcher functionalities. The default is 0.
        
        Class attributes
        ----------
        self.chunk_positions
        
        self.affine_transform_operator_x_y_z_1 :
            The affine transform to be applied to the image, X_1 = F @ X_0, 4x4 matrix containing the rigid displacements in the last column.
        self.affine_transform_center_x_y_z : tuple of integers (c_x, c_y, c_z)
            the position of the local coordinate center on a global system from which the affine transform is to be applied
        self.affine_transform_scaling : float
            the coordinates are writen as X_0 = [x_0, y_0, z_0, affine_transform_scaling]. This accounts for the scaling of the rigid displacement if for example the affine operator originates from on a scaled version of the img_np.
        self.spline_order : int 

        """
        self.img_np = img_h5_pointer
        self.chunk_size = chunk_size
        # check the initializers
        self._check_init()
        # get the shape of the image
        self.depth, self.height, self.width = img_h5_pointer.shape
        # Initialize an array to get the positions of the chunks
        self.chunk_positions = None
        # initialize the parameters related to the transformation
        self.affine_transform_operator_x_y_z_1 = None # the affine transform to be applied to the image, X_1 = F @ X_0, 4x4 matrix containing the rigid displacements in the last column.
        self.affine_transform_center_x_y_z = None # the center from which the affine transform will be computed FX.
        self.affine_transform_scaling = None # the coordinates are writen as X_0 = [x_0, y_0, z_0, affine_transform_scaling]. This accounts for the scaling of the rigid displacement if for example the affine operator originates from on a scaled version of the img_np.
        # initialize the parameters related to the transformation
        self.spline_order = None
        # initialize the add_one_for_mask
        self.add_value_for_mask = add_value_for_mask
        
    def _check_init(self):
        """
        Set of checks to make sure that the provided data is correct.
        """
        # check if the provided chunk_size is a tuple and it has three components
        if not isinstance(self.chunk_size, int):
            sys.exit("Please provide a positive integer for the chunk size!")
        else:
            # check if the provided chunk size is greater than 
            if self.chunk_size < 1:
                sys.exit("Please provide a positive integer for the chunk size!")
            # check if the we are dealing with a numpy image
            if isinstance(self.img_np, h5py._hl.dataset.Dataset):
                # check if we are dealing with a 3D array
                if not len(self.img_np.shape) == 3:
                    sys.exit("The image must be a 3D numpy array!")
            else:
                sys.exit("The image must be a 3D numpy array!")
    
    # Compute the total number of chunks and their bounding boxes (z_0 and z_1)
    def get_chunks_position(self, start_slice, end_slice):
        """
        This function gets the start and end position in z for every chunk so that the chunks are later easily extracted from the image of interest.
        """
        # initialize a list that will contain the start and end height of the chunks
        chunk_positions = np.append(np.arange(start_slice, end_slice, self.chunk_size), end_slice)
        # form lists of lists for every chunk position
        chunk_positions_list = [(chunk_positions[i], chunk_positions[i+1]) for i in range(chunk_positions.shape[0] - 1)]
        # return positions of the chunks
        self.chunk_positions = chunk_positions_list
    
    def transform_chunk(self, chunk_index=0, use_mask=False):
        # Check if the chunks have been defined
        if type(self.chunk_positions) == type(None): sys.exit("Please run the get_chunks_position() first to define the chunks!")
        # Check for any user defined operators
        self._check_operators()
        # Check if the index is in the list
        if chunk_index >= len(self.chunk_positions):
            sys.exit("The required chunk index is out of range!")
        # Get the coordinates of the chunk (corresponding to the transformed position)
        z_1, y_1, x_1 = cp.mgrid[self.chunk_positions[chunk_index][0]:self.chunk_positions[chunk_index][1], :self.height, :self.width].astype(cp.float32)
        # Get the coordinates back to the transformation center if the center is different from (0,0,0)
        if self.affine_transform_center_x_y_z != (0,0,0):
            z_1 += self.affine_transform_center_x_y_z[2]
            y_1 += self.affine_transform_center_x_y_z[1]
            x_1 += self.affine_transform_center_x_y_z[0]
        # Stack the coordinates and append a last row accounting for the scaling factor
        x_y_z_1_coord = cp.vstack((x_1.ravel(), y_1.ravel(), z_1.ravel(), cp.ones(x_1.size) * self.affine_transform_scaling))
        # get the inverse transform of the coordinates x_1, y_1 and z_1
        x_y_z_0_coord = cp.linalg.inv(cp.asarray(self.affine_transform_operator_x_y_z_1)) @ x_y_z_1_coord
        # Return the coordinates to their original position if the center is different from (0,0,0)
        if self.affine_transform_center_x_y_z != (0,0,0):
            x_y_z_0_coord[0] -= self.affine_transform_center_x_y_z[0]
            x_y_z_0_coord[1] -= self.affine_transform_center_x_y_z[1]
            x_y_z_0_coord[2] -= self.affine_transform_center_x_y_z[2]
        # Get the extents of the x_y_z_0_coord in z
        z_0_min, z_0_max = (x_y_z_0_coord[2, :].min(), x_y_z_0_coord[2, :].max())
        # Extract the image for pixel mapping
            # Get image boundaries
        higher_bound, lower_bound = self._check_bounds(z_0_min, z_0_max)
            # Extract the slices
        # Return the requested slices
        if self.add_value_for_mask == 0:
            img_extract_cp = cp.asarray(self.img_np[lower_bound:higher_bound, :, :])
        else:
            img_extract_cp = cp.asarray(np.add(self.img_np[lower_bound:higher_bound, :, :], self.add_value_for_mask, dtype=self.img_np.dtype))
        # Map coordinates
            # redifine the z in the initial coordinates to correspond to the local coordinate system
        x_y_z_0_coord[2, :] = x_y_z_0_coord[2, :] - lower_bound
            # extract mapped coordinates if the values are found within the boundaries
        if higher_bound != lower_bound:
            interpolated_values = map_coordinates(img_extract_cp, x_y_z_0_coord[[2,1,0], :], order=self.spline_order)
            interpolated_values = interpolated_values.get()
            if use_mask:
                # Interpolate mask with SAME interpolator type to avoid anything touching zeros
                mask = (img_extract_cp != 0)
                interpolated_mask = map_coordinates(mask, x_y_z_0_coord[[2, 1, 0], :], order=self.spline_order)
                interpolated_mask = interpolated_mask.get()
                # Enforce binary mask: anything < 1 → 0
                interpolated_values = np.where(interpolated_mask, interpolated_values, 0.0)
            # reshape the mapped coordinates to their original shape
            interpolated_values = interpolated_values.reshape(x_1.shape)
        else:
            interpolated_values = np.zeros(x_1.shape, dtype=self.img_np.dtype)
        return interpolated_values
    
    # user defined transformation
        # Set the affine transform operator
    def set_affine_transform_operator(self, affine_operator_x_y_z_1=np.eye(4)):
        self.affine_transform_operator_x_y_z_1 = affine_operator_x_y_z_1
        # Set the affine transform center
    def set_affine_transform_center(self, affine_transform_center_x_y_z=(0,0,0)):
        self.affine_transform_center_x_y_z = affine_transform_center_x_y_z
        # Set the affine transform scaling factor
    def scale_affine_transform(self, affine_transform_scaling=1):
        self.affine_transform_scaling = affine_transform_scaling
        # Set the spline order for mapping
    def set_spline_order(self, spline_order=1):
        # check if the spline order is correct
        spline_orders = (0,1,3,5)
        if spline_order not in spline_orders:
            sys.exit("Supported spline orders: ", spline_orders)
        # set the class attribute
        self.spline_order = spline_order
        
        # check if all operators have been asigned
    def _check_operators(self):
        if type(self.affine_transform_operator_x_y_z_1) == type(None): self.set_affine_transform_operator()
        if type(self.affine_transform_center_x_y_z) == type(None): self.set_affine_transform_center()
        if type(self.affine_transform_scaling) == type(None): self.scale_affine_transform()
        if type(self.spline_order) == type(None): self.set_spline_order()
        
    def _check_bounds(self, z_min, z_max):
        # get the initial bounds
        lower_bound = cp.floor(z_min - self.spline_order)
        higher_bound = cp.ceil(z_max + self.spline_order)
        
        # check if the lower or the higher bounds go beyond the image in +z direction
        if lower_bound > self.depth: lower_bound = self.depth
        if higher_bound > self.depth: higher_bound = self.depth
        # check if the lower or the higher bounds go beyond the image in -z direction
        if lower_bound < 0: lower_bound = 0
        if higher_bound < 0: higher_bound = 0
        
        return int(higher_bound), int(lower_bound)
    
    def _get_transform_txt(self):
        # Compose the file to be saved
        tr_matrix = [row for row in self.affine_transform_operator_x_y_z_1]
        tr_matrix.append([self.affine_transform_center_x_y_z[0],
                          self.affine_transform_center_x_y_z[1],
                          self.affine_transform_center_x_y_z[2],
                          self.affine_transform_scaling])
        # Compose the file saving format
        tr_matrix_format = [["du/dx, du/dy, du/dz,     u"],
                            ["dv/dx, dv/dy, dv/dz,     v"],
                            ["dw/dx, dw/dy, dw/dz,     w"],
                            ["    0,     0,     0,     1"],
                            ["  x_c,   y_c,   z_c, scale"]]
        return tr_matrix, tr_matrix_format
    
    def chunk_by_chunk_transform(self, path=None, file_name=None, overwrite=False):
        # Get the path
        if type(path) == type(None):
            folder_path = os.path.join(os.getcwd(), f"Transformed_slices_{self.spline_order}")
            txt_path = os.getcwd()
            print("Will be saved at:", folder_path)
        else:
            folder_path = os.path.join(path, f"Transformed_slices_{self.spline_order}")
            txt_path = path
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=overwrite)
        
        # Get the filename
        if type(file_name) == type(None):
            file_name = "transformed_slice"

        # Iterate over every slice corresponding to the transformed image coordinates
        t_0 = time()
        for i, chunk_position in enumerate(self.chunk_positions):
            tif.imwrite(folder_path + "/" + file_name + f"_{i}.tif", self.transform_chunk(i))
        print(f"It took {np.round(time() - t_0, 1)} seconds")
        
        # Compose the file to be saved
        tr_matrix, tr_matrix_format = self._get_transform_txt()
        
        # Save to CSV
        with open(txt_path + "/transform_matrix_" + file_name + ".csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(tr_matrix)

        with open(txt_path + "/transform_matrix_format_" + file_name + ".csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(tr_matrix_format)
            
        return 0

if __name__ == "__main__":
    """
    Minimal smoke test for :class:`affine_transform_large_data`.

    The test below applies the identity affine to a stack of slices and
    writes the result to a temporary directory. It is intentionally free of
    hard-coded paths so it can be run from any working directory.
    """
    import tempfile

    # Build a small synthetic volume.
    img = np.random.randint(0, 65535, size=(8, 64, 64), dtype=np.uint16)

    # Wrap the volume in an in-memory h5 dataset (the class expects an
    # ``h5py``-style object with a ``.shape`` and array indexing).
    import h5py
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    with h5py.File(tmp.name, "w") as fh:
        fh.create_dataset("image", data=img)

    with h5py.File(tmp.name, "r") as fh:
        transform = affine_transform_large_data(fh["image"], chunk_size=1)

        # Identity transform with a cubic-spline interpolation.
        identity = np.eye(4)
        transform.set_affine_transform_operator(identity)
        transform.set_spline_order(3)
        transform.set_affine_transform_center(
            (img.shape[2] / 2, img.shape[1] / 2, img.shape[0] / 2)
        )

        out_dir = tempfile.mkdtemp(prefix="tomo_image_stitcher_test_")
        transform.chunk_by_chunk_transform(path=out_dir, file_name="identity")
        print(f"Identity transform output written to: {out_dir}")

    os.unlink(tmp.name)

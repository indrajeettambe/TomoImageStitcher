"""
Registration kit — ZNCC pixel search and IC-GN Lucas-Kanade refinement.

This module provides :class:`RegistrationKIT`, a static-method-only class that
exposes the two registration engines used by the Stitcher pipeline:

* :meth:`RegistrationKIT.correlate_NCC` — a multi-stage downscaled
  zero-normalised cross-correlation pixel search on the GPU.
* :meth:`RegistrationKIT.lucas_kanade_3D_inv_mask` — an inverse-compositional
  Gauss-Newton Lucas-Kanade refinement with optional affine or rigid warp
  and mask-aware error metrics.
"""
import numpy as np
from copy import deepcopy
import sys
import tifffile as tif
import cupy as cp
from cupyx.scipy.ndimage import binary_erosion as binary_erosion_cp
from cupyx.scipy.ndimage import affine_transform as affine_transform_cp
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_cp
from cupyx.scipy.ndimage import shift as shift_transform_cp
import SimpleITK as itk
from cupyx.scipy.ndimage import correlate
from scipy.linalg import polar

class RegistrationKIT:
    @staticmethod
    def lucas_kanade_3D_inv_mask(template_img, 
                                 moving_img,
                                 derivatives="gaussian", 
                                 sigma_z_y_x=(1, 1, 1), 
                                 margins_xyz=(20, 20, 20), 
                                 max_iter = 20, 
                                 convergence_criteria=0.001, 
                                 mask=False, 
                                 erodeMask=False, 
                                 erosionElement=np.ones((1, 1, 1)),
                                 initial_guess=None,
                                 interp_order=1,
                                 regulate=False,
                                 slice_extract=None,
                                 save=False, 
                                 affine_warp=False, 
                                 affine_guess=True,
                                 rigid_warp=False,
                                 xy_reg=False):
        """
        This function does correlation between two 2 3D images using a lucas kanade algorithm and an affine or rigid displacement warp. To avoid computing the Hessian for every iteration the 
        incremental transform is applied to the template image rather than the moving one (also known as IC-GN or "Inverse Compositional Gauss-Newton")
        
        Parameters
        ----------
        template_img : ndarray of 3D shape
            The reference image.
        moving_img : ndarray of 3D shape
            The translated image.
        derivatives : string, optional
            Defines the type of derivatives, Gaussian -> "gaussian" or anything else for central differences. The default is "gaussian".
        sigma : (float, float, float), optional
            The sigma defining the gaussian window to compute image derivatives in case of "gaussian" derivatives. The default is (1, 1, 1).
        margins_xyz: (int, int, int), optional
            The margin defines the margins at the x, y, z borders respectively where the correlation will be avoided due to border effects.It is active only if the mask is set to False. The default is (20, 20, 20).
        max_iter : int, optional
            The maximal number of iterations in case of non convergence. The default is 20.
        convergence_criteria : float, optional
            The criteria for convergence set on update magnitude of the vector P, |dP| The default is 0.01.
        mask: boolean, optional
            When used it considers for correlation only valued greater or equal to 1. 0 is rezerved for masking purposes. The default is False.
        erodeMask: boolean, optional
            When used with the mask option, it erodes the mask (region where both images have values greater or equal to 1) with the erosionElement. The default is False.
        erosionElement = array of int np.array((int, int, int))
            Check erodeMask. Default np.ones((1, 1, 1)).
        initial_guess: np.array of shape 4, 4.
            When passed the template image is initially transformed by this matrix and then enters the masking and correlation stages. The affine_guess option is used to avoid an affine_transform when only a shift is being applied.
            Default is None
        interp_order: int, optional
            The spline order of the interpolation when updating the warp on the template image. The default is 1.
        regulate: boolean, optioal
            The regulate option slows down the convergence in regions of high ZNCC by multiplying every factor of the vector dP with 1-ZNCC. The default is False.
        slice_extract: int, optional
            The slice to be looked at during the correlation process which will be returned from the function. The default is None and returns the first slice (index 0).
        affine_warp: boolean
            If set to false only rigid translation registration is performed (dx, dy, dz)
        affine_guess: boolean, optional
            Check initial guess.
        save: boolean, optional
            If set to True saves the warped template at each iteration.
        xy_reg: boolean, optional
            If set to True the optimization only accounts for degrees of freedom in xy plane. z is neglected.
        Returns
        -------
        disp_x, disp_y, disp_z, affine_transform_operator, no_iterations, stack of images during registration at the middle slice, a (-1, -1) tuple just to keep the same format with previous versions.
            The computations components required to register the moving_img to the template_img in x, y and z and other helping data.
            The affine transform operator acts on the coordinates of the template image to register it to the moving one: moving_img_coord = affine_transform_operator @ template_img_coord
        """
        
        # Define the shape of the image (height, width, depth)
        img_depth, img_height, img_width = template_img.shape
        
        # Convert the input images into cupy ndarray
        template_warped = cp.asarray(template_img)
        template_img = cp.asarray(template_img)
        moving_img = cp.asarray(moving_img)
        
        # An initial displacement step greater than the convergence criteria for the loop to initiate
        norm_step = 2*convergence_criteria
        # A dummy variable to contain the previous displacement step for checking the convergence
        previous_norm = 0
        # A dummy variable counting the number of iterations in the loop below
        count = 0
        # A  list to contain the values of normalized correlation coefficients
        ncc_list = cp.array((0, ))
        # A list to contain the values of affine transform
        affine_transform_list = cp.asarray((cp.eye(4), ))
        
        # Variable to accumulate (matrix multiplication) the affine transform operators of each increment
        if type(initial_guess) != np.ndarray:
            affine_transform_operator = cp.eye(4)
        else:
            if initial_guess.shape != (4, 4):
                sys.exit("The initial guess must be a 4x4 ndarray!")
            else:
                affine_transform_operator = cp.asanyarray(np.linalg.inv(initial_guess))
                if affine_guess:
                    template_warped = affine_transform_cp(template_img, affine_transform_operator, order=interp_order)
                else:
                    template_warped = shift_transform_cp(template_img, (-affine_transform_operator[0][-1], 
                                                                        -affine_transform_operator[1][-1], 
                                                                        -affine_transform_operator[2][-1]), order=interp_order)
                #print("Initial guess passed:")
                #print(affine_transform_operator)

        # A list of slices to check registration process at each iteration on the middle of the volume
        if type(slice_extract) == int:
            #middle_slices = cp.asarray([template_warped[:, template_warped.shape[1]//2, :]])
            middle_slices = cp.asarray([template_warped[slice_extract, :, :]])
        else:
            middle_slices = cp.asarray([template_warped[0, :, :]])
        
        # A coefficient to slow down the convergence in case of a good normalized correlation coefficient
        slow_down_coeff = 1

        # The center of transform
        xyz_c = np.array((img_width  / 2, img_height / 2, img_depth  / 2))
        def _extract_rigid_transform(T_cp, xyz_0=np.zeros(3, dtype=np.float32)):
            """
            Provided an affine transform T_cp, this function returns only the rigid transform accounting for a rotation around the subset (template) center.
    
            Parameters
            ----------
            T_cp : cp.ndarray(4x4)
                THE CUPY TRANSFORM MATRIX.
    
            Returns
            -------
            TYPE
                cp.ndarray(4x4).
    
            """
            # Extract the coordinates of the centroid of the template
            xyz_0 = xyz_0.astype(np.float32)
            # Convert the transform into numpy array
            T = T_cp.get()
            # Get the polar decomposition F = R @ U
            F = T[:3, :3]
            R, U = polar(F)
            # Compensate for the center shift
            T[:-1, -1] += (F - R) @ xyz_0
            # Get the rigid transform
            T_rigid = np.pad(R, ((0, 1), (0, 1))) + np.pad(T[:, -1][..., np.newaxis], ((0, 0), (3, 0)))
            return cp.asarray(T_rigid, dtype=cp.float32)
        
        while (norm_step > convergence_criteria) and (count < max_iter):
            # t_0 = timing()
            if (mask):
                # Get the mask: it has to be computed each time as the template image will be transformed at every step
                mask_img = (template_warped != 0) & (moving_img != 0)
                # Notify the user
                if (erodeMask):
                    #print(f"Mask being eroded step {count}")
                    # Create a boolean dummy array with similar dimensions to template_img.
                    mask_to_erode = cp.zeros(template_img.shape, dtype=np.bool_)

                    # Get the mask and expand it with one pixel to unlock the borders
                    mask_to_erode[mask_img] = True
                    #mask_to_erode = cp.pad(mask_to_erode, ((1, 1), (1, 1), (1, 1)))
                    
                    # Erode the mask
                    eroded_mask = binary_erosion_cp(mask_to_erode, structure=cp.ones(erosionElement.shape), border_value=False)
    
                    # Remove the added rows and columsn
                    #eroded_mask = eroded_mask[1:-1, 1:-1, 1:-1]
                    
                    # Update the mask indices
                    mask_img = (eroded_mask == True)

                    # Check if there is any pixel left in mask_img
                    if cp.sum(mask_img) == 0:
                        break
                
            if (count == 0):
                if not mask:
                    # Create a boolean dummy array with similar dimensions to template_img.
                    mask_to_erode = cp.zeros(template_img.shape, dtype=cp.bool_)
                    # Set the mask within the given margins
                    mask_to_erode[margins_xyz[2]:-margins_xyz[2], margins_xyz[1]:-margins_xyz[1], margins_xyz[0]:-margins_xyz[0]] = True
                    mask_img = mask_to_erode
                    
                # Get the derivatives of the moving image. The d list of derivatives contains derivative_x, _y, _z at the indices 0,1,2 respectively
                d = []
                if derivatives == "gaussian":
                    #print("Getting derivatives")
                    d.append(gaussian_filter_cp(moving_img, sigma=cp.array((0,0,sigma_z_y_x[2])), order=1, mode="constant"))
                    d.append(gaussian_filter_cp(moving_img, sigma=cp.array((0,sigma_z_y_x[1],0)), order=1, mode="constant"))
                    d.append(gaussian_filter_cp(moving_img, sigma=cp.array((sigma_z_y_x[0],0,0)), order=1, mode="constant"))
                else:
                    #print("Getting derivatives")
                    d.append(cp.gradient(moving_img, axis=2))
                    d.append(cp.gradient(moving_img, axis=1))
                    d.append(cp.gradient(moving_img, axis=0))

                # Get the coordinates of the moving image
                # Use numpy.mgrid to generate Z, Y, X grids -> c_initial. The c list contains the X, Y, Z masked coordinates at 0, 1, 2 indices respectively.
                if affine_warp:
                    c = cp.flip(cp.mgrid[0:img_depth, 0:img_height, 0:img_width], axis=0)
                
                    # Get the indices for the A matrix components
                    if not xy_reg:
                        a_m_indices_str = ["M_x", "M_y", "M_z", "M_x", "M_x", "M_x", "M_y", "M_y", "M_y", "M_z", "M_z", "M_z"]
                        a_m_indices = [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]
                        a_w_indices_str = ["", "", "", "x", "y", "z", "x", "y", "z", "x", "y", "z"]
                        a_w_indices = [3, 3, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2]
                    else:
                        # Get the indices for the A matrix components
                        a_m_indices_str = ["M_x", "M_y", "M_x", "M_x", "M_y", "M_y"]
                        a_m_indices = [0, 1, 0, 0, 1, 1]
                        a_w_indices_str = ["", "", "x", "y", "x", "y", "x", "y"]
                        a_w_indices = [3, 3, 0, 1, 0, 1]   
                else:
                    if not xy_reg:
                        # Get the indices for the A matrix components
                        a_m_indices_str = ["M_x", "M_y", "M_z"]
                        a_m_indices = [0, 1, 2]
                        a_w_indices_str = ["", "", ""]
                        a_w_indices = [3, 3, 3]
                    else:
                        # Get the indices for the A matrix components
                        a_m_indices_str = ["M_x", "M_y"]
                        a_m_indices = [0, 1]
                        a_w_indices_str = ["", ""]
                        a_w_indices = [3, 3]
                
                # The component A_i of the A matrix is obtained as d[a_m_indices][i]*c[a_w_indices][i]
            if (count == 0) or (mask): # Even though we are using IC-GN algorithm since the mask gets updated at every step it is necessary to recompute the hessian
                # Compose the Hessian matrix A_i*A_j
                H = cp.ones((len(a_m_indices), len(a_m_indices)))
                for i, a_m in enumerate(a_m_indices_str):
                    for j, a in enumerate(a_m_indices_str):
                        if j >= i:
                            # print(f"Index:{i}{j}: " + a_m_indices_str[i] + " " + a_w_indices_str[i] + " " + a_m_indices_str[j] + " " + a_w_indices_str[j])
                            # d[a_m_indices[i]] * c[a_w_indices[i]] * [a_m_indices[j]] * c[a_w_indices[j]]
                            
                            # When coordinates are included in both ij and ji hessian component
                            if (a_w_indices[i] < 3) and (a_w_indices[j] < 3):
                                H_i_j = cp.sum(d[a_m_indices[i]][mask_img] * c[a_w_indices[i]][mask_img] * d[a_m_indices[j]][mask_img] * c[a_w_indices[j]][mask_img])
                            # When only second term coordinates are included in the current hessian component
                            if (a_w_indices[i] == 3) and (a_w_indices[j] < 3):
                                H_i_j = cp.sum(d[a_m_indices[i]][mask_img] * d[a_m_indices[j]][mask_img] * c[a_w_indices[j]][mask_img])
                            # When only first term coordinates are included in the current hessian component
                            if (a_w_indices[i] < 3) and (a_w_indices[j] == 3):
                                H_i_j = cp.sum(d[a_m_indices[i]][mask_img] * c[a_w_indices[i]][mask_img] * d[a_m_indices[j]][mask_img])
                            # When none of the coordinates are included in the current hessian component
                            if (a_w_indices[i] == 3) and (a_w_indices[j] == 3):
                                H_i_j = cp.sum(d[a_m_indices[i]][mask_img] * d[a_m_indices[j]][mask_img])
                                
                            # Fill in the hessian matrix with the component ij and its symmetric one if applicable (j != i)
                            H[i][j] = H_i_j
                            if j != i:
                                H[j][i] = H_i_j
            #print("Getting weighted residuals")
            # Get the weighted residuals vector from the template to the moving image --> the vector driving the convergence process which is computed at every iteration
            R = cp.ones((len(a_m_indices),1))
            for i, a_m in enumerate(a_m_indices_str):
                # When coordinates are included
                if (a_w_indices[i] < 3):
                    R_i = cp.sum(d[a_m_indices[i]][mask_img] * c[a_w_indices[i]][mask_img] * ((template_warped - moving_img)[mask_img]))
                # When coordinates are not included
                else:
                    R_i = cp.sum(d[a_m_indices[i]][mask_img] * ((template_warped - moving_img)[mask_img]))
                # Fill in the residuals array
                R[i] = R_i
    
            # Get the affine transform parameters P(u, v, w, du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz)
            # Get the affine transform parameters P(u, v, du/dx, du/dy, dv/dx, dv/dy)
            # Get the affine transform parameters P(u, v, w)
            # Get the affine transform parameters P(u, v)
            try:
                dP = -cp.linalg.inv(H)@R
            except Exception as ex:
                # In case of an exception print the error type and set the displacement values to 0
                template = "An exception of type {0} occurred. Arguments:/n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print (message)
                dP = cp.zeros((4,4))
                break
                
            # affine transformation matrix
            """
            T_affine = cp.array([[dw/dz, dw/dy, dw/dx, w],
                                 [dv/dz, dv/dy, dv/dx, v],
                                 [du/dz, du/dy, du/dx, u],
                                 [    0,     0,     0, 1]])
            
            T_affine = cp.array([[1 + dP[11][0],    dP[10][0],     dP[9][0], dP[2][0]],
                                 [     dP[8][0], 1 + dP[7][0],     dP[6][0], dP[1][0]],
                                 [     dP[5][0],     dP[4][0], 1 + dP[3][0], dP[0][0]],
                                 [            0,           0,            0,        1]])
            """
            T_affine = cp.eye(4)
            if affine_warp:
                if not xy_reg:
                    T_affine[0][0] = 1 + dP[11][0]*slow_down_coeff
                    T_affine[0][1] = dP[10][0]*slow_down_coeff
                    T_affine[0][2] = dP[9][0]*slow_down_coeff
                    T_affine[0][3] = dP[2][0]*slow_down_coeff
                    T_affine[1][0] = dP[8][0]*slow_down_coeff
                    T_affine[1][1] = 1 + dP[7][0]*slow_down_coeff
                    T_affine[1][2] = dP[6][0]*slow_down_coeff
                    T_affine[1][3] = dP[1][0]*slow_down_coeff
                    T_affine[2][0] = dP[5][0]*slow_down_coeff
                    T_affine[2][1] = dP[4][0]*slow_down_coeff
                    T_affine[2][2] = 1 + dP[3][0]*slow_down_coeff
                    T_affine[2][3] = dP[0][0]*slow_down_coeff
                else:
                    T_affine[1][1] = 1 + dP[5][0]*slow_down_coeff
                    T_affine[1][2] = dP[4][0]*slow_down_coeff
                    T_affine[1][3] = dP[1][0]*slow_down_coeff
                    T_affine[2][1] = dP[3][0]*slow_down_coeff
                    T_affine[2][2] = 1 + dP[2][0]*slow_down_coeff
                    T_affine[2][3] = dP[0][0]*slow_down_coeff
            else:
                if not xy_reg:
                    T_affine[0][-1] = dP[2][0]*slow_down_coeff
                    T_affine[1][-1] = dP[1][0]*slow_down_coeff
                    T_affine[2][-1] = dP[0][0]*slow_down_coeff
                else:
                    T_affine[1][-1] = dP[1][0]*slow_down_coeff
                    T_affine[2][-1] = dP[0][0]*slow_down_coeff
            if (rigid_warp) and (not cp.isnan(T_affine).any()):
                T_affine = _extract_rigid_transform(T_affine, xyz_c)
            # accumulating the affine transformation matrix that computes the transform from moving to template
            affine_transform_operator = T_affine@affine_transform_operator

            if affine_warp:
                # Use warpAffine to transform --> the order of spline interpolation is 3 by default
                template_warped = affine_transform_cp(template_img, affine_transform_operator, order=interp_order)
            else:
                # Use shift to transform in case of a non affine warp
                template_warped = shift_transform_cp(template_img, (-affine_transform_operator[0][-1], 
                                                                    -affine_transform_operator[1][-1], 
                                                                    -affine_transform_operator[2][-1]), order=interp_order)
                
            if save:
                tif.imwrite(f"template_img_reg_{count}.tif", template_warped.get())
                if count == 0:
                    moving_reduced = deepcopy(moving_img)
                    moving_reduced[~mask_img] = 0
                tif.imwrite(f"moving_img_reg_{count}.tif", moving_reduced.get())
            
            # Compute the current norm of the dP vector
            current_norm = cp.sqrt(cp.dot(dP[:, 0], dP[:, 0]))
            
            # Get the difference with the previous norm
            norm_step = cp.absolute(current_norm - previous_norm)
            
            # Update the norm for the next iteration
            previous_norm = current_norm.copy()
            
            # Count loop index
            
            count += 1
            # Get the normalized cross correlation between the registered images: template_warped, template_img and moving_img
            # final warp of the template
            b_0 = template_warped[mask_img] - cp.mean(template_warped[mask_img]) # The size includes the product sqrt(N)*sqrt(N) from both denumerators and is thus not added to the c_1
            b_1 = cp.std(template_warped[mask_img]) * template_warped[mask_img].size
            
            # Moving image
            c_0 = moving_img[mask_img] - cp.mean(moving_img[mask_img])
            c_1 = cp.std(moving_img[mask_img])
            
            ncc = cp.round(np.sum(b_0*c_0) / (b_1*c_1) * 100, 0)
            
            if regulate:
                if np.isnan(ncc) == False:
                    if ncc >= 0:
                        slow_down_coeff = 1 - cp.abs(ncc)/100
                    else:
                        slow_down_coeff = 1
                else:
                    print(f"An NCC = nan found, it was set to the previous one as {slow_down_coeff}")
                
            # Append the values of ncc and affine transform operator to their lists
            ncc_list = cp.append(ncc_list, ncc)
            affine_transform_list = cp.concatenate((affine_transform_list, affine_transform_operator[cp.newaxis, ...]), axis=0)
            
            # Append the middle slices to record the registration process
            if type(slice_extract) == int:
                #middle_slices = cp.concatenate((middle_slices, template_warped[:, template_warped.shape[1]//2, :][cp.newaxis, ...]), axis=0)
                middle_slices = cp.concatenate((middle_slices, template_warped[slice_extract, :, :][cp.newaxis, ...]), axis=0)
            else:
                middle_slices = cp.concatenate((middle_slices, template_warped[0, :, :][cp.newaxis, ...]), axis=0)
            #print(f"For the iteration {count-1} it took {np.round(timing() - t_0, 2)} seconds")
        # Append the middle slice of the moving image so that the process evolves from template to moving throughout the convergence
        if type(slice_extract) == int:
            #middle_slices = cp.concatenate((middle_slices, moving_img[:, template_warped.shape[1]//2, :][cp.newaxis, ...]), axis=0)
            middle_slices = cp.concatenate((middle_slices, moving_img[slice_extract, :, :][cp.newaxis, ...]), axis=0)
        else:
            middle_slices = cp.concatenate((middle_slices, moving_img[0, :, :][cp.newaxis, ...]), axis=0)
            
        # Retun the inverse of the final transform which represents the transformation that when applied to the template takes it to the moving one:
        # return count, cp.linalg.inv(affine_transform_operator).get(), middle_slices.get(), ncc_list.get(), affine_transform_list.get()
        # Returning the results in the same form as the previous LC to not add much modification to the code below
        T = cp.linalg.inv(affine_transform_operator).get()
        disp_x = T[2][-1]
        disp_y = T[1][-1]
        disp_z = T[0][-1]
        last_img = middle_slices.get()
        ncc = (ncc_list[1], ncc_list[-1])
        stats = (-1, -1) # Set it to -1, -1 as it is not being used anywhere anyway
    
        del middle_slices
        del template_warped
        del moving_img
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
        
        return disp_x, disp_y, disp_z, count, T, last_img, ncc, stats
        #-d_x_s, -d_y_s, -d_z_s, count, np.linalg.inv(rigid_transform_operator), np.array(middle_slices), ncc, (mean_diff, std_diff)

    @staticmethod
    def correlate_NCC(search, 
                      template, 
                      downscale=1,
                      downscale_stages = 1,
                      use_spline=False,
                      use_mask_template=False,
                      use_mask_search=False,
                      use_minimun_count=False,
                      mask_threshold=(-1E-10, 1E-10),
                      minimum_count=1E6,
                      apply_gaussian_img_x_y_z=(0,0,0),
                      apply_gaussian_NCC_x_y_z=(0,0,0)):
        """
        This function does a pixel-wise correlation procedure based on the normalized cross-correlation using the correlation functions implemented in CUDA from cupy and allows for masking capabilities.
        
        Parameters
        ----------
        search : 3-dimensional ndarray
            The image where the template will be looked for. It has to have dimensions greater or equal to the template.
        template : 3-dimensional ndarray
            The image that will be looked for in the search..
        downscale : INT, optional
            Allows for the image to be downscaled before the analysis. Only integers 1,2,3,... allowed. The default is 1.
        use_mask_template : Boolean, optional
            If set to True only the values out of the range determined in the mask_threshold will be considered for the correlation on the template image. The default is False.
        use_mask_search : Boolean, optional
            If set to True only the values out of the range determined in the mask_threshold will be considered for the correlation on the search image. The default is False.
        use_minimun_count : Boolean, optional
            The search of the maximum NCC is limited within regions where the correlation included more than the minimum_count pixels. Effective only when use_mask_template and use_mask_search are set to True.. The default is False.
        mask_threshold : Tuple with two entries, optional
            The range to be used for the mask threshold (Tuple of two entries). Default is (-1E-10, 1E-10). The default is (-1E-10, 1E-10).
        minimum_count : TYPE, optional
            The minimum number of pixels to account for when using masking and use_minimum_count option. The default is 1E6.
    
        Returns
        -------
        (1x3 ndarray), float, tuple(2D ndarray, 2D ndarray)
            the position (z, y, x) with the maximal correlation coefficient (ncc) within the search image.
            the optimal NCC in percentage for this position
            two slices extracted from the registered volumes: If full of 0 it means that they could not be extracted
    
        """
        
        # Apply a gaussian filter to the images if requested
        if apply_gaussian_img_x_y_z != (0,0,0):
            template = gaussian_filter_cp(cp.asarray(template), sigma=cp.asarray((apply_gaussian_img_x_y_z[2], 
                                                                               apply_gaussian_img_x_y_z[1], 
                                                                               apply_gaussian_img_x_y_z[0]))).get()
    
            search = gaussian_filter_cp(cp.asarray(search), sigma=cp.asarray((apply_gaussian_img_x_y_z[2],
                                                                           apply_gaussian_img_x_y_z[1],
                                                                           apply_gaussian_img_x_y_z[0]))).get()
        # Downscale function from itk
            # interpolator
        if use_spline:
            interpolator = itk.sitkBSpline
        else:
            interpolator = itk.sitkLinear
            # function        
        def downscale_itk(img_np, scale_factor):
            """
            This function uses tri-linear interpolation to downscale an image from itk.
            """
            # Convert to a SimpleITK image
            image = itk.GetImageFromArray(img_np)
            # Define the new size parameters
            new_size = [int(s * scale_factor) for s in image.GetSize()]
            new_spacing = [o_s / n_s * s for o_s, n_s, s in zip(image.GetSpacing(), new_size, image.GetSize())]
            # Resample with linear interpolation
            resampled_image = itk.Resample(image, new_size, itk.Transform(), interpolator,
                                            image.GetOrigin(), new_spacing, image.GetDirection(), 0, 
                                            image.GetPixelID())
            # Convert back to NumPy array
            resampled_array = itk.GetArrayFromImage(resampled_image)
            return resampled_array
                
        # Check the compatability of the dimensions
        if (template.shape[0] > search.shape[0]) or (template.shape[1] > search.shape[1]) or (template.shape[2] > search.shape[2]):
            #sys.exit("The template must have dimensions smaller than the search one.")
            print("The template must have dimensions smaller than the search one.")
            pos_opt_z_y_x = np.array((0,0,0))
            ncc_opt = -1        
            N_opt = -1
            template_slice = cp.zeros((template.shape[1], template.shape[2]))
            search_slice = template_slice
        try:
            # Check the compatability of the downscale
            if downscale <= 0:
                sys.exit("The scale must be greater or equal to 0")
            else:
                # Extract the mask from the search and downscale it if required
                if use_mask_search:
                    mask_search = ~((search >= mask_threshold[0]) & (search <= mask_threshold[1]))
                if downscale != 1:
                    if downscale > 1:
                        search = itk.GetArrayFromImage(itk.BinShrink(itk.GetImageFromArray(search), (downscale, downscale, downscale)))
                    else:
                        for stage in range(downscale_stages):
                            search = downscale_itk(search, downscale)
                
                # Extract the mask from the template and downscale it if required
                if use_mask_template:
                    mask_template = ~((template >= mask_threshold[0]) & (template <= mask_threshold[1]))
                if downscale != 1:
                    if downscale > 1:
                        template = itk.GetArrayFromImage(itk.BinShrink(itk.GetImageFromArray(template), (downscale, downscale, downscale)))
                    else:
                        for stage in range(downscale_stages):
                            template = downscale_itk(template, downscale)
                
                # make sure to remove the interpolation outliers in case of a mask
                if use_mask_template:
                    if downscale != 1:
                        if downscale > 1:
                            mask_template = itk.GetArrayFromImage(itk.BinShrink(itk.GetImageFromArray(mask_template.astype(np.float32)), (downscale, downscale, downscale)))
                        else:
                            for stage in range(downscale_stages):
                                mask_template = downscale_itk(mask_template.astype(np.float32), downscale)
                            
                    mask_template[mask_template != 1] = 0
                    template = template * mask_template # turning some of the values to 0 makes sure that these do not add anything to the operators on the correlation process
                    
                if use_mask_search:
                    if downscale != 1:
                        if downscale > 1:
                            mask_search = itk.GetArrayFromImage(itk.BinShrink(itk.GetImageFromArray(mask_search.astype(np.float32)), (downscale, downscale, downscale)))
                        else:
                            for stage in range(downscale_stages):
                                mask_search = downscale_itk(mask_search.astype(np.float32), downscale)
                    mask_search[mask_search != 1] = 0
                    search = search * mask_search # turning some of the values to 0 makes sure that these do not add anything to the operators on the correlation process
                        
            # Convert the arrays to cupy ones and float 32 data type
                # The images
            search = cp.asarray(search, dtype=cp.float32)
            template = cp.asarray(template, dtype=cp.float32)
            
                # Their respective mask
            if use_mask_search:
                mask_search = cp.asarray(mask_search)
            if use_mask_template:
                mask_template = cp.asarray(mask_template)
            
            # Get the operators needed for computing the NCC: not affected by the mask
            a = correlate(search, template, mode="constant", cval=0.0)
            
            if (use_mask_template == False):
                # When no mask on the template is requested
                N = template.size # the size of the template array
                c = correlate(search, cp.ones(template.shape), mode="constant", cval=0.0)
                d = correlate(search**2, cp.ones(template.shape), mode="constant", cval=0.0)
            else:
                # When the mask is in template is requested
                    # mask the template and get the correlation
                c = correlate(search, mask_template, mode="constant", cval=0.0)
                d = correlate(search**2, mask_template, mode="constant", cval=0.0)
                # When the mask in only requested on the template make sure to get the right number of voxels that enter the correlation process
                if use_mask_search == False:
                    N = cp.sum(mask_template) # the size of the template array
            
            if use_mask_search == False:
                # Compute b and e
                b = cp.sum(template)
                e = cp.sum(template**2)
            else:
                # Compute b and e
                b = correlate(mask_search, template, mode="constant", cval=0.0)
                e = correlate(mask_search, template**2, mode="constant", cval=0.0)
                # When the mask is requested in both template and search make sure that the number of correlated pixels is obtained as the intersection of both masks at every voxel position.
                N = correlate(mask_search, mask_template, mode="constant", cval=0.0)
            
            # Calculate the CC
            cc = (a - b*c/N)/cp.sqrt(d-c**2/N)/cp.sqrt(e-b**2/N)

            # smoothen the CC if requested
            if apply_gaussian_NCC_x_y_z != (0,0,0):
                cc = gaussian_filter_cp(cc, sigma=(apply_gaussian_NCC_x_y_z[2], 
                                                   apply_gaussian_NCC_x_y_z[1], 
                                                   apply_gaussian_NCC_x_y_z[0]))
            
            # Coverage which works only in case of a mask
            if (use_minimun_count) and (use_mask_search or use_mask_template):
                cc[N < minimum_count] = -1
            
            # Get the maximum position
            try:
                max_pos = cp.asarray(cp.where(cc == cp.nanmax(cc*(cc != cp.inf)))).get()
            
                # Get the first one if more than one values are found at the maximum one
                if max_pos.shape[1] > 1:
                    max_pos = max_pos[:, 0]
                # Check if there is no value returned by the function
                if max_pos.size != 0:
                    pos_opt_z_y_x = max_pos.ravel()
                    pos_opt_z_y_x = pos_opt_z_y_x
                    ncc_opt = cc[pos_opt_z_y_x[0]][pos_opt_z_y_x[1]][pos_opt_z_y_x[2]].get()
                    if use_mask_search == True:
                        N_opt = N[pos_opt_z_y_x[0]][pos_opt_z_y_x[1]][pos_opt_z_y_x[2]].get()
                    else:
                        N_opt = N
                else:
                    print("Could not find the maximum position!")
                    pos_opt_z_y_x = np.array((0,0,0))
                    ncc_opt = -1        
                    N_opt = -1
            except:
                print("Something wrong while finding the maxima!")
                pos_opt_z_y_x = np.array((0,0,0))
                ncc_opt = -2 
                N_opt = -2
            
            # get two registered slices to show the correspondance: use the displacement to avoid getting negative indexes
            s_t_l = (pos_opt_z_y_x - np.array(template.shape) // 2).astype(int)
            s_t_r = (pos_opt_z_y_x - np.array(template.shape) // 2 + np.array(template.shape)).astype(int)
            
            check_left = s_t_l < np.zeros(3)
            check_right = s_t_r > np.array(search.shape)
            
            pad_left = check_left * (-s_t_l)
            pad_right = check_right * (s_t_r - np.array(search.shape))
            
            try:
                template_slice = template[pad_left[0] + template.shape[0]//2 - pad_right[0], 
                                          pad_left[1]:template.shape[1]-pad_right[1], 
                                          pad_left[2]:template.shape[2]-pad_right[2]]
                
                search_slice = search[pad_left[0] + pos_opt_z_y_x[0] - pad_right[0], 
                                      pad_left[1] + pos_opt_z_y_x[1] - template.shape[1] // 2:pos_opt_z_y_x[1] - template.shape[1] // 2 + template.shape[1]-pad_right[1], 
                                      pad_left[2] + pos_opt_z_y_x[2] - template.shape[2] // 2:pos_opt_z_y_x[2] - template.shape[2] // 2 + template.shape[2]-pad_right[2]]
            except:
                print("Could not extract corresponding slices!")
                template_slice = cp.zeros((template.shape[1], template.shape[2]))
                search_slice = template_slice
        except:
            print("Could not correlate!")
            pos_opt_z_y_x = np.array((0,0,0))
            ncc_opt = -1        
            N_opt = -1
            template_slice = cp.zeros((template.shape[1], template.shape[2]))
            search_slice = template_slice
            
        # Compute the correction factor due to downscale
        if downscale < 1: downscale = 1 / (downscale ** downscale_stages)
                
        return pos_opt_z_y_x*downscale, np.round(ncc_opt*100,1), N_opt, (template_slice.get(), search_slice.get()), template.get(), search.get()
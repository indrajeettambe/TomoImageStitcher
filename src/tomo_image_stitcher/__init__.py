"""
TomoImageStitcher — GPU-accelerated 3D volumetric stitching.

A package for stitching 3D sub-volumes (tomography reconstructions,
projection volumes, etc.) using a ZNCC pixel search followed by an
inverse-compositional Lucas-Kanade refinement, with mask-aware
blending on the GPU.

Modules
-------
stitcher       Main ``Stitcher`` class driving the full pipeline.
registration   ``RegistrationKIT`` with ZNCC and IC-GN Lucas-Kanade engines.
transform      ``affine_transform_large_data`` for chunked GPU affine warps.
utilities      H5 I/O helpers, circular masks, distance functions.
danmax         Optional DanMAX beamline utilities.
"""

from .stitcher import Stitcher, Utilities
from .registration import RegistrationKIT
from .transform import affine_transform_large_data

__version__ = "0.2.0"
__author__ = "Endri Lacaj, Indrajeet Tambe"
__license__ = "MIT"

__all__ = [
    "Stitcher",
    "Utilities",
    "RegistrationKIT",
    "affine_transform_large_data",
    "__version__",
]

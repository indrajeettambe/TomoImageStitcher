"""Verify constant values stay constant at a partially covered mask boundary."""
import numpy as np
from scipy.ndimage import map_coordinates
y,x=np.mgrid[:304,:304];mask=((x-151.5)**2+(y-151.5)**2<151.**2).astype('f4');img=mask*7.25
coords=np.array([[151.5]*8,[.55,.8,1.1,1.4,1.8,2.2,2.6,3.]])
alpha=map_coordinates(mask,coords,order=1,mode='constant',cval=0,prefilter=False);value=map_coordinates(img,coords,order=1,mode='constant',cval=0,prefilter=False)
assert np.any((alpha>0)&(alpha<1));np.testing.assert_allclose(value[alpha>0]/alpha[alpha>0],7.25,rtol=1e-6)
print('edge interpolation preserves a constant value through partial footprint coverage')

#!/usr/bin/env python3
"""Screen adjacent layers. Scaled images are registration diagnostics, never output data."""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parent.parent
steps = sorted(ROOT.glob('[0-9][0-9][0-9][0-9]um'))
results = {}
for folder in steps:
 results[folder.name]=[]
 for i in range(3):
  with h5py.File(folder/f'layers/layer{i}.h5','r') as f, h5py.File(folder/f'layers/layer{i+1}.h5','r') as g:
   a=f['stitched_data/stitched_image']; b=g['stitched_data/stitched_image']; h=min(a.shape[1],b.shape[1]);w=min(a.shape[2],b.shape[2]);
   def plane(d,z):
    # Read full slices to avoid inefficient HDF5 point selection.
    return gaussian_filter(d[z,:h,:w][::4,::4],0.7)
   az=[2,5,8,11]; aa=np.stack([plane(a,z) for z in az]); bb=np.stack([plane(b,z) for z in range(300,324)])
   scores=[]
   for off in range(300,313):
    v=bb[np.array(az)+off-300]; mask=(aa!=0)&(v!=0); x=aa[mask];y=v[mask]
    scores.append(float(np.corrcoef(x,y)[0,1]))
   off=300+int(np.argmax(scores));v=bb[np.array(az)+off-300];
   # Register using separate, summed low-resolution slices; retain individual estimates for consistency assessment.
   estimates=[]
   for x,y in zip(aa,v):
    delta,error,_=phase_cross_correlation(x,y,upsample_factor=10,normalization=None)
    estimates.append((delta*4).tolist())
   rec={'pair':[i,i+1], 'source_a_z':az,'tested_offsets':list(range(300,313)), 'correlations':scores,'best_offset':off,
        'offset_peak_at_search_boundary': off in (300,312), 'shift_to_apply_to_layer_b_yx_pixels':estimates,
        'median_shift_yx_pixels':np.median(estimates,axis=0).tolist()}
   results[folder.name].append(rec)
   print(folder.name,i,'offset',off,'correlation',round(max(scores),4),'shift yx',np.round(np.median(estimates,axis=0),2),'range',np.round(np.ptp(estimates,axis=0),2),flush=True)
   (ROOT/'review_20261103/vertical_measurements.json').write_text(json.dumps(results,indent=2)+'\n')

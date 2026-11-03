#!/usr/bin/env python3
"""Diagnostic XZ and YZ crops around each vertical overlap."""
from pathlib import Path
import argparse,json,h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];O=R/'candidate_restitch';p=argparse.ArgumentParser();p.add_argument('step');p.add_argument('--prefix',default='raw_candidate_volume');a=p.parse_args()
with h5py.File(O/f'{a.prefix}{a.step}.h5') as f:
 d=f['stitched_data/stitched_image'];starts=f['layer_z_starts'][:];H,W=d.shape[1:];rows=[]
 for l in (2,1,0):
  z=int(starts[l]);lo=max(0,z-25);hi=min(d.shape[0],z+40);xz=[];yz=[]
  for k in range(lo,hi):
   slab=d[k,:,:];xz.append(slab[H//2,W//4:3*W//4]);yz.append(slab[H//4:3*H//4,W//2])
  rows.append((l,lo,z,np.array(xz),np.array(yz)))
 fig,ax=plt.subplots(3,2,figsize=(13,9),constrained_layout=True)
 for r,(l,lo,z,x,y) in enumerate(rows):
  for c,img in enumerate((x,y)):
   ax[r,c].imshow(img,cmap='gray',vmin=-.00008,vmax=.0003,aspect='auto',interpolation='nearest');ax[r,c].axhline(z-lo,color='red',lw=.5);ax[r,c].set_title(f'layer {l+1} to {l}, '+('XZ' if c==0 else 'YZ'));ax[r,c].set_ylabel('slices in crop')
 fig.suptitle(f'{a.step}: raw intensity seam overview; red = start of overlap')
 fig.savefig(O/f'{a.prefix}seams_{a.step}.png',dpi=140);plt.close(fig)

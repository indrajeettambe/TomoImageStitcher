#!/usr/bin/env python3
"""Compare same-XY reconstruction tiles from adjacent vertical layers.

Matches by motor-position metadata and fits a translation on two slices; a third
slice is held out. Measurements are diagnostic and not used to alter output data.
"""
from pathlib import Path
import argparse,json
import h5py,numpy as np
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import minimize
from skimage.registration import phase_cross_correlation
R=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--step',default='0000um');p.add_argument('--pair',type=int,default=0);p.add_argument('--limit',type=int,default=38);args=p.parse_args()
i=args.pair;step=args.step
regs=[np.load(R/step/f'registration/layer{k}.npz') for k in (i,i+1)]
coords=[r['coords_px'][:,:2] for r in regs];paths=[r['paths'] for r in regs]
meas=json.loads((R/'review_20261103/vertical_measurements.json').read_text())[step][i];off=meas['best_offset']
with h5py.File(R/step/f'layers/layer{i}.h5') as f:ga=f['stitched_data/layer_final_global_coordinates'][:]
with h5py.File(R/step/f'layers/layer{i+1}.h5') as f:gb=f['stitched_data/layer_final_global_coordinates'][:]
results=[]
for j in range(min(args.limit,len(coords[0]))):
 dist=np.linalg.norm(coords[1]-coords[0][j],axis=1);k=int(np.argmin(dist))
 if dist[k]>3: continue
 with h5py.File(paths[0][j]) as fa,h5py.File(paths[1][k]) as fb:
  da=fa['exchange/data'];db=fb['exchange/data'];zs=[3,7,10]
  a=np.stack([da[z,52:252,52:252] for z in zs]);b=np.stack([db[z+off,52:252,52:252] for z in zs])
 def corr(sh,inds):
  vals=[]
  for n in inds:
   x=gauss_a[n,12:-12,12:-12];y=shift(gauss_b[n],sh,order=1,mode='nearest',prefilter=False)[12:-12,12:-12]
   vals.append(np.corrcoef(x.ravel(),y.ravel())[0,1])
  return float(np.mean(vals))
 gauss_a=gaussian_filter(a,(0,1,1));gauss_b=gaussian_filter(b,(0,1,1))
 guess=phase_cross_correlation(gauss_a[0],gauss_b[0],upsample_factor=2,normalization=None)[0]
 guess=np.clip(guess,-10,10)
 opt=minimize(lambda v:-corr(v,[0,1]),guess,method='Powell',bounds=[(-10,10),(-10,10)],options={'xtol':0.05,'maxiter':35})
 r={'tile_a':j,'tile_b':k,'motor_xy_a':coords[0][j].tolist(),'motor_xy_b':coords[1][k].tolist(),'motor_distance_px':float(dist[k]),'source_a':str(paths[0][j]),'source_b':str(paths[1][k]),'fit_shift_to_b_yx':opt.x.tolist(),'shift_implied_by_mosaic_yx':(gb[k,:2]-ga[j,:2])[::-1].tolist(),'heldout_corr_zero':corr([0,0],[2]),'heldout_corr_fit':corr(opt.x,[2]),'heldout_corr_mosaic':corr((gb[k,:2]-ga[j,:2])[::-1],[2]),'a_std':float(a.std()),'b_std':float(b.std()),'search_edge':bool(np.max(np.abs(opt.x))>9.8)}
 results.append(r)
 print(j,k,'fit',np.round(opt.x,1),'mosaic',np.round(r['shift_implied_by_mosaic_yx'],1),'test',round(r['heldout_corr_zero'],2),round(r['heldout_corr_fit'],2),round(r['heldout_corr_mosaic'],2),flush=True)
 out=R/'review_20261103'/f'raw_tile_alignment_{step}_pair{i}.json';out.write_text(json.dumps({'step':step,'pair':i,'offset':off,'tiles':results,'caveats':'Tile matching uses motor position; single-translation fits and correlation do not verify precipitate preservation.'},indent=2)+'\n')

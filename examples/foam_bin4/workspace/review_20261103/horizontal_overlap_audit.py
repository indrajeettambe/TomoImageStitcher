#!/usr/bin/env python3
"""Evaluate paired source-tile overlap NCC before/after recorded XY placements."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--step',default='0000um');p.add_argument('--layer',type=int,default=0);p.add_argument('--solution',action='store_true');a=p.parse_args()
reg=np.load(R/a.step/f'registration/layer{a.layer}.npz');motor=reg['coords_px'][:,:2].astype(float)
with h5py.File(R/a.step/f'layers/layer{a.layer}.h5') as f: placed=f['stitched_data/layer_final_global_coordinates'][:,:2]
if a.solution:
 corrections=np.asarray(json.loads((R/'review_20261103'/f'position_solution_{a.step}.json').read_text())['xy_corrections_yx'])[a.layer];placed=placed+corrections[:,::-1]
neighbours=json.loads((R/a.step/f'registration/layer{a.layer}_neighbours.json').read_text())
zs=[80,162,240];tiles=[]
for p in reg['paths']:
 with h5py.File(str(p)) as f: tiles.append(np.asarray(f['exchange/data'][zs,:,:]))
def ncc(i,j,centers):
 ci=centers[i];cj=centers[j];radius=145.;
 xmin=int(np.ceil(max(ci[0]-radius,cj[0]-radius)));xmax=int(np.floor(min(ci[0]+radius,cj[0]+radius)))
 ymin=int(np.ceil(max(ci[1]-radius,cj[1]-radius)));ymax=int(np.floor(min(ci[1]+radius,cj[1]+radius)))
 if xmin>xmax or ymin>ymax:return None
 yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
 m=((xx-ci[0])**2+(yy-ci[1])**2<radius**2)&((xx-cj[0])**2+(yy-cj[1])**2<radius**2)
 if m.sum()<200:return None
 yi=yy[m]-ci[1]+151.5;xi=xx[m]-ci[0]+151.5;yj=yy[m]-cj[1]+151.5;xj=xx[m]-cj[0]+151.5
 result=[]
 for z in range(3):
  x=map_coordinates(tiles[i][z],[yi,xi],order=1,mode='constant',prefilter=False);y=map_coordinates(tiles[j][z],[yj,xj],order=1,mode='constant',prefilter=False)
  result.append(float(np.corrcoef(x,y)[0,1]))
 return {'ncc':result,'pixels':int(m.sum())}
result=[]
for i,ns in enumerate(neighbours):
 for j in ns:
  if j<=i:continue
  x=ncc(i,j,motor);y=ncc(i,j,placed)
  if x is not None and y is not None:result.append({'pair':[i,j],'motor':x,'registered':y})
O=R/'review_20261103'/(f'horizontal_overlap_{a.step}_layer{a.layer}' + ('_solution.json' if a.solution else '.json'));O.write_text(json.dumps({'step':a.step,'layer':a.layer,'z_slices':zs,'pairs':result,'caveat':'Motor and registered overlaps use different sampling locations; high/low NCC is not itself feature-preservation evidence.'},indent=2)+'\n')
x=np.array([q['motor']['ncc'] for q in result]);y=np.array([q['registered']['ncc'] for q in result]);print(a.step,a.layer,'pairs',len(result),'median motor',np.median(x),'median registered',np.median(y),'registered worse >=.1',np.sum(np.mean(y-x,axis=1)<-.1),flush=True)

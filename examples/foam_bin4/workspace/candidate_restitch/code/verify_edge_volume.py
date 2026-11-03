#!/usr/bin/env python3
"""Independent sampled source-to-output numerical verification, not scientific QC."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[2];O=R/'candidate_restitch';p=argparse.ArgumentParser();p.add_argument('step');a=p.parse_args()
path=O/f'edge_candidate_volume{a.step}.h5'
with h5py.File(path) as f:
 d=f['stitched_data/stitched_image'];m=json.loads(f.attrs['provenance_json']);origx,origy=m['origin_xy'];starts=m['layer_z_starts'];centers=np.array(m['tile_centers_xy']);radius=m['radius_pixels'];H,W=d.shape[1:]
 # Verify a fixed central crop so we independently recompute each tile's weights.
 y0=H//2-128;x0=W//2-128;yy,xx=np.mgrid[y0+origy:y0+origy+256,x0+origx:x0+origx+256];sy,sx=np.mgrid[:304,:304];source_support=((sx-151.5)**2+(sy-151.5)**2<151.**2).astype('f4');
 def render(l,z):
  num=np.zeros((256,256),'f4');den=np.zeros_like(num)
  for c,src in zip(centers[l],m['source_paths'][l]):
   radius_map=np.hypot(xx-c[0],yy-c[1]);mask=radius_map<radius
   if not mask.any():continue
   weight=np.maximum(radius-radius_map,0).astype('f4')
   with h5py.File(src) as source:img=source['exchange/data'][z,:,:]
   coords=[yy[mask]-c[1]+151.5,xx[mask]-c[0]+151.5];alpha=map_coordinates(source_support,coords,order=1,mode='constant',cval=0,prefilter=False);num[mask]+=map_coordinates(img,coords,order=1,mode='constant',cval=0,prefilter=False)*weight[mask];den[mask]+=weight[mask]*alpha
  out=np.zeros_like(num);np.divide(num,den,out=out,where=den>0);return out,den>0
 zs=sorted(set([starts[i]+162 for i in range(4)]+[starts[i]+6 for i in range(3)]));checks=[]
 for z in zs:
  ids=[i for i in range(4) if 0<=z-starts[i]<324];ids.sort(key=lambda i:starts[i]);aa,ma=render(ids[0],z-starts[ids[0]])
  if len(ids)==1:expected=aa
  else:
   bb,mb=render(ids[1],z-starts[ids[1]]);w=(z-starts[ids[1]])/(starts[ids[0]]+323-starts[ids[1]]);expected=np.zeros_like(aa);both=ma&mb;expected[both]=aa[both]*(1-w)+bb[both]*w;expected[ma&~mb]=aa[ma&~mb];expected[mb&~ma]=bb[mb&~ma]
  actual=d[z,y0:y0+256,x0:x0+256];np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-11);checks.append(int(z))
 result={'step':a.step,'shape':d.shape,'checked_crop_origin_yx':[y0,x0],'checked_z':checks,'result':'source recomposition matches output within float tolerance','limitation':'sampled implementation verification; does not establish scientific correctness'}
(O/f'edge_verification_{a.step}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

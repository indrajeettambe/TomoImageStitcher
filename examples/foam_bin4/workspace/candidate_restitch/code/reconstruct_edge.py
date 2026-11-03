#!/usr/bin/env python3
"""Recompose experimental bin4 volumes from original reconstruction tiles.

WARNING: graph registrations are not scientifically validated. Original sources
and current volumes must be kept. This program does not denoise or segment.
"""
import argparse,json,os
from pathlib import Path
import h5py,numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[2];O=R/'candidate_restitch';
def main(step, test_pilot=False):
 sol=json.loads((R/'review_20261103'/f'position_solution_{step}.json').read_text())
 if not sol['success']:raise ValueError('Position optimizer did not converge')
 corrections=np.array(sol['xy_corrections_yx']);vertical=json.loads((R/'review_20261103/vertical_measurements.json').read_text())[step];offsets=[r['best_offset'] for r in vertical]
 starts=[sum(offsets[i:]) for i in range(3)]+[0];depth=max(starts)+324
 regs=[np.load(R/step/f'registration/layer{i}.npz') for i in range(4)];centers=[]
 for l in range(4):
  with h5py.File(R/step/f'layers/layer{l}.h5') as f:c=f['stitched_data/layer_final_global_coordinates'][:,:2]
  centers.append(c+corrections[l,:,::-1])
 centers=np.asarray(centers);radius=151.;syy,sxx=np.mgrid[:304,:304];source_support=((sxx-151.5)**2+(syy-151.5)**2<151.**2).astype('f4');xmin=int(np.floor(centers[:,:,0].min()-radius));ymin=int(np.floor(centers[:,:,1].min()-radius));xmax=int(np.ceil(centers[:,:,0].max()+radius));ymax=int(np.ceil(centers[:,:,1].max()+radius));H=ymax-ymin+1;W=xmax-xmin+1
 allfiles=[];layers=[]
 try:
  for l in range(4):
   tiles=[];den=np.zeros((H,W),'f4');count=np.zeros((H,W),'u1')
   for i,c in enumerate(centers[l]):
    xa=max(0,int(np.floor(c[0]-radius))-xmin);xb=min(W,int(np.ceil(c[0]+radius))-xmin+1);ya=max(0,int(np.floor(c[1]-radius))-ymin);yb=min(H,int(np.ceil(c[1]+radius))-ymin+1)
    yy,xx=np.mgrid[ya+ymin:yb+ymin,xa+xmin:xb+xmin];r=np.hypot(xx-c[0],yy-c[1]);mask=r<radius;coords=(yy[mask]-c[1]+151.5,xx[mask]-c[0]+151.5);alpha=map_coordinates(source_support,coords,order=1,mode='constant',cval=0,prefilter=False);w=np.maximum(radius-r,0).astype('f4');effective=np.zeros_like(w);effective[mask]=w[mask]*alpha;view=np.s_[ya:yb,xa:xb];den[view]+=effective;count[view]+=(effective>0).astype('u1')
    f=h5py.File(str(regs[l]['paths'][i]));allfiles.append(f);tiles.append((f,view,mask,w,coords))
   layers.append((tiles,den,count))
  def render(l,z):
   tiles,den,_=layers[l];out=np.zeros((H,W),'f4')
   for f,view,mask,w,coords in tiles:
    a=f['exchange/data'][z,:,:];v=out[view];v[mask]+=map_coordinates(a,coords,order=1,mode='constant',cval=0,prefilter=False)*w[mask]
   np.divide(out,den,out=out,where=den>0);return out
  if test_pilot:
   raise ValueError('old radius-150 pilot is not applicable to the edge-aware compositor')
   if step!='0000um': raise ValueError('pilot exists only for 0000um')
   with h5py.File(O/'raw_seam_pilot/raw_seam_pilot.h5') as pilot:
    meta=json.loads(pilot.attrs['provenance_json']);py,px=meta['roi_origin_yx'];ys=slice(py-ymin,py-ymin+384);xs=slice(px-xmin,px-xmin+384)
    for l,t in [(0,3),(0,8),(1,310),(1,318)]:
     index=int(np.where(pilot['layer1_z'][:]==(t+offsets[0] if l==0 else t))[0][0]);reference=pilot[f'layer{l}_raw_mosaic'][index];actual=render(l,t)[ys,xs]
     np.testing.assert_allclose(actual,reference,rtol=2e-6,atol=1e-11)
   print('Raw assembly agrees with independent seam pilot on four planes',flush=True);return
  final=O/f'edge_candidate_volume{step}.h5';partial=O/f'edge_candidate_volume{step}.h5.partial'
  if final.exists():raise FileExistsError(final)
  provenance={'status':'EXPERIMENTAL; registration and preservation are not validated','source':'bin4 source reconstruction tiles, unchanged native intensity units','step':step,'source_paths':[[str(x) for x in reg['paths']] for reg in regs],'position_solution':str(R/'review_20261103'/f'position_solution_{step}.json'),'tile_centers_xy':centers.tolist(),'origin_xy':[xmin,ymin],'layer_z_starts':starts,'z_offsets':offsets,'radius_pixels':radius,'method':'support-normalized linear XY interpolation with radius-151 geometric footprint and normalized radial feathering; linear axial feather; no denoising or intensity correction', 'limitations':['bin4 inputs have reduced spatial resolution','some alignment constraints are uncertain','some horizontal overlaps can worsen','radius-151 disk omits source tile corners and edges','interpolation and blending alter fine features; retain all original inputs']}
  with h5py.File(partial,'a') as out:
   if 'stitched_data/stitched_image' in out:
    d=out['stitched_data/stitched_image'];old=json.loads(out.attrs['provenance_json'])
    if d.shape!=(depth,H,W) or old['tile_centers_xy']!=provenance['tile_centers_xy'] or old['layer_z_starts']!=starts:raise ValueError('incompatible partial file')
    n=d.id.get_num_chunks();indices=sorted(d.id.get_chunk_info(i).chunk_offset[0] for i in range(n))
    if indices!=list(range(n)):raise ValueError('partial chunks are not a contiguous prefix')
    zstart=n;print(step,'resuming at',zstart,flush=True)
   else:
    d=out.create_dataset('stitched_data/stitched_image',shape=(depth,H,W),dtype='f4',chunks=(1,H,W))
    out.create_dataset('coverage_count_by_layer',data=np.stack([x[2] for x in layers]),chunks=(1,H,W),compression='lzf');out.create_dataset('layer_z_starts',data=starts);out.attrs['provenance_json']=json.dumps(provenance);zstart=0
   for z in range(zstart,depth):
    ids=[i for i in range(4) if 0<=z-starts[i]<324]
    if len(ids)==1:plane=render(ids[0],z-starts[ids[0]])
    elif len(ids)==2:
     a,b=sorted(ids,key=lambda k:starts[k]);A=render(a,z-starts[a]);B=render(b,z-starts[b]);ma=layers[a][1]>0;mb=layers[b][1]>0;w=(z-starts[b])/(starts[a]+323-starts[b]);plane=np.zeros((H,W),'f4');both=ma&mb;plane[both]=A[both]*(1-w)+B[both]*w;plane[ma&~mb]=A[ma&~mb];plane[mb&~ma]=B[mb&~ma]
    else:raise ValueError('unexpected coverage')
    if not np.isfinite(plane).all():raise ValueError(f'nonfinite output slice {z}')
    d[z]=plane
    if z%50==0:print(step,z,'/',depth,flush=True)
   out.attrs['complete']=True;out.flush()
  with h5py.File(partial) as f:
   if not f.attrs['complete'] or f['stitched_data/stitched_image'].shape!=(depth,H,W):raise ValueError('invalid output')
  os.rename(partial,final);print('complete',final,flush=True)
 finally:
  for f in allfiles:f.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--test-pilot',action='store_true');p.add_argument('steps',nargs='+',choices=['0000um','0250um','0500um','0750um','1000um']);a=p.parse_args()
 for step in a.steps:main(step,a.test_pilot)

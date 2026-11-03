#!/usr/bin/env python3
"""Evaluate accepted graph edges at unselected axial slices z=1 and z=12."""
from pathlib import Path
import argparse,json,h5py,numpy as np
from scipy.ndimage import gaussian_filter,shift
R=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser();p.add_argument('step',default='0000um',nargs='?');args=p.parse_args();step=args.step
sol=json.loads((R/'review_20261103'/f'position_solution_{step}.json').read_text());out=[]
for edge in sol['heldout_evaluations']:
 layer=edge['pair'];data=json.loads((R/'review_20261103'/f'raw_tile_alignment_{step}_pair{layer}.json').read_text());rec=next(x for x in data['tiles'] if x['tile_a']==edge['tile_a'] and x['tile_b']==edge['tile_b']);off=data['offset']
 with h5py.File(rec['source_a']) as a,h5py.File(rec['source_b']) as b:
  for z in [1,12]:
   # z=12 may fall outside layer B for offset 312; skip instead of extrapolating.
   if z+off>=b['exchange/data'].shape[0]:continue
   x=gaussian_filter(a['exchange/data'][z,52:252,52:252],1)[12:-12,12:-12];y=gaussian_filter(b['exchange/data'][z+off,52:252,52:252],1)
   def corr(sh):return float(np.corrcoef(x.ravel(),shift(y,sh,order=1,mode='nearest',prefilter=False)[12:-12,12:-12].ravel())[0,1])
   out.append({'pair':layer,'tile_a':edge['tile_a'],'z':z,'ncc_old':corr(edge['old_shift_yx']),'ncc_new':corr(edge['new_shift_yx'])})
(R/'review_20261103'/f'position_independent_{step}.json').write_text(json.dumps(out,indent=2)+'\n');d=np.array([x['ncc_new']-x['ncc_old'] for x in out]);print(step,'n',len(d),'median',np.median(d),'improved',sum(d>0),'worsened>.05',sum(d<-.05),flush=True)

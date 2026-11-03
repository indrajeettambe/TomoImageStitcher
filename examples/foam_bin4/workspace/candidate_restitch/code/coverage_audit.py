#!/usr/bin/env python3
"""Screen geometric gaps in per-layer and overlap coverage; no segmentation."""
from pathlib import Path
import argparse,json,h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument("--prefix",default="raw_candidate_volume");args=p.parse_args()
from scipy.ndimage import binary_fill_holes,label
R=Path(__file__).resolve().parents[2];out={}
for p in sorted((R/'candidate_restitch').glob(args.prefix+'[0-9]*um.h5')):
 step=p.name[len(args.prefix):-3]
 with h5py.File(p) as f:
  c=f['coverage_count_by_layer'][:]>0;record=[]
  for l,mask in enumerate(c):
   holes=binary_fill_holes(mask)&~mask;labels,n=label(holes);sizes=np.bincount(labels.ravel())[1:];sizes=sorted(map(int,sizes),reverse=True)
   record.append({'layer':l,'covered_pixels':int(mask.sum()),'enclosed_uncovered_pixels':int(holes.sum()),'enclosed_gap_count':int(n),'largest_enclosed_gaps_pixels':sizes[:10]})
  overlaps=[]
  for i in range(3):
   a,b=c[i],c[i+1];common=(a&b);union=(a|b);overlaps.append({'pair':[i,i+1],'union_pixels':int(union.sum()),'shared_pixels':int(common.sum()),'shared_fraction_of_union':float(common.sum()/union.sum())})
  out[step]={'layers':record,'vertical_overlaps':overlaps,'warning':'2D geometric masks only; does not establish correct alignment or specimen connectivity.'}
  print(step,'enclosed gaps',[x['enclosed_uncovered_pixels'] for x in record],'vertical shared',[round(x['shared_fraction_of_union'],4) for x in overlaps])
(R/('candidate_restitch/'+args.prefix+'coverage_audit.json')).write_text(json.dumps(out,indent=2)+'\n')

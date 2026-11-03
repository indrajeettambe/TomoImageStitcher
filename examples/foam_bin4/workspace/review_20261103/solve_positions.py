#!/usr/bin/env python3
"""Conservative pilot global XY corrections from independently held-out overlaps."""
from pathlib import Path
import argparse,json
import h5py,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.ndimage import gaussian_filter,shift
R=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('step',nargs='?',default='0000um');args=p.parse_args();step=args.step
N=152; positions=[]
for l in range(4):
 with h5py.File(R/step/f'layers/layer{l}.h5') as f:positions.append(f['stitched_data/layer_final_global_coordinates'][:,:2][:,::-1])
positions=np.asarray(positions)
edges=[];accepted=[]
for layer in range(4):
 data=json.loads((R/'review_20261103'/f'horizontal_overlap_{step}_layer{layer}.json').read_text())
 for x in data['pairs']:
  if np.median(x['registered']['ncc']) >= .7:
   i,j=x['pair'];edges.append((layer*38+i,layer*38+j,np.array([0.,0.]),2.0,'horizontal'))
for layer in range(3):
 data=json.loads((R/'review_20261103'/f'raw_tile_alignment_{step}_pair{layer}.json').read_text())
 for x in data['tiles']:
  if x['heldout_corr_fit']>=.8 and not x['search_edge'] and x['heldout_corr_fit']>=x['heldout_corr_zero']-.03:
   i,j=x['tile_a'],x['tile_b'];desired=np.array(x['fit_shift_to_b_yx'])-(positions[layer+1,j]-positions[layer,i]);edges.append((layer*38+i,(layer+1)*38+j,desired,1.0,'vertical'));accepted.append((layer,x))
# Linear residuals plus robust Huber loss; preserve horizontal positions softly, anchor layer 0,
# and regularize all correction vectors. No deformation or confidence guarantees.
nedge=len(edges);jac=lil_matrix((2*nedge+2*N,2*N),dtype='i1')
for q,(i,j,_,_,_) in enumerate(edges):
 for axis in range(2):jac[2*q+axis,2*i+axis]=1;jac[2*q+axis,2*j+axis]=1
for i in range(N):
 for axis in range(2):jac[2*nedge+2*i+axis,2*i+axis]=1
jac=jac.tocsr()
def residual(flat):
 v=flat.reshape(N,2);out=[]
 for i,j,d,w,_ in edges:out.extend((v[j]-v[i]-d)*w)
 for i in range(N):out.extend(v[i]*(.8 if i<38 else .15))
 return np.array(out)
fit=least_squares(residual,np.zeros(2*N),jac_sparsity=jac,loss='huber',f_scale=2.,max_nfev=500,ftol=1e-4)
correction=fit.x.reshape(4,38,2)
# Held-out z=10 was not used for fitting the tile-level translations; its scores
# did select high-confidence edges, so evaluation is not fully independent.
evaluations=[]
for layer,x in accepted:
 i,j=x['tile_a'],x['tile_b'];off=json.loads((R/'review_20261103'/f'raw_tile_alignment_{step}_pair{layer}.json').read_text())['offset']
 with h5py.File(x['source_a']) as fa,h5py.File(x['source_b']) as fb:
  a=gaussian_filter(fa['exchange/data'][10,52:252,52:252],1);b=gaussian_filter(fb['exchange/data'][off+10,52:252,52:252],1)
 def corr(shift_yx):
  y=shift(b,shift_yx,order=1,mode='nearest',prefilter=False)[12:-12,12:-12];xx=a[12:-12,12:-12];return float(np.corrcoef(xx.ravel(),y.ravel())[0,1])
 old=positions[layer+1,j]-positions[layer,i];new=old+correction[layer+1,j]-correction[layer,i]
 evaluations.append({'pair':layer,'tile_a':i,'tile_b':j,'old_shift_yx':old.tolist(),'new_shift_yx':new.tolist(),'ncc_old':corr(old),'ncc_new':corr(new)})
out={'step':step,'method':'Huber graph fit with strong horizontal saved-overlap constraints, high-correlation cross-layer tile matches, and position priors','parameters':{'horizontal_weight':2.0,'vertical_weight':1.0,'layer0_prior':.8,'other_prior':.15,'huber_f_scale':2.0},'success':bool(fit.success),'message':fit.message,'edges_horizontal':sum(e[4]=='horizontal' for e in edges),'edges_vertical':len(accepted),'xy_corrections_yx':correction.tolist(),'heldout_evaluations':evaluations,'caveat':'Experimental; held-out slice participates in selecting edges, horizontal constraints assume saved positions, and unobserved tiles depend on priors.'}
(R/'review_20261103'/f'position_solution_{step}.json').write_text(json.dumps(out,indent=2)+'\n')
diffs=np.array([e['ncc_new']-e['ncc_old'] for e in evaluations]);print(step,'edges',out['edges_horizontal'],len(accepted),'success',fit.success,'heldout median change',float(np.median(diffs)),'improved',int((diffs>0).sum()),'worse >0.05',int((diffs<-.05).sum()),'of',len(diffs),'max correction',float(np.max(np.linalg.norm(correction,axis=2))),flush=True)

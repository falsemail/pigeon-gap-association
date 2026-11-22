"""Frozen partial assignment and classical matching algorithms."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from data_protocol import descriptors
from baselines import components, predict as previous
import pygmtools as pygm

def partial(cost,null):
 n,m=cost.shape;out=np.full(n,-1,int)
 if not n or not m:return out
 D=np.full((n,n),1e6);np.fill_diagonal(D,np.broadcast_to(null,n));i,j=linear_sum_assignment(np.c_[cost,D]);valid=j<m;out[i[valid]]=j[valid];return out

def cost(q,cfg):
 s=q['sample'];kind=cfg['kind'];a,b=descriptors(s['a']),descriptors(s['b']);sig=s['meta']['sigma'];h=float(s['h']);n,m=q['pair'].shape[:2]
 if kind in ['cv','multi_model','geometry','structure','kbest']:
  p=components(s)
  if kind=='cv':return p['cv']
  if kind=='multi_model':return p['multi_model']*cfg.get('scale',.1)
  return p['geometry']+cfg.get('structure',0)*p['structure']
 if kind=='relative':
  dr=b['rel'][None]-a['rel'][:,None];dv=b['rv'][None]-a['rv'][:,None];scale=cfg.get('scale',10)
  return np.log1p(np.sum(dr**2,2)/scale**2)+cfg.get('velocity',.2)*np.log1p(np.sum(dv**2,2)/4)
 if kind in ['rrwm','ipfp']:
  # Official pygmtools uses column-major candidate ordering i + j*N.
  C=q['compatible'].transpose(1,0,3,2).reshape(n*m,n*m).astype(np.float64)
  un=np.exp(-np.sum((a['rel'][:,None]-b['rel'][None])**2,2)/(2*cfg.get('scale',10)**2))
  np.fill_diagonal(C,cfg.get('unary',1)*un.T.reshape(-1));C+=np.eye(n*m)*1e-9
  fun=pygm.rrwm if kind=='rrwm' else pygm.ipfp
  P=fun(C,n1=n,n2=m,backend='numpy',max_iter=50)
  return -np.log(np.maximum(P,1e-12))
 raise ValueError(kind)

def predict(q,cfg,precomputed=None):
 if cfg['kind']=='kbest':
  c=dict(method='kbest',structure=cfg.get('structure',1),null=cfg['null'],k=8,rerank=cfg.get('rerank',3));return previous(q['sample'],c,components(q['sample']))
 C=precomputed if precomputed is not None else cost(q,cfg);return partial(C,cfg['null'])

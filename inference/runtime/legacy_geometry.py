"""Frozen project helper extracted from wide-area-maneuver/association.py.
Source SHA256 6279a072ee653f7cd16c40085b5d37b8b7b307a5f1733503684d7d5d4ca44919.
No original simulation, labels, or checkpoints are imported.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
def ct(p,v,w,h):
 a=w*h;d=np.abs(w)>1e-5;s=np.where(d,np.sin(a)/np.where(d,w,1),h);c=np.where(d,(1-np.cos(a))/np.where(d,w,1),.5*w*h*h)
 out=p+v*h;out=out.copy();out[:,0]=p[:,0]+s*v[:,0]-c*v[:,1];out[:,1]=p[:,1]+c*v[:,0]+s*v[:,1];return out

def build(a,b,available,observed):
 h=max(.2,b['t']-a['t']);p,v,acc=a['x'][:,:3],a['x'][:,3:6],a['x'][:,6:9];q,u,ac=b['x'][:,:3],b['x'][:,3:6],b['x'][:,6:9]
 forward=ct(p,v,a['w'],h);backward=ct(q,u,b['w'],-h)
 df=[np.sum((r[:,None]-q[None])**2,2) for r in (p+v*h,forward,p+v*h+.5*acc*h*h)]
 db=[np.sum((p[:,None]-r[None])**2,2) for r in (q-u*h,backward,q-u*h+.5*ac*h*h)]
 same=(a['ids'][:,None]==b['ids'][None]).astype(np.float32);dv=np.linalg.norm(v[:,None]-u[None],axis=2)/250
 scale=300+55*h
 cost=.65*np.log1p(np.minimum.reduce(df)/scale**2)+.35*np.log1p(np.minimum.reduce(db)/scale**2)+.25*dv**2-.6*same
 pair=np.stack([*(np.log1p(d/1e6) for d in df+db),dv,same,np.linalg.norm(p[:,None]-q[None],axis=2)/(320*h+1000),np.broadcast_to(a['sd'][:,:3].mean(1)[:,None]/1000,same.shape),np.broadcast_to(b['sd'][:,:3].mean(1)[None]/1000,same.shape)],-1).astype(np.float32)
 reachable=np.linalg.norm(p[:,None]-q[None],axis=2)<=320*h+3000
 return dict(a=a,b=b,h=np.float32(h),context=np.r_[h/30,min(observed/3,2),np.asarray(available).reshape(-1)].astype(np.float32),cost=cost.astype(np.float32),pair=pair,reachable=reachable,forward=forward.astype(np.float32),backward=backward.astype(np.float32))

def assignment(cost,row_null=3.,quality=None,threshold=0.):
 n,m=cost.shape
 if not n or not m:return np.full(n,-1,int)
 if quality is not None:cost=np.where(quality[None]>=threshold,cost,1e6)
 dummy=np.full((n,n),1e6,np.float32);np.fill_diagonal(dummy,np.broadcast_to(row_null,(n,)))
 ii,jj=linear_sum_assignment(np.c_[cost,dummy]);out=np.full(n,-1,int);use=jj<m;out[ii[use]]=jj[use];return out

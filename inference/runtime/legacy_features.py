"""Frozen project helper extracted from wide-area-dense/dense_model.py.
Source SHA256 e56ec47cb4bf143cf1e80c40e4d050e8b44cc9613883254f35f1eee76b79eda1.
No original simulation, labels, or checkpoints are imported.
"""
import numpy as np
from legacy_geometry import ct
def neighborhood(q):
 x=np.asarray(q['x'],float);p=x[:,:3];v=x[:,3:6]
 dp=p[:,None]-p[None];dv=v[:,None]-v[None]
 w=np.exp(-np.minimum(np.sum(dp*dp,2)/(2*900**2)+np.sum(dv*dv,2)/(2*140**2),30))
 np.fill_diagonal(w,0);mass=w.sum(1);w+=np.eye(len(x));w/=w.sum(1,keepdims=True)
 center=w@x;relative=p-center[:,:3]
 sd=np.sqrt(np.maximum(w@(p*p)-(w@p)**2,0)+100**2)
 velocity=center[:,3:6];speed=np.linalg.norm(velocity[:,:2],axis=1);u=velocity[:,:2]/np.maximum(speed[:,None],1)
 local=np.c_[relative[:,0]*u[:,0]+relative[:,1]*u[:,1],-relative[:,0]*u[:,1]+relative[:,1]*u[:,0],relative[:,2]]
 return dict(center=center,relative=relative,sd=sd,u=u,local=local,mass=mass)

def features(s):
 a,b=s['a'],s['b'];x=np.asarray(a['x'],float);y=np.asarray(b['x'],float);n,m=len(x),len(y);h=float(s['h']);ga,gb=neighborhood(a),neighborhood(b)
 expand_a=lambda q:np.broadcast_to(q[:,None],(n,m,q.shape[-1]))
 expand_b=lambda q:np.broadcast_to(q[None],(n,m,q.shape[-1]))
 def orient(d,u):return np.stack([d[:,:,0]*u[:,None,0]+d[:,:,1]*u[:,None,1],-d[:,:,0]*u[:,None,1]+d[:,:,1]*u[:,None,0],d[:,:,2]],-1)
 def orient_b(d,u):return np.stack([d[:,:,0]*u[None,:,0]+d[:,:,1]*u[None,:,1],-d[:,:,0]*u[None,:,1]+d[:,:,1]*u[None,:,0],d[:,:,2]],-1)
 delta=y[None,:,:3]-x[:,None,:3]
 forward=y[None,:,:3]-s['forward'][:,None];backward=s['backward'][None]-x[:,None,:3]
 rv=ga['local'][:,None]-gb['local'][None];scale=np.sqrt(a['sd'][:,:3,None]**2+b['sd'].T[None,:3,:]**2).transpose(0,2,1)+50
 ca,cb=ga['center'],gb['center'];gf=ct(ca[:,:3],ca[:,3:6],a['w'],h);gr=ct(cb[:,:3],cb[:,3:6],b['w'],-h)
 context=np.broadcast_to(np.asarray(s['context'])[None,None],(n,m,10))
 f=np.concatenate([s['pair'],orient(delta,ga['u'])/5000,orient(forward,ga['u'])/(300+50*h),orient_b(backward,gb['u'])/(300+50*h),
   (y[None,:,3:6]-x[:,None,3:6])/120,(y[None,:,6:9]-x[:,None,6:9])/20,
   rv/500,rv/np.maximum((ga['sd'][:,None]+gb['sd'][None])/2,100),
   orient(cb[None,:,:3]-gf[:,None],ga['u'])/(500+60*h),orient_b(gr[None]-ca[:,None,:3],gb['u'])/(500+60*h),
   expand_a(ga['local']/700),expand_b(gb['local']/700),expand_a(ga['sd']/700),expand_b(gb['sd']/700),
   expand_a((x[:,3:6]-ca[:,3:6])/100),expand_b((y[:,3:6]-cb[:,3:6])/100),
   expand_a(np.c_[np.log1p(ga['mass']),a['w']/.1]),expand_b(np.c_[np.log1p(gb['mass']),b['w']/.1]),
   delta/np.maximum(scale+200*h,100),context],-1)
 return np.clip(f,-15,15).astype(np.float32),ga['mass'].astype(np.float32),gb['mass'].astype(np.float32)

"""Generate one benchmark condition from a registered recorded flight."""
import numpy as np
from core import DT, SCALE, pack_observed, build_sample, label_sample

def episode(seed,n=80,kind='coherent',spacing=1.,sigma=1.5,gap=15.,recovery=2.,engine='leader',turnover=.12,truth=None,onset=10.,clutter=.03):
    if truth is None:raise ValueError('A registered pigeon trajectory is required')
    else:n=truth.shape[1];physics=dict(engine=engine,kind=kind,spacing=spacing,source='provided_trajectory')
    rng=np.random.default_rng(seed+920000000)
    start=round(onset/DT);stop=round((onset+gap+recovery)/DT);first=round((onset+gap)/DT)
    assert stop<len(truth) and start>=20,(stop,len(truth),start)
    # Entry and exit of tracklets are sampled independently of algorithm scores.
    # These schedules describe observation-set turnover, not simulated deaths.
    order=rng.permutation(n);dropa=min(n-1,int(round(n*turnover*rng.uniform(.6,1.4))));dropb=min(n-1,int(round(n*turnover*rng.uniform(.6,1.4))));left=order[dropa:];right=order[:n-dropb]
    left=rng.permutation(left);right=rng.permutation(right)
    origin=np.zeros(3)
    def observed(indices,t0,t1,offset,is_post=False):
        # Draw the complete recovery noise first, then reveal only its prefix.
        # This makes all recovery times share exactly the same observations.
        last=min(t0+20,len(truth)-1) if is_post else t1
        t=np.arange(t0,last+1)*DT;z=truth[t0:last+1,indices,:3].copy();z+=rng.normal(size=z.shape)*sigma*np.array([1,2,1.2])
        nf=int(round(len(indices)*clutter))
        if nf:
            base=z[0,rng.integers(len(indices),size=nf)]+rng.normal(size=(nf,3))*[40,40,10]
            velocity=rng.normal(size=(nf,3))*[15,15,2]
            ghost=base[None]+(t-t[0])[:,None,None]*velocity[None]+rng.normal(size=(len(t),nf,3))*sigma*np.array([1,2,1.2])
            z=np.concatenate([z,ghost],1)
        valid=rng.random(z.shape[:2])>.05;valid[0]=True
        ids=np.arange(z.shape[1])+offset;count=t1-t0+1
        return pack_observed(z[:count],valid[:count],t[:count],ids,sigma,origin)
    a=observed(left,start-20,start,10000)
    # Coordinate origin comes exclusively from the observed old tracklets.
    origin=np.median(a['x'][:,:3],axis=0)
    a['origin']=origin.astype(np.float32);a['static'][:,:3]=(a['x'][:,:3]-origin)/10000
    b=observed(right,first,stop,20000,True)
    left_labels=np.r_[left,np.full(len(a['x'])-len(left),-1)];right_labels=np.r_[right,np.full(len(b['x'])-len(right),-1)]
    sample=label_sample(build_sample(a,b),left_labels,right_labels)
    sample['meta']=dict(seed=seed,n=n,kind=kind,spacing=spacing,sigma=sigma,gap=gap,recovery=recovery,engine=engine,turnover=turnover,clutter=clutter,physics=physics)
    sample['forward_target']=np.r_[truth[stop,left,:3]*SCALE,np.zeros((len(a['x'])-len(left),3))].astype(np.float32)
    sample['forward_valid']=left_labels>=0
    # Geometry truth is training-only; it is excluded from tensorize.
    return sample

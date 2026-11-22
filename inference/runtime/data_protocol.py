"""Group descriptors, pair features and horizontal projection for recorded trajectories."""
import numpy as np
from scipy.stats import rankdata
from core import build_sample, label_sample

def descriptors(q):
 """Observed kinematics, relative roles, and short temporal statistics."""
 x=np.asarray(q['x'],float)/10;n=len(x);p=x[:,:3];v=x[:,3:6];acc=x[:,6:9]
 seq=q['seq'];valid=seq[:,:,18]>.5
 hist=seq[:,:,:3]*100+p[:,None];vh=seq[:,:,3:6]*25
 center=np.median(p,axis=0);vc=np.median(v,axis=0);speed=np.linalg.norm(vc[:2]);u=vc[:2]/max(speed,1e-5)
 if speed<1e-5:u=np.array([1.,0.])
 R=np.array([[u[0],-u[1],0],[u[1],u[0],0],[0,0,1.]])
 rel=(p-center)@R;rv=(v-vc)@R;ra=(acc-np.median(acc,axis=0))@R
 d2=np.sum((p[:,None]-p[None])**2,axis=2);d=np.sqrt(d2);np.fill_diagonal(d,np.inf)
 nearest=np.min(d,axis=1) if n>1 else np.ones(n)*10
 scale=np.clip(np.median(nearest),2,100);extent=np.maximum(np.median(abs(rel),axis=0)*1.4826,2)
 # A common observation-based frame at each visible time, never inside outage.
 hrel=[];hvel=[];hrank=[]
 for t in range(21):
  ok=valid[:,t];c=np.median(hist[ok,t],axis=0) if ok.any() else center
  relt=(hist[:,t]-c)@R;hrel.append(relt)
  vv=np.median(vh[ok,t],axis=0) if ok.any() else vc
  hvel.append((vh[:,t]-vv)@R)
  hrank.append((rankdata(relt,axis=0,method='min')-1)/n-.5)
 hrel=np.stack(hrel,1)
 mu=np.stack([np.mean(hrel[i,valid[i]],axis=0) if valid[i].any() else rel[i] for i in range(n)])
 sd=np.stack([np.std(hrel[i,valid[i]],axis=0) if valid[i].any() else rel[i]*0 for i in range(n)])
 rank=(rankdata(rel,axis=0,method='min')-1)/n
 radii=[5.,20.,80.];local=[]
 for radius in radii:
  w=np.exp(-np.minimum(d2/radius**2,50));w/=w.sum(1,keepdims=True)
  local.append((p-w@p)@R)
 nearest=np.clip(nearest,0,1000)
 feat=np.c_[rel/20,rel/extent,rv/5,ra/3,mu/20,sd/10,rank,
             np.log1p(nearest)[:,None],np.full((n,1),np.log1p(scale)),np.full((n,1),q['span']/4),
             *(l/20 for l in local),q['sd'][:,:3]/100,q['sd'][:,3:6]/50,
             np.full((n,1),np.log1p(n)),np.full((n,1),speed/25)]
 temporal=np.concatenate([hrel/20,np.stack(hvel,1)/5,(hrel-rel[:,None])/20,np.stack(hrank,1),seq[:,:,9:12]*10,
                          valid[:,:,None],seq[:,:,12:13],np.broadcast_to(np.linspace(-1,0,21)[None,:,None],(n,21,1)),np.full((n,21,1),q['span']/4)],2)
 temporal=np.clip(temporal,-20,20)*valid[:,:,None]
 return dict(p=p,v=v,a=acc,rel=rel,rv=rv,ra=ra,mu=mu,sd=sd,rank=rank,scale=scale,extent=extent,local=local,temporal=temporal.astype(np.float32),
             node=np.clip(feat,-20,20).astype(np.float32),center=center,vc=vc,R=R,seq=seq,valid=valid)

def features(s,need_compat=True):
 a,b=descriptors(s['a']),descriptors(s['b']);n,m=len(a['p']),len(b['p']);h=float(s['h']);sig=float(s['meta']['sigma'])
 ea=lambda z:np.broadcast_to(z[:,None],(n,m,z.shape[-1]))
 eb=lambda z:np.broadcast_to(z[None],(n,m,z.shape[-1]))
 rel=b['rel'][None]-a['rel'][:,None];vel=b['rv'][None]-a['rv'][:,None];scale=np.sqrt(a['scale']*b['scale']);ext=np.sqrt(a['extent']*b['extent'])
 coords=[rel/10,rel/(scale+sig),rel/ext,
         (b['mu'][None]-a['mu'][:,None])/10,(b['sd'][None]-a['sd'][:,None])/5,
         vel/5,(b['ra'][None]-a['ra'][:,None])/3,(b['rank'][None]-a['rank'][:,None]),
         (rel-a['rv'][:,None]*h)/(5+2*h),(rel-b['rv'][None]*h)/(5+2*h)]
 for la,lb in zip(a['local'],b['local']):coords.append((lb[None]-la[:,None])/10)
 delta=b['p'][None]-a['p'][:,None];dv=b['v'][None]-a['v'][:,None]
 coords.extend([delta/(20+20*h),dv/10,(delta-a['v'][:,None]*h)/(10+5*h),(delta-b['v'][None]*h)/(10+5*h)])
 for distance in [rel,vel,delta-a['v'][:,None]*h,delta-b['v'][None]*h]:
  coords.append(np.log1p((distance**2).sum(2))[...,None])
 coords.extend([ea(a['rel']/20),eb(b['rel']/20),ea(a['rv']/5),eb(b['rv']/5),ea(a['sd']/10),eb(b['sd']/10),
                np.broadcast_to(np.array([h/20,sig/5,np.log1p(n),np.log1p(m),np.log1p(scale),s['b']['span']/4]),(n,m,6)),
                s['pair'],s['cost'][...,None]])
 pair=np.clip(np.concatenate(coords,-1),-20,20).astype(np.float32)
 # Directed relative edge changes, useful to geometric and learned baselines.
 edge_a=(a['rel'][:,None]-a['rel'][None]);edge_b=(b['rel'][:,None]-b['rel'][None])
 # n,m,n,m ordering: candidate i-j compared with candidate k-l.
 compatible=None
 if need_compat:
  diff=edge_a[:,None,:,None,:]-edge_b[None,:,None,:,:]
  denom=4+sig+np.sqrt(np.linalg.norm(edge_a,axis=2)[:,None,:,None]*np.linalg.norm(edge_b,axis=2)[None,:,None,:])*.25+h*.1
  compatible=np.exp(-np.minimum(np.sum(diff**2,axis=-1)/(denom**2),30)).astype(np.float32)
  mask=(1-np.eye(n))[:,None,:,None]*(1-np.eye(m))[None,:,None,:]
  compatible*=mask
 return dict(pair=pair,node_a=a['node'],node_b=b['node'],history_a=a['temporal'],history_b=b['temporal'],edge_a=edge_a.astype(np.float32),edge_b=edge_b.astype(np.float32),
             compatible=compatible,rel_a=a['rel'].astype(np.float32),rel_b=b['rel'].astype(np.float32),
             target=s['target'],reverse_target=s['reverse_target'],base=s['cost'],sample=s)

def project2d(s):
 import copy
 a,b=copy.deepcopy(s['a']),copy.deepcopy(s['b'])
 for q in [a,b]:
  q['x'][:,[2,5,8]]=0;q['sd'][:,[2,5,8]]=1e-6;q['seq'][:,:,[2,5,8,11]]=0;q['static'][:,[2,5,8,11,14]]=0;q['origin'][2]=0
 out=label_sample(build_sample(a,b),s['truth_a'],s['truth_b']);out['meta']={**s['meta'],'observed_dimensions':2}
 return out

"""Short-tracklet filtering, feature construction and scoring used with recorded pigeon trajectories."""
import numpy as np

DT=.2

SCALE=10.0  # canonical metres = 10 physical metres; time is never scaled

def limit(x,bound):return x*np.minimum(1,np.asarray(bound)/np.maximum(np.linalg.norm(x,axis=-1),1e-12))[...,None]

def inherited():
    from legacy_geometry import build
    return build

def structure_features(sample):
    from legacy_features import features
    return features(sample)

def pack_observed(z,valid,times,ids,sigma,origin):
    """Causal CV filtering of an already formed short tracklet; no truth input.

    All positions in physical metres. A tracklet is one column of z. Stable
    membership inside this short window is an explicit primary-benchmark
    assumption, separately removed in the front-end stress experiment.
    """
    z=np.asarray(z);valid=np.asarray(valid,bool);nt,n,_=z.shape
    mean=np.zeros((n,6));P=np.tile(np.diag([sigma**2*4]*3+[25**2]*3),(n,1,1))
    started=np.zeros(n,bool);hist=[];last_velocity=np.zeros((n,3));acc=np.zeros((n,3))
    R=np.diag(np.array([1,2,1.2])**2*sigma**2)
    F=np.eye(6);F[:3,3:]=np.eye(3)*DT
    B=np.r_[np.eye(3)*DT**2/2,np.eye(3)*DT];Q=B@B.T*3**2+np.eye(6)*1e-8
    seen=np.zeros(n,int);first=np.full(n,nt)
    for k in range(nt):
        mean=mean@F.T;P=F@P@F.T+Q
        init=valid[k]&~started;mean[init,:3]=z[k,init];mean[init,3:]=0;first[init]=k;started|=valid[k]
        take=np.where(valid[k])[0];innov=np.zeros(n)
        if len(take):
            pp=P[take];S=pp[:,:3,:3]+R;K=pp[:,:,:3]@np.linalg.inv(S);res=z[k,take]-mean[take,:3]
            mean[take]+=np.einsum('nij,nj->ni',K,res);A=np.eye(6)[None]-K@np.eye(6)[:3]
            P[take]=A@pp@A.transpose(0,2,1)+K@R@K.transpose(0,2,1)
            innov[take]=np.sqrt(np.einsum('ni,nij,nj->n',res,np.linalg.inv(S),res))
        seen+=valid[k];dv=(mean[:,3:]-last_velocity)/DT
        acc=.8*acc+.2*limit(dv,8);acc[seen<3]=0;last_velocity=mean[:,3:].copy()
        sd=np.sqrt(np.maximum(np.diagonal(P,axis1=1,axis2=2),1e-10))
        hist.append(dict(x=np.c_[mean,acc].copy(),sd=sd,valid=started.copy(),seen=valid[k],innov=innov))
    x=hist[-1]['x']*SCALE;sd=np.c_[hist[-1]['sd']*SCALE,np.full((n,3),3*SCALE)]
    seq=np.zeros((n,21,19),np.float32)
    for j,h in enumerate(hist[-21:]):
        k=21-min(21,nt)+j;q=h['x']*SCALE;mask=h['valid'][:,None]
        seq[:,k]=np.c_[(q[:,:3]-x[:,:3])/1000,q[:,3:6]/250,q[:,6:9]/40,h['sd'][:,:3]*SCALE/1000,
                       h['seen'],h['seen'],h['innov']/20,np.ones((n,2)),h['seen'],h['valid']]*mask
    v=x[:,3:6];a=x[:,6:9];w=np.clip((v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum((v[:,:2]**2).sum(1),100),-.3,.3)
    age=np.maximum(nt-first,0)*DT
    static=np.c_[(x[:,:3]-origin)/10000,v/250,a/40,sd[:,:3]/1000,sd[:,3:6]/100,age/60,w/.15,1-valid[-1]].astype(np.float32)
    return dict(ids=np.asarray(ids,int),x=x.astype(np.float32),sd=sd.astype(np.float32),w=w.astype(np.float32),seq=seq,static=static,origin=np.asarray(origin,np.float32),t=float(times[-1]),span=float(times[-1]-times[0]))

def geometry_sample(a,b):
    # No future recovery duration or true motion class is present in context.
    return inherited()(a,b,np.zeros((2,4),bool),b['span'])

def build_sample(a,b):
    s=geometry_sample(a,b)
    f,ma,mb=structure_features(s);s['dense_features']=f;s['mass_a']=ma;s['mass_b']=mb
    return s

def label_sample(sample,left_truth,right_truth):
    """Exact source identity labels; never depend on a position gate."""
    n,m=len(left_truth),len(right_truth);lookup={int(x):j for j,x in enumerate(right_truth) if x>=0};reverse={int(x):i for i,x in enumerate(left_truth) if x>=0}
    sample['target']=np.array([lookup.get(int(x),m) if x>=0 else m for x in left_truth],np.int64)
    sample['reverse_target']=np.array([reverse.get(int(x),n) if x>=0 else n for x in right_truth],np.int64)
    sample['truth_a']=np.asarray(left_truth,int);sample['truth_b']=np.asarray(right_truth,int)
    return sample

def score_assignment(s,assignment):
    n,m=s['cost'].shape;target=s['target'];pred=np.asarray(assignment);matched=pred>=0;real=target<m
    correct=(pred==target)&matched&real;tp=int(correct.sum());fp=int((matched&~correct).sum());fn=int((real&~correct).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);used=set(pred[matched].tolist());new=np.where((s['reverse_target']>=n)&(s['truth_b']>=0))[0]
    true_exits=(~real)&(s['truth_a']>=0)
    union_true=len(set(s['truth_a'][s['truth_a']>=0])|set(s['truth_b'][s['truth_b']>=0]));union_est=n+m-int(matched.sum())
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=2*tp/max(2*tp+fp+fn,1),union_count_error=abs(union_est-union_true),union_count_true=union_true,union_count_estimated=union_est,current_count_error=int(np.sum(s['truth_b']<0)),
                retention=recall,correct=tp,continuing=int(real.sum()),old_count=n,current_count=m,
                null_correct=int(((~matched)&(~real)).sum()),null_total=int((~real).sum()),
                new_stolen=sum(int(j in used) for j in new),new_total=len(new),true_exit_rejected=int(((~matched)&true_exits).sum()),true_exit_total=int(true_exits.sum()),false_current_matched=sum(int(j in used) for j in np.where(s['truth_b']<0)[0]),assignments=int(matched.sum()))

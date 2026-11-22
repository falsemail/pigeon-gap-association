"""Non-learned, development-tuned partial assignment controls.

The bounded k-best method is tracklet association with a structural reranker,
not a claim to reproduce a full birth/death MHT filtering implementation.
"""
import heapq,itertools,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from core import inherited,SCALE
inherited()
from legacy_geometry import assignment,ct

def solve(cost,null):
    return assignment(np.asarray(cost,np.float64),null)

def components(s,only=None):
    a,b=s['a'],s['b'];p,v,acc=np.split(a['x'][:,:9],3,axis=1);q,u,bc=np.split(b['x'][:,:9],3,axis=1);h=float(s['h'])
    norm=(30+5.5*h)*SCALE
    cv=np.log1p(((p[:,None]+v[:,None]*h-q[None])**2).sum(2)/norm**2)
    if only=='cv':return {'cv':cv}
    if only=='geometry':return {'geometry':s['cost'].astype(float)}
    # Gaussian mixture of CV, exponentially damped acceleration, and turn
    # extrapolation, scored against every causally available recovery state.
    good=b['seq'][:,:,18]>.5
    ts=np.arange(21)*.2;ts-=ts[-1]
    z=b['seq'][:,:,:3]*1000+q[:,None]
    models=[]
    for mode in (range(3) if only not in ('structure','kbest') else []):
        energy=np.zeros((len(p),len(q)));weight=np.zeros(len(q))
        for k in np.where(good.any(0))[0]:
            hk=max(.0,h+ts[k]);w=good[:,k].astype(float)
            if mode==0:r=p+v*hk
            elif mode==1:r=p+v*hk+acc*(3*hk-9*(1-np.exp(-hk/3)))
            else:r=ct(p,v,a['w'],hk)
            var=(a['sd'][:,:3]**2)[:,None]+(b['sd'][:,:3]**2)[None]+(a['sd'][:,3:6]**2*hk**2)[:,None]+(SCALE*(3+1.5*hk))**2
            e=.5*np.sum((r[:,None]-z[None,:,k])**2/var+np.log(var/SCALE**2),2)
            energy+=np.minimum(e,80)*w;weight+=w
        models.append(energy/np.maximum(weight,1))
    mm=-logsumexp(-np.stack(models),axis=0)+np.log(3) if models else np.zeros_like(cv)
    mm+=.15*np.sum((v[:,None]-u[None])**2,2)/(15*SCALE)**2
    if only=='multi_model':return {'multi_model':mm}
    # Observed local relative coordinates; no true groups are supplied.
    from legacy_features import neighborhood
    ga,gb=neighborhood(a),neighborhood(b)
    dr=ga['local'][:,None]-gb['local'][None]
    sd=(ga['sd'][:,None]+gb['sd'][None])/2
    struct=np.log1p(np.sum(dr**2/(sd**2+(5*SCALE)**2),2))
    rv=(v-ga['center'][:,3:6])[:,None]-(u-gb['center'][:,3:6])[None]
    struct+=.15*np.log1p(np.sum(rv**2,2)/(5*SCALE)**2)
    return {'cv':cv,'multi_model':mm,'geometry':s['cost'].astype(float),'structure':struct}

def augment(cost,null):
    n,m=cost.shape;dummy=np.full((n,n),1e10);np.fill_diagonal(dummy,np.broadcast_to(null,n));return np.c_[cost,dummy]

def kbest(cost,null,k=8):
    """Murty partitioning with disjoint fixed-prefix / forbidden-edge nodes.

    Returns exactly ranked distinct assignments for finite rectangular costs;
    each row has its own null column. Ties follow SciPy's deterministic order.
    """
    C=augment(cost,null);n,m0=cost.shape;counter=itertools.count();heap=[]
    def push(fixed,banned):
        W=C.copy()
        for i,j in banned:W[i,j]=1e10
        for i,j in fixed:W[i,:]=1e10;W[:,j]=1e10;W[i,j]=C[i,j]
        ii,jj=linear_sum_assignment(W)
        if len(ii)<n or np.any(W[ii,jj]>=1e9):return
        a=np.empty(n,int);a[ii]=jj;value=float(C[ii,jj].sum())
        heapq.heappush(heap,(value,next(counter),a,tuple(fixed),frozenset(banned)))
    push((),frozenset());out=[]
    while heap and len(out)<k:
        value,_,a,fixed,banned=heapq.heappop(heap);out.append((np.where(a<m0,a,-1),value))
        prefix=list(fixed);fixedrows={i for i,j in fixed}
        for i in range(n):
            if i in fixedrows:continue
            push(prefix,banned|{(i,int(a[i]))});prefix.append((i,int(a[i])))
    return out

def relation_energy(s,matched):
    rows=np.where(matched>=0)[0]
    if len(rows)<2:return 0.
    a=s['a']['x'][rows,:3]/SCALE;b=s['b']['x'][matched[rows],:3]/SCALE
    da=np.linalg.norm(a[:,None]-a[None],axis=2);db=np.linalg.norm(b[:,None]-b[None],axis=2)
    w=np.exp(-(da/80)**2);np.fill_diagonal(w,0)
    return float(np.sum(w*np.minimum(np.log((db+3)/(da+3))**2,4))/max(len(rows),1))

def predict(s,cfg,parts=None):
    n,m=s['cost'].shape
    if not n or not m:return np.full(n,-1,int)
    kind=cfg['method'];c=parts or components(s,only=kind);null=cfg.get('null',3.)
    if kind=='cv':cost=c['cv']
    elif kind=='multi_model':cost=c['multi_model']*cfg.get('scale',1.)
    else:cost=c['geometry']+cfg.get('structure',0)*c['structure']
    cost=np.where(s['reachable'],cost,1e6)
    if kind=='kbest':
        candidates=kbest(cost,null,cfg.get('k',8))
        return min(candidates,key=lambda z:z[1]+cfg.get('rerank',1.)*relation_energy(s,z[0]))[0]
    return solve(cost,null)

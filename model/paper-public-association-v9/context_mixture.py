"""Set-conditioned mixture association and partial transport ablation.

Cross-attention / dustbin transport follow the SuperGlue design principle;
the motion-input architecture is an original adaptation, not its image system.
The conditional mixture foundation is Bishop1994 and our frozen V6 model.
"""
from pathlib import Path
import sys,numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'paper-public-association-v6'));sys.path.insert(0,str(ROOT/'paper-public-association-v3'))
from mixture_bridge import MixtureBridge,inputs as original_inputs,auxiliary
from learners import Cross,mlp

def context(raw,reverse=False):
    aa=raw['b'] if reverse else raw['a'];bb=raw['a'] if reverse else raw['b'];h=float(raw['h'])*(-1 if reverse else 1)
    sets=[]
    for q in [aa,bb]:
        x=np.asarray(q['x'],float)/10;v=np.median(x[:,3:6],0);speed=np.linalg.norm(v[:2]);u=v[:2]/max(speed,1e-5)
        if speed<1e-5:u=np.array([1.,0.])
        R=np.array([[u[0],-u[1],0],[u[1],u[0],0],[0,0,1.]])
        sets.append(dict(v=v,c=np.median(x[:,:3],0),R=R,n=len(x)))
    a,b=sets;ua=a['R'][:2,0];ub=b['R'][:2,0]
    c=[h/20,float(np.median(bb['sd'][:,:3]))/50,np.log1p(a['n']),np.log1p(b['n']),aa['span']/4,bb['span']/4,float(ua@ub),float(ua[0]*ub[1]-ua[1]*ub[0])]
    c.extend((b['v']-a['v'])@a['R']/10);c.extend((b['c']-a['c']-a['v']*h)@a['R']/(20+10*abs(h)))
    return c+([0.,1.] if reverse else [1.,0.])

def inputs(samples,device,corrected=True):
    s=original_inputs(samples,device)
    if corrected:
        s['context_a']=torch.tensor(np.array([context(q['sample']) for q in samples],np.float32),device=device)
        s['context_b']=torch.tensor(np.array([context(q['sample'],True) for q in samples],np.float32),device=device)
    return s

def partial_transport(L,nr,nc,ma,mb,iterations=15):
    B,N,M=L.shape
    z=torch.cat([torch.cat([L,nr[...,None]],2),torch.cat([nc[:,None,:],L.new_zeros(B,1,1)],2)],1)
    maska=torch.cat([ma,torch.ones(B,1,dtype=torch.bool,device=L.device)],1);maskb=torch.cat([mb,torch.ones(B,1,dtype=torch.bool,device=L.device)],1)
    z=z.masked_fill(~(maska[:,:,None]&maskb[:,None,:]),-1e4);n=ma.sum(1);m=mb.sum(1);total=(n+m).float()
    mu=torch.cat([ma.float(),m[:,None]],1)/total[:,None];nu=torch.cat([mb.float(),n[:,None]],1)/total[:,None]
    lm=mu.clamp_min(1e-30).log();ln=nu.clamp_min(1e-30).log();u=torch.zeros_like(lm);v=torch.zeros_like(ln)
    for _ in range(iterations):u=lm-torch.logsumexp(z+v[:,None,:],2);v=ln-torch.logsumexp(z+u[:,:,None],1)
    return z+u[:,:,None]+v[:,None,:]+total.log()[:,None,None]

class ContextMixture(MixtureBridge):
    def __init__(self,attention=False,transport=False):
        super().__init__();self.attention=attention;self.use_transport=transport
        if attention:
            self.blocks=nn.ModuleList([Cross(64) for _ in range(2)])
            self.contextual_null=mlp([144,64,32,1])
    def forward(self,s):
        if not self.attention:return super().forward(s)
        a=self.encode(s['history_a'],s['a']);b=self.encode(s['history_b'],s['b']);ma,mb=s['ma'],s['mb']
        for block in self.blocks:a,b=block(a,b,ma,mb)
        pa=self.params(a,b,mb,s['a'],s['context_a']);pb=self.params(b,a,ma,s['b'],s['context_b'])
        fa=self.density(pa,s['b'],s['coordinates']);fb=self.density(pb,s['a'],s['coordinates']).transpose(1,2)
        L=(fa+fb)*.5+torch.nn.functional.softplus(self.mix)*self.pair(s['pair']).squeeze(-1)
        L=L.masked_fill(~(ma[:,:,None]&mb[:,None,:]),-1e4)
        ga=(a*ma[...,None]).sum(1)/ma.sum(1)[:,None];gb=(b*mb[...,None]).sum(1)/mb.sum(1)[:,None]
        nr=self.contextual_null(torch.cat([a,gb[:,None].expand_as(a),s['context_a'][:,None].expand(-1,a.shape[1],-1)],-1)).squeeze(-1)
        nc=self.contextual_null(torch.cat([b,ga[:,None].expand_as(b),s['context_b'][:,None].expand(-1,b.shape[1],-1)],-1)).squeeze(-1)
        o=dict(logits=L,rownull=nr,colnull=nc,forward_density=fa,backward_density=fb)
        if self.use_transport:
            Z=partial_transport(L,nr,nc,ma,mb);o.update(logits=Z[:,:-1,:-1],rownull=Z[:,:-1,-1],colnull=Z[:,-1,:-1],transport=Z)
        return o

"""Explicit, motion-input adaptations of association families.

SuperGlue-style self/cross attention + dustbin optimal transport is implemented
from the paper's equations; it is not its image descriptor/pretrained system.
MPN-style directed row/column edge messages adapt the time-aware bipartite
message-passing principle to the two-window problem. All modifications are
listed in the comparison matrix, rather than called exact end-to-end trackers.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def mlp(dims):
 layers=[]
 for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
  layers.append(nn.Linear(a,b))
  if i<len(dims)-2:layers.append(nn.SiLU())
 return nn.Sequential(*layers)

def batch(samples,device,need_compat=True):
 B=len(samples);N=max(q['pair'].shape[0] for q in samples);M=max(q['pair'].shape[1] for q in samples);D=samples[0]['pair'].shape[-1];C=samples[0]['node_a'].shape[-1]
 arrays=dict(pair=np.zeros((B,N,M,D),np.float32),a=np.zeros((B,N,C),np.float32),b=np.zeros((B,M,C),np.float32),ma=np.zeros((B,N),bool),mb=np.zeros((B,M),bool),target=np.full((B,N),M,np.int64),reverse=np.full((B,M),N,np.int64))
 if need_compat:arrays['compat']=np.zeros((B,N,M,N,M),np.float32)
 for k,q in enumerate(samples):
  n,m=q['pair'].shape[:2];arrays['pair'][k,:n,:m]=q['pair'];arrays['a'][k,:n]=q['node_a'];arrays['b'][k,:m]=q['node_b'];arrays['ma'][k,:n]=True;arrays['mb'][k,:m]=True
  if need_compat:arrays['compat'][k,:n,:m,:n,:m]=q['compatible']
  arrays['target'][k,:n]=np.where(q['target']<m,q['target'],M);arrays['reverse'][k,:m]=np.where(q['reverse_target']<n,q['reverse_target'],N)
 arrays['history_a']=np.zeros((B,N,21,19),np.float32);arrays['history_b']=np.zeros((B,M,21,19),np.float32)
 for k,q in enumerate(samples):
  n,m=q['pair'].shape[:2];arrays['history_a'][k,:n]=q['history_a'];arrays['history_b'][k,:m]=q['history_b']
 result={k:torch.as_tensor(v,device=device) for k,v in arrays.items()};result['_samples']=[{'pair':q['pair'],'sample':{k:q['sample'][k] for k in ['a','b','h','context','cost','pair','reachable','forward','backward','dense_features']}} for q in samples]
 return result

def transport(scores,ma,mb,bin_score,iterations=12):
 """Log Sinkhorn with capacity dustbins for variable-size partial matches."""
 B,N,M=scores.shape;device=scores.device
 z=bin_score.expand(B,N+1,M+1).clone();z[:,:N,:M]=scores
 maska=torch.cat([ma,torch.ones((B,1),dtype=torch.bool,device=device)],1);maskb=torch.cat([mb,torch.ones((B,1),dtype=torch.bool,device=device)],1)
 z=z.masked_fill(~(maska[:,:,None]&maskb[:,None,:]),-1e4)
 n=ma.sum(1);m=mb.sum(1);total=(n+m).float()
 mu=torch.cat([ma.float(),m[:,None].float()],1)/total[:,None];nu=torch.cat([mb.float(),n[:,None].float()],1)/total[:,None]
 lm=torch.log(mu.clamp_min(1e-30));ln=torch.log(nu.clamp_min(1e-30));u=torch.zeros_like(lm);v=torch.zeros_like(ln)
 for _ in range(iterations):
  u=lm-torch.logsumexp(z+v[:,None,:],2);v=ln-torch.logsumexp(z+u[:,:,None],1)
 return z+u[:,:,None]+v[:,None,:]+torch.log(total)[:,None,None]

class Cross(nn.Module):
 def __init__(self,d=64):
  super().__init__();self.selfatt=nn.MultiheadAttention(d,4,batch_first=True);self.cross=nn.MultiheadAttention(d,4,batch_first=True);self.norm=nn.LayerNorm(d);self.update=mlp([2*d,2*d,d])
 def forward(self,a,b,ma,mb):
  aa=self.selfatt(a,a,a,key_padding_mask=~ma,need_weights=False)[0];bb=self.selfatt(b,b,b,key_padding_mask=~mb,need_weights=False)[0];a=self.norm(a+self.update(torch.cat([a,aa],-1)));b=self.norm(b+self.update(torch.cat([b,bb],-1)))
  aa=self.cross(a,b,b,key_padding_mask=~mb,need_weights=False)[0];bb=self.cross(b,a,a,key_padding_mask=~ma,need_weights=False)[0]
  return self.norm(a+self.update(torch.cat([a,aa],-1))),self.norm(b+self.update(torch.cat([b,bb],-1)))

class Matcher(nn.Module):
 def __init__(self,pair_dim,node_dim,kind='relational'):
  super().__init__();self.kind=kind;self.pair_dim=pair_dim;self.node_dim=node_dim
  self.pair=mlp([pair_dim,128,64,1]);self.null=mlp([node_dim,64,1]);self.node=mlp([node_dim,64,64]);self.bin=nn.Parameter(torch.tensor(1.))
  if kind in ['superglue','relational_attention']:self.blocks=nn.ModuleList([Cross() for _ in range(4)]);self.proj=nn.Linear(64,64)
  if kind=='mpn':
   self.edge=mlp([pair_dim,64,64]);self.messages=nn.ModuleList([mlp([64*5,128,64]) for _ in range(3)]);self.head=mlp([64,32,1])
  if kind in ['relational','relational_attention','no_geometry']:
   self.refine=mlp([pair_dim+5,64,32,1]);self.weight=nn.Parameter(torch.tensor(.0))
 def forward(self,s):
  x=s['pair'];ma,mb=s['ma'],s['mb'];mask=ma[:,:,None]&mb[:,None,:];na=self.null(s['a']).squeeze(-1);nb=self.null(s['b']).squeeze(-1)
  if self.kind=='superglue':
   a,b=self.node(s['a']),self.node(s['b'])
   for block in self.blocks:a,b=block(a,b,ma,mb)
   L=torch.einsum('bid,bjd->bij',self.proj(a),self.proj(b))/8
   Z=transport(L,ma,mb,self.bin)
   return dict(logits=Z[:,:-1,:-1],rownull=Z[:,:-1,-1],colnull=Z[:,-1,:-1],transport=Z)
  elif self.kind=='mpn':
   e0=self.edge(x);e=e0
   for layer in self.messages:
    row=(e*mask[...,None]).sum(2)/mb.sum(1)[:,None,None];col=(e*mask[...,None]).sum(1)/ma.sum(1)[:,None,None]
    rm=e.masked_fill(~mask[...,None],-1e4).max(2).values;cm=e.masked_fill(~mask[...,None],-1e4).max(1).values
    e=layer(torch.cat([e0,row[:,:,None].expand_as(e),col[:,None].expand_as(e),rm[:,:,None].expand_as(e),cm[:,None].expand_as(e)],-1))
   L=self.head(e).squeeze(-1)
  else:L=self.pair(x).squeeze(-1)
  if self.kind=='relational_attention':
   a,b=self.node(s['a']),self.node(s['b'])
   for block in self.blocks:a,b=block(a,b,ma,mb)
   L=L+torch.einsum('bid,bjd->bij',self.proj(a),self.proj(b))/8
  L=L.masked_fill(~mask,-1e4)
  if self.kind in ['relational','relational_attention','no_geometry']:
   base=L
   for _ in range(3):
    P=transport(L,ma,mb,self.bin,8)[:,:-1,:-1].exp()*mask
    C=s['compat'] if self.kind!='no_geometry' else s['compat']*0
    agree=torch.einsum('bijkl,bkl->bij',C,P)/P.sum((1,2))[:,None,None].clamp_min(1)
    rowmax=L.max(2,keepdim=True).values;colmax=L.max(1,keepdim=True).values
    ctx=torch.stack([agree,P,L-rowmax,L-colmax,L.clamp(-20,20)],-1)
    L=(base+3*torch.tanh(self.refine(torch.cat([x,ctx],-1)).squeeze(-1)/3)).masked_fill(~mask,-1e4)
  return dict(logits=L,rownull=na,colnull=nb)

def objective(o,s):
 if 'transport' in o:
  Z=o['transport'];B,N,M=o['logits'].shape
  r=-torch.gather(Z[:,:N,:],2,s['target'][...,None]).squeeze(-1);c=-torch.gather(Z[:,:,:M].transpose(1,2),2,s['reverse'][...,None]).squeeze(-1)
  return .5*((r*s['ma']).sum()/s['ma'].sum()+(c*s['mb']).sum()/s['mb'].sum())
 L=o['logits'];r=F.cross_entropy(torch.cat([L,o['rownull'][...,None]],2).transpose(1,2),s['target'],reduction='none');c=F.cross_entropy(torch.cat([L.transpose(1,2),o['colnull'][...,None]],2).transpose(1,2),s['reverse'],reduction='none')
 return .5*((r*s['ma']).sum()/s['ma'].sum()+(c*s['mb']).sum()/s['mb'].sum())

def assignment(o,k,n,m,threshold=0.):
 from scipy.optimize import linear_sum_assignment
 L=o['logits'][k,:n,:m].detach().cpu().numpy();nr=o['rownull'][k,:n].detach().cpu().numpy();nc=o['colnull'][k,:m].detach().cpu().numpy()
 # Symmetric conditional log odds. Adding the same constant to every pair
 # and null logit leaves both conditional CE and these odds unchanged.
 # A one-edge versus two-null raw-logit objective lacks that invariance.
 cost=-L+.5*(nr[:,None]+nc[None]);dummy=np.full((n,n),1e5);np.fill_diagonal(dummy,-threshold);ii,jj=linear_sum_assignment(np.c_[cost,dummy]);a=np.full(n,-1);a[ii[jj<m]]=jj[jj<m];return a

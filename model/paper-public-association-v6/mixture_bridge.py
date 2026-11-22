"""Bidirectional conditional mixture endpoint model, development candidate.

Mixture density principle: Bishop (1994). Motion-association use is our own
implementation, not a reproduction of TrajE. Supervision is correspondence
between noisy OBSERVED endpoints, including 2D; no unobserved height is filled.
The returned densities are conditional scores, not a joint tracking posterior.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'paper-public-association-v3'))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from learners import batch,mlp
from bridge_model import observed_summary

def inputs(samples,device):
 s=batch(samples,device,need_compat=False);ca=[];cb=[];dims=[]
 for q in samples:
  raw=q['sample'];ca.append(observed_summary(raw)[0]+[1.,0.])
  cb.append(observed_summary(dict(a=raw['b'],b=raw['a'],h=raw['h']))[0]+[0.,1.])
  dims.append([1.,1.,float(not (np.max(raw['a']['sd'][:,2])<1e-5 and np.max(raw['b']['sd'][:,2])<1e-5))])
 s['context_a']=torch.tensor(np.asarray(ca,np.float32),device=device)
 s['context_b']=torch.tensor(np.asarray(cb,np.float32),device=device)
 s['coordinates']=torch.tensor(np.asarray(dims,np.float32),device=device)
 return s

class MixtureBridge(nn.Module):
 def __init__(self,components=4,residual=True,bidirectional=True):
  super().__init__();self.components=components;self.residual=residual;self.bidirectional=bidirectional
  self.gru=nn.GRU(19,32,batch_first=True);self.encode_node=mlp([73,64,64])
  self.predict=mlp([144,128,64,7*components]);self.null=mlp([64,32,1]);self.pair=mlp([91,128,64,1]);self.mix=nn.Parameter(torch.tensor(-1.))
 def encode(self,h,x):
  B,N,T,D=h.shape;_,z=self.gru(h.reshape(B*N,T,D));return self.encode_node(torch.cat([z[0].reshape(B,N,32),x],-1))
 def params(self,a,b,mb,node,context):
  pool=(b*mb[...,None]).sum(1)/mb.sum(1)[:,None]
  z=self.predict(torch.cat([a,pool[:,None].expand_as(a),context[:,None].expand(-1,a.shape[1],-1)],-1)).reshape(a.shape[0],a.shape[1],self.components,7)
  mean=node[:,:,:3,None].transpose(-1,-2)+z[...,1:4]
  var=.0025+F.softplus(z[...,4:])**2
  return mean,var,F.log_softmax(z[...,0],-1)
 @staticmethod
 def density(params,target,coords):
  mean,var,weight=params
  # B,N,M,K,D: old distribution evaluated at each recovered endpoint.
  delta=target[:,None,:,None,:3]-mean[:,:,None]
  lp=-.5*((delta.square()/var[:,:,None]+var[:,:,None].log()+np.log(2*np.pi))*coords[:,None,None,None,:]).sum(-1)
  return torch.logsumexp(lp+weight[:,:,None],-1)
 def forward(self,s):
  a=self.encode(s['history_a'],s['a']);b=self.encode(s['history_b'],s['b'])
  pa=self.params(a,b,s['mb'],s['a'],s['context_a']);pb=self.params(b,a,s['ma'],s['b'],s['context_b'])
  fa=self.density(pa,s['b'],s['coordinates']);fb=self.density(pb,s['a'],s['coordinates']).transpose(1,2)
  L=(fa+fb)*.5 if self.bidirectional else fa
  if self.residual:L=L+F.softplus(self.mix)*self.pair(s['pair']).squeeze(-1)
  L=L.masked_fill(~(s['ma'][:,:,None]&s['mb'][:,None,:]),-1e4)
  return dict(logits=L,rownull=self.null(a).squeeze(-1),colnull=self.null(b).squeeze(-1),forward_density=fa,backward_density=fb)

def auxiliary(o,s):
 # Identity is read only here, in supervised loss; never in model.forward.
 N,M=o['logits'].shape[1:];va=s['ma']&(s['target']<M);vb=s['mb']&(s['reverse']<N)
 la=-o['forward_density'].gather(2,s['target'].clamp_max(M-1)[...,None]).squeeze(-1)
 lb=-o['backward_density'].transpose(1,2).gather(2,s['reverse'].clamp_max(N-1)[...,None]).squeeze(-1)
 return .5*((la*va).sum()/va.sum().clamp_min(1)+(lb*vb).sum()/vb.sum().clamp_min(1))

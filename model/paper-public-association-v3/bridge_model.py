"""Observed-window-conditioned stochastic endpoint prediction for association.

Prediction is conditioned on the recovered SET summary, not hidden identities.
The auxiliary future position is training supervision only. This is our own
motion-input candidate, not a reproduction of a named trajectory predictor.
"""
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from learners import mlp,batch
def observed_summary(raw):
 out=[]
 for key in ['a','b']:
  x=np.asarray(raw[key]['x'],float)/10;p=x[:,:3];v=np.median(x[:,3:6],axis=0);speed=np.linalg.norm(v[:2]);u=v[:2]/max(speed,1e-5)
  if speed<1e-5:u=np.array([1.,0.])
  R=np.array([[u[0],-u[1],0],[u[1],u[0],0],[0,0,1.]])
  out.append(dict(p=p,vc=v,center=np.median(p,axis=0),R=R))
 a,b=out;h=float(raw['h']);ua=a['R'][:2,0];ub=b['R'][:2,0]
 c=[h/20,float(np.median(raw['b']['sd'][:,:3]))/50,np.log1p(len(a['p'])),np.log1p(len(b['p'])),raw['a']['span']/4,raw['b']['span']/4,float(ua@ub),float(ua[0]*ub[1]-ua[1]*ub[0])]
 c.extend((b['vc']-a['vc'])/10);c.extend((b['center']-a['center']-a['vc']*h)/(20+10*h));return c,b

def bridge_batch(samples,device,supervision=False):
 s=batch(samples,device,need_compat=False);B,N=s['ma'].shape;context=[];target=np.zeros((B,N,3),np.float32);valid=np.zeros((B,N),bool)
 for i,q in enumerate(samples):
  raw=q['sample'];c,b=observed_summary(raw);context.append(c)
  if supervision and 'forward_target' in raw:
   n=len(raw['a']['x']);target[i,:n]=((raw['forward_target']/10-b['center'])@b['R']/20);valid[i,:n]=raw['forward_valid']
 s['bridge_context']=torch.tensor(np.asarray(context,np.float32),device=device)
 if supervision:s['endpoint_target']=torch.tensor(target,device=device);s['endpoint_valid']=torch.tensor(valid,device=device)
 return s

class BridgeMatcher(nn.Module):
 def __init__(self,kind='bridge'):
  super().__init__();self.kind=kind;self.pair_dim=91;self.node_dim=41
  self.gru=nn.GRU(19,32,batch_first=True);self.embed=mlp([73,64,64]);self.predictor=mlp([142,128,64,6]);self.null=mlp([64,32,1]);self.pair=mlp([91,128,64,1]);self.mix=nn.Parameter(torch.tensor(-1.))
 def encode(self,h,node):
  B,N,T,D=h.shape;_,z=self.gru(h.reshape(B*N,T,D));return self.embed(torch.cat([z[0].reshape(B,N,32),node],-1))
 def forward(self,s):
  if 'bridge_context' not in s:s={**s,'bridge_context':torch.tensor(np.asarray([observed_summary(q['sample'])[0] for q in s['_samples']],np.float32),device=s['pair'].device)}
  ma,mb=s['ma'],s['mb'];a=self.encode(s['history_a'],s['a']);b=self.encode(s['history_b'],s['b']);global_b=(b*mb[...,None]).sum(1)/mb.sum(1)[:,None]
  x=torch.cat([a,global_b[:,None].expand_as(a),s['bridge_context'][:,None].expand(-1,a.shape[1],-1)],-1);pred=self.predictor(x);mu=s['a'][:,:,:3]+pred[:,:,:3];variance=.0025+F.softplus(pred[:,:,3:])**2
  residual=s['b'][:,None,:,:3]-mu[:,:,None];L=-.5*(residual.square()/variance[:,:,None]+variance[:,:,None].log()).sum(-1)
  # A projected 2D observation has explicitly absent z uncertainty; it must
  # not become an artificial, perfectly measured height cue in the density.
  z_available=torch.tensor([not (np.max(q['sample']['a']['sd'][:,2])<1e-5 and np.max(q['sample']['b']['sd'][:,2])<1e-5) for q in s['_samples']],device=L.device)
  coordinate_mask=torch.ones((len(z_available),3),device=L.device);coordinate_mask[:,2]=z_available
  density=(residual.square()/variance[:,:,None]+variance[:,:,None].log());L=-.5*(density*coordinate_mask[:,None,None,:]).sum(-1)
  if self.kind!='bridge_prediction_only':L=L+F.softplus(self.mix)*self.pair(s['pair']).squeeze(-1)
  L=L.masked_fill(~(ma[:,:,None]&mb[:,None,:]),-1e4)
  return dict(logits=L,rownull=self.null(a).squeeze(-1),colnull=self.null(b).squeeze(-1),endpoint_mean=mu,endpoint_variance=variance)

def endpoint_loss(o,s):
 z=.5*((s['endpoint_target']-o['endpoint_mean']).square()/o['endpoint_variance']+o['endpoint_variance'].log()).sum(-1)
 return (z*s['endpoint_valid']).sum()/s['endpoint_valid'].sum().clamp_min(1)

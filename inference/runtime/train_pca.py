"""PCA-GM with the frozen motion-node and dummy-node adaptation."""
import torch
from torch import nn
import pygmtools as pygm

class PCAMatcher(nn.Module):
 def __init__(self,node_dim=41):
  super().__init__();self.node_dim=node_dim
  x=torch.zeros((1,2,node_dim));A=torch.eye(2)[None]
  _,self.network=pygm.pca_gm(x,x,A,A,in_channel=node_dim,hidden_channel=64,out_channel=64,num_layers=3,sk_max_iter=20,sk_tau=.1,return_network=True,pretrain=False,backend='pytorch')
  self.dummy=nn.Parameter(torch.zeros(node_dim))
 def forward(self,s):
  ma,mb=s['ma'],s['mb'];B,N=ma.shape;M=mb.shape[1];K=N+M
  # Each graph has K real-or-dummy slots; masked padding is a dummy, not a label.
  a=self.dummy.expand(B,K,-1).clone();b=a.clone();a[:,:N]=torch.where(ma[...,None],s['a'],self.dummy);b[:,:M]=torch.where(mb[...,None],s['b'],self.dummy)
  A=torch.eye(K,device=a.device)[None].repeat(B,1,1);C=A.clone()
  pa=s['a'][:,:,:3]*20;pb=s['b'][:,:,:3]*20
  A[:,:N,:N]=torch.exp(-torch.cdist(pa,pa).square()/400)*ma[:,:,None]*ma[:,None,:]+torch.diag_embed((~ma).float())
  C[:,:M,:M]=torch.exp(-torch.cdist(pb,pb).square()/400)*mb[:,:,None]*mb[:,None,:]+torch.diag_embed((~mb).float())
  counts=ma.sum(1)+mb.sum(1)
  P=pygm.pca_gm(a,b,A,C,n1=counts,n2=counts,in_channel=self.node_dim,hidden_channel=64,out_channel=64,num_layers=3,sk_max_iter=20,sk_tau=.1,network=self.network,pretrain=False,backend='pytorch')
  dm_b=torch.cat([~mb,torch.ones((B,N),dtype=torch.bool,device=a.device)],1);dm_a=torch.cat([~ma,torch.ones((B,M),dtype=torch.bool,device=a.device)],1)
  nr=(P[:,:N]*dm_b[:,None]).sum(2);nc=(P[:,:,:M]*dm_a[:,:,None]).sum(1)
  L=P[:,:N,:M].clamp_min(1e-9).log().masked_fill(~(ma[:,:,None]&mb[:,None]),-1e4)
  return dict(logits=L,rownull=nr.clamp_min(1e-9).log(),colnull=nc.clamp_min(1e-9).log())

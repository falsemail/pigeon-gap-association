"""Permutation-equivariant temporal role descriptors and relational matching."""
import torch
from torch import nn
import torch.nn.functional as F
from learners import mlp,Cross,transport
class TemporalMatcher(nn.Module):
 def __init__(self,pair_dim=91,node_dim=41,kind='temporal_graph'):
  super().__init__();self.kind=kind;self.pair_dim=pair_dim;self.node_dim=node_dim
  self.gru=nn.GRU(19,32,batch_first=True);self.node=mlp([node_dim,32,32]);self.join=mlp([64,64,64]);self.blocks=nn.ModuleList([Cross() for _ in range(2)])
  self.proj=nn.Linear(64,32);self.base=mlp([pair_dim,128,64,1]);self.residual=mlp([64,64,32,1]);self.null=mlp([64,32,1]);self.bin=nn.Parameter(torch.tensor(1.));self.graph_strength=nn.Parameter(torch.tensor(0.));self.dropout=nn.Dropout(.15)
  nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
 def encode(self,h,node):
  B,N,T,D=h.shape;_,z=self.gru(h.reshape(B*N,T,D));z=z[0].reshape(B,N,-1)
  return self.join(torch.cat([z,self.node(node)],-1))
 def forward(self,s):
  ma,mb=s['ma'],s['mb'];mask=ma[:,:,None]&mb[:,None,:];a=self.encode(s['history_a'],s['a']);b=self.encode(s['history_b'],s['b'])
  if self.kind!='temporal':
   for layer in self.blocks:a,b=layer(a,b,ma,mb)
  aa,bb=self.proj(a),self.proj(b);pair=torch.cat([aa[:,:,None]*bb[:,None],(aa[:,:,None]-bb[:,None]).abs()],-1)
  L=self.base(s['pair']).squeeze(-1)+self.residual(self.dropout(pair)).squeeze(-1)
  L=L.masked_fill(~mask,-1e4)
  if self.kind=='temporal_graph':
   base=L
   for _ in range(3):
    P=transport(L,ma,mb,self.bin,12)[:,:-1,:-1].exp()*mask
    agree=torch.einsum('bijkl,bkl->bij',s['compat'],P)/P.sum((1,2))[:,None,None].clamp_min(1)
    L=(base+F.softplus(self.graph_strength)*4*agree).masked_fill(~mask,-1e4)
  return dict(logits=L,rownull=self.null(a).squeeze(-1),colnull=self.null(b).squeeze(-1))

"""The published point adapter for the pinned author TWiX core."""
import numpy as np
import torch
from torch import nn
from twix_vendor import twix as author

class NormPoints(nn.Module):
    def forward(self,z):
        low=torch.where(torch.isnan(z),torch.inf,z).amin(0,keepdim=True)
        high=torch.where(torch.isnan(z),-torch.inf,z).amax(0,keepdim=True)
        span=high-low;good=torch.isfinite(span)&(span>0)
        return torch.where(good,2*(z-low)/torch.where(good,span,torch.ones_like(span))-1,torch.zeros_like(z))

class PointTWiX(author.TWiX):
    def __init__(self,layers=1,inter_pair=True,device='cuda'):
        super().__init__(32,16,32,layers,.1,inter_pair=inter_pair,device=device)
        self.norm_coordinates=NormPoints();self.projection=nn.Linear(3,32)
        self.to(device)

def trajectory_input(q,device='cuda'):
    # Read only observed state/history/timestamps. No IDs, labels or metadata.
    sample=q['sample'];origin=np.median(np.asarray(sample['a']['x'][:,:3],np.float64)/10,0)
    time_origin=float(sample['b']['t']);out=[]
    for side in ['a','b']:
        a=sample[side];seq=np.asarray(a['seq']);valid=seq[:,:,18]>.5
        xyz=seq[:,:,:3].astype(np.float64)*100+np.asarray(a['x'][:,:3],np.float64)[:,None]/10-origin
        frame=np.broadcast_to((float(a['t'])-time_origin+np.arange(-20,1)*.2)*5,valid.shape).copy()
        xyz[~valid]=np.nan;frame[~valid]=np.nan
        # Author time encoding needs at least one valid target in the first
        # row of the new window; remove common leading/trailing padding.
        ix=np.flatnonzero(valid.any(0));assert len(ix)>0
        out.extend([torch.tensor(xyz[:,ix[0]:ix[-1]+1].transpose(1,0,2),dtype=torch.float32,device=device),torch.tensor(frame[:,ix[0]:ix[-1]+1].T[...,None],dtype=torch.float32,device=device)])
    return (*out,5.)

"""Physical anchors for a learned conditional mixture, a redevelopment pilot.

P/E coordinated-turning hypotheses follow Yomosa et al. (2015), Fig.7,
https://doi.org/10.1371/journal.pone.0140558 . The neural residual/variance and
bidirectional association construction are our adaptation, not their tracker.
Only observations outside the outage determine frames and anchor rotations.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'paper-public-association-v9'))
from context_mixture import ContextMixture,inputs,auxiliary
import torch

def equal_radius_anchor(node,context):
    p=node[...,:3];c=context[:,None,6];s=context[:,None,7]
    return torch.stack([p[...,0]*c+p[...,1]*s,-p[...,0]*s+p[...,1]*c,p[...,2]],-1)

class PhysicalMixture(ContextMixture):
    def __init__(self,physical=True,attention=False,residual=True):
        super().__init__(attention=attention,transport=False)
        self.physical=physical;self.residual=residual
        if attention and not residual:raise ValueError('Attention parent includes residual; unsupported ablation')
    def params(self,a,b,mb,node,context):
        mean,var,weight=super().params(a,b,mb,node,context)
        if self.physical:
            correction=equal_radius_anchor(node,context)-node[...,:3]
            select=mean.new_tensor([0,1,0,1])[None,None,:,None]
            mean=mean+select*correction[:,:,None,:]
        return mean,var,weight

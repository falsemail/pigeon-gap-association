"""Pinned OPGM author core with the frozen motion backbone."""
from pathlib import Path
import ast, types, hashlib, os
import numpy as np, torch
from torch import nn
from temporal_models import TemporalMatcher
from opgm_sources import checked_source
VENDOR=Path(os.environ.get("PIGEON_OPGM_DIR",Path.home()/".cache"/"pigeon-gap-association"/"opgm"))

def author_module(name,relative):
    path=checked_source(VENDOR,relative);tree=ast.parse(path.read_text(),filename=str(path));skipped=[];body=[]
    for item in tree.body:
        if isinstance(item,ast.ImportFrom) and (item.module=='IPython.core.debugger' or item.module in ['src.lap_solvers.hungarian','src.lap_solvers.ILP']):
            skipped.append(ast.unparse(item));continue
        body.append(item)
    tree.body=body;module=types.ModuleType(name);module.__file__=str(path);exec(compile(tree,str(path),'exec'),module.__dict__)
    module.audit=dict(relative=relative,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),skipped_unused_imports=skipped,function_bodies_changed=False);return module

solver=author_module("opgm_author_solver","src/pgm_solver/pma.py")
feasibility=author_module("opgm_author_feasibility","src/pgm_solver/pmatnew.py")
loss_module=author_module("opgm_author_loss","src/loss_func.py")
sinkhorn_module=author_module("opgm_author_sinkhorn","src/lap_solvers/sinkhorn.py")

class MotionOPGM(nn.Module):
    def __init__(self,tau=.05):
        super().__init__();self.tau=tau;self.backbone=TemporalMatcher(91,41,'temporal_graph');self.wrs=nn.Parameter(torch.tensor(1.))
        self.sinkhorn=sinkhorn_module.Sinkhorn(max_iter=20,tau=tau,epsilon=1e-10)
        self.feasible=feasibility.PMATA(self.wrs,torch.tensor(.8),torch.tensor(.8))
        self.criterion=loss_module.PMLoss()
    def forward(self,s,rho=.4):
        a=self.backbone(s)['logits'];n=s['ma'].sum(1);m=s['mb'].sum(1)
        prob=self.sinkhorn(a,n,m,dummy_row=True)
        possible,alpha,beta=self.feasible(prob,a,rho,n,m)
        return dict(affinity=a,probability=prob,possible=possible,alpha=alpha,beta=beta,n=n,m=m)
    def loss(self,o,s,weight=.4):
        n,m=o['probability'].shape[1:];target=s['target'];gt=(target[:,:,None]==torch.arange(m,device=target.device)[None,None,:]).float()*s['ma'][:,:,None]*s['mb'][:,None,:]
        return self.criterion(o['probability'],gt,o['possible'],o['alpha'],o['beta'],weight,o['n'],o['m'])
    def decode(self,probability,affinity,rho):
        P=solver.pma(torch.as_tensor(probability),torch.as_tensor(affinity),self.wrs,torch.tensor(.8),torch.tensor(.8),rho)
        out=np.full(P.shape[0],-1,int);i,j=np.where(P.cpu().numpy()>.5);out[i]=j;return out

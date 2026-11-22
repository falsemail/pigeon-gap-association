"""Observed-only inputs for the frozen neural matchers."""
import numpy as np
import torch
from data_protocol import descriptors
from context_mixture import context

def prepare_observations(samples):
    B=len(samples);N=max(len(s['a']['x']) for s in samples);M=max(len(s['b']['x']) for s in samples)
    result=dict(a=np.zeros((B,N,41),np.float32),b=np.zeros((B,M,41),np.float32),
        history_a=np.zeros((B,N,21,19),np.float32),history_b=np.zeros((B,M,21,19),np.float32),
        ma=np.zeros((B,N),bool),mb=np.zeros((B,M),bool),
        context_a=np.zeros((B,16),np.float32),context_b=np.zeros((B,16),np.float32),coordinates=np.zeros((B,3),np.float32))
    for k,raw in enumerate(samples):
        # Restrict context to observed fields even if callers retain labels.
        observed={key:raw[key] for key in ['a','b','h']}
        for side,mask in [('a','ma'),('b','mb')]:
            d=descriptors(observed[side]);n=len(d['node'])
            result[side][k,:n]=d['node'];result['history_'+side][k,:n]=d['temporal'];result[mask][k,:n]=True
        result['context_a'][k]=context(observed);result['context_b'][k]=context(observed,True)
        result['coordinates'][k]=[1.,1.,float(not (np.max(observed['a']['sd'][:,2])<1e-5 and np.max(observed['b']['sd'][:,2])<1e-5))]
    return result

def tensor_inputs(prepared,device):return {k:torch.as_tensor(v,device=device) for k,v in prepared.items()}

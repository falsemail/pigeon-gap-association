"""Remove reference identities and retain only observable fields."""
import copy
import numpy as np

OBSERVED_KEYS=['a','b','h','context','cost','pair','reachable','forward','backward','dense_features']

def observed(raw):
    s=copy.deepcopy({k:raw[k] for k in OBSERVED_KEYS})
    n,m=s['cost'].shape
    for side,offset in [('a',10000),('b',20000)]:s[side]['ids']=np.arange(len(s[side]['x']))+offset
    s['meta']={k:raw['meta'][k] for k in ['sigma','gap']}
    s['target']=np.full(n,m,np.int64);s['reverse_target']=np.full(m,n,np.int64)
    return s

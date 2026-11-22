"""Frozen BCPD scoring using the separately prepared author binary."""
import os, subprocess, tempfile
from pathlib import Path
import numpy as np

def core(Y,X,cfg,original=False):
    with tempfile.TemporaryDirectory() as t:
        p=Path(t);np.savetxt(p/'Y.txt',Y,fmt='%.17g');np.savetxt(p/'X.txt',X,fmt='%.17g')
        args=[os.environ['PIGEON_BCPD_BIN'],'-x',str(p/'X.txt'),'-y',str(p/'Y.txt'),'-T'+cfg['mode'],'-b'+str(cfg['beta']),'-l'+str(cfg['lam']),'-w'+str(cfg['outlier']),'-g'+str(cfg['gamma']),'-ux','-N1','-n100','-c1e-4','-d1000000','-sPce','-o',str(p/'fit_'),'-q']
        run=subprocess.run(args,env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),cwd=p,capture_output=True,text=True,timeout=10)
        if run.returncode:raise RuntimeError(run.stderr[-1500:])
        s=(p/'fit_P.txt').read_text().splitlines();P=np.zeros((len(Y),len(X)))
        for line in s[1:]:
            a,b,v=line.split();P[int(b)-1,int(a)-1]=float(v)
        shape=np.loadtxt(p/'fit_y.txt',ndmin=2);best=np.loadtxt(p/'fit_e.txt',skiprows=1,ndmin=2)
        assert np.isfinite(P).all() and np.isfinite(shape).all() and P.min()>=0 and np.max(P.sum(0))<=1+1e-7
        return P,shape,best

def coords(q):
    s=q['sample'];dims=2 if np.max(s['a']['sd'][:,2])<1e-5 and np.max(s['b']['sd'][:,2])<1e-5 else 3
    return q['node_a'][:,:dims].astype(float)*20,q['node_b'][:,:dims].astype(float)*20

def predict_cost(q,cfg):
    a,b=coords(q)
    try:
        p,_,_=core(a,b,cfg);c=-np.log(p.clip(1e-12,1))
        if cfg.get('symmetric'):
            pp,_,_=core(b,a,cfg);c=.5*(c-np.log(pp.T.clip(1e-12,1)))
        return c,None
    except (RuntimeError,subprocess.TimeoutExpired,AssertionError) as error:
        # Prespecified deterministic fallback includes failed conditions in all
        # metrics; never drop hard cases or use the hidden true identity.
        c=np.log1p(np.sum((a[:,None]-b[None])**2,2)/100)
        return c,str(error)

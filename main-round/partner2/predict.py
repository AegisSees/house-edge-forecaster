"""House Edge forecaster (see README.md). Deterministic; no training at predict time."""
import argparse, csv, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import numpy as np
from common import load
import model4, model6, latent

LATENT_HL=8.0; LATENT_K=0.4; LATENT_PRIOR=2.0

def series(d,procs):
    cfg={**model4.CFG,'sd_b':0.3}; eps={}
    for r in d['q']: eps.setdefault(r['episode_id'],[]).append(r)
    jobs=[(rows,{r['machine_id']:d['H'][(e,r['machine_id'])] for r in rows},cfg) for e,rows in eps.items()]
    if procs>1:
        from multiprocessing import Pool
        with Pool(procs) as p: res=p.starmap(latent.episode_series,jobs,chunksize=1)
    else: res=[latent.episode_series(*j) for j in jobs]
    return dict(zip(eps.keys(),res)), eps

def adjust(P,S,eps):
    lg=lambda p: np.log(np.clip(p,1e-12,1-1e-12)/(1-np.clip(p,1e-12,1-1e-12)))
    Q={}
    for e,rs in eps.items():
        cu=rs[0]['cutoff']; num=den=0.0
        for ser in S[e].values():
            for t,(y,p) in ser.items():
                w=0.5**((cu-1-t)/LATENT_HL); num+=w*(y-p); den+=w*p*(1-p)
        u=num/(den+LATENT_PRIOR)
        for r in rs:
            p=np.array(P[r['query_id']],float); pw=1-p[0]; pw2=1/(1+np.exp(-(lg(pw)+LATENT_K*u))); sh=p[1:]/p[1:].sum()
            Q[r['query_id']]=np.concatenate([[1-pw2],pw2*sh])
    return Q

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--split',default='test'); ap.add_argument('--procs',type=int,default=max(1,min(os.cpu_count() or 1,8)))
    a=ap.parse_args()
    d=load(a.split,root=a.data)
    P=model6.predict(d,procs=a.procs)
    S,eps=series(d,a.procs)
    Q=adjust(P,S,eps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['query_id','p_0','p_1','p_2','p_5','p_20'])
        for r in d['q']:
            p=[float(x) for x in Q[r['query_id']]]; s=sum(p); p=[x/s for x in p]
            w.writerow([r['query_id']]+[repr(x) for x in p])

if __name__=='__main__': main()

import sys, os, numpy as np, collections, pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *; import model4 as M
def machine_series(r,s,others,cfg):
    """MAP top hypothesis in-sample logit series and info at cutoff"""
    cu=r['cutoff']; ticks=[t for t in sorted(s) if t<cu]
    ep=M.ep_feat(others,ticks,cfg.get('ep_lag',3)); en=M.ep_feat(others,[cu],cfg.get('ep_lag',3))[0]
    p,w,info=M.win_model(s,ticks,cu,M.BIAS[(r['machine_type'],r['signal'])],cfg,return_post=True,ep=ep,en=en)
    ch,j,sg=info[int(np.argmax(w))]
    a,b=M.PAIRS[j]; tk=np.array(ticks,float); y=np.array([s[t]>0 for t in ticks],float)
    z=np.array([(1.0 if (s[t-a]>0)!=(s[t-b]>0) else -1.0) if (t-a in s and t-b in s) else 0.0 for t in ticks])*M.gfun(tk,ch)
    bb,_=M.fit2(z,y,np.array([M.BIAS[(r['machine_type'],r['signal'])],sg*cfg['W'],cfg['c_mu']]),np.array([cfg['sd_b']**2,cfg['sd_w']**2,cfg['c_sd']**2]),extra=ep)
    e=bb[0]+bb[1]*z+bb[2]*ep
    return dict(zip(ticks,zip(y,1/(1+np.exp(-e)))))
def episode_series(rows,Hs,cfg):
    out={}
    for r in rows:
        others=[Hs[m] for m in Hs if m!=r['machine_id']]
        out[r['machine_id']]=machine_series(r,Hs[r['machine_id']],others,cfg)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    split=sys.argv[1]; d=load(split); cfg={**M.CFG,'sd_b':0.3}
    eps={}
    for r in d['q']: eps.setdefault(r['episode_id'],[]).append(r)
    jobs=[(rows,{r['machine_id']:d['H'][(e,r['machine_id'])] for r in rows},cfg) for e,rows in eps.items()]
    with Pool(7) as p: res=p.starmap(episode_series,jobs)
    pickle.dump(dict(zip(eps.keys(),res)),open(f'work/series_{split}.pkl','wb'))

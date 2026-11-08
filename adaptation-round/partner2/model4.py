import sys, numpy as np, time
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from common import *
PAIRS=[(1,3),(2,5),(4,7)]; SL=(1,2,4)
SH={'steady':([0.754,0.197,0.048,0.001],[0.455,0.436,0.107,0.002]),
    'balanced':([0.5,0.348,0.13,0.022],[0.214,0.547,0.205,0.034]),
    'jackpot':([0.335,0.245,0.315,0.105],[0.121,0.324,0.416,0.139])}
BIAS={('balanced',-1):-1.09,('balanced',0):-0.72,('balanced',1):-0.33,('jackpot',-1):-2.51,('jackpot',0):-2.21,('jackpot',1):-1.71,
      ('steady',-1):0.27,('steady',0):0.51,('steady',1):0.93}
CFG=dict(c_mu=0.7, c_sd=0.6, ep_lag=3, use_ep=True, W=2.4, sd_w=0.3, sd_b=0.45, step=4, Ls=(16,32,64,128), pc=0.5, p_abrupt=0.5, size_pc=0.3, fit_w=True)

# Round override: params.json (written by the team pipeline from the round's fitted rules) replaces the constants above.
import json as _json, os as _os
_PP=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'params.json')
PARAMS=_json.load(open(_PP)) if _os.path.exists(_PP) else None
if PARAMS:
    PAIRS=[tuple(x) for x in PARAMS['pairs']]; SL=tuple(PARAMS['size_lags'])
    SH={k:tuple(v) for k,v in PARAMS['sh'].items()}
    BIAS={(k.split('|')[0],int(k.split('|')[1])):v for k,v in PARAMS['bias'].items()}
    CFG['W']=PARAMS.get('W',CFG['W']); CFG['ep_lag']=PARAMS.get('ep_lag',CFG['ep_lag'])
def gfun(tk,ch):
    if ch is None: return np.ones_like(tk)
    T,L=ch
    if L==0: return np.where(tk>=T,-1.0,1.0)
    return 1-2*np.clip((tk-T)/L,0,1)
def fit2(z,y,mu,pv,iters=30,extra=None):
    # logit = b + w*z (+ c*extra); Laplace evidence
    b=np.array(mu,float); P=1/np.array(pv)
    X=np.stack([np.ones_like(z),z],1) if extra is None else np.column_stack([np.ones_like(z),z,extra])
    for _ in range(iters):
        e=X@b; p=1/(1+np.exp(-e)); g=X.T@(y-p)-P*(b-mu); H=(X*(p*(1-p))[:,None]).T@X+np.diag(P)
        st=np.linalg.solve(H,g); b=b+st
        if np.abs(st).max()<1e-6: break
    e=X@b; ll=np.sum(y*e-np.logaddexp(0,e))
    ev=ll-0.5*np.sum(P*(b-mu)**2)+0.5*np.sum(np.log(P))-0.5*np.linalg.slogdet(H)[1]
    return b,ev
def changes(cu,tmin,cfg):
    out=[(None,np.log(1-cfg['pc']))]
    Ts=list(range(max(tmin,8),cu-1,cfg['step']))
    n_ab=len(Ts); n_r=len(Ts)*len(cfg['Ls'])
    out+=[((T,0),np.log(cfg['pc']*cfg['p_abrupt']/n_ab)) for T in Ts]
    out+=[((T,L),np.log(cfg['pc']*(1-cfg['p_abrupt'])/n_r)) for L in cfg['Ls'] for T in Ts]
    return out
def win_model(s,ticks,cu,mu_b,cfg,return_post=False,ep=None,en=None):
    tk=np.array(ticks,float); y=np.array([s[t]>0 for t in ticks],float)
    X=np.zeros((len(ticks),len(PAIRS))); xn=np.zeros(len(PAIRS))
    for j,(a,b) in enumerate(PAIRS):
        X[:,j]=[(1.0 if (s[t-a]>0)!=(s[t-b]>0) else -1.0) if (t-a in s and t-b in s) else 0.0 for t in ticks]
        xn[j]=(1.0 if (s[cu-a]>0)!=(s[cu-b]>0) else -1.0) if (cu-a in s and cu-b in s) else 0.0
    evs=[];preds=[];info=[]
    for ch,lp in changes(cu,int(tk.min()),cfg):
        g=gfun(tk,ch); gn=float(gfun(np.array([cu],float),ch)[0])
        for j in range(len(PAIRS)):
            z=X[:,j]*g
            for sg in (1,-1):
                if ep is None:
                    b,ev=fit2(z,y,np.array([mu_b,sg*cfg['W']]),np.array([cfg['sd_b']**2,cfg['sd_w']**2])); en_=0.0; c=0.0
                    evs.append(ev+lp+np.log(1/6)); preds.append(1/(1+np.exp(-(b[0]+b[1]*xn[j]*gn))))
                else:
                    b,ev=fit2(z,y,np.array([mu_b,sg*cfg['W'],cfg['c_mu']]),np.array([cfg['sd_b']**2,cfg['sd_w']**2,cfg['c_sd']**2]),extra=ep)
                    evs.append(ev+lp+np.log(1/6)); preds.append(1/(1+np.exp(-(b[0]+b[1]*xn[j]*gn+b[2]*en))))
                info.append((ch,j,sg))
    evs=np.array(evs); w=np.exp(evs-evs.max()); w/=w.sum()
    p=float(w@np.array(preds))
    return (p,w,info) if return_post else p
def size_model(s,ticks,cu,typ,cfg):
    hi,lo=map(np.array,SH[typ]); win=[t for t in ticks if s[t]>0]
    idx={1:0,2:1,5:2,20:3}; tk=np.array(win,float)
    yi=np.array([idx[s[t]] for t in win],int)
    lhi=np.log(hi)[yi]; llo=np.log(lo)[yi]
    evs=[];preds=[]
    Ts=list(range(max(int(tk.min()) if len(tk) else 0,8),cu-1,cfg['step']))
    chs=[(None,np.log(1-cfg['size_pc']))]+[((T,0),np.log(cfg['size_pc']/max(len(Ts),1))) for T in Ts]
    for k in SL:
        f=np.array([(1.0 if s[t-k]>=2 else -1.0) if t-k in s else 0.0 for t in win])
        fn=(1.0 if s[cu-k]>=2 else -1.0) if cu-k in s else 0.0
        for sg in (1,-1):
            for ch,lp in chs:
                g=gfun(tk,ch); gn=float(gfun(np.array([cu],float),ch)[0])
                st=sg*f*g  # +1 => hi, -1 => lo, 0 => unknown (mix)
                ll=np.where(st>0,lhi,np.where(st<0,llo,np.logaddexp(lhi,llo)-np.log(2))).sum()
                sn=sg*fn*gn
                a=1.0 if sn>0 else 0.0 if sn<0 else 0.5
                evs.append(ll+lp+np.log(1/6)); preds.append(a*hi+(1-a)*lo)
    evs=np.array(evs); w=np.exp(evs-evs.max()); w/=w.sum()
    return w@np.array(preds)
def ep_feat(others,ticks,k):
    out=[]
    for t in ticks:
        v=[o[t-k]>0 for o in others if t-k in o]
        out.append(np.mean(v)-0.4 if v else 0.0)
    return np.array(out)
def predict_one(r,s,cfg,others=()):
    cu=r['cutoff']; ticks=[t for t in sorted(s) if t<cu]
    if cfg['use_ep'] and others:
        ep=ep_feat(others,ticks,cfg['ep_lag']); en=ep_feat(others,[cu],cfg['ep_lag'])[0]
    else: ep=en=None
    pw=win_model(s,ticks,cu,BIAS[(r['machine_type'],r['signal'])],cfg,ep=ep,en=en)
    sh=size_model(s,ticks,cu,r['machine_type'],cfg)
    p=np.concatenate([[1-pw],pw*sh]); return p/p.sum()
def predict(d,cfg=None,procs=7):
    cfg={**CFG,**(cfg or {})}; t0=time.time()
    from multiprocessing import Pool
    byep={}
    for r in d['q']: byep.setdefault(r['episode_id'],[]).append(r['machine_id'])
    jobs=[(r,d['H'][(r['episode_id'],r['machine_id'])],cfg,[d['H'][(r['episode_id'],m)] for m in byep[r['episode_id']] if m!=r['machine_id']]) for r in d['q']]
    with Pool(procs) as pool: res=pool.starmap(predict_one,jobs,chunksize=4)
    print(f'  predicted {len(res)} in {time.time()-t0:.0f}s',flush=True)
    return {r['query_id']:p for r,p in zip(d['q'],res)}
if __name__=='__main__':
    d=load('validation'); P=predict(d); score(P,d)
    import pickle; pickle.dump(P,open(sys.argv[1] if len(sys.argv)>1 else 'work/p4.pkl','wb'))

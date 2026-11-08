import sys, os, numpy as np, time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from model4 import PAIRS, SL, SH, BIAS, gfun, fit2, ep_feat
CFG=dict(W=2.4, sd_w=0.3, sd_b=0.3, step=4, Ls=(16,32,64,128), pc=0.5, p_abrupt=0.5, size_pc=0.3,
         c_mu=0.7, c_sd=0.6, ep_lags=(3,), c_mu6=0.2, c_sd6=0.4, use_ep=True, p_shared=0.25, rho=0.99, short_pc=0.001, short_spc=0.001, short_ps=0.001, ep_cond=False, p_stable=0.5, p_drift=0.5)
from model4 import PARAMS
if PARAMS:
    CFG['W']=PARAMS.get('W',CFG['W']); CFG['ep_lags']=(PARAMS.get('ep_lag',3),)
DROP=True   # short histories: fit each lag pair only on ticks where both lags are observed (CV +0.010)
def lse(a,axis=None):
    a=np.asarray(a); m=a.max(axis=axis,keepdims=True); r=m+np.log(np.exp(a-m).sum(axis=axis,keepdims=True))
    return r.squeeze(axis) if axis is not None else float(r.squeeze())
def win_parts(s,ticks,cu,mu_b,cfg,ep,en,chs):
    tk=np.array(ticks,float); y=np.array([s[t]>0 for t in ticks],float)
    X=np.zeros((len(ticks),len(PAIRS))); xn=np.zeros(len(PAIRS))
    for j,(a,b) in enumerate(PAIRS):
        X[:,j]=[(1.0 if (s[t-a]>0)!=(s[t-b]>0) else -1.0) if (t-a in s and t-b in s) else 0.0 for t in ticks]
        xn[j]=(1.0 if (s[cu-a]>0)!=(s[cu-b]>0) else -1.0) if (cu-a in s and cu-b in s) else 0.0
    out={}
    for ch in chs:
        g=gfun(tk,ch); gn=float(gfun(np.array([cu],float),ch)[0]); L=np.zeros(len(PAIRS)); PR=np.zeros(len(PAIRS))
        for j in range(len(PAIRS)):
            z=X[:,j]*g; evs=[];pr=[]
            if cfg.get('drop'):   # short histories: fit pair j only on ticks where both of its lags are observed
                mk=X[:,j]!=0
                if mk.sum()>=2: z=z[mk]; yy=y[mk]; epp=None if ep is None else ep[mk]
                else: yy=y; epp=ep
            else: yy=y; epp=ep
            for sg in (1,-1):
                if ep is None:
                    b,ev=fit2(z,yy,np.array([mu_b,sg*cfg['W']]),np.array([cfg['sd_b']**2,cfg['sd_w']**2])); e=b[0]+b[1]*xn[j]*gn
                else:
                    nl=ep.shape[1]; cm=[cfg['c_mu']]+[cfg['c_mu6']]*(nl-1); cs=[cfg['c_sd']**2]+[cfg['c_sd6']**2]*(nl-1)
                    b,ev=fit2(z,yy,np.array([mu_b,sg*cfg['W']]+cm),np.array([cfg['sd_b']**2,cfg['sd_w']**2]+cs),extra=epp); e=b[0]+b[1]*xn[j]*gn+b[2:]@en
                evs.append(ev+np.log(0.5)); pr.append(1/(1+np.exp(-e)))
            evs=np.array(evs); L[j]=lse(evs); PR[j]=np.exp(evs-L[j])@np.array(pr)
        out[ch]=(L,PR)
    return out
def size_parts(s,ticks,cu,typ,chs):
    hi,lo=map(np.array,SH[typ]); win=[t for t in ticks if s[t]>0]
    idx={1:0,2:1,5:2,20:3}; tk=np.array(win,float); yi=np.array([idx[s[t]] for t in win],int)
    lhi=np.log(hi)[yi]; llo=np.log(lo)[yi]; out={}
    for ch in chs:
        g=gfun(tk,ch); gn=float(gfun(np.array([cu],float),ch)[0]); L=np.zeros(len(SL)); PR=np.zeros((len(SL),4))
        for j,k in enumerate(SL):
            f=np.array([(1.0 if s[t-k]>=2 else -1.0) if t-k in s else 0.0 for t in win]); fn=(1.0 if s[cu-k]>=2 else -1.0) if cu-k in s else 0.0
            evs=[];pr=[]
            for sg in (1,-1):
                st=sg*f*g; ll=np.where(st>0,lhi,np.where(st<0,llo,np.logaddexp(lhi,llo)-np.log(2))).sum()
                sn=sg*fn*gn; a=1.0 if sn>0 else 0.0 if sn<0 else 0.5
                evs.append(ll+np.log(0.5)); pr.append(a*hi+(1-a)*lo)
            evs=np.array(evs); L[j]=lse(evs); PR[j]=np.exp(evs-L[j])@np.array(pr)
        out[ch]=(L,PR)
    return out
def marg(parts,lps):
    """combine over change hypotheses per j: returns L[j], pred[j]"""
    keys=list(parts); A=np.array([lps[k]+parts[k][0] for k in keys])  # (nch,3)
    L=lse(A,axis=0); w=np.exp(A-L)  # (nch,3)
    P=np.array([parts[k][1] for k in keys])  # (nch,3) or (nch,3,4)
    pred=np.einsum('cj,cj...->j...',w,P)
    return L,pred
def predict_episode(rows,Hs,cfg):
    cu=rows[0]['cutoff']; mids=[r['machine_id'] for r in rows]
    short=np.mean([len([t for t in h if t<cu]) for h in Hs.values()])<0.6*cu
    if short and cfg.get('short_pc') is not None:
        cfg={**cfg,'pc':cfg['short_pc'],'size_pc':cfg['short_spc'],'p_shared':cfg['short_ps'],'drop':DROP}
    tmin=min(min(h) for h in Hs.values()); Ts=list(range(max(tmin,8),cu-1,cfg['step']))
    wch=[None]+[(T,0) for T in Ts]+[(T,L) for L in cfg['Ls'] for T in Ts]
    schs=[None]+[(T,0) for T in Ts]
    lpw={None:np.log(1-cfg['pc'])}
    for T in Ts: lpw[(T,0)]=np.log(cfg['pc']*cfg['p_abrupt']/len(Ts))
    for L in cfg['Ls']:
        for T in Ts: lpw[(T,L)]=np.log(cfg['pc']*(1-cfg['p_abrupt'])/(len(Ts)*len(cfg['Ls'])))
    lps={None:np.log(1-cfg['size_pc'])}
    for T in Ts: lps[(T,0)]=np.log(cfg['size_pc']/len(Ts))
    rho=cfg['rho']; nP=len(PAIRS); LK=np.log(np.where(np.eye(nP)>0,rho,(1-rho)/max(nP-1,1)))+np.log(1/nP)  # log p(j)p(k|j)
    res={}
    for r in rows:
        m=r['machine_id']; s=Hs[m]; ticks=[t for t in sorted(s) if t<cu]; others=[Hs[o] for o in mids if o!=m]
        ep=en=None
        if cfg['use_ep'] and others:
            ep=np.column_stack([ep_feat(others,ticks,k) for k in cfg['ep_lags']]); en=np.array([ep_feat(others,[cu],k)[0] for k in cfg['ep_lags']])
        W=win_parts(s,ticks,cu,BIAS[(r['machine_type'],r['signal'])],cfg,ep,en,wch)
        S=size_parts(s,ticks,cu,r['machine_type'],schs)
        SL_,SP=marg(S,lps)
        WL,WP=marg(W,lpw)
        res[m]=(W,WL,WP,SL_,SP)
    def joint(WLj,WPj,SLk,SPk):
        A=LK+WLj[:,None]+SLk[None,:]; Lt=lse(A); w=np.exp(A-Lt)
        pw=w.sum(1)@WPj; sh=w.sum(0)@SPk
        return Lt,pw,sh
    if cfg.get('ep_cond') and not short:
        # episode-level: stable (no win change on any machine) vs all-machines-change (independent T/L)
        def sub(keys):
            lp={k:lpw[k] for k in keys}; tot=lse([lp[k] for k in keys]); lp={k:v-tot for k,v in lp.items()}
            return lp
        lp_none={None:0.0}; lp_chg=sub([k for k in wch if k is not None])
        st={m:joint(*marg({None:res[m][0][None]},lp_none),res[m][3],res[m][4]) for m in mids}
        cg={m:joint(*marg({k:res[m][0][k] for k in wch if k is not None},lp_chg),res[m][3],res[m][4]) for m in mids}
        Ls=sum(v[0] for v in st.values()); Lc=sum(v[0] for v in cg.values())
        a=np.array([np.log(cfg['p_stable'])+Ls,np.log(cfg['p_drift'])+Lc]); a=np.exp(a-lse(a))
        ind={m:(lse([np.log(cfg['p_stable'])+st[m][0],np.log(cfg['p_drift'])+cg[m][0]]),a[0]*st[m][1]+a[1]*cg[m][1],a[0]*st[m][2]+a[1]*cg[m][2]) for m in mids}
        L_ind=lse([np.log(cfg['p_stable'])+Ls,np.log(cfg['p_drift'])+Lc])
    else:
        ind={m:joint(res[m][1],res[m][2],res[m][3],res[m][4]) for m in mids}
        L_ind=sum(v[0] for v in ind.values())
    shT=[]
    for T in Ts:
        vals={m:joint(res[m][0][(T,0)][0],res[m][0][(T,0)][1],res[m][3],res[m][4]) for m in mids}
        shT.append((sum(v[0] for v in vals.values()),vals))
    lp=np.array([np.log(1-cfg['p_shared'])+L_ind]+[np.log(cfg['p_shared']/len(Ts))+x[0] for x in shT])
    post=np.exp(lp-lse(lp))
    out={}
    for r in rows:
        m=r['machine_id']
        pw=post[0]*ind[m][1]+sum(post[i+1]*shT[i][1][m][1] for i in range(len(Ts)))
        sh=post[0]*ind[m][2]+sum(post[i+1]*shT[i][1][m][2] for i in range(len(Ts)))
        p=np.concatenate([[1-pw],pw*np.asarray(sh)]); out[r['query_id']]=p/p.sum()
    return out
def predict(d,cfg=None,procs=7,verbose=True):
    cfg={**CFG,**(cfg or {})}; t0=time.time()
    eps={}
    for r in d['q']: eps.setdefault(r['episode_id'],[]).append(r)
    jobs=[(rows,{r['machine_id']:d['H'][(e,r['machine_id'])] for r in rows},cfg) for e,rows in eps.items()]
    if procs>1:
        from multiprocessing import Pool
        with Pool(procs) as pool: res=pool.starmap(predict_episode,jobs,chunksize=1)
    else: res=[predict_episode(*j) for j in jobs]
    out={}
    for o in res: out.update(o)
    if verbose: print(f'  predicted {len(out)} in {time.time()-t0:.0f}s',flush=True)
    return out
if __name__=='__main__':
    from score import score; import pickle
    d=load('validation'); P=predict(d); score(P,d)
    pickle.dump(P,open(sys.argv[1] if len(sys.argv)>1 else 'work/p6.pkl','wb'))

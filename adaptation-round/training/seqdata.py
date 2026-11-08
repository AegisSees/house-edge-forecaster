import numpy as np, pandas as pd
from common import *
T = 257
TYP = {'steady':0,'balanced':1,'jackpot':2}
def episode_tensors(d):
    """returns X (E,8,T,D) float32, Y (E,8,T) int64 (-1 missing), cut (E,), qids (E,8)"""
    q = d['q']; h = d['h']
    qm = q.set_index('machine_id')
    Xs, Ys, cuts, qids = [], [], [], []
    hg = dict(tuple(h.groupby('episode_id')))
    for ep, qe in q.groupby('episode_id', sort=False):
        qe = qe.sort_values('machine_id'); ms = list(qe.machine_id); cut = int(qe.cutoff.iloc[0])
        nM = len(ms); mi = {m:i for i,m in enumerate(ms)}
        ge = hg[ep]
        M = np.full((nM, T), np.nan)
        M[ge.machine_id.map(mi).values, ge.tick.values] = ge.payout.values
        M[:, cut:] = np.nan
        C = np.full((nM, T), 5)  # category 0..4, 5=missing
        ok = ~np.isnan(M); C[ok] = np.vectorize(PIDX.get)(M[ok].astype(int))
        OH = np.eye(6)[C]  # nM,T,6
        feats = []
        for k in range(1,9):
            sh = np.zeros_like(OH); sh[:,:,5]=1; sh[:,k:] = OH[:,:-k]; feats.append(sh)
        # others summary at lags 1..8: win frac among observed, observed frac
        W = np.where(ok, (np.nan_to_num(M)>0).astype(float), 0); O = ok.astype(float)
        for k in range(1,9):
            Wk = np.zeros_like(W); Ok = np.zeros_like(O); Wk[:,k:]=W[:,:-k]; Ok[:,k:]=O[:,:-k]
            sw = Wk.sum(0,keepdims=True)-Wk; so = Ok.sum(0,keepdims=True)-Ok
            feats.append(np.stack([sw/np.maximum(so,1), so/(nM-1)],-1))
        st = np.zeros((nM,T,5))
        for i,m in enumerate(ms):
            st[i,:,TYP[qm.loc[m,'machine_type']]] = 1; st[i,:,3] = qm.loc[m,'signal']
        st[:,:,4] = np.arange(T)/256
        X = np.concatenate(feats+[st],-1).astype(np.float32)
        Y = np.where(ok, C, -1); Y[:, cut:] = -1
        Xs.append(X); Ys.append(Y); cuts.append(cut); qids.append(list(qe.query_id))
    return np.stack(Xs), np.stack(Ys).astype(np.int64), np.array(cuts), np.array(qids)

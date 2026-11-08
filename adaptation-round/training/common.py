import numpy as np, pandas as pd
from pathlib import Path
import os
D = Path(os.environ.get('KIT','kit')) / 'public'
PAY = [0,1,2,5,20]; PIDX = {p:i for i,p in enumerate(PAY)}
PC = ['p_0','p_1','p_2','p_5','p_20']
def load(split):
    q = pd.read_csv(D/split/'queries.csv')
    h = pd.read_csv(D/split/'history.csv')
    out = {'q': q, 'h': h}
    for f in ['labels','conditions','targets','reference']:
        p = D/split/f'{f}.csv'
        if p.exists(): out[f] = pd.read_csv(p)
    return out
def score(pred, d):
    """pred: DataFrame query_id + PC. returns points, ratios dict"""
    t = d['targets'].set_index('query_id'); r = d['reference'].set_index('query_id')
    c = d['conditions'].set_index('query_id')['condition']
    q = d['q'].set_index('query_id')
    p = pred.set_index('query_id').loc[t.index]
    e = ((p[PC].values - t[PC].values)**2).sum(1)
    eb = ((r.loc[t.index, PC].values - t[PC].values)**2).sum(1)
    df = pd.DataFrame({'e':e,'eb':eb,'ep':q.loc[t.index,'episode_id'].values,'c':c.loc[t.index].values})
    g = df.groupby(['c','ep'])[['e','eb']].mean().groupby('c').mean()
    ratios = (g.e/np.maximum(g.eb,1e-9))
    R = ratios.mean()
    return 60/(1+R), ratios.round(3).to_dict()
def seqs(h):
    """dict (ep,machine)->dict tick->payout"""
    out = {}
    for (ep,m), g in h.groupby(['episode_id','machine_id']):
        out[(ep,m)] = dict(zip(g.tick.values, g.payout.values))
    return out

"""Write partner2/params.json for a round from our fitted rules.json plus the round's histories.
usage: python partner_params.py <rules.json> <public folder> <out params.json>"""
import json, sys, numpy as np, pandas as pd
rules = json.load(open(sys.argv[1])); pub = sys.argv[2]
lg = lambda p: float(np.log(p / (1 - p)))
pairs = [[r[0], r[1]] for r in rules['rules'] if r[0] is not None]
sl = [r[2] for r in rules['rules'] if r[0] is not None]
sh = {t: [v['low'], v['high']] for t, v in rules['vectors'].items()}
bias, ws = {}, []
# per-machine cell rates on stable train machines (conditions are public for train), best rule pair per machine
q = pd.read_csv(f'{pub}/train/queries.csv'); cnd = pd.read_csv(f'{pub}/train/conditions.csv'); q = q.merge(cnd, on='query_id')
ht = pd.read_csv(f'{pub}/train/history.csv'); per = {}
for _, r in q[q.condition == 'stable'].iterrows():
    g = ht[ht.machine_id == r.machine_id]; Y = np.full(int(r.cutoff), -1); Y[g.tick.values] = g.payout.values; W_ = (Y > 0).astype(float)
    bestm = None
    for a_, b_ in pairs:
        t = np.arange(max(a_, b_), len(Y)); x = W_[t - a_] != W_[t - b_]; y = W_[t]
        n1, s1, n0, s0 = x.sum(), y[x].sum(), (~x).sum(), y[~x].sum()
        r1, r0 = (s1 + .5) / (n1 + 1), (s0 + .5) / (n0 + 1)
        if bestm is None or abs(lg(r1) - lg(r0)) > bestm[0]: bestm = (abs(lg(r1) - lg(r0)), (lg(r1) + lg(r0)) / 2)
    per.setdefault(f'{r.machine_type}|{int(r.signal)}', []).append(bestm); ws.append(bestm[0] / 2)
bias = {k: float(np.median([v[1] for v in vs])) for k, vs in per.items()}
W = float(np.median(ws)) if ws else 2.4
# cross-machine coupling lag: which lag of the other machines' mean win best tracks a machine's win (within machine)
rows = []
for split in ('train', 'validation', 'test'):
    h = pd.read_csv(f'{pub}/{split}/history.csv'); h['win'] = (h.payout > 0).astype(float); rows.append(h)
h = pd.concat(rows)
best = None
for k in range(1, 13):
    num = den = 0.0
    for ep, g in h.groupby('episode_id'):
        P = g.pivot_table(index='tick', columns='machine_id', values='win')
        if len(P) < 150: continue
        A = P.values; n = A.shape[1]
        oth = (np.nansum(A, 1, keepdims=True) - np.nan_to_num(A)) / np.maximum((~np.isnan(A)).sum(1, keepdims=True) - (~np.isnan(A)), 1)
        x = oth[:-k]; y = A[k:]; ok = ~np.isnan(y)
        y = y - np.nanmean(A, 0); x = x - np.nanmean(oth, 0)
        num += np.nansum((x * y)[ok]); den += np.sqrt(np.nansum((x ** 2)[ok]) * np.nansum((y ** 2)[ok]))
    c = num / den
    if best is None or c > best[0]: best = (c, k)
    print('coupling lag', k, round(c, 4))
out = {'pairs': pairs, 'size_lags': sl, 'sh': sh, 'bias': bias, 'W': W, 'ep_lag': best[1]}
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print('W', round(W, 3), 'ep_lag', best[1], 'pairs', pairs, 'size lags', sl)

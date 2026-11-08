"""Pick the team blend for a round from each model's own validation predictions; write models/blend.json.
usage: python blend_fit.py --kit <public folder> --out blend.json ours.csv[:name] partnerA.csv:folderA ...
The first CSV is ours (sizes come from it). Names after ':' are the partner folder names inside the entry."""
import argparse, json, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--kit', required=True); ap.add_argument('--out', required=True); ap.add_argument('csvs', nargs='+')
a = ap.parse_args(); PC = ['p_0', 'p_1', 'p_2', 'p_5', 'p_20']
T = pd.read_csv(f'{a.kit}/validation/targets.csv').set_index('query_id').sort_index(); idx = T.index
ref = pd.read_csv(f'{a.kit}/validation/reference.csv').set_index('query_id').loc[idx]
q = pd.read_csv(f'{a.kit}/validation/queries.csv').set_index('query_id').loc[idx]; ep = q.episode_id.values; typ = q.machine_type.values
cond = pd.read_csv(f'{a.kit}/validation/conditions.csv').set_index('query_id').loc[idx, 'condition'].values
nobs = pd.read_csv(f'{a.kit}/validation/history.csv').groupby(['episode_id', 'machine_id']).size()
short = np.array([nobs.get((e, m), 0) < 100 for e, m in zip(q.episode_id, q.machine_id)]).astype(int)
paths = [c.split(':')[0] for c in a.csvs]; names = [c.split(':')[1] if ':' in c else None for c in a.csvs]
C = [pd.read_csv(p).set_index('query_id').loc[idx] for p in paths]
SZ = C[0][PC[1:]].values / (1 - C[0].p_0.values)[:, None]
M = np.clip(np.stack([1 - c.p_0.values for c in C], 1), 1e-6, 1 - 1e-6); L = np.log(M / (1 - M)); K = M.shape[1]
pt = 1 - T.p_0.values; Tv = T[PC].values; eref = ((ref[PC].values - Tv) ** 2).sum(1)
SZS = [c[PC[1:]].values / np.maximum(1 - c.p_0.values, 1e-12)[:, None] for c in C]
def R(pw, sz=None):
    sz = SZ if sz is None else sz
    e = ((np.column_stack([1 - pw, pw[:, None] * sz]) - Tv) ** 2).sum(1)
    df = pd.DataFrame({'e': e, 'eb': eref, 'ep': ep, 'c': cond}); g = df.groupby(['c', 'ep'])[['e', 'eb']].mean().groupby('c').mean(); return (g.e / g.eb).mean()
import itertools
steps = np.round(np.arange(0, 1.0001, 0.1), 1)
grid = [np.array(w) for w in itertools.product(steps, repeat=K) if abs(sum(w) - 1) < 1e-9]
def mix(w, ii, logit): return 1 / (1 + np.exp(-(L[ii] @ w))) if logit else M[ii] @ w
thr = {t: float(np.median(M[typ == t].mean(1))) for t in np.unique(typ)}
low = (M.mean(1) < np.array([thr[t] for t in typ])).astype(int)
def fitW(ii, st, logit): return {g: min(grid, key=lambda w: ((mix(w, ii[st[ii] == g], logit) - pt[ii[st[ii] == g]]) ** 2).mean()) for g in np.unique(st)}
def apply(W, st, logit):
    out = np.zeros(len(idx))
    for g, w in W.items(): ii = np.where(st == g)[0]; out[ii] = mix(w, ii, logit)
    return out
cser = pd.Series(cond, index=ep).groupby(level=0).first()
def cv(st, logit, seeds=range(6)):
    r = []
    for s in seeds:
        rng = np.random.default_rng(s); folds = {}
        for c, g in cser.groupby(cser):
            for i, x in enumerate(rng.permutation(g.index.values)): folds[x] = i % 4
        f = np.array([folds[e] for e in ep]); out = np.zeros(len(idx))
        for k in range(4):
            tr = np.where(f != k)[0]; te = np.where(f == k)[0]; out[te] = apply(fitW(tr, st, logit), st, logit)[te]
        r.append(R(out))
    return float(np.mean(r))
for i in range(K): print('model', i, paths[i], 'alone ratio', round(R(M[:, i]), 5))
best = ('alone', cv(np.zeros(len(idx), int), True) if K == 1 else None)
cands = []
for name, st in [('global', np.zeros(len(idx), int)), ('state', low), ('state x short', low + 2 * short)]:
    for logit in [True]:   # predict.py blends in log-odds
        c = cv(st, logit); cands.append((c, name, logit, st)); print(f'{name:14s} {"logit" if logit else "linear"} CV ratio {c:.5f}')
c, name, logit, st = min(cands, key=lambda z: z[0])
ours_alone = cv(np.zeros(len(idx), int), True) if False else None
W = fitW(np.arange(len(idx)), st, logit)
gname = {0: 'high', 1: 'low', 2: 'short_high', 3: 'short_low'}
if name == 'global': weights = {k: W[0].tolist() for k in ['high', 'low', 'short_high', 'short_low']}
elif name == 'state': weights = {'high': W[0].tolist(), 'low': W[1].tolist(), 'short_high': W[0].tolist(), 'short_low': W[1].tolist()}
else: weights = {gname[int(g)]: w.tolist() for g, w in W.items()}
if not logit: print('NOTE: linear blend chosen by CV; predict.py blends in log-odds, so the logit option is used instead')
# payout-size source (ours or a partner), separately for full and short histories, judged on validation
pwf = apply(W, st, logit); opts = []
for fi in range(K):
    for si in range(K):
        sz = np.where(short[:, None] == 1, SZS[si], SZS[fi]); opts.append((R(pwf, sz), fi, si))
for r_, fi, si in sorted(opts)[:4]: print('sizes full from', fi, 'short from', si, 'ratio', round(r_, 5))
r_, fi, si = min(opts)
blend = {'partners': [n for n in names[1:]], 'gate_threshold': {k: round(v, 4) for k, v in thr.items()}, 'weights': weights,
         'full_size_partner': names[fi] if fi > 0 else None, 'short_size_partner': names[si] if si > 0 else None,
         'cv_ratio': c, 'choice': name, 'val_ratio_with_sizes': r_}
json.dump(blend, open(a.out, 'w'), indent=1); print('wrote', a.out, json.dumps(blend))

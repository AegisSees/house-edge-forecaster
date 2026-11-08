"""Decide on validation whether the episode-latent correction helps this round; record it in blend.json.
usage: python latent_choice.py <public> <val with latent> <val without latent> <blend.json>"""
import json, sys, shutil, numpy as np, pandas as pd
pub, on, off, bj = sys.argv[1:5]; PC = ['p_0', 'p_1', 'p_2', 'p_5', 'p_20']
T = pd.read_csv(f'{pub}/validation/targets.csv').set_index('query_id').sort_index(); idx = T.index
ref = pd.read_csv(f'{pub}/validation/reference.csv').set_index('query_id').loc[idx]
q = pd.read_csv(f'{pub}/validation/queries.csv').set_index('query_id').loc[idx]
c = pd.read_csv(f'{pub}/validation/conditions.csv').set_index('query_id').loc[idx, 'condition'].values
eb = ((ref[PC].values - T[PC].values) ** 2).sum(1)
def R(path):
    p = pd.read_csv(path).set_index('query_id').loc[idx]; e = ((p[PC].values - T[PC].values) ** 2).sum(1)
    g = pd.DataFrame({'e': e, 'eb': eb, 'ep': q.episode_id.values, 'c': c}).groupby(['c', 'ep'])[['e', 'eb']].mean().groupby('c').mean()
    return float((g.e / g.eb).mean())
r_on, r_off = R(on), R(off); use = r_on < r_off
b = json.load(open(bj)); b['episode_latent'] = bool(use); json.dump(b, open(bj, 'w'), indent=1)
if not use: shutil.copyfile(off, on)
print(f'episode latent: ratio with {r_on:.6f} without {r_off:.6f} -> {"ON" if use else "OFF"}')

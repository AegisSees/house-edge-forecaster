"""Fit the payout-size model configuration for a round (run once per round, before predicting).

1. Size vectors: the two payout-size distributions per machine type, read from
   validation targets (public). Reports their spread to confirm there are exactly two.
2. Rule family: on train+validation+test histories (inputs only, no answers), finds
   which lags k and payout threshold best explain winning payout sizes per machine,
   and keeps the lags that win for a meaningful share of machines.

Usage: python fit_size_vectors.py --data public --out models/size_config.json
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from houseedge.features import build_split, read_csv

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min-share", type=float, default=0.12); a = ap.parse_args()
pub = Path(a.data)
qtype = {r["query_id"]: r["machine_type"] for r in read_csv(pub / "validation/queries.csv")}
by = {}
for r in read_csv(pub / "validation/targets.csv"):
    p = np.array([float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")])
    by.setdefault(qtype[r["query_id"]], []).append(p[1:] / max(1 - p[0], 1e-12))
vec = {}
for t, rows in by.items():
    s = np.array(rows); mid = (s[:, 0].max() + s[:, 0].min()) / 2
    lo, hi = s[s[:, 0] > mid], s[s[:, 0] <= mid]
    vec[t] = {"low": lo.mean(0).tolist(), "high": hi.mean(0).tolist(),
              "spread": float(max(lo.std(0).max(), hi.std(0).max())), "n_low": len(lo), "n_high": len(hi)}
    print(t, "low", np.round(lo.mean(0), 4), "high", np.round(hi.mean(0), 4), "spread", vec[t]["spread"])

# rule family discovery
wins = {}
for split in ("train", "validation", "test"):
    _, Y, cuts, qids = build_split(pub / split, extended=False)
    qt = {r["query_id"]: r["machine_type"] for r in read_csv(pub / split / "queries.csv")}
    for i, q in enumerate(qids):
        wins[q] = (Y[i], int(cuts[i]), qt[q])
score = {(k, th): 0.0 for k in range(1, 9) for th in (1, 2, 3, 4)}
best_count = {key: 0 for key in score}
for q, (y, cut, t) in wins.items():
    low, high = (np.log(np.array(vec[t]["low"]) + 1e-12), np.log(np.array(vec[t]["high"]) + 1e-12))
    ticks = np.where(y[:cut] >= 1)[0]
    if len(ticks) < 20:
        continue
    j = y[ticks] - 1; res = []
    for (k, th) in score:
        prev = np.where(ticks - k >= 0, y[np.maximum(ticks - k, 0)], -1); ok = prev >= 0; big = prev >= th
        lls = [np.where(np.logical_xor(big, f), high[j], low[j])[ok].sum() for f in (0, 1)]
        res.append((max(lls), (k, th)))
    res.sort(reverse=True)
    if res[0][0] - res[1][0] > 3:
        best_count[res[0][1]] += 1
total = sum(best_count.values())
ranked = sorted(best_count.items(), key=lambda kv: -kv[1])
print("best rule counts (k, threshold index):", [(k, c) for k, c in ranked[:10]], "machines with clear rule:", total)
thr = max((1, 2, 3, 4), key=lambda th: sum(c for (k, t2), c in best_count.items() if t2 == th))
lags = sorted(k for (k, t2), c in best_count.items() if t2 == thr and total and c / total >= a.min_share)
cfg = {"vectors": vec, "lags": lags or list(range(1, 9)), "threshold_index": thr}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("lags", cfg["lags"], "threshold index", thr, "->", a.out)

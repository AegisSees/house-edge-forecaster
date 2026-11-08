"""Fit the rule-model configuration for a round (run once per round, before predicting).

Uses only public files: histories of all splits (inputs only) and validation targets
(for the two payout-size vectors per machine type).
1. Size vectors per type from validation targets (reports spread; ~0 means exactly two).
2. Size rule family: which lag k / payout threshold switches the size regime.
3. Win rule family: which lag pairs (a, b) drive win chance through an XOR pattern.
4. Link: the size lag that goes with each win pair.
5. Prior win rates (high/low state) per machine type and signal.

Usage: python fit_rules.py --data public --out models/rules.json
"""
import argparse, csv, json, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--maxlag", type=int, default=int(os.environ.get("HE_MAXLAG", "16")), help="largest lag searched")
a = ap.parse_args(); pub = Path(a.data); MAXLAG = a.maxlag
# tick range of this round: cutoffs may differ from the main round
cut_max = max(int(float(r["cutoff"])) for split in ("train", "validation", "test")
              for r in csv.DictReader(open(pub / split / "queries.csv", newline="", encoding="utf-8")))
TICKS = max(257, cut_max + 1)
os.environ["HE_T"] = str(TICKS)
from houseedge.features import build_split, read_csv
from houseedge.joint_model import betaln, _pair_cells
print("ticks", TICKS, "max lag searched", MAXLAG)

# 1. size vectors
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
              "spread": float(max(lo.std(0).max(), hi.std(0).max()))}
    print(t, "low", np.round(lo.mean(0), 4), "high", np.round(hi.mean(0), 4), "spread", vec[t]["spread"])

machines = {}
for split in ("train", "validation", "test"):
    _, Y, cuts, qids = build_split(pub / split, extended=False)
    meta = {r["query_id"]: (r["machine_type"], int(float(r["signal"]))) for r in read_csv(pub / split / "queries.csv")}
    for i, q in enumerate(qids):
        machines[q] = (split, Y[i], int(cuts[i]), *meta[q])

# 2. size rule family
def size_best(y, cut, t):
    lo, hi = np.log(np.array(vec[t]["low"]) + 1e-12), np.log(np.array(vec[t]["high"]) + 1e-12)
    mix = np.logaddexp(lo, hi) - np.log(2)   # no-information baseline: scores are log-likelihood ratios
    ticks = np.where(y[:cut] >= 1)[0]
    if len(ticks) < 20:
        return None
    j = y[ticks] - 1; res = []
    for k in range(1, MAXLAG + 1):
        prev = np.where(ticks - k >= 0, y[np.maximum(ticks - k, 0)], -1); ok = prev >= 0
        for th in (1, 2, 3, 4):
            big = prev >= th
            res.append((max((np.where(np.logical_xor(big, f), hi[j], lo[j]) - mix[j])[ok].sum() for f in (0, 1)), (k, th)))
    res.sort(reverse=True)
    return res[0][1] if res[0][0] - res[1][0] > 3 else None

size_rule = {q: size_best(y, c, t) for q, (_, y, c, t, _) in machines.items()}
counts = {}
for r in size_rule.values():
    if r: counts[r] = counts.get(r, 0) + 1
thr = max((1, 2, 3, 4), key=lambda th: sum(c for (k, t2), c in counts.items() if t2 == th))
tot = sum(c for (k, t2), c in counts.items() if t2 == thr)
size_lags = sorted(k for (k, t2), c in counts.items() if t2 == thr and c / tot >= 0.12)
print("size rule counts", sorted(counts.items(), key=lambda kv: -kv[1])[:8], "-> lags", size_lags, "threshold", thr)

# 3. win rule family (long histories), chunked free-cell XOR evidence
PAIRS = [(i, j) for i in range(1, MAXLAG + 1) for j in range(i + 1, MAXLAG + 1)]
from scipy.special import betaln as _vbetaln
def win_evidence(y, cut, pa, pb, chunk=32, kappa=4.0):
    """Chunked Beta-binomial evidence for an XOR split on lags (pa, pb), as a log ratio vs. no split."""
    obs = y[:cut] >= 0; pbar = ((y[:cut] >= 1).sum() + 1) / (obs.sum() + 2); A, B = kappa * pbar, kappa * (1 - pbar)
    t, ok, x, win = _pair_cells(y, cut, pa, pb); ch = (cut - 1 - t) // chunk
    nc = int(ch.max()) + 1; idx = (2 * ch + x)[ok]; w = win[ok].astype(float)
    n2 = np.bincount(idx, minlength=2 * nc); s2 = np.bincount(idx, weights=w, minlength=2 * nc)
    n1 = n2[0::2] + n2[1::2]; s1 = s2[0::2] + s2[1::2]
    return float((_vbetaln(A + s2, B + n2 - s2)).sum() - (_vbetaln(A + s1, B + n1 - s1)).sum() - nc * _vbetaln(A, B))
win_pair = {}
for q, (_, y, c, t, _) in machines.items():
    if (y[:c] >= 0).sum() < 150:
        continue
    ev = sorted(((win_evidence(y, c, pa, pb), (pa, pb)) for pa, pb in PAIRS), reverse=True)
    if ev[0][0] - ev[1][0] > 5:
        win_pair[q] = ev[0][1]
pc = {}
for p in win_pair.values(): pc[p] = pc.get(p, 0) + 1
pairs = sorted(p for p, c in pc.items() if c / max(len(win_pair), 1) >= 0.10)
print("win pair counts", sorted(pc.items(), key=lambda kv: -kv[1])[:6], "-> pairs", pairs)

# 4. link pair -> size lag
rules, linked = [], True
for p in pairs:
    ks = [size_rule[q][0] for q, wp in win_pair.items() if wp == p and size_rule[q] and size_rule[q][1] == thr]
    if not ks:
        linked = False; continue
    k = max(set(ks), key=ks.count); share = ks.count(k) / len(ks)
    print("pair", p, "size lag", k, "share", round(share, 3), "n", len(ks))
    linked &= share >= 0.7
    rules.append([p[0], p[1], k])
if not linked or not rules:
    rules = [[None, None, k] for k in size_lags]

# 5. prior high/low win rates per (type, signal), long histories of train+test, best pair by rate gap
cent = {}
for q, (split, y, c, t, sg) in machines.items():
    if split == "validation" or (y[:c] >= 0).sum() < 150 or rules[0][0] is None:
        continue
    best = None
    for pa, pb, _ in rules:
        _, ok, x, win = _pair_cells(y, c, pa, pb)
        r1 = (win[ok & x].sum() + 1) / ((ok & x).sum() + 2); r0 = (win[ok & ~x].sum() + 1) / ((ok & ~x).sum() + 2)
        if best is None or abs(r1 - r0) > best[0]:
            best = (abs(r1 - r0), max(r1, r0), min(r1, r0))
    cent.setdefault(f"{t}|{sg}", []).append(best[1:])
centers = {k: [float(np.median([v[0] for v in vs])), float(np.median([v[1] for v in vs]))] for k, vs in cent.items()}
used = [x for r in rules for x in r if x] + list(size_lags)
lags = int(min(MAXLAG, max([8] + used)))
cfg = {"vectors": vec, "threshold_index": int(thr), "rules": rules, "centers": centers,
       "chunk": 32, "kappa_size": 4.0, "kappa_win": 3.0, "short_obs": 100, "short_blend": 0.2 if centers else 0.0,
       "lags": lags, "ticks": TICKS}
print("network input lags", lags, "ticks", TICKS)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("rules", rules, "->", a.out)

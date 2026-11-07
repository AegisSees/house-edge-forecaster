"""Build per-episode input tensors from the public CSVs (numpy + stdlib only).

Every input for tick t uses only observations at ticks < t, and nothing at or after
the query cutoff is read. Missing ticks are encoded as a separate 'missing' category.
"""
import csv
from pathlib import Path
import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PIDX = {p: i for i, p in enumerate(PAYOUTS)}
TYPES = {"steady": 0, "balanced": 1, "jackpot": 2}
T = 257
LAGS = 8


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(folder):
    folder = Path(folder)
    queries = read_csv(folder / "queries.csv")
    history = {}
    for row in read_csv(folder / "history.csv"):
        history.setdefault(row["episode_id"], []).append(
            (row["machine_id"], int(row["tick"]), int(float(row["payout"]))))
    return queries, history


def episode_features(machines, cut, rows):
    """machines: list of (machine_id, type, signal) sorted by machine_id."""
    n = len(machines)
    index = {m[0]: i for i, m in enumerate(machines)}
    C = np.full((n, T), 5, dtype=np.int64)          # 5 = missing / unknown
    for machine_id, tick, payout in rows:
        if 0 <= tick < cut and machine_id in index:
            C[index[machine_id], tick] = PIDX[payout]
    ok = C < 5
    onehot = np.eye(6)[C]
    feats = []
    for k in range(1, LAGS + 1):
        shifted = np.zeros_like(onehot); shifted[:, :, 5] = 1; shifted[:, k:] = onehot[:, :-k]
        feats.append(shifted)
    win = np.where(ok, (C > 0) & (C < 5), False).astype(float)
    obs = ok.astype(float)
    for k in range(1, LAGS + 1):
        wk = np.zeros_like(win); ok_k = np.zeros_like(obs)
        wk[:, k:] = win[:, :-k]; ok_k[:, k:] = obs[:, :-k]
        sw = wk.sum(0, keepdims=True) - wk; so = ok_k.sum(0, keepdims=True) - ok_k
        feats.append(np.stack([sw / np.maximum(so, 1), so / (n - 1)], -1))
    static = np.zeros((n, T, 5))
    for i, (_, mtype, signal) in enumerate(machines):
        static[i, :, TYPES[mtype]] = 1
        static[i, :, 3] = signal
    static[:, :, 4] = np.arange(T) / 256
    X = np.concatenate(feats + [static], -1)
    Y = np.where(ok, C, -1); Y[:, cut:] = -1
    return X, Y


def build_split(folder):
    """Returns X (N,T,D), Y (N,T), cut (N,), query_ids (N,) with one row per query."""
    queries, history = load_split(folder)
    episodes = {}
    for q in queries:
        episodes.setdefault(q["episode_id"], []).append(q)
    Xs, Ys, cuts, qids = [], [], [], []
    for ep in sorted(episodes):
        qs = sorted(episodes[ep], key=lambda q: q["machine_id"])
        cut = int(qs[0]["cutoff"])
        if any(int(q["cutoff"]) != cut for q in qs):
            raise ValueError(f"Mixed cutoffs in episode {ep}")
        machines = [(q["machine_id"], q["machine_type"], int(float(q["signal"]))) for q in qs]
        X, Y = episode_features(machines, cut, history.get(ep, []))
        Xs.append(X); Ys.append(Y); cuts += [cut] * len(qs); qids += [q["query_id"] for q in qs]
    return np.concatenate(Xs), np.concatenate(Ys), np.array(cuts), np.array(qids)

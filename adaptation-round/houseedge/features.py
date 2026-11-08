"""Build per-episode input tensors from the public CSVs (numpy + stdlib only).

Every input for tick t uses only observations at ticks < t, and nothing at or after
the query cutoff is read. Missing ticks are encoded as a separate 'missing' category.
"""
import csv
import os
from pathlib import Path
import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PIDX = {p: i for i, p in enumerate(PAYOUTS)}
TYPES = {"steady": 0, "balanced": 1, "jackpot": 2}
T = int(os.environ.get("HE_T", "257"))        # ticks 0..T-1 (cutoffs up to T-1)
LAGS = int(os.environ.get("HE_LAGS", "8"))    # lags used as network inputs


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


def episode_features(machines, cut, rows, extended=True):
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
    if extended:
        feats.append(static); static = running_stats(C)
    X = np.concatenate(feats + [static], -1)
    Y = np.where(ok, C, -1); Y[:, cut:] = -1
    return X, Y


def build_split(folder, extended=True):
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
        X, Y = episode_features(machines, cut, history.get(ep, []), extended)
        Xs.append(X); Ys.append(Y); cuts += [cut] * len(qs); qids += [q["query_id"] for q in qs]
    return np.concatenate(Xs), np.concatenate(Ys), np.array(cuts), np.array(qids)


# ---------------------------------------------------------------------------
# Per-machine running statistics (v4 inputs). All sums are causal: the value at
# tick t only aggregates targets observed at ticks < t.
HALF_LIVES = (None, 48.0)
PAIRS = [(a, b) for a in range(1, LAGS + 1) for b in range(a + 1, LAGS + 1)]


def _kernel(hl):
    t = np.arange(T, dtype=np.float64)
    d = t[:, None] - 1 - t[None, :]
    k = np.where(d >= 0, 0.5 ** (np.maximum(d, 0) / hl) if hl else 1.0, 0.0)
    return k


_K = {hl: _kernel(hl) for hl in HALF_LIVES}


def _ewsum(x, hl):
    """x (M,T,F) -> causal (exponentially weighted) sums over earlier ticks."""
    return np.einsum("ts,msf->mtf", _K[hl], x)


def _lags(win, obs):
    LW, LO = {}, {}
    for k in range(1, LAGS + 1):
        lw = np.zeros_like(win); lo = np.zeros_like(obs)
        lw[:, k:] = win[:, :-k]; lo[:, k:] = obs[:, :-k]
        LW[k], LO[k] = lw, lo
    return LW, LO


def running_stats(C):
    """C (M,T) categories with 5 = missing. Returns (M,T,44+84)."""
    win = ((C > 0) & (C < 5)).astype(np.float64); obs = (C < 5).astype(np.float64)
    LW, LO = _lags(win, obs)
    q = []
    for k in range(1, LAGS + 1):
        a1 = LW[k] * LO[k] * obs; a0 = (1 - LW[k]) * LO[k] * obs
        q += [a1, a1 * win, a0, a0 * win]
    q += [obs, win] + [(C == j).astype(np.float64) for j in (1, 2, 3, 4)]
    Q = np.stack(q, -1)
    out = []
    for hl in HALF_LIVES:
        S = _ewsum(Q, hl); n_obs = S[..., 4 * LAGS]; n_win = S[..., 4 * LAGS + 1]
        pbar = (n_win + 1) / (n_obs + 2)
        for k in range(LAGS):
            n1, s1, n0, s0 = S[..., 4 * k], S[..., 4 * k + 1], S[..., 4 * k + 2], S[..., 4 * k + 3]
            out += [(s1 + 4 * pbar) / (n1 + 4) - pbar, (s0 + 4 * pbar) / (n0 + 4) - pbar]
        out.append(pbar)
        for j in range(4):
            out.append((S[..., 4 * LAGS + 2 + j] + 0.5) / (n_win + 2))
        out.append(np.log1p(n_obs) / 6)
    single = np.stack(out, -1)
    # lag-pair statistics: empirical win rate for the pair state active at tick t
    q, cur = [], []
    for a, b in PAIRS:
        ob = LO[a] * LO[b]; combo = 2 * LW[a] + LW[b]
        for c in range(4):
            ind = ob * (combo == c); cur.append(ind); n = ind * obs; q += [n, n * win]
    Q = np.stack(q, -1); CUR = np.stack(cur, -1)
    M = C.shape[0]
    pair_out = []
    for hl in HALF_LIVES:
        S = _ewsum(Q, hl); Sn = S[..., 0::2]; Ss = S[..., 1::2]
        no = _ewsum(obs[..., None], hl)[..., 0]; nw = _ewsum(win[..., None], hl)[..., 0]
        pbar = ((nw + 1) / (no + 2))[..., None]
        rate = (Ss + 4 * pbar) / (Sn + 4) - pbar
        pair_out.append((rate * CUR).reshape(M, T, len(PAIRS), 4).sum(-1))
        if hl is None:
            pair_out.append((np.log1p(Sn) * CUR).reshape(M, T, len(PAIRS), 4).sum(-1) / 5)
    return np.concatenate([single] + pair_out, -1)

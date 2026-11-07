"""Exact Bayesian model for payout sizes.

In this simulator every machine type has exactly two payout-size distributions
(given a win). A machine uses the "high" one when the payout k ticks earlier was
2x or more (or the opposite, depending on the machine), for one machine-specific
lag k from a small candidate set. We average over all (k, direction) rules,
weighted by how well each explains the machine's observed winning payouts.
"""
import json
import numpy as np


def load_size_config(path):
    cfg = json.loads(open(path, encoding="utf-8").read())
    vectors = {t: (np.array(v["low"]), np.array(v["high"])) for t, v in cfg["vectors"].items()}
    return vectors, tuple(cfg["lags"]), int(cfg["threshold_index"])


def high_probability(Y_row, cut, low, high, lags, thr_idx):
    """Y_row: (T,) categories 0..4 with -1 missing (nothing at or after cut). Returns P(high at cut)."""
    ll_lo, ll_hi = np.log(low), np.log(high)
    ticks = np.where(Y_row[:cut] >= 1)[0]                     # observed wins before the cutoff
    j = Y_row[ticks] - 1
    log_w, says = [], []
    for k in lags:
        prev = np.where(ticks - k >= 0, Y_row[np.maximum(ticks - k, 0)], -1)
        known = prev >= 0
        big = prev >= thr_idx
        mixed = np.logaddexp(ll_hi[j] + np.log(0.5), ll_lo[j] + np.log(0.5))
        at_cut = Y_row[cut - k] if cut - k >= 0 else -1
        for flip in (0, 1):
            state_high = np.logical_xor(big, bool(flip))
            ll = np.where(known, np.where(state_high, ll_hi[j], ll_lo[j]), mixed).sum()
            log_w.append(ll - np.log(2 * len(lags)))
            says.append(0.5 if at_cut < 0 else float((at_cut >= thr_idx) != bool(flip)))
    log_w = np.array(log_w)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return float(w @ np.array(says))


def size_distribution(Y, cuts, types, vectors, lags, thr_idx):
    out = np.zeros((len(cuts), 4))
    for i in range(len(cuts)):
        low, high = vectors[types[i]]
        p = high_probability(Y[i], int(cuts[i]), low, high, lags, thr_idx)
        out[i] = p * high + (1 - p) * low
    return out

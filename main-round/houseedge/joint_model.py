"""Joint Bayesian rule model (numpy + stdlib).

Main-round structure found in the public data:
- win chance follows an XOR rule on one lag pair (a, b) per machine (machine-specific sign);
- the payout-size regime is set by whether the payout at lag k = a was 2x or more
  (machine-specific direction), choosing between exactly two size vectors per type.
Both kinds of evidence identify the same pair, so they are combined.

`high_probability` -> P(high size regime at the cutoff), robust to win-rule sign flips
(win evidence uses free cell rates inside 32-tick chunks).
`structured_win` -> P(win at the cutoff) under a static rule (used only for short histories).
"""
from math import lgamma
import numpy as np


def betaln(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _pair_cells(Y, cut, a, b):
    t = np.arange(cut)
    ok = (Y[:cut] >= 0) & (t - a >= 0) & (t - b >= 0)
    ya = np.where(t - a >= 0, Y[np.maximum(t - a, 0)], -1)
    yb = np.where(t - b >= 0, Y[np.maximum(t - b, 0)], -1)
    ok &= (ya >= 0) & (yb >= 0)
    x = (ya >= 1) != (yb >= 1)
    win = Y[:cut] >= 1
    return t, ok, x, win


def _size_ll(Y, cut, k, lo, hi, thr):
    ticks = np.where(Y[:cut] >= 1)[0]
    j = Y[ticks] - 1
    lh, ll = np.log(hi[j]), np.log(lo[j])
    prev = np.where(ticks - k >= 0, Y[np.maximum(ticks - k, 0)], -1)
    known = prev >= 0
    big = prev >= thr
    mixed = np.logaddexp(lh, ll) + np.log(0.5)
    at_cut = Y[cut - k] if cut - k >= 0 else -1
    res = []
    for flip in (0, 1):
        st = np.logical_xor(big, bool(flip))
        L = np.where(known, np.where(st, lh, ll), mixed).sum()
        says = 0.5 if at_cut < 0 else float((at_cut >= thr) != bool(flip))
        res.append((L, says))
    return res


def high_probability(Y, cut, lo, hi, rules, thr, chunk=32, kappa=4.0):
    obs = Y[:cut] >= 0
    nw = (Y[:cut] >= 1).sum(); no = obs.sum()
    pbar = (nw + 1) / (no + 2)
    A, B = kappa * pbar, kappa * (1 - pbar)
    L, says = [], []
    for a, b, k in rules:
        Lw = 0.0
        if a is not None:
            t, ok, x, win = _pair_cells(Y, cut, a, b)
            ch = (cut - 1 - t) // chunk
            for c in np.unique(ch[ok]):
                sel = ok & (ch == c)
                for xv in (True, False):
                    s2 = sel & (x == xv); n = int(s2.sum()); s = int(win[s2].sum())
                    Lw += betaln(A + s, B + n - s) - betaln(A, B)
        for Ls, sy in _size_ll(Y, cut, k, lo, hi, thr):
            L.append(Lw + Ls); says.append(sy)
    L = np.array(L); w = np.exp(L - L.max()); w /= w.sum()
    return float(w @ np.array(says))


def structured_win(Y, cut, lo, hi, rules, thr, c_hi, c_lo, kappa=3.0):
    L, PR = [], []
    for a, b, k in rules:
        t, ok, x, win = _pair_cells(Y, cut, a, b)
        sl = _size_ll(Y, cut, k, lo, hi, thr)
        Ls = np.logaddexp(sl[0][0], sl[1][0]) + np.log(0.5)
        ya = Y[cut - a] if cut - a >= 0 else -1; yb = Y[cut - b] if cut - b >= 0 else -1
        qx = None if (ya < 0 or yb < 0) else bool((ya >= 1) != (yb >= 1))
        for s in (0, 1):
            hx = x if s == 0 else ~x
            nh = int((ok & hx).sum()); sh = int(win[ok & hx].sum()); nl = int((ok & ~hx).sum()); sl_ = int(win[ok & ~hx].sum())
            Lw = (betaln(kappa * c_hi + sh, kappa * (1 - c_hi) + nh - sh) - betaln(kappa * c_hi, kappa * (1 - c_hi))
                  + betaln(kappa * c_lo + sl_, kappa * (1 - c_lo) + nl - sl_) - betaln(kappa * c_lo, kappa * (1 - c_lo)))
            ph = (kappa * c_hi + sh) / (kappa + nh); pl = (kappa * c_lo + sl_) / (kappa + nl)
            p = 0.5 * (ph + pl) if qx is None else (ph if (qx if s == 0 else not qx) else pl)
            L.append(Lw + Ls); PR.append(p)
    L = np.array(L); w = np.exp(L - L.max()); w /= w.sum()
    return float(w @ np.array(PR))

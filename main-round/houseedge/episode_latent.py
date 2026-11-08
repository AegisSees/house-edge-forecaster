"""Episode-latent correction (team analysis, 2026-11-08).
Machines in one episode share a slowly drifting luck term in the win logit. For each query:
logit(win) += beta * x, x = half-life-8 exponentially weighted mean of the OTHER machines' one-step residuals
(observed win minus the GRU ensemble's in-sample win chance) over ticks < cutoff. Skipped for short histories.
beta is fitted by maximum likelihood on the split's own history ticks (no labels or targets), grid 0..4 step 0.25."""
import numpy as np
from houseedge.gru_numpy import gelu, _gru

HALF_LIFE = 8.0


def _all_ticks(w, X):
    D = w["inp.0.weight"].shape[1]; X = X[..., :D]; z = gelu(X @ w["inp.0.weight"].T + w["inp.0.bias"])
    if "g1.weight_ih_l0" in w:
        h1 = _gru(z, w["g1.weight_ih_l0"], w["g1.weight_hh_l0"], w["g1.bias_ih_l0"], w["g1.bias_hh_l0"]); H = h1.shape[-1]
        if w["g2.weight_ih_l0"].shape[1] == 2 * H:
            N, TT, _ = h1.shape; h = h1.reshape(N // 8, 8, TT, H); mix = (h.sum(1, keepdims=True) - h) / 7
            h1 = np.concatenate([h, mix], -1).reshape(N, TT, 2 * H)
        hh = _gru(h1, w["g2.weight_ih_l0"], w["g2.weight_hh_l0"], w["g2.bias_ih_l0"], w["g2.bias_hh_l0"])
    else:
        hh = z
        for l in sorted({int(k.split("_l")[1]) for k in w if k.startswith("rnn.weight_ih_l")}):
            hh = _gru(hh, w[f"rnn.weight_ih_l{l}"], w[f"rnn.weight_hh_l{l}"], w[f"rnn.bias_ih_l{l}"], w[f"rnn.bias_hh_l{l}"])
    feat = np.concatenate([hh, z], -1); hid = gelu(feat @ w["head.1.weight"].T + w["head.1.bias"])
    lg = hid @ w["head.3.weight"].T + w["head.3.bias"]; lg -= lg.max(-1, keepdims=True); p = np.exp(lg); p /= p.sum(-1, keepdims=True)
    return 1 - p[..., 0]


def latent_shift(model_paths, X, Y, cuts, episodes):
    """Returns (shift per query in log-odds before beta, fitted beta)."""
    L = []
    for path in model_paths:
        with np.load(path) as z:
            w = {k: z[k].astype(np.float64) for k in z.files}
        pw = np.clip(_all_ticks(w, X), 1e-6, 1 - 1e-6); L.append(np.log(pw / (1 - pw)))
    off = np.mean(L, 0); pw = 1 / (1 + np.exp(-off))
    N, T = Y.shape; obs = Y >= 0; W = ((Y > 0) & obs).astype(float)
    R = np.where(obs, W - pw, 0.0); O = obs.astype(float)
    Ro = np.zeros_like(R); Oo = np.zeros_like(O)
    for e in np.unique(episodes):                       # sums over the other machines of the same episode
        m = episodes == e; Ro[m] = R[m].sum(0) - R[m]; Oo[m] = O[m].sum(0) - O[m]
    dec = 0.5 ** (1 / HALF_LIFE); num = np.zeros_like(Ro); den = np.zeros_like(Oo)
    for t in range(1, T):
        num[:, t] = dec * num[:, t - 1] + Ro[:, t - 1]; den[:, t] = dec * den[:, t - 1] + Oo[:, t - 1]
    x = num / np.maximum(den, 1e-9)
    m = obs & (np.arange(T)[None, :] >= 16); xx, yy, oo = x[m], W[m], off[m]
    def ll(b):
        p = 1 / (1 + np.exp(-(oo + b * xx))); return (yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum()
    grid = np.arange(0, 4.01, 0.25); beta = float(grid[int(np.argmax([ll(b) for b in grid]))])
    return x[np.arange(N), cuts], beta

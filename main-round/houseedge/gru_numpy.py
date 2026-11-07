"""Float64 numpy forward pass of the trained GRU forecaster (no torch needed)."""
import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(weights, X, cuts):
    """weights: dict of float64 arrays named like torch state_dict. Returns (N,5) probs at cuts."""
    w = weights
    z = gelu(X @ w["inp.0.weight"].T + w["inp.0.bias"])
    layers = sorted({int(k.split("_l")[1]) for k in w if k.startswith("rnn.weight_ih_l")})
    h_in = z
    for l in layers:
        Wih, Whh = w[f"rnn.weight_ih_l{l}"], w[f"rnn.weight_hh_l{l}"]
        bih, bhh = w[f"rnn.bias_ih_l{l}"], w[f"rnn.bias_hh_l{l}"]
        H = Whh.shape[1]
        gi = h_in @ Wih.T + bih                      # (N,T,3H)
        h = np.zeros((X.shape[0], H))
        out = np.empty((X.shape[0], X.shape[1], H))
        for t in range(X.shape[1]):
            gh = h @ Whh.T + bhh
            r = sigmoid(gi[:, t, :H] + gh[:, :H])
            u = sigmoid(gi[:, t, H:2 * H] + gh[:, H:2 * H])
            n = np.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
            h = (1 - u) * n + u * h
            out[:, t] = h
        h_in = out
    rows = np.arange(X.shape[0])
    feat = np.concatenate([h_in[rows, cuts], z[rows, cuts]], -1)
    hid = gelu(feat @ w["head.1.weight"].T + w["head.1.bias"])
    logits = hid @ w["head.3.weight"].T + w["head.3.bias"]
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def _gru(x, Wih, Whh, bih, bhh):
    H = Whh.shape[1]
    gi = x @ Wih.T + bih
    h = np.zeros((x.shape[0], H))
    out = np.empty((x.shape[0], x.shape[1], H))
    for t in range(x.shape[1]):
        gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, t, :H] + gh[:, :H])
        u = sigmoid(gi[:, t, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - u) * n + u * h
        out[:, t] = h
    return out


def forward_v2(w, X, cuts, machines_per_episode=8):
    """Two stacked GRUs (g1, g2); with mixing, g2 also sees the mean of the other machines' g1 states."""
    z = gelu(X @ w["inp.0.weight"].T + w["inp.0.bias"])
    h1 = _gru(z, w["g1.weight_ih_l0"], w["g1.weight_hh_l0"], w["g1.bias_ih_l0"], w["g1.bias_hh_l0"])
    H = h1.shape[-1]
    if w["g2.weight_ih_l0"].shape[1] == 2 * H:
        N, TT, _ = h1.shape; M = machines_per_episode
        h = h1.reshape(N // M, M, TT, H)
        mix = (h.sum(1, keepdims=True) - h) / (M - 1)
        h1 = np.concatenate([h, mix], -1).reshape(N, TT, 2 * H)
    h2 = _gru(h1, w["g2.weight_ih_l0"], w["g2.weight_hh_l0"], w["g2.bias_ih_l0"], w["g2.bias_hh_l0"])
    rows = np.arange(X.shape[0])
    feat = np.concatenate([h2[rows, cuts], z[rows, cuts]], -1)
    hid = gelu(feat @ w["head.1.weight"].T + w["head.1.bias"])
    logits = hid @ w["head.3.weight"].T + w["head.3.bias"]
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def predict_any(w, X, cuts):
    """Dispatch on weight names; models use the first D input columns (base 69, +44, +84)."""
    D = w["inp.0.weight"].shape[1]
    if D > X.shape[-1]:
        raise ValueError(f"Unexpected input width {D}")
    X = X[..., :D]
    if "g1.weight_ih_l0" in w:
        return forward_v2(w, X, cuts)
    return forward(w, X, cuts)

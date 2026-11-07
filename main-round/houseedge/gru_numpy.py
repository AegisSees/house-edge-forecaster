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

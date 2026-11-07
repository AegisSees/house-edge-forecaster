"""Recreate the House Edge forecast CSV from saved models (numpy only, no training).

Win chance: median over the GRU forecasters in models/*.npz (blended with a
structured rule model for short histories).
Payout size given a win: exact Bayesian rule model (houseedge/joint_model.py).
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from houseedge.features import build_split, read_csv
from houseedge.gru_numpy import predict_any
from houseedge.joint_model import high_probability, structured_win
import json

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="public/ folder of the current round")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    folder = Path(a.data) / a.split
    X, Y, cuts, qids = build_split(folder)
    qrows = {r["query_id"]: r for r in read_csv(folder / "queries.csv")}
    types = [qrows[q]["machine_type"] for q in qids]
    signals = [int(float(qrows[q]["signal"])) for q in qids]
    models = sorted((HERE / "models").glob("*.npz"))
    if not models:
        raise SystemExit("No model weights found in models/")
    wins = []
    for path in models:
        with np.load(path) as z:
            w = {k: z[k].astype(np.float64) for k in z.files}
        wins.append(1.0 - predict_any(w, X, cuts)[:, 0])
        print(f"  {path.name}", flush=True)
    win = np.median(np.stack(wins), 0)
    P = None
    cfg_path = HERE / "models" / "rules.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        rules = [tuple(r) for r in cfg["rules"]]; thr = cfg["threshold_index"]
        sizes = np.zeros((len(qids), 4))
        for i, q in enumerate(qids):
            vec = cfg["vectors"][types[i]]; lo, hi = np.array(vec["low"]), np.array(vec["high"])
            ph = high_probability(Y[i], int(cuts[i]), lo, hi, rules, thr, cfg["chunk"], cfg["kappa_size"])
            sizes[i] = ph * hi + (1 - ph) * lo
            key = f"{types[i]}|{signals[i]}"
            if cfg["short_blend"] > 0 and (Y[i, :cuts[i]] >= 0).sum() < cfg["short_obs"] and key in cfg["centers"]:
                c_hi, c_lo = cfg["centers"][key]
                sw = structured_win(Y[i], int(cuts[i]), lo, hi, rules, thr, c_hi, c_lo, cfg["kappa_win"])
                win[i] = cfg["short_blend"] * sw + (1 - cfg["short_blend"]) * win[i]
        P = np.column_stack([1.0 - win, win[:, None] * sizes])
    if P is None:
        raise SystemExit("models/rules.json is required")
    P = np.clip(P, 1e-12, 1.0)
    P /= P.sum(1, keepdims=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for q, p in zip(qids, P):
            wr.writerow([q] + [repr(float(v)) for v in p])
    print(f"Wrote {len(qids)} forecasts from {len(models)} model(s) to {out}")


if __name__ == "__main__":
    main()

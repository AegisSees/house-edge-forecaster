"""Recreate the House Edge forecast CSV from saved models (numpy only, no training).

Win chance: geometric-mean ensemble of the GRU forecasters in models/*.npz.
Payout size given a win: exact Bayesian rule model (houseedge/size_model.py).
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from houseedge.features import build_split, read_csv
from houseedge.gru_numpy import predict_any
from houseedge.size_model import load_size_config, size_distribution

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="public/ folder of the current round")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    folder = Path(a.data) / a.split
    X, Y, cuts, qids = build_split(folder)
    qtype = {r["query_id"]: r["machine_type"] for r in read_csv(folder / "queries.csv")}
    types = [qtype[q] for q in qids]
    models = sorted((HERE / "models").glob("*.npz"))
    if not models:
        raise SystemExit("No model weights found in models/")
    logp = np.zeros((len(qids), 5))
    for path in models:
        with np.load(path) as z:
            w = {k: z[k].astype(np.float64) for k in z.files}
        logp += np.log(np.clip(predict_any(w, X, cuts), 1e-9, 1.0))
        print(f"  {path.name}", flush=True)
    P = np.exp(logp / len(models))
    P /= P.sum(1, keepdims=True)
    win = 1.0 - P[:, 0]
    cfg_path = HERE / "models" / "size_config.json"
    if cfg_path.exists():   # exact size model (skipped if the round has no size config)
        vectors, lags, thr = load_size_config(cfg_path)
        sizes = size_distribution(Y, cuts, types, vectors, lags, thr)
        P = np.column_stack([1.0 - win, win[:, None] * sizes])
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

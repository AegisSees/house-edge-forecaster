"""Recreate the House Edge forecast CSV from saved GRU weights (numpy only, no training)."""
import argparse
import csv
from pathlib import Path
import numpy as np
from houseedge.features import build_split
from houseedge.gru_numpy import forward

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="public/ folder of the current round")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    X, _, cuts, qids = build_split(Path(a.data) / a.split)
    models = sorted((HERE / "models").glob("*.npz"))
    if not models:
        raise SystemExit("No model weights found in models/")
    P = np.zeros((len(qids), 5))
    for path in models:
        with np.load(path) as z:
            w = {k: z[k].astype(np.float64) for k in z.files}
        P += forward(w, X, cuts)
    P /= len(models)
    P = np.clip(P, 1e-9, 1.0)
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

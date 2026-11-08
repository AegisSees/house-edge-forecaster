#!/usr/bin/env bash
# Package and verify the round-two entry made by run_round2.sh.
# usage: bash package_round2.sh <kit root (has build_entry.py, check_submission.py, public/)> <work dir> <out dir>
set -euo pipefail
KIT=$(readlink -f "$1"); W=$(readlink -f "$2"); OUT=$(readlink -f "$3"); R2=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
E="$W/entry"
find "$E" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$E/training/round2"
cp "$R2/run_round2.sh" "$R2/package_round2.sh" "$R2/final_forecast.sh" "$R2/latent_choice.py" "$R2/add_batch.sh" "$R2/partner_params.py" "$R2/partner_refit.py" "$R2/blend_fit.py" "$E/training/round2/"
cp "$R2/train/v3.py" "$R2/train/gru.py" "$E/training/round2/"
cp "$W"/*.log "$E/training/round2/" 2>/dev/null || true
python3 - "$E" "$W" <<'PY'
import json, sys, pathlib
E, W = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
r = json.loads((E / "models/rules.json").read_text()); b = json.loads((E / "models/blend.json").read_text())
nets = sorted(p.name for p in (E / "models").glob("*.npz"))
txt = f"""

## Round two (adaptation) refit

Everything below was re-learned from the adaptation kit's training data by `training/round2/run_round2.sh`
(one command; logs are in `training/round2/`). Validation targets were used only to choose the blend.

- Win rules found (lag a, lag b, size lag): {r['rules']}; network input lags {r.get('lags')}, ticks {r.get('ticks')}.
- Win-chance networks retrained on this kit: {', '.join(nets)}.
- Teammate models refit on this kit: partner (joint Bayesian filter, `partner_refit.py`), partner2 (`partner2/params.json`, `partner_params.py`).
- Blend (`models/blend.json`): partners {b['partners']}, choice {b.get('choice')}, weights {b['weights']},
  payout sizes for full histories from {b.get('full_size_partner') or 'ours'}, for short histories from {b.get('short_size_partner') or 'ours'}.

The command to recreate the forecast is unchanged: `python predict.py --data /data --out /output/submission.csv`.
"""
p = E / "README.md"; s = p.read_text()
s = s.split("\n\n## Round two (adaptation) refit")[0]
p.write_text(s + txt)
PY
rm -f "$OUT/code-entry.zip"
(cd "$KIT" && python3 build_entry.py --source "$E" --out "$OUT/code-entry.zip")
cp "$W/test_final.csv" "$OUT/submission.csv"; cp "$W/val_final.csv" "$OUT/validation_predictions.csv"
# reproduce from the ZIP in a clean folder with 2 threads, compare with the CSV
RD=$(mktemp -d); (cd "$RD" && python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$OUT/code-entry.zip")
SRC=$(dirname "$(find "$RD" -name reproduce.json | head -1)")
t0=$(date +%s)
SPLIT=${REPRO_SPLIT:-validation}   # validation is half the size of test and runs the same code path
(cd "$SRC" && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 predict.py --data "$KIT/public" --split "$SPLIT" --out "$RD/repro.csv") > "$RD/repro.log" 2>&1
echo "reproduction ($SPLIT) took $(( $(date +%s) - t0 ))s"
REF="$OUT/submission.csv"; [ "$SPLIT" = validation ] && REF="$OUT/validation_predictions.csv"
python3 - "$REF" "$RD/repro.csv" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_csv(sys.argv[1]).set_index('query_id').sort_index(); b = pd.read_csv(sys.argv[2]).set_index('query_id').sort_index()
assert list(a.index) == list(b.index), 'query ids differ'
d = np.abs(a.values - b.values).max(); print('reproduced max abs diff', d); assert d < 1e-8, 'REPRODUCTION MISMATCH'
PY
(cd "$KIT" && python3 check_submission.py "$OUT/submission.csv" --split test --data "$KIT/public") 2>&1 | tail -3 || true
(cd "$KIT" && python3 check_submission.py "$OUT/validation_predictions.csv" --split validation --data "$KIT/public") 2>&1 | grep -i "points\|ratio" || true
ls -la "$OUT"

#!/usr/bin/env bash
# Final validation forecast, episode-latent on/off choice on validation, then the test forecast.
# usage: bash final_forecast.sh <public folder> <work dir>
set -euo pipefail
PUBDIR=$(readlink -f "$1"); W=$(readlink -f "$2"); R2=$(dirname "$(readlink -f "$0")")
python3 - "$W/entry/models/blend.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); b['episode_latent'] = True; json.dump(b, open(sys.argv[1], 'w'), indent=1)
PY
(cd "$W/entry" && OMP_NUM_THREADS=4 HE_DUMP_NOLAT="$W/val_final_nolat.csv" python3 predict.py --data "$PUBDIR" --split validation --out "$W/val_final.csv") > "$W/pred_val_final.log" 2>&1
python3 "$R2/latent_choice.py" "$PUBDIR" "$W/val_final.csv" "$W/val_final_nolat.csv" "$W/entry/models/blend.json" | tee "$W/latent_choice.log"
(cd "$W/entry" && OMP_NUM_THREADS=4 python3 predict.py --data "$PUBDIR" --split test --out "$W/test_final.csv") > "$W/pred_test_final.log" 2>&1

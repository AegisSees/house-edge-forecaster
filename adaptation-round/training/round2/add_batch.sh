#!/usr/bin/env bash
# Add one more batch of nets to an existing round-two work dir (made by run_round2.sh), then re-blend and re-forecast.
# usage: bash add_batch.sh <public folder> <work dir> <batch number, e.g. 2>
set -euo pipefail
PUBDIR=$(readlink -f "$1"); W=$(readlink -f "$2"); b=$3
R2=$(dirname "$(readlink -f "$0")"); t0=$(date +%s)
log(){ echo "[$(( $(date +%s) - t0 ))s] $*"; }
LAGS=$(python3 -c "import json;print(json.load(open('$W/entry/models/rules.json'))['lags'])")
TICKS=$(python3 -c "import json;print(json.load(open('$W/entry/models/rules.json'))['ticks'])")
log "train batch $b (lags $LAGS ticks $TICKS)"
s1=$((10*b+1)); s2=$((10*b+2)); s3=$((10*b+3))
for s in $s1 $s2 $s3; do HE_LAGS=$LAGS HE_T=$TICKS ENTRY="$W/entry" KIT=$(dirname "$PUBDIR") OUTD="$W" OMP_NUM_THREADS=1 \
    python3 "$R2/train/v3.py" --threads 1 --ep "${EP:-28}" --evalfrom $(( ${EP:-28} - 2 )) --seed $s --tag F$s > "$W/log_F$s.txt" 2>&1 & done
HE_LAGS=$LAGS HE_T=$TICKS ENTRY="$W/entry" KIT=$(dirname "$PUBDIR") OUTD="$W" OMP_NUM_THREADS=1 THREADS=1 \
    python3 "$R2/train/gru.py" --ep "${EP:-28}" --hid 128 --layers 2 --seed $b --tag E$b > "$W/log_E$b.txt" 2>&1 &
wait
grep -h "^$(( ${EP:-28} - 1 )) " "$W"/log_F$s1.txt "$W"/log_F$s2.txt "$W"/log_F$s3.txt "$W"/log_E$b.txt | cut -c1-120 || true
for f in "$W"/w_*.npz; do n=$(basename "$f" .npz); cp "$f" "$W/entry/models/gru_${n#w_}.npz"; done
cp "$W/entry/models/blend.json" "$W/blend_batch$((b-1)).json"
python3 - "$W/entry/models/blend.json" <<'PY'
import json, sys
json.dump({"partners": [], "gate_threshold": {}, "weights": {"low": [1.0], "high": [1.0], "short_low": [1.0], "short_high": [1.0]},
           "short_size_partner": None}, open(sys.argv[1], "w"), indent=1)
PY
log "our validation forecast with $(ls "$W"/entry/models/*.npz | wc -l) nets"
(cd "$W/entry" && OMP_NUM_THREADS=4 python3 predict.py --data "$PUBDIR" --split validation --out "$W/val_ours.csv") > "$W/pred_val_ours.log" 2>&1
BL=("$W/val_ours.csv")
[ -s "$W/val_partner.csv" ] && BL+=("$W/val_partner.csv:partner")
[ -s "$W/val_partner2.csv" ] && BL+=("$W/val_partner2.csv:partner2")
if [ ${#BL[@]} -gt 1 ]; then python3 "$R2/blend_fit.py" --kit "$PUBDIR" --out "$W/entry/models/blend.json" "${BL[@]}" | tee "$W/blend_fit_batch$b.log"; fi
log "final validation and test forecasts"
bash "$R2/final_forecast.sh" "$PUBDIR" "$W"
log "done; compare blend_fit_batch$b.log cv_ratio with the previous blend_fit.log before resubmitting"

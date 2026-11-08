#!/usr/bin/env bash
# Round-two (adaptation) one-command pipeline.
# usage: bash run_round2.sh <public folder of the new kit> <work dir> [n_batches=1]
# Result: <work>/entry (ready to zip), <work>/val_final.csv, <work>/test_final.csv
set -euo pipefail
PUBDIR=$(readlink -f "$1"); W=$(readlink -f "$2"); NB=${3:-1}
R2=$(dirname "$(readlink -f "$0")")
mkdir -p "$W"; t0=$(date +%s)
log(){ echo "[$(( $(date +%s) - t0 ))s] $*"; }
rm -rf "$W/entry"; cp -r "$R2/entry" "$W/entry"; find "$W/entry" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
rm -f "$W"/entry/models/*.npz "$W"/entry/models/rules.json "$W"/entry/models/blend.json "$W"/entry/partner2/params.json "$W"/w_*.npz
log "fit rules"
(cd "$W/entry" && OMP_NUM_THREADS=4 python3 fit_rules.py --data "$PUBDIR" --out models/rules.json) | tee "$W/fit_rules.log"
python3 "$R2/partner_params.py" "$W/entry/models/rules.json" "$PUBDIR" "$W/entry/partner2/params.json" | tail -1 | tee "$W/partner_params.log"
LAGS=$(python3 -c "import json;print(json.load(open('$W/entry/models/rules.json'))['lags'])")
TICKS=$(python3 -c "import json;print(json.load(open('$W/entry/models/rules.json'))['ticks'])")
log "lags $LAGS ticks $TICKS; partner2 validation in background"
(cd "$W/entry/partner2" && OMP_NUM_THREADS=1 python3 predict.py --data "$PUBDIR" --split validation --out "$W/val_partner2.csv" --procs 1) > "$W/partner2_val.log" 2>&1 &
P2=$!
log "refit the friend's joint filter (win pairs within lags 1-8) and its validation forecast in background"
( python3 "$R2/partner_refit.py" "$W/entry/partner" "$PUBDIR" "$W/entry/partner/model.json" "$R2/entry/partner/model.json" > "$W/partner_refit.log" 2>&1 && \
  cd "$W/entry/partner" && OMP_NUM_THREADS=1 python3 partner_predict.py --data "$PUBDIR" --split validation --out "$W/val_partner.csv" ) > "$W/partner_val.log" 2>&1 &
P1=$!
for b in $(seq 1 "$NB"); do
  log "train batch $b"
  s1=$((10*b+1)); s2=$((10*b+2)); s3=$((10*b+3))
  for s in $s1 $s2 $s3; do HE_LAGS=$LAGS HE_T=$TICKS ENTRY="$W/entry" KIT=$(dirname "$PUBDIR") OUTD="$W" OMP_NUM_THREADS=1 \
      python3 "$R2/train/v3.py" --threads 1 --ep "${EP:-28}" --evalfrom $(( ${EP:-28} - 2 )) --seed $s --tag F$s > "$W/log_F$s.txt" 2>&1 & done
  HE_LAGS=$LAGS HE_T=$TICKS ENTRY="$W/entry" KIT=$(dirname "$PUBDIR") OUTD="$W" OMP_NUM_THREADS=1 THREADS=1 \
      python3 "$R2/train/gru.py" --ep "${EP:-28}" --hid 128 --layers 2 --seed $b --tag E$b > "$W/log_E$b.txt" 2>&1 &
  for j in $(jobs -p); do [ "$j" = "$P2" ] || [ "$j" = "$P1" ] || wait "$j"; done
  grep -h "^$(( ${EP:-28} - 1 )) " "$W"/log_F*.txt "$W"/log_E*.txt | cut -c1-120 || true
done
for f in "$W"/w_*.npz; do n=$(basename "$f" .npz); cp "$f" "$W/entry/models/gru_${n#w_}.npz"; done
python3 - "$W/entry/models/blend.json" <<'PY'
import json, sys
json.dump({"partners": [], "gate_threshold": {}, "weights": {"low": [1.0], "high": [1.0], "short_low": [1.0], "short_high": [1.0]},
           "short_size_partner": None}, open(sys.argv[1], "w"), indent=1)
PY
log "our validation forecast"
(cd "$W/entry" && OMP_NUM_THREADS=2 python3 predict.py --data "$PUBDIR" --split validation --out "$W/val_ours.csv") > "$W/pred_val_ours.log" 2>&1
wait "$P2" || log "partner2 validation FAILED (see partner2_val.log)"
wait "$P1" || log "friend refit/validation FAILED (see partner_refit.log, partner_val.log)"
cat "$W/partner_refit.log" | tail -1
BL=("$W/val_ours.csv")
[ -s "$W/val_partner.csv" ] && BL+=("$W/val_partner.csv:partner")
[ -s "$W/val_partner2.csv" ] && BL+=("$W/val_partner2.csv:partner2")
if [ ${#BL[@]} -gt 1 ]; then
  python3 "$R2/blend_fit.py" --kit "$PUBDIR" --out "$W/entry/models/blend.json" "${BL[@]}" | tee "$W/blend_fit.log"
else
  log "using our model alone"
fi
log "final validation and test forecasts"
bash "$R2/final_forecast.sh" "$PUBDIR" "$W"
log "done"

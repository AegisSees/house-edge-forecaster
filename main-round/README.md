# House Edge entry: team blend (GRU ensemble + joint Bayesian rule model + two teammate Bayesian models)

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13). Extra packages pinned in `requirements.txt` (numpy and scipy).
- Hardware: CPU only. Prediction takes about 15 minutes on 2 cores, with peak RAM under 4 GB. Nothing is trained during the rerun.

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

1. `houseedge/features.py` rebuilds per-machine input sequences from `/data/test/queries.csv` and `history.csv`. The inputs at tick t use only ticks before t. Missing ticks are "unknown", never losses.
2. Every GRU checkpoint in `models/*.npz` runs as a float64 numpy forward pass (`houseedge/gru_numpy.py`). Each gives a win chance, 1 - p_0, and the script averages them in log-odds space across checkpoints.
3. `houseedge/joint_model.py` with `models/rules.json` supplies the payout-size distribution given a win. For short histories (under 100 observed ticks), it also supplies a structured win estimate, blended in at 20%.
4. The script writes `p_0 = 1 - P(win)` and `p_x = P(win) * P(size x | win)`.

The run is deterministic, with no random numbers at prediction time.

## Team blend

`predict.py` also runs the two teammates' Bayesian models:

- `partner/`: Bayesian filter. Code and `model.json` are unchanged from that teammate's entry; see `partner/PARTNER_README.md`. It runs in-process.
- `partner2/`: Bayesian model with abrupt and linear flips, episode-level flips and an episode latent; see `partner2/PARTNER_README.md`. It runs as a subprocess (`partner2/predict.py`) and its `p_0` is read back. One change from that teammate's entry: for short-history episodes, each lag pair's win model is fitted only on ticks where both of the pair's lags were observed (`DROP` in `partner2/model6.py`; full histories are unaffected). This raised `partner2/` alone from 58.07 to 58.15 on validation and the team blend by about 0.01 points in cross-validation (better in 16 of 20 fold splits). `model4.py`, `model6.py` and `latent.py` also read optional round parameters from `params.json`, which is absent in this entry, so the main-round constants are used.

The blend combines win chances in log-odds space with a two-state gate. Error analysis on validation showed that the Bayesian models are more accurate when a machine is in its low-win state, and the GRU ensemble when it is in its high-win state. A query counts as "low" when the three models' average win chance is below its machine type's validation median (balanced 0.7358, jackpot 0.0215, steady 0.9168). Short histories (under 100 observed ticks) get their own weights. Weights on the log-odds of win chance, as [ours, partner, partner2]:

| state | full history | short history |
| --- | --- | --- |
| low | [0.2, 0.2, 0.6] | [0.0, 0.5, 0.5] |
| high | [0.6, 0.0, 0.4] | [0.7, 0.3, 0.0] |

The payout-size distribution comes from our rule model for full histories. For short histories it comes from `partner/`, whose size forecast was more accurate there in cross-validation. The gate design and weights were chosen by 4-fold, condition-stratified episode cross-validation on public validation.

| entry | validation points / 60 |
| --- | --- |
| our GRU + rule model alone (median of 22 nets, earlier version) | 58.33 |
| our GRU + rule model alone (log-odds average of 22 nets) | 58.29 |
| `partner/` alone | 58.08 |
| `partner2/` alone | 58.07 |
| two-way gated blend (ours + partner) | 58.46 |
| three-way gated blend, linear | 58.51 (cross-validated estimate 58.48) |
| three-way gated blend, log-odds, short-history weights | 58.55 (cross-validated estimate 58.50) |
| same, plus log-odds averaging of our nets and `partner/` sizes for short histories | 58.57 (cross-validated estimate 58.52) |
| same, using only the 16 five-class nets and a 20% structured blend for short histories | 58.60 (cross-validated estimate 58.55) |
| same, without the three v4 five-class nets (13 nets) | 58.60 (cross-validated estimate 58.56) |
| same, with `partner2/` skipping unobserved-lag ticks on short histories and re-picked weights | 58.60 (58.6038; cross-validated estimate 58.56, +0.010 over the previous row) |
| same, plus the episode-latent correction (this entry) | **58.63** (58.6264) |

Condition loss ratios of this entry: stable 0.0166, short history 0.0431, drift 0.0183, shared change 0.0176.

## Episode-latent correction

Machines in one episode share a slowly drifting luck term in the win chance. After the blend, `houseedge/episode_latent.py` adds `beta * x` to the log-odds of win, where `x` is the half-life-8 exponentially weighted mean, over ticks before the cutoff, of the other machines' one-step residuals (observed win minus our GRU ensemble's in-sample win chance at that tick). It is skipped for short histories. `beta` is fitted by maximum likelihood on the predicted split's own history ticks (grid 0 to 4, step 0.25; no labels or targets are read); it is 1.25 on validation. Payout sizes are unchanged. Validation: 58.604 -> 58.626 (stratified episode bootstrap 90% interval for the gain with three nets: +0.012 to +0.043).

## Rule model (found in the public data)

- Each machine type has exactly two payout-size distributions given a win. Their spread in the validation targets is about 1e-15.
- A machine's win chance follows an XOR pattern on one lag pair, (1,3), (2,5) or (4,7), with a machine-specific direction that can drift or flip over time.
- The size regime is set by whether the payout at the pair's first lag (1, 2 or 4) was 2x or more, again with a machine-specific direction.
- `joint_model.high_probability` weighs every (pair, direction) hypothesis by two kinds of evidence. Win evidence uses free cell rates inside 32-tick chunks, so it is robust to flips. Size evidence comes from the winning payouts. The result is the posterior probability of the high size regime at the cutoff.
- `fit_rules.py` rediscovers all of this from public files: histories of every split, plus the validation targets for the two size vectors. It writes `models/rules.json`.

```sh
python fit_rules.py --data public --out models/rules.json
```

Validation score of our model alone (without the team blend, earlier 22-net median version): **58.33 / 60** (`check_submission.py --split validation`). Condition loss ratios: stable 0.0205, short history 0.0543, drift 0.0197, shared change 0.0203.

## Win-chance networks

There are 13 recurrent nets (2 stacked GRUs, hidden size 128). Six win/loss-only nets and three v4 five-class nets from earlier versions were dropped because the team blend scored better without them in cross-validation. Their inputs are the one-hot payouts at lags 1 to 8, the other machines' win rates at lags 1 to 8, machine type, signal and tick. The v3 and v4 variants add causal per-machine running statistics: conditional win rates for each lag and for each lag pair. Each net is trained with cross-entropy on the next observed payout at every history tick of the round's train, validation and test histories. Validation and test answers are never training inputs.

The training scripts are in `training/` and need torch and pandas. They read the kit through the `KIT` environment variable and write `w_<tag>.npz` to `OUTD`.

| file | script and settings |
| --- | --- |
| gru_e1 | `gru.py --ep 28 --hid 128 --layers 2 --seed 1` |
| gru_g3 | `gru.py --ep 60 --hid 128 --layers 2 --seed 0` |
| gru_A, gru_B, gru_C, gru_D, gru_C2, gru_C3 | `v2.py` (A: --aug 0.5; B: --aug 0.5 --mix 1; C: defaults; D: --mix 1; C2/C3: --seed 2/3) |
| gru_F1, gru_F4, gru_F5, gru_F6, gru_F7 | `v3.py --seed 1/4/5/6/7` |

CPU training in torch 2.15 is not bit-for-bit deterministic across machines, so the exact weights are included.

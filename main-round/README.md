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
3. `houseedge/joint_model.py` with `models/rules.json` supplies the payout-size distribution given a win. For short histories (under 100 observed ticks), it also supplies a structured win estimate, blended in at 40%.
4. The script writes `p_0 = 1 - P(win)` and `p_x = P(win) * P(size x | win)`.

The run is deterministic, with no random numbers at prediction time.

## Team blend

`predict.py` also runs the two teammates' Bayesian models:

- `partner/`: Bayesian filter. Code and `model.json` are unchanged from that teammate's entry; see `partner/PARTNER_README.md`. It runs in-process.
- `partner2/`: Bayesian model with abrupt and linear flips, episode-level flips and an episode latent. Code is unchanged from that teammate's entry; see `partner2/PARTNER_README.md`. It runs as a subprocess (`partner2/predict.py`) and its `p_0` is read back.

The blend combines win chances in log-odds space with a two-state gate. Error analysis on validation showed that the Bayesian models are more accurate when a machine is in its low-win state, and the GRU ensemble when it is in its high-win state. A query counts as "low" when the three models' average win chance is below its machine type's validation median (balanced 0.7477, jackpot 0.0226, steady 0.9149). Short histories (under 100 observed ticks) get their own weights. Weights on the log-odds of win chance, as [ours, partner, partner2]:

| state | full history | short history |
| --- | --- | --- |
| low | [0.3, 0.2, 0.5] | [0.0, 0.6, 0.4] |
| high | [0.7, 0.0, 0.3] | [0.8, 0.2, 0.0] |

The payout-size distribution comes from our rule model for full histories. For short histories it comes from `partner/`, whose size forecast was more accurate there in cross-validation. The gate design and weights were chosen by 4-fold, condition-stratified episode cross-validation on public validation.

| entry | validation points / 60 |
| --- | --- |
| our GRU + rule model alone (median of nets) | 58.33 |
| our GRU + rule model alone (log-odds average of nets) | 58.29 |
| `partner/` alone | 58.08 |
| `partner2/` alone | 58.07 |
| two-way gated blend (ours + partner) | 58.46 |
| three-way gated blend, linear | 58.51 (cross-validated estimate 58.48) |
| three-way gated blend, log-odds, short-history weights | 58.55 (cross-validated estimate 58.50) |
| same, plus log-odds averaging of our nets and `partner/` sizes for short histories (this entry) | **58.57** (cross-validated estimate 58.52) |

Condition loss ratios of this entry: stable 0.0174, short history 0.0440, drift 0.0186, shared change 0.0178.

## Rule model (found in the public data)

- Each machine type has exactly two payout-size distributions given a win. Their spread in the validation targets is about 1e-15.
- A machine's win chance follows an XOR pattern on one lag pair, (1,3), (2,5) or (4,7), with a machine-specific direction that can drift or flip over time.
- The size regime is set by whether the payout at the pair's first lag (1, 2 or 4) was 2x or more, again with a machine-specific direction.
- `joint_model.high_probability` weighs every (pair, direction) hypothesis by two kinds of evidence. Win evidence uses free cell rates inside 32-tick chunks, so it is robust to flips. Size evidence comes from the winning payouts. The result is the posterior probability of the high size regime at the cutoff.
- `fit_rules.py` rediscovers all of this from public files: histories of every split, plus the validation targets for the two size vectors. It writes `models/rules.json`.

```sh
python fit_rules.py --data public --out models/rules.json
```

Validation score of our model alone (without the team blend, median of nets): **58.33 / 60** (`check_submission.py --split validation`). Condition loss ratios: stable 0.0205, short history 0.0543, drift 0.0197, shared change 0.0203.

## Win-chance networks

There are 22 recurrent nets (2 stacked GRUs, hidden size 128). Their inputs are the one-hot payouts at lags 1 to 8, the other machines' win rates at lags 1 to 8, machine type, signal and tick. The v3 and v4 variants add causal per-machine running statistics: conditional win rates for each lag and for each lag pair. Each net is trained with cross-entropy on the next observed payout at every history tick of the round's train, validation and test histories. Validation and test answers are never training inputs.

The training scripts are in `training/` and need torch and pandas. They read the kit through the `KIT` environment variable and write `w_<tag>.npz` to `OUTD`.

| file | script and settings |
| --- | --- |
| gru_e1 | `gru.py --ep 28 --hid 128 --layers 2 --seed 1` |
| gru_g3 | `gru.py --ep 60 --hid 128 --layers 2 --seed 0` |
| gru_A, gru_B, gru_C, gru_D, gru_C2, gru_C3 | `v2.py` (A: --aug 0.5; B: --aug 0.5 --mix 1; C: defaults; D: --mix 1; C2/C3: --seed 2/3) |
| gru_F1, gru_F4, gru_F5, gru_F6, gru_F7 | `v3.py --seed 1/4/5/6/7` |
| gru_FB1, gru_FB2, gru_FB3 | `v3.py --binary 1 --seed 11/21/22` (win/loss loss only) |
| gru_PB1, gru_PB2, gru_PB3 | `v4.py --binary 1 --seed 12/23/24` (win/loss loss only) |
| gru_P1, gru_P2, gru_P3 | `v4.py --seed 1/2/3` |

CPU training in torch 2.15 is not bit-for-bit deterministic across machines, so the exact weights are included.

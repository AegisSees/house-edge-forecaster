# House Edge entry: GRU ensemble + joint Bayesian rule model

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13). Extra package pinned in `requirements.txt` (numpy only).
- Hardware: CPU only. Prediction takes about 10 to 15 minutes on 2 cores, with peak RAM under 4 GB. Nothing is trained during the rerun.

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

1. `houseedge/features.py` rebuilds per-machine input sequences from `/data/test/queries.csv` and `history.csv`. The inputs at tick t use only ticks before t. Missing ticks are "unknown", never losses.
2. Every GRU checkpoint in `models/*.npz` runs as a float64 numpy forward pass (`houseedge/gru_numpy.py`). Each gives a win chance, 1 - p_0, and the script takes the median across checkpoints.
3. `houseedge/joint_model.py` with `models/rules.json` supplies the payout-size distribution given a win. For short histories (under 100 observed ticks), it also supplies a structured win estimate, blended in at 40%.
4. The script writes `p_0 = 1 - P(win)` and `p_x = P(win) * P(size x | win)`.

The run is deterministic, with no random numbers at prediction time.

## Rule model (found in the public data)

- Each machine type has exactly two payout-size distributions given a win. Their spread in the validation targets is about 1e-15.
- A machine's win chance follows an XOR pattern on one lag pair, (1,3), (2,5) or (4,7), with a machine-specific direction that can drift or flip over time.
- The size regime is set by whether the payout at the pair's first lag (1, 2 or 4) was 2x or more, again with a machine-specific direction.
- `joint_model.high_probability` weighs every (pair, direction) hypothesis by two kinds of evidence. Win evidence uses free cell rates inside 32-tick chunks, so it is robust to flips. Size evidence comes from the winning payouts. The result is the posterior probability of the high size regime at the cutoff.
- `fit_rules.py` rediscovers all of this from public files: histories of every split, plus the validation targets for the two size vectors. It writes `models/rules.json`.

```sh
python fit_rules.py --data public --out models/rules.json
```

Validation score of this exact pipeline: **58.30 / 60** (`check_submission.py --split validation`). Condition loss ratios: stable 0.0196, short history 0.0548, drift 0.0211, shared change 0.0214.

## Win-chance networks

There are 14 recurrent nets (2 stacked GRUs, hidden size 128). Their inputs are the one-hot payouts at lags 1 to 8, the other machines' win rates at lags 1 to 8, machine type, signal and tick. The v3 and v4 variants add causal per-machine running statistics: conditional win rates for each lag and for each lag pair. Each net is trained with cross-entropy on the next observed payout at every history tick of the round's train, validation and test histories. Validation and test answers are never training inputs.

The training scripts are in `training/` and need torch and pandas. They read the kit through the `KIT` environment variable and write `w_<tag>.npz` to `OUTD`.

| file | script and settings |
| --- | --- |
| gru_e1 | `gru.py --ep 28 --hid 128 --layers 2 --seed 1` |
| gru_g3 | `gru.py --ep 60 --hid 128 --layers 2 --seed 0` |
| gru_A, gru_B, gru_C, gru_D, gru_C2, gru_C3 | `v2.py` (A: --aug 0.5; B: --aug 0.5 --mix 1; C: defaults; D: --mix 1; C2/C3: --seed 2/3) |
| gru_F1, gru_F4, gru_F5 | `v3.py --seed 1/4/5` |
| gru_P1, gru_P2, gru_P3 | `v4.py --seed 1/2/3` |

CPU training in torch 2.15 is not bit-for-bit deterministic across machines, so the exact weights are included.

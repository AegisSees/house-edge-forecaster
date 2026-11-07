# House Edge entry: GRU ensemble + exact payout-size model

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13). Extra package pinned in `requirements.txt` (numpy only).
- Hardware: CPU only. Prediction takes about 10 minutes on 2 cores, with peak RAM under 4 GB. Nothing is trained during the rerun.

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

1. `houseedge/features.py` rebuilds per-machine input sequences from `/data/test/queries.csv` and `history.csv`. The inputs at tick t use only ticks before t. Missing ticks are "unknown", never losses.
2. Every GRU checkpoint in `models/*.npz` runs as a float64 numpy forward pass (`houseedge/gru_numpy.py`). Their probabilities are combined by geometric mean, and the ensemble's chance of winning, 1 - p_0, is kept.
3. Given a win, the payout-size distribution comes from `houseedge/size_model.py` with `models/size_config.json`. It is an exact Bayesian average over machine-specific lag rules (see below).
4. The script writes `p_0 = 1 - P(win)` and `p_x = P(win) * P(size x | win)`.

The run is deterministic, with no random numbers at prediction time.

## What the models are

- **Win chance:** 10 recurrent nets (2 stacked GRUs, hidden size 128). Their inputs are the one-hot payouts at lags 1 to 8, the other machines' win rates at lags 1 to 8, machine type and signal. Some variants add causal per-machine running statistics: conditional win rates for each lag and for each lag pair. Each net is trained with cross-entropy on the next observed payout at every history tick of the round's train, validation and test histories. Validation and test answers are never training inputs.
- **Payout size:** the validation targets show that each machine type has exactly two size distributions given a win, and the machine switches between them. `fit_size_vectors.py` reads those two vectors from the public validation targets. It then finds, from histories only, which lags drive the switch: the payout at lag 1, 2 or 4 being 2x or more, with a machine-specific lag and direction. At prediction time, each machine's observed winning payouts give a posterior over the (lag, direction) rules, and the rules' verdicts at the cutoff are averaged.
- Validation score of this exact pipeline: 57.70 / 60 (`check_submission.py --split validation`).

## Training (separate, needs torch + pandas)

The training scripts are in `training/`. They read the kit through the `KIT` environment variable and write `w_<tag>.npz` to `OUTD`. The checkpoints included here are:

| file | script and settings |
| --- | --- |
| gru_e1 | `gru.py --ep 28 --hid 128 --layers 2 --seed 1` |
| gru_g3 | `gru.py --ep 60 --hid 128 --layers 2 --seed 0` |
| gru_A, gru_B, gru_C, gru_D, gru_C2 | `v2.py` (A: --aug 0.5; B: --aug 0.5 --mix 1; C: defaults; D: --mix 1; C2: --seed 2) |
| gru_F1 | `v3.py --seed 1` |
| gru_P1, gru_P2 | `v4.py --seed 1/2` |

CPU training in torch 2.15 is not bit-for-bit deterministic across machines, so the exact weights are included. Size config: `python fit_size_vectors.py --data public --out models/size_config.json`.

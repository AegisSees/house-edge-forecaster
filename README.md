# House Edge Forecaster

Probabilistic forecasting of slot-machine payouts for the **House Edge** challenge at TAMU Datathon 2026.

The commit history follows the versions submitted during the event, from the first GRU (56.4 / 60) to the final team blend (58.63 / 60).

For every query (a machine at a cutoff tick) the model predicts the probability of each payout (0, 1, 2, 5 or 20) using only the history before the cutoff. Entries are scored against the simulator's true probabilities, not the observed payouts, so the work was mostly about recovering how the hidden simulator generates outcomes and then building models around that structure.

| Round | Validation score |
| --- | --- |
| Main round | **58.63 / 60** |
| Adaptation round (new hidden rules, 90-minute window) | **19.49 / 20** |

## Approach

### 1. Recovering the hidden structure

Exploratory analysis of the public data showed that the true probabilities factor as

```
P(payout = x) = P(win) * P(size = x | win)
```

- **Payout size.** Each machine type has exactly two size distributions given a win. Which one applies is set by an XOR of a lagged payout (was the payout at lag 1, 2 or 4 a 2x or larger win?) and a per-machine flip.
- **Win chance.** The win probability follows an XOR rule on one pair of lags, (1,3), (2,5) or (4,7), with a machine-specific direction that can drift or flip over time in the "drift" and "shared change" conditions.
- **Shared luck.** Machines in the same episode share a slowly drifting luck term.

`fit_rules.py` rediscovers all of this from the public files and writes `models/rules.json`.

### 2. Models

- **GRU ensemble (backbone).** 13 recurrent networks (2 stacked GRU layers, hidden size 128) predict the win chance from one-hot payouts at lags 1 to 8, the other machines' win rates, machine type, signal and tick, plus causal running statistics such as conditional win rates per lag and lag pair. Trained in PyTorch, served as a pure NumPy float64 forward pass, averaged in log-odds space.
- **Exact Bayesian rule model.** `houseedge/joint_model.py` computes the posterior over every (lag pair, direction) hypothesis, combining win evidence from 32-tick chunks (robust to flips) and size evidence from winning payouts. It supplies the payout-size distribution and a structured win estimate for short histories.
- **Two Bayesian teammate models** (see [Team](#team)): a Bayesian state filter by Frank Atienza (`partner/`) and Bayesian logistic models with abrupt and gradual rule flips plus an episode latent by Andy Tran (`partner2/`).

### 3. Gated blend

Error analysis showed the Bayesian models are more accurate when a machine is in its low-win state and the GRUs when it is in its high-win state. The models are combined in log-odds space with a two-state gate and separate weights for short histories (under 100 observed ticks). Gate and weights were chosen by 4-fold, condition-stratified episode cross-validation, and every change had to win in cross-validation to stay in.

### 4. Episode-latent correction

After blending, each machine's win log-odds gets `beta * x`, where `x` is a recency-weighted (half-life 8) mean of the other machines' one-step prediction residuals in the same episode. `beta` is fitted by maximum likelihood on history only.

### Progression on main-round validation

| Step | Points / 60 |
| --- | --- |
| Best single model (GRU + rule model) | 58.33 |
| Bayesian state filter alone | 58.08 |
| Bayesian logistic flip model alone | 58.07 |
| Two-way gated blend | 58.46 |
| Three-way log-odds blend with short-history weights | 58.55 |
| Pruned ensemble, refined short-history handling | 58.60 |
| Plus episode-latent correction (final) | **58.63** |

### Adaptation round

The adaptation round changed which temporal relationships drive the machines. A one-command pipeline built and rehearsed in advance (`adaptation-round/training/round2/run_round2.sh`) re-searched the lags and found the new win pairs (1,4), (2,6) and (3,8) with size lags 1 to 3, retrained the GRUs, refit both Bayesian models on the new training data, and re-picked the blend by cross-validation. The blend scored 19.49 / 20 on validation, above every single model (19.27 to 19.40).

## Repository layout

```
main-round/          Final main-round entry (58.63 / 60)
  predict.py           Runs the full blend and writes submission.csv
  fit_rules.py         Rediscovers the simulator rules from public data
  houseedge/           Features, NumPy GRU, Bayesian rule model, episode latent
  models/              GRU weights (.npz) and rules.json
  partner/, partner2/  Teammates' Bayesian models
  training/            PyTorch training scripts for the GRUs
adaptation-round/    Round-two entry (19.49 / 20), same layout
  training/round2/     Automated refit pipeline and logs
```

Each round folder has its own `README.md` with full technical detail, including every blend weight and ablation.

## Running

Requires Python 3.13 with numpy and scipy (`requirements.txt`). CPU only, no network, deterministic.

```sh
cd main-round
pip install -r requirements.txt
python predict.py --data /path/to/kit --out submission.csv
```

Prediction takes about 15 minutes on 2 cores. The competition data is not included in this repository. Retraining the GRUs additionally needs torch and pandas; see `main-round/README.md`.

## Team

Team of three for TAMU Datathon 2026:

- **V** ([AegisSees](https://github.com/AegisSees)): GRU ensemble, Bayesian rule model, gated blend, episode-latent correction and the round-two refit pipeline (`houseedge/`, `predict.py`, `fit_rules.py`, `training/`).
- **Frank Atienza**: Bayesian state filter (`partner/`).
- **Andy Tran**: Bayesian logistic model with rule flips and an episode latent (`partner2/`).

Each teammate's folder keeps its own README.

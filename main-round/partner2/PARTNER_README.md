# House Edge entry: structured Bayesian per-machine forecaster

## Recreate the CSV

    pip install -r requirements.txt        # numpy==2.3.5
    python predict.py --data /data --out /output/submission.csv

- Python 3.13 (also tested on 3.14) and numpy 2.3.5. No other dependencies, no network, no GPU.
- Fully deterministic: no random numbers, so no seeds are needed.
- Reads only `<data>/test/queries.csv` and `<data>/test/history.csv`. To score locally on validation, add `--split validation`.
- There is no training step at predict time. The model's few constants (below) are hard-coded.
- Runtime: about 3 minutes on 8 cores, and about 12 minutes on the 2-core runner (limit 30). Use `--procs N` to set the worker count.

## Model

Each machine's next-spin distribution is P(win) multiplied by a payout-size profile given a win.

**Win chance** (`model6.py`):

    logit P(win at t) = b + w * g(t) * XOR(win[t-a], win[t-b]) + c * (mean win of other machines at t-3)

- One lag pair (a, b) is active per machine, from {(1,3), (2,5), (4,7)}.
- The prior on w is N(±2.4, 0.3²). The prior on b is N(mean for machine type and signal, 0.3²). The prior on c is N(0.7, 0.6²).
- g(t) is a regime function: no change; an abrupt sign flip at tick T; or a linear reversal starting at T over L ticks, with L in {16, 32, 64, 128}.
- An episode-level hypothesis lets all machines flip together at a common tick T, for shared changes.
- Short histories (detected from the number of observed ticks) use a near-zero prior on changes.
- Predictions average over pair, sign, T and L, weighted by Laplace-approximated evidence.

**Payout size given a win:**
- The machine is in a "hi" or "lo" state. Each state has a fixed size profile per machine type.
- The state is ±big(payout[t-k]), where big means 2× or more. The lag k matches the first lag of the win pair 99% of the time (k is in {1, 2, 4}).
- The relationship can flip at a tick T. The model averages over these hypotheses jointly with the win model.

**Episode latent** (`predict.py`, `latent.py`): pooled in-sample win residuals across the episode's machines, exponentially weighted with an 8-tick half-life, nudge the cutoff logit (scale 0.4).

All inputs come from ticks strictly before each query's cutoff, and other machines are used only within the same episode. Missing ticks are treated as unknown. The constants (lag pairs, size profiles, bias means, |w|, coupling, latent scale) were estimated from the organizer-released train and validation data.

Local validation: 58.07 / 60.

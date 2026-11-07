# House Edge entry: GRU sequence forecaster

- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13); extra package pinned in `requirements.txt` (numpy only)
- Hardware: CPU only. Prediction takes about 1 minute on 2 cores and uses less than 2 GB of RAM.

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

The command loads every weight file in `models/`, which are saved checkpoints, so nothing is trained during the rerun. It rebuilds the input features from `/data/test/queries.csv` and `/data/test/history.csv`, runs a float64 numpy forward pass of each GRU, averages their probabilities and writes the CSV. The code is deterministic and has no randomness at prediction time.

## Model

For each machine, a recurrent network reads the tick-by-tick history up to the cutoff. Its inputs at tick t use only observations from ticks before t:
- one-hot payout (or "missing") at lags 1 to 8;
- the win rate and observed fraction of the other machines in the same episode at lags 1 to 8;
- machine type, signal and tick / 256.

Missing ticks are encoded as unknown, never as a loss. The network is trained with cross-entropy to predict the next observed payout at every history tick of the round's train, validation and test histories. Validation and test answers are never used for training. Validation targets were used only to choose settings.

## Training (separate, needs torch)

```sh
python train.py --data path/to/public --seed 1 --out models/gru_s1.npz
```

Settings: 2-layer GRU, hidden 128, dropout 0.1, AdamW (lr 3e-3 OneCycle, wd 1e-4), batch 64 sequences, 28 epochs, torch 2.15 on CPU. CPU training is not bit-for-bit deterministic across machines, so the exact submitted weights are included in `models/`.

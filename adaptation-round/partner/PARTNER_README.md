# House Edge — joint Bayesian model, version 3

Main round, season `5fffdc4d0c700bbf61c3f911`. Team identity is supplied by the participant in the submission form. Version: `joint-missing-history-v3`. Submit the separately supplied `submission.csv` and this matching code ZIP together.

## Measured results

| Evaluation | Points / 60 |
| --- | ---: |
| Previous version 2, full validation | 55.224863 |
| Version 3, full validation | **58.084714** |
| Four-fold model-selection check, pooled held-out predictions | **58.074671** |

The organizer's participant checker confirms both version-3 scores. The 3,000-replicate condition-stratified episode bootstrap gives a 95% interval of **57.647990–58.444743** for the full-validation score and **57.634709–58.433088** for the fold-selection score. The paired improvement over version 2 has a 95% interval of **+2.376080 to +3.387306 points**.

The validation point estimate meets 58; the interval crosses 58. These results do not guarantee a hidden-test score of 58. Public validation has been used repeatedly during development. The four folds exclude the evaluated episodes when selecting settings from the fixed candidate catalog, but model design and the catalog benefited from previous validation use. This is a robustness check, not a fresh external holdout. Bootstrap intervals reflect episode sampling variation conditional on this development process; they do not include all model-selection or distribution-shift uncertainty. Hidden test targets and condition labels were unavailable.

## Run offline prediction

Install dependencies before the offline run:

```sh
python -m pip install -r requirements.txt
```

From this extracted source folder:

```sh
python predict.py --data /absolute/path/to/kit/public --out /absolute/path/to/submission.csv
```

The root `reproduce.json` gives the organizer command using `/data` and `/output/submission.csv`. Prediction loads `model.json`, `metadata.json`, and only the requested split's queries and history. It requires no labels, conditions, training files, existing prediction CSVs, network requests, or external APIs. There are no query-specific forecast lookup tables. It writes only the requested output CSV. Round and season are checked before prediction.

The environment tested locally was Linux, Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. The organizer's exact Python 3.13 image was unavailable. Requirements are pinned. BLAS/OpenMP use one thread; a GPU is not required. Runtime, memory, hashes, and reproduction results appear in `verification.json`.

## Model and changes from version 2

1. Training histories identify three recurring win lag pairs: (1,3), (2,5), and (4,7). A machine's pattern and response direction are uncertain latent variables, not inferred from query IDs or condition labels at prediction time.
2. Stable training episodes estimate shared type/signal effects and the population distribution of baseline win tendency and interaction strength. The population fit integrates over uncertainty in individual machines. It stopped after 95 iterations at the specified parameter-change tolerance.
3. Training observations identify a corresponding payout-size lag for each pattern: 1, 2, or 4 ticks. The joint filter uses the likelihood of loss, 1-chip payout, and larger payout to update the latent pattern. Version 2 treated the 1-chip share largely as a smoothed frequency; this version models its temporal dependence.
4. Complete histories use a Bayesian filter over pattern, signed interaction strength, baseline state, and payout-size direction. Baseline diffusion is 0.03 and the win-direction reversal probability is 0.0002 per tick. The small learned auxiliary win-lag-7 effect is retained in this branch.
5. Incomplete histories use all observed old and recent spins. A forward algorithm sums over missing binary win/loss sequences instead of discarding the old observations. This marginalization is exact under the branch's static primary-pair model, with uniform unknown prehistory. The branch omits the small auxiliary lag-7 effect and parameter drift, and ignores payout-size evidence whose predecessor size is unknown. It is not exact inference of the full simulator.
6. Relative 2/5/20-chip shares are estimated from pooled training observations by machine type. Predictions average over latent-state uncertainty and form a valid five-category distribution.

Parameters are estimated from training data. A small, fixed catalog of five candidate settings is compared on public validation: three prior widths, an original-prior alternative, and an equal-weight prior ensemble. The selected model uses the learned prior at width 1.0. Eight-node Gauss-Hermite integration handles narrow parameter distributions. Increasing to 12 nodes changed the development-half score by only 0.000041 points; this approximation check is separate from exact reproduction of the fixed eight-node model.

Features for tick t use only observations strictly before t. Missing ticks remain unknown, and histories are grouped by episode and machine. The inference branch is chosen from observable history completeness, never from test condition labels.

## Verify the score and implementation

Generate validation predictions:

```sh
python predict.py --data /absolute/path/to/kit/public --split validation --out /absolute/path/to/validation_predictions.csv
```

From the original participant kit, use its unmodified checker:

```sh
python check_submission.py /absolute/path/to/validation_predictions.csv --split validation
python check_submission.py /absolute/path/to/cross_validation_predictions.csv --split validation
python check_submission.py /absolute/path/to/submission.csv
```

The first two CSVs are for checking results, not competition uploads. The separately supplied `submission.csv` contains the test forecasts.

Run the regression tests from this source folder:

```sh
python -m unittest -v test_history_rules.py test_state_model.py test_joint_model.py
```

The 12 tests cover future/current-tick leakage, missing observations, episode separation, batch invariance, empty histories, joint-filter agreement with explicit latent-state enumeration, and missing-spin agreement with exhaustive binary-sequence enumeration. The package is also tested from a fresh extraction with only test inputs. `verification.json` records reproduction and a complete independent refit. `robustness_report.json` contains every candidate score, fold membership and choices, and bootstrap intervals.

## Refit from public inputs

```sh
python train.py --data /absolute/path/to/kit/public --out /absolute/path/to/new-run
```

Choose a new output directory. The script refits the population and size models using training observations, evaluates the fixed candidate catalog, conducts four-fold episode selection and 3,000 bootstrap resamples, freezes the final model, and only then loads test inputs. It writes:

- `model.json` and `submission.csv`
- `validation_predictions.csv` and `cross_validation_predictions.csv`
- `training_report.json` and `robustness_report.json`
- `candidate_catalog.json` and `population_em_trace.json`

The training report records dependency versions, seeds, input hashes, model hash, and runtime. Parameter fitting and prediction are deterministic; the bootstrap seed is 20261107. No external experimental caches are required.

For the adaptation kit, refit with its public directory and examine the new results. The script reads the new season and point maximum; main-round performance does not establish adaptation performance. If the new data do not support the three-pair structural model, revise it before submission. Never reuse this main checkpoint directly on a different season.

After refitting, copy the source folder, replace the checkpoint and reports, update this README and verification results, and use the participant kit to package the new source:

```sh
python build_entry.py --source /absolute/path/to/new-source --out /absolute/path/to/new-entry.zip
```

Upload the matching test CSV and code ZIP as two separate files through the correct round's form. Local checks do not submit an entry.

## Artifact fingerprints

- `submission.csv` SHA-256: `4e7bdb609cebd14f18ecffe4282f2d2906106796ca5fd6cadbfd146a5dafa452`.
- `validation_predictions.csv` SHA-256: `d5f634a1e522b6ea8fac1a18fff7a82655f06d0469b1817c12ba03b2849f897b`.
- `cross_validation_predictions.csv` SHA-256: `55fdcb1779e0157f9629ad9bbba0f3de2feff8d038b197176364cdc86096c75c`.
- `model.json` SHA-256: `b266dc321523bf8274ca67a423f82c306382d568123343f3ba17f282e79f27df`.

Independent full refit: 145.5 seconds. Fresh extraction prediction: 43.6 seconds; peak child RSS: 850.2 MiB. These measurements were made locally; the exact organizer image was not tested.

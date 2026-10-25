# Non-linear and interaction features for Deep Forest

Experiment code for an MSc ensemble-learning project at Warsaw University of Technology, by Wojciech
Jasiński and Sandra Adamiec. The research plan is the spec: [docs/plan.pdf](docs/plan.pdf).

Deep Forest stacks layers of tree ensembles. Each layer receives the input features and the
previous layer's predictions. We add new columns between layers and compare each variant with the
same cascade without them (the *base cascade*) on 32 public datasets.

## Quick start

Needs [uv](https://docs.astral.sh/uv/); every dependency is pinned in `uv.lock`. On macOS, LightGBM also
needs OpenMP: `brew install libomp`.

```sh
uv sync
uv run pytest            # unit tests + an end-to-end smoke run, about 1 min
uv run dfx run smoke     # every method on tiny built-in data, about 30 s (not a result)
```

## Reproduce the study

```sh
uv run dfx data                          # download the 32 datasets (sha256-pinned) -> docs/data-audit.md
uv run dfx run main --workers 8          # 5,280 tasks, 38 CPU-hours; interrupt and rerun to resume
uv run dfx run sensitivity --workers 8   # 3,840 tasks, 59 CPU-hours
uv run dfx run positive                  # 330 tasks, 2 minutes: the positive control
uv run dfx report main                   # statistics, tables and figures -> reports/main/
uv run dfx report sensitivity
uv run dfx report positive
```

`dfx run` also takes `--datasets`, `--methods`, `--seeds`, `--folds` and `--checks` (comma-separated)
to run a subset, `--dry-run` to only count done / stale / failed / missing tasks, and `--results DIR`
for scratch runs. CPU hours as measured on an Apple M4 (10 cores); the main run took 7.5 h on 8 workers,
at most 0.7 GB each.

## What is compared

Each layer of the base cascade has 2 random forests (RF) and 2 extremely randomized forests (ERF) of
100 trees, with the defaults of the DF21 0.1.7 package. The variants add columns to each layer's input:

| Method | Added columns | Chosen by | From layer | Tests |
|---|---|---|---|---|
| `base` | none | – | – | – |
| `random_xx` | K feature × feature products | at random, per layer and fold model | 1 | – |
| `target_xx` | K feature × feature products | correlation with the target | 1 | – |
| `residual_xx` | K feature × feature products | correlation with the previous layer's error | 1 | H1 vs base; H2a, H2b vs `random_xx`, `target_xx` |
| `xp_mi` | K feature × prediction products | mutual information with the target | 2 | H3 vs base |
| `nonlinear` | uncertainty of each forest type, disagreement between types | fixed | 2 | H4 vs base |
| `monotone` | sigmoid of every prediction column | fixed | 2 | H5 vs base |
| `combined` | `nonlinear` + `residual_xx` | as above | 1 / 2 | – |
| `rf`, `rf_target_xx` | no cascade: one 400-tree RF, without / with K target-chosen products | – | – | – |
| `lightgbm` | no cascade: LightGBM tuned over 10 random settings on 3 inner folds | – | – | – |

K = min(d, ⌊d(d−1)/8⌋), where d is the number of features after one-hot encoding. The error is
1 − balanced accuracy (classification) or RMSE (regression). Each hypothesis is a two-sided Wilcoxon
signed-rank test over the 32 datasets, with Holm's correction over the six; H2 needs both H2a and H2b.

Added after the main run, as exploratory: the report repeats the comparisons within each task, on
other metrics (accuracy, ROC AUC, average precision, R²) and at fixed depths, with Hodges–Lehmann
intervals (Analysis in [docs/decisions.md](docs/decisions.md)); `experiments/positive.toml` is a
positive control: every method on two synthetic datasets with one planted interaction.

- Every setting and its source (the plan, the DF21 source, a library default, or ours): [docs/decisions.md](docs/decisions.md).
- What each dataset contributes and what is excluded: [docs/data-audit.md](docs/data-audit.md).
- On four datasets the base cascade scores within fold-to-fold noise of the official DF21 package:
  [docs/df21-check.md](docs/df21-check.md).

Terms used in the code:
- **task**: one method on one dataset, seed and outer fold.
- **check**: one sensitivity setting. In the main experiment there is a single check, `default`.
- **fold model**: the forests of one layer trained on two of the three inner parts of the outer-training
  rows. They predict the third part, and these out-of-fold (**OOF**) predictions feed the next layer.
- **P**: a layer's prediction columns.
- **XX** / **XP**: feature × feature / feature × prediction products.

The outer test rows pass through every layer as it is trained but never fit anything, so every layer's
test score is recorded and no model is stored.

## Result files

Each task writes `results/<experiment>/<check>/<dataset>/<method>/s<seed>-f<fold>.npz`, then the `.json`
next to it. The JSON holds a fingerprint of the task, its settings, the code and `uv.lock`, the data
and the split. A rerun skips a task only if its fingerprint still matches. JSON keys:

- `task` (with its full settings), `fingerprint`, `task_type`, `n_train`, `n_test`, `n_classes`, `d`.
- `best_layer`, `n_layers` (counted from 1), and `metrics`: the test metrics at the best layer.
- `layers`: per layer, `oof_score`, `metrics`, `cpu_seconds` and `selections`.
  - `oof_score` is the depth score of the OOF predictions: balanced accuracy, or MSE in regression. It is `null` for baselines.
  - `selections` has one dict per fold model, e.g. `{"residual_xx": {"pairs": [[i, j], ...], "pool": n}}`.
  - `pairs` index encoded features; for XP, j indexes the p_{t,c} columns, type-major.
  - `pool` counts the non-constant candidates.
  - `xp_mi` also stores `max_corr_with_P`.
- `info`: K, or LightGBM's chosen and tried settings.
- `cpu_seconds`, `wall_seconds`, and `worker_peak_rss_mb` (the worker's peak so far).

The NPZ holds `test_index` (rows of the prepared dataset), `y_true`, `pred_by_layer` (layers × test
rows × classes) and, for cascades and LightGBM, `inner_folds` (each outer-training row's inner fold).

`dfx report` writes `results/<experiment>/runs.csv`, `layers.csv` and `reports/<experiment>/`; it adds
accuracy, ROC AUC, average precision and R², computed from the NPZ predictions. These and
`manifest.json` (environment, commit, dataset hashes) are committed after each experiment.

The per-run files are not in git; they will be attached to a GitHub release. To rebuild a report from
them, extract the tarball into `results/` at the release's commit, run `uv run dfx data` (the report
checks each result's data and split hashes), then `uv run dfx report <experiment>`.

## Reproducibility

- Python 3.12 (`.python-version`) and every package version are fixed by `uv.lock`.
- Downloads are checked against pinned sha256 hashes; the outer splits are committed in `splits/`.
- Every seed is fixed, and none depends on the method, so variants grow identical trees wherever their
  inputs coincide (Seeds row of [docs/decisions.md](docs/decisions.md)). Results are identical with 1
  or many workers (tested).

## Layout

```
src/dfx/
  datasets.py   the 32 dataset specs: source, target, exclusions, groups
  data.py       download, prepare, related-row groups, 20,000-row cap, splits, data audit
  prep.py       imputation, one-hot encoding, standardization; DF21 binning
  forest.py     RF and ERF with DF21 semantics, on scikit-learn trees
  features.py   the added columns, their selection, and which method adds which
  cascade.py    the cascade loop, depth rule and seed streams
  baselines.py  RF, RF + products, LightGBM
  metrics.py    test metrics
  run.py        experiment files -> tasks, worker pool, result files, resume
  report.py     statistics, tables, figures
  cli.py        the dfx command
experiments/    main.toml, sensitivity.toml, positive.toml, smoke.toml: the tasks and their settings
tests/          test_core.py (unit), test_smoke.py (end to end)
scripts/        check_df21.py: comparison with the official DF21 package (own environment)
docs/           plan.pdf, decisions.md, data-audit.md, df21-check.md
splits/         outer fold assignments
data/           downloads and prepared arrays (not in git)
results/        per-run files (not in git); runs.csv, layers.csv, manifest.json
reports/        generated summaries, tables and figures
```

## Credits

- The cascade follows Zhou & Feng, *Deep Forest* (National Science Review, 2019) and the defaults of the
  official DF21 package, [LAMDA-NJU/Deep-Forest](https://github.com/LAMDA-NJU/Deep-Forest) 0.1.7. Its
  code is not copied; `forest.py` and `prep.py` reimplement its forests and binning on scikit-learn.
- Built on scikit-learn, LightGBM, NumPy, SciPy, pandas and matplotlib.
- Datasets: the UCI Machine Learning Repository (CC BY 4.0) and California Housing via scikit-learn.
  Sources are listed in `src/dfx/datasets.py`.

## AI assistance

The code was written with Claude Code (Opus 5.5) from the authors' research plan. The implementation
plan and the code were reviewed by Claude agents and by Codex (gpt-6-astra), and Claude Code applied the
findings. The authors decided the design and are responsible for the code and results.

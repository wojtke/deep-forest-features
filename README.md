# Non-linear and interaction features for Deep Forest

Experiment code for an MSc ensemble-learning project at Warsaw University of Technology, by Wojciech
Jasiński and Sandra Adamiec. The research plan is the spec: [docs/plan.pdf](docs/plan.pdf).

Deep Forest stacks layers of tree ensembles. Each layer receives the input features and the
previous layer's predictions. We add new columns between layers and compare each variant with the
same cascade without them (the *base cascade*) on 32 public datasets. The new columns are
products of two features or of a feature and a prediction, chosen by simple scores, and transforms
of the predictions.

## Quick start

Needs [uv](https://docs.astral.sh/uv/). Every dependency is pinned in `uv.lock`. On macOS, LightGBM also
needs OpenMP: `brew install libomp`.

```sh
uv sync
uv run pytest                      # unit tests + an end-to-end smoke run, about 1 min
uv run dfx run smoke --workers 4   # every method on tiny built-in data, about 30 s
```

## Reproduce the study

```sh
uv run dfx data                          # download the 32 datasets (sha256-pinned), write data/AUDIT.md
uv run dfx run main --workers 8          # 5,280 tasks; safe to interrupt, rerun to resume
uv run dfx run sensitivity --workers 8   # 3,840 tasks
uv run dfx report main                   # tables, tests and figures -> reports/main/
uv run dfx report sensitivity
```

`dfx run` takes `--datasets`, `--methods`, `--seeds`, `--folds` and `--checks` (comma-separated)
to run a subset, and `--dry-run` to list what would run. `dfx status main` counts finished,
failed and missing tasks. A one-fold pilot on 22.10 puts the main run at about 40 CPU-hours
(about 5 hours on 8 workers) and the sensitivity checks at roughly 60.

## What is compared

| Method | Added columns | Chosen by |
|---|---|---|
| `base` | none | – |
| `random_xx` | K feature–feature products | random, redrawn per layer |
| `target_xx` | K feature–feature products | correlation with the target |
| `residual_xx` | K feature–feature products | correlation with the previous layer's error |
| `xp_mi` | K feature–prediction products | mutual information with the target |
| `nonlinear` | uncertainty of each forest type, disagreement between types | fixed |
| `monotone` | sigmoid of every prediction column | fixed |
| `combined` | `nonlinear` + `residual_xx` | as above |
| `rf`, `rf_target_xx` | a 400-tree random forest, without / with target-chosen products | – |
| `lightgbm` | tuned LightGBM | 10 random settings, 3 inner folds |

Hypotheses H1–H5 are decided by two-sided Wilcoxon tests with Holm's correction over the 32 datasets.
Every setting and its source (the plan, the DF21 package source, a library default, or our own choice)
is in [docs/decisions.md](docs/decisions.md). On four datasets the base cascade scores within
fold-to-fold noise of the official DF21 package: [docs/df21-check.md](docs/df21-check.md).

## How it works

- **Cascade** (`src/dfx/cascade.py`). Each layer has 2 random forests + 2 extremely randomized forests,
  100 trees each, with DF21 0.1.7 defaults reimplemented on scikit-learn trees (`forest.py`). Layer
  predictions are 3-fold out-of-fold, with the same folds at every layer. Columns are added from raw
  (unbinned) features and predictions, chosen on each fold model's own training rows, and then
  binned into 255 quantile bins. Layers are added until two in a row bring no gain. The outer test
  rows pass through every layer but are never used for fitting, so each layer's test score is recorded.
- **Tasks** (`src/dfx/run.py`). An experiment file in `experiments/` expands into tasks, one per
  dataset × method × seed × outer fold. They run in a pool of single-threaded processes, largest first.
- **Results**. Each task writes `results/<experiment>/<check>/<dataset>/<method>/s<seed>-f<fold>.npz`
  and `.json` (schema in [AGENTS.md](AGENTS.md)), with a fingerprint of the code, `uv.lock`, data,
  split and settings. A rerun skips a task only if the fingerprint still matches. If a worker
  process dies, the tasks it may have been running are retried one at a time and the culprit is
  recorded as failed.
- **Report** (`src/dfx/report.py`). Writes `results/<experiment>/runs.csv` and `layers.csv` (committed),
  plus `reports/<experiment>/summary.md` with LaTeX tables and figures.

The per-run files are not in git; they are attached to the GitHub release of each experiment.

## Reproducibility

- Python 3.12 and every package version come from `uv.lock`; `results/<experiment>/manifest.json`
  records versions, platform, git commit and dataset hashes.
- Downloads are checked against pinned sha256 hashes. Outer splits are committed in `splits/`.
- Seeds are fixed: outer splits use seeds 0, 1, 2 and the 20,000-row subsample seed 0. Every random
  choice inside a task comes from `SeedSequence([seed, dataset, fold, layer, inner fold, stream])` and
  never depends on the method, so variants grow identical trees wherever their inputs coincide.
- Results are identical with 1 or many workers (tested).

## Layout

```
src/dfx/      data.py datasets.py prep.py forest.py features.py cascade.py baselines.py metrics.py run.py report.py cli.py
experiments/  main.toml sensitivity.toml smoke.toml
tests/        test_core.py (unit) test_smoke.py (end to end)
scripts/      check_df21.py (comparison with the official DF21 package)
docs/         plan.pdf decisions.md df21-check.md
data/         downloads and prepared arrays (not in git); AUDIT.md
splits/       outer fold assignments
results/      per-run files (not in git); runs.csv, layers.csv, manifest.json
reports/      summary.md, LaTeX tables, figures
```

## Credits

- The cascade follows Zhou & Feng, *Deep Forest* (National Science Review, 2019) and the defaults of the
  official DF21 package, [LAMDA-NJU/Deep-Forest](https://github.com/LAMDA-NJU/Deep-Forest) 0.1.7. Its
  code is not copied; `forest.py` and `prep.py` reimplement its forests and binning on scikit-learn.
- Built on scikit-learn, LightGBM, NumPy, SciPy, pandas and matplotlib.
- Datasets: the UCI Machine Learning Repository (CC BY 4.0) and California Housing via scikit-learn.
  Sources are listed in `src/dfx/datasets.py` and data/AUDIT.md.

## AI assistance

The code was written with Claude Code (Opus 5.5) from the authors' research plan and an implementation
plan. The implementation plan was reviewed by a Claude agent and by Codex (gpt-6-astra). The authors
decided the design and are responsible for the code and results.

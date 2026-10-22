# AGENTS.md

Notes for AI agents (and people) working here. Read [README.md](README.md) first. The research plan
[docs/plan.pdf](docs/plan.pdf) is the spec; [docs/decisions.md](docs/decisions.md) gives the source of
every setting.

## Commands

```sh
uv sync                                   # environment from uv.lock
uv run pytest                             # all tests, about 1 min; run before every commit
uv run dfx run smoke --workers 2          # end-to-end check on built-in data
uv run dfx data [NAME ...]                # download + prepare datasets, rewrite data/AUDIT.md
uv run dfx run EXP --workers N [--datasets a,b --methods m --seeds 0 --folds 0 --checks c --dry-run]
uv run dfx status EXP                     # done / stale / failed / missing per method
uv run dfx report EXP [--allow-incomplete]
```

Long runs: start them in the background and follow `results/EXP/run.log`. A failed task leaves
`<task>.error.txt` with the traceback, and the run continues.

## Rules that keep the results valid

1. **Outer test rows never fit anything.** `cascade.fit_predict` and `baselines.run_baseline` receive
   test features but never test labels; preprocessing, selection, scaling, binning, forests and tuning
   use training rows only. Tests in `tests/test_core.py` check this; keep them passing.
2. **Inside the cascade, every added column is fitted on that fold model's training rows** (`tr`),
   then applied unchanged to validation and test rows. Test rows use the mean over the previous fold
   models (`cascade.new_columns_test`).
3. **Randomness comes only from `cascade.seed_for(...)`**, and seeds never depend on the method name.
4. **No invented numbers.** A setting is sourced (plan, DF21 source, library default) or recorded as
   "ours" in docs/decisions.md. A new value of our own also gets a sensitivity check or an explicit note.
5. **Never edit `results/` by hand.** Only `dfx run` writes results; only `dfx report` writes `runs.csv`,
   `layers.csv` and `reports/`.
6. **Changing a file in `run.COMPUTE_FILES` changes every fingerprint.** Finished results then count
   as stale and will be re-run. Do not change those files while a production run is in progress;
   commit first, run from the commit, and say in the commit message when results must be redone.

## Where to change what

| To change | Edit |
|---|---|
| a dataset (source, target, exclusions, groups) | `src/dfx/datasets.py`, then `uv run dfx data NAME` and check data/AUDIT.md |
| added columns or selection | `src/dfx/features.py` (`CASCADE_METHODS`, `components`) |
| the cascade loop, depth rule, seeds | `src/dfx/cascade.py` |
| forests or binning | `src/dfx/forest.py`, `src/dfx/prep.py` |
| RF / LightGBM baselines | `src/dfx/baselines.py` |
| which runs exist | `experiments/*.toml` (settings override `CascadeConfig` / `LGBMConfig` fields) |
| statistics, tables, figures | `src/dfx/report.py` |

A new method also needs a row in the README table, in docs/decisions.md if it has settings, and in
`report.METHODS`. Test it in `tests/test_core.py` and add it to `experiments/smoke.toml`.

## Result files

`s<seed>-f<fold>.json` keys:
- `task`, with the settings overrides.
- `fingerprint`, `git`, `task_type`, `n_train`, `n_test`, `n_classes`, `d` (encoded features).
- `best_layer`, `n_layers`, `metrics` (test metrics at the best layer).
- `layers`: a list of `{layer, oof_score, metrics, cpu_seconds, selections}`. `selections` holds one
  dict per fold model, e.g. `{"xx_residual": {"pairs": [[i, j], ...], "pool": n}}`.
- `info` (K; LightGBM's chosen and tried settings), `cpu_seconds`, `wall_seconds`, `worker_peak_rss_mb`.

`s<seed>-f<fold>.npz` holds `test_index`, `y_true`, `pred_by_layer` (layers × test rows × classes)
and `inner_folds`. Metrics: classification `balanced_accuracy` (primary), `log_loss`, `macro_f1`;
regression `rmse` (primary), `mae`.

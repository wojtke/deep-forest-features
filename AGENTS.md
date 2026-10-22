# AGENTS.md

Notes for AI agents (and people) changing this repo. Read [README.md](README.md) first: commands,
methods, result files and layout. The research plan [docs/plan.pdf](docs/plan.pdf) is the spec;
[docs/decisions.md](docs/decisions.md) gives the source of every setting.

Run `uv run pytest` (about 1 min) before every commit. For scratch runs, pass `--results` and `--out`
pointing outside the repo. Start long runs in the background and follow `results/<experiment>/run.log`.
A failed task leaves `<task>.error.txt` with the traceback, and the run continues.

## Rules that keep the results valid

1. **Outer test rows never fit anything.** `cascade.fit_predict` and `baselines.run_baseline` receive
   test features but never test labels. Preprocessing, selection, scaling, binning, forests and tuning
   use training rows only. Tests in `tests/test_core.py` check this; keep them passing.
2. **Inside the cascade, every added column is fitted on that fold model's training rows** (`tr`),
   then applied unchanged to validation and test rows. For test rows, a column is computed from each
   previous fold model's outputs and then averaged (`cascade.new_columns_test`).
3. **Randomness inside a task comes only from `cascade.seed_for(...)`, and seeds never depend on the
   method name.** The outer-split seeds are in `experiments/main.toml`; the subsample seed is
   `data.SUBSAMPLE_SEED`.
4. **No invented numbers.** A setting is sourced (plan, DF21 source, library default) or is a row
   marked *ours* in docs/decisions.md.
5. **Never edit `results/` by hand.** Only `dfx run` writes results; only `dfx report` writes
   `runs.csv`, `layers.csv` and `reports/`.
6. **Any edit to a file in `run.COMPUTE_FILES` or to `uv.lock`, even a comment, makes every finished
   result stale.** `dfx run` re-runs stale tasks and `dfx report` ignores them. Workers refuse tasks
   once the code has changed mid-run, so never edit those files during a run. Commit first, run from the
   commit, and say in the commit message when results must be redone.

## Changing a dataset

1. Edit its `Spec` in `src/dfx/datasets.py`.
2. If the prepared data changes, the committed split no longer fits. Run
   `rm splits/NAME.npz && uv run dfx data NAME`.
3. Check docs/data-audit.md and commit the new split.

That dataset's results go stale, because its data hash is in the fingerprint.

## Adding a cascade method

1. In `src/dfx/features.py`:
   - write a component with `fit(Xs, prev, y, *, task, n_classes, seed)` and `transform(Xs, prev)`;
   - add it to `components()`. Names ending in `_xx` enter at layer 1, all others at layer 2;
   - add the method to `CASCADE_METHODS`.
2. Add a test to `tests/test_core.py`.
3. Record its settings in docs/decisions.md: source, K, first layer.
4. Add a row to the README methods table and update the task counts in README and `experiments/main.toml`.
5. Decide whether the sensitivity checks in `experiments/sensitivity.toml` include it.

Experiments with `methods = "all"` and the report pick it up. Editing `features.py` makes every result
stale (rule 6). A baseline goes into `baselines.BASELINES` and `run_baseline` instead.

"""Collect results and run the analysis of plan §5.

`dfx report main` writes results/main/runs.csv and layers.csv (one row per run / per run and
layer; committed), then reports/main/: summary.md with every table, LaTeX versions of the tables
and figures. Only runs whose fingerprint matches the current code, data and settings are used.
The hypothesis tests refuse to run unless every run of the methods they compare is finished.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from .baselines import BASELINES
from .features import CASCADE_METHODS
from .metrics import PRIMARY, SECONDARY, error
from .run import experiment_settings, fingerprints, state, tasks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASCADES = list(CASCADE_METHODS)
METHODS = CASCADES + list(BASELINES)
# Plan §5: one Holm family of six paired comparisons; H2 needs both H2a and H2b.
HYPOTHESES = [("H1", "residual_xx", "base"), ("H2a", "residual_xx", "random_xx"),
              ("H2b", "residual_xx", "target_xx"), ("H3", "xp_mi", "base"), ("H4", "nonlinear", "base"),
              ("H5", "monotone", "base")]
ALPHA = 0.05


# --- collecting ---------------------------------------------------------------------------------

def collect(experiment: str, results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Runs and per-layer rows of every finished, current task, and the unfinished tasks per method."""
    expected = tasks(experiment)
    fps, _ = fingerprints(expected)
    runs, layers, missing = [], [], []
    for t in expected:
        if state(t, results, fps[t]) != "done":
            missing.append(t.method)
            continue
        r = json.loads(t.path(results).with_suffix(".json").read_text())
        key = {"check": t.check, "dataset": t.dataset, "method": t.method, "seed": t.seed, "fold": t.fold,
               "task": r["task_type"]}
        xp = [c for layer in r["layers"] for sel in layer["selections"]
              for c in sel.get("xp_mi", {}).get("max_corr_with_P", [])]
        runs.append(key | {"error": error(r["metrics"], r["task_type"]), **r["metrics"],
                           "best_layer": r["best_layer"], "n_layers": r["n_layers"], "d": r["d"],
                           "K": r["info"].get("K"), "cpu_seconds": r["cpu_seconds"], "n_train": r["n_train"],
                           "xp_corr_median": float(np.median(xp)) if xp else np.nan})
        for layer in r["layers"]:
            layers.append(key | {"layer": layer["layer"], "oof_score": layer["oof_score"],
                                 "error": error(layer["metrics"], r["task_type"]), **layer["metrics"]})
    return pd.DataFrame(runs), pd.DataFrame(layers), pd.Series(missing, dtype=str).value_counts()


# --- statistics ---------------------------------------------------------------------------------

def relative_reduction(e_ref, e):
    """(e_ref - e) / max(e_ref, e), zero if both are zero (plan §5); positive favours e."""
    e_ref, e = np.asarray(e_ref, float), np.asarray(e, float)
    top = np.maximum(e_ref, e)
    return np.where(top > 0, (e_ref - e) / np.where(top > 0, top, 1), 0.0)


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values (step-down, monotone)."""
    p = np.asarray(p, float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def wilcoxon(reduction) -> float:
    """Two-sided Wilcoxon signed-rank p-value (scipy default zero handling); 1 if all are zero."""
    reduction = np.asarray(reduction)
    return 1.0 if np.all(reduction == 0) else float(stats.wilcoxon(reduction).pvalue)


def mean_errors(runs: pd.DataFrame) -> pd.DataFrame:
    """Dataset x method table of errors averaged over the runs (15 in the main experiment)."""
    return runs.groupby(["dataset", "method"])["error"].mean().unstack()


def reductions(errors: pd.DataFrame, method: str, ref: str) -> pd.Series:
    """Relative error reduction of `method` vs `ref` per dataset where both have results."""
    both = errors[[method, ref]].dropna()
    return pd.Series(relative_reduction(both[ref], both[method]), index=both.index)


def comparisons(errors: pd.DataFrame, pairs, family: bool = True) -> pd.DataFrame:
    rows = []
    for name, method, ref in pairs:
        if method not in errors or ref not in errors:
            continue
        red = reductions(errors, method, ref)
        rows.append({"comparison": name, "method": method, "reference": ref, "datasets": len(red),
                     "median_reduction": float(red.median()), "wins": int((red > 0).sum()),
                     "losses": int((red < 0).sum()), "zeros": int((red == 0).sum()), "p": wilcoxon(red)})
    table = pd.DataFrame(rows)
    if family and len(table):
        table["p_holm"] = holm(table["p"])
        table["supported"] = (table["p_holm"] < ALPHA) & (table["median_reduction"] > 0)
    return table


def ranks(errors: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Mean ranks (1 = lowest error), Friedman test and Nemenyi critical difference (Demšar 2006)."""
    errors = errors[[m for m in METHODS if m in errors]].dropna()
    k, n = errors.shape[1], errors.shape[0]
    mean_rank = errors.rank(axis=1).mean().sort_values()
    friedman = stats.friedmanchisquare(*[errors[m] for m in errors]) if k > 2 and n > 1 else None
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2) if k > 1 else np.nan
    return mean_rank, {"datasets": n, "methods": k, "chi2": getattr(friedman, "statistic", np.nan),
                       "p": getattr(friedman, "pvalue", np.nan), "cd": float(q * np.sqrt(k * (k + 1) / (6 * n)))}


def common_depth(layers: pd.DataFrame) -> pd.DataFrame:
    """Median relative error reduction vs the base cascade at each depth that every cascade
    variant reached in a run (plan §3); columns: variants, plus the number of datasets."""
    cas = layers[layers.method.isin(CASCADES)]
    reach = cas.groupby(["dataset", "seed", "fold", "method"])["layer"].max().unstack()
    cas = cas.join(reach.dropna().min(axis=1).rename("common"), on=["dataset", "seed", "fold"])
    out = {}
    for depth, at in cas[cas.layer <= cas.common].groupby("layer"):
        err = mean_errors(at)
        if "base" in err:
            out[depth] = {m: float(reductions(err, m, "base").median()) for m in CASCADES if m in err and m != "base"}
            out[depth]["datasets"] = len(err)
    return pd.DataFrame(out).T.rename_axis("depth")


def per_dataset(runs: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Mean ± sd of a metric over the runs, dataset x method (plan §5)."""
    sub = runs[runs.task == task]
    if sub.empty or metric not in sub:
        return pd.DataFrame()
    g = sub.groupby(["dataset", "method"])[metric]
    table = (g.mean().map("{:.4g}".format) + " ± " + g.std(ddof=1).fillna(0).map("{:.2g}".format)).unstack()
    return table[[m for m in METHODS if m in table]]


def sensitivity(runs: pd.DataFrame, settings: dict, results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) each check vs the reference run of the same method; (2) each variant vs the base cascade
    under the same setting: the check's own base runs, the check named by `base` in the experiment
    file, or else the reference experiment's base (descriptive, no tests)."""
    ref_runs, _, _ = collect(settings["reference"], results)
    keys = ["dataset", "method", "seed", "fold"]
    ref = ref_runs[ref_runs.seed.isin(settings["seeds"])]
    stable, versus = [], []
    for check in settings["check"]:
        sub = runs[runs.check == check["name"]]
        if sub.empty:
            continue
        joined = sub.join(ref.set_index(keys)["error"].rename("ref_error"), on=keys).dropna(subset=["ref_error"])
        for method, part in joined.groupby("method"):
            err = part.groupby("dataset")[["error", "ref_error"]].mean()
            red = pd.Series(relative_reduction(err["ref_error"], err["error"]))
            stable.append({"check": check["name"], "method": method, "datasets": len(red),
                           "median_reduction": float(red.median()), "wins": int((red > 0).sum()),
                           "losses": int((red < 0).sum())})
        base_name = check.get("base", check["name"])
        own = runs[(runs.check == base_name) & (runs.method == "base")]
        base, base_from = (own, base_name) if len(own) else (ref[ref.method == "base"], settings["reference"])
        errors = mean_errors(pd.concat([sub[sub.method != "base"], base]))
        table = comparisons(errors, [(check["name"], m, "base") for m in errors if m != "base"], family=False)
        versus.extend(table.assign(check=check["name"], base_from=base_from).to_dict("records"))
    columns = ["check", "method", "base_from", "datasets", "median_reduction", "wins", "losses", "zeros"]
    return pd.DataFrame(stable), pd.DataFrame(versus, columns=columns)


# --- writing ------------------------------------------------------------------------------------

def _fmt(x) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        return "-" if np.isnan(x) else (f"{x:.3g}" if abs(x) < 1e-3 and x != 0 else f"{x:.4f}".rstrip("0").rstrip("."))
    return str(x)


def markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(df.columns), *body])


def latex(df: pd.DataFrame) -> str:
    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    lines = [r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(_fmt(v)) for v in row) + r" \\" for row in df.itertuples(index=False)]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def cd_diagram(mean_rank: pd.Series, cd: float, path: Path) -> None:
    """Critical-difference diagram: methods on the mean-rank axis; bars join methods whose mean
    ranks differ by less than the critical difference."""
    k = len(mean_rank)
    fig, ax = plt.subplots(figsize=(7, 0.35 * k + 1.2))
    ax.set_xlim(1, k)
    ax.set_ylim(-((k + 1) // 2) * 0.5 - 0.3, 1.5)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black")
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.15, color="black")
        ax.text(r, 0.3, str(r), ha="center", fontsize=8)
    ax.plot([1, 1 + cd], [1.1, 1.1], color="black")
    ax.text(1 + cd / 2, 1.25, f"CD = {cd:.2f}", ha="center", fontsize=8)
    for i, (method, r) in enumerate(mean_rank.items()):
        left = i < k / 2
        y = -(i if left else k - 1 - i) * 0.5 - 0.5
        x_text = 1 - 0.1 if left else k + 0.1
        ax.plot([r, r, x_text], [0, y, y], color="0.4", lw=0.8)
        ax.text(x_text, y, f"{method} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=8)
    values, level = mean_rank.to_numpy(), 0
    for i in range(k):
        j = max(j for j in range(i, k) if values[j] - values[i] < cd)
        if j > i and (i == 0 or values[j] - values[i - 1] >= cd):
            level += 1
            ax.plot([values[i] - 0.03, values[j] + 0.03], [-0.15 * level] * 2, color="black", lw=2.5)
    _save(fig, path)


def reductions_plot(errors: pd.DataFrame, task_of: pd.Series, path: Path) -> None:
    """Relative error reduction per dataset for the six hypothesis comparisons."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (_, method, ref) in enumerate(HYPOTHESES):
        red = reductions(errors, method, ref)
        cls = task_of.reindex(red.index).eq("classification").to_numpy()
        x = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(red))
        ax.scatter(x[cls], red[cls], s=12, marker="o", color="C0", label="classification" if i == 0 else None)
        ax.scatter(x[~cls], red[~cls], s=12, marker="^", color="C1", label="regression" if i == 0 else None)
        ax.hlines(red.median(), i - 0.3, i + 0.3, color="black")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(HYPOTHESES)), [f"{n}\n{m}\nvs {r}" for n, m, r in HYPOTHESES], fontsize=7)
    ax.set_ylabel("relative error reduction")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


class Summary:
    """summary.md plus one LaTeX file per table."""

    def __init__(self, folder: Path, title: str):
        self.folder, self.lines = folder, [title]
        (folder / "tables").mkdir(parents=True, exist_ok=True)

    def text(self, *lines: str) -> None:
        self.lines.extend(["", *lines])

    def table(self, name: str, heading: str, df: pd.DataFrame, note: str = "", index: bool = False) -> None:
        if df.empty:
            return
        df = df.reset_index() if index else df
        self.text(f"## {heading}", *([note, ""] if note else []), markdown(df))
        (self.folder / "tables" / f"{name}.tex").write_text(latex(df))

    def write(self) -> Path:
        path = self.folder / "summary.md"
        path.write_text("\n".join(self.lines) + "\n")
        return path


def _main_report(s: Summary, runs: pd.DataFrame, layers: pd.DataFrame, missing: pd.Series) -> None:
    errors, task_of = mean_errors(runs), runs.groupby("dataset")["task"].first()
    needed = {x for _, a, b in HYPOTHESES for x in (a, b)}
    if needed <= set(errors.columns) and not missing.reindex(sorted(needed)).fillna(0).any():
        tests = comparisons(errors, HYPOTHESES)
        h2 = bool(tests.set_index("comparison").loc[["H2a", "H2b"], "supported"].all())
        s.table("hypotheses", "Hypotheses (plan §5)", tests,
                f"Two-sided Wilcoxon signed-rank over datasets, Holm over the six comparisons; supported iff "
                f"adjusted p < {ALPHA} and median reduction > 0. **H2 supported: {'yes' if h2 else 'no'}** "
                "(needs H2a and H2b).")
        reductions_plot(errors, task_of, s.folder / "reductions")
        s.text("![relative reductions](reductions.png)")
    else:
        s.text("## Hypotheses (plan §5)", "", "Not tested: some runs of the compared methods are unfinished.")
    vs_base = [(m, m, "base") for m in METHODS if m != "base"]
    for scope in ("all", "classification", "regression"):
        sub = errors if scope == "all" else errors[task_of.reindex(errors.index) == scope]
        if sub.empty:
            continue
        suffix = "" if scope == "all" else f": {scope}"
        if scope != "all":
            s.table(f"hypotheses_{scope}", f"Hypotheses by task{suffix} (descriptive)",
                    comparisons(sub, HYPOTHESES, family=False))
        s.table(f"vs_base_{scope}", f"Every method vs the base cascade{suffix} (descriptive)",
                comparisons(sub, vs_base, family=False))
        mean_rank, fr = ranks(sub)
        if fr["methods"] > 2:
            s.table(f"ranks_{scope}", f"Mean ranks{suffix}", mean_rank.rename("mean rank").to_frame(),
                    f"Friedman over {fr['methods']} methods and {fr['datasets']} datasets: chi2 = {fr['chi2']:.2f}, "
                    f"p = {_fmt(fr['p'])}; Nemenyi critical difference = {fr['cd']:.2f} (descriptive).", index=True)
            cd_diagram(mean_rank, fr["cd"], s.folder / f"cd_{scope}")
            s.text(f"![CD diagram{suffix}](cd_{scope}.png)")
    s.table("common_depth", "Common depth (descriptive)", common_depth(layers),
            "Median relative reduction vs base at each depth reached by every cascade variant in a run.", index=True)
    xp = runs.loc[runs.method == "xp_mi", "xp_corr_median"].dropna()
    if len(xp):
        s.text("## XP products vs the predictions", "",
               f"Median over runs of the median max |corr| between chosen XP products and P: {xp.median():.3f} "
               f"(IQR {xp.quantile(.25):.3f}-{xp.quantile(.75):.3f}).")


def report(experiment: str, results: Path, out: Path, allow_incomplete: bool = False) -> None:
    runs, layers, missing = collect(experiment, results)
    if runs.empty:
        raise SystemExit(f"{experiment}: no finished runs in {results}")
    if missing.sum() and not allow_incomplete:
        raise SystemExit(f"{experiment}: {missing.sum()} tasks not finished ({dict(missing)}); see "
                         f"`dfx status {experiment}`, or pass --allow-incomplete for an interim report")
    runs.to_csv(results / experiment / "runs.csv", index=False)
    layers.to_csv(results / experiment / "layers.csv", index=False)
    settings = experiment_settings(experiment)
    title = f"# Report: {experiment}" + (f" (INCOMPLETE: {missing.sum()} tasks missing)" if missing.sum() else "")
    s = Summary(out / experiment, title)
    s.text(f"{len(runs)} runs on {runs.dataset.nunique()} datasets. Error = 1 - balanced accuracy "
           "(classification) or RMSE (regression); relative reduction = (e_ref - e) / max(e_ref, e) of errors "
           "averaged over a dataset's runs; positive favours the method.")
    if "reference" in settings:
        stable, versus = sensitivity(runs, settings, results)
        s.table("sensitivity_vs_base", "Sensitivity: each variant vs base under the same setting (descriptive)",
                versus, "Base runs from the same check, the check named in `base_from`, or the reference experiment.")
        s.table("sensitivity_vs_reference", f"Sensitivity: each check vs `{settings['reference']}` (descriptive)",
                stable, "Same method, dataset, seed and fold in the reference experiment.")
    else:
        _main_report(s, runs, layers, missing)
    for task in ("classification", "regression"):
        for metric in (PRIMARY[task], *SECONDARY[task], "cpu_seconds"):
            s.table(f"per_dataset_{task}_{metric}", f"Per dataset: {metric} mean ± sd ({task})",
                    per_dataset(runs, task, metric), index=True)
    print(f"wrote {results / experiment / 'runs.csv'} and {s.write()}")

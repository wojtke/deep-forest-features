"""Collect results and run the analysis of plan §5.

`dfx report main` writes results/main/runs.csv and layers.csv (one row per run / per run and
layer; committed), then reports/main/: summary.md with every table, LaTeX versions of the tables
and figures. Only runs whose fingerprint matches the current code, data and settings are used.
The hypothesis tests refuse to run on an incomplete experiment unless --allow-incomplete is given.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = {"classification": "balanced_accuracy", "regression": "rmse"}
CASCADES = ["base", "random_xx", "target_xx", "residual_xx", "xp_mi", "nonlinear", "monotone", "combined"]
METHODS = CASCADES + ["rf", "rf_target_xx", "lightgbm"]
# Plan §5: one Holm family of six paired comparisons; H2 needs both H2a and H2b.
HYPOTHESES = [("H1", "residual_xx", "base"), ("H2a", "residual_xx", "random_xx"),
              ("H2b", "residual_xx", "target_xx"), ("H3", "xp_mi", "base"), ("H4", "nonlinear", "base"),
              ("H5", "monotone", "base")]
ALPHA = 0.05


# --- collecting ---------------------------------------------------------------------------------

def collect(experiment: str, results: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Runs and per-layer rows of every finished, current task; also the number not finished."""
    from .run import fingerprints, state, tasks

    expected = tasks(experiment)
    fps, _ = fingerprints(expected)
    runs, layers, missing = [], [], 0
    for t in expected:
        if state(t, results, fps[t]) != "done":
            missing += 1
            continue
        r = json.loads(t.path(results).with_suffix(".json").read_text())
        key = {"check": t.check, "dataset": t.dataset, "method": t.method, "seed": t.seed, "fold": t.fold,
               "task": r["task_type"]}
        xp = [c for layer in r["layers"] for sel in layer["selections"]
              for c in sel.get("xp_mi", {}).get("max_corr_with_P", [])]
        runs.append(key | {"error": _error(r["metrics"], r["task_type"]), **r["metrics"],
                           "best_layer": r["best_layer"], "n_layers": r["n_layers"], "d": r["d"],
                           "K": r["info"].get("K"), "cpu_seconds": r["cpu_seconds"], "n_train": r["n_train"],
                           "xp_corr_median": float(np.median(xp)) if xp else np.nan})
        for layer in r["layers"]:
            layers.append(key | {"layer": layer["layer"], "oof_score": layer["oof_score"],
                                 "error": _error(layer["metrics"], r["task_type"]), **layer["metrics"]})
    return pd.DataFrame(runs), pd.DataFrame(layers), missing


def _error(metrics: dict, task: str) -> float:
    """Plan §5: one minus balanced accuracy, or RMSE."""
    return 1 - metrics["balanced_accuracy"] if task == "classification" else metrics["rmse"]


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


def comparisons(errors: pd.DataFrame, pairs, family: bool = True) -> pd.DataFrame:
    rows = []
    for name, method, ref in pairs:
        both = errors[[method, ref]].dropna()
        red = relative_reduction(both[ref], both[method])
        rows.append({"comparison": name, "method": method, "reference": ref, "datasets": len(red),
                     "median_reduction": float(np.median(red)), "wins": int((red > 0).sum()),
                     "losses": int((red < 0).sum()), "zeros": int((red == 0).sum()), "p": wilcoxon(red)})
    table = pd.DataFrame(rows)
    if family:
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
    reach = reach.dropna().min(axis=1).rename("common")
    cas = cas.join(reach, on=["dataset", "seed", "fold"])
    cas = cas[cas.layer <= cas.common]
    out = {}
    for depth, at in cas.groupby("layer"):
        err = mean_errors(at)
        if "base" not in err:
            continue
        row = {m: float(np.median(relative_reduction(err["base"], err[m]))) for m in CASCADES if m in err and m != "base"}
        out[depth] = row | {"datasets": len(err)}
    return pd.DataFrame(out).T.rename_axis("depth")


# --- writing ------------------------------------------------------------------------------------

def _fmt(x) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        return "-" if np.isnan(x) else (f"{x:.3g}" if abs(x) < 1e-3 and x != 0 else f"{x:.4f}".rstrip("0").rstrip("."))
    return str(x)


def markdown(df: pd.DataFrame, index: bool = True) -> str:
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def latex(df: pd.DataFrame, index: bool = True) -> str:
    df = df.reset_index() if index else df
    esc = (lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$"))
    lines = [r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(_fmt(v)) for v in row) + r" \\" for row in df.itertuples(index=False)]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def per_dataset(runs: pd.DataFrame, task: str) -> pd.DataFrame:
    """Mean ± std of the primary metric over the runs, dataset x method (plan §5)."""
    sub = runs[runs.task == task]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["dataset", "method"])[PRIMARY[task]]
    table = (g.mean().map("{:.4f}".format) + " ± " + g.std(ddof=1).fillna(0).map("{:.4f}".format)).unstack()
    return table[[m for m in METHODS if m in table]]


def cd_diagram(mean_rank: pd.Series, cd: float, path: Path) -> None:
    """Critical-difference diagram: methods on the mean-rank axis; bars join methods whose mean
    ranks differ by less than the critical difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def reductions_plot(errors: pd.DataFrame, tasks: pd.Series, path: Path) -> None:
    """Relative error reduction per dataset for the six hypothesis comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (name, method, ref) in enumerate(HYPOTHESES):
        both = errors[[method, ref]].dropna()
        red = relative_reduction(both[ref], both[method])
        cls = tasks.reindex(both.index).eq("classification").to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(red))
        ax.scatter(i + jitter[cls], red[cls], s=12, marker="o", color="C0", label="classification" if i == 0 else None)
        ax.scatter(i + jitter[~cls], red[~cls], s=12, marker="^", color="C1", label="regression" if i == 0 else None)
        ax.hlines(np.median(red), i - 0.3, i + 0.3, color="black")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(HYPOTHESES)), [f"{n}\n{m}\nvs {r}" for n, m, r in HYPOTHESES], fontsize=7)
    ax.set_ylabel("relative error reduction")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def report(experiment: str, results: Path, out: Path, allow_incomplete: bool = False) -> None:
    from .run import experiment_settings

    runs, layers, missing = collect(experiment, results)
    if runs.empty:
        raise SystemExit(f"{experiment}: no finished runs in {results}")
    if missing and not allow_incomplete:
        raise SystemExit(f"{experiment}: {missing} tasks not finished; run `dfx status {experiment}`, "
                         "or pass --allow-incomplete for an interim report")
    folder = out / experiment
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    runs.to_csv(results / experiment / "runs.csv", index=False)
    layers.to_csv(results / experiment / "layers.csv", index=False)
    settings = experiment_settings(experiment)
    title = f"# Report: {experiment}" + (f" — INCOMPLETE ({missing} tasks missing)" if missing else "")
    sections = [title, "", f"{len(runs)} runs on {runs.dataset.nunique()} datasets. Error = 1 - balanced accuracy "
                "(classification) or RMSE (regression); relative reduction = (e_ref - e) / max(e_ref, e) of "
                "errors averaged over a dataset's runs; positive favours the method."]

    def add(name: str, heading: str, df: pd.DataFrame, note: str = "", index: bool = True) -> None:
        if df.empty:
            return
        sections.extend(["", f"## {heading}", "", note, "", markdown(df, index)] if note else
                        ["", f"## {heading}", "", markdown(df, index)])
        (folder / "tables" / f"{name}.tex").write_text(latex(df, index))

    if "reference" in settings:
        add("sensitivity", "Sensitivity checks (descriptive)", sensitivity(runs, settings, results),
            f"Each check vs the `{settings['reference']}` run of the same method, dataset, seed and fold.",
            index=False)
    else:
        errors = mean_errors(runs)
        task_of = runs.groupby("dataset")["task"].first()
        if {x for _, a, b in HYPOTHESES for x in (a, b)} <= set(errors.columns):
            tests = comparisons(errors, HYPOTHESES)
            h2 = bool(tests.set_index("comparison").loc[["H2a", "H2b"], "supported"].all())
            add("hypotheses", "Hypotheses (plan §5)", tests,
                f"Two-sided Wilcoxon signed-rank over datasets, Holm over the six comparisons; supported iff "
                f"adjusted p < {ALPHA} and median reduction > 0. **H2 supported: {'yes' if h2 else 'no'}** "
                "(needs H2a and H2b).", index=False)
            for task in ("classification", "regression"):
                sub = errors[task_of.reindex(errors.index) == task]
                add(f"hypotheses_{task}", f"Hypotheses by task: {task} (descriptive)",
                    comparisons(sub, HYPOTHESES, family=False), index=False)
            reductions_plot(errors, task_of, folder / "reductions")
            sections.extend(["", "![relative reductions](reductions.png)"])
        others = [(m, m, "base") for m in METHODS if m in errors and m != "base"]
        if "base" in errors:
            add("vs_base", "Every method vs the base cascade (descriptive)",
                comparisons(errors, others, family=False), "Includes the combined variant.", index=False)
        mean_rank, fr = ranks(errors)
        if fr["methods"] > 2:
            add("ranks", "Mean ranks", mean_rank.rename("mean rank").to_frame(),
                f"Friedman over {fr['methods']} methods and {fr['datasets']} datasets: chi2 = {fr['chi2']:.2f}, "
                f"p = {_fmt(fr['p'])}; Nemenyi critical difference = {fr['cd']:.2f} (descriptive).")
            cd_diagram(mean_rank, fr["cd"], folder / "cd_diagram")
            sections.extend(["", "![CD diagram](cd_diagram.png)"])
        add("common_depth", "Common depth (descriptive)", common_depth(layers),
            "Median relative reduction vs base at each depth reached by every cascade variant in a run.")
        xp = runs.loc[runs.method == "xp_mi", "xp_corr_median"].dropna()
        if len(xp):
            sections.extend(["", "## XP products vs the predictions", "",
                             f"Median over runs of the median max |corr| between chosen XP products and P: "
                             f"{xp.median():.3f} (IQR {xp.quantile(.25):.3f}-{xp.quantile(.75):.3f})."])
    for task in ("classification", "regression"):
        add(f"per_dataset_{task}", f"Per dataset: {PRIMARY[task]} mean ± sd ({task})", per_dataset(runs, task))
    cpu = runs.groupby("method")["cpu_seconds"].agg(["mean", "sum"])
    cpu["sum"] = cpu["sum"] / 3600
    add("cpu", "CPU time", cpu.rename(columns={"mean": "mean s per run", "sum": "total h"}),
        "Measured on mixed performance/efficiency cores; indicative only.")
    (folder / "summary.md").write_text("\n".join(sections) + "\n")
    print(f"wrote {results / experiment / 'runs.csv'} and {folder / 'summary.md'}")


def sensitivity(runs: pd.DataFrame, settings: dict, results: Path) -> pd.DataFrame:
    """Median relative reduction of each check vs the reference experiment's matching runs."""
    ref_runs, _, _ = collect(settings["reference"], results)
    if ref_runs.empty:
        return pd.DataFrame()
    keys = ["dataset", "method", "seed", "fold"]
    ref = ref_runs[ref_runs.seed.isin(settings["seeds"])].set_index(keys)["error"].rename("ref_error")
    rows = []
    for (check, method), sub in runs.groupby(["check", "method"]):
        joined = sub.join(ref, on=keys).dropna(subset=["ref_error"])
        err = joined.groupby("dataset")[["error", "ref_error"]].mean()
        red = relative_reduction(err["ref_error"], err["error"])
        rows.append({"check": check, "method": method, "datasets": len(err), "median_reduction": float(np.median(red)),
                     "wins": int((red > 0).sum()), "losses": int((red < 0).sum())})
    return pd.DataFrame(rows)

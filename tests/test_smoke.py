"""End to end: the smoke experiment in two worker processes, determinism, resume and the report."""
import json
from importlib.metadata import version

from dfx.report import report
from dfx.run import PACKAGES, _git, code_hash, fingerprints, run_experiment, run_task, state, tasks

TIMINGS = ("cpu_seconds", "wall_seconds", "worker_peak_rss_mb")


def without_timings(record):
    record = {k: v for k, v in record.items() if k not in TIMINGS}
    record["layers"] = [{k: v for k, v in layer.items() if k != "cpu_seconds"} for layer in record["layers"]]
    return record


def test_smoke_experiment_end_to_end(tmp_path):
    results = tmp_path / "results"
    assert run_experiment("smoke", workers=2, results=results)
    selected = tasks("smoke")
    fps, _ = fingerprints(selected)
    assert all(state(t, results, fps[t]) == "done" for t in selected)
    records = {t: json.loads(t.path(results).with_suffix(".json").read_text()) for t in selected}
    assert all(r["metrics"] == r["layers"][r["best_layer"] - 1]["metrics"] for r in records.values())

    # Every method on one dataset, run again in this process, gives the same results as the pool.
    meta = {"code": code_hash(), "git": _git(), "versions": {p: version(p) for p in PACKAGES}}
    for t in [t for t in selected if t.dataset == "smoke_pair" and t.fold == 0]:
        assert without_timings(records[t]) == without_timings(run_task(t, tmp_path / "again", fps[t], meta))

    # A result with another fingerprint is stale and is re-run; finished ones are skipped.
    task = next(t for t in selected if t.method == "residual_xx")
    path = task.path(results).with_suffix(".json")
    path.write_text(json.dumps(records[task] | {"fingerprint": "old"}))
    assert state(task, results, fps[task]) == "stale"
    assert run_experiment("smoke", workers=2, results=results)
    assert state(task, results, fps[task]) == "done"

    report("smoke", results, tmp_path / "reports")
    summary = (tmp_path / "reports" / "smoke" / "summary.md").read_text()
    assert "## Hypotheses (plan §5)" in summary and (results / "smoke" / "runs.csv").exists()

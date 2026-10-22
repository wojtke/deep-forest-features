"""Expand experiments/<name>.toml into tasks, run them in parallel and store one result per task.

A task is (experiment, check, dataset, method, seed, outer fold). A finished task leaves
results/<experiment>/<check>/<dataset>/<method>/s<seed>-f<fold>.npz and .json. The JSON is written
last and holds a fingerprint of everything that determines the result (task, settings, code, data,
split). A resumed run skips a task only if both files exist and the fingerprint matches.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import sys
import time
import tomllib
import traceback
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .data import ROOT

EXPERIMENTS = ROOT / "experiments"
SRC = Path(__file__).resolve().parent
COMPUTE_FILES = ("data.py", "datasets.py", "prep.py", "forest.py", "features.py", "cascade.py",
                 "baselines.py", "run.py")  # changing these changes results; report.py/cli.py do not
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "lightgbm")


def code_hash() -> str:
    h = hashlib.sha256()
    for name in COMPUTE_FILES:
        h.update((SRC / name).read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Task:
    experiment: str
    check: str
    dataset: str
    method: str
    seed: int
    fold: int
    outer_folds: int
    cascade: str = "{}"  # JSON of CascadeConfig overrides
    lightgbm: str = "{}"  # JSON of LGBMConfig overrides

    def path(self, results: Path) -> Path:
        return results / self.experiment / self.check / self.dataset / self.method / f"s{self.seed}-f{self.fold}"

    def label(self) -> str:
        check = "" if self.check == "default" else f" [{self.check}]"
        return f"{self.dataset} {self.method}{check} s{self.seed} f{self.fold}"


def experiment_settings(name: str) -> dict:
    """experiments/<name>.toml with defaults filled in."""
    from .datasets import ALL

    settings = tomllib.loads((EXPERIMENTS / f"{name}.toml").read_text())
    if settings.get("datasets", "all") == "all":
        settings["datasets"] = list(ALL)
    settings.setdefault("cascade", {})
    settings.setdefault("lightgbm", {})
    settings.setdefault("check", [{"name": "default", "methods": settings.get("methods", [])}])
    return settings


def tasks(name: str, datasets=None, methods=None, seeds=None, folds=None, checks=None) -> list[Task]:
    s = experiment_settings(name)
    out = []
    for check in s["check"]:
        if checks and check["name"] not in checks:
            continue
        cascade = json.dumps({**s["cascade"], **check.get("cascade", {})}, sort_keys=True)
        lightgbm = json.dumps(s["lightgbm"], sort_keys=True)
        for dataset in s["datasets"]:
            for method in check["methods"]:
                for seed in s["seeds"]:
                    for fold in range(s["outer_folds"]):
                        t = Task(name, check["name"], dataset, method, seed, fold, s["outer_folds"], cascade, lightgbm)
                        if ((datasets is None or dataset in datasets) and (methods is None or method in methods)
                                and (seeds is None or seed in seeds) and (folds is None or fold in folds)):
                            out.append(t)
    return out


def metrics(y: np.ndarray, pred: np.ndarray, task: str) -> dict:
    """Plan §5: balanced accuracy (primary), log-loss, macro-F1; RMSE (primary), MAE."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

    if task == "classification":
        p = pred / pred.sum(1, keepdims=True)
        label = p.argmax(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {"balanced_accuracy": float(balanced_accuracy_score(y, label)),
                    "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
                    "macro_f1": float(f1_score(y, label, average="macro", zero_division=0))}
    err = y - pred[:, 0]
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def run_task(task: Task, results: Path, fingerprint: str, git: dict) -> dict:
    """Fit one method on one outer training fold and score it on the outer test fold."""
    from .baselines import LGBMConfig, run_baseline
    from .cascade import CascadeConfig, fit_predict
    from .data import load, outer_folds
    from .features import CASCADE_METHODS
    from .prep import Preprocessor

    wall, cpu = time.time(), time.process_time()
    data = load(task.dataset)
    fold_of, _ = outer_folds(data, task.seed, task.outer_folds)
    train, test = np.flatnonzero(fold_of != task.fold), np.flatnonzero(fold_of == task.fold)
    prep = Preprocessor().fit(data.X[train], data.categorical)  # outer-training rows only
    xs_tr, xs_te = prep.transform(data.X[train]), prep.transform(data.X[test])
    cfg = CascadeConfig(**json.loads(task.cascade))
    common = dict(task=data.task, n_classes=data.n_classes, origin=prep.origin, dummy=prep.dummy,
                  keys=(task.seed, zlib.crc32(task.dataset.encode()), task.fold), cfg=cfg)
    if task.method in CASCADE_METHODS:
        out = fit_predict(xs_tr, data.y[train], data.groups[train], xs_te, method=task.method, **common)
    else:
        out = run_baseline(task.method, xs_tr, data.y[train], data.groups[train], xs_te,
                           lgbm=LGBMConfig(**json.loads(task.lightgbm)), **common)
    y_te = data.y[test]
    layers = [{"layer": i + 1, "oof_score": layer.oof_score, "metrics": metrics(y_te, layer.test_pred, data.task),
               "cpu_seconds": round(layer.cpu_seconds, 3), "selections": layer.selections}
              for i, layer in enumerate(out.layers)]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)
    record = {"task": task.__dict__ | {"cascade": json.loads(task.cascade), "lightgbm": json.loads(task.lightgbm)},
              "fingerprint": fingerprint, "git": git, "task_type": data.task,
              "n_train": len(train), "n_test": len(test), "n_classes": data.n_classes, "d": len(prep.origin),
              "best_layer": out.best, "n_layers": len(out.layers), "metrics": layers[out.best - 1]["metrics"],
              "layers": layers, "info": out.info, "cpu_seconds": round(time.process_time() - cpu, 3),
              "wall_seconds": round(time.time() - wall, 3), "worker_peak_rss_mb": round(rss, 1)}
    base = task.path(results)
    base.parent.mkdir(parents=True, exist_ok=True)
    preds = np.stack([layer.test_pred for layer in out.layers]).astype(np.float32)
    arrays = {"test_index": test, "y_true": y_te, "pred_by_layer": preds}
    if out.inner_folds is not None:
        arrays["inner_folds"] = out.inner_folds
    _atomic_write(base.with_suffix(".npz"), lambda f: np.savez_compressed(f, **arrays))
    _atomic_write(base.with_suffix(".json"), lambda f: f.write((json.dumps(record, indent=1) + "\n").encode()))
    return record


def _safe_run(task: Task, results: Path, fingerprint: str, git: dict) -> dict:
    error = task.path(results).with_suffix(".error.txt")
    try:
        record = run_task(task, results, fingerprint, git)
        error.unlink(missing_ok=True)
        return {"ok": True, "wall": record["wall_seconds"], "layers": record["n_layers"]}
    except Exception:
        error.parent.mkdir(parents=True, exist_ok=True)
        error.write_text(traceback.format_exc())
        return {"ok": False, "error": traceback.format_exc().strip().splitlines()[-1]}


_LIMITS = None


def _init_worker() -> None:
    global _LIMITS
    from threadpoolctl import threadpool_limits

    _LIMITS = threadpool_limits(1)
    warnings.filterwarnings("ignore", category=UserWarning)


def _git() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def fingerprints(all_tasks: list[Task]) -> tuple[dict[Task, str], dict[str, tuple[int, int]]]:
    """Fingerprint per task and (rows, features) per dataset; creates missing outer splits
    (parent process only, so workers never write split files)."""
    from .cascade import CascadeConfig
    from .baselines import LGBMConfig
    from .data import load, outer_folds

    code, out, cache = code_hash(), {}, {}
    for t in all_tasks:
        key = (t.dataset, t.seed, t.outer_folds)
        if key not in cache:
            data = load(t.dataset)
            cache[key] = (data.sha256, outer_folds(data, t.seed, t.outer_folds, create=True)[1], len(data.y),
                          data.X.shape[1])
        data_sha, split_sha, *_ = cache[key]
        payload = {"task": t.__dict__, "cascade": CascadeConfig(**json.loads(t.cascade)).__dict__,
                   "lightgbm": LGBMConfig(**json.loads(t.lightgbm)).__dict__, "code": code,
                   "data": data_sha, "split": split_sha}
        out[t] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return out, {k[0]: v[2:] for k, v in cache.items()}


def state(task: Task, results: Path, fingerprint: str) -> str:
    """'done', 'stale' (finished with another fingerprint), 'failed' or 'missing'."""
    base = task.path(results)
    if base.with_suffix(".json").exists() and base.with_suffix(".npz").exists():
        stored = json.loads(base.with_suffix(".json").read_text())["fingerprint"]
        return "done" if stored == fingerprint else "stale"
    return "failed" if base.with_suffix(".error.txt").exists() else "missing"


def write_manifest(name: str, results: Path, git: dict, datasets: dict) -> None:
    lock = ROOT / "uv.lock"
    manifest = {"experiment": name, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "settings": (EXPERIMENTS / f"{name}.toml").read_text(), "python": platform.python_version(),
                "platform": platform.platform(), "packages": {p: version(p) for p in PACKAGES},
                "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
                "code_sha256": code_hash(), "git": git, "datasets": datasets}
    folder = results / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def run_experiment(name, *, workers, results: Path, datasets=None, methods=None, seeds=None, folds=None,
                   checks=None, dry_run=False) -> bool:
    from .data import load

    selected = tasks(name, datasets, methods, seeds, folds, checks)
    fps, sizes = fingerprints(selected)
    states = {t: state(t, results, fps[t]) for t in selected}
    todo = [t for t in selected if states[t] != "done"]
    counts = {s: sum(v == s for v in states.values()) for s in ("done", "stale", "failed", "missing")}
    print(f"{name}: {len(selected)} tasks selected; " + ", ".join(f"{v} {k}" for k, v in counts.items()), flush=True)
    if counts["stale"]:
        print("  stale = finished with different code/data/settings; they will be re-run", flush=True)
    if dry_run or not todo:
        for t in todo[:20]:
            print(f"  would run: {t.label()} ({states[t]})")
        return True
    git = _git()
    write_manifest(name, results, git, {t.dataset: load(t.dataset).sha256 for t in selected})
    weight = {"rf": 0.2, "rf_target_xx": 0.2, "lightgbm": 0.4}
    todo.sort(key=lambda t: -sizes[t.dataset][0] * np.sqrt(sizes[t.dataset][1]) * weight.get(t.method, 1.0))
    log = open(results / name / "run.log", "a")
    remaining, failed, finished, start = list(todo), [], 0, time.time()

    def report(t, res):
        nonlocal finished
        finished += 1
        status = f"ok {res['wall']:.1f}s, {res['layers']} layers" if res["ok"] else f"FAILED: {res['error']}"
        line = f"[{finished}/{len(todo)}] {t.label()}: {status}"
        print(line, flush=True)
        log.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")
        log.flush()
        if not res["ok"]:
            failed.append(t)

    if workers == 1:
        _init_worker()
        for t in remaining:
            report(t, _safe_run(t, results, fps[t], git))
        remaining = []
    restarts = 0
    while remaining and restarts <= 3:
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
                futures = {pool.submit(_safe_run, t, results, fps[t], git): t for t in remaining}
                for future in as_completed(futures):
                    t = futures[future]
                    report(t, future.result())
                    remaining.remove(t)
        except BrokenProcessPool:
            restarts += 1
            print(f"a worker died (out of memory?); restarting the pool ({restarts}/3) "
                  f"for {len(remaining)} unfinished tasks", flush=True)
    log.close()
    hours = (time.time() - start) / 3600
    print(f"{name}: {finished - len(failed)} finished, {len(failed)} failed, {len(remaining)} not run, "
          f"{hours:.2f} h", flush=True)
    for t in failed:
        print(f"  failed: {t.label()} -> {t.path(results).with_suffix('.error.txt')}")
    return not failed and not remaining


def status(name: str, results: Path) -> None:
    selected = tasks(name)
    fps, _ = fingerprints(selected)
    table: dict[tuple, dict] = {}
    for t in selected:
        row = table.setdefault((t.check, t.method), {"done": 0, "stale": 0, "failed": 0, "missing": 0})
        row[state(t, results, fps[t])] += 1
    print(f"{'check':<16} {'method':<14} {'done':>6} {'stale':>6} {'failed':>6} {'missing':>8}")
    for (check, method), row in table.items():
        print(f"{check:<16} {method:<14} {row['done']:>6} {row['stale']:>6} {row['failed']:>6} {row['missing']:>8}")
    total = {k: sum(r[k] for r in table.values()) for k in ("done", "stale", "failed", "missing")}
    print(f"{'total':<31} {total['done']:>6} {total['stale']:>6} {total['failed']:>6} {total['missing']:>8}")

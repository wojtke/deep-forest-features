"""Expand experiments/<name>.toml into tasks, run them in parallel and store one result per task.

A task is (experiment, check, dataset, method, seed, outer fold). A finished task leaves
results/<experiment>/<check>/<dataset>/<method>/s<seed>-f<fold>.npz and .json. The JSON is written
last and holds a fingerprint of everything that determines the result (task, settings, code,
locked dependencies, data, split). A resumed run skips a task only if both files exist and the
fingerprint matches.
"""
import hashlib
import io
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .baselines import BASELINES, LGBMConfig, run_baseline
from .cascade import CascadeConfig, fit_predict
from .data import ROOT, load, outer_split
from .datasets import ALL
from .features import CASCADE_METHODS
from .metrics import metrics
from .prep import Preprocessor

EXPERIMENTS = ROOT / "experiments"
SRC = Path(__file__).resolve().parent
# Files whose content can change a result. datasets.py acts through the data hash; report.py and
# cli.py never change a result.
COMPUTE_FILES = ("data.py", "prep.py", "forest.py", "features.py", "cascade.py", "baselines.py", "metrics.py",
                 "run.py")
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "lightgbm")
METHODS = (*CASCADE_METHODS, *BASELINES)


def code_hash() -> str:
    """sha256 of the compute files and uv.lock."""
    h = hashlib.sha256()
    for path in [SRC / name for name in COMPUTE_FILES] + [ROOT / "uv.lock"]:
        h.update(path.read_bytes() if path.exists() else b"")
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


def _require(kind: str, chosen, valid) -> None:
    unknown = sorted(set(chosen or ()) - set(valid))
    if unknown:
        raise SystemExit(f"unknown {kind}: {', '.join(map(str, unknown))}; valid: {', '.join(map(str, valid))}")


def experiment_settings(name: str) -> dict:
    """experiments/<name>.toml with defaults filled in ("all" expands to every dataset / method)."""
    path = EXPERIMENTS / f"{name}.toml"
    _require("experiment", [name] if not path.exists() else [], sorted(p.stem for p in EXPERIMENTS.glob("*.toml")))
    s = tomllib.loads(path.read_text())
    if s.get("datasets", "all") == "all":
        s["datasets"] = list(ALL)
    if s.get("methods") == "all":
        s["methods"] = list(METHODS)
    s.setdefault("cascade", {})
    s.setdefault("lightgbm", {})
    s.setdefault("check", [{"name": "default", "methods": s.get("methods", [])}])
    _require("method in the experiment file", [m for c in s["check"] for m in c["methods"]], METHODS)
    return s


def tasks(name: str, datasets=None, methods=None, seeds=None, folds=None, checks=None) -> list[Task]:
    """Every task of an experiment, optionally filtered; unknown filter values are an error."""
    s = experiment_settings(name)
    _require("dataset", datasets, s["datasets"])
    _require("method", methods, sorted({m for c in s["check"] for m in c["methods"]}))
    _require("seed", seeds, s["seeds"])
    _require("fold", folds, range(s["outer_folds"]))
    _require("check", checks, [c["name"] for c in s["check"]])
    out = []
    for check in s["check"]:
        cascade = json.dumps({**s["cascade"], **check.get("cascade", {})}, sort_keys=True)
        lightgbm = json.dumps(s["lightgbm"], sort_keys=True)
        for dataset in s["datasets"]:
            for method in check["methods"]:
                for seed in s["seeds"]:
                    for fold in range(s["outer_folds"]):
                        if all(f is None or v in f for f, v in ((checks, check["name"]), (datasets, dataset),
                                                               (methods, method), (seeds, seed), (folds, fold))):
                            out.append(Task(name, check["name"], dataset, method, seed, fold, s["outer_folds"],
                                            cascade, lightgbm))
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_task(task: Task, results: Path, fingerprint: str, meta: dict) -> dict:
    """Fit one method on one outer training fold and score it on the outer test fold."""
    wall, cpu = time.time(), time.process_time()
    data = load(task.dataset)
    fold_of, _ = outer_split(data, task.seed, task.outer_folds)
    train, test = np.flatnonzero(fold_of != task.fold), np.flatnonzero(fold_of == task.fold)
    prep = Preprocessor().fit(data.X[train], data.categorical)  # outer-training rows only
    xs_tr, xs_te = prep.transform(data.X[train]), prep.transform(data.X[test])
    common = dict(task=data.task, n_classes=data.n_classes, cfg=CascadeConfig(**json.loads(task.cascade)),
                  keys=(task.seed, zlib.crc32(task.dataset.encode()), task.fold))
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
    record = {"task": asdict(task) | {"cascade": json.loads(task.cascade), "lightgbm": json.loads(task.lightgbm)},
              "fingerprint": fingerprint, "git": meta["git"], "versions": meta["versions"], "task_type": data.task,
              "n_train": len(train), "n_test": len(test), "n_classes": data.n_classes, "d": xs_tr.shape[1],
              "best_layer": out.best, "n_layers": len(out.layers), "metrics": layers[out.best - 1]["metrics"],
              "layers": layers, "info": out.info, "cpu_seconds": round(time.process_time() - cpu, 3),
              "wall_seconds": round(time.time() - wall, 3), "worker_peak_rss_mb": round(rss, 1)}
    arrays = {"test_index": test, "y_true": y_te,
              "pred_by_layer": np.stack([layer.test_pred for layer in out.layers]).astype(np.float32)}
    if out.inner_folds is not None:
        arrays["inner_folds"] = out.inner_folds
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    base = task.path(results)
    base.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(base.with_suffix(".npz"), buffer.getvalue())  # predictions first ...
    _atomic_write(base.with_suffix(".json"), (json.dumps(record, indent=1) + "\n").encode())  # ... then the marker
    return record


def _write_error(task: Task, results: Path, text: str) -> None:
    path = task.path(results).with_suffix(".error.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _safe_run(task: Task, results: Path, fingerprint: str, meta: dict) -> dict:
    try:
        if code_hash() != meta["code"]:
            raise RuntimeError("code or uv.lock changed since this run started; start it again")
        record = run_task(task, results, fingerprint, meta)
        task.path(results).with_suffix(".error.txt").unlink(missing_ok=True)
        return {"ok": True, "wall": record["wall_seconds"], "layers": record["n_layers"]}
    except Exception:
        _write_error(task, results, traceback.format_exc())
        return {"ok": False, "error": traceback.format_exc().strip().splitlines()[-1]}


def _init_worker() -> None:
    threadpool_limits(1)  # stays in force for the process
    warnings.filterwarnings("ignore", category=UserWarning)


def _git() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    changed = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results", ":(exclude)reports")
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(changed)}


def fingerprints(selected: list[Task], create: bool = False) -> tuple[dict[Task, str], dict[str, int]]:
    """Fingerprint per task and rows per dataset. Missing outer splits are created only if
    `create` (by `dfx run`, in the parent process, never by workers)."""
    code, datasets, splits, out = code_hash(), {}, {}, {}
    for t in selected:
        if t.dataset not in datasets:
            datasets[t.dataset] = load(t.dataset)
        key = (t.dataset, t.seed, t.outer_folds)
        if key not in splits:
            splits[key] = outer_split(datasets[t.dataset], t.seed, t.outer_folds, create)[1]
        payload = {"task": asdict(t), "cascade": asdict(CascadeConfig(**json.loads(t.cascade))),
                   "lightgbm": asdict(LGBMConfig(**json.loads(t.lightgbm))), "code": code,
                   "data": datasets[t.dataset].sha256, "split": splits[key]}
        out[t] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return out, {name: len(d.y) * np.sqrt(d.X.shape[1]) for name, d in datasets.items()}


def state(task: Task, results: Path, fingerprint: str) -> str:
    """'done', 'stale' (finished with another fingerprint), 'failed' or 'missing'."""
    base = task.path(results)
    try:
        stored = json.loads(base.with_suffix(".json").read_text())["fingerprint"]
        if base.with_suffix(".npz").exists():
            return "done" if stored == fingerprint else "stale"
    except (OSError, ValueError, KeyError):  # absent or unreadable
        pass
    return "failed" if base.with_suffix(".error.txt").exists() else "missing"


def write_manifest(name: str, results: Path, meta: dict) -> None:
    """results/<name>/manifest.json: environment and data of the whole experiment."""
    manifest = {"experiment": name, "settings": (EXPERIMENTS / f"{name}.toml").read_text(),
                "python": platform.python_version(), "platform": platform.platform(), "packages": meta["versions"],
                "code_and_lock_sha256": meta["code"], "git": meta["git"],
                "datasets": {n: load(n).sha256 for n in experiment_settings(name)["datasets"]}}
    folder = results / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def _run_pool(queue: list[Task], workers: int, results: Path, fps: dict, meta: dict, report) -> list[Task]:
    """Run tasks from `queue` (consumed) with at most one task per worker in flight. Returns []
    when the queue is done, or the tasks that were running when a worker process died."""
    running = {}
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
            while queue or running:
                while queue and len(running) < workers:
                    t = queue.pop(0)
                    running[pool.submit(_safe_run, t, results, fps[t], meta)] = t
                for future in wait(running, return_when=FIRST_COMPLETED)[0]:
                    report(running[future], future.result())
                    del running[future]
    except BrokenProcessPool:
        return list(running.values())
    return []


def run_experiment(name, *, workers, results: Path, datasets=None, methods=None, seeds=None, folds=None,
                   checks=None, dry_run=False) -> bool:
    selected = tasks(name, datasets, methods, seeds, folds, checks)
    fps, cost = fingerprints(selected, create=True)
    states = {t: state(t, results, fps[t]) for t in selected}
    todo = [t for t in selected if states[t] != "done"]
    counts = {s: sum(v == s for v in states.values()) for s in ("done", "stale", "failed", "missing")}
    print(f"{name}: {len(selected)} tasks selected; " + ", ".join(f"{v} {k}" for k, v in counts.items()), flush=True)
    if counts["stale"]:
        print("  stale = finished with other code, data or settings; they will be re-run", flush=True)
    if dry_run or not todo:
        for t in todo[:20]:
            print(f"  would run: {t.label()} ({states[t]})")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return True
    meta = {"code": code_hash(), "git": _git(), "versions": {p: version(p) for p in PACKAGES}}
    write_manifest(name, results, meta)
    todo.sort(key=lambda t: -cost[t.dataset] * {"rf": 0.2, "rf_target_xx": 0.2, "lightgbm": 0.4}.get(t.method, 1))
    failed, finished, start = [], 0, time.time()

    def report(t, res):
        nonlocal finished
        finished += 1
        layers = f"{res.get('layers')} layer{'s' if res.get('layers') != 1 else ''}"
        line = f"[{finished}/{len(todo)}] {t.label()}: " + (f"ok {res['wall']:.1f}s, {layers}" if res["ok"]
                                                              else f"FAILED: {res['error']}")
        print(line, flush=True)
        with open(results / name / "run.log", "a") as log:
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
        if not res["ok"]:
            failed.append(t)

    if workers == 1:
        _init_worker()
        for t in todo:
            report(t, _safe_run(t, results, fps[t], meta))
    else:
        queue = list(todo)
        while queue:
            for t in _run_pool(queue, workers, results, fps, meta, report):
                # A worker died while these ran; run each alone so a crash names its task.
                print(f"a worker process died; re-running {t.label()} on its own", flush=True)
                if _run_pool([t], 1, results, fps, meta, report):
                    message = "the worker process died while running this task (out of memory?)"
                    _write_error(t, results, message)
                    report(t, {"ok": False, "error": message})
    print(f"{name}: {finished - len(failed)} finished, {len(failed)} failed, {(time.time() - start) / 60:.1f} min",
          flush=True)
    for t in failed:
        print(f"  failed: {t.label()} -> {t.path(results).with_suffix('.error.txt')}")
    return not failed


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

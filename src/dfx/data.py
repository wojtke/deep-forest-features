"""Datasets: download, prepare, related-row groups, subsampling and splits (plan §4, §5).

Each dataset is a `Spec` in `datasets.py`. `prepare()` turns its raw files into
`data/prepared/<name>.npz`: features with NaN for missing values (categorical columns as integer
codes), the target, related-row groups and the original row numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SPLITS = ROOT / "splits"
MAX_ROWS = 20_000  # plan §4
SUBSAMPLE_SEED = 0  # ours


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    sha256: str | None = None  # pinned after the first download; `dfx data` prints it


@dataclass(frozen=True)
class Spec:
    name: str
    task: str  # "classification" or "regression"
    target: str
    read: Callable[[Path], pd.DataFrame]  # raw-file directory -> one frame with every column
    sources: tuple[Source, ...] = ()
    drop: tuple[str, ...] = ()  # ids, timestamps, weights and columns that reveal the target
    group: str | None = None  # rows with equal values here are related (plan §5)
    categorical: tuple[str, ...] = ()  # numeric-coded categoricals; text columns always are
    missing: tuple = ()  # extra missing-value markers, e.g. -200 in Air Quality
    plan: tuple[int, int] | None = None  # rows x features in plan Table 2
    note: str = ""  # one line for data/AUDIT.md
    smoke: bool = False  # tiny data for tests only, never part of "all"


@dataclass
class Dataset:
    name: str
    task: str
    X: np.ndarray  # float64; NaN = missing; categorical columns hold codes 0..L-1
    y: np.ndarray  # class codes 0..C-1 or float target
    groups: np.ndarray  # related rows share a value
    row_ids: np.ndarray  # row number in the raw frame
    features: list[str]
    categorical: np.ndarray  # bool per feature
    classes: list[str]  # class labels in code order (classification)
    sha256: str

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.task == "classification" else 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(spec: Spec) -> Path:
    """Fetch the raw files once into data/raw/<name>/ and check their pinned hashes."""
    folder = DATA / "raw" / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    for source in spec.sources:
        path = folder / source.filename
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            request = urllib.request.Request(source.url, headers={"User-Agent": "dfx"})
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as f:
                f.write(response.read())
            os.replace(tmp, path)
        digest = sha256_file(path)
        if source.sha256 is None:
            print(f"  {spec.name}/{source.filename}: unpinned sha256 {digest}")
        elif digest != source.sha256:
            raise ValueError(f"{spec.name}/{source.filename}: sha256 {digest} != pinned {source.sha256}")
    return folder


def related_groups(X: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
    """Rows with identical inputs, or equal `key`, share a group (connected components)."""
    filled = np.where(np.isnan(X), np.inf, X)
    dup = np.unique(filled, axis=0, return_inverse=True)[1].reshape(-1)
    if key is None:
        return dup
    k = np.unique(np.asarray(key, dtype=str), return_inverse=True)[1].reshape(-1)
    n, nd, nk = len(X), dup.max() + 1, k.max() + 1
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[n + dup, n + nd + k]
    graph = coo_matrix((np.ones(2 * n), (rows, cols)), shape=(n + nd + nk,) * 2)
    labels = connected_components(graph, directed=False)[1][:n]
    return np.unique(labels, return_inverse=True)[1]


def subsample(y: np.ndarray, groups: np.ndarray, task: str, max_rows: int, seed: int) -> np.ndarray:
    """Rows to keep: whole groups, at most `max_rows`, class proportions kept as closely as
    whole groups allow (ours). A group's class is the class of its first row."""
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    ids, first, sizes = np.unique(groups, return_index=True, return_counts=True)
    strata = y[first] if task == "classification" else np.zeros(len(ids))
    keep = []
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        quota = int(max_rows * sizes[members].sum() / len(y))
        total = 0
        for m in rng.permutation(members):
            if total + sizes[m] <= quota:
                keep.append(ids[m])
                total += sizes[m]
    return np.flatnonzero(np.isin(groups, keep))


def encode(frame: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features as float codes: text or listed columns become categorical codes (sorted labels)."""
    columns, categorical = [], []
    for name in frame.columns:
        col = frame[name]
        is_cat = name in spec.categorical or not pd.api.types.is_numeric_dtype(col)
        if is_cat:
            text = col.astype("string").str.strip()
            labels = sorted(text.dropna().unique())
            codes = text.map({v: i for i, v in enumerate(labels)}).astype("Float64")
            columns.append(codes.to_numpy(dtype=float, na_value=np.nan))
        else:
            columns.append(col.to_numpy(dtype=float, na_value=np.nan))
        categorical.append(is_cat)
    return np.column_stack(columns), np.array(categorical), list(frame.columns)


def prepare(spec: Spec) -> tuple[Dataset, dict]:
    """Raw frame -> Dataset (plan §4): drop rows with a missing target and excluded columns,
    form related-row groups, subsample to 20,000 rows. Returns the dataset and audit facts."""
    frame = spec.read(download(spec))
    raw_rows, raw_features = frame.shape[0], frame.shape[1] - 1
    unknown = [c for c in (*spec.drop, spec.target, spec.group, *spec.categorical) if c and c not in frame]
    if unknown:
        raise KeyError(f"{spec.name}: columns not in data: {unknown}; have {list(frame.columns)}")
    for marker in spec.missing:
        frame = frame.replace(marker, np.nan)
    target = frame[spec.target]
    if spec.task == "classification":
        target = target.astype("string").str.strip()
    has_target = target.notna().to_numpy()
    frame, target = frame[has_target], target[has_target]
    row_ids = np.flatnonzero(has_target)
    key = frame[spec.group].to_numpy() if spec.group else None
    features = frame.drop(columns=[c for c in (spec.target, spec.group, *spec.drop) if c])
    X, categorical, names = encode(features, spec)
    if spec.task == "classification":
        classes = sorted(target.unique())
        y = target.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=int)
    else:
        classes, y = [], target.to_numpy(dtype=float)
    groups = related_groups(X, key)
    keep = subsample(y, groups, spec.task, MAX_ROWS, SUBSAMPLE_SEED)
    X, y, groups, row_ids = X[keep], y[keep], np.unique(groups[keep], return_inverse=True)[1], row_ids[keep]
    digest = hashlib.sha256(X.tobytes() + y.tobytes() + groups.tobytes()).hexdigest()
    data = Dataset(spec.name, spec.task, X, y, groups, row_ids, names, categorical, classes, digest)
    facts = {"raw_rows": raw_rows, "raw_features": raw_features, "dropped_missing_target": int((~has_target).sum()),
             "excluded": [c for c in (spec.group, *spec.drop) if c], "rows": len(y), "features": len(names),
             "categorical": int(categorical.sum()), "groups": int(groups.max() + 1),
             "missing_pct": float(100 * np.isnan(X).mean())}
    return data, facts


def save(data: Dataset, facts: dict) -> None:
    """Write data/prepared/<name>.npz and a JSON with metadata and audit facts."""
    from .features import k_products
    from .prep import Preprocessor

    folder = DATA / "prepared"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{data.name}.npz", X=data.X, y=data.y, groups=data.groups,
                        row_ids=data.row_ids, categorical=data.categorical)
    d = len(Preprocessor().fit(data.X, data.categorical).origin)
    facts = {**facts, "d": d, "K": k_products(d)}
    if data.task == "classification":
        facts["class_counts"] = np.bincount(data.y, minlength=len(data.classes)).tolist()
    else:
        facts["target_mean_std"] = [float(data.y.mean()), float(data.y.std())]
    meta = {"name": data.name, "task": data.task, "features": data.features, "classes": data.classes,
            "sha256": data.sha256, "rows": len(data.y), "facts": facts}
    (folder / f"{data.name}.json").write_text(json.dumps(meta, indent=1) + "\n")


def write_audit() -> Path:
    """data/AUDIT.md: one row per prepared dataset, compared with plan Table 2."""
    from .datasets import ALL, DATASETS

    head = ("| dataset | task | target | plan rows x feat | raw rows x feat | no target | rows used | "
            "features | categ. | d | K | groups | classes / target mean±sd | missing % | excluded | note |")
    lines = ["# Data audit", "",
             "Generated by `dfx data`. *raw* = the source file before exclusions; *rows used* after "
             "dropping missing targets and subsampling to 20,000; *d* = features after one-hot "
             "encoding on all rows (per outer fold it can differ slightly); *K* = min(d, floor(d(d-1)/8)).",
             "", head, "|" + "---|" * (head.count("|") - 1)]
    for name in ALL:
        path = DATA / "prepared" / f"{name}.json"
        if not path.exists():
            lines.append(f"| {name} | not prepared |" + " |" * 14)
            continue
        meta, spec = json.loads(path.read_text()), DATASETS[name]
        f = meta["facts"]
        plan = f"{spec.plan[0]:,} x {spec.plan[1]}" if spec.plan else "-"
        if meta["task"] == "classification":
            summary = ", ".join(f"{c}: {n}" for c, n in zip(meta["classes"], f["class_counts"]))
        else:
            summary = "{:.4g} ± {:.4g}".format(*f["target_mean_std"])
        lines.append(f"| {name} | {meta['task'][:3]} | {spec.target} | {plan} | {f['raw_rows']:,} x {f['raw_features']} | "
                     f"{f['dropped_missing_target']} | {f['rows']:,} | {f['features']} | {f['categorical']} | {f['d']} | "
                     f"{f['K']} | {f['groups']:,} | {summary} | {f['missing_pct']:.1f} | {', '.join(f['excluded']) or '-'} | "
                     f"{spec.note} |")
    path = DATA / "AUDIT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def load(name: str) -> Dataset:
    """A prepared dataset; smoke datasets are built in memory and need no files."""
    from .datasets import DATASETS, smoke_dataset

    if DATASETS[name].smoke:
        return smoke_dataset(name)
    folder = DATA / "prepared"
    meta = json.loads((folder / f"{name}.json").read_text())
    with np.load(folder / f"{name}.npz") as z:
        return Dataset(name, meta["task"], z["X"], z["y"], z["groups"], z["row_ids"], meta["features"],
                       z["categorical"], meta["classes"], meta["sha256"])


def _splitter(task: str, n_folds: int, seed: int):
    if task == "classification":
        return StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    return GroupKFold(n_folds, shuffle=True, random_state=seed)


def fold_ids(y: np.ndarray, groups: np.ndarray, task: str, n_folds: int, seed: int) -> np.ndarray:
    """Fold number of every row; related rows share a fold, classes stratified."""
    ids = np.empty(len(y), dtype=np.int8)
    for f, (_, test) in enumerate(_splitter(task, n_folds, seed).split(np.zeros(len(y)), y, groups)):
        ids[test] = f
    return ids


def inner_folds(y: np.ndarray, groups: np.ndarray, task: str, n_folds: int, seed: int) -> list:
    """(train, validation) row indices of the OOF folds inside an outer training set."""
    ids = fold_ids(y, groups, task, n_folds, seed)
    return [(np.flatnonzero(ids != f), np.flatnonzero(ids == f)) for f in range(n_folds)]


def outer_folds(data: Dataset, seed: int, n_folds: int, create: bool = False) -> tuple[np.ndarray, str]:
    """Saved outer fold ids for (seed, n_folds) and their sha256. Splits live in splits/<name>.npz;
    they are created by the parent process only (`create=True`), never by workers."""
    if _is_smoke(data.name):
        ids = fold_ids(data.y, data.groups, data.task, n_folds, seed)
        return ids, hashlib.sha256(ids.tobytes()).hexdigest()
    path = SPLITS / f"{data.name}.npz"
    key = f"seed{seed}_folds{n_folds}"
    stored = dict(np.load(path)) if path.exists() else {}
    if stored and str(stored["dataset_sha256"]) != data.sha256:
        raise ValueError(f"{path} was made for other data; delete it to re-split")
    if key not in stored:
        if not create:
            raise FileNotFoundError(f"missing split {key} in {path}")
        stored[key] = fold_ids(data.y, data.groups, data.task, n_folds, seed)
        stored["dataset_sha256"] = np.array(data.sha256)
        SPLITS.mkdir(exist_ok=True)
        np.savez_compressed(path, **stored)
    ids = stored[key]
    return ids, hashlib.sha256(ids.tobytes()).hexdigest()


def _is_smoke(name: str) -> bool:
    """Smoke datasets are split on the fly and never written to splits/."""
    from .datasets import DATASETS

    return DATASETS[name].smoke

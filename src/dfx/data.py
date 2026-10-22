"""Datasets: download, prepare, related-row groups, subsampling and splits (plan §4, §5).

Each dataset is a `Spec` in `datasets.py`. `prepare()` turns its raw archive into
`data/prepared/<name>.npz`: features with NaN for missing values (categorical columns as integer
codes), the target, related-row groups and the original row numbers.
"""
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
class Spec:
    name: str
    task: str  # "classification" or "regression"
    target: str
    read: Callable[[Path], pd.DataFrame]  # raw folder -> one frame with every column
    url: str | None = None  # archive saved as data/raw/<name>/<name>.zip
    sha256: str | None = None  # pinned after the first download; `dfx data` prints it
    drop: tuple[str, ...] = ()  # ids, timestamps, weights and columns that reveal the target
    group: str | None = None  # rows with equal values here are related (plan §5)
    categorical: tuple[str, ...] = ()  # numeric-coded categoricals; text columns always are
    missing: tuple = ()  # extra missing-value markers, e.g. -200 in Air Quality
    plan: tuple[int, int] | None = None  # rows x features in plan Table 2
    note: str = ""  # one line for data/AUDIT.md


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
    """Fetch the archive once into data/raw/<name>/ and check its pinned hash."""
    folder = DATA / "raw" / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    if spec.url:
        path = folder / f"{spec.name}.zip"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            request = urllib.request.Request(spec.url, headers={"User-Agent": "dfx"})
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as f:
                f.write(response.read())
            os.replace(tmp, path)
        digest = sha256_file(path)
        if spec.sha256 is None:
            print(f"  {spec.name}: unpinned sha256 {digest}")
        elif digest != spec.sha256:
            raise ValueError(f"{spec.name}: sha256 {digest} != pinned {spec.sha256}")
    return folder


def related_groups(X: np.ndarray, key=None) -> np.ndarray:
    """Rows with identical inputs, or equal non-missing `key`, share a group (connected components)."""
    dup = np.unique(np.where(np.isnan(X), np.inf, X), axis=0, return_inverse=True)[1].reshape(-1)
    if key is None:
        return dup
    k = pd.factorize(pd.Series(key))[0]  # missing keys get -1 ...
    k = np.where(k < 0, k.max() + 1 + np.arange(len(k)), k)  # ... and then a group of their own
    n, nd = len(X), dup.max() + 1
    edges = (np.r_[np.arange(n), np.arange(n)], np.r_[n + dup, n + nd + k])
    graph = coo_matrix((np.ones(2 * n), edges), shape=(n + nd + k.max() + 1,) * 2)
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
        quota, total = int(max_rows * sizes[members].sum() / len(y)), 0
        for m in rng.permutation(members):
            if total + sizes[m] <= quota:
                keep.append(ids[m])
                total += sizes[m]
    return np.flatnonzero(np.isin(groups, keep))


def encode(frame: pd.DataFrame, categorical: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Features as floats: text or listed columns become categorical codes (sorted labels)."""
    columns, is_cat = [], []
    for name in frame.columns:
        col = frame[name]
        cat = name in categorical or not pd.api.types.is_numeric_dtype(col)
        if cat:
            text = col.astype("string").str.strip()
            col = text.map({v: i for i, v in enumerate(sorted(text.dropna().unique()))}).astype("Float64")
        columns.append(col.to_numpy(dtype=float, na_value=np.nan))
        is_cat.append(cat)
    return np.column_stack(columns), np.array(is_cat)


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
    features = frame.drop(columns=[c for c in (spec.target, spec.group, *spec.drop) if c])
    has_target, has_inputs = target.notna().to_numpy(), features.notna().any(axis=1).to_numpy()
    usable = has_target & has_inputs  # rows without any feature value carry nothing (ours)
    frame, target, features = frame[usable], target[usable], features[usable]
    X, categorical = encode(features, spec.categorical)
    if spec.task == "classification":
        classes = sorted(target.unique())
        y = target.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=int)
    else:
        classes, y = [], target.to_numpy(dtype=float)
    groups = related_groups(X, frame[spec.group].to_numpy() if spec.group else None)
    keep = subsample(y, groups, spec.task, MAX_ROWS, SUBSAMPLE_SEED)
    X, y, row_ids = X[keep], y[keep], np.flatnonzero(usable)[keep]
    groups = np.unique(groups[keep], return_inverse=True)[1]
    digest = hashlib.sha256(X.tobytes() + y.tobytes() + groups.tobytes()).hexdigest()
    data = Dataset(spec.name, spec.task, X, y, groups, row_ids, list(features.columns), categorical, classes, digest)
    facts = {"raw_rows": raw_rows, "raw_features": raw_features, "dropped_missing_target": int((~has_target).sum()),
             "dropped_no_inputs": int((has_target & ~has_inputs).sum()),
             "excluded": [c for c in (spec.group, *spec.drop) if c], "rows": len(y), "features": X.shape[1],
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
    d = Preprocessor().fit(data.X, data.categorical).width
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

    head = ("| dataset | task | target | plan rows x feat | raw rows x feat | dropped rows | rows used | "
            "features | categ. | d | K | groups | classes / target mean±sd | missing % | excluded | note |")
    lines = ["# Data audit", "",
             "Generated by `dfx data`. *raw* = the source file before exclusions; *dropped rows* = rows "
             "without a target + rows without any feature value; *rows used* after those and subsampling to "
             "20,000; *d* = features after one-hot "
             "encoding on all rows (per outer fold it can differ slightly); *K* = min(d, floor(d(d-1)/8)). "
             "Notes starting with PROPOSED are choices the plan does not make; they need confirming.",
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
        cells = [name, meta["task"][:3], spec.target, plan, f"{f['raw_rows']:,} x {f['raw_features']}",
                 f"{f['dropped_missing_target']} + {f['dropped_no_inputs']}", f"{f['rows']:,}", f["features"],
                 f["categorical"], f["d"], f["K"], f"{f['groups']:,}", summary, f"{f['missing_pct']:.1f}",
                 ", ".join(f["excluded"]) or "-", spec.note]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    path = DATA / "AUDIT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def load(name: str) -> Dataset:
    """A prepared dataset; smoke datasets are built in memory and need no files."""
    from .datasets import SMOKE, smoke_dataset

    if name in SMOKE:
        return smoke_dataset(name)
    folder = DATA / "prepared"
    meta = json.loads((folder / f"{name}.json").read_text())
    with np.load(folder / f"{name}.npz") as z:
        return Dataset(name, meta["task"], z["X"], z["y"], z["groups"], z["row_ids"], meta["features"],
                       z["categorical"], meta["classes"], meta["sha256"])


def fold_ids(y: np.ndarray, groups: np.ndarray, task: str, n_folds: int, seed: int) -> np.ndarray:
    """Fold number of every row; related rows share a fold, classes stratified."""
    splitter = StratifiedGroupKFold if task == "classification" else GroupKFold
    ids = np.empty(len(y), dtype=np.int8)
    for f, (_, test) in enumerate(splitter(n_folds, shuffle=True, random_state=seed).split(y, y, groups)):
        ids[test] = f
    return ids


def inner_split(y: np.ndarray, groups: np.ndarray, task: str, n_folds: int, seed: int) -> tuple[np.ndarray, list]:
    """OOF folds inside an outer training set: fold id per row and (train, validation) index pairs."""
    ids = fold_ids(y, groups, task, n_folds, seed)
    return ids, [(np.flatnonzero(ids != f), np.flatnonzero(ids == f)) for f in range(n_folds)]


def outer_split(data: Dataset, seed: int, n_folds: int, create: bool = False) -> tuple[np.ndarray, str]:
    """Saved outer fold ids for (seed, n_folds) and their sha256. They live in splits/<name>.npz and
    are created only when `create` is set, by the parent process (`dfx data`, `dfx run`)."""
    from .datasets import SMOKE

    if data.name in SMOKE:  # split on the fly, never written
        ids = fold_ids(data.y, data.groups, data.task, n_folds, seed)
        return ids, hashlib.sha256(ids.tobytes()).hexdigest()
    path, key = SPLITS / f"{data.name}.npz", f"seed{seed}_folds{n_folds}"
    stored = dict(np.load(path)) if path.exists() else {}
    if stored and str(stored["dataset_sha256"]) != data.sha256:
        raise ValueError(f"{path} was made for other data; delete it to re-split")
    if key not in stored:
        if not create:
            raise FileNotFoundError(f"missing split {key} in {path}; run `dfx data {data.name}`")
        stored[key] = fold_ids(data.y, data.groups, data.task, n_folds, seed)
        stored["dataset_sha256"] = np.array(data.sha256)
        SPLITS.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **stored)
        os.replace(tmp, path)
    return stored[key], hashlib.sha256(stored[key].tobytes()).hexdigest()

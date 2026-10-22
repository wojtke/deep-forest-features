"""Columns added between cascade layers (plan §3, Table 1).

Every added column is computed from raw state: standardized unbinned features `Xs` and the
previous layer's outputs on the original scale (`Outputs`). Binning happens later, at the forest
input. A component is fitted on one fold model's training rows only and then applied unchanged.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

CHUNK = 512  # candidate columns scored at once: about 45 MB at 11k rows
CONSTANT = 1e-12  # variance at or below this counts as constant (ours)

# The cascade methods of plan Table 1 and the components each adds.
CASCADE_METHODS = {
    "base": (),
    "random_xx": ("random_xx",),
    "target_xx": ("target_xx",),
    "residual_xx": ("residual_xx",),
    "xp_mi": ("xp_mi",),
    "nonlinear": ("nonlinear",),
    "monotone": ("monotone",),
    "combined": ("nonlinear", "residual_xx"),
}


def k_products(d: int, mode: str = "base") -> int:
    """K = min(d, floor(M_XX / 4)) with M_XX = d(d-1)/2 (plan §3, ours). The sensitivity checks
    use ceil(K/2) ("half") and 2K ("double")."""
    k = min(d, d * (d - 1) // 2 // 4)
    return {"base": k, "half": -(-k // 2), "double": 2 * k}[mode]


@dataclass
class Outputs:
    """One layer's outputs for a set of rows, original scale.

    forests: (n, F, C) output of each forest, RF forests first (classification: class
    probabilities; regression: C = 1). sq: (n, F) mean squared tree prediction of each forest
    (regression only)."""
    forests: np.ndarray
    n_rf: int
    sq: np.ndarray | None = None

    def rows(self, idx) -> "Outputs":
        return Outputs(self.forests[idx], self.n_rf, None if self.sq is None else self.sq[idx])

    @property
    def P(self) -> np.ndarray:
        """The prediction columns a layer passes on: every forest's output, concatenated."""
        return self.forests.reshape(len(self.forests), -1)

    def mean(self) -> np.ndarray:
        """(n, C) mean over all forests: the layer's prediction."""
        return self.forests.mean(axis=1)

    def type_means(self) -> np.ndarray:
        """(n, 2, C): p_RF and p_ERF, each the mean of that type's forests."""
        return np.stack([self.forests[:, :self.n_rf].mean(1), self.forests[:, self.n_rf:].mean(1)], 1)

    def tree_std(self) -> np.ndarray:
        """(n, 2) standard deviation (ddof 0) of the tree predictions pooled within each type
        (all forests have the same number of trees)."""
        sq = np.column_stack([self.sq[:, :self.n_rf].mean(1), self.sq[:, self.n_rf:].mean(1)])
        return np.sqrt(np.maximum(sq - self.type_means()[:, :, 0] ** 2, 0))


def _abs_corr(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """max over target columns of |corr(value column, target column)|; undefined -> 0."""
    v = values - values.mean(0)
    t = targets - targets.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(v.T @ t) / (np.linalg.norm(v, axis=0)[:, None] * np.linalg.norm(t, axis=0)[None, :])
    return np.nan_to_num(r, nan=0.0, posinf=0.0).max(axis=1)


def _chunked(fn, idx: np.ndarray) -> np.ndarray:
    """fn applied to slices of idx of at most CHUNK candidates, results concatenated."""
    parts = [fn(idx[s:s + CHUNK]) for s in range(0, len(idx), CHUNK)]
    return np.concatenate(parts) if parts else np.empty(0)


class Products:
    """K products x_i x_j (XX) or x_i p_{t,c} (XP), chosen on training rows (plan §3).

    selector: "random" (redrawn for every layer and fold model), "target" (max_c |corr| with
    y_c), "residual" (max_c |corr| with the layer error y_c - p̄_c; equals target without a
    previous layer) or "mi" (mutual information with y, 3 neighbours). Candidates constant on the
    training rows, e.g. products of two dummies of one categorical feature, are dropped first.
    """

    def __init__(self, kind: str, selector: str, k: int):
        self.kind, self.selector, self.k = kind, selector, k

    def _right(self, Xs, prev):
        if self.kind == "xx":
            return Xs
        p = prev.type_means().reshape(len(Xs), -1)  # (n, 2C)
        return (p - self.p_mean) / self.p_scale if self.regression else p

    def fit(self, Xs, prev, y, *, task, n_classes, seed):
        self.regression = task == "regression"
        d = Xs.shape[1]
        if self.kind == "xx":
            self.a, self.b = np.triu_indices(d, 1)
        else:
            p = prev.type_means().reshape(len(Xs), -1)
            self.p_mean, self.p_scale = p.mean(0), np.where(p.std(0) > 0, p.std(0), 1.0)
            self.a, self.b = np.repeat(np.arange(d), p.shape[1]), np.tile(np.arange(p.shape[1]), d)
        right = self._right(Xs, prev)

        def values(idx):
            return Xs[:, self.a[idx]] * right[:, self.b[idx]]

        candidates = np.arange(len(self.a))
        pool = candidates[_chunked(lambda c: values(c).var(0) > CONSTANT, candidates).astype(bool)]
        k = min(self.k, len(pool))
        if self.selector == "random":
            chosen = np.random.default_rng(seed).choice(pool, k, replace=False)
        else:
            if self.selector == "mi":
                mi = mutual_info_classif if task == "classification" else mutual_info_regression
                score = mi(values(pool), y, n_neighbors=3, random_state=seed)
            else:
                target = np.eye(n_classes)[y] if task == "classification" else y[:, None].astype(float)
                if self.selector == "residual" and prev is not None:
                    target = target - prev.mean()
                score = _chunked(lambda c: _abs_corr(values(c), target), pool)
            chosen = pool[np.argsort(-score, kind="stable")[:k]]
        chosen = np.sort(chosen)
        self.a, self.b = self.a[chosen], self.b[chosen]
        self.info = {"pairs": np.column_stack([self.a, self.b]).tolist(), "pool": len(pool)}
        if self.kind == "xp" and k:
            # Plan §3: how strongly the chosen XP products repeat the prediction columns.
            self.info["max_corr_with_P"] = _abs_corr(self.transform(Xs, prev), prev.P).round(4).tolist()
        return self

    def transform(self, Xs, prev):
        return Xs[:, self.a] * self._right(Xs, prev)[:, self.b]


class NonLinear:
    """g(P) = H_RF, H_ERF, D (plan eq. 3). Classification: entropy of each type's mean
    probabilities and L1 distance between the types. Regression: standard deviation of the
    type's tree predictions and |mean_RF - mean_ERF|."""
    info: dict = {}

    def fit(self, Xs, prev, y, **_):
        return self

    def transform(self, Xs, prev):
        tm = prev.type_means()
        if prev.sq is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -np.where(tm > 0, tm * np.log(tm), 0.0).sum(-1)
            d = np.abs(tm[:, 0] - tm[:, 1]).sum(-1)
        else:
            h, d = prev.tree_std(), np.abs(tm[:, 0, 0] - tm[:, 1, 0])
        return np.column_stack([h, d])


ACTIVATIONS = {"sigmoid": expit, "relu": lambda x: np.maximum(x, 0), "gelu": lambda x: x * ndtr(x)}


class Monotone:
    """A monotone activation of every prediction column (plan §3). Regression outputs are first
    scaled to [0, 1] with the training rows' min and max (ours; not clipped)."""
    info: dict = {}

    def __init__(self, activation: str):
        self.act = ACTIVATIONS[activation]

    def fit(self, Xs, prev, y, *, task, **_):
        self.lo = self.scale = None
        if task == "regression":
            P = prev.P
            self.lo, span = P.min(0), P.max(0) - P.min(0)
            self.scale = np.where(span > 0, span, 1.0)
        return self

    def transform(self, Xs, prev):
        P = prev.P
        return self.act(P if self.lo is None else (P - self.lo) / self.scale)


def components(method: str, layer: int, k: int, activation: str) -> list[tuple[str, object]]:
    """Fresh, unfitted components of `method` that are active in `layer`: XX products enter
    every layer, the others start at layer 2 (plan Table 1)."""
    out = []
    for name in CASCADE_METHODS[method]:
        if name.endswith("_xx"):
            out.append((name, Products("xx", name.removesuffix("_xx"), k)))
        elif layer >= 2 and name == "xp_mi":
            out.append((name, Products("xp", "mi", k)))
        elif layer >= 2 and name == "nonlinear":
            out.append((name, NonLinear()))
        elif layer >= 2 and name == "monotone":
            out.append((name, Monotone(activation)))
    return out

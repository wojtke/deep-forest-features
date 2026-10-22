"""Preprocessing fitted on training rows only (plan §4) and the DF21 percentile binner."""
import numpy as np


class Preprocessor:
    """Median / mode imputation, one-hot encoding of categorical columns, standardization of
    numeric ones. Fitted on outer-training rows only (plan §4); every method reuses the same fit.

    Unseen categories become all-zero dummies, which stay 0/1. Columns constant on the training
    rows are dropped (ours).
    """

    def fit(self, X: np.ndarray, categorical: np.ndarray):
        self.parts = []
        for j in range(X.shape[1]):
            seen = X[~np.isnan(X[:, j]), j]
            if seen.size == 0:
                continue
            if categorical[j]:
                levels, counts = np.unique(seen, return_counts=True)
                if len(levels) > 1:
                    self.parts.append((j, "cat", levels[np.argmax(counts)], levels))
            else:
                median = np.median(seen)
                filled = np.where(np.isnan(X[:, j]), median, X[:, j])
                if filled.std() > 0:
                    self.parts.append((j, "num", median, (filled.mean(), filled.std())))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        blocks = []
        for j, kind, fill, extra in self.parts:
            col = np.where(np.isnan(X[:, j]), fill, X[:, j])
            if kind == "cat":
                blocks.append((col[:, None] == extra[None, :]).astype(float))
            else:
                mean, std = extra
                blocks.append(((col - mean) / std)[:, None])
        return np.hstack(blocks) if blocks else np.zeros((len(X), 0))


class Binner:
    """DF21 percentile binning (deepforest/_binner.py L23-86): at most 255 bins, thresholds at
    `np.percentile(col, linspace(0, 100, 256)[1:-1], method="midpoint")`; a column with at most
    255 distinct values is binned exactly at midpoints between them. x <= threshold goes to the
    lower bin. `n_bins = 0` disables binning (sensitivity check). Output is float32 for sklearn.
    """

    def __init__(self, n_bins: int = 255):
        self.n_bins = n_bins

    def fit(self, Z: np.ndarray):
        self.edges = []
        if self.n_bins:
            for col in np.asarray(Z, dtype=float).T:
                distinct = np.unique(col)
                if len(distinct) <= self.n_bins:
                    edges = (distinct[:-1] + distinct[1:]) / 2
                else:
                    q = np.linspace(0, 100, self.n_bins + 1)[1:-1]
                    edges = np.unique(np.percentile(col, q, method="midpoint"))
                self.edges.append(edges)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if not self.n_bins:
            return np.asarray(Z, dtype=np.float32)
        out = np.empty(Z.shape, dtype=np.float32)
        for j, edges in enumerate(self.edges):
            out[:, j] = np.searchsorted(edges, Z[:, j], side="left")
        return out

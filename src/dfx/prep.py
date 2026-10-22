"""Preprocessing fitted on training rows only (plan §4) and the DF21 percentile binner."""
from __future__ import annotations

import numpy as np


class Preprocessor:
    """Median / mode imputation, one-hot encoding of categorical columns, standardization of
    numeric ones. Fitted on outer-training rows only (plan §4); every method reuses the same fit.

    Unseen categories become all-zero dummies. Columns constant on the training rows are dropped
    (ours). `origin[j]` is the raw column behind encoded column j; `dummy[j]` marks one-hot
    columns, which stay 0/1.
    """

    def fit(self, X: np.ndarray, categorical: np.ndarray) -> "Preprocessor":
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
                filled = np.where(np.isnan(X[:, j]), np.median(seen), X[:, j])
                if filled.std() > 0:
                    self.parts.append((j, "num", np.median(seen), (filled.mean(), filled.std())))
        widths = [len(extra) if kind == "cat" else 1 for _, kind, _, extra in self.parts]
        self.origin = np.repeat([p[0] for p in self.parts], widths).astype(int)
        self.dummy = np.repeat([p[1] == "cat" for p in self.parts], widths).astype(bool)
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

    def fit(self, Z: np.ndarray) -> "Binner":
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

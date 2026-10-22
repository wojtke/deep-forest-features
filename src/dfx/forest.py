"""Random forests with DF21 semantics, grown one tree at a time (plan §3, Table 3).

DF21 v0.1.7 (deepforest/forest.py L117-129, tree/tree.py L245-247): each tree is fit on the
distinct rows of a bootstrap draw, unweighted; sqrt(m) features per split in classification and
regression; trees fully grown (min_samples_leaf 1); gini / squared error. RF trees use the best
split, ERF trees a random threshold (ExtraTrees style). Each tree predicts the requested rows and
is discarded, so memory never holds more than one tree.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def grow_forest(kind: str, task: str, n_trees: int, Z: np.ndarray, y: np.ndarray, evals: list,
                n_classes: int, seed: int) -> list[tuple]:
    """Fit a forest on (Z, y) and return, for each array in `evals`, (mean, s1, s2):
    mean is the forest prediction (n, C) (C = 1 in regression); in regression s1 and s2 are the
    per-row sum of tree predictions and of their squares (for the tree std of plan eq. 3)."""
    rng = np.random.default_rng(seed)
    classification = task == "classification"
    tree_class = DecisionTreeClassifier if classification else DecisionTreeRegressor
    splitter = {"rf": "best", "erf": "random"}[kind]
    s1 = [np.zeros((len(e), n_classes if classification else 1)) for e in evals]
    s2 = [np.zeros(len(e)) for e in evals]
    for _ in range(n_trees):
        rows = np.unique(rng.integers(0, len(Z), len(Z)))  # distinct rows of a bootstrap draw
        tree = tree_class(splitter=splitter, max_features="sqrt", random_state=int(rng.integers(2**31 - 1)))
        tree.fit(Z[rows], y[rows])
        for i, e in enumerate(evals):
            if classification:
                s1[i][:, tree.classes_] += tree.predict_proba(e)
            else:
                p = tree.predict(e)
                s1[i][:, 0] += p
                s2[i] += p**2
    if classification:
        return [(s / n_trees, None, None) for s in s1]
    return [(a / n_trees, a[:, 0], b) for a, b in zip(s1, s2)]

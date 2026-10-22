"""The cascade (plan §3): layers of 2 RF + 2 ERF, 3-fold out-of-fold (OOF) predictions with the
same folds at every layer, added columns chosen on each fold model's own training rows.

Transductive evaluation: the outer test rows pass through every layer as it is trained but are
never used to fit anything, so no model is kept and every layer's test prediction is returned
(for the common-depth comparison).
"""
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .data import inner_split
from .features import Outputs, components, k_products
from .forest import grow_forest
from .prep import Binner

# Seed streams. Seeds depend on (seed, dataset, outer fold, layer, inner fold, stream, index) but
# never on the method, so variants grow identical trees wherever their inputs coincide. MI is the
# stream of every component except random_xx; only XP-MI uses it.
TREES, RANDOM_XX, MI, INNER, LIGHTGBM, SINGLE_RF = range(6)


def seed_for(*keys: int) -> int:
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0])


@dataclass(frozen=True)
class CascadeConfig:
    """Cascade settings (plan Table 3); sources in docs/decisions.md."""
    n_forests: int = 2  # per type: DF21 n_estimators=2 -> 2 RF + 2 ERF
    n_trees: int = 100  # DF21
    n_bins: int = 255  # DF21; 0 = no binning (sensitivity check)
    oof_folds: int = 3  # Zhou & Feng (2019)
    max_layers: int = 20  # DF21
    patience: int = 2  # DF21 n_tolerant_rounds
    delta: float = 1e-5  # DF21
    k_mode: str = "base"  # K, ceil(K/2) or 2K (sensitivity check)
    activation: str = "sigmoid"  # monotone variant (plan §3)


@dataclass
class Layer:
    test_pred: np.ndarray  # (n_test, C): mean over forests and fold models
    oof_score: float | None  # depth score of the OOF predictions (None for baselines)
    cpu_seconds: float
    selections: list  # per fold model: what was added


@dataclass
class Result:
    layers: list[Layer]
    best: int  # 1-based index of the layer whose prediction is the output
    inner_folds: np.ndarray | None  # OOF fold of every training row
    info: dict


def depth_score(y: np.ndarray, pred: np.ndarray, task: str) -> float:
    """Balanced accuracy (plan §3; DF21 uses accuracy) or mean squared error (DF21)."""
    if task == "classification":
        return float(balanced_accuracy_score(y, pred.argmax(1)))
    return float(np.mean((y - pred[:, 0]) ** 2))


def improved(score: float, best: float | None, task: str, delta: float) -> bool:
    """DF21 cascade.py L742-747: a layer improves if it beats the best layer so far by delta."""
    if best is None:
        return True
    return score >= best + delta if task == "classification" else score <= best - delta


def depth_rule(scores: list[float], task: str, delta: float, patience: int) -> tuple[int, bool]:
    """(best layer so far, stop now?) under the DF21 rule (cascade.py L930-962): stop once
    `patience` layers in a row bring no improvement. Layers are 1-based."""
    best, best_score, wait = 0, None, 0
    for layer, score in enumerate(scores, 1):
        if improved(score, best_score, task, delta):
            best, best_score, wait = layer, score, 0
        else:
            wait += 1
    return best, wait >= patience


def new_columns(fitted: list, xs: np.ndarray, prev: Outputs | None) -> list[np.ndarray]:
    """Raw columns a layer adds for training rows: the previous OOF predictions P (from layer 2)
    and every fitted component's columns."""
    return ([] if prev is None else [prev.P]) + [c.transform(xs, prev) for c in fitted]


def new_columns_test(fitted: list, xs: np.ndarray, prev: list[Outputs]) -> list[np.ndarray]:
    """The same columns for test rows (plan §3): P is the mean of the previous fold models'
    predictions; each added column is the mean of its values computed from each fold model."""
    if not prev:
        return [c.transform(xs, None) for c in fitted]
    return [np.mean([o.P for o in prev], 0)] + [np.mean([c.transform(xs, o) for o in prev], 0) for c in fitted]


def fit_predict(xs_tr, y_tr, groups_tr, xs_te, *, method: str, task: str, n_classes: int, keys: tuple,
                cfg: CascadeConfig) -> Result:
    """Train the cascade on (xs_tr, y_tr) and predict xs_te at every layer.

    xs_*: preprocessed, unbinned features (products are computed from these). keys: (seed,
    dataset hash, outer fold) for the seed streams."""
    regression, n_out = task == "regression", n_classes if task == "classification" else 1
    fold_of, folds = inner_split(y_tr, groups_tr, task, cfg.oof_folds, seed_for(*keys, INNER))
    x_binner = Binner(cfg.n_bins).fit(xs_tr)  # X binned once, as DF21
    xb_tr, xb_te = x_binner.transform(xs_tr), x_binner.transform(xs_te)
    k = k_products(xs_tr.shape[1], cfg.k_mode)
    kinds = ["rf"] * cfg.n_forests + ["erf"] * cfg.n_forests

    def empty(n):  # every forest's outputs for n rows, filled below
        return Outputs(np.empty((n, len(kinds), n_out)), np.empty((n, len(kinds))) if regression else None)

    prev_tr: Outputs | None = None  # OOF outputs of the previous layer
    prev_te: list[Outputs] = []  # test outputs of each previous fold model
    layers, scores = [], []
    for layer in range(1, cfg.max_layers + 1):
        cpu = time.process_time()
        oof, outputs_te, selections = empty(len(y_tr)), [], []
        for f, (tr, va) in enumerate(folds):
            named = components(method, layer, k, cfg.activation)
            for name, comp in named:
                stream = RANDOM_XX if name == "random_xx" else MI
                comp.fit(xs_tr[tr], None if prev_tr is None else prev_tr.rows(tr), y_tr[tr], task=task,
                         n_classes=n_classes, seed=seed_for(*keys, layer, f, stream))
            selections.append({name: comp.info for name, comp in named})
            fitted = [comp for _, comp in named]
            new_tr, new_te = new_columns(fitted, xs_tr, prev_tr), new_columns_test(fitted, xs_te, prev_te)
            z_tr, z_te = xb_tr, xb_te
            if new_tr:
                a_tr, a_te = np.hstack(new_tr), np.hstack(new_te)
                binner = Binner(cfg.n_bins).fit(a_tr[tr])  # new columns: this fold model's rows
                z_tr = np.hstack([xb_tr, binner.transform(a_tr)])
                z_te = np.hstack([xb_te, binner.transform(a_te)])
            test = empty(len(xs_te))
            for j, kind in enumerate(kinds):
                (va_mean, va_sq), (te_mean, te_sq) = grow_forest(
                    kind, task, cfg.n_trees, z_tr[tr], y_tr[tr], [z_tr[va], z_te], n_classes,
                    seed_for(*keys, layer, f, TREES, j))
                oof.forests[va, j], test.forests[:, j] = va_mean, te_mean
                if regression:
                    oof.sq[va, j], test.sq[:, j] = va_sq, te_sq
            outputs_te.append(test)
        prev_tr, prev_te = oof, outputs_te
        scores.append(depth_score(y_tr, prev_tr.mean(), task))
        test_pred = np.mean([o.mean() for o in outputs_te], 0)
        layers.append(Layer(test_pred, scores[-1], time.process_time() - cpu, selections))
        best, stop = depth_rule(scores, task, cfg.delta, cfg.patience)
        if stop:
            break
    return Result(layers, best, fold_of, {"K": k})

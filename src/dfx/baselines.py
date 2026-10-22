"""Reference methods of plan Table 1: one random forest, a random forest with target-chosen XX
products (products without a cascade), and tuned LightGBM."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .cascade import INNER, LIGHTGBM, SINGLE_RF, CascadeConfig, Layer, Result, depth_score, seed_for
from .data import inner_folds
from .features import Products, k_products
from .forest import grow_forest
from .prep import Binner

RF_TREES = 400  # plan §5: as many trees as the four forests of one layer


@dataclass(frozen=True)
class LGBMConfig:
    """LightGBM tuning (plan §5): random settings scored on grouped inner folds."""
    n_settings: int = 10  # plan
    inner_folds: int = 3  # plan
    max_trees: int = 1000  # plan
    patience: int = 50  # ours, not sourced (LightGBM has no default)
    learning_rate: tuple = (0.01, 0.3)  # plan; log-uniform (ours)
    max_depth: tuple = (3, 8)  # plan; integer uniform (ours)
    subsample: tuple = (0.6, 1.0)  # plan; uniform (ours)
    colsample_bytree: tuple = (0.6, 1.0)  # plan; uniform (ours)
    reg_lambda: tuple = (0.1, 10.0)  # plan; log-uniform (ours)


def _forest(xs_tr, y_tr, xs_te, extra_tr, extra_te, *, task, n_classes, cfg, seed):
    """One RF on binned [X | extra]; binners fitted on all outer-training rows (like layer 1)."""
    x_binner = Binner(cfg.n_bins).fit(xs_tr)
    z_tr, z_te = x_binner.transform(xs_tr), x_binner.transform(xs_te)
    if extra_tr is not None and extra_tr.shape[1]:
        b = Binner(cfg.n_bins).fit(extra_tr)
        z_tr, z_te = np.hstack([z_tr, b.transform(extra_tr)]), np.hstack([z_te, b.transform(extra_te)])
    (pred, _, _), = grow_forest("rf", task, RF_TREES, z_tr, y_tr, [z_te], n_classes, seed)
    return pred


def _proba(model, x, n_classes):
    out = np.zeros((len(x), n_classes))
    out[:, model.classes_] = model.predict_proba(x)
    return out


def _lightgbm(xs_tr, y_tr, groups_tr, xs_te, *, task, n_classes, keys, lgbm: LGBMConfig):
    """Pick the setting with the best mean inner-fold primary metric; early stopping on
    LightGBM's default metric (log-loss / L2); refit with the rounded mean best iteration."""
    import lightgbm as lgb

    classification = task == "classification"
    model_class = lgb.LGBMClassifier if classification else lgb.LGBMRegressor
    rng = np.random.default_rng(seed_for(*keys, LIGHTGBM))
    log_uniform = (lambda r: float(np.exp(rng.uniform(np.log(r[0]), np.log(r[1])))))
    settings = [{"learning_rate": log_uniform(lgbm.learning_rate),
                 "max_depth": int(rng.integers(lgbm.max_depth[0], lgbm.max_depth[1] + 1)),
                 "subsample": float(rng.uniform(*lgbm.subsample)),
                 "colsample_bytree": float(rng.uniform(*lgbm.colsample_bytree)),
                 "reg_lambda": log_uniform(lgbm.reg_lambda)} for _ in range(lgbm.n_settings)]
    fixed = {"subsample_freq": 1, "n_jobs": 1, "deterministic": True, "force_row_wise": True, "verbose": -1,
             "random_state": seed_for(*keys, LIGHTGBM, 1)}
    folds = inner_folds(y_tr, groups_tr, task, lgbm.inner_folds, seed_for(*keys, INNER))
    tried = []
    for setting in settings:
        scores, iters = [], []
        for tr, va in folds:
            model = model_class(n_estimators=lgbm.max_trees, **setting, **fixed)
            model.fit(xs_tr[tr], y_tr[tr], eval_X=(xs_tr[va],), eval_y=(y_tr[va],),
                      callbacks=[lgb.early_stopping(lgbm.patience, verbose=False)])
            pred = _proba(model, xs_tr[va], n_classes) if classification else model.predict(xs_tr[va])[:, None]
            score = depth_score(y_tr[va], pred, task)
            scores.append(score if classification else float(np.sqrt(score)))  # balanced accuracy / RMSE
            iters.append(max(1, int(model.best_iteration_ or lgbm.max_trees)))
        tried.append({"setting": setting, "score": float(np.mean(scores)), "trees": int(round(np.mean(iters)))})
    best = (max if classification else min)(tried, key=lambda t: t["score"])
    model = model_class(n_estimators=best["trees"], **best["setting"], **fixed).fit(xs_tr, y_tr)
    pred = _proba(model, xs_te, n_classes) if classification else model.predict(xs_te)[:, None]
    fold_of = np.empty(len(y_tr), dtype=np.int8)
    for f, (_, va) in enumerate(folds):
        fold_of[va] = f
    return pred, {"chosen": best, "tried": tried}, fold_of


def run_baseline(method: str, xs_tr, y_tr, groups_tr, xs_te, *, task, n_classes, origin, dummy, keys,
                 cfg: CascadeConfig, lgbm: LGBMConfig) -> Result:
    cpu = time.process_time()
    common = dict(task=task, n_classes=n_classes, cfg=cfg, seed=seed_for(*keys, SINGLE_RF))
    info, selections, fold_of = {}, [], None
    if method == "rf":
        pred = _forest(xs_tr, y_tr, xs_te, None, None, **common)
    elif method == "rf_target_xx":
        k = k_products(xs_tr.shape[1], cfg.k_mode)
        products = Products("xx", "target", k).fit(xs_tr, None, y_tr, task=task, n_classes=n_classes,
                                                   origin=origin, dummy=dummy, seed=0)
        pred = _forest(xs_tr, y_tr, xs_te, products.transform(xs_tr, None), products.transform(xs_te, None), **common)
        selections, info = [{"xx_target": products.info}], {"K": k}
    elif method == "lightgbm":
        pred, info, fold_of = _lightgbm(xs_tr, y_tr, groups_tr, xs_te, task=task, n_classes=n_classes, keys=keys,
                                        lgbm=lgbm)
    else:
        raise KeyError(method)
    return Result([Layer(pred, None, time.process_time() - cpu, selections)], 1, fold_of, info)

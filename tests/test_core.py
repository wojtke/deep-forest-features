"""Fast checks of the pieces the results depend on: splits, preprocessing, forests, added columns,
the cascade's fold and test-row rules, the depth rule and the statistics."""
import numpy as np
import pandas as pd
import pytest

from dfx import forest as forest_module
from dfx.cascade import CascadeConfig, depth_rule, fit_predict, new_columns_test
from dfx.data import fold_ids, inner_split, related_groups, subsample
from dfx.features import Monotone, NonLinear, Outputs, Products, k_products
from dfx.forest import grow_forest
from dfx.prep import Binner, Preprocessor
from dfx.report import holm, paired, relative_reduction

TINY = CascadeConfig(n_trees=4, max_layers=3)


def problem(task="classification", n=240, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] > 0).astype(int) if task == "classification" else X[:, 0] + 2 * X[:, 1] * X[:, 2]
    return X, y


def cascade(method, X, y, task="classification", X_te=None, cfg=TINY):
    prep = Preprocessor().fit(X, np.zeros(X.shape[1], bool))
    xs, xs_te = prep.transform(X), prep.transform(X if X_te is None else X_te)
    return fit_predict(xs, y, np.arange(len(y)), xs_te, method=method, task=task,
                       n_classes=2 if task == "classification" else 1, keys=(0, 1, 0), cfg=cfg)


def outputs(probs_rf, probs_erf):
    return Outputs(np.stack([probs_rf, probs_rf, probs_erf, probs_erf], 1), n_rf=2)


# --- data ---------------------------------------------------------------------------------------

def test_groups_join_duplicates_and_keys_but_not_missing_keys():
    X = np.array([[1.0, np.nan], [1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    g = related_groups(X, key=np.array(["a", "b", "b", "c", None, None], dtype=object))
    assert g[0] == g[1] == g[2] != g[3]  # rows 0-1 identical, rows 1-2 share a key
    assert len({g[3], g[4], g[5]}) == 3  # missing keys do not join rows


def test_folds_keep_groups_together_and_are_deterministic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    y = rng.integers(0, 2, 180)
    for task in ("classification", "regression"):
        ids = fold_ids(y, groups, task, 5, seed=1)
        assert all(len(set(ids[groups == g])) == 1 for g in np.unique(groups))
        np.testing.assert_array_equal(ids, fold_ids(y, groups, task, 5, seed=1))
        for tr, va in inner_split(y, groups, task, 3, seed=2)[1]:
            assert not set(groups[tr]) & set(groups[va])


def test_subsample_keeps_whole_groups_and_proportions():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(3000), 10)
    y = np.repeat(rng.random(3000) < 0.2, 10).astype(int)
    keep = subsample(y, groups, "classification", 20_000, seed=0)
    assert len(keep) <= 20_000
    assert np.all(np.bincount(groups[keep])[np.unique(groups[keep])] == 10)
    assert abs(y[keep].mean() - y.mean()) < 0.01


# --- preprocessing and binning -------------------------------------------------------------------

def test_preprocessing_uses_training_rows_only():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, np.nan], [1.0, 4.0]])
    prep = Preprocessor().fit(X, np.array([True, False]))
    assert prep.width == 3  # two dummies, one numeric column
    out = prep.transform(np.array([[7.0, 100.0]]))  # unseen category, extreme value
    assert out[0, :2].tolist() == [0.0, 0.0]
    assert np.isclose(prep.transform(X)[:, 2].mean(), 0)  # standardized on training rows


def test_binner_is_exact_for_few_values_and_has_255_bins():
    few = np.array([[3.0], [1.0], [2.0], [2.0]])
    np.testing.assert_array_equal(Binner().fit(few).transform(few)[:, 0], [2, 0, 1, 1])
    many = np.random.default_rng(0).normal(size=(5000, 1))
    assert len(np.unique(Binner().fit(many).transform(many))) == 255
    np.testing.assert_array_equal(Binner(0).fit(many).transform(many), many.astype(np.float32))


def test_monotone_columns_bin_like_raw_predictions():
    p = np.random.default_rng(0).random((400, 4))
    sig = 1 / (1 + np.exp(-p))
    np.testing.assert_array_equal(Binner().fit(p).transform(p), Binner().fit(sig).transform(sig))


# --- forests ------------------------------------------------------------------------------------

def test_trees_see_distinct_bootstrap_rows_and_sqrt_features(monkeypatch):
    seen = []

    class Recording(forest_module.DecisionTreeClassifier):
        def fit(self, X, y, **kw):
            seen.append((len(X), len(np.unique(X, axis=0)), self.max_features, self.splitter))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(forest_module, "DecisionTreeClassifier", Recording)
    X, y = problem()
    [(pred, _)] = grow_forest("erf", "classification", 5, X, y, [X], 3, seed=0)  # 3 classes, one unseen
    assert all(rows == distinct < len(X) for rows, distinct, *_ in seen)
    assert all(mf == "sqrt" and sp == "random" for *_, mf, sp in seen)
    assert pred.shape == (len(X), 3) and np.allclose(pred[:, 2], 0) and np.allclose(pred.sum(1), 1)


# --- added columns ------------------------------------------------------------------------------

def test_k_rule():
    assert [k_products(d) for d in (2, 4, 5, 8, 60)] == [0, 1, 2, 7, 60]
    assert k_products(8, "half") == 4 and k_products(7, "half") == 3 and k_products(8, "double") == 14


def test_products_find_a_pure_interaction_and_drop_constant_candidates():
    X, _ = problem("regression", n=600, d=6)
    y = np.sign(X[:, 1] * X[:, 3])
    for selector in ("target", "residual"):
        p = Products("xx", selector, 1).fit(X, None, y, task="regression", n_classes=1, seed=0)
        assert p.info["pairs"] == [[1, 3]]
    D = np.column_stack([X[:, :2], (X[:, 2] > 0), (X[:, 2] <= 0)]).astype(float)  # two dummies of one feature
    p = Products("xx", "random", 50).fit(D, None, y, task="regression", n_classes=1, seed=0)
    assert [2, 3] not in p.info["pairs"] and len(p.info["pairs"]) == 5  # their product is constant 0


def test_residual_selection_uses_the_layer_error():
    X, _ = problem("regression", n=600, d=4)
    first, second = X[:, 0] * X[:, 1], X[:, 2] * X[:, 3]
    y = first + 0.5 * second
    prev = Outputs(np.repeat(first[:, None, None], 4, axis=1), n_rf=2)  # the layer already predicts `first`
    pick = {s: Products("xx", s, 1).fit(X, prev, y, task="regression", n_classes=1, seed=0).info["pairs"]
            for s in ("target", "residual")}
    assert pick == {"target": [[0, 1]], "residual": [[2, 3]]}


def test_entropy_and_disagreement_by_hand():
    out = outputs(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]))
    h_rf, h_erf, d = NonLinear().fit(None, out, None).transform(None, out)[0]
    assert np.isclose(h_rf, np.log(2)) and h_erf == 0 and np.isclose(d, 1.0)


def regression_outputs(rf_mean, rf_sq):
    """One row; both RF forests predict `rf_mean` with mean squared tree prediction `rf_sq`."""
    forests = np.array([[[rf_mean], [rf_mean], [0.0], [0.0]]])
    return Outputs(forests, 2, np.array([[rf_sq, rf_sq, 0.0, 0.0]]))


def test_regression_uncertainty_is_the_pooled_tree_std():
    assert np.isclose(regression_outputs(3.0, 10.0).tree_std()[0, 0], 1.0)  # trees 2 and 4: mean 3, sd 1


def test_test_rows_average_columns_not_predictions():
    """Plan §3: each added column is the mean of its values computed from each fold model."""
    first, second = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    a, b = outputs(first, first), outputs(second, second)
    p, cols = new_columns_test([NonLinear()], np.zeros((1, 0)), [a, b])
    np.testing.assert_allclose(p, [[0.5, 0.5] * 4])  # P is averaged
    np.testing.assert_allclose(cols, [[0.0, 0.0, 0.0]])  # entropy of each is 0; of the mean, log 2
    c, d = regression_outputs(0.0, 1.0), regression_outputs(10.0, 101.0)  # each has tree std 1
    _, cols = new_columns_test([NonLinear()], np.zeros((1, 0)), [c, d])
    assert np.isclose(cols[0, 0], 1.0)  # not the std of the pooled trees (about 5.1)


def test_fitted_scalers_are_applied_unchanged_to_test_rows():
    train, test = regression_outputs(0.0, 1.0), regression_outputs(10.0, 101.0)
    train = Outputs(np.concatenate([train.forests, test.forests * 0.5]), 2, np.concatenate([train.sq, test.sq]))
    mono = Monotone("relu").fit(None, train, None, task="regression")  # min 0, max 5 per column
    np.testing.assert_allclose(mono.transform(None, test)[0, :2], [2.0, 2.0])  # (10 - 0) / 5, not refitted
    xs = np.ones((2, 1))
    xp = Products("xp", "random", 1).fit(xs, train, np.array([0.0, 1.0]), task="regression", n_classes=1, seed=0)
    assert np.isclose(xp.transform(np.ones((1, 1)), test)[0, 0], 3.0)  # (10 - 2.5) / 2.5: training-row scale


# --- the cascade --------------------------------------------------------------------------------

def test_variants_match_the_base_cascade_where_inputs_coincide():
    X, y = problem()
    layer1 = {m: cascade(m, X, y).layers[0] for m in ("base", "nonlinear", "monotone", "xp_mi", "target_xx",
                                                       "residual_xx")}
    for m in ("nonlinear", "monotone", "xp_mi"):
        np.testing.assert_array_equal(layer1[m].test_pred, layer1["base"].test_pred)
    np.testing.assert_array_equal(layer1["residual_xx"].test_pred, layer1["target_xx"].test_pred)
    same = [list(sel.values()) for sel in layer1["residual_xx"].selections]
    assert same == [list(sel.values()) for sel in layer1["target_xx"].selections]


@pytest.mark.parametrize("method", ["residual_xx", "xp_mi", "combined"])
def test_test_rows_never_change_training(method):
    X, y = problem()
    rng = np.random.default_rng(5)
    a = cascade(method, X, y, X_te=rng.normal(size=(30, 5)))
    b = cascade(method, X, y, X_te=rng.normal(size=(50, 5)) * 10)
    assert [la.oof_score for la in a.layers] == [lb.oof_score for lb in b.layers]
    assert [la.selections for la in a.layers] == [lb.selections for lb in b.layers] and a.best == b.best


def test_selection_sees_only_the_fold_models_training_rows():
    X, y = problem("regression")
    cfg = CascadeConfig(n_trees=2, max_layers=1)
    a = cascade("target_xx", X, y, "regression", cfg=cfg)
    va = np.flatnonzero(a.inner_folds == 0)
    y2 = y.copy()
    y2[va] = np.random.default_rng(1).permutation(y2[va]) * 5  # labels the fold-0 model never trains on
    b = cascade("target_xx", X, y2, "regression", cfg=cfg)
    np.testing.assert_array_equal(a.inner_folds, b.inner_folds)
    assert a.layers[0].selections[0] == b.layers[0].selections[0]


def test_new_columns_are_binned_on_the_fold_models_training_rows(monkeypatch):
    import dfx.cascade

    fitted_rows = []

    class Recording(Binner):
        def fit(self, Z):
            fitted_rows.append(len(Z))
            return super().fit(Z)

    monkeypatch.setattr(dfx.cascade, "Binner", Recording)
    X, y = problem()
    result = cascade("residual_xx", X, y, cfg=CascadeConfig(n_trees=2, max_layers=2, patience=5))
    train_sizes = [int((result.inner_folds != f).sum()) for f in range(3)]
    assert fitted_rows == [len(y)] + train_sizes * 2  # X once on all rows; then per layer, per fold model


def test_depth_rule_follows_df21():
    assert depth_rule([0.80, 0.800005], "classification", 1e-5, 2) == (1, False)
    assert depth_rule([0.80, 0.800005, 0.79], "classification", 1e-5, 2) == (1, True)
    assert depth_rule([0.70, 0.75, 0.74, 0.76], "classification", 1e-5, 2) == (4, False)
    assert depth_rule([5.0, 4.0, 4.5, 4.2], "regression", 1e-5, 2) == (2, True)


# --- statistics ---------------------------------------------------------------------------------

def test_relative_reduction_and_holm():
    np.testing.assert_allclose(relative_reduction([0.2, 0.0, 0.1], [0.1, 0.0, 0.2]), [0.5, 0.0, -0.5])
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_interim_report_pairs_runs():
    runs = pd.DataFrame({"check": "default", "dataset": "a", "method": ["base", "base", "rf"], "seed": 0,
                         "fold": [0, 1, 0]})
    assert paired(runs).fold.tolist() == [0, 0]

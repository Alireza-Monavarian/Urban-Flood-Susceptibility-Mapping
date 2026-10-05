"""Tests for pipeline.model_compare -- the 6-model comparison
(MaxEnt reference + 5 presence-background ML comparators) on an IDENTICAL,
matched test set, with NO test-set early-stopping for the boosting
comparators.

Split into two tiers, mirroring test_model_families.py / test_evaluate_oos.py's
own convention:
  1. Pure-logic / synthetic-data tests (fast, no dependency on the real,
     gitignored predictor stack or MaxEnt artifacts) -- ``_fit_configs()``'s
     static registry (the key regression test), ``_inner_split``'s
     stratification contract, direct unit tests of the three boosting
     ``_fit_*`` helpers confirming their ``eval_set`` argument is really
     the inner-validation fold (never a "test" stand-in), and
     ``build_background``/``_presence_features`` against tiny synthetic
     on-disk rasters (real GeoTIFFs written to ``tmp_path``, not the real
     predictor stack).
  2. Real-fixture tests against the actual on-disk canonical stack +
     ``maxent.fit_eval()``'s ``holdout_model/flood.tif`` -- skipped (not hard-failed)
     if either is not present locally.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from rasterio.transform import from_origin, rowcol, xy

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from pipeline import config, evaluate, maxent, model_compare


# ============================================================================
# 1. _fit_configs() -- the key regression test.
# ============================================================================

def test_no_model_early_stops_on_test():
    cfg = model_compare._fit_configs()
    assert all(v != "test" for v in cfg.values())


def test_fit_configs_lists_all_six_models():
    cfg = model_compare._fit_configs()
    assert set(cfg.keys()) == {
        "MaxEnt", "LogisticRegression", "RandomForest",
        "XGBoost", "LightGBM", "CatBoost",
    }


def test_fit_configs_boosting_models_use_train_inner_val():
    cfg = model_compare._fit_configs()
    for name in ("XGBoost", "LightGBM", "CatBoost"):
        assert cfg[name] == "train_inner_val"


# ============================================================================
# 2. _inner_split -- carves the boosting validation fold OUT OF TRAIN ONLY.
# ============================================================================

def test_inner_split_covers_all_train_rows_disjointly():
    X = pd.DataFrame({"f": np.arange(100)})
    y = np.tile([0, 1], 50)
    X_tr, X_val, y_tr, y_val = model_compare._inner_split(X, y)
    assert len(X_tr) + len(X_val) == len(X)
    assert set(X_tr.index).isdisjoint(set(X_val.index))
    assert sorted(X_tr.index.tolist() + X_val.index.tolist()) == list(range(100))


def test_inner_split_is_stratified():
    X = pd.DataFrame({"f": np.arange(200)})
    y = np.array([0] * 180 + [1] * 20)   # ~1:9 imbalance, echoes the real ~1:35
    _X_tr, _X_val, y_tr, y_val = model_compare._inner_split(X, y)
    # Stratified split preserves the minority-class fraction in both folds.
    assert y_tr.mean() == pytest.approx(y.mean(), abs=0.02)
    assert y_val.mean() == pytest.approx(y.mean(), abs=0.05)


def test_inner_split_deterministic_for_fixed_seed():
    X = pd.DataFrame({"f": np.arange(100)})
    y = np.tile([0, 1], 50)
    a = model_compare._inner_split(X, y, seed=config.GLOBAL_SEED)
    b = model_compare._inner_split(X, y, seed=config.GLOBAL_SEED)
    assert list(a[0].index) == list(b[0].index)


# ============================================================================
# 3. THE CRITICAL FIX -- direct unit tests of the boosting _fit_* helpers:
#    each one's `eval_set` must be exactly the (X_val, y_val) fold passed
#    in, never silently substituted, and must never overlap the (disjoint,
#    shifted) synthetic "would-be-test" data these tests construct but
#    never pass to the function at all (there is no `X_test` parameter on
#    any of these functions -- structurally impossible for them to see it).
# ============================================================================

def _tiny_train_val(n_tr=60, n_val=16, seed=0):
    rng = np.random.default_rng(seed)
    X_tr = pd.DataFrame({
        "a": rng.normal(size=n_tr),
        "b": rng.normal(size=n_tr),
        "hydrologic_soil_group": np.tile([1, 2, 3, 4], n_tr // 4),
    })
    y_tr = np.tile([0, 1], n_tr // 2)
    X_val = pd.DataFrame({
        "a": rng.normal(size=n_val) + 5.0,   # shifted -- never a row of X_tr
        "b": rng.normal(size=n_val) + 5.0,
        "hydrologic_soil_group": np.tile([1, 2, 3, 4], n_val // 4),
    })
    y_val = np.tile([0, 1], n_val // 2)
    return X_tr, y_tr, X_val, y_val


def test_xgboost_eval_set_is_the_inner_validation_fold(monkeypatch):
    X_tr, y_tr, X_val, y_val = _tiny_train_val()
    captured = {}
    real_fit = xgb.XGBClassifier.fit

    def spy_fit(self, X, y, **kwargs):
        captured["eval_set"] = kwargs.get("eval_set")
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", spy_fit)
    model_compare._fit_xgboost(X_tr, y_tr, X_val, y_val)

    assert captured["eval_set"] is not None
    seen_X, seen_y = captured["eval_set"][0]
    assert seen_X.equals(X_val)
    assert np.array_equal(np.asarray(seen_y), y_val)
    assert not seen_X["a"].isin(X_tr["a"]).any(), "eval_set leaked a train row"


def test_lightgbm_eval_set_is_the_inner_validation_fold(monkeypatch):
    X_tr, y_tr, X_val, y_val = _tiny_train_val()
    captured = {}
    real_fit = lgb.LGBMClassifier.fit

    def spy_fit(self, X, y, **kwargs):
        captured["eval_set"] = kwargs.get("eval_set")
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy_fit)
    model_compare._fit_lightgbm(X_tr, y_tr, X_val, y_val)

    assert captured["eval_set"] is not None
    seen_X, seen_y = captured["eval_set"][0]
    assert seen_X.equals(X_val)
    assert np.array_equal(np.asarray(seen_y), y_val)
    assert not seen_X["a"].isin(X_tr["a"]).any(), "eval_set leaked a train row"


def test_catboost_eval_set_is_the_inner_validation_fold(monkeypatch):
    X_tr, y_tr, X_val, y_val = _tiny_train_val()
    captured = {}
    real_fit = CatBoostClassifier.fit

    def spy_fit(self, X, y=None, **kwargs):
        captured["eval_set"] = kwargs.get("eval_set")
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(CatBoostClassifier, "fit", spy_fit)
    model_compare._fit_catboost(X_tr, y_tr, X_val, y_val)

    assert captured["eval_set"] is not None
    seen_X, seen_y = captured["eval_set"]      # CatBoost's own convention: one tuple, not a list
    assert seen_X.equals(X_val)
    assert np.array_equal(np.asarray(seen_y), y_val)
    assert not seen_X["a"].isin(X_tr["a"]).any(), "eval_set leaked a train row"


# ============================================================================
# 4. build_background / _presence_features -- tiny synthetic on-disk
#    rasters (real GeoTIFFs under tmp_path), never the real predictor stack.
# ============================================================================

FAKE_LAYERS = ["dem", "slope", "hydrologic_soil_group", "nlcd_landcover"]


def _write_fake_layer(pred_dir, name, values, cell=10.0, crs="EPSG:26914"):
    values = np.asarray(values, dtype="float64")
    da = xr.DataArray(values, dims=("y", "x"))
    transform = from_origin(0, values.shape[0] * cell, cell, cell)
    da = da.rio.write_transform(transform)
    da = da.rio.write_crs(crs)
    da.rio.write_nodata(np.nan, inplace=True)
    da.rio.to_raster(Path(pred_dir) / f"{name}.tif")
    return transform


@pytest.fixture
def fake_stack(tmp_path):
    n = 10
    grids = {
        "dem": np.arange(n * n, dtype=float).reshape(n, n),
        "slope": np.arange(n * n, dtype=float).reshape(n, n) * 0.1,
        "hydrologic_soil_group": np.full((n, n), 2.0),
        "nlcd_landcover": np.full((n, n), 41.0),
    }
    transform = None
    for name, arr in grids.items():
        transform = _write_fake_layer(tmp_path, name, arr)
    return tmp_path, transform, grids


def test_build_background_excludes_presence_coincident_pixels(fake_stack):
    tmp_path, transform, _grids = fake_stack
    pres_rc = [(2, 3), (5, 5), (0, 0)]
    xs, ys = xy(transform, [rc[0] for rc in pres_rc], [rc[1] for rc in pres_rc])
    presence = pd.DataFrame({"x": xs, "y": ys})

    bg = model_compare.build_background(n=20, seed=0, pred_dir=tmp_path,
                                         layers=FAKE_LAYERS, presence=presence)

    assert len(bg) == 20
    assert bg.attrs["n_presence_excluded"] == 3

    bg_rows, bg_cols = rowcol(transform, bg["x"].to_numpy(), bg["y"].to_numpy())
    bg_rc = set(zip(np.asarray(bg_rows).tolist(), np.asarray(bg_cols).tolist()))
    assert bg_rc.isdisjoint(set(pres_rc)), "background must never draw a presence-coincident pixel"


def test_build_background_categorical_columns_cast_to_int(fake_stack):
    tmp_path, transform, _grids = fake_stack
    xs, ys = xy(transform, [0], [0])
    presence = pd.DataFrame({"x": xs, "y": ys})

    bg = model_compare.build_background(n=10, seed=1, pred_dir=tmp_path,
                                         layers=FAKE_LAYERS, presence=presence)
    for c in model_compare.CATEGORICAL:
        assert np.issubdtype(bg[c].dtype, np.integer), f"{c} must be cast to int"


def test_build_background_deterministic_seed(fake_stack):
    tmp_path, transform, _grids = fake_stack
    xs, ys = xy(transform, [0], [0])
    presence = pd.DataFrame({"x": xs, "y": ys})

    bg1 = model_compare.build_background(n=15, seed=7, pred_dir=tmp_path,
                                          layers=FAKE_LAYERS, presence=presence)
    bg2 = model_compare.build_background(n=15, seed=7, pred_dir=tmp_path,
                                          layers=FAKE_LAYERS, presence=presence)
    assert list(bg1["x"]) == list(bg2["x"])
    assert list(bg1["y"]) == list(bg2["y"])


def test_build_background_different_seed_differs(fake_stack):
    tmp_path, transform, _grids = fake_stack
    xs, ys = xy(transform, [0], [0])
    presence = pd.DataFrame({"x": xs, "y": ys})

    bg1 = model_compare.build_background(n=15, seed=1, pred_dir=tmp_path,
                                          layers=FAKE_LAYERS, presence=presence)
    bg2 = model_compare.build_background(n=15, seed=2, pred_dir=tmp_path,
                                          layers=FAKE_LAYERS, presence=presence)
    assert list(bg1["x"]) != list(bg2["x"])


def test_presence_features_matches_hand_computed_values(fake_stack):
    tmp_path, transform, grids = fake_stack
    row, col = 4, 6
    x, y = xy(transform, row, col)
    canonical = pd.DataFrame({"x": [x], "y": [y]})

    feats = model_compare._presence_features(canonical, layers=FAKE_LAYERS, pred_dir=tmp_path)

    assert feats["dem"].iloc[0] == pytest.approx(grids["dem"][row, col])
    assert feats["slope"].iloc[0] == pytest.approx(grids["slope"][row, col])
    assert feats["hydrologic_soil_group"].iloc[0] == int(grids["hydrologic_soil_group"][row, col])
    assert np.issubdtype(feats["hydrologic_soil_group"].dtype, np.integer)


def test_presence_features_raises_on_off_grid_nan(fake_stack):
    tmp_path, transform, _grids = fake_stack
    # A point far outside the 10x10 grid's extent samples NaN on every layer.
    canonical = pd.DataFrame({"x": [-99999.0], "y": [99999.0]})
    with pytest.raises(AssertionError):
        model_compare._presence_features(canonical, layers=FAKE_LAYERS, pred_dir=tmp_path)


# ============================================================================
# 5. run() -- real-fixture tests against the actual canonical stack +
#    maxent.fit_eval()'s holdout_model/flood.tif. Skipped if either is missing.
# ============================================================================

@pytest.fixture(scope="module")
def real_comparison():
    if not evaluate.EVAL_RASTER.exists():
        pytest.skip(f"{evaluate.EVAL_RASTER} not built locally (run maxent.fit_eval() first)")
    if not model_compare.PRED_DIR.exists() or not any(model_compare.PRED_DIR.glob("*.tif")):
        pytest.skip(f"{model_compare.PRED_DIR} not populated locally")

    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    background = model_compare.build_background()
    results = model_compare.run(train_idx, test_idx, background)
    return dict(results=results, train_idx=train_idx, test_idx=test_idx,
                samples=samples, background=background)


EXPECTED_MODELS = {
    "MaxEnt (reference)", "Logistic Regression", "Random Forest",
    "XGBoost", "LightGBM", "CatBoost",
}


def test_run_returns_six_models_on_identical_test_set(real_comparison):
    results = real_comparison["results"]
    assert len(results) == 6
    assert set(results["model"]) == EXPECTED_MODELS
    assert results["auc"].notna().all()
    assert results["auc"].between(0, 1).all()
    # Every row's threshold-metrics block was computed over the SAME-sized
    # test set (83 test presence + 30% of the 10,000-pixel background pool).
    n_test_presence = len(real_comparison["test_idx"])
    n_test_bg = round(model_compare.DEFAULT_BACKGROUND_N * 0.30)
    expected_n = n_test_presence + n_test_bg
    assert (results["tp"] + results["tn"] + results["fp"] + results["fn"] == expected_n).all()


def test_run_maxent_train_idx_matches_recorded_provenance(real_comparison):
    """Cross-checks that the MaxEnt row really was scored against Task
    6.2's own recorded train_idx (proves the invariant-2 assert inside
    _score_maxent fired against the REAL provenance, not a stale/assumed
    train_idx)."""
    recorded = np.loadtxt(maxent.HOLDOUT_MODEL_DIR / "train_idx.csv", skiprows=1, dtype=int)
    assert sorted(recorded.tolist()) == sorted(real_comparison["train_idx"].tolist())


def test_run_background_excludes_all_canonical_presence_pixels(real_comparison):
    """None of the 277 canonical presence points' pixels should be
    drawable as background."""
    assert real_comparison["background"].attrs["n_presence_excluded"] >= 1


def test_run_pinned_auc_ranking(real_comparison):
    """Pins the ACTUAL AUC observed for all 6 models on the real,
    matched, honest (no test-set early-stopping) test set.
    Values are read directly off a real run (data/processed/
    model_comparison/metrics_comparison.csv), never forced to any
    particular order: MaxEnt lands 5th of 6 here, ahead of only Logistic
    Regression -- NOT the winner, and NOT forced to be."""
    auc = real_comparison["results"].set_index("model")["auc"]
    assert auc["MaxEnt (reference)"] == pytest.approx(0.984269, abs=5e-4)
    assert auc["Logistic Regression"] == pytest.approx(0.980486, abs=5e-4)
    assert auc["Random Forest"] == pytest.approx(0.993096, abs=5e-4)
    # re-pinned 2026-08-12 to a clean rebuild (xgboost 3.3.0, env/RUNTIME.md)
    assert auc["XGBoost"] == pytest.approx(0.992703, abs=5e-4)
    assert auc["LightGBM"] == pytest.approx(0.987679, abs=5e-4)
    assert auc["CatBoost"] == pytest.approx(0.990839, abs=5e-4)

    # The actual, un-forced ranking (highest AUC first) -- 4 tree-ensemble
    # comparators (XGBoost/RF/CatBoost/LightGBM) all outrank MaxEnt on
    # this honest, matched, no-test-leakage evaluation; MaxEnt itself
    # outranks only the linear Logistic Regression baseline.
    ranking = auc.sort_values(ascending=False).index.tolist()
    # Re-pinned 2026-08-12 after a clean rebuild. THE RANKING FLIPPED:
    # under xgboost 3.3.0 Random Forest (0.9931) now edges XGBoost (0.9927),
    # where XGBoost previously led at 0.9937.
    assert ranking == [
        "Random Forest",
        "XGBoost",
        "CatBoost",
        "LightGBM",
        "MaxEnt (reference)",
        "Logistic Regression",
    ]


# ============================================================================
# 6. _fit_all_models -- the shared fitting helper run() and
#    susceptibility_rasters() both call. Fast synthetic-data unit tests
#    (no real predictor stack needed).
# ============================================================================

_TINY_LAYERS = ["a", "b", "hydrologic_soil_group", "nlcd_landcover"]


def _tiny_synthetic_train(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        # _fit_logistic_regression's ColumnTransformer hardcodes BOTH
        # CATEGORICAL columns (never filtered by `layers`, unlike
        # _fit_catboost's own cat_features intersection) -- both must be
        # present or its "cat" transformer raises a column-not-found error.
        "hydrologic_soil_group": rng.integers(1, 5, size=n),
        "nlcd_landcover": rng.integers(1, 5, size=n),
    })
    y = rng.integers(0, 2, size=n)
    # Guarantee both classes are present (stratified _inner_split needs >=1
    # of each in both folds).
    y[:2] = [0, 1]
    return X, y


def test_fit_all_models_returns_exactly_ml_model_labels():
    X, y = _tiny_synthetic_train()
    fitted = model_compare._fit_all_models(X, y, _TINY_LAYERS)
    assert tuple(fitted.keys()) == model_compare.ML_MODEL_LABELS


def test_fit_all_models_produces_working_predict_proba():
    X, y = _tiny_synthetic_train()
    fitted = model_compare._fit_all_models(X, y, _TINY_LAYERS)
    for label, model in fitted.items():
        proba = model.predict_proba(X)[:, 1]
        assert len(proba) == len(X), f"{label}: predict_proba length mismatch"
        assert np.all((proba >= 0) & (proba <= 1)), f"{label}: predict_proba out of [0,1]"


def test_fit_all_models_deterministic_for_fixed_seed():
    X, y = _tiny_synthetic_train()
    fitted_a = model_compare._fit_all_models(X, y, _TINY_LAYERS, seed=config.GLOBAL_SEED)
    fitted_b = model_compare._fit_all_models(X, y, _TINY_LAYERS, seed=config.GLOBAL_SEED)
    for label in model_compare.ML_MODEL_LABELS:
        proba_a = fitted_a[label].predict_proba(X)[:, 1]
        proba_b = fitted_b[label].predict_proba(X)[:, 1]
        assert np.allclose(proba_a, proba_b), f"{label}: not deterministic across two fits"


# ============================================================================
# 7. _raster_fingerprint / _cache_hit -- susceptibility_rasters()'s
#    cache-guard primitives. Fast, pure-logic unit tests (no raster I/O).
# ============================================================================

def test_raster_fingerprint_differs_when_train_idx_differs():
    bg = pd.DataFrame({"x": [0.0, 10.0, 20.0], "y": [0.0, 10.0, 20.0]})
    fp1 = model_compare._raster_fingerprint([1, 2, 3], [4, 5], bg, ["a", "b"], seed=42)
    fp2 = model_compare._raster_fingerprint([1, 2, 9], [4, 5], bg, ["a", "b"], seed=42)
    assert fp1 != fp2


def test_raster_fingerprint_differs_when_background_content_differs():
    bg1 = pd.DataFrame({"x": [0.0, 10.0, 20.0], "y": [0.0, 10.0, 20.0]})
    bg2 = pd.DataFrame({"x": [0.0, 10.0, 999.0], "y": [0.0, 10.0, 999.0]})
    fp1 = model_compare._raster_fingerprint([1, 2, 3], [4, 5], bg1, ["a", "b"], seed=42)
    fp2 = model_compare._raster_fingerprint([1, 2, 3], [4, 5], bg2, ["a", "b"], seed=42)
    assert fp1 != fp2


def test_raster_fingerprint_is_order_independent_for_train_idx():
    bg = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    fp1 = model_compare._raster_fingerprint([1, 2, 3], [4, 5], bg, ["a", "b"], seed=42)
    fp2 = model_compare._raster_fingerprint([3, 1, 2], [5, 4], bg, ["a", "b"], seed=42)
    assert fp1 == fp2


def test_cache_hit_false_when_raster_missing(tmp_path):
    fp = {"train_idx": [1, 2]}
    assert model_compare._cache_hit(tmp_path / "nope.tif", fp) is False


def test_cache_hit_true_when_fingerprint_matches(tmp_path):
    raster_path = tmp_path / "model.tif"
    raster_path.write_bytes(b"not a real tif -- cache_hit only checks the sidecar JSON")
    fp = {"train_idx": [1, 2], "seed": 42}
    model_compare._fingerprint_path(raster_path).write_text(json.dumps(fp))
    assert model_compare._cache_hit(raster_path, fp) is True


def test_cache_hit_false_when_fingerprint_mismatches(tmp_path):
    raster_path = tmp_path / "model.tif"
    raster_path.write_bytes(b"placeholder")
    model_compare._fingerprint_path(raster_path).write_text(json.dumps({"train_idx": [1, 2], "seed": 42}))
    assert model_compare._cache_hit(raster_path, {"train_idx": [9, 9], "seed": 42}) is False


def test_cache_hit_false_when_sidecar_missing(tmp_path):
    raster_path = tmp_path / "model.tif"
    raster_path.write_bytes(b"placeholder")
    # No .fingerprint.json written -- must never be treated as a match.
    assert model_compare._cache_hit(raster_path, {"train_idx": [1, 2]}) is False


# ============================================================================
# 8. susceptibility_rasters() -- real-fixture tests against the actual
#    canonical stack + maxent.fit_eval()'s holdout_model/flood.tif. Skipped if
#    either is missing (mirrors `real_comparison` above).
# ============================================================================

@pytest.fixture(scope="module")
def real_susceptibility_rasters(real_comparison, tmp_path_factory):
    if not evaluate.EVAL_RASTER.exists():
        pytest.skip(f"{evaluate.EVAL_RASTER} not built locally (run maxent.fit_eval() first)")
    if not model_compare.PRED_DIR.exists() or not any(model_compare.PRED_DIR.glob("*.tif")):
        pytest.skip(f"{model_compare.PRED_DIR} not populated locally")

    out_dir = tmp_path_factory.mktemp("susceptibility_rasters")
    paths = model_compare.susceptibility_rasters(
        real_comparison["train_idx"], real_comparison["test_idx"], real_comparison["background"],
        run_results=real_comparison["results"], out_dir=out_dir,
    )
    return dict(paths=paths, out_dir=out_dir, **real_comparison)


def test_susceptibility_rasters_returns_a_path_per_ml_model(real_susceptibility_rasters):
    paths = real_susceptibility_rasters["paths"]
    assert set(paths.keys()) == set(model_compare.ML_MODEL_LABELS)
    for label, p in paths.items():
        assert p.exists() and p.stat().st_size > 0, f"{label}: {p} missing/empty"
        assert p.suffix == ".tif"


def test_susceptibility_rasters_finite_in_mask_nan_outside(real_susceptibility_rasters):
    """Every raster shares the SAME valid-pixel footprint (the modeling
    grid's own valid_mask): finite exactly where the grid is valid, NaN
    (nodata) everywhere else -- never a raster with a different footprint
    from another model's."""
    _arrays, valid_mask, _transform, _layers = model_compare._load_modeling_arrays()
    for label, p in real_susceptibility_rasters["paths"].items():
        da = rioxarray.open_rasterio(p).squeeze("band", drop=True)
        finite = np.isfinite(da.values)
        assert np.array_equal(finite, valid_mask), f"{label}: raster footprint != valid_mask"
        assert (da.values[finite] >= 0).all() and (da.values[finite] <= 1).all(), (
            f"{label}: susceptibility values escaped [0, 1]"
        )


def test_susceptibility_rasters_consistency_auc_matches_run(real_susceptibility_rasters):
    """Re-derives the SAME consistency check susceptibility_rasters()
    itself asserts internally (raster-sampled test AUC vs. run()'s own
    reported AUC) -- proving the guard actually held for this real run,
    not merely that no AssertionError happened to propagate."""
    fx = real_susceptibility_rasters
    canonical = fx["samples"]
    background = fx["background"]
    test_idx, train_idx = fx["test_idx"], fx["train_idx"]
    _bg_train_idx, bg_test_idx = config.split(background)

    test_pts = canonical.iloc[test_idx][["x", "y"]]
    bg_test_pts = background.iloc[bg_test_idx][["x", "y"]]
    y_true = np.concatenate([np.ones(len(test_pts)), np.zeros(len(bg_test_pts))])
    all_xy = pd.concat([test_pts, bg_test_pts], ignore_index=True)

    run_auc = fx["results"].set_index("model")["auc"]
    for label, p in fx["paths"].items():
        da = rioxarray.open_rasterio(p).squeeze("band", drop=True)
        rows, cols = rowcol(da.rio.transform(), all_xy["x"].to_numpy(), all_xy["y"].to_numpy())
        sampled = da.values[np.asarray(rows), np.asarray(cols)]
        from sklearn.metrics import roc_auc_score
        raster_auc = roc_auc_score(y_true, sampled)
        assert abs(raster_auc - run_auc.loc[label]) < 0.02, (
            f"{label}: raster-sampled AUC {raster_auc:.4f} vs run() AUC {run_auc.loc[label]:.4f}"
        )


def test_susceptibility_rasters_cache_hit_skips_refit(real_susceptibility_rasters, monkeypatch):
    """A second call with the IDENTICAL inputs must not re-fit anything --
    proven directly by making a refit raise."""
    fx = real_susceptibility_rasters

    def _boom(*args, **kwargs):
        raise AssertionError("_fit_all_models must not be called on a full cache hit")

    monkeypatch.setattr(model_compare, "_fit_all_models", _boom)
    paths_again = model_compare.susceptibility_rasters(
        fx["train_idx"], fx["test_idx"], fx["background"],
        run_results=fx["results"], out_dir=fx["out_dir"],
    )
    assert paths_again == fx["paths"]


def test_susceptibility_rasters_consistency_guard_raises_on_auc_mismatch(real_susceptibility_rasters):
    fx = real_susceptibility_rasters
    bad_results = fx["results"].copy()
    bad_results.loc[bad_results["model"] == "Logistic Regression", "auc"] = 0.01
    with pytest.raises(AssertionError, match="CONSISTENCY GUARD FAILED"):
        model_compare.susceptibility_rasters(
            fx["train_idx"], fx["test_idx"], fx["background"],
            run_results=bad_results, out_dir=fx["out_dir"], models=["Logistic Regression"],
        )


def test_susceptibility_rasters_rejects_unknown_model_label(real_susceptibility_rasters):
    fx = real_susceptibility_rasters
    with pytest.raises(AssertionError, match="unknown model"):
        model_compare.susceptibility_rasters(
            fx["train_idx"], fx["test_idx"], fx["background"],
            run_results=fx["results"], out_dir=fx["out_dir"], models=["Neural Net"],
        )


def test_susceptibility_rasters_deterministic_across_two_out_dirs(real_comparison, tmp_path_factory):
    """Two INDEPENDENT calls (separate out_dirs, so the second cannot
    shortcut via the first's cache -- both genuinely fit+predict from
    scratch) must write byte-identical rasters for a seeded ML model."""
    if not evaluate.EVAL_RASTER.exists() or not model_compare.PRED_DIR.exists():
        pytest.skip("real predictor stack / eval raster not built locally")

    out_a = tmp_path_factory.mktemp("det_a")
    out_b = tmp_path_factory.mktemp("det_b")
    models = ["Logistic Regression", "Random Forest"]

    paths_a = model_compare.susceptibility_rasters(
        real_comparison["train_idx"], real_comparison["test_idx"], real_comparison["background"],
        run_results=real_comparison["results"], out_dir=out_a, models=models,
    )
    paths_b = model_compare.susceptibility_rasters(
        real_comparison["train_idx"], real_comparison["test_idx"], real_comparison["background"],
        run_results=real_comparison["results"], out_dir=out_b, models=models,
    )

    for label in models:
        da_a = rioxarray.open_rasterio(paths_a[label]).squeeze("band", drop=True).values
        da_b = rioxarray.open_rasterio(paths_b[label]).squeeze("band", drop=True).values
        finite_a, finite_b = np.isfinite(da_a), np.isfinite(da_b)
        assert np.array_equal(finite_a, finite_b), f"{label}: valid-pixel footprint differs"
        assert np.allclose(da_a[finite_a], da_b[finite_b]), f"{label}: not deterministic"


# ============================================================================
# 7. variable_importance_comparison() -- cross-model importance on the
#    CANONICAL layer set. Real-fixture; skipped if the stack isn't built.
#    Guards the defect this function was added to fix: the paper's cross-model
#    importance artefacts used to be computed on the superseded 21-predictor
#    stack, so they silently disagreed with every other number in the paper.
# ============================================================================

@pytest.fixture(scope="module")
def real_importance(real_comparison, tmp_path_factory):
    """NOTE: writes to a tmp dir, never model_compare.OUT_DIR -- the canonical
    importance_comparison.csv is a committed paper artefact built at the full
    n_repeats, and a test run must not silently overwrite it with its own
    faster, noisier version."""
    vic = config.DATA / "maxent" / "variable_importance_combined.csv"
    if not vic.exists():
        pytest.skip(f"{vic} not built locally (run maxent.jackknife_gains() first)")
    out_dir = tmp_path_factory.mktemp("importance")
    df = model_compare.variable_importance_comparison(
        real_comparison["train_idx"], real_comparison["test_idx"],
        real_comparison["background"], n_repeats=3, out_dir=out_dir,
    )
    return df, out_dir


def test_importance_covers_exactly_the_canonical_modeling_layers(real_importance):
    real_importance, _out_dir = real_importance
    """THE regression guard: one row per canonical predictor and nothing else.
    A stale artefact carrying the old 21-predictor set (with precipitation,
    SPI or flow accumulation in it) fails here."""
    expected = set(maxent.modeling_layers())
    assert set(real_importance["predictor"]) == expected
    assert len(real_importance) == len(expected)
    for dropped in ("precip_annual_mean", "precip_rx1day", "spi",
                    "flow_accumulation", "tri", "tpi", "median_income"):
        assert dropped not in set(real_importance["predictor"])


def test_importance_normalised_columns_are_shares(real_importance):
    real_importance, _out_dir = real_importance
    for col in ("maxent_jk_norm", "xgb_gain_norm", "lgb_gain_norm", "cat_gain_norm"):
        vals = real_importance[col]
        assert (vals >= 0).all(), f"{col} has negative shares"
        assert vals.sum() == pytest.approx(1.0, abs=1e-6), f"{col} does not sum to 1"


def test_importance_writes_both_csvs(real_importance):
    _df, out_dir = real_importance
    imp = out_dir / "importance_comparison.csv"
    rank = out_dir / "mean_importance_rank.csv"
    assert imp.exists() and rank.exists()
    on_disk = pd.read_csv(imp)
    assert set(on_disk["predictor"]) == set(maxent.modeling_layers())
    ranks = pd.read_csv(rank)
    assert len(ranks) == len(maxent.modeling_layers())
    # rank 1 is the most important; means are averages of five per-model ranks
    assert ranks["mean_rank_5models"].min() >= 1.0
    assert ranks["mean_rank_5models"].max() <= len(maxent.modeling_layers())


def test_importance_logistic_regression_excluded(real_importance):
    real_importance, _out_dir = real_importance
    """LR coefficients are not a comparable per-predictor importance (they
    depend on the one-hot encoding), so no LR column may appear."""
    assert not any("logistic" in c.lower() or c.startswith("lr_")
                   for c in real_importance.columns)
    assert "Logistic Regression" not in model_compare.IMPORTANCE_MODELS


# ============================================================================
# 8. shap_importance() -- exact TreeExplainer attributions for the four tree
#    ensembles. Real-fixture; skipped if the predictor stack isn't built.
# ============================================================================

@pytest.fixture(scope="module")
def real_shap(real_comparison, tmp_path_factory):
    """Same isolation rule as real_importance: never write to
    model_compare.OUT_DIR, whose shap_importance.csv is a paper artefact."""
    pytest.importorskip("shap")
    out_dir = tmp_path_factory.mktemp("shap")
    df = model_compare.shap_importance(
        real_comparison["train_idx"], real_comparison["test_idx"],
        real_comparison["background"], out_dir=out_dir, max_samples=300,
    )
    return df, out_dir


def test_shap_includes_maxent_and_excludes_logistic_regression(real_shap):
    """MaxEnt is included, but NOT via TreeExplainer -- it is not a tree model.
    Its attributions come from maxent.exact_shap, a closed form off the fitted
    .lambdas. Logistic regression is excluded because its one-hot design matrix
    has no one-to-one mapping back to the 17 predictors."""
    df, _out_dir = real_shap
    assert "MaxEnt" in model_compare.SHAP_MODELS
    assert "MaxEnt" not in model_compare.SHAP_TREE_MODELS
    assert "Logistic Regression" not in model_compare.SHAP_MODELS
    assert set(df["model"]) == set(model_compare.SHAP_MODELS)


def test_maxent_shap_sums_to_the_centred_linear_predictor():
    """THE exactness guard. For an additive model the Shapley values must sum,
    row by row, to that row's model output minus the baseline. Here the output
    is MaxEnt's linear predictor, recomputed independently from the .lambdas
    below; if exact_shap ever drops or double-counts a term, this fails."""
    lambdas = Path(maxent.HOLDOUT_MODEL_DIR) / "flood.lambdas"
    if not lambdas.exists():
        pytest.skip(f"{lambdas} not built locally")
    parsed = maxent.read_lambdas(lambdas)
    assert parsed["linear"], "no linear terms parsed"
    assert set(parsed["categorical"]) <= set(maxent.CATEGORICAL)

    layers = list(maxent.modeling_layers())
    rng = np.random.default_rng(0)
    X = pd.DataFrame(index=range(120), columns=layers, dtype="float64")
    for var in layers:
        if var in parsed["categorical"]:
            codes = sorted(parsed["categorical"][var])
            X[var] = rng.choice(codes, size=len(X))
        else:
            lo, hi = parsed["linear"].get(var, (0.0, 0.0, 1.0))[1:]
            X[var] = rng.uniform(lo, hi, size=len(X))

    def eta(frame):
        """MaxEnt's linear predictor, computed independently of exact_shap."""
        out = np.zeros(len(frame))
        for var in frame.columns:
            v = frame[var].to_numpy(dtype="float64")
            if var in parsed["categorical"]:
                table = parsed["categorical"][var]
                out = out + np.array([table.get(float(c), 0.0) for c in v])
                continue
            if var in parsed["linear"]:
                a, mn, mx = parsed["linear"][var]
                out = out + a * np.clip((v - mn) / (mx - mn), 0, 1)
            if var in parsed["quadratic"]:
                a, mn, mx = parsed["quadratic"][var]
                out = out + a * np.clip((v ** 2 - mn) / (mx - mn), 0, 1)
        return out

    phi = maxent.exact_shap(X, lambdas)
    assert list(phi.columns) == layers
    expected = eta(X) - eta(X).mean()
    assert np.allclose(phi.to_numpy().sum(axis=1), expected, atol=1e-9),         "Shapley values do not sum to the centred linear predictor"
    # centring: with background = X, every predictor's mean contribution is 0
    assert np.allclose(phi.mean(axis=0), 0.0, atol=1e-9)


def test_read_lambdas_rejects_non_additive_features(tmp_path):
    """A hinge or product feature makes the per-predictor decomposition
    invalid. It must fail loudly, not silently drop the term."""
    bad = tmp_path / "flood.lambdas"
    bad.write_text("dem*slope, 1.0, 0.0, 1.0\nlinearPredictorNormalizer, 0.0\n")
    with pytest.raises(ValueError, match="non-additive"):
        maxent.read_lambdas(bad)


def test_shap_covers_exactly_the_canonical_modeling_layers(real_shap):
    df, _out_dir = real_shap
    expected = set(maxent.modeling_layers())
    for model in model_compare.SHAP_MODELS:
        sub = df[df["model"] == model]
        assert set(sub["predictor"]) == expected, model
        assert len(sub) == len(expected), model
    assert "median_income" not in set(df["predictor"])


def test_shap_shares_are_nonnegative_and_sum_to_one_per_model(real_shap):
    df, _out_dir = real_shap
    for model, sub in df.groupby("model"):
        assert (sub["share"] >= 0).all(), model
        assert sub["share"].sum() == pytest.approx(1.0, abs=1e-6), model
        assert (sub["mean_abs_shap"] >= 0).all(), model


def test_shap_direction_is_a_correlation_not_a_raw_signed_value(real_shap):
    """direction_corr must be a unit-free rank correlation in [-1, 1]. Signed
    SHAP is in each model's own output units (log-odds for the boosters,
    probability for the forest) and is NOT comparable across models, which is
    why the figure plots this column instead."""
    df, _out_dir = real_shap
    d = df["direction_corr"].dropna()
    assert len(d) > 0
    assert d.between(-1.0, 1.0).all()
    # the two are genuinely different quantities, not a rescaling of each other
    assert not np.allclose(d, df["mean_signed_shap"].dropna())


def test_shap_recovers_the_known_physical_directions(real_shap):
    """Distance to outfall, elevation and HAND must reduce predicted
    susceptibility in every ensemble; available water storage must raise it.
    A sign flip here means the presence class was read off the wrong axis of
    the (n, features, classes) SHAP array."""
    df, _out_dir = real_shap
    for pred in ("distance_to_outfall", "dem", "hand"):
        vals = df[df["predictor"] == pred]["direction_corr"]
        assert (vals < 0).all(), f"{pred} should push susceptibility down: {list(vals)}"
    aws = df[df["predictor"] == "available_water_storage"]["direction_corr"]
    assert (aws > 0).all(), f"available_water_storage should push up: {list(aws)}"


def test_shap_writes_csv(real_shap):
    _df, out_dir = real_shap
    path = out_dir / "shap_importance.csv"
    assert path.exists()
    on_disk = pd.read_csv(path)
    assert set(on_disk["model"]) == set(model_compare.SHAP_MODELS)
    for col in ("mean_abs_shap", "mean_signed_shap", "direction_corr", "share"):
        assert col in on_disk.columns

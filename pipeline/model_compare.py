"""6-model comparison (MaxEnt reference + 5 presence-background
ML comparators) on an IDENTICAL, matched test set, with NO test-set
early-stopping for the boosting comparators.

Ports the model-fitting recipe of the original notebook
``04b_model_comparison`` (Logistic Regression / Random Forest / XGBoost /
LightGBM / CatBoost, with a presence-excluded pseudo-absence pool) into this canonical pipeline, with
ONE deliberate, load-bearing fix vs. the source:

**THE CRITICAL FIX.** The source notebook's three boosting comparators
(XGBoost/LightGBM/CatBoost) each watched ``X_test``/``y_test`` directly as
their ``eval_set`` for early stopping (``early_stopping_rounds``/
``use_best_model``) -- i.e. each model implicitly selected its own best
iteration count using the "held-out" test labels, which is test-set
leakage dressed up as regularization: the reported test AUC is no longer
an honest estimate of generalization, because the model's own complexity
was tuned to that exact test set. This module instead carves an INNER
validation split OUT OF TRAIN ONLY (``_inner_split``, stratified,
``config.GLOBAL_SEED``-seeded) and uses THAT as every boosting model's
``eval_set`` -- ``X_test``/``y_test`` are never passed to any
``eval_set``/``early_stopping_rounds``/``use_best_model`` argument
anywhere in this module. ``_fit_configs()`` is a small, static registry
of each model's eval-set source, asserted by
``tests/test_model_compare.py::test_no_model_early_stops_on_test`` to
never be ``"test"``.

**MaxEnt is NOT retrained here.** It is scored by reusing ``maxent.fit_eval()``'s
train-only evaluation raster (``maxent.HOLDOUT_MODEL_DIR / "flood.tif"``,
``evaluate.EVAL_RASTER``) -- the same raster the honest
out-of-sample AUC (``evaluate.oos_auc``) already uses -- sampled at the
SAME test presence points and the SAME test-partition background this
module builds for the 5 ML comparators, so all 6 rows in the returned
table score against literally the same test set.

**Pseudo-absence/background.** ``build_background()`` draws
``n`` (default 10,000) random valid AOI pixels from the canonical
17-layer modeling stack (``maxent.modeling_layers()``,
``predictors_screened/``), EXCLUDING any pixel that coincides with a
canonical presence point -- so the same 10 m cell can never appear as
both a presence and a background row (a fix the source notebook already
made, ported here; NOT the same background
``evaluate.sample_background`` builds, which has no presence-exclusion
because the MaxEnt-only OOS-AUC check in ``evaluate`` never needed it). That pool
is split 70/30 IN PARALLEL with the presence split, via the SAME
canonical ``config.split()`` (never a bespoke re-derivation), so
``run()``'s ``background`` argument and its internal train/test
partition are exactly as reproducible as the presence side.

**Categorical handling (ported from 04b, unchanged):** ``nlcd_landcover``/
``hydrologic_soil_group`` are one-hot encoded for Logistic Regression,
left as integer codes for Random Forest/XGBoost/LightGBM (see the
CORRECTION below), and passed as native categorical features to
CatBoost via ``cat_features``.

**CORRECTION (2026-08-13).** This docstring previously
justified the integer codes with "tree splits do not assume ordinality". That is
**wrong**: a threshold split on ``nlcd_landcover <= 43`` imposes exactly an
ordering on NLCD's nominal class codes, and the model can only carve that axis
into intervals. ``categorical_encoding_comparison()`` measures the consequence
rather than asserting it away -- it refits random forest (one-hot), XGBoost
(``enable_categorical``) and LightGBM (``categorical_feature``) and compares.
Observed effect: **|ΔAUC| <= 0.0008** for all three, so the canonical recipe is
left unchanged and the null result is reported.

**Full-AOI susceptibility rasters** (``susceptibility_rasters()``): fits each of the 5 ML
comparators via the SAME train-only, seeded recipe ``run()`` itself uses
(``_fit_all_models`` -- the shared helper both functions call, so there is
still only ONE fitting implementation in this module), then
``predict_proba``s over every valid pixel of the canonical modeling grid
and writes one ``<model>.tif`` per model under ``RASTER_DIR`` (gitignored).
A consistency guard re-samples each freshly-written raster at ``run()``'s
own test set and asserts the resulting AUC matches ``run()``'s reported
number for that model -- see ``susceptibility_rasters()``'s own docstring.
This is what ``pipeline.report``'s ``susceptibility_maps_comparison``
figure reads (report.py itself never fits/predicts anything).

**Still deliberately out of scope**: permutation/gain importance plots,
ROC-curve figures, and MaxEnt-vs-ML spatial (Spearman) correlation --
the metrics table never needed these, and nothing
about the full-AOI raster addition above changes that.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
from rasterio.transform import rowcol, xy as _xy

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from pipeline import config, evaluate, maxent, qa

PRED_DIR = config.DATA / "predictors_screened"
OUT_DIR = config.DATA / "model_comparison"
RASTER_DIR = OUT_DIR / "rasters"   # susceptibility_rasters()'s per-model .tif output dir -- gitignored.

# Same two nominal-class layers `pipeline.screen`/`pipeline.maxent` already
# single out for special handling -- re-exported here under this module's
# own name rather than a second hardcoded tuple.
CATEGORICAL = maxent.CATEGORICAL

DEFAULT_BACKGROUND_N = 10_000
# Fraction of TRAIN carved out for the boosting models' early-stopping
# validation fold -- THE CRITICAL FIX; never touches X_test/y_test.
INNER_VAL_FRAC = 0.20

# The 5 ML comparators susceptibility_rasters() can raster -- run()'s own
# row order for the non-MaxEnt rows (MaxEnt is excluded: it is never refit
# by this module -- see susceptibility_rasters()'s own docstring).
ML_MODEL_LABELS = ("Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost")

# label -> on-disk raster filename stem (`<out_dir>/<stem>.tif`) --
# lowercase/underscored so a raster's filename never carries the
# human-readable label's own space.
_RASTER_STEM = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
}


# ============================================================================
# Raster loading + point sampling
# ============================================================================

def _load_modeling_arrays(layers=None, pred_dir=PRED_DIR):
    """Load the canonical modeling stack (default ``maxent.modeling_layers()``,
    17 layers) from ``pred_dir`` (default ``predictors_screened/``) into a
    dict of in-memory float32 ndarrays, one shared full-grid validity mask
    (True where EVERY loaded layer is finite), and the common affine
    transform.

    ``predictors_screened/`` actually holds 18 files (17 modeling layers +
    ``median_income``, already excluded by ``maxent.modeling_layers()`` --
    see that function's docstring) -- ``layers`` restricts the read to
    exactly the canonical set. Every raster here already encodes nodata as
    NaN (this pipeline's own convention -- see
    ``pipeline.predictors.apply_modeling_transform``'s docstring), so
    ``np.isfinite`` alone is a correct validity test -- no sentinel-value
    translation needed (a deliberate departure from
    the original notebook's defensive ``arr[arr == nd] =
    np.nan``, written for an older, sentinel-based raster generation this
    pipeline no longer uses).

    Returns ``(arrays, valid_mask, transform, layers)`` -- ``layers`` is
    echoed back (resolved list, not ``None``) so callers that didn't pass
    one explicitly still know what was actually loaded.
    """
    if layers is None:
        layers = maxent.modeling_layers()
    layers = list(layers)
    pred_dir = Path(pred_dir)

    ref = rioxarray.open_rasterio(pred_dir / f"{layers[0]}.tif").squeeze("band", drop=True)
    nrows, ncols = ref.shape
    transform = ref.rio.transform()

    arrays = {}
    valid = np.ones((nrows, ncols), dtype=bool)
    for name in layers:
        da = rioxarray.open_rasterio(pred_dir / f"{name}.tif").squeeze("band", drop=True)
        arr = da.values.astype("float32")
        arrays[name] = arr
        valid &= np.isfinite(arr)
    return arrays, valid, transform, layers


def _presence_features(canonical, layers=None, pred_dir=PRED_DIR):
    """Sample the canonical modeling stack at every row of ``canonical``
    (an x/y point table, e.g. ``config.canonical_samples()``) -- nearest
    pixel, the SAME ``rasterio.transform.rowcol`` floor convention every
    other raster-sampling function in this pipeline shares
    (``predictors._sample_at_points``, ``config._modeling_grid_rowcol``,
    ``evaluate._sample_xy``).

    Every canonical presence point is already guaranteed finite across
    these layers: ``predictors.valid_mask()`` gates on the superset
    21-layer ``predictors.MODELING_LAYERS``, of which the (<=21) screened
    modeling layers are always a subset (screening only ever REMOVES
    layers from that 21-layer set; it never adds one). Asserted below
    rather than merely assumed, so a future predictor-stack change that
    breaks this invariant fails loudly here instead of silently feeding
    NaNs into sklearn.

    Out-of-bounds points (row/col outside the raster's own extent) are
    NaN-filled rather than indexed -- the same ``in_bounds`` guard
    ``predictors._sample_at_points`` uses, and for the same reason: a
    small negative row/col from ``rowcol()`` would otherwise silently
    wrap around to a valid-looking but WRONG pixel under plain numpy
    fancy indexing, rather than raising or NaN-ing. The completeness
    assert below then catches it explicitly instead of that silent
    corruption.

    Returns a DataFrame with one column per layer name, aligned to
    ``canonical``'s own row order (categorical columns already cast to
    ``int``).
    """
    arrays, _valid, transform, layers = _load_modeling_arrays(layers=layers, pred_dir=pred_dir)
    nrows, ncols = next(iter(arrays.values())).shape
    rows, cols = rowcol(transform, canonical["x"].to_numpy(), canonical["y"].to_numpy())
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_bounds = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)

    feats = {}
    for name in layers:
        vals = np.full(len(canonical), np.nan, dtype="float64")
        vals[in_bounds] = arrays[name][rows[in_bounds], cols[in_bounds]]
        feats[name] = vals
    feats = pd.DataFrame(feats)
    assert not feats.isna().any().any(), (
        "[model_compare] a canonical presence point sampled NaN (or fell "
        "off the raster extent) against the screened modeling stack -- "
        "valid_mask()'s 21-layer guarantee no longer covers this screened "
        "subset; investigate before trusting any downstream metric"
    )
    for c in CATEGORICAL:
        feats[c] = feats[c].astype(int)
    return feats


def build_background(n=DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED,
                      pred_dir=PRED_DIR, layers=None, presence=None):
    """The presence-excluded pseudo-absence pool: ``n`` random valid AOI pixels
    drawn from the canonical modeling stack, EXCLUDING every pixel that
    coincides with a canonical presence point -- so the same 10 m cell can
    never appear as both a presence row and a background row for the 5
    ML comparators below (a fix already made in the original notebook).
    This is a DIFFERENT (stricter) background than
    ``evaluate.sample_background`` builds for the MaxEnt-only OOS-AUC
    check -- that helper never excludes presence pixels, because scoring
    a fixed MaxEnt raster at presence vs. background points never risks
    training on the same cell twice the way jointly-fit presence-
    background classifiers do.

    Seeded (default ``config.GLOBAL_SEED``) for exact reproducibility --
    same seed always draws the identical ``n`` pixels.

    Returns a DataFrame: ``x``, ``y`` (pixel-center coordinates, the SAME
    column convention ``config.canonical_samples()`` uses) plus one column
    per layer in ``layers`` (default ``maxent.modeling_layers()``,
    feature values already extracted so callers never need to re-touch
    the rasters), with the two ``CATEGORICAL`` columns cast to ``int``
    (safe -- every row is drawn only from pixels finite across every
    layer, by construction of the ``valid`` mask below).
    """
    arrays, valid, transform, layers = _load_modeling_arrays(layers=layers, pred_dir=pred_dir)
    nrows, ncols = valid.shape

    if presence is None:
        presence = config.canonical_samples()
    pres_rows, pres_cols = rowcol(transform, presence["x"].to_numpy(), presence["y"].to_numpy())
    pres_rows = np.asarray(pres_rows)
    pres_cols = np.asarray(pres_cols)
    # Faithful port of 04b's own `in_bounds` guard: an off-grid presence
    # point cannot coincide with any in-grid background pixel anyway, so
    # it is simply dropped from the exclusion set rather than raising
    # (np.ravel_multi_index would otherwise ValueError on an out-of-range
    # row/col below).
    in_bounds = (pres_rows >= 0) & (pres_rows < nrows) & (pres_cols >= 0) & (pres_cols < ncols)
    pres_rows, pres_cols = pres_rows[in_bounds], pres_cols[in_bounds]

    pool = np.flatnonzero(valid.ravel())
    pres_flat = np.ravel_multi_index((pres_rows, pres_cols), (nrows, ncols))
    n_pool_before = len(pool)
    pool = np.setdiff1d(pool, pres_flat)   # drop presence-coincident pixels
    n_excluded = n_pool_before - len(pool)

    rng = np.random.default_rng(seed)
    bg_flat = rng.choice(pool, size=n, replace=False)
    bg_rows, bg_cols = np.unravel_index(bg_flat, (nrows, ncols))

    xs, ys = _xy(transform, bg_rows, bg_cols)
    data = {"x": np.asarray(xs, dtype="float64"), "y": np.asarray(ys, dtype="float64")}
    for name in layers:
        data[name] = arrays[name][bg_rows, bg_cols]

    df = pd.DataFrame(data)
    for c in CATEGORICAL:
        df[c] = df[c].astype(int)
    df.attrs["n_presence_excluded"] = int(n_excluded)
    return df


# ============================================================================
# Per-model early-stopping registry -- a regression guard against test-set early stopping.
# ============================================================================

def _fit_configs():
    """Per-model early-stopping ``eval_set`` source. Asserted by
    ``tests/test_model_compare.py::test_no_model_early_stops_on_test``:
    ``all(v != "test" for v in cfg.values())``.

    - ``"n/a"`` -- MaxEnt is not refit at all here; ``maxent.fit_eval()``'s train-only
      evaluation raster is reused as-is.
    - ``"none"`` -- Logistic Regression / Random Forest fit in one shot
      (fixed iteration count / no boosting rounds); there is nothing to
      early-stop against.
    - ``"train_inner_val"`` -- XGBoost/LightGBM/CatBoost early-stop
      against ``_inner_split``'s validation fold, carved OUT OF TRAIN
      ONLY, never ``X_test``/``y_test``. THIS is the fix this module exists
      to make true -- confirmed by inspection of ``_fit_xgboost``/
      ``_fit_lightgbm``/``_fit_catboost`` below, each of which only ever
      receives ``X_tr_inner``/``X_val_inner`` from ``run()``, never
      ``X_test``/``y_test``.
    """
    return {
        "MaxEnt": "n/a",
        "LogisticRegression": "none",
        "RandomForest": "none",
        "XGBoost": "train_inner_val",
        "LightGBM": "train_inner_val",
        "CatBoost": "train_inner_val",
    }


def _inner_split(X_train, y_train, val_frac=INNER_VAL_FRAC, seed=config.GLOBAL_SEED):
    """THE CRITICAL FIX: carve a validation split OUT OF TRAIN ONLY, for
    the three boosting models' early stopping. The original notebook
    ``04b_model_comparison`` (the source this module ports) watched
    ``X_test``/``y_test`` directly in every boosting ``eval_set`` --
    test-set early-stopping, where the model implicitly tunes its own
    complexity (iteration count) using the "held-out" labels, contaminating
    the very number that is supposed to measure generalization. This
    function is the one and only place ``run()`` derives a validation
    fold from, and it only ever receives ``X_train``/``y_train`` (never
    ``X_test``/``y_test``) as arguments.

    Stratified by ``y_train`` so the ~1:35 presence:background imbalance
    is preserved in both the inner-train and inner-validation folds.
    """
    return train_test_split(X_train, y_train, test_size=val_frac,
                             random_state=seed, stratify=y_train)


# ============================================================================
# The 5 presence-background ML comparators -- ported from 04b's §5-7b.
# ============================================================================

def _continuous_cols(layers):
    return [c for c in layers if c not in CATEGORICAL]


def _fit_logistic_regression(X_train, y_train, layers):
    """One-hot encoding for the 2 categorical predictors; standard scaling
    for continuous predictors (required for LR convergence). L2
    regularization, ``lbfgs`` solver. No early stopping -- fits in one
    shot on the FULL X_train/y_train (04b's own recipe, unchanged)."""
    continuous = _continuous_cols(layers)
    pre = ColumnTransformer([
        ("num", StandardScaler(), continuous),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                    random_state=config.GLOBAL_SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _fit_random_forest(X_train, y_train):
    """500 trees, ``class_weight='balanced'`` corrects for the
    presence:background imbalance. No early stopping -- fits in one shot
    on the FULL X_train/y_train (04b's own recipe, unchanged)."""
    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=config.GLOBAL_SEED,
    )
    rf.fit(X_train, y_train)
    return rf


def _fit_xgboost(X_tr, y_tr, X_val, y_val):
    """``scale_pos_weight`` corrects for class imbalance, computed from
    the actual fit data (``X_tr``/``y_tr`` -- the inner-train fold, NOT
    the full X_train). ``eval_set``/``early_stopping_rounds`` watch ONLY
    ``(X_val, y_val)`` -- ``_inner_split``'s validation fold -- never
    ``X_test``/``y_test``."""
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos, eval_metric="auc",
        early_stopping_rounds=30, random_state=config.GLOBAL_SEED,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _fit_lightgbm(X_tr, y_tr, X_val, y_val):
    """Leaf-wise gradient boosting; same ``scale_pos_weight``/early-
    stopping-on-inner-validation contract as ``_fit_xgboost``."""
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, num_leaves=15, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos, random_state=config.GLOBAL_SEED,
        n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)], eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _fit_catboost(X_tr, y_tr, X_val, y_val):
    """Ordered boosting with native categorical handling via
    ``cat_features``. ``eval_set``/``use_best_model`` watch ONLY
    ``(X_val, y_val)`` -- never ``X_test``/``y_test``.

    ``cat_features`` is filtered to whichever of ``CATEGORICAL``'s two
    names are actually columns of ``X_tr`` (mirrors
    ``maxent._categorical_layers_present``'s own reason for the identical
    filter: ``run()``'s real call always passes both, since
    ``maxent.modeling_layers()`` always includes both, but a caller
    exercising this helper directly against a reduced/synthetic frame
    should get CatBoost's native categorical handling for whichever
    subset is present, not a ``ValueError`` from a hardcoded name that
    isn't there).
    """
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    cat_features = [c for c in CATEGORICAL if c in X_tr.columns]
    model = CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=n_neg / n_pos, random_seed=config.GLOBAL_SEED,
        early_stopping_rounds=30, verbose=0,
    )
    model.fit(
        X_tr, y_tr,
        cat_features=cat_features, eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return model


def _fit_all_models(X_train, y_train, layers, seed=config.GLOBAL_SEED):
    """Fits all 5 presence-background ML comparators on the FULL
    ``X_train``/``y_train`` -- the EXACT recipe ``run()`` itself always
    used (this function IS that recipe, factored out so ``run()`` and
    ``susceptibility_rasters()`` share ONE fitting path rather than a
    second, possibly-drifting reimplementation). The three boosting
    models early-stop against ``_inner_split``'s validation fold, carved
    out of ``X_train``/``y_train`` ONLY -- see this module's docstring,
    "THE CRITICAL FIX", and ``_fit_configs()``'s registry.

    ``seed`` only seeds ``_inner_split`` itself -- every individual
    ``_fit_*`` helper called below already hardcodes its own
    ``random_state``/``random_seed`` at ``config.GLOBAL_SEED``, unchanged
    by this parameter; it exists so a caller can vary the inner-validation
    split alone without editing this function, not to vary each model's
    own fit.

    Returns a dict ``{label: fitted_model_or_pipeline}`` in
    ``ML_MODEL_LABELS`` order (Python dicts preserve insertion order) --
    ``run()`` iterates this directly to build its own row order, so that
    ordering is part of this function's contract, not an implementation
    accident.
    """
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = _inner_split(X_train, y_train, seed=seed)
    return {
        "Logistic Regression": _fit_logistic_regression(X_train, y_train, layers),
        "Random Forest": _fit_random_forest(X_train, y_train),
        "XGBoost": _fit_xgboost(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner),
        "LightGBM": _fit_lightgbm(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner),
        "CatBoost": _fit_catboost(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner),
    }


# ============================================================================
# Scoring -- shared MaxSS-threshold + confusion-matrix metrics, reusing
# `pipeline.evaluate`'s already-tested implementation (never a second,
# possibly-drifting reimplementation for the 5 ML comparators).
# ============================================================================

def _score_ml_model(model, X_train, y_train, X_test, y_test, label):
    """AUC (threshold-free) + MaxSS-threshold confusion-matrix metrics for
    one fitted presence-background classifier. The MaxSS threshold is
    Youden's J on the model's OWN TRAIN predictions
    (``evaluate.maxss_threshold_from_roc``, algebraically identical to
    04b's ``maxss_threshold`` -- ``tpr - fpr == tpr + (1-fpr) - 1``) --
    never derived from test, though this would be leakage-free either way
    since a THRESHOLD (not a fitted parameter) cannot itself overfit the
    labels it's picked against the same way an early-stopped iteration
    count can."""
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    threshold = evaluate.maxss_threshold_from_roc(y_train, prob_train)
    auc = float(roc_auc_score(y_test, prob_test))
    m = evaluate.threshold_metrics(y_test, prob_test, threshold, label=label)
    return {"model": label, "auc": auc, **m}


def _score_maxent(train_idx, test_idx, canonical, background, bg_test_idx,
                   eval_raster=None, results_csv=None):
    """MaxEnt's row: sample the Task-6.2 train-only evaluation raster
    (``evaluate.EVAL_RASTER`` = ``maxent.HOLDOUT_MODEL_DIR / "flood.tif"``)
    at the SAME test presence points and the SAME test-partition
    background the 5 ML comparators use -- reuses
    ``evaluate.oos_scores``/``evaluate.threshold_metrics``, the exact
    tested plumbing the honest OOS-AUC (``evaluate.oos_auc``) already runs on, rather than
    a second raster-sampling implementation.

    Invariant 2 (``qa.assert_eval_model_excludes_test``) is asserted
    explicitly here, BEFORE any raster is touched -- the same guard
    ``evaluate.oos_auc`` itself enforces internally, made visible at this
    call site too since this function reimplements the AUC computation
    (via ``oos_scores`` + a local ``roc_auc_score``, not by calling
    ``oos_auc`` a second time) so the threshold-metrics dict and the AUC
    number are guaranteed to come from ONE sampling pass, not two.
    """
    eval_raster = eval_raster if eval_raster is not None else evaluate.EVAL_RASTER
    results_csv = results_csv if results_csv is not None else evaluate.EVAL_RESULTS_CSV

    qa.assert_eval_model_excludes_test(train_idx, test_idx)

    test_pts = canonical.iloc[test_idx][["x", "y"]]
    bg_test_pts = background.iloc[bg_test_idx][["x", "y"]]
    y_true, y_score = evaluate.oos_scores(eval_raster, test_pts, bg_test_pts)

    auc = float(roc_auc_score(y_true, y_score))
    threshold = evaluate.maxss_threshold_from_results(results_csv)
    m = evaluate.threshold_metrics(y_true, y_score, threshold, label="MaxEnt (reference)")
    return {"model": "MaxEnt (reference)", "auc": auc, **m}


# ============================================================================
# Shared X_train/X_test assembly -- factored out of run() so
# susceptibility_rasters() (below) builds an IDENTICAL X_train (and
# re-derives the SAME X_test/bg_test_idx for its own consistency-AUC
# guard) rather than a second, possibly-drifting reimplementation of this
# feature-assembly step. run() itself now calls this -- same concat calls,
# same order, same arguments, just relocated; its own outputs are
# unchanged.
# ============================================================================

def _build_train_test_features(canonical, train_idx, test_idx, background, layers,
                                 pred_dir=PRED_DIR):
    """The exact X/y train/test tables ``run()`` fits/scores against.

    Returns ``(X_train, y_train, X_test, y_test, bg_train_idx, bg_test_idx)``
    -- ``bg_train_idx``/``bg_test_idx`` are ``config.split(background)``'s
    own return, echoed back so a caller (``_score_maxent``,
    ``susceptibility_rasters``) never has to re-derive that split a second
    time from a different call site.
    """
    pres_feats = _presence_features(canonical, layers=layers, pred_dir=pred_dir)

    bg_feats = background[list(layers)].copy()
    for c in CATEGORICAL:
        bg_feats[c] = bg_feats[c].astype(int)

    y_pres = np.ones(len(pres_feats), dtype=int)
    y_bg = np.zeros(len(bg_feats), dtype=int)

    bg_train_idx, bg_test_idx = config.split(background)
    qa.assert_no_leakage(bg_train_idx, bg_test_idx)

    X_train = pd.concat(
        [pres_feats.iloc[train_idx].reset_index(drop=True),
         bg_feats.iloc[bg_train_idx].reset_index(drop=True)],
        ignore_index=True,
    )
    y_train = np.concatenate([y_pres[train_idx], y_bg[bg_train_idx]])

    X_test = pd.concat(
        [pres_feats.iloc[test_idx].reset_index(drop=True),
         bg_feats.iloc[bg_test_idx].reset_index(drop=True)],
        ignore_index=True,
    )
    y_test = np.concatenate([y_pres[test_idx], y_bg[bg_test_idx]])

    return X_train, y_train, X_test, y_test, bg_train_idx, bg_test_idx


# ============================================================================
# THE comparison
# ============================================================================

def run(train_idx, test_idx, background, *, out_dir=OUT_DIR, layers=None,
        eval_raster=None, results_csv=None):
    """Compare 6 models -- MaxEnt reference + 5 presence-background ML
    comparators (Logistic Regression, Random Forest, XGBoost, LightGBM,
    CatBoost) -- on the IDENTICAL test set: the ``test_idx`` presence
    points plus ``config.split(background)``'s test partition. Returns one
    row per model (``auc`` + ``evaluate.threshold_metrics``'s full
    confusion-matrix-derived metrics), and also writes the table to
    ``<out_dir>/metrics_comparison.csv``.

    Parameters
    ----------
    train_idx, test_idx : positional indices into
        ``config.canonical_samples()`` -- ``config.split()``'s own return
        convention (194 train / 83 test on the real canonical 277-point
        set).
    background : the FULL presence-excluded background pool
        (``build_background()``'s return, default n=10,000) -- split
        70/30 IN PARALLEL with presence via the SAME canonical
        ``config.split()`` (never a bespoke re-derivation), so
        ``background``'s own train/test partition is exactly as
        reproducible as the presence side.

    THE CRITICAL FIX: XGBoost/LightGBM/CatBoost early-stop against
    ``_inner_split``'s validation fold, carved out of TRAIN ONLY --
    ``X_test``/``y_test`` are never passed to any ``eval_set``/
    ``early_stopping_rounds``/``use_best_model`` argument. See
    ``_fit_configs()`` for the per-model eval-set-source registry this
    claim is regression-tested against.

    See also ``susceptibility_rasters()`` -- fits the SAME 5 ML comparators
    (``_fit_all_models``, the helper this function itself calls below) and
    persists a full-AOI prediction raster per model, instead of only the
    aggregated metrics row this function returns.
    """
    if layers is None:
        layers = maxent.modeling_layers()

    qa.assert_no_leakage(train_idx, test_idx)

    canonical = config.canonical_samples()
    X_train, y_train, X_test, y_test, bg_train_idx, bg_test_idx = _build_train_test_features(
        canonical, train_idx, test_idx, background, layers)

    rows = []
    fitted = _fit_all_models(X_train, y_train, layers)
    for label, model in fitted.items():
        rows.append(_score_ml_model(model, X_train, y_train, X_test, y_test, label))

    maxent_row = _score_maxent(train_idx, test_idx, canonical, background, bg_test_idx,
                                eval_raster=eval_raster, results_csv=results_csv)
    rows = [maxent_row] + rows

    out = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "metrics_comparison.csv", index=False)
    return out


# ============================================================================
# Full-AOI susceptibility RASTERS -- closes the gap run() deliberately
# leaves open (per-model METRICS only). Fits via _fit_all_models (the SAME
# recipe run() itself uses -- never a second fitting implementation), then
# predict_proba's over the full valid grid and writes one <model>.tif per
# model, with a consistency-AUC guard against run()'s own reported number.
# ============================================================================

def _reference_layer(layers, pred_dir=PRED_DIR):
    """Reopens ``layers[0]``'s own raster file purely for its CRS
    (``_load_modeling_arrays`` already returns this layer's shape/
    transform, but not its CRS, and its 4-tuple return is kept stable for
    its two existing callers rather than widened for this one new need).
    Any ONE of the aligned modeling layers carries an identical CRS/
    transform/shape (``predictors._alignment_audit``), so ``layers[0]`` --
    whatever ``_load_modeling_arrays`` itself already used as its own
    shape/transform reference -- is as valid a CRS source as any other.
    """
    return rioxarray.open_rasterio(Path(pred_dir) / f"{layers[0]}.tif").squeeze("band", drop=True)


def _grid_features(layers, pred_dir=PRED_DIR):
    """The full valid-pixel grid as a model-ready DataFrame -- the SAME
    columns/order/dtypes ``_presence_features()``/``build_background()``
    hand every fitted model at TRAIN time (continuous stay float,
    ``CATEGORICAL`` cast to int) -- so ``predict_proba()`` sees an
    IDENTICAL column schema whether scoring a training row or a grid
    pixel. This identity is exactly what ``susceptibility_rasters()``'s own
    consistency-AUC guard checks empirically after the fact.

    Returns ``(X_grid, valid_mask, transform, layers, ref_da)`` -- ``X_grid``
    has one row per ``True`` entry of ``valid_mask``, taken in
    ``arr[valid_mask]``'s own (row-major, C-order) raveling order, so
    ``full[valid_mask] = probs`` (the inverse scatter) always lines back up
    correctly. ``ref_da`` is ``layers[0]``'s own opened DataArray -- the
    CRS/transform/shape reference for writing the output raster.
    """
    arrays, valid_mask, transform, layers = _load_modeling_arrays(layers=layers, pred_dir=pred_dir)
    ref_da = _reference_layer(layers, pred_dir=pred_dir)

    grid_data = {}
    for name in layers:
        col = arrays[name][valid_mask]
        if name in CATEGORICAL:
            col = col.astype(int)
        grid_data[name] = col
    X_grid = pd.DataFrame(grid_data)
    return X_grid, valid_mask, transform, layers, ref_da


def _write_susceptibility_raster(probs, valid_mask, ref_da, out_path):
    """Scatters ``probs`` (1-D, aligned to ``valid_mask``'s ``True``
    entries, in the SAME order ``_grid_features`` produced them) back into
    a full-grid float32 array (NaN outside ``valid_mask`` -- this
    pipeline's own nodata convention, see ``_load_modeling_arrays``'s
    docstring) and writes it as a GeoTIFF sharing ``ref_da``'s CRS/
    transform/shape -- the same ``ref.copy(data=arr)`` + ``.rio.
    to_raster()`` recipe ``maxent._asc_to_geotiff`` already established in
    this pipeline for its other model-output rasters.
    """
    full = np.full(valid_mask.shape, np.nan, dtype="float32")
    full[valid_mask] = np.asarray(probs, dtype="float32")
    da = ref_da.copy(data=full)
    da.rio.write_nodata(np.nan, inplace=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(out_path)
    return out_path


def _raster_fingerprint(train_idx, test_idx, background, layers, seed):
    """A cheap, content-based fingerprint of every input that determines a
    susceptibility raster's pixel values -- mirrors ``sensitivity``'s /
    ``robustness``'s own "hash the real content, refit on any mismatch"
    cache-guard convention (never a bare path-exists check, which could
    silently keep serving a raster fit under some earlier train_idx/
    test_idx/background/layers/seed combination). ``background``'s content
    (not just its length) is hashed -- order-independent, rounded to mm
    precision -- so a same-sized but DIFFERENT background draw is still
    caught as a mismatch.
    """
    train_key = sorted(int(i) for i in np.asarray(train_idx))
    test_key = sorted(int(i) for i in np.asarray(test_idx))
    bg_xy = np.round(background[["x", "y"]].to_numpy(dtype="float64"), 3)
    order = np.lexsort((bg_xy[:, 1], bg_xy[:, 0]))
    bg_hash = hashlib.sha256(bg_xy[order].tobytes()).hexdigest()[:16]
    return {
        "train_idx": train_key,
        "test_idx": test_key,
        "n_background": int(len(background)),
        "background_hash": bg_hash,
        "layers": list(layers),
        "seed": int(seed),
    }


def _fingerprint_path(raster_path):
    raster_path = Path(raster_path)
    return raster_path.parent / f"{raster_path.stem}.fingerprint.json"


def _cache_hit(raster_path, fingerprint):
    """True iff ``raster_path`` and its fingerprint sidecar both already
    exist AND the sidecar's recorded fingerprint matches ``fingerprint``
    exactly -- the ONE place ``susceptibility_rasters()`` decides to skip a
    refit, so a stale/missing sidecar always fails this (never a silent
    false-positive cache hit)."""
    raster_path = Path(raster_path)
    fp_path = _fingerprint_path(raster_path)
    if not (raster_path.exists() and fp_path.exists()):
        return False
    try:
        recorded = json.loads(fp_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == fingerprint


def susceptibility_rasters(train_idx, test_idx, background, *, run_results=None,
                            models=None, out_dir=RASTER_DIR, layers=None,
                            pred_dir=PRED_DIR, canonical=None,
                            seed=config.GLOBAL_SEED, metrics_csv=None, auc_tol=0.02):
    """Full-AOI susceptibility RASTERS for the presence-background ML
    comparators -- closes the one gap ``run()`` deliberately leaves open
    (per-model METRICS only; the 5 fitted model objects are local
    variables inside ``run()`` and are never persisted). Fits each of
    ``models`` (default ``ML_MODEL_LABELS``, all 5) TRAIN-ONLY and seeded,
    via the EXACT SAME recipe ``run()`` itself uses (``_fit_all_models``),
    then ``predict_proba``s over every valid pixel of the canonical
    modeling grid (``_load_modeling_arrays()``) and writes
    ``<out_dir>/<model>.tif`` (float32, nodata=NaN, sharing the modeling
    grid's own transform + CRS).

    The models NEVER see the 83 canonical test points (``test_idx``) or
    their matched background partition -- invariant #2, the SAME train/
    test contract ``run()`` itself enforces -- so mapping susceptibility
    across the WHOLE AOI (including the pixels under those held-out
    points) is not leakage: it is exactly what a susceptibility map is
    FOR. Every model here is seeded (default ``config.GLOBAL_SEED``), so
    two calls with the same inputs write byte-identical rasters.

    **CONSISTENCY GUARD** -- catches an encoding/leakage bug loudly rather
    than shipping a wrong map: after writing (or confirming the cache of)
    each model's raster, this function samples it -- nearest-pixel, the
    SAME ``rasterio.transform.rowcol`` floor convention every other
    raster-sampling function in this pipeline shares -- at the IDENTICAL
    test presence (``test_idx``) + test-partition background points
    ``run()`` itself scores that model against, recomputes AUC, and
    asserts it matches ``run()``'s own reported ``auc`` for that model
    (read from ``run_results`` if given, else parsed from
    ``metrics_csv``/``<out_dir's parent>/metrics_comparison.csv`` --
    i.e. ``run()``'s own default output path) within ``auc_tol`` (default
    0.02). A mismatch means the grid's column encoding (order/dtype/
    categorical coding) diverged from what the model was actually trained
    on -- raised immediately, never silently shipped. This guard re-runs
    on EVERY call, cached raster or not, so a stale/corrupted cached
    raster is still caught rather than trusted forever.

    **Cache-aware**: each ``<model>.tif`` carries a JSON fingerprint
    sidecar (``<model>.fingerprint.json``) recording exactly which
    ``train_idx``/``test_idx``/``background``/``layers``/``seed`` produced
    it (``_raster_fingerprint``/``_cache_hit``) -- mirrors
    ``sensitivity``/``robustness``'s own "hash the real inputs, refit on
    any mismatch" cache-guard convention, never a bare path-exists check.
    A model whose raster+fingerprint already match the CURRENT call's
    inputs is not refit.

    Parameters
    ----------
    train_idx, test_idx : positional indices into ``canonical`` (default
        ``config.canonical_samples()``) -- the SAME two arrays passed to
        ``run()``.
    background : the FULL presence-excluded background pool
        (``build_background()``'s return) -- the SAME object passed to
        ``run()``; this function re-derives ``run()``'s own 70/30
        ``config.split(background)`` partition from it internally.
    run_results : optional DataFrame matching ``run()``'s own return value
        (has a ``"model"``/``"auc"`` column pair). Default ``None`` reads
        ``metrics_csv`` (default ``<out_dir>.parent / "metrics_comparison.
        csv"``) from disk instead -- ``run()``'s own default write
        location, so calling ``run()`` immediately beforehand (the
        notebook's own Stage-4 order) needs no extra argument here.
    models : optional subset of ``ML_MODEL_LABELS`` to raster (default:
        all 5). Never ``"MaxEnt (reference)"`` -- MaxEnt is not refit
        here; ``pipeline.report``'s comparison figure reuses ``maxent.fit_eval()``'s
        own train-only evaluation raster (``evaluate.EVAL_RASTER``)
        directly for its MaxEnt panel instead.
    canonical, pred_dir : injectable for testability against a small
        synthetic stack (mirrors ``build_background(presence=...,
        pred_dir=...)``'s own convention) -- default to
        ``config.canonical_samples()`` / ``PRED_DIR`` (the real canonical
        stack) when omitted.

    Returns ``{label: Path}`` for exactly the models requested.
    """
    if layers is None:
        layers = maxent.modeling_layers()
    layers = list(layers)

    models = list(models) if models is not None else list(ML_MODEL_LABELS)
    unknown = [m for m in models if m not in ML_MODEL_LABELS]
    assert not unknown, (
        f"[model_compare] susceptibility_rasters: unknown model(s) {unknown} -- "
        f"must be a subset of {ML_MODEL_LABELS}"
    )

    qa.assert_no_leakage(train_idx, test_idx)
    bg_train_idx, bg_test_idx = config.split(background)
    qa.assert_no_leakage(bg_train_idx, bg_test_idx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_results is None:
        metrics_path = Path(metrics_csv) if metrics_csv is not None \
            else (out_dir.parent / "metrics_comparison.csv")
        assert metrics_path.exists(), (
            f"[model_compare] susceptibility_rasters: {metrics_path} not found -- call "
            f"model_compare.run(train_idx, test_idx, background) first (it writes this "
            f"CSV), or pass run_results=<run()'s own return value> explicitly. The "
            f"consistency-AUC guard below needs a real run()-reported number to check "
            f"every raster against."
        )
        run_results = pd.read_csv(metrics_path)
    run_auc = run_results.set_index("model")["auc"]
    missing_auc = [m for m in models if m not in run_auc.index]
    assert not missing_auc, (
        f"[model_compare] susceptibility_rasters: run_results has no row for "
        f"{missing_auc} -- was it really produced by model_compare.run() on this same "
        f"train_idx/test_idx/background?"
    )

    if canonical is None:
        canonical = config.canonical_samples()

    fp = _raster_fingerprint(train_idx, test_idx, background, layers, seed)
    raster_paths = {label: out_dir / f"{_RASTER_STEM[label]}.tif" for label in models}
    to_fit = [label for label in models if not _cache_hit(raster_paths[label], fp)]

    fitted = {}
    if to_fit:
        X_train, y_train, _X_test, _y_test, _bg_train_idx, _bg_test_idx = _build_train_test_features(
            canonical, train_idx, test_idx, background, layers, pred_dir=pred_dir)
        all_fitted = _fit_all_models(X_train, y_train, layers, seed=seed)
        fitted = {label: all_fitted[label] for label in to_fit}

    X_grid, valid_mask, transform, _layers_echo, ref_da = _grid_features(layers, pred_dir=pred_dir)

    test_pts = canonical.iloc[test_idx][["x", "y"]]
    bg_test_pts = background.iloc[bg_test_idx][["x", "y"]]
    y_true_test = np.concatenate([np.ones(len(test_pts)), np.zeros(len(bg_test_pts))])
    all_test_xy = pd.concat([test_pts, bg_test_pts], ignore_index=True)
    rows_rc, cols_rc = rowcol(transform, all_test_xy["x"].to_numpy(), all_test_xy["y"].to_numpy())
    rows_rc, cols_rc = np.asarray(rows_rc), np.asarray(cols_rc)

    out_paths = {}
    for label in models:
        raster_path = raster_paths[label]

        if label in fitted:
            probs = fitted[label].predict_proba(X_grid)[:, 1]
            _write_susceptibility_raster(probs, valid_mask, ref_da, raster_path)
            _fingerprint_path(raster_path).write_text(json.dumps(fp, indent=2))
            print(f"[model_compare] susceptibility_rasters: wrote {raster_path}")
        else:
            print(f"[model_compare] susceptibility_rasters: {raster_path} cached (fingerprint match)")

        sampled_full = rioxarray.open_rasterio(raster_path).squeeze("band", drop=True).values
        sampled = sampled_full[rows_rc, cols_rc]
        finite = np.isfinite(sampled)
        assert finite.all(), (
            f"[model_compare] susceptibility_rasters: {raster_path} sampled "
            f"{int((~finite).sum())} NaN of {len(sampled)} test/background points -- a "
            f"test point fell outside the raster's valid footprint?"
        )
        raster_auc = float(roc_auc_score(y_true_test, sampled))
        expected_auc = float(run_auc.loc[label])
        diff = abs(raster_auc - expected_auc)
        assert diff < auc_tol, (
            f"[model_compare] susceptibility_rasters: CONSISTENCY GUARD FAILED for "
            f"{label} -- raster-sampled test AUC {raster_auc:.4f} vs. run()-reported "
            f"{expected_auc:.4f} (diff {diff:.4f} >= tolerance {auc_tol}); the grid's "
            f"column encoding has diverged from what this model was actually trained "
            f"on -- investigate before trusting {raster_path}"
        )
        print(f"[model_compare] susceptibility_rasters: {label} consistency OK (raster "
              f"AUC {raster_auc:.4f} vs run() AUC {expected_auc:.4f}, diff {diff:.4f})")
        out_paths[label] = raster_path

    return out_paths


# ============================================================================
# Cross-model variable importance.
#
# Previously OUT OF SCOPE for this module (see the module docstring's
# "Deliberately out of scope" note), which left the paper's cross-model
# importance figures and the mean-rank column of its importance table sourced
# from the SUPERSEDED 21-predictor notebook run -- the only artefacts in the
# manuscript still computed on a predictor set the pipeline no longer uses.
# This section closes that gap: it derives every model's importance on the SAME
# canonical layer set, presence set, split and fitted models that `run()` scores,
# by reusing `_build_train_test_features` + `_fit_all_models` rather than
# re-deriving either.
#
# Importance is NOT comparable in absolute terms across these model families
# (permutation drop in AUC, split-gain, and jackknife gain are different
# quantities on different scales), so each column is normalised to sum to 1
# within its own model and the paper compares RANKS, never raw magnitudes.
# ============================================================================

# Logistic regression is deliberately absent: its coefficients depend on the
# one-hot encoding of the two categorical layers and are not a per-predictor
# importance comparable with the others (the manuscript states this too).
IMPORTANCE_MODELS = ("MaxEnt", "Random Forest", "XGBoost", "LightGBM", "CatBoost")

IMPORTANCE_CSV = OUT_DIR / "importance_comparison.csv"
MEAN_RANK_CSV = OUT_DIR / "mean_importance_rank.csv"


def _normalise(series, layers):
    """Reindex onto ``layers`` and scale to sum to 1 (0-filled, negatives
    clipped). Negative permutation importance means "shuffling this predictor
    *helped*", i.e. no signal; clipping it to 0 keeps the column a proper
    share-of-importance rather than letting a negative value inflate the
    positive entries through the normalising sum.
    """
    s = pd.Series(series, dtype="float64").reindex(layers).fillna(0.0).clip(lower=0)
    total = s.sum()
    return s / total if total > 0 else s


def variable_importance_comparison(train_idx, test_idx, background, *, layers=None,
                                    pred_dir=PRED_DIR, out_dir=OUT_DIR,
                                    n_repeats=10, seed=config.GLOBAL_SEED):
    """Per-predictor importance for the five models that admit a comparable
    measure, on the canonical modeling stack.

    Sources, one per model family (never a re-derivation of a quantity another
    stage already owns):
      - **MaxEnt** -- the jackknife unique contribution from
        ``maxent/variable_importance_combined.csv`` (read, not recomputed:
        that file is the output of ~2N+1 real MaxEnt fits).
      - **Random Forest** -- permutation importance (drop in score when a
        predictor is shuffled) on the TEST partition, ``n_repeats`` shuffles.
      - **XGBoost / LightGBM / CatBoost** -- the boosters' own split-gain
        importance.

    Every model is the one ``_fit_all_models`` produces from
    ``_build_train_test_features``' tables, so the fits here are identical to
    the ones ``run()`` scores -- the boosting models therefore also inherit
    the inner-split early stopping (never the test set).

    Returns a DataFrame with one row per predictor and columns
    ``predictor, maxent_jk_unique, maxent_jk_norm, rf_perm_mean, rf_perm_std,
    xgb_gain_norm, lgb_gain_norm, cat_gain_norm``; also writes it to
    ``<out_dir>/importance_comparison.csv`` and the derived mean-rank table
    (1 = most important, averaged over the five models) to
    ``<out_dir>/mean_importance_rank.csv``.
    """
    from sklearn.inspection import permutation_importance

    if layers is None:
        layers = maxent.modeling_layers()
    layers = list(layers)

    qa.assert_no_leakage(train_idx, test_idx)
    canonical = config.canonical_samples()
    X_train, y_train, X_test, y_test, _bg_tr, _bg_te = _build_train_test_features(
        canonical, train_idx, test_idx, background, layers, pred_dir=pred_dir)

    models = _fit_all_models(X_train, y_train, layers, seed=seed)

    perm = permutation_importance(models["Random Forest"], X_test, y_test,
                                   n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    rf_mean = pd.Series(perm.importances_mean, index=layers)
    rf_std = pd.Series(perm.importances_std, index=layers)

    xgb_gain = models["XGBoost"].get_booster().get_score(importance_type="gain")
    lgb_model = models["LightGBM"]
    lgb_gain = dict(zip(lgb_model.feature_name_,
                        lgb_model.booster_.feature_importance("gain")))
    cat_model = models["CatBoost"]
    cat_gain = dict(zip(cat_model.feature_names_, cat_model.get_feature_importance()))

    vic_path = config.DATA / "maxent" / "variable_importance_combined.csv"
    if not vic_path.exists():
        raise FileNotFoundError(
            f"[model_compare] variable_importance_comparison: {vic_path} not found -- "
            f"run maxent.variable_importance_combined() first"
        )
    vic = pd.read_csv(vic_path).set_index("predictor")
    jk = vic["unique_contribution"].reindex(layers).fillna(0.0)

    df = pd.DataFrame({
        "predictor": layers,
        "maxent_jk_unique": jk.values,
        "maxent_jk_norm": _normalise(jk, layers).values,
        "rf_perm_mean": rf_mean.values,
        "rf_perm_std": rf_std.values,
        "xgb_gain_norm": _normalise(xgb_gain, layers).values,
        "lgb_gain_norm": _normalise(lgb_gain, layers).values,
        "cat_gain_norm": _normalise(cat_gain, layers).values,
    })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "importance_comparison.csv", index=False)

    rank_cols = ["maxent_jk_norm", "rf_perm_mean", "xgb_gain_norm",
                 "lgb_gain_norm", "cat_gain_norm"]
    ranks = df.set_index("predictor")[rank_cols].rank(ascending=False, method="min")
    mean_rank = ranks.mean(axis=1).sort_values().rename("mean_rank_5models")
    mean_rank.to_frame().to_csv(out_dir / "mean_importance_rank.csv")

    print(f"[model_compare] variable_importance_comparison: {len(df)} predictors x "
          f"{len(IMPORTANCE_MODELS)} models -> {out_dir / 'importance_comparison.csv'}")
    print(f"[model_compare] top by mean rank: "
          + ", ".join(f"{k} ({v:.1f})" for k, v in mean_rank.head(3).items()))
    return df


# ============================================================================
# SHAP -- per-observation attributions for the tree ensembles.
#
# This literature's standard interpretability device (29 of the 36 papers
# we surveyed use it), and it
# answers a question the rank-based measures cannot: not just which predictors
# a model relies on, but in which direction each one pushes an individual
# prediction.
#
# Deliberately restricted to the four tree ensembles. MaxEnt runs as a Java
# subprocess with no sklearn-style predict interface, so a SHAP explanation of
# it would require KernelExplainer -- a sampling approximation, slow and with
# its own error -- when MaxEnt already reports the exact fitted marginal
# response directly (maxent.response_curves). Approximating a model that can be
# read exactly would be a step backwards, so the primary model is interpreted
# through its own response curves and SHAP is used for the comparators, where
# TreeExplainer is exact and cheap.
# ============================================================================

SHAP_TREE_MODELS = ("Random Forest", "XGBoost", "LightGBM", "CatBoost")
# MaxEnt is first because it is the paper's reference model. Its attributions
# come from maxent.exact_shap (a closed form read off the fitted .lambdas),
# not from TreeExplainer -- see shap_importance.
SHAP_MODELS = ("MaxEnt",) + SHAP_TREE_MODELS
SHAP_CSV = OUT_DIR / "shap_importance.csv"


CATEGORICAL_ENCODING_CSV = OUT_DIR / "categorical_encoding_comparison.csv"


def categorical_encoding_comparison(train_idx, test_idx, background, *, layers=None,
                                    pred_dir=PRED_DIR, seed=config.GLOBAL_SEED):
    """Does treating NLCD as ORDINAL integers instead of a nominal category
    change the tree models' performance?

    This module's docstring used to say the tree models receive the
    categoricals "as integer codes ... (tree splits do not assume ordinality)".
    **That parenthetical was wrong.** A threshold split on ``nlcd_landcover <= 43``
    imposes exactly an ordering on NLCD's class codes (11 open water, 21-24
    developed, 41-43 forest, 81-82 agriculture ...), which are nominal. The
    model can only carve the code axis into intervals, so it cannot isolate
    e.g. {open water, developed high} without dragging everything between them
    along.

    CatBoost is unaffected -- it already uses ``cat_features``. Random forest,
    XGBoost and LightGBM are not.

    Compares, on the identical matched test set:
      * **ordinal** -- the current behaviour, integer codes;
      * **nominal** -- one-hot for random forest (scikit-learn has no native
        categorical support) and native handling for XGBoost
        (``enable_categorical``) and LightGBM (``categorical_feature``).

    Returns a DataFrame with both AUCs per model and the delta, and writes it to
    ``OUT_DIR/categorical_encoding_comparison.csv``.
    """
    if layers is None:
        layers = maxent.modeling_layers()
    layers = list(layers)
    qa.assert_no_leakage(train_idx, test_idx)

    canonical = config.canonical_samples()
    X_train, y_train, X_test, y_test, _bt, _bte = _build_train_test_features(
        canonical, train_idx, test_idx, background, layers, pred_dir=pred_dir)
    cats = [c for c in CATEGORICAL if c in X_train.columns]
    X_tr_i, X_val_i, y_tr_i, y_val_i = _inner_split(X_train, y_train, seed=seed)

    rows = []

    # ---- ordinal (current behaviour) -------------------------------------
    ordinal = {
        "Random Forest": _fit_random_forest(X_train, y_train),
        "XGBoost": _fit_xgboost(X_tr_i, y_tr_i, X_val_i, y_val_i),
        "LightGBM": _fit_lightgbm(X_tr_i, y_tr_i, X_val_i, y_val_i),
    }
    for name, m in ordinal.items():
        rows.append(dict(model=name, encoding="ordinal_integer_codes",
                         auc=float(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))))

    # ---- nominal ---------------------------------------------------------
    # random forest: one-hot, since sklearn has no native categorical support
    oh = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)],
        remainder="passthrough")
    rf_pipe = Pipeline([("prep", oh),
                        ("rf", RandomForestClassifier(
                            n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                            class_weight="balanced", n_jobs=-1,
                            random_state=seed))])
    rf_pipe.fit(X_train, y_train)
    rows.append(dict(model="Random Forest", encoding="nominal_one_hot",
                     auc=float(roc_auc_score(y_test, rf_pipe.predict_proba(X_test)[:, 1]))))

    def _as_category(df):
        out = df.copy()
        for c in cats:
            out[c] = out[c].astype("category")
        return out

    Xtr_c, Xval_c, Xte_c = (_as_category(d) for d in (X_tr_i, X_val_i, X_test))
    n_neg, n_pos = int((y_tr_i == 0).sum()), int((y_tr_i == 1).sum())

    xgb_c = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos, eval_metric="auc",
        early_stopping_rounds=30, random_state=seed, n_jobs=-1, verbosity=0,
        enable_categorical=True, tree_method="hist")
    xgb_c.fit(Xtr_c, y_tr_i, eval_set=[(Xval_c, y_val_i)], verbose=False)
    rows.append(dict(model="XGBoost", encoding="nominal_native",
                     auc=float(roc_auc_score(y_test, xgb_c.predict_proba(Xte_c)[:, 1]))))

    lgb_c = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, num_leaves=15, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos,
        random_state=seed, n_jobs=-1, verbosity=-1)
    lgb_c.fit(Xtr_c, y_tr_i, eval_set=[(Xval_c, y_val_i)], eval_metric="auc",
              categorical_feature=cats,
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    rows.append(dict(model="LightGBM", encoding="nominal_native",
                     auc=float(roc_auc_score(y_test, lgb_c.predict_proba(Xte_c)[:, 1]))))

    df = pd.DataFrame(rows)
    wide = df.pivot(index="model", columns="encoding", values="auc").reset_index()
    nominal_col = [c for c in wide.columns if c.startswith("nominal")]
    wide["auc_nominal"] = wide[nominal_col].bfill(axis=1).iloc[:, 0]
    wide = wide[["model", "ordinal_integer_codes", "auc_nominal"]]
    wide.columns = ["model", "auc_ordinal", "auc_nominal"]
    wide["delta"] = wide["auc_nominal"] - wide["auc_ordinal"]
    CATEGORICAL_ENCODING_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(CATEGORICAL_ENCODING_CSV, index=False)
    print(f"[model_compare] wrote {CATEGORICAL_ENCODING_CSV.name}")
    return wide


def shap_importance(train_idx, test_idx, background, *, layers=None,
                     pred_dir=PRED_DIR, out_dir=OUT_DIR, max_samples=2000,
                     seed=config.GLOBAL_SEED):
    """Mean |SHAP value| per predictor for each tree ensemble, plus the signed
    mean, computed on the held-out test rows.

    Returns a long DataFrame (``model``, ``predictor``, ``mean_abs_shap``,
    ``mean_signed_shap``, ``share``) and writes it to
    ``<out_dir>/shap_importance.csv``. ``share`` is ``mean_abs_shap``
    normalised within each model, so magnitudes are comparable across models
    that operate on different output scales.

    The signed mean is reported alongside the magnitude because the two answer
    different questions: magnitude says how much a predictor moves predictions,
    sign says which way on average. A predictor can be influential and
    direction-neutral, and only the pair distinguishes that from an
    unimportant one.

    ``max_samples`` caps the explained rows (the background rows dominate the
    test frame and add little to a per-predictor mean); sampling is seeded.
    """
    import shap
    from scipy.stats import spearmanr

    if layers is None:
        layers = maxent.modeling_layers()
    layers = list(layers)

    qa.assert_no_leakage(train_idx, test_idx)
    canonical = config.canonical_samples()
    X_train, y_train, X_test, y_test, _bt, _bte = _build_train_test_features(
        canonical, train_idx, test_idx, background, layers, pred_dir=pred_dir)
    models = _fit_all_models(X_train, y_train, layers, seed=seed)

    if len(X_test) > max_samples:
        X_expl = X_test.sample(n=max_samples, random_state=seed)
    else:
        X_expl = X_test

    rows = []

    def _emit(label, vals):
        mean_abs = np.abs(vals).mean(axis=0)
        mean_signed = vals.mean(axis=0)
        total = mean_abs.sum()
        for k, pred in enumerate(layers):
            # Direction: Spearman correlation between a predictor's VALUE and
            # its own SHAP contribution. Signed SHAP alone is not comparable
            # across these models -- each reports in its own output units
            # (log-odds for the boosters, probability for the forest, the
            # linear predictor for MaxEnt) -- so a shared axis is dominated by
            # whichever model has the widest scale. The rank correlation is
            # unit-free and reads directly: negative means higher predictor
            # values push the prediction down.
            xv = X_expl[pred].to_numpy(dtype="float64")
            sv = vals[:, k]
            if np.std(xv) > 0 and np.std(sv) > 0:
                direction = float(spearmanr(xv, sv).statistic)
            else:
                direction = float("nan")
            rows.append(dict(model=label, predictor=pred,
                             mean_abs_shap=float(mean_abs[k]),
                             mean_signed_shap=float(mean_signed[k]),
                             direction_corr=direction,
                             share=float(mean_abs[k] / total) if total > 0 else 0.0))
        print(f"[model_compare] SHAP {label}: explained {len(vals)} rows, "
              f"top = {layers[int(np.argmax(mean_abs))]}")

    # MaxEnt: exact closed-form Shapley values off the TRAIN-ONLY fit, so the
    # attribution set matches the tree ensembles (fit on train, explained on
    # the held-out rows). No sampling and no KernelExplainer approximation --
    # at FINAL_CONFIG the linear predictor is additive, so each predictor's
    # Shapley value is just its own centred contribution.
    lambdas = Path(maxent.HOLDOUT_MODEL_DIR) / "flood.lambdas"
    if lambdas.exists():
        me = maxent.exact_shap(X_expl[layers], lambdas)
        _emit("MaxEnt", me.to_numpy(dtype="float64"))
    else:
        print(f"[model_compare] SHAP: skipping MaxEnt, {lambdas} not found "
              f"(run the held-out MaxEnt fit first)")

    for label in SHAP_TREE_MODELS:
        model = models[label]
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_expl)
        vals = np.asarray(vals)
        if vals.ndim == 3:                 # (n, features, classes) -> positive class
            vals = vals[..., -1]
        _emit(label, vals)

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "shap_importance.csv", index=False)
    return df

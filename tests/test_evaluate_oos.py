"""Tests for pipeline.evaluate -- the honest
out-of-sample AUC (sampling the EVAL/train-only raster at TEST points,
never the FINAL/full-model raster -- that substitution was the original
bug: a resubstitution AUC of 0.9842 silently
standing in for a true 0.9684 holdout) + threshold metrics + the two
missing field-standard SDM figures (success/prediction-rate curves,
density-classed susceptibility map).

Split into two tiers, mirroring test_model_families.py's own convention:
  1. Pure-logic / synthetic-data tests (fast, no real raster I/O) --
     threshold_metrics against a hand-computed 2x2 toy (matching this
     repo's test_maxent_aicc.py convention of verifying arithmetic against
     an independently hand-computed ground truth, not just "runs without
     crashing"), the _success_prediction_rate_arrays/_classify_values pure
     numpy cores against hand-computed toys, and sample_background's
     determinism against a small synthetic in-memory raster.
  2. Real-fixture tests against the actual on-disk MaxEnt rasters
     (data/processed/maxent/holdout_model/flood.tif,
     final/cloglog/flood_avg.tif) -- gitignored build artifacts, so each
     real-data test/fixture skips (rather than hard-failing) if the file
     is not present locally, matching this repo's tolerance for a fresh
     checkout that has not yet run the MaxEnt fits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from rasterio.transform import from_origin

from pipeline import config, evaluate, maxent


def _fake_raster(values, cell=10.0, crs="EPSG:26914"):
    """A minimal synthetic georeferenced raster (rioxarray DataArray) for
    fast, real-file-free tests of raster-sampling wrappers -- top-left
    origin at (0, values.shape[0]*cell), `cell`-metre square pixels, no
    coordinate arrays needed (``rio.write_transform`` sets the transform
    directly, verified against a hand-computed rowcol lookup)."""
    values = np.asarray(values, dtype="float64")
    da = xr.DataArray(values, dims=("y", "x"))
    transform = from_origin(0, values.shape[0] * cell, cell, cell)
    da = da.rio.write_transform(transform)
    da = da.rio.write_crs(crs)
    return da


# ==========================================================================
# 1. oos_auc -- the key regression test: the leakage
#    assert must fire BEFORE eval_raster/test_pts/background are ever
#    touched (all three are None here -- any code path that dereferences
#    them before the assert would raise TypeError/AttributeError instead
#    of AssertionError, which pytest.raises(AssertionError) would catch as
#    a failure, not a pass).
# ==========================================================================

def test_oos_auc_refuses_resubstitution():
    with pytest.raises(AssertionError):
        evaluate.oos_auc(eval_raster=None, test_pts=None,
                          train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([2]),
                          background=None)


def test_oos_auc_no_overlap_does_not_raise_the_leakage_assert():
    """Disjoint train/test must NOT raise -- proves the guard is a real
    leakage check (fires on bad input, silent on good input), not a
    tautology that always raises regardless of overlap."""
    with pytest.raises(Exception) as exc_info:
        evaluate.oos_auc(eval_raster=None, test_pts=None,
                          train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([9]),
                          background=None)
    # It still fails (eval_raster=None is unusable past the guard), but NOT
    # with the leakage AssertionError -- confirms the guard itself passed.
    assert not isinstance(exc_info.value, AssertionError)


# ==========================================================================
# 2. sample_background -- determinism against a small synthetic raster
#    (no dependency on the real, gitignored predictor stack).
# ==========================================================================

def test_sample_background_same_seed_is_deterministic():
    da = _fake_raster(np.arange(400, dtype=float).reshape(20, 20))
    bg1 = evaluate.sample_background(da, n=50, seed=42)
    bg2 = evaluate.sample_background(da, n=50, seed=42)
    assert list(bg1["x"]) == list(bg2["x"])
    assert list(bg1["y"]) == list(bg2["y"])


def test_sample_background_different_seed_differs():
    da = _fake_raster(np.arange(400, dtype=float).reshape(20, 20))
    bg1 = evaluate.sample_background(da, n=50, seed=42)
    bg2 = evaluate.sample_background(da, n=50, seed=7)
    assert list(bg1["x"]) != list(bg2["x"])


def test_sample_background_excludes_nan_pixels():
    vals = np.arange(100, dtype=float).reshape(10, 10)
    vals[0:3, :] = np.nan   # 30 of 100 pixels invalid
    da = _fake_raster(vals)
    bg = evaluate.sample_background(da, n=70, seed=42)
    sampled = evaluate._sample_xy(da, bg["x"].to_numpy(), bg["y"].to_numpy())
    assert np.all(np.isfinite(sampled)), "background must only land on valid (non-NaN) AOI pixels"


# ==========================================================================
# 3. oos_auc real-fixture tests -- maxent.fit_eval()'s actual holdout_model/flood.tif
#    + the real canonical 277-point/194-train/83-test split.
# ==========================================================================

@pytest.fixture(scope="module")
def real_eval_fixture():
    if not evaluate.EVAL_RASTER.exists():
        pytest.skip(f"{evaluate.EVAL_RASTER} not built locally (run maxent.fit_eval() first)")
    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx]
    background = evaluate.sample_background(evaluate.EVAL_RASTER, n=10_000, seed=config.GLOBAL_SEED)
    return dict(eval_raster=evaluate.EVAL_RASTER, test_pts=test_pts,
                train_idx=train_idx, test_idx=test_idx, background=background)


def test_oos_auc_matches_recorded_train_idx_provenance(real_eval_fixture):
    """maxent.fit_eval() recorded the exact train_idx it fit on to
    holdout_model/train_idx.csv -- confirm it agrees with a FRESH
    config.split() call (the same cross-check test_model_families.py's
    own test_eval_model_never_sees_test performs for fit_eval itself), so
    this test's fixture is provably using the real eval model's actual
    training set, not an assumption about it."""
    recorded = np.loadtxt(maxent.HOLDOUT_MODEL_DIR / "train_idx.csv", skiprows=1, dtype=int)
    assert sorted(recorded.tolist()) == sorted(real_eval_fixture["train_idx"].tolist())


def test_oos_auc_in_expected_band(real_eval_fixture):
    auc = evaluate.oos_auc(**real_eval_fixture)
    print(f"\n[test_oos_auc_in_expected_band] honest out-of-sample AUC = {auc:.6f}")
    # NOTE: an earlier 0.94-0.98 band was sized for the OLD model; this is
    # the rebuilt 17-predictor/TWI/beta=0.5-LQ/277-pt model. The real,
    # honestly-computed value is reported here (not forced to the stale band).
    assert 0.95 < auc < 1.0
    assert auc == pytest.approx(0.982577108433735, abs=5e-4)


def test_oos_auc_deterministic_background_seed(real_eval_fixture):
    """Regenerating background from the SAME seed twice (two independent
    sample_background calls, not the same array reused) must reproduce
    the identical AUC."""
    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx]

    bg1 = evaluate.sample_background(evaluate.EVAL_RASTER, n=10_000, seed=config.GLOBAL_SEED)
    bg2 = evaluate.sample_background(evaluate.EVAL_RASTER, n=10_000, seed=config.GLOBAL_SEED)

    auc1 = evaluate.oos_auc(evaluate.EVAL_RASTER, test_pts, train_idx, test_idx, bg1)
    auc2 = evaluate.oos_auc(evaluate.EVAL_RASTER, test_pts, train_idx, test_idx, bg2)
    assert auc1 == auc2


def test_oos_auc_uses_eval_raster_not_final_raster(real_eval_fixture):
    """The defining resubstitution regression guard: scoring against the FINAL
    (full-277-point) raster at the same test points/background must give
    a DIFFERENT number than scoring against the EVAL (train-only) raster
    -- if these two ever silently coincided it would mean oos_auc had
    been pointed at the full model again."""
    honest = evaluate.oos_auc(**real_eval_fixture)
    bugged = evaluate.oos_auc(
        eval_raster=evaluate.FINAL_RASTER,
        test_pts=real_eval_fixture["test_pts"],
        train_idx=real_eval_fixture["train_idx"],
        test_idx=real_eval_fixture["test_idx"],
        background=real_eval_fixture["background"],
    )
    assert honest != bugged


# ==========================================================================
# 4. apparent_auc -- re-export of maxent.apparent_auc -- never a second,
#    independently-drifting
#    resubstitution-AUC implementation.
# ==========================================================================

def test_apparent_auc_is_maxent_apparent_auc():
    assert evaluate.apparent_auc is maxent.apparent_auc


def test_apparent_auc_real_checkpoint_value():
    if not (maxent.FINAL_DIR / "cloglog" / "maxentResults.csv").exists():
        pytest.skip("final/cloglog/maxentResults.csv not built locally (run the MaxEnt fits first)")
    result = evaluate.apparent_auc()
    assert result["n_replicates"] == 10
    # Training-AUC mean of the 10-replicate bootstrap FINAL model. The MaxEnt
    # bootstrap is unseeded (no CLI seed; see maxent.fit_final docstring), so this
    # drifts run-to-run (observed 0.9675->0.9714, one-directional up). Band-pinned
    # like the other final-model quantities -- the conclusion (near-perfect
    # resubstitution AUC, NEVER the headline number) is invariant.
    assert result["mean"] == pytest.approx(0.9675, abs=0.02)


# ==========================================================================
# 5. threshold_metrics -- ported from the original notebook, verified
#    against a hand-computed 2x2 confusion-matrix toy:
#      y_true  = [1,1,1,0,0,0,0,0,0,0]  (3 presence, 7 background)
#      y_score = [.9,.8,.3,.7,.6,.2,.1,.05,.4,.5]
#      threshold = 0.5 -> y_pred = [1,1,0,1,1,0,0,0,0,1]
#      tp=2 tn=4 fp=3 fn=1 (hand-counted from the table above)
#      sens=2/3=.66667  spec=4/7=.57143  tss=5/21=.238095
#      precision=2/5=.4  f1=2*2/(2*2+3+1)=4/8=.5
#      po=.6 pe=.5 kappa=(.6-.5)/(1-.5)=.2  hss==kappa for a 2x2 table (.2)
#      pred_frac = 5/10 = .5
# ==========================================================================

Y_TRUE = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
Y_SCORE = np.array([.9, .8, .3, .7, .6, .2, .1, .05, .4, .5])


def test_threshold_metrics_hand_computed_confusion_counts():
    m = evaluate.threshold_metrics(Y_TRUE, Y_SCORE, threshold=0.5, label="toy")
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 4, 3, 1)


def test_threshold_metrics_hand_computed_derived_scores():
    m = evaluate.threshold_metrics(Y_TRUE, Y_SCORE, threshold=0.5, label="toy")
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 7)
    assert m["tss"] == pytest.approx(5 / 21)
    assert m["precision"] == pytest.approx(0.4)
    assert m["f1"] == pytest.approx(0.5)
    assert m["kappa"] == pytest.approx(0.2)
    assert m["hss"] == pytest.approx(0.2)          # == kappa for a 2x2 table
    assert m["pred_frac"] == pytest.approx(0.5)


def test_threshold_metrics_extreme_threshold_guards_against_division_by_zero():
    """threshold above every score -> tp=fn... wait tp+fn is presence count,
    always > 0 here; the guard that actually gets exercised is tp+fp==0
    (nothing predicted positive) -> precision/f1 must return 0., not NaN
    or a ZeroDivisionError."""
    m = evaluate.threshold_metrics(Y_TRUE, Y_SCORE, threshold=1.5, label="above-all")
    assert m["tp"] == 0 and m["fp"] == 0
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0
    assert m["sensitivity"] == 0.0


def test_threshold_metrics_below_all_scores_perfect_sensitivity():
    m = evaluate.threshold_metrics(Y_TRUE, Y_SCORE, threshold=-1.0, label="below-all")
    assert m["tp"] == 3 and m["tn"] == 0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 0.0


# ==========================================================================
# 6. Threshold sourcing -- MaxSS / P10 from the eval model's own (single-
#    row, non-bootstrapped) maxentResults.csv, matching the original
#    notebook's column names exactly; plus the ROC-based MaxSS
#    cross-check/fallback (Youden's J, algebraically identical to
#    maximizing TSS = sens+spec-1).
# ==========================================================================

def test_maxss_threshold_from_results_reads_real_eval_csv():
    if not evaluate.EVAL_RESULTS_CSV.exists():
        pytest.skip(f"{evaluate.EVAL_RESULTS_CSV} not built locally (run the MaxEnt fits first)")
    t = evaluate.maxss_threshold_from_results(evaluate.EVAL_RESULTS_CSV)
    assert 0.0 < t < 1.0   # a cloglog threshold must be a valid probability


def test_p10_threshold_from_results_reads_real_eval_csv():
    if not evaluate.EVAL_RESULTS_CSV.exists():
        pytest.skip(f"{evaluate.EVAL_RESULTS_CSV} not built locally (run the MaxEnt fits first)")
    t = evaluate.p10_threshold_from_results(evaluate.EVAL_RESULTS_CSV)
    assert 0.0 < t < 1.0


def test_maxss_threshold_from_roc_matches_manual_youden_j():
    # A small toy where Youden's J = tpr-fpr is unambiguously maximized at
    # one particular sklearn roc_curve threshold -- hand-checkable: the
    # perfect separator at 0.5 has tpr=1, fpr=0 -> J=1, the best possible.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    t = evaluate.maxss_threshold_from_roc(y_true, y_score)
    m = evaluate.threshold_metrics(y_true, y_score, threshold=t)
    assert m["tss"] == pytest.approx(1.0)   # perfect separation is achievable here


# ==========================================================================
# 7. success_prediction_rate -- pure numeric core hand-computed:
#      valid_vals      = [0..9]           (10 "AOI pixels")
#      presence_scores = [2, 5, 8]        (3 "test presence" points)
#      thresholds      = [0, 3, 6, 9, 10]
#    success_rate(t)    = mean(presence_scores >= t)
#    prediction_rate(t) = mean(valid_vals >= t)
# ==========================================================================

def test_success_prediction_rate_arrays_hand_computed():
    valid_vals = np.arange(10, dtype=float)
    presence_scores = np.array([2.0, 5.0, 8.0])
    thresholds = np.array([0.0, 3.0, 6.0, 9.0, 10.0])

    success_rate, prediction_rate = evaluate._success_prediction_rate_arrays(
        valid_vals, presence_scores, thresholds
    )

    assert success_rate == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0, 0.0])
    assert prediction_rate == pytest.approx([1.0, 0.7, 0.4, 0.1, 0.0])


def test_success_prediction_rate_is_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    valid_vals = rng.uniform(0, 1, size=5000)
    presence_scores = rng.uniform(0, 1, size=50)
    thresholds = np.linspace(0, 1, 100)
    success_rate, prediction_rate = evaluate._success_prediction_rate_arrays(
        valid_vals, presence_scores, thresholds
    )
    assert np.all(np.diff(success_rate) <= 1e-12)
    assert np.all(np.diff(prediction_rate) <= 1e-12)


def test_success_prediction_rate_writes_png(tmp_path, real_eval_fixture):
    out_path = tmp_path / "success_prediction_rate.png"
    df = evaluate.success_prediction_rate(
        evaluate.EVAL_RASTER, real_eval_fixture["test_pts"], out_path=out_path
    )
    assert out_path.exists() and out_path.stat().st_size > 0
    assert {"threshold", "success_rate", "prediction_rate"} <= set(df.columns)
    assert df["success_rate"].between(0, 1).all()
    assert df["prediction_rate"].between(0, 1).all()


# ==========================================================================
# 8. classed_map -- pure quantile-binning core hand-computed:
#      vals = [0..9], n_classes=5 -> quantile breaks (numpy linear
#      interpolation) = [0, 1.8, 3.6, 5.4, 7.2, 9] -- a perfectly even
#      2-2-2-2-2 split (worked by hand).
# ==========================================================================

def test_classify_values_quantile_hand_computed_breaks_and_counts():
    vals = np.arange(10, dtype=float)
    breaks, counts_per_class = evaluate._classify_values(vals, n_classes=5, method="quantile")
    assert breaks == pytest.approx([0.0, 1.8, 3.6, 5.4, 7.2, 9.0])
    assert counts_per_class.tolist() == [2, 2, 2, 2, 2]


def test_classify_values_rejects_unknown_method():
    with pytest.raises(ValueError):
        evaluate._classify_values(np.arange(10, dtype=float), n_classes=5, method="bogus")


def test_classed_map_rejects_mismatched_labels():
    with pytest.raises(AssertionError):
        evaluate.classed_map(final_raster=_fake_raster(np.arange(100, dtype=float).reshape(10, 10)),
                              n_classes=5, labels=["only", "two"])


def test_classed_map_writes_png(tmp_path):
    if not evaluate.FINAL_RASTER.exists():
        pytest.skip(f"{evaluate.FINAL_RASTER} not built locally (run the MaxEnt fits first)")
    out_path = tmp_path / "classed_map.png"
    breaks, class_counts = evaluate.classed_map(out_path=out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
    assert len(breaks) == 6          # n_classes=5 default -> 6 bin edges
    assert len(class_counts) == 5
    assert sum(class_counts.values()) > 0


# ==========================================================================
# 9. internal_validation_table -- the convenience composition the report
#    needs (MaxSS + P10 rows over the honest test set) -- built entirely
#    from the primitives above, not a new independently-scored path.
# ==========================================================================

def test_internal_validation_table_real_checkpoint(real_eval_fixture):
    df = evaluate.internal_validation_table(
        eval_raster=real_eval_fixture["eval_raster"],
        test_pts=real_eval_fixture["test_pts"],
        background=real_eval_fixture["background"],
    )
    assert set(df["label"]) == {"MaxSS", "10th percentile"}
    for col in ("sensitivity", "specificity", "tss", "f1", "kappa", "hss"):
        assert df[col].between(-1, 1).all()

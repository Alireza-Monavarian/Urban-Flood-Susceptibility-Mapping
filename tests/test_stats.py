"""Tests for pipeline.robustness -- matched-threshold stats
(AUPRC/Brier/bootstrap-CI/Moran's I) + two fixes on top of model_compare's
matched test set and the spatial-CV/LOEO per-fold AUCs.

Ports the original notebook 04c_robustness_checks, Sections 13 (Moran's I) and 15-18
(AUPRC, calibration/Brier, bootstrap CI, Wilcoxon+McNemar), with the two
fixes documented in robustness.py's own matched-threshold section banner:

1. **THE THRESHOLD FIX** -- McNemar (and any other binary/threshold metric
   comparing models) binarizes at EACH model's OWN MaxSS threshold, never a
   fixed 0.5.
2. **THE SIGNIFICANCE FIX** -- the Nadeau-Bengio (2003) corrected resampled
   t-test replaces 04c's Section 18a Wilcoxon-at-its-own-p-floor test.

Two tiers, mirroring test_model_compare.py's/test_robustness_folds.py's own
established convention:

1. Pure-logic / synthetic-data tests for ``nadeau_bengio_corrected_t``,
   ``bootstrap_auc_ci``, ``mcnemar_pairs``'s own-threshold behavior, and
   ``morans_i_residuals``'s core spatial-autocorrelation math against
   synthetic clustered/random point sets -- fast, no dependency on the real,
   gitignored predictor stack or MaxEnt artifacts.
2. Real end-to-end tests against the actual on-disk canonical
   stack + ``holdout_model/flood.tif`` + the cached
   spatial-CV/LOEO fold rasters (matching test_robustness_folds.py's own
   tier-2 precedent: NOT gated behind a skip -- they assume the full run
   has been done with local data already built). These PIN the
   ACTUAL observed values -- never a value assumed in advance.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import config, evaluate, model_compare, robustness

# ============================================================================
# Tier 1 -- pure logic / synthetic data, no real pipeline data needed.
# ============================================================================

# ----------------------------------------------------------------------
# THE SIGNIFICANCE FIX -- no Wilcoxon anywhere in this module, ever.
# ----------------------------------------------------------------------

def test_significance_test_is_not_wilcoxon_at_p_floor():
    """Static regression guard: the old Section-18 Wilcoxon-at-p-floor test
    must never be reinstated anywhere in robustness.py, under any name.
    Checks for an actual import/call (``wilcoxon(`` or a scipy.stats
    ``wilcoxon`` import) rather than a bare substring match, since this
    module's own docstrings legitimately NAME "Wilcoxon" in prose when
    explaining what THE SIGNIFICANCE FIX replaced."""
    import inspect
    source = inspect.getsource(robustness)
    assert "wilcoxon(" not in source.lower(), (
        "robustness.py must never CALL scipy.stats.wilcoxon -- the "
        "Nadeau-Bengio corrected t-test replaces it entirely (THE "
        "SIGNIFICANCE FIX)"
    )
    assert "import wilcoxon" not in source.lower(), (
        "robustness.py must never IMPORT scipy.stats.wilcoxon"
    )


def test_significance_test_method_name_is_regression_guarded():
    """The required assertion shape: the reported method name
    must be one of the two sanctioned replacements for Wilcoxon-at-p-floor."""
    result = robustness.nadeau_bengio_corrected_t(np.array([0.01, 0.02, -0.01]), n_train=250, n_test=27)
    assert result["method"] in ("nadeau_bengio_corrected_t", "effect_size_ci")


# ----------------------------------------------------------------------
# nadeau_bengio_corrected_t -- the Nadeau-Bengio (2003) correction itself.
# ----------------------------------------------------------------------

def test_nadeau_bengio_corrected_t_matches_hand_computed_example():
    """Cross-checked against an independently hand-derived value (mean_d,
    sd_d, the 1/k + n_test/n_train correction, se, t, p computed directly
    with scipy.stats.t rather than by calling the function under test)."""
    diffs = np.array([0.05, 0.04, 0.06, 0.05, 0.03])
    result = robustness.nadeau_bengio_corrected_t(diffs, n_train=250.0, n_test=27.0)

    assert result["method"] == "nadeau_bengio_corrected_t"
    assert result["k"] == 5
    assert result["df"] == 4
    assert result["mean_diff"] == pytest.approx(0.046, abs=1e-9)
    assert result["sd_diff"] == pytest.approx(0.011401754250991379, abs=1e-9)
    assert result["t_stat"] == pytest.approx(7.2696047242716615, abs=1e-6)
    assert result["p_value"] == pytest.approx(0.0019020225313715056, abs=1e-6)


def test_nadeau_bengio_corrected_t_zero_diffs_gives_p_one_not_error():
    """All-zero paired differences must give t=0, p=1 (no effect, no
    significance) -- not a division-by-zero error or a NaN."""
    result = robustness.nadeau_bengio_corrected_t(np.zeros(3), n_train=250, n_test=27)
    assert result["t_stat"] == 0.0
    assert result["p_value"] == pytest.approx(1.0, abs=1e-12)


def test_nadeau_bengio_corrected_t_requires_at_least_two_folds():
    with pytest.raises(AssertionError):
        robustness.nadeau_bengio_corrected_t(np.array([0.1]), n_train=250, n_test=27)


def test_nadeau_bengio_larger_test_train_ratio_widens_interval():
    """THE point of the correction: a larger n_test/n_train ratio (folds
    whose training sets overlap MORE, relatively) must inflate the
    estimated variance -- smaller |t|, larger p -- for the IDENTICAL
    diffs. Proves this is not a no-op dressed up as a new name."""
    diffs = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.045, 0.055, 0.02, 0.04, 0.05, 0.06])
    small_ratio = robustness.nadeau_bengio_corrected_t(diffs, n_train=1000, n_test=1)
    large_ratio = robustness.nadeau_bengio_corrected_t(diffs, n_train=100, n_test=100)
    assert abs(large_ratio["t_stat"]) < abs(small_ratio["t_stat"])
    assert large_ratio["p_value"] > small_ratio["p_value"]


# ----------------------------------------------------------------------
# bootstrap_auc_ci -- pure-logic structure/bounds check.
# ----------------------------------------------------------------------

def test_bootstrap_auc_ci_structure_and_bounds():
    rng = np.random.default_rng(7)
    y_true = np.array([1] * 15 + [0] * 60)
    y_score = np.concatenate([rng.normal(0.6, 0.25, 15), rng.normal(0.35, 0.25, 60)])

    result = robustness.bootstrap_auc_ci(y_true, y_score, n_boot=500, seed=11)

    assert result["point_auc"] == pytest.approx(0.8266666666666667, abs=1e-9)
    assert result["ci_lo"] == pytest.approx(0.719688814849363, abs=1e-6)
    assert result["ci_hi"] == pytest.approx(0.9155348890143832, abs=1e-6)
    assert 0.0 <= result["ci_lo"] < result["point_auc"] < result["ci_hi"] <= 1.0
    assert result["n_boot_used"] <= result["n_boot_requested"] == 500
    assert result["ci_level"] == 0.95


def test_bootstrap_auc_ci_deterministic_for_fixed_seed():
    rng = np.random.default_rng(3)
    y_true = np.array([1] * 10 + [0] * 40)
    y_score = np.concatenate([rng.normal(0.7, 0.2, 10), rng.normal(0.3, 0.2, 40)])
    r1 = robustness.bootstrap_auc_ci(y_true, y_score, n_boot=200, seed=5)
    r2 = robustness.bootstrap_auc_ci(y_true, y_score, n_boot=200, seed=5)
    assert r1 == r2


def test_bootstrap_auc_ci_skips_single_class_resamples():
    """A near-degenerate class balance (1 positive in 30) makes SOME
    bootstrap resamples draw zero positives -- those must be skipped
    (n_boot_used < n_boot_requested), never crash roc_auc_score."""
    y_true = np.array([1] + [0] * 29)
    y_score = np.linspace(0, 1, 30)
    result = robustness.bootstrap_auc_ci(y_true, y_score, n_boot=200, seed=1)
    assert result["n_boot_used"] < result["n_boot_requested"]
    assert result["n_boot_used"] > 0


# ----------------------------------------------------------------------
# mcnemar_pairs -- THE THRESHOLD FIX, isolated on synthetic scores.
# ----------------------------------------------------------------------

def test_mcnemar_pairs_uses_each_models_own_threshold():
    """Two models share the IDENTICAL score array but different
    thresholds -- any nonzero b/c can ONLY come from the threshold
    difference (a shared fixed cutoff on identical scores would give
    IDENTICAL predictions, hence b=c=0 necessarily)."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    scores = {
        "A": dict(y_true=y_true, y_score=score, threshold=0.5),
        "B": dict(y_true=y_true, y_score=score, threshold=0.03),
    }

    df = robustness.mcnemar_pairs(scores=scores)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["threshold_a"] == pytest.approx(0.5)
    assert row["threshold_b"] == pytest.approx(0.03)

    # Independently recomputed expected b/c (own-threshold binarization).
    pred_a = (score >= 0.5).astype(int)
    pred_b = (score >= 0.03).astype(int)
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    expected_b = int(np.sum(ok_a & ~ok_b))
    expected_c = int(np.sum(~ok_a & ok_b))
    assert (row["b"], row["c"]) == (expected_b, expected_c)
    assert (expected_b, expected_c) != (0, 0), (
        "a fixed shared threshold on identical scores would force b=c=0 -- "
        "this nonzero result proves the per-model threshold was actually used"
    )
    assert row["chi2"] == pytest.approx(0.8, abs=1e-9)


def test_mcnemar_pairs_rejects_mismatched_test_sets():
    scores = {
        "A": dict(y_true=np.array([1, 0, 0]), y_score=np.array([0.9, 0.1, 0.2]), threshold=0.5),
        "B": dict(y_true=np.array([1, 0, 1]), y_score=np.array([0.9, 0.1, 0.2]), threshold=0.5),
    }
    with pytest.raises(AssertionError):
        robustness.mcnemar_pairs(scores=scores)


# ----------------------------------------------------------------------
# morans_i_residuals -- core spatial-autocorrelation math on synthetic
# clustered vs. spatially-random residuals (no real raster/point data).
# ----------------------------------------------------------------------

def test_morans_i_detects_synthetic_spatial_clustering():
    rng = np.random.default_rng(0)
    n_per_cluster = 40
    x1 = rng.normal(0, 10, n_per_cluster)
    y1 = rng.normal(0, 10, n_per_cluster)
    x2 = rng.normal(10000, 10, n_per_cluster)
    y2 = rng.normal(10000, 10, n_per_cluster)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    # Two tight, well-separated clusters, opposite-sign residual in each --
    # textbook positive spatial autocorrelation.
    residual = np.concatenate([np.ones(n_per_cluster), -np.ones(n_per_cluster)])
    scores = {"synthetic": dict(y_true=np.zeros(2 * n_per_cluster), y_score=residual, x=x, y=y)}

    result = robustness.morans_i_residuals(scores=scores, model="synthetic", k=8, n_perm=199, seed=1)

    assert result["observed_I"] > 0.5
    assert result["p_value"] < 0.05
    assert result["verdict"] == "clustered"


def test_morans_i_null_on_spatially_random_residuals():
    rng = np.random.default_rng(2)
    n = 80
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    residual = rng.normal(0, 1, n)   # iid noise -- no spatial structure at all
    scores = {"synthetic": dict(y_true=np.zeros(n), y_score=residual, x=x, y=y)}

    result = robustness.morans_i_residuals(scores=scores, model="synthetic", k=8, n_perm=199, seed=1)

    assert abs(result["observed_I"]) < 0.3
    assert result["verdict"] == "spatially_random"


def test_morans_i_result_has_observed_p_and_verdict():
    """Structural check: the result must carry observed I + p-value + a
    spatial-randomness verdict, always."""
    rng = np.random.default_rng(9)
    n = 30
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    residual = rng.normal(0, 1, n)
    scores = {"m": dict(y_true=np.zeros(n), y_score=residual, x=x, y=y)}
    result = robustness.morans_i_residuals(scores=scores, model="m", k=5, n_perm=99, seed=0)
    assert set(["observed_I", "expected_I", "p_value", "verdict", "n", "k", "n_perm"]) <= set(result.keys())
    assert result["verdict"] in ("clustered", "spatially_random")
    assert -1.0 <= result["observed_I"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0


# ============================================================================
# Tier 2 -- real on-disk run: the actual matched-test-set model fits
# (5 ML comparators refit here + MaxEnt's existing, NOT refit, eval raster)
# and the real 11-event LOEO XGBoost refit. `_matched_test_scores` is
# in-memory-cached (see its own docstring), so every test below that needs
# it after the FIRST call is fast within one pytest process.
# ============================================================================

@pytest.fixture(scope="module")
def real_scores():
    return robustness._matched_test_scores()


@pytest.fixture(scope="module")
def real_significance():
    return robustness.loeo_significance_test()


# ----------------------------------------------------------------------
# The no-fixed-0.5-threshold regression test.
# ----------------------------------------------------------------------

def test_no_fixed_half_threshold():
    from pipeline import robustness
    thr = robustness._binarization_thresholds()     # model -> threshold used
    assert all(abs(t - 0.5) > 1e-9 for t in thr.values())   # per-model MaxSS, not 0.5


def test_binarization_thresholds_covers_all_six_models_and_matches_model_compare():
    """Pins the ACTUAL observed thresholds -- identical (to 4 decimals) to
    model_compare's own reported table,
    confirming this module's from-scratch refit reproduces the exact same
    per-model MaxSS operating points, never a re-derivation that happens to
    merely look similar."""
    thr = robustness._binarization_thresholds()
    assert set(thr.keys()) == {
        "MaxEnt (reference)", "Logistic Regression", "Random Forest",
        "XGBoost", "LightGBM", "CatBoost",
    }
    assert thr["MaxEnt (reference)"] == pytest.approx(0.0842, abs=5e-4)
    assert thr["Logistic Regression"] == pytest.approx(0.0421, abs=5e-4)
    assert thr["Random Forest"] == pytest.approx(0.7996, abs=5e-4)
    # XGBoost/CatBoost re-pinned 2026-08-12 to a clean rebuild under the
    # pinned environment (env/RUNTIME.md); xgboost 3.3.0 shifts both models'
    # MaxSS operating point. The other four are unchanged.
    assert thr["XGBoost"] == pytest.approx(0.6721, abs=5e-4)
    assert thr["LightGBM"] == pytest.approx(0.3704, abs=5e-4)
    assert thr["CatBoost"] == pytest.approx(0.6571, abs=5e-4)
    print("\n[test_binarization_thresholds_covers_all_six_models_and_matches_model_compare] thresholds:")
    for m, t in sorted(thr.items()):
        print(f"  {m:<22}: {t:.4f}")


def test_matched_scores_agree_with_model_compare_run(real_scores):
    """Cross-check: _matched_test_scores' own from-scratch refit must
    reproduce model_compare.run()'s own tested AUC per model -- proves this
    module's necessary duplication of run()'s train/test assembly (see
    robustness.py's own matched-threshold banner) has not silently drifted from
    the ACTUAL tested model_compare recipe. Observed: bit-identical to 6
    decimals on this machine."""
    from sklearn.metrics import roc_auc_score

    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    background = model_compare.build_background()
    official = model_compare.run(train_idx, test_idx, background).set_index("model")["auc"]

    for model, d in real_scores.items():
        my_auc = roc_auc_score(d["y_true"], d["y_score"])
        assert my_auc == pytest.approx(official[model], abs=1e-4), (
            f"{model}: recomputed AUC {my_auc:.6f} vs model_compare.run()'s {official[model]:.6f}"
        )


def test_imbalance_metrics_shape_and_bounds_pinned(real_scores):
    """Pins the ACTUAL observed AUPRC/Brier for all 6 models (MaxEnt at
    minimum)."""
    df = robustness.imbalance_metrics(scores=real_scores).set_index("model")
    assert len(df) == 6
    assert df["auprc"].between(0, 1).all()
    assert df["brier"].between(0, 1).all()

    assert df.loc["MaxEnt (reference)", "auprc"] == pytest.approx(0.669129, abs=5e-3)
    assert df.loc["MaxEnt (reference)", "brier"] == pytest.approx(0.014564, abs=5e-3)
    print("\n[test_imbalance_metrics_shape_and_bounds_pinned] AUPRC/Brier/threshold:")
    print(df.to_string())


def test_bootstrap_auc_ci_all_models_pinned(real_scores):
    df = robustness.bootstrap_auc_ci_all_models(scores=real_scores, n_boot=1000).set_index("model")
    assert len(df) == 6
    for model, row in df.iterrows():
        assert row["ci_lo"] <= row["point_auc"] + 1e-6 <= row["ci_hi"] + 1e-6

    mx = df.loc["MaxEnt (reference)"]
    assert mx["point_auc"] == pytest.approx(0.984269, abs=5e-4)
    assert mx["ci_lo"] == pytest.approx(0.977031, abs=5e-3)
    assert mx["ci_hi"] == pytest.approx(0.990538, abs=5e-3)
    print("\n[test_bootstrap_auc_ci_all_models_pinned] bootstrap 95% CIs:")
    print(df.to_string())


def test_mcnemar_pairs_real_data_uses_binarization_thresholds(real_scores):
    thr = robustness._binarization_thresholds()
    df = robustness.mcnemar_pairs(scores=real_scores)
    assert len(df) == 15   # C(6,2) pairs
    for _, row in df.iterrows():
        a, b = row["pair"].split(" vs ")
        assert row["threshold_a"] == pytest.approx(thr[a])
        assert row["threshold_b"] == pytest.approx(thr[b])
        assert abs(row["threshold_a"] - 0.5) > 1e-9
        assert abs(row["threshold_b"] - 0.5) > 1e-9

    # Re-pinned 2026-08-12 to a clean rebuild (xgboost 3.3.0 shifts XGBoost's
    # MaxSS threshold 0.790 -> 0.672, which moves its discordant counts).
    mx_vs_xgb = df[df["pair"] == "XGBoost vs MaxEnt (reference)"].iloc[0]
    assert mx_vs_xgb["b"] == 161
    assert mx_vs_xgb["c"] == 5
    print("\n[test_mcnemar_pairs_real_data_uses_binarization_thresholds] McNemar pairs:")
    print(df.to_string(index=False))


def test_morans_i_residuals_real_maxent_pinned(real_scores):
    """Pins the ACTUAL observed Moran's I on MaxEnt's real matched-test-set
    residuals: modest positive spatial autocorrelation (I~0.15, far from
    the synthetic worst-case I=1.0 in the Tier-1 test above), but still
    clearly significant at 999 permutations (p=0.001, the floor at this
    permutation count) -- reported honestly, not forced toward either
    "clean" or "broken". Caveat: background points, ~36x the
    presence count here, dominate this number."""
    result = robustness.morans_i_residuals(scores=real_scores, n_perm=999)
    assert result["model"] == "MaxEnt (reference)"
    assert result["n"] == 3083
    assert -1.0 <= result["observed_I"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["verdict"] in ("clustered", "spatially_random")

    assert result["observed_I"] == pytest.approx(0.1509, abs=5e-3)
    assert result["p_value"] == pytest.approx(0.001, abs=2e-3)
    assert result["verdict"] == "clustered"
    print(f"\n[test_morans_i_residuals_real_maxent_pinned] observed_I={result['observed_I']:+.4f} "
          f"expected_I={result['expected_I']:+.4f} p={result['p_value']:.4f} verdict={result['verdict']}")


def test_loeo_significance_test_reports_nadeau_bengio(real_significance):
    """Pins the ACTUAL observed significance-test result: MaxEnt vs.
    XGBoost (the AUC leader in model_compare) across the 11 real LOEO event-folds --
    p=0.23, nowhere near 0.05, i.e. NOT a real difference under the
    corrected test (contrast with 04c's own Wilcoxon-at-p-floor, which
    would have reported p=0.0078 as "significant" on an analogous 8-fold
    comparison purely from folds-too-few, never from this effect size)."""
    result = real_significance
    assert result["method"] == "nadeau_bengio_corrected_t"
    assert result["k"] == 11
    assert result["second_model"] == "XGBoost"
    assert result["verdict"] in ("real_difference", "within_noise")

    # Re-pinned 2026-08-12 to a clean rebuild under the pinned environment
    # (env/RUNTIME.md). The previous values (p=0.53) came from an unrecorded
    # XGBoost version; xgboost 3.3.0 shifts the per-event AUCs and hence the
    # paired differences. The VERDICT is unchanged -- within_noise either way --
    # which is the claim the paper rests on.
    assert result["mean_diff"] == pytest.approx(-0.0109136597627861, abs=1e-6)
    assert result["t_stat"] == pytest.approx(-1.2841043778380234, abs=1e-6)
    assert result["p_value"] == pytest.approx(0.2280577602018274, abs=1e-6)
    assert result["cohens_d"] == pytest.approx(-0.5610655861583898, abs=1e-6)
    assert result["verdict"] == "within_noise"
    print(f"\n[test_loeo_significance_test_reports_nadeau_bengio] "
          f"mean_diff={result['mean_diff']:+.4f} cohens_d={result['cohens_d']:+.3f} "
          f"t={result['t_stat']:.3f} df={result['df']} p={result['p_value']:.4f} "
          f"verdict={result['verdict']}")


def test_loeo_significance_events_match_loeo_membership(real_significance):
    """The paired comparison must run over the SAME 11-event fold
    structure loeo()/loeo_membership() use for MaxEnt -- never a
    different partition for the second model."""
    membership = robustness.loeo_membership(config.canonical_samples())
    assert set(real_significance["events"]) == set(membership.keys())
    assert len(real_significance["maxent_aucs"]) == len(real_significance["second_model_aucs"]) == 11

"""Tests for pipeline.maxent's pure-logic AICc helpers -- ``_aicc``,
``read_lambdas_k``, ``aicc_from_raw``, ``select_best`` -- plus a couple of
constant/shape regression guards for the tuning grid (15 candidates as of
the log-transform adoption -- originally 12).

Deliberately fast and I/O-free (no MaxEnt subprocess, no raster reads): this
file is scoped to "pure-logic" coverage of the AICc arithmetic and
its two feeder helpers, each validated against a hand-computable ground
truth. The real grid search (``pipeline.maxent.aicc_grid``, which DOES
invoke the real MaxEnt subprocess against the real canonical predictor
stack -- 17 layers as of the log-transform adoption, previously 19) is
run once via a standalone script -- not a pytest test -- and its actual
result is written to ``data/processed/maxent/tuning/aicc_results.csv``,
not asserted here as a pinned literal (unlike ``test_screen.py``'s ``EXPECTED_KEPT``, there is no reason
to re-run a ~15-minute subprocess grid on every ``pytest`` invocation just
to re-confirm arithmetic these unit tests already pin directly).

The required toy -- ``_aicc(loglik=-10.008848, k=2, n=5)``
== 30.017696 -- is the same hand-computed example the original notebook
``04_maxent``
§2 validates its own ``aicc_from_raw`` against:
    loglik = ln(.10)+ln(.20)+ln(.05)+ln(.30)+ln(.15) = -10.008848
    AIC    = 2*2 - 2*(-10.008848)                    = 24.017696
    AICc   = AIC + 2*2*3/(5-2-1)                     = 24.017696 + 6 = 30.017696
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import maxent

TOY_K = 2
TOY_RAW = np.array([0.10, 0.20, 0.05, 0.30, 0.15])
TOY_LOGLIK = -10.008848
TOY_AICC = 30.017696347449302


# --------------------------------------------------------------------------
# _aicc() -- the hand-computed toy
# --------------------------------------------------------------------------

def test_aicc_toy_matches_hand_value():
    assert abs(maxent._aicc(loglik=-10.008848, k=2, n=5) - 30.017696) < 1e-5


def test_aicc_matches_full_precision_hand_value():
    assert maxent._aicc(loglik=TOY_LOGLIK, k=TOY_K, n=5) == pytest.approx(TOY_AICC, abs=1e-6)


def test_aicc_returns_inf_when_denominator_nonpositive():
    """n - k - 1 <= 0 (too few points for the parameter count) must return
    +inf rather than divide-by-zero/negative-denominator garbage -- mirrors
    ``aicc_from_raw``'s own guard in the source notebook."""
    assert maxent._aicc(loglik=-1.0, k=4, n=5) == float("inf")   # denom == 0
    assert maxent._aicc(loglik=-1.0, k=10, n=5) == float("inf")  # denom < 0


def test_aicc_increases_with_k_holding_loglik_n_fixed():
    """More parameters should never IMPROVE (lower) AICc when the fit
    (loglik) is held artificially constant -- the complexity penalty is
    monotone increasing in k over this range."""
    low_k = maxent._aicc(loglik=-50.0, k=2, n=100)
    high_k = maxent._aicc(loglik=-50.0, k=10, n=100)
    assert high_k > low_k


# --------------------------------------------------------------------------
# aicc_from_raw() -- loglik = sum(log(raw)), then _aicc -- the same toy,
# validated end-to-end from raw predictions rather than a precomputed loglik
# --------------------------------------------------------------------------

def test_aicc_from_raw_toy_matches_hand_value():
    computed = maxent.aicc_from_raw(TOY_K, TOY_RAW)
    assert computed == pytest.approx(TOY_AICC, abs=1e-6)


def test_aicc_from_raw_matches_aicc_directly():
    """aicc_from_raw must be exactly _aicc(sum(log(raw)), k, len(raw)) --
    no independent arithmetic path that could silently diverge."""
    loglik = np.sum(np.log(TOY_RAW))
    expected = maxent._aicc(loglik=loglik, k=TOY_K, n=len(TOY_RAW))
    assert maxent.aicc_from_raw(TOY_K, TOY_RAW) == pytest.approx(expected)


def test_aicc_from_raw_accepts_plain_list():
    """Real callers pass a pandas Series (.values) from a samplePredictions
    CSV column; a plain list must work identically to the pinned ndarray."""
    assert maxent.aicc_from_raw(TOY_K, list(TOY_RAW)) == pytest.approx(TOY_AICC, abs=1e-6)


# --------------------------------------------------------------------------
# read_lambdas_k() -- feature-parameter count, excluding the 4 metadata rows
# --------------------------------------------------------------------------

def _write_lambdas(path, rows):
    """rows: list of (name, a, b, c) tuples -- mirrors a real .lambdas
    file's headerless 4-column CSV layout."""
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_read_lambdas_k_excludes_metadata_rows(tmp_path):
    """3 real feature rows (2 non-zero, 1 exactly zero) + the 4 metadata
    rows (linearPredictorNormalizer/densityNormalizer/numBackgroundPoints/
    entropy) -- deliberately given large NON-zero 'a' values here too, so a
    naive `(a != 0).sum()` over the whole file (without the name-based
    exclusion) would wrongly return 6 instead of the correct 2."""
    path = tmp_path / "flood.lambdas"
    _write_lambdas(path, [
        ("slope", 0.42, 1.0, 2.0),
        ("nlcd_landcover=21", 0.0, 1.0, 2.0),          # zero coefficient -- not counted
        ("dem", -0.13, 1.0, 2.0),
        ("linearPredictorNormalizer", 1.23, 0.0, 0.0),  # metadata -- excluded regardless of value
        ("densityNormalizer", 4.56, 0.0, 0.0),
        ("numBackgroundPoints", 10000.0, 0.0, 0.0),
        ("entropy", 2.34, 0.0, 0.0),
    ])
    assert maxent.read_lambdas_k(path) == 2


def test_read_lambdas_k_all_zero_features_is_zero(tmp_path):
    path = tmp_path / "flood.lambdas"
    _write_lambdas(path, [
        ("slope", 0.0, 1.0, 2.0),
        ("dem", 0.0, 1.0, 2.0),
        ("linearPredictorNormalizer", 1.23, 0.0, 0.0),
        ("densityNormalizer", 4.56, 0.0, 0.0),
        ("numBackgroundPoints", 10000.0, 0.0, 0.0),
        ("entropy", 2.34, 0.0, 0.0),
    ])
    assert maxent.read_lambdas_k(path) == 0


def test_read_lambdas_k_counts_negative_coefficients_too(tmp_path):
    """A negative lambda is still a NON-zero (active) feature -- only
    exactly-zero coefficients are excluded from k."""
    path = tmp_path / "flood.lambdas"
    _write_lambdas(path, [
        ("hand", -0.87, 1.0, 2.0),
        ("linearPredictorNormalizer", 1.23, 0.0, 0.0),
        ("densityNormalizer", 4.56, 0.0, 0.0),
        ("numBackgroundPoints", 10000.0, 0.0, 0.0),
        ("entropy", 2.34, 0.0, 0.0),
    ])
    assert maxent.read_lambdas_k(path) == 1


# --------------------------------------------------------------------------
# select_best() -- lowest-AICc row, ignoring failed candidates
# --------------------------------------------------------------------------

def _toy_grid():
    # n=277 -- the canonical pixel-unique modeling count; purely illustrative for
    # these pure-logic select_best() tests (any n works), updated so a
    # future reader doesn't mistake the old 279 placeholder for a
    # currently-meaningful number.
    return pd.DataFrame([
        {"beta": 0.5, "feature_classes": "L",    "k": 10, "n": 277, "AICc": 3800.0, "status": "ok"},
        {"beta": 1.0, "feature_classes": "L",    "k": 12, "n": 277, "AICc": 3750.5, "status": "ok"},
        {"beta": 1.0, "feature_classes": "LQ",   "k": 15, "n": 277, "AICc": 3760.0, "status": "ok"},
        {"beta": 2.0, "feature_classes": "LQHP", "k": 40, "n": 277, "AICc": np.inf, "status": "failed"},
    ])


def test_select_best_returns_lowest_aicc_row():
    best = maxent.select_best(_toy_grid())
    assert best["beta"] == 1.0
    assert best["feature_classes"] == "L"
    assert best["k"] == 12
    assert best["AICc"] == pytest.approx(3750.5)


def test_select_best_ignores_failed_candidates_even_if_lower_aicc_slipped_in():
    """A 'failed' row must never win even if its AICc field were somehow not
    inf (e.g. a stale/partial value) -- status == 'ok' gates eligibility,
    not AICc alone."""
    grid = _toy_grid()
    grid.loc[grid["status"] == "failed", "AICc"] = 1.0  # artificially "best" but failed
    best = maxent.select_best(grid)
    assert best["status"] == "ok"
    assert best["feature_classes"] == "L" and best["beta"] == 1.0


def test_select_best_raises_when_grid_is_all_failed():
    grid = _toy_grid()
    grid["status"] = "failed"
    with pytest.raises(ValueError):
        maxent.select_best(grid)


# --------------------------------------------------------------------------
# Tuning-grid shape constants -- originally "12 candidates" (4
# betas x 3 feature classes); the log-transform adoption added a 5th beta (0.25, a diagnostic
# check that the beta=0.5/LQ winner found on the log-transformed stack
# isn't merely sitting at the low edge of the original grid) for 15
# candidates -- pinned as a pure regression guard (no subprocess I/O).
# --------------------------------------------------------------------------

def test_grid_constants_define_15_candidates():
    assert sorted(maxent.BETAS) == [0.25, 0.5, 1, 2, 3]
    assert set(maxent.FEATURE_CLASSES) == {"L", "LQ", "LQHP"}
    assert len(maxent.BETAS) * len(maxent.FEATURE_CLASSES) == 15


def test_feature_classes_have_the_five_maxent_flags():
    expected_keys = {"linear", "quadratic", "product", "threshold", "hinge"}
    for name, flags in maxent.FEATURE_CLASSES.items():
        assert set(flags) == expected_keys, name


def test_feature_classes_are_nested_l_lq_lqhp():
    """L subset of LQ subset of LQHP, matching the source notebook's grid
    design (each class only ADDS feature types, never removes one)."""
    l, lq, lqhp = maxent.FEATURE_CLASSES["L"], maxent.FEATURE_CLASSES["LQ"], maxent.FEATURE_CLASSES["LQHP"]
    assert l["linear"] and lq["linear"] and lqhp["linear"]
    assert not l["quadratic"] and lq["quadratic"] and lqhp["quadratic"]
    assert not lq["product"] and lqhp["product"]
    assert not lq["hinge"] and lqhp["hinge"]

"""Tests for pipeline.screen.bivariate -- the presence-vs-background
predictor CHARACTERIZATION (Mann-Whitney U + rank-biserial r for continuous
predictors, chi-square + Cramer's V for categorical predictors, Bonferroni
correction across every predictor actually tested), ported from
the original notebook ``02c_predictor_bivariate``.

Deliberately exercises the REAL on-disk screened predictor stack
(``data/processed/predictors_screened/*.tif``) and the real canonical
286-point HW+EOW presence set (``data/processed/canonical_presence.csv``),
matching this project's established testing philosophy (test_screen.py's own
docstring: "Deliberately exercises the real stack, not a synthetic fixture").

``02c``'s own ``PRED_DIR`` is ``'../data/processed/predictors_screened'`` --
i.e. the SOURCE notebook already characterized the SCREENED stack, not the
raw 22-layer one -- so ``bivariate()``'s default ``pred_dir=SCREENED_DIR`` is
a faithful port, not a judgment call. The real screen keeps 18 layers
as of the log-transform adoption + TWI-retention preference (originally 20,
before the transform's Stage-1/2 re-screen additionally
dropped ``spi``/``flow_accumulation`` (``twi`` retained -- see
test_screen.py's own ``EXPECTED_KEPT``/``test_actual_kept_set_is_18_layers``)
-- not the original planning estimate of 21. Every assertion below
about "18" is an empirical regression pin on that already-verified fact,
exactly like test_screen.py's own pins; ``screen.py`` itself must derive
``n_tests`` live from whatever is really on disk, never hardcode any of 18,
20, or 21.
"""
import numpy as np
import pandas as pd
import pytest
import rasterio

from pipeline import config, screen


@pytest.fixture(scope="module", autouse=True)
def ensure_screened_stack():
    """tests/test_bivariate.py must be runnable in isolation
    (``pytest tests/test_bivariate.py``) without depending
    on tests/test_screen.py having already populated
    data/processed/predictors_screened/ as a side effect -- regenerate it
    explicitly, exactly like test_screen.py's own module-scoped ``screened``
    fixture does.
    """
    screen.multicollinearity(screen._load_stack())


@pytest.fixture(scope="module")
def presence():
    """The canonical 286-point HW+EOW presence set -- read straight from the
    CSV ``hwm.build_presence()`` already writes (avoids re-parsing 13 raw
    survey files per test run; same data either way). Deliberately NOT
    ``config.canonical_samples()``'s smaller, predictor-NaN-gated subset:
    ``bivariate()`` mirrors ``02c``'s own per-predictor ``np.isfinite`` drop
    instead of requiring simultaneous validity across every layer -- see
    ``bivariate``'s docstring.
    """
    df = pd.read_csv(config.DATA / "canonical_presence.csv")
    assert len(df) == 286
    return df


@pytest.fixture(scope="module")
def result(presence):
    return screen.bivariate(presence)


# --------------------------------------------------------------------------
# (a) Bonferroni applied over the ACTUAL n_tests
# --------------------------------------------------------------------------

def test_n_tests_derived_from_actual_screened_stack_not_hardcoded(result):
    """screen.py must never hardcode the stale planning estimate of 21 -- n_tests
    is derived live from whatever predictors are actually on disk under
    SCREENED_DIR. Empirically that is 18 today (the real screen,
    through the log-transform adoption's modeling transform + TWI-retention
    preference, keeps 18 -- 20 before the transform additionally dropped
    ``spi``/``flow_accumulation`` (``twi`` retained), and neither the
    assumed 21) -- pinned here as a regression guard,
    same philosophy as test_actual_kept_set_is_18_layers in test_screen.py:
    if a future predictor-stack rebuild changes this, this test SHOULD fail
    so a human notices, rather than silently drifting.
    """
    on_disk = sorted(p.stem for p in screen.SCREENED_DIR.glob("*.tif"))
    assert len(on_disk) == 18
    assert result.attrs["n_tests"] == 18
    assert len(result) == 18
    assert sorted(result["predictor"]) == on_disk


def test_bonferroni_applied_over_actual_n_tests(result):
    n_tests = result.attrs["n_tests"]
    assert n_tests == len(result)
    expected = (result["p_raw"] * n_tests).clip(upper=1.0)
    np.testing.assert_allclose(result["p_bonferroni"].to_numpy(), expected.to_numpy())
    assert (result["significant"] == (result["p_bonferroni"] < 0.05)).all()


def test_bonferroni_never_exceeds_one(result):
    assert (result["p_bonferroni"] <= 1.0).all()
    assert (result["p_bonferroni"] >= 0.0).all()


# --------------------------------------------------------------------------
# (b) Result is explicitly labeled non-validation
# --------------------------------------------------------------------------

def test_labeled_characterization_not_validation(result):
    assert result.attrs.get("kind") == "characterization"
    assert result.attrs.get("not_validation") is True
    framing = result.attrs.get("framing", "")
    assert "not" in framing.lower() and "validation" in framing.lower()


def test_docstring_states_not_validation():
    doc = (screen.bivariate.__doc__ or "").lower()
    assert "not model validation" in doc or "not validation" in doc
    assert "characteriz" in doc  # characterization/characterizes


def test_saved_csv_filename_signals_characterization(result):
    """The on-disk artifact name itself carries the framing cue -- deviates
    deliberately from 02c's ``predictor_bivariate_stats.csv`` so even a bare
    ``ls data/processed/`` can't mistake this for a validation artifact.
    """
    assert "characterization" in screen.BIVARIATE_REPORT_PATH.name
    assert screen.BIVARIATE_REPORT_PATH.exists()
    saved = pd.read_csv(screen.BIVARIATE_REPORT_PATH)
    assert list(saved["predictor"]) == list(result["predictor"])


# --------------------------------------------------------------------------
# (c) Continuous rows use Mann-Whitney/rank-biserial; categorical rows use
# chi-square/Cramer's V
# --------------------------------------------------------------------------

def test_continuous_rows_use_mann_whitney_rank_biserial(result):
    cont = result[result["type"] == "continuous"]
    assert len(cont) == 16  # 18 screened - 2 categorical
    assert (cont["test"] == "Mann-Whitney U").all()
    assert (cont["effect_metric"] == "rank-biserial r").all()
    assert cont["effect"].between(-1.0, 1.0).all()
    assert set(cont["direction"]) <= {"higher", "lower"}


def test_categorical_rows_use_chi_square_cramers_v(result):
    cat = result[result["type"] == "categorical"]
    assert set(cat["predictor"]) == set(screen.CATEGORICAL)
    assert (cat["test"] == "chi-square").all()
    assert (cat["effect_metric"] == "Cramer's V").all()
    assert cat["effect"].between(0.0, 1.0 + 1e-9).all()
    assert (cat["direction"] == "n/a").all()


def test_categorical_effect_reuses_module_cramers_v(presence):
    """The categorical rows' 'effect' column must be the SAME statistic
    screen.cramers_v computes (the nominal-aware fix), not a fresh,
    independently-implemented rebinning helper -- verified by reproducing
    one categorical row's effect via a direct screen.cramers_v(a_nominal=
    True, b_nominal=True) call on the same presence/background group-vs-
    category-code arrays bivariate() itself builds internally.
    """
    arrays, transform, valid_mask = screen._load_arrays_2d(screen.SCREENED_DIR)
    pres_row, pres_col = screen._xy_to_rowcol(
        presence["x"].to_numpy(), presence["y"].to_numpy(), transform
    )
    nrows, ncols = valid_mask.shape
    in_bounds = (pres_row >= 0) & (pres_row < nrows) & (pres_col >= 0) & (pres_col < ncols)
    pres_row, pres_col = pres_row[in_bounds], pres_col[in_bounds]
    bg_row, bg_col = screen._draw_background(valid_mask, n=10_000, random_state=config.GLOBAL_SEED)

    result = screen.bivariate(presence)
    for cat_name in screen.CATEGORICAL:
        arr = arrays[cat_name]
        pv = arr[pres_row, pres_col]
        bv = arr[bg_row, bg_col]
        pv = pv[np.isfinite(pv)]
        bv = bv[np.isfinite(bv)]
        group = np.concatenate([np.ones(len(pv), dtype=int), np.zeros(len(bv), dtype=int)])
        codes = np.concatenate([pv, bv]).astype(int)
        expected_v = screen.cramers_v(group, codes, a_nominal=True, b_nominal=True)
        actual_v = result.loc[result["predictor"] == cat_name, "effect"].iloc[0]
        assert actual_v == pytest.approx(expected_v)


def test_cramers_v_stats_matches_public_cramers_v():
    """_cramers_v_stats (the shared core bivariate() uses to also get the
    chi-square p-value) must return the exact same V as the public
    cramers_v() wrapper -- confirming the refactor reuses one code path
    rather than adding a second, parallel computation.
    """
    rng = np.random.default_rng(9)
    group = rng.integers(0, 2, size=2000)
    codes = (group * 3 + rng.integers(0, 3, size=2000))  # dependent on group
    v, chi2, p, table = screen._cramers_v_stats(group, codes, a_nominal=True, b_nominal=True)
    v_public = screen.cramers_v(group, codes, a_nominal=True, b_nominal=True)
    assert v == pytest.approx(v_public)
    assert 0.0 <= p <= 1.0
    assert chi2 >= 0.0
    assert table.values.sum() == 2000


# --------------------------------------------------------------------------
# (d) Determinism: same background seed -> same result
# --------------------------------------------------------------------------

def test_deterministic_given_same_seed(presence):
    r1 = screen.bivariate(presence, random_state=42)
    r2 = screen.bivariate(presence, random_state=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_different_seed_changes_background_draw(presence):
    r1 = screen.bivariate(presence, random_state=42)
    r2 = screen.bivariate(presence, random_state=7)
    assert not r1["stat"].equals(r2["stat"])


def test_draw_background_deterministic_given_seed():
    valid = np.ones((50, 50), dtype=bool)
    row1, col1 = screen._draw_background(valid, n=100, random_state=42)
    row2, col2 = screen._draw_background(valid, n=100, random_state=42)
    assert np.array_equal(row1, row2) and np.array_equal(col1, col2)

    row3, col3 = screen._draw_background(valid, n=100, random_state=1)
    assert not (np.array_equal(row1, row3) and np.array_equal(col1, col3))


def test_background_drawn_with_global_seed_by_default(presence):
    """Default call (no explicit random_state) must match an explicit
    random_state=config.GLOBAL_SEED call -- config.GLOBAL_SEED (42) is the
    real default, not merely a coincidentally-equal literal.
    """
    r_default = screen.bivariate(presence)
    r_explicit = screen.bivariate(presence, random_state=config.GLOBAL_SEED)
    pd.testing.assert_frame_equal(r_default, r_explicit)


# --------------------------------------------------------------------------
# Supporting pure-function tests (no raster I/O)
# --------------------------------------------------------------------------

def test_xy_to_rowcol_matches_affine_inverse():
    transform = rasterio.transform.from_origin(west=0, north=100, xsize=10, ysize=10)
    row, col = screen._xy_to_rowcol(np.array([5, 15, 95]), np.array([95, 85, 5]), transform)
    assert list(row) == [0, 1, 9]
    assert list(col) == [0, 1, 9]


# --------------------------------------------------------------------------
# _mwu_rank_biserial() -- ground-truth sign-convention guard
#
# The continuous-row assertions in section (c) above only check
# `effect.between(-1.0, 1.0)` and `direction in {"higher", "lower"}` -- both
# of which would STILL PASS even if the rank-biserial formula were silently
# inverted (e.g. `r = 1 - 2*U/(n1*n2)`, a real alternative convention in the
# literature: both conventions land in [-1, 1], and only differ in which
# side of the presence/background contrast gets called "higher"). Since this
# sign sets the reported DIRECTION of every predictor's effect in the paper
# table, it must be pinned against a hand-computable ground truth, not just
# range-checked against the real (but not independently verified by the
# test itself) CSV values.
# --------------------------------------------------------------------------

def test_mwu_rank_biserial_presence_strictly_higher_is_r_plus_one():
    """presence uniformly 10, background uniformly 1 -- every presence value
    outranks every background value, so U = n1*n2 (25) and
    r = (2*25/25) - 1 = 1.0 exactly.
    """
    pv = np.full(5, 10.0)
    bv = np.full(5, 1.0)
    u, p, r, direction = screen._mwu_rank_biserial(pv, bv)
    assert r == pytest.approx(1.0)
    assert direction == "higher"


def test_mwu_rank_biserial_presence_strictly_lower_is_r_minus_one():
    """Swapped: presence uniformly 1, background uniformly 10 -- every
    presence value ranks below every background value, so U = 0 and
    r = (2*0/25) - 1 = -1.0 exactly. This is the case that would flip sign
    (to +1.0) under the alternative `r = 1 - 2U/(n1*n2)` convention -- if
    this assertion ever starts reading +1.0 instead, the formula has been
    silently inverted.
    """
    pv = np.full(5, 1.0)
    bv = np.full(5, 10.0)
    u, p, r, direction = screen._mwu_rank_biserial(pv, bv)
    assert r == pytest.approx(-1.0)
    assert direction == "lower"


def test_mwu_rank_biserial_balanced_overlap_near_zero():
    """A symmetric, thoroughly-interleaved case with no systematic shift
    either direction. Combined and ranked 1..10: pv occupies ranks
    {1, 4, 5, 8, 10} (rank-sum 28), bv occupies {2, 3, 6, 7, 9} (rank-sum
    27) -- so U = 28 - 5*6/2 = 13 and r = (2*13/25) - 1 = 0.04, close to (but
    not exactly, since n=5 is odd and the two rank-sums can't split
    perfectly evenly) zero. Guards against a degenerate helper that always
    returns +-1 or that mislabels a near-tie.
    """
    pv = np.array([1.0, 4.0, 5.0, 8.0, 10.0])
    bv = np.array([2.0, 3.0, 6.0, 7.0, 9.0])
    u, p, r, direction = screen._mwu_rank_biserial(pv, bv)
    assert r == pytest.approx(0.04)
    assert -1.0 <= r <= 1.0
    assert abs(r) < 0.1
    assert direction == "higher"  # r's own sign here is (barely) positive

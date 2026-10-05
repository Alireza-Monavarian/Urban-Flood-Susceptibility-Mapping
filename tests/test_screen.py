"""Tests for pipeline.screen -- the two-stage multicollinearity
screen (Pearson |r| >= 0.8, then VIF > 10) over the REAL on-disk 22-layer
predictor stack (data/processed/predictors/*.tif), plus cramers_v() for
categorical association.

Deliberately exercises the real stack, not a synthetic fixture -- the whole
point is to find out whether the D8-routing fix
(flow_accumulation/twi/spi/hand, previously capped/broken at 2451) introduces
any NEW multicollinearity the old, broken hydrology never had a chance to
show.

Empirical result on the RAW stack: the D8 fix alone does NOT introduce new
collinearity. Stage 1 dropped exactly ``tri``/``tpi`` -- same as
the original notebook 03_multicollinearity, for the same terrain reason (slope r=0.984
vs tri, curvature r=0.972 vs tpi) -- and Stage 2 needed no additional
drops (max VIF 3.42, ``available_water_storage``). ``flow_accumulation``/
``spi`` was the closest the corrected hydrology came to a Stage-1 hit, at
r=0.79 -- just under the 0.8 threshold.

UPDATE (log-transform adoption): ``pipeline.screen.multicollinearity()``
now assesses Stage 1/2 on ``pipeline.predictors.apply_modeling_transform()``
-transformed values (log10 for ``flow_accumulation``, log1p for ``spi``,
identity elsewhere) -- the same values MaxEnt actually models -- rather
than raw physical units. This DOES change the result: log10
(``flow_accumulation``) vs log1p(``spi``) now correlates r=0.81, crossing
the 0.8 Stage-1 threshold the raw pair's r=0.79 stayed just under, so
``spi`` is now ALSO dropped at Stage 1. That cascades one step further: of
``flow_accumulation``/``twi``, one of the two must go at Stage 2 (VIF) --
see UPDATE 2 below for which one and why. The full ACTUAL drop set on the
transformed stack is 4 (not 2) -- 18 kept, not 20. Nothing here is
hardcoded to force any particular outcome; ``multicollinearity()`` computes
it fresh from whatever 22-layer stack is on disk (through the modeling
transform), and the "pinned" tests below assert the ACTUAL observed result
on today's transformed stack, not a value copied from the old
(pre-transform) run.

UPDATE 2 (TWI retention): of the ``flow_accumulation``/``twi`` pair Stage 2 must
resolve (VIF=10.37 for whichever one is evaluated with the other still
present -- they are not a Stage-1 hit; their own pairwise r=0.768 stays
under 0.8), the project owner's standing preference is to retain ``twi``
(the canonical topographic wetness index in flood-susceptibility
literature) and drop ``flow_accumulation`` instead -- the reverse of what
an unmodified "drop the single highest VIF column" rule would do (that
rule flags ``twi`` first, since its VIF is marginally higher than
``flow_accumulation``'s at the iteration both are still present -- see
``screen._choose_vif_drop``'s docstring). ``EXPECTED_KEPT``/
``EXPECTED_DROPPED`` below reflect this: ``twi`` is now KEPT,
``flow_accumulation`` is now DROPPED (swapped from the log-transform
-adoption-only result immediately above) -- kept/dropped COUNTS are
unchanged (18/4), only WHICH of the two survives is different.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import screen

# The exact kept set found running the real screen (through the
# log-transform adoption's modeling transform AND the TWI-retention
# preference -- module docstring, "UPDATE" / "UPDATE 2") against the
# corrected 22-layer stack. Not a hoped-for target -- if
# a future predictor-stack rebuild (or transform/preference change) changes
# this, this test SHOULD start failing so a human notices, the same
# philosophy as test_valid_presence_actual_count_is_279 in
# test_predictor_qa.py. ``flow_accumulation``/``spi`` are NOT here (``twi``
# IS, unlike the log-transform-only run) -- see EXPECTED_DROPPED below.
EXPECTED_KEPT = sorted([
    "available_water_storage", "curvature", "curve_number", "dem",
    "depth_to_restriction", "distance_to_outfall", "distance_to_streams",
    "drainage_density", "hand", "hydrologic_soil_group",
    "impervious_pct", "ksat", "median_income", "nlcd_landcover",
    "population_density", "slope", "stormwater_density", "twi",
])
# tri/tpi: Stage 1, pre-existing (terrain vs slope/curvature). spi: Stage 1,
# NEW post-transform (log10(flow_accumulation) vs log1p(spi), r=0.81 >= 0.8).
# flow_accumulation: Stage 2, NEW post-transform+TWI-retention (VIF=10.37 at
# the iteration both flow_accumulation/twi are present; screen._choose_vif_drop
# removes flow_accumulation instead of twi per the owner's TWI-retention
# preference) -- see module docstring, "UPDATE" / "UPDATE 2".
EXPECTED_DROPPED = sorted(["tri", "tpi", "flow_accumulation", "spi"])


# One load + one full screen run (Pearson + VIF + raster copy), shared by
# every test that needs it below -- real I/O/compute (~2-3s), no reason to
# pay for it more than once per test session.
@pytest.fixture(scope="module")
def stack():
    return screen._load_stack()


@pytest.fixture(scope="module")
def screened(stack):
    return screen.multicollinearity(stack)


# --------------------------------------------------------------------------
# _load_stack()
# --------------------------------------------------------------------------

def test_load_stack_shape_and_no_nan(stack):
    assert len(stack) == 200_000
    assert len(stack.columns) == 22
    assert not stack.isna().any().any()


def test_load_stack_includes_all_predictor_names(stack):
    on_disk = {p.stem for p in screen.PRED_DIR.glob("*.tif")}
    assert set(stack.columns) == on_disk
    assert "tri" in stack.columns and "tpi" in stack.columns  # present pre-screen
    assert "nlcd_landcover" in stack.columns and "hydrologic_soil_group" in stack.columns


def test_load_stack_deterministic(stack):
    again = screen._load_stack()
    pd.testing.assert_frame_equal(stack, again)


def test_flow_accumulation_reflects_corrected_d8_routing(stack):
    """Regression guard: this screen must be running against the
    FIXED D8 routing (max ~5.5M), not the old xarray-spatial bug that
    capped real-AOI flow accumulation at 2451. If this ever
    silently reverted, the whole premise of re-running this screen (rather
    than trusting the old notebook's tri/tpi-only result) would be moot.
    """
    assert stack["flow_accumulation"].max() > 50_000


# --------------------------------------------------------------------------
# multicollinearity() -- the core screen test
# --------------------------------------------------------------------------

def test_drops_tri_and_tpi_and_no_categorical_pearson():
    kept, report = screen.multicollinearity(screen._load_stack())
    assert "tri" not in kept and "tpi" not in kept
    assert "nlcd_landcover" not in report.index and "hydrologic_soil_group" not in report.index


# --------------------------------------------------------------------------
# multicollinearity() -- the ACTUAL observed full result, pinned
# --------------------------------------------------------------------------

def test_actual_kept_set_is_18_layers(screened):
    kept, _report = screened
    assert kept == EXPECTED_KEPT
    assert len(kept) == 18


def test_actual_dropped_set_is_tri_tpi_flow_accumulation_spi(screened):
    """The log-transform adoption + TWI-retention preference (module
    docstring, "UPDATE" / "UPDATE 2") change the drop set from the
    pre-transform {tri, tpi} to {tri, tpi, flow_accumulation, spi}:
    ``spi`` is a Stage-1 drop (log10(flow_accumulation) vs log1p(spi),
    r=0.81 >= 0.8 -- see
    test_flow_accumulation_spi_crosses_08_threshold_post_transform below),
    and ``flow_accumulation`` is a Stage-2 drop that cascades from it
    (VIF=10.37 at the iteration both flow_accumulation/twi are present --
    see test_vif_le_10_for_every_kept_continuous_predictor below, and
    screen._choose_vif_drop for why flow_accumulation -- not twi -- is the
    one removed). This is
    verified, not assumed: see
    test_hand_is_not_dropped_but_flow_accumulation_now_is and
    test_spi_and_flow_accumulation_are_dropped_twi_retained below for the
    underlying evidence.
    """
    _kept, report = screened
    dropped = sorted(report.index[report["dropped"]].tolist())
    assert dropped == EXPECTED_DROPPED


def test_stage1_terrain_pairs_exceed_threshold(screened):
    _kept, report = screened
    assert report.loc["tri", "slope"] >= 0.8
    assert report.loc["tpi", "curvature"] >= 0.8


def test_dropped_rows_have_nan_vif(screened):
    """tri/tpi/spi never reach Stage 2 (dropped in Stage 1), so their VIF is
    undefined (NaN), not silently 0 or omitted. ``spi`` is the log-transform
    adoption's new Stage-1 drop (module docstring, "UPDATE") -- same
    rationale as the pre-existing tri/tpi pins."""
    _kept, report = screened
    assert np.isnan(report.loc["tri", "VIF"])
    assert np.isnan(report.loc["tpi", "VIF"])
    assert np.isnan(report.loc["spi", "VIF"])


def test_flow_accumulation_dropped_at_stage2_but_report_vif_also_reads_nan(screened):
    """``flow_accumulation`` is different from tri/tpi/spi: it IS reached by
    Stage 2 (it has a real, finite VIF partway through the iterative
    procedure -- present at the SAME iteration ``twi``'s VIF hits 10.37,
    over the >10 threshold, which is WHY one of the two must be dropped --
    ``screen._choose_vif_drop`` picks ``flow_accumulation`` per the
    owner's TWI-retention preference). ``report["VIF"]`` still
    reads NaN for it, same as a Stage-1 drop, simply because
    ``multicollinearity()`` reindexes onto ``vif_history[-1]`` (the LAST
    iteration's surviving columns only) rather than recording each column's
    own triggering VIF at the moment it was removed -- a pre-existing
    reporting characteristic of ``_stage2_vif``'s history mechanism, first
    surfaced when the log-transform adoption made Stage 2 drop something
    for the first time. Pinned here explicitly (rather than folded into
    test_dropped_rows_have_nan_vif above) so this NaN is never mistaken for
    "VIF was undefined for flow_accumulation" -- it wasn't; it just isn't
    the value retained in the final report.
    """
    _kept, report = screened
    assert bool(report.loc["flow_accumulation", "dropped"]) is True
    assert np.isnan(report.loc["flow_accumulation", "VIF"])


def test_vif_le_10_for_every_kept_continuous_predictor(screened):
    kept, report = screened
    kept_continuous = [c for c in kept if c not in screen.CATEGORICAL]
    vif = report.loc[kept_continuous, "VIF"]
    assert vif.notna().all()
    assert (vif <= 10).all()


def test_categorical_layers_never_enter_pearson_report(screened):
    _kept, report = screened
    for cat in screen.CATEGORICAL:
        assert cat not in report.index
        assert cat not in report.columns


# --------------------------------------------------------------------------
# The hydrology block, post log-transform adoption -- where the transform's
# own effect on collinearity shows up (module docstring, "UPDATE").
# --------------------------------------------------------------------------

def test_hand_is_not_dropped_but_flow_accumulation_now_is(screened):
    """Of the 4 D8-routing hydrology layers, ``hand`` is unaffected by the
    log-transform adoption's new Stage-1/2 drops. ``flow_accumulation``
    WAS unaffected too, immediately after the log-transform adoption alone
    -- but the TWI-retention preference (module docstring, "UPDATE 2") then
    swaps which of ``flow_accumulation``/``twi`` Stage 2 removes, so
    ``flow_accumulation`` is now dropped and ``twi`` is not (see
    test_spi_and_flow_accumulation_are_dropped_twi_retained below)."""
    _kept, report = screened
    assert bool(report.loc["hand", "dropped"]) is False
    assert bool(report.loc["flow_accumulation", "dropped"]) is True
    assert bool(report.loc["twi", "dropped"]) is False


def test_spi_and_flow_accumulation_are_dropped_twi_retained(screened):
    """Unlike the pre-transform screen (which introduced NO new drop beyond
    tri/tpi), the log-transform
    adoption's re-screen DOES drop 2 more layers -- but WHICH 2 depends on
    the TWI-retention preference: ``spi`` at Stage 1 (paired
    with ``flow_accumulation``, see
    test_flow_accumulation_spi_crosses_08_threshold_post_transform) and
    ``flow_accumulation`` (not ``twi``) at Stage 2 (VIF, see
    test_flow_accumulation_dropped_at_stage2_but_report_vif_also_reads_nan).
    ``twi`` survives. This is the real, computed consequence of adopting
    the transform + the retention preference, not a forced or hoped-for
    result.
    """
    _kept, report = screened
    for h in ("spi", "flow_accumulation"):
        assert bool(report.loc[h, "dropped"]) is True
    assert bool(report.loc["twi", "dropped"]) is False


def test_flow_accumulation_spi_crosses_08_threshold_post_transform(screened):
    """The raw pair's r=0.79 stayed just under the 0.8
    Stage-1 threshold because flow_accumulation's extreme skew (spans 6+
    orders of magnitude) suppresses the linear (Pearson) relationship.
    log10(flow_accumulation) vs log1p(spi) removes that skew and reveals a
    much stronger linear relationship between the two -- r=0.81, now OVER
    the threshold. Bounded to a range (rather than an exact literal) so a
    future stack rebuild landing anywhere in the same "just over 0.8" zone
    still passes, but a rebuild that fell back under 0.8 (undoing this
    finding) -- or that moved far past this range -- would fail loudly
    instead of silently drifting.
    """
    _kept, report = screened
    r = abs(report.loc["flow_accumulation", "spi"])
    assert 0.8 <= r < 0.85


def test_hydrology_layers_not_strongly_collinear_with_terrain(screened):
    """slope/curvature (the two terrain layers involved in the actual Stage
    1 drops) do not show a hidden near-threshold pairing with any of the 4
    corrected hydrology layers either -- the (post-transform)
    multicollinearity risk in this block is isolated to the
    flow_accumulation/spi pairing (and twi's cascading Stage-2 VIF) above,
    not a hidden terrain interaction."""
    _kept, report = screened
    for h in ("flow_accumulation", "twi", "spi", "hand"):
        assert abs(report.loc[h, "slope"]) < 0.8
        assert abs(report.loc[h, "curvature"]) < 0.8


# --------------------------------------------------------------------------
# Report/CSV outputs
# --------------------------------------------------------------------------

def test_report_saved_to_csv_matches_returned_report(screened):
    _kept, report = screened
    assert screen.REPORT_PATH.exists()
    saved = pd.read_csv(screen.REPORT_PATH, index_col=0)
    assert list(saved.index) == list(report.index)
    numeric_cols = [c for c in report.columns if c != "dropped"]
    np.testing.assert_allclose(
        saved[numeric_cols].to_numpy(dtype=float),
        report[numeric_cols].to_numpy(dtype=float),
        rtol=1e-8, equal_nan=True,
    )
    assert (saved["dropped"].astype(bool).to_numpy() == report["dropped"].to_numpy()).all()


def test_predictors_screened_dir_has_exactly_the_kept_rasters(screened):
    kept, _report = screened
    files = {p.stem for p in screen.SCREENED_DIR.glob("*.tif")}
    assert files == set(kept)


def test_categorical_association_report_saved(screened):
    _kept, _report = screened
    assert screen.CATEGORICAL_REPORT_PATH.exists()
    cat_report = pd.read_csv(screen.CATEGORICAL_REPORT_PATH, index_col=0)
    assert set(cat_report.index) == set(screen.CATEGORICAL)
    for cat in screen.CATEGORICAL:
        vals = cat_report.loc[cat].drop(labels=[cat]).dropna()
        assert len(vals) > 0
        assert ((vals >= 0) & (vals <= 1 + 1e-9)).all()


# --------------------------------------------------------------------------
# _choose_drop() -- Stage 1 tie-break: documented rationale for the two
# known pairs, generic fallback for anything else (never a hardcoded list)
# --------------------------------------------------------------------------

def test_choose_drop_uses_known_rationale_for_slope_tri_either_order():
    empty_corr = pd.DataFrame()
    assert screen._choose_drop("slope", "tri", empty_corr) == "tri"
    assert screen._choose_drop("tri", "slope", empty_corr) == "tri"


def test_choose_drop_uses_known_rationale_for_curvature_tpi_either_order():
    empty_corr = pd.DataFrame()
    assert screen._choose_drop("curvature", "tpi", empty_corr) == "tpi"
    assert screen._choose_drop("tpi", "curvature", empty_corr) == "tpi"


def test_choose_drop_uses_known_rationale_for_flow_accumulation_twi_either_order():
    """The TWI-retention rule: the
    ``flow_accumulation``/``twi`` entry in ``_KNOWN_PAIR_RATIONALE`` says
    keep ``twi``, drop ``flow_accumulation`` -- directly testable through
    ``_choose_drop`` itself (order-independent), even though in PRACTICE
    this specific pair is never actually flagged by Stage 1's pairwise
    Pearson computation (it is resolved by ``_choose_vif_drop`` at Stage 2
    instead -- see the tests below and ``_choose_vif_drop``'s own
    docstring)."""
    empty_corr = pd.DataFrame()
    assert screen._choose_drop("flow_accumulation", "twi", empty_corr) == "flow_accumulation"
    assert screen._choose_drop("twi", "flow_accumulation", empty_corr) == "flow_accumulation"


def test_choose_drop_falls_back_to_generic_rule_for_unknown_pair():
    """A pair NOT in the known-rationale table: the generic "drop the more
    broadly redundant variable" heuristic must fire instead of raising or
    guessing. Here 'x' is more correlated with the rest of this synthetic
    stack (0.9, 0.85) than 'y' is (0.1, 0.1), so 'x' should be dropped.
    """
    corr = pd.DataFrame(
        {"x": [1.0, 0.9, 0.9, 0.85], "y": [0.9, 1.0, 0.1, 0.1],
         "z": [0.9, 0.1, 1.0, 0.1], "w": [0.85, 0.1, 0.1, 1.0]},
        index=["x", "y", "z", "w"],
    )
    assert screen._choose_drop("x", "y", corr) == "x"


# --------------------------------------------------------------------------
# _choose_vif_drop() -- Stage 2's per-iteration column-removal choice:
# normally the mechanical highest-VIF column, EXCEPT the documented
# flow_accumulation/twi override (the TWI-retention rule). Pure-function unit tests, synthetic (no
# raster I/O) -- see test_flow_accumulation_dropped_at_stage2_but_report_vif
# _also_reads_nan / test_spi_and_flow_accumulation_are_dropped_twi_retained
# above for the end-to-end confirmation against the real stack.
# --------------------------------------------------------------------------

def test_choose_vif_drop_overrides_to_flow_accumulation_when_twi_is_top():
    """The documented override: if the mechanical top-VIF column is ``twi``
    (the "keep" side of the ``_KNOWN_PAIR_RATIONALE`` entry) and
    ``flow_accumulation`` (its known partner) is still present in
    ``working_cols``, drop ``flow_accumulation`` instead of ``twi``."""
    vifs = pd.Series([15.0, 8.0, 3.0], index=["twi", "flow_accumulation", "dem"])
    working_cols = ["twi", "flow_accumulation", "dem"]
    assert screen._choose_vif_drop(vifs, working_cols) == "flow_accumulation"


def test_choose_vif_drop_falls_back_to_mechanical_top_when_partner_absent():
    """If ``twi`` is the mechanical top but ``flow_accumulation`` is NOT in
    ``working_cols`` (e.g. already removed by an earlier iteration, or
    never present in this particular call), there is no partner to
    redirect to -- the mechanical top (``twi``) is returned unchanged."""
    vifs = pd.Series([15.0, 3.0], index=["twi", "dem"])
    working_cols = ["twi", "dem"]
    assert screen._choose_vif_drop(vifs, working_cols) == "twi"


def test_choose_vif_drop_unaffected_for_unrelated_top_column():
    """A top column with no ``_KNOWN_PAIR_RATIONALE`` entry at all (e.g.
    ``available_water_storage``) is returned unchanged -- the override only
    ever fires for a documented pair's "keep" side."""
    vifs = pd.Series([12.0, 5.0], index=["available_water_storage", "dem"])
    working_cols = ["available_water_storage", "dem"]
    assert screen._choose_vif_drop(vifs, working_cols) == "available_water_storage"


def test_choose_vif_drop_never_fires_for_pre_existing_stage1_only_pairs():
    """Regression guard: even if ``slope`` or ``curvature`` (the "keep"
    side of the two PRE-EXISTING ``_KNOWN_PAIR_RATIONALE`` entries) somehow
    were the mechanical top at Stage 2, the override must NOT fire unless
    their Stage-1 partner (``tri``/``tpi`` respectively) is ALSO still in
    ``working_cols`` -- which never happens in practice (tri/tpi are always
    removed at Stage 1), but this pins the guard condition itself rather
    than relying on that always being true elsewhere."""
    vifs = pd.Series([12.0, 5.0], index=["slope", "dem"])
    working_cols = ["slope", "dem"]  # tri NOT present
    assert screen._choose_vif_drop(vifs, working_cols) == "slope"


# --------------------------------------------------------------------------
# cramers_v() -- pure-function unit tests, synthetic (no raster I/O)
# --------------------------------------------------------------------------

def test_cramers_v_perfect_association_is_one():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=5000)
    assert screen.cramers_v(a, a) == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_independent_categoricals_near_zero():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, size=20_000)
    b = rng.integers(0, 5, size=20_000)
    assert screen.cramers_v(a, b) < 0.05


def test_cramers_v_symmetric():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 3, size=5000)
    b = (a + rng.integers(0, 3, size=5000)) % 3
    assert screen.cramers_v(a, b) == pytest.approx(screen.cramers_v(b, a), rel=1e-9)


def test_cramers_v_categorical_continuous_pair_detects_dependency():
    """A continuous variable driven mostly by a categorical group (small
    noise relative to the between-group spacing) must show strong, but not
    necessarily perfect, association -- exercises the auto-discretization
    (quantile-binning) branch for the continuous side of the pair.
    """
    rng = np.random.default_rng(3)
    groups = rng.integers(0, 3, size=20_000)
    cont = groups * 10 + rng.normal(size=20_000)
    v = screen.cramers_v(groups, cont)
    assert 0.3 < v <= 1.0


def test_cramers_v_bounded_0_to_1():
    v = screen.cramers_v([1, 1, 2, 2, 3, 3] * 100, [1, 2, 1, 2, 1, 2] * 100)
    assert 0.0 <= v <= 1.0


def test_cramers_v_drops_pairwise_nan():
    a = pd.Series([1, 1, 2, 2, np.nan, 1])
    b = pd.Series([1, 2, 1, 2, 1, np.nan])
    v = screen.cramers_v(a, b)
    expected = screen.cramers_v(a.iloc[:4], b.iloc[:4])
    assert v == pytest.approx(expected)


def test_cramers_v_nominal_side_not_quantile_rebinned():
    """Regression guard for a review finding: a >10-class
    genuinely NOMINAL input (like ``nlcd_landcover``'s 15 native NLCD
    legend codes) must NOT be silently ``pd.qcut``-rebinned by raw code
    value just because its cardinality exceeds ``bins``. ``a_nominal=True``
    must keep it at its native categories instead.

    Uses 15 real NLCD legend codes (non-contiguous, unevenly spaced --
    ``nlcd_landcover``'s actual values) so this reproduces the exact shape
    of the bug rather than an idealized 0..14 range. ``b``'s dependence on
    ``a`` is wired through a SHUFFLED per-code effect -- i.e. structured by
    category IDENTITY, not by numeric code MAGNITUDE -- so a code-value
    quantile-rebin (which merges numerically-adjacent codes regardless of
    how different their true category effect is) necessarily destroys
    signal a native-category treatment preserves.
    """
    rng = np.random.default_rng(4)
    codes = [11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    a = pd.Series(rng.choice(codes, size=20_000))
    assert a.nunique() == 15  # > bins=10 -- the exact cardinality that
                              # used to trigger the qcut-rebin bug

    shuffled_effect = rng.permutation(len(codes)) * 10
    b = a.map(dict(zip(codes, shuffled_effect))) + rng.normal(scale=1.0, size=20_000)

    v_nominal = screen.cramers_v(a, b, a_nominal=True)
    v_rebinned = screen.cramers_v(a, b)  # default: cardinality-based, buggy path

    # Independent confirmation that the nominal path's contingency table
    # really is built from all 15 native categories, not <=10 code-value
    # quantile bins (the same crosstab cramers_v(a_nominal=True) builds
    # internally: native a x quantile-binned b).
    table = pd.crosstab(a, pd.qcut(b, q=10, duplicates="drop"))
    assert table.shape[0] == 15

    # Native categories capture far more of the (identity-keyed) signal
    # than a code-value rebin, which blends codes with unrelated effects.
    assert v_nominal > v_rebinned + 0.1

"""Tests for pipeline.equity -- the corrected ACS equity/
environmental-justice overlay (honest index composition, Holm-corrected
Spearman family, disclosed population_density predictor<->outcome
coupling, MAUP/ecological-fallacy/ACS-MOE/student-housing caveats).

Two tiers, mirroring this project's established pattern (test_sensitivity.py/
test_bivariate.py): Tier 1 is pure-logic/synthetic-data unit tests with zero
I/O (fast); Tier 2 exercises the REAL Census API + TIGER shapefile + the
real published susceptibility/population_density rasters, through one
module-scoped ``table``/``corr`` fixture pair reused across every Tier-2
test so the ~network+zonal-stats cost (a few seconds once cached, see
below) is paid once per test session, not once per test.

Every Tier-2 pinned number below was observed by actually running
``pipeline.equity`` against the real 2019-2023 5-year ACS pull for Riley
County and the real ``data/processed/maxent/final/cloglog/flood_avg.tif``
-- never guessed or forced toward an expected finding.
The central, honest result these pins encode: of the 6 indicators with
enough non-null data to test, only ``bg_pop_density_mean`` -- the
variable module docstring correction 3 explicitly discloses as
mechanically coupled to the model, not an independent equity signal --
survives multiple-comparison correction. None of the 5 genuine
sociodemographic indicators (including median household income) reach
even nominal (uncorrected) significance.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import config, equity


# ============================================================================
# Tier 1: pure-logic / synthetic-data unit tests (no network, no disk I/O)
# ============================================================================

# ----------------------------------------------------------------------
# (a) index_composition() -- the core composition test, plus
# structural honesty checks.
# ----------------------------------------------------------------------

def test_vulnerability_uses_five_indicators():
    """The vulnerability index uses exactly five indicators."""
    comp = equity.index_composition()
    assert set(comp["vulnerability"]) == {
        "pct_nonwhite", "pct_renter", "rent_burden_pct", "pct_elderly", "median_income_inv"
    }
    assert len(comp["adaptive_capacity"]) == 2


def test_index_composition_is_pure_no_io():
    """index_composition() must be callable with zero network/disk access
    -- a plain, static recipe, not something that silently depends on a
    live ACS pull having already happened. Calling it twice must be
    perfectly stable (no hidden state)."""
    comp1 = equity.index_composition()
    comp2 = equity.index_composition()
    assert comp1 == comp2


def test_adaptive_capacity_drops_vehicle_access_only():
    comp = equity.index_composition()
    assert set(comp["adaptive_capacity"]) == {"pct_owner", "median_income"}
    assert set(comp["adaptive_capacity_dropped"]) == {"pct_no_vehicle"}


def test_vulnerability_drops_poverty_and_disability_only():
    comp = equity.index_composition()
    assert set(comp["vulnerability_dropped"]) == {"poverty_rate", "pct_disability"}


def test_dropped_indicators_document_data_availability_not_redundancy():
    """The REAL reason must be documented, not a vague
    hand-wave -- and specifically not a mismatched description like the
    old notebook's. Each dropped-indicator reason must reference the
    empirical data-quality failure (null/suppressed), not merely assert
    'redundant'."""
    comp = equity.index_composition()
    for reason in comp["vulnerability_dropped"].values():
        assert "null" in reason.lower() or "suppress" in reason.lower()
    for reason in comp["adaptive_capacity_dropped"].values():
        assert "null" in reason.lower() or "suppress" in reason.lower()


def test_old_notebook_description_documented_for_contrast():
    """index_composition() must document the OLD (mismatched) notebook
    description for provenance -- it must not simply reproduce it as if
    it were still accurate."""
    comp = equity.index_composition()
    desc = comp["old_notebook_description"]
    assert "7" in desc  # names the old 7-indicator vulnerability count
    assert "07_equity_analysis" in desc


# ----------------------------------------------------------------------
# (b) compute_indices() -- must use index_composition()'s recipe exactly,
# no drift possible (the class of bug this module exists to eliminate).
# ----------------------------------------------------------------------

@pytest.fixture
def synthetic_bg():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "pct_nonwhite": rng.uniform(0, 0.4, n),
        "pct_renter": rng.uniform(0, 1, n),
        "rent_burden_pct": rng.uniform(15, 45, n),
        "pct_elderly": rng.uniform(0, 0.3, n),
        "median_income_inv": -rng.uniform(20000, 90000, n),
        "pct_owner": rng.uniform(0, 1, n),
        "median_income": rng.uniform(20000, 90000, n),
    })


def test_compute_indices_uses_exactly_the_declared_columns(synthetic_bg):
    out = equity.compute_indices(synthetic_bg)
    assert "vulnerability_index" in out.columns
    assert "adaptive_capacity_index" in out.columns

    comp = equity.index_composition()
    manual_vuln = pd.concat(
        [equity._zscore(synthetic_bg[c]) for c in comp["vulnerability"]], axis=1
    ).mean(axis=1, skipna=True)
    manual_ac = pd.concat(
        [equity._zscore(synthetic_bg[c]) for c in comp["adaptive_capacity"]], axis=1
    ).mean(axis=1, skipna=True)
    np.testing.assert_allclose(out["vulnerability_index"].to_numpy(), manual_vuln.to_numpy())
    np.testing.assert_allclose(out["adaptive_capacity_index"].to_numpy(), manual_ac.to_numpy())


def test_compute_indices_changing_composition_changes_computation(synthetic_bg, monkeypatch):
    """Proves there is no second, independently-maintained column list:
    patching index_composition()'s return value must change what
    compute_indices() actually computes."""
    baseline = equity.compute_indices(synthetic_bg)

    def _fewer_indicators():
        return {
            "vulnerability": ["pct_nonwhite"],
            "vulnerability_dropped": {},
            "adaptive_capacity": ["pct_owner"],
            "adaptive_capacity_dropped": {},
            "old_notebook_description": "",
        }
    monkeypatch.setattr(equity, "index_composition", _fewer_indicators)
    patched = equity.compute_indices(synthetic_bg)

    expected_vuln = equity._zscore(synthetic_bg["pct_nonwhite"])
    np.testing.assert_allclose(patched["vulnerability_index"].to_numpy(), expected_vuln.to_numpy())
    assert not np.allclose(baseline["vulnerability_index"].to_numpy(),
                           patched["vulnerability_index"].to_numpy())


# ----------------------------------------------------------------------
# (c) correlations() correction logic on a synthetic table (isolates the
# multiple-comparison math from the real data pull).
# ----------------------------------------------------------------------

def test_correlations_synthetic_holm_math_is_correct():
    rng = np.random.default_rng(1)
    n = 40
    susc = rng.uniform(0, 1, n)
    # One variable strongly related to susc (should survive correction),
    # others pure noise (should not).
    strong = susc * 2 + rng.normal(0, 0.05, n)
    noise1 = rng.uniform(0, 1, n)
    noise2 = rng.uniform(0, 1, n)
    synth = pd.DataFrame({"susc_mean": susc, "strong_var": strong,
                          "noise1": noise1, "noise2": noise2})
    variables = {"strong_var": "Strong", "noise1": "Noise 1", "noise2": "Noise 2"}

    df = equity.correlations(table=synth, variables=variables, method="holm")
    assert df.attrs["n_tests"] == 3
    assert df.loc[df["variable"] == "strong_var", "significant_corrected"].iloc[0]
    assert df["p_adjusted"].notna().all()
    assert (df["p_adjusted"] >= df["p_raw"]).all()  # correction never makes p smaller
    assert (df["p_adjusted"] <= 1.0).all()


def test_coupled_predictor_is_corrected_in_its_own_family():
    """Population density is corrected on its own, never alongside the
    sociodemographic indicators (manuscript Methods). Sharing a family let its
    near-zero p-value take the first rank and lower every indicator's
    BH-adjusted value; the indicators' adjustment must equal a correction over
    the indicators alone."""
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(3)
    n = 40
    susc = rng.uniform(0, 1, n)
    synth = pd.DataFrame({
        "susc_mean": susc,
        "bg_pop_density_mean": susc * 3 + rng.normal(0, 0.05, n),   # near-perfect
        "moderate": susc + rng.normal(0, 0.4, n),
        "weak": susc + rng.normal(0, 1.5, n),
        "noise": rng.uniform(0, 1, n),
    })
    variables = {"moderate": "Moderate", "weak": "Weak", "noise": "Noise",
                 "bg_pop_density_mean": "Population density"}

    df = equity.correlations(table=synth, variables=variables)
    assert df.attrs["n_tests"] == 4
    assert df.attrs["n_tests_by_family"] == {"sociodemographic": 3, "coupled_predictor": 1}

    pd_row = df.loc[df["variable"] == "bg_pop_density_mean"].iloc[0]
    assert pd_row["correction_family"] == "coupled_predictor"
    assert pd_row["p_adjusted_holm"] == pytest.approx(pd_row["p_raw"])   # family of one
    assert pd_row["p_adjusted_fdr_bh"] == pytest.approx(pd_row["p_raw"])

    demo = df[df["correction_family"] == "sociodemographic"]
    assert set(demo["variable"]) == {"moderate", "weak", "noise"}
    np.testing.assert_allclose(demo["p_adjusted_holm"],
                               multipletests(demo["p_raw"], method="holm")[1])
    np.testing.assert_allclose(demo["p_adjusted_fdr_bh"],
                               multipletests(demo["p_raw"], method="fdr_bh")[1])


def test_correlations_skips_sparse_variables_below_min_n():
    synth = pd.DataFrame({
        "susc_mean": np.linspace(0, 1, 20),
        "good_var": np.linspace(1, 2, 20),
        "sparse_var": [np.nan] * 17 + [1.0, 2.0, 3.0],
    })
    df = equity.correlations(table=synth,
                             variables={"good_var": "Good", "sparse_var": "Sparse"},
                             min_n=5)
    assert list(df["variable"]) == ["good_var"]


def test_correlations_rejects_unsupported_method():
    synth = pd.DataFrame({"susc_mean": np.arange(10.0), "x": np.arange(10.0) * 2})
    with pytest.raises(ValueError):
        equity.correlations(table=synth, variables={"x": "X"}, method="bonferroni")


def test_coupled_and_demographic_vars_are_disjoint():
    """Structural guarantee that the mechanically-coupled predictor can
    never be miscategorized as a genuine demographic indicator."""
    assert set(equity.DEMOGRAPHIC_VARS) & set(equity.COUPLED_VARS) == set()
    assert equity.ALL_CORRELATION_VARS == {**equity.DEMOGRAPHIC_VARS, **equity.COUPLED_VARS}


# ============================================================================
# Tier 2: real end-to-end -- ACS API + TIGER shapefile + real MaxEnt
# susceptibility/population_density rasters. One module-scoped fetch,
# reused everywhere below.
# ============================================================================

@pytest.fixture(scope="module")
def table():
    return equity.equity_table()


@pytest.fixture(scope="module")
def corr(table):
    return equity.correlations(table=table)


# ----------------------------------------------------------------------
# (d) Real ACS acquisition + zonal stats sanity
# ----------------------------------------------------------------------

def test_census_api_key_present():
    assert equity.CENSUS_KEY_PATH.exists()
    assert equity.CENSUS_KEY_PATH.read_text().strip()


def test_fetch_acs_returns_expected_columns():
    acs = equity.fetch_acs()
    expected = {"GEOID", "median_income", "median_income_inv", "poverty_rate",
                "pct_nonwhite", "pct_renter", "pct_owner", "rent_burden_pct",
                "pct_elderly", "pct_disability", "pct_no_vehicle", "pop_total",
                "pct_18_24", "median_age"}
    assert expected.issubset(set(acs.columns))
    assert len(acs) > 0


def test_fetch_acs_cache_short_circuits_network(monkeypatch):
    """After the real (module-fixture) pull has already cached to disk, a
    fresh fetch_acs() call must not touch the network at all."""
    assert equity.CENSUS_CACHE.exists(), "expected the real ACS pull to have cached by now"

    def _boom(*a, **kw):
        raise AssertionError("fetch_acs() hit the network despite an existing cache")
    monkeypatch.setattr(equity.requests, "get", _boom)

    acs = equity.fetch_acs()
    assert len(acs) > 0


def test_block_group_geometries_cover_exactly_the_configured_counties():
    """Updated 2026-08-12: the overlay is TWO counties (Riley 161 + Pottawatomie 149),
    not Riley alone -- equity.COUNTIES, not equity.COUNTY. The old single-county
    assertion silently passed only because the geography had not yet changed."""
    bg = equity.block_group_geometries()
    assert len(bg) > 0
    assert set(bg.columns) >= {"GEOID", "geometry"}
    prefixes = tuple(equity.STATE + c for c in equity.COUNTIES)
    assert bg["GEOID"].str.startswith(prefixes).all()
    # both counties must actually be represented, or the second is silently absent
    for pref in prefixes:
        assert bg["GEOID"].str.startswith(pref).any(), f"no block groups for {pref}"


def test_equity_table_row_count_and_columns(table):
    """Pinned: 56 of 76 block groups across BOTH counties (Riley 161 +
    Pottawatomie 149) have >= 50 valid susceptibility pixels (MIN_PIXELS gate) -- a
    regression guard on the real zonal-stats pipeline, not a hardcoded
    assumption inside pipeline/equity.py itself (which derives this count
    live). Was 51 of 59 when the overlay was Riley-only."""
    assert len(table) == 56
    assert {"susc_mean", "susc_n", "bg_pop_density_mean", "bg_pop_density_n",
            "GEOID", "median_income", "pct_18_24"}.issubset(set(table.columns))
    assert table["susc_mean"].notna().all()
    # cloglog output is a probability-like [0, 1] surface
    assert table["susc_mean"].between(0, 1).all()


def test_only_disability_remains_null_in_the_real_pull(table):
    """Empirically grounds which indicators are genuinely unavailable.

    Updated 2026-08-14 (plan 2.6). Poverty and vehicle access ARE published at
    block-group geography -- just not in the tables originally queried. C17002
    (ratio of income to poverty) and B25044 (tenure by vehicles available) both
    return complete estimates, so both are now tested. B17001 and B08201 remain
    empty, which is why the original exclusion looked correct.

    DISABILITY genuinely is unavailable: neither B18101 nor C18108 returns
    estimates for any block group in this vintage. If the Bureau ever backfills
    it, THIS test fails first and prompts a human to reconsider the family."""
    assert table["poverty_rate"].notna().sum() > 0, "C17002 should populate poverty_rate"
    assert table["pct_no_vehicle"].notna().sum() > 0, "B25044 should populate pct_no_vehicle"
    assert table["pct_disability"].notna().sum() == 0, \
        "disability is not published at block group; if it now is, revisit the family"


# ----------------------------------------------------------------------
# (e) correlations() -- the correction test, plus pinned
# actual observed numbers (honest reporting).
# ----------------------------------------------------------------------

def test_spearman_pvalues_are_corrected(corr):
    """The p-values must be corrected (using the module-scoped
    real ``corr`` fixture rather than a fresh call, so this doesn't
    re-trigger the whole ACS+zonal-stats pipeline a second time)."""
    assert corr["p_adjusted"].notna().all()


def test_n_tests_is_eight_only_disability_suppressed(corr):
    """Of the 9 candidate variables (8 demographic + population_density), only
    pct_disability has zero valid rows, leaving 8 actually tested.

    Was 6 until 2026-08-14, when poverty and vehicle access were recovered from
    C17002 and B25044 -- see test_only_disability_remains_null_in_the_real_pull.
    Growing the family also tightens both corrections,
    which is why the Holm and Benjamini--Hochberg pins below are re-derived
    rather than carried over."""
    assert corr.attrs["n_tests"] == 8
    assert len(corr) == 8
    # population density is corrected on its own (manuscript Methods), so the
    # sociodemographic Holm/BH family is the 7 indicators, not all 8 rows
    assert corr.attrs["n_tests_by_family"] == {"sociodemographic": 7, "coupled_predictor": 1}


# Observed Spearman results from the real ACS pull + the real published
# susceptibility raster. These are asserted below as INVARIANTS + loose BANDS, never the old
# ``rel=1e-4`` exact pins.
#
# Why band-pins, not exact pins: the published raster
# (``data/processed/maxent/final/cloglog/flood_avg.tif``) is a 10-replicate
# MaxEnt *bootstrap ensemble* fit with NO fixed CLI seed -- MaxEnt exposes no
# ``randomseed`` hook and ``pipeline.maxent.fit_final`` passes none (see that
# function's docstring). So ``susc_mean`` -- and every ``rho``/``p`` derived
# from it -- carries small, unavoidable run-to-run jitter (``bg_pop_density_
# mean`` rho was observed across regenerations in ~0.61-0.64). The old exact
# pins broke the instant the raster was *legitimately* regenerated, while the
# scientific CONCLUSION was invariant every time. We therefore assert the
# stable invariants (sole survivor, significance verdicts, signs) plus loose
# bands, adopting statistical -- not bit -- reproducibility for this ensemble.
#
# ``n`` (block-group count per indicator) IS deterministic: it comes from the
# ACS null pattern intersected with the deterministic 51-row zonal table
# (``test_equity_table_row_count_and_columns``: all 51 rows have non-null
# ``susc_mean``), NOT from the raster's numeric values -- so it stays an exact
# pin and still guards the zonal-stats pipeline against drift.
# Re-pinned 2026-08-12 for the TWO-county overlay (Riley 161 + Pottawatomie 149).
# Previously 51/45/50/50/45/50 when the overlay was Riley-only.
OBSERVED_N = {
    "bg_pop_density_mean": 56, "median_income": 49, "pct_elderly": 55,
    "pct_renter": 55, "rent_burden_pct": 48, "pct_nonwhite": 55,
    # added 2026-08-14: poverty and vehicle access ARE published at block group,
    # in C17002 and B25044 respectively
    "poverty_rate": 55, "pct_no_vehicle": 55,
}
DEMOGRAPHICS = ["median_income", "pct_elderly", "pct_renter", "rent_burden_pct",
                "pct_nonwhite", "poverty_rate", "pct_no_vehicle"]
# Sign of rho, asserted ONLY for indicators whose |rho| sits comfortably clear
# of zero (~0.2+, observed) so the sign cannot flip under the raster's small
# jitter. ``rent_burden_pct`` (~0.08) and ``pct_nonwhite`` (~0.06) sit
# essentially AT zero -- their sign is not a stable, meaningful quantity, so it
# is deliberately NOT asserted (only their non-significance is).
DEMOGRAPHIC_SIGN = {
    "poverty_rate": +1,    # strongest genuine indicator; raw-significant, NOT after Holm
    "median_income": -1,   # NEGATIVE income gradient; raw-significant, NOT after Holm
    "pct_elderly": -1,     # negative; raw-significant, NOT after Holm
    "pct_renter": +1,      # positive; raw-significant, NOT after Holm
}

# Indicators reaching NOMINAL (uncorrected) significance under the two-county
# overlay. Under the old Riley-only geography none of them did, so the original
# test asserted `significant_raw is False` for every demographic. That assertion
# is now wrong: the larger sample (55 vs 50 block groups) resolves three weak
# associations to p<0.05 raw. NONE survives Holm, so the paper's claim -- "no
# association surviving correction for multiple comparisons" -- is unaffected.
# Pinned explicitly rather than dropped: a change here is a real change in the
# equity result and should fail loudly.
RAW_SIGNIFICANT_DEMOGRAPHICS = {"median_income", "pct_elderly", "pct_renter",
                               "poverty_rate"}


def test_population_density_is_sole_strong_positive_survivor(corr):
    """``bg_pop_density_mean`` -- the variable module docstring correction 3
    discloses as MECHANICALLY coupled to the model, not an independent equity
    signal -- is the sole Holm survivor, a strong POSITIVE association.

    Band, not exact pin: the published raster is a non-deterministic bootstrap
    ensemble (see ``OBSERVED_N``'s comment), so rho jitters run-to-run
    (observed ~0.61-0.64) while the CONCLUSION -- sole significant survivor,
    strong positive -- is invariant. ``n`` is deterministic and stays pinned."""
    row = corr.loc[corr["variable"] == "bg_pop_density_mean"].iloc[0]
    assert bool(row["significant_corrected"]) is True
    assert row["rho"] > 0
    assert 0.55 <= row["rho"] <= 0.72
    assert int(row["n"]) == OBSERVED_N["bg_pop_density_mean"]


@pytest.mark.parametrize("variable", DEMOGRAPHICS)
def test_demographic_indicator_is_not_significant(corr, variable):
    """The honest equity null: every genuine sociodemographic indicator is
    NON-significant after Holm correction. Three (income, elderly, renter) DO
    reach nominal uncorrected significance under the two-county overlay -- see
    ``RAW_SIGNIFICANT_DEMOGRAPHICS`` -- which is why raw significance is pinned
    per-indicator rather than asserted uniformly False. The significance VERDICTS and the deterministic
    per-indicator block-group count ``n`` are pinned exactly; rho is NOT pinned
    (raster jitter -- see ``OBSERVED_N``'s comment), only its SIGN is checked,
    and only where |rho| is clear enough of zero to have a stable sign
    (``DEMOGRAPHIC_SIGN``)."""
    row = corr.loc[corr["variable"] == variable].iloc[0]
    assert bool(row["significant_corrected"]) is False
    assert bool(row["significant_raw"]) is (variable in RAW_SIGNIFICANT_DEMOGRAPHICS)
    assert int(row["n"]) == OBSERVED_N[variable]
    if variable in DEMOGRAPHIC_SIGN:
        assert np.sign(row["rho"]) == DEMOGRAPHIC_SIGN[variable]


def test_only_population_density_survives_correction_honest_finding(corr):
    """THE central honest result (if most associations are not
    significant after correction, say so plainly). Only the
    mechanically-coupled population_density variable survives Holm
    correction; every genuine sociodemographic indicator -- including
    median household income -- does not.

    Updated 2026-08-12 for the two-county overlay: three demographics now DO
    reach nominal uncorrected p<0.05 (see ``RAW_SIGNIFICANT_DEMOGRAPHICS``).
    The corrected result -- the one the paper rests on -- is unchanged."""
    survivors = corr.loc[corr["significant_corrected"], "variable"].tolist()
    assert survivors == ["bg_pop_density_mean"]

    demographic_rows = corr[~corr["coupled_with_predictor"]]
    raw_sig = set(demographic_rows.loc[demographic_rows["significant_raw"], "variable"])
    assert raw_sig == RAW_SIGNIFICANT_DEMOGRAPHICS, (
        f"raw-significant demographics changed: {sorted(raw_sig)}. This is a real "
        f"change in the equity result, not noise -- reconcile the manuscript."
    )
    assert not demographic_rows["significant_corrected"].any()


def test_income_susceptibility_gradient_is_not_significant(corr):
    """Explicit callout: is the income-
    susceptibility gradient significant?

    Under the primary Holm correction, NO. Updated 2026-08-12: under the
    two-county overlay it IS nominally significant (rho=-0.31, p_raw=0.028) and
    it also clears Benjamini--Hochberg (p_bh=0.0497 once population density is
    corrected in its own family; 0.045 when it shared the family). The paper's claim is
    Holm-specific and the manuscript already discloses the BH outcome; this test
    pins that distinction so it cannot regress into an unqualified 'not
    significant'."""
    row = corr.loc[corr["variable"] == "median_income"].iloc[0]
    assert row["p_adjusted"] > 0.05          # Holm -- the primary correction
    assert row["p_raw"] < 0.05               # but nominally significant
    assert row["p_adjusted_fdr_bh"] < 0.05   # and it clears BH -- disclosed, not hidden
    assert row["significant_raw"]            # nominally yes (was False, Riley-only)
    assert not row["significant_corrected"]  # under Holm, no -- the paper's claim


def test_coupled_with_predictor_flags_only_population_density(corr):
    coupled = corr.loc[corr["coupled_with_predictor"], "variable"].tolist()
    assert coupled == ["bg_pop_density_mean"]


def test_p_adjusted_never_less_than_p_raw(corr):
    assert (corr["p_adjusted"] >= corr["p_raw"] - 1e-12).all()


def test_p_adjusted_fdr_bh_reported_alongside_primary_holm(corr):
    """Secondary FDR column present for transparency about correction-method
    sensitivity.

    Updated 2026-08-12 -- and this reverses the previous claim. Under the
    Riley-only overlay the BH set matched the Holm set, so the conclusion was
    robust to the choice of correction. Under the two-county overlay it is NOT:
    Holm keeps one survivor, BH keeps four. The manuscript already reports this
    ("three indicators fall below 0.05 with the unweighted mean" under BH), and
    it is precisely why Holm is named as primary rather than chosen post hoc.
    Pinned exactly so the divergence stays visible."""
    assert corr["p_adjusted_fdr_bh"].notna().all()
    bh_survivors = set(corr.loc[corr["p_adjusted_fdr_bh"] < 0.05, "variable"])
    holm_survivors = set(corr.loc[corr["p_adjusted"] < 0.05, "variable"])
    assert holm_survivors == {"bg_pop_density_mean"}
    assert bh_survivors == {"bg_pop_density_mean"} | RAW_SIGNIFICANT_DEMOGRAPHICS
    assert holm_survivors < bh_survivors, "Holm must be no less strict than BH"


# ----------------------------------------------------------------------
# (f) Metadata: population_density coupling disclosure + caveats +
# exploratory/non-causal framing (all required).
# ----------------------------------------------------------------------

def test_population_density_coupling_disclosure_present_and_specific(corr):
    disclosure = corr.attrs["population_density_coupling"]
    assert isinstance(disclosure, str) and len(disclosure) > 0
    lower = disclosure.lower()
    assert "predictor" in lower
    assert "mechanical" in lower
    assert "population_density" in lower
    # must cite concrete, live-derived evidence, not just an assertion
    assert "lambda" in lower or "coefficient" in lower


def test_population_density_lambda_summary_matches_disk(table):
    summ = equity.population_density_lambda_summary()
    assert summ["n_replicates"] == 10
    assert summ["mean"] > 0
    assert summ["min"] > 0  # positive in EVERY replicate, not just on average
    assert summ["max"] >= summ["min"]


def test_caveats_present_maup_ecological_fallacy_moe_student_housing(corr):
    caveats = corr.attrs["caveats"]
    assert set(caveats) == {"maup", "ecological_fallacy", "acs_margin_of_error",
                            "student_housing_confound"}
    assert "modifiable areal unit" in caveats["maup"].lower()
    assert "ecological fallacy" in caveats["ecological_fallacy"].lower()
    assert "margin" in caveats["acs_margin_of_error"].lower()
    assert "student" in caveats["student_housing_confound"].lower()
    assert "ksu" in caveats["student_housing_confound"].lower()


def test_student_housing_caveat_grounded_in_real_block_group_count(corr):
    """The caveat must cite the ACTUAL observed count, not a generic
    sentence -- pinned: 19 of 55 block groups meet the >=30% aged-18-24
    threshold in the real two-county pull (was 19 of 50 under Riley-only).
    The manuscript quotes the same 19/55."""
    text = corr.attrs["caveats"]["student_housing_confound"]
    assert "19" in text
    assert "55" in text


def test_framing_is_exploratory_correlational_not_causal(corr):
    assert corr.attrs["kind"] == "exploratory_correlational"
    assert corr.attrs["causal"] is False
    framing = corr.attrs["framing"].lower()
    assert "causal" in framing
    assert "not" in framing  # "never a causal estimate" / "not validated"


def test_module_never_claims_validated_or_proven():
    """Framing-enforcement check on the module source itself -- mirrors
    pipeline.evaluate's/pipeline.screen's own no-overclaim guards. This
    equity overlay must never describe its own correlations as
    'validated', 'proven', or a demonstrated 'disparity'."""
    import inspect
    src = inspect.getsource(equity).lower()
    forbidden = ["is validated", "proves that", "demonstrated disparity",
                 "causal effect of"]
    for phrase in forbidden:
        assert phrase not in src, f"overclaiming phrase found in source: {phrase!r}"


# ----------------------------------------------------------------------
# (g) Orchestration
# ----------------------------------------------------------------------

def test_run_ties_everything_together():
    result = equity.run()
    # income_quintiles added 2026-08-14: run() now also persists the quintile
    # summary, which previously had no producer in pipeline/ at all.
    assert set(result) == {"table", "correlations", "index_composition",
                           "income_quintiles"}
    assert "vulnerability_index" in result["table"].columns
    assert "adaptive_capacity_index" in result["table"].columns
    assert result["correlations"]["p_adjusted"].notna().all()
    assert result["index_composition"]["vulnerability"] == equity.index_composition()["vulnerability"]


# ============================================================================
# Two-county geography. Manhattan spans Riley AND Pottawatomie, and the
# susceptibility surface is modelled over both; analysing equity on Riley alone
# silently covered less ground than the model it is describing.
# ============================================================================

def test_equity_counties_match_predictors():
    """The equity county list must equal the predictor county list. If they
    diverge again, the equity analysis is describing a different area than the
    surface it samples."""
    from pipeline import predictors
    assert set(equity.COUNTIES) == set(predictors.CENSUS_COUNTIES), (
        f"equity.COUNTIES={equity.COUNTIES} != "
        f"predictors.CENSUS_COUNTIES={predictors.CENSUS_COUNTIES}"
    )
    assert "149" in equity.COUNTIES, "Pottawatomie (149) must be included"


def test_block_group_geometries_covers_both_counties():
    if not equity.BG_CACHE.exists():
        pytest.skip(f"{equity.BG_CACHE} not cached locally")
    bg = equity.block_group_geometries()
    counties = {g[2:5] for g in bg["GEOID"]}
    assert counties == set(equity.COUNTIES), f"got counties {counties}"
    assert len(bg) > 50


def test_dasymetric_columns_present_and_bounded():
    """The dasymetric statistics must exist and be restricted to developed land
    -- population_density cannot serve as the weight because it is rasterised
    from the block-group polygons and is constant within each one."""
    import inspect
    src = inspect.getsource(equity._zonal_dasymetric)
    assert "NLCD_DEVELOPED" in src
    assert "population_density" in src.lower()      # the no-op weight is documented
    assert equity.NLCD_DEVELOPED == (21, 22, 23, 24)

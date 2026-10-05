"""Tests for pipeline.maxent's variable-importance combining logic:
jackknife unique contribution (``gain_all - gain_without``) + bootstrap
permutation importance/percent contribution (the final model) +
an optional Mann-Whitney rank-biserial join (screen.bivariate()'s output).

Two tiers, mirroring test_maxent_aicc.py's/test_model_families.py's own
scoping precedent ("pure-logic vs. real-subprocess-run"):

1. Pure-logic / synthetic-data tests for ``_categorical_layers_present``,
   ``_build_layer_subset``, ``_read_regularized_gain``,
   ``_bootstrap_importance``, and ``combine_variable_importance`` -- fast,
   no MaxEnt subprocess, no real predictor stack, no real bootstrap output.
2. A handful of tests that call the REAL ``maxent.variable_importance_combined()``
   orchestrator (itself pure CSV I/O + merge -- no MaxEnt subprocess) against
   the REAL on-disk ``data/processed/maxent/jackknife_gains.csv`` (the
   output of ~35 leave-one-out/single-variable MaxEnt fits at
   FINAL_CONFIG, run once via a standalone script, never inside pytest --
   same rationale test_maxent_aicc.py gives for ``aicc_grid()``) and the
   real final-model bootstrap ``maxentResults.csv``. These assert
   real structural/sanity properties and PIN the ACTUAL observed top
   predictor -- never a value assumed in advance.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import maxent

# ============================================================================
# Tier 1 -- pure logic / synthetic data, no I/O beyond tmp_path
# ============================================================================

# --------------------------------------------------------------------------
# _categorical_layers_present -- run()'s togglelayertype= flags must be
# computed from what's actually in a (possibly reduced) layers_dir, not
# unconditionally emitted for both of screen.CATEGORICAL's names -- Task
# 6.3's single-variable/leave-one-out subset dirs may contain neither, one,
# or both categorical layers.
# --------------------------------------------------------------------------

def test_categorical_layers_present_detects_subset(tmp_path):
    (tmp_path / "nlcd_landcover.asc").write_text("x")
    (tmp_path / "slope.asc").write_text("x")
    assert maxent._categorical_layers_present(tmp_path) == ["nlcd_landcover"]


def test_categorical_layers_present_detects_both(tmp_path):
    (tmp_path / "nlcd_landcover.asc").write_text("x")
    (tmp_path / "hydrologic_soil_group.asc").write_text("x")
    assert maxent._categorical_layers_present(tmp_path) == [
        "hydrologic_soil_group", "nlcd_landcover",
    ]


def test_categorical_layers_present_empty_when_neither(tmp_path):
    (tmp_path / "slope.asc").write_text("x")
    assert maxent._categorical_layers_present(tmp_path) == []


# --------------------------------------------------------------------------
# _build_layer_subset -- idempotent symlink-based reduced-layer directories
# (the mechanism behind gain_all/gain_without/gain_only, never copying the
# (tens-of-MB) .asc grids).
# --------------------------------------------------------------------------

def test_build_layer_subset_symlinks_only_requested_layers(tmp_path):
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    for name in ["a", "b", "c"]:
        (layers_dir / f"{name}.asc").write_text(f"data-{name}")

    subset_dir = tmp_path / "subset"
    result = maxent._build_layer_subset(["a", "c"], subset_dir, layers_dir=layers_dir)

    assert result == subset_dir
    present = sorted(p.name for p in subset_dir.glob("*.asc"))
    assert present == ["a.asc", "c.asc"]
    assert (subset_dir / "a.asc").read_text() == "data-a"


def test_build_layer_subset_is_idempotent_and_drops_stale_entries(tmp_path):
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    for name in ["a", "b"]:
        (layers_dir / f"{name}.asc").write_text(name)

    subset_dir = tmp_path / "subset"
    maxent._build_layer_subset(["a", "b"], subset_dir, layers_dir=layers_dir)
    # Rebuild with a SMALLER subset reusing the same directory -- "b" must
    # be dropped, not left behind as a stale extra layer.
    maxent._build_layer_subset(["a"], subset_dir, layers_dir=layers_dir)

    present = sorted(p.name for p in subset_dir.glob("*.asc"))
    assert present == ["a.asc"]


def test_build_layer_subset_second_call_is_a_cheap_noop(tmp_path):
    """Calling twice with the SAME names must not raise (symlinks already
    exist) and must leave the directory in the same state."""
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "a.asc").write_text("a")
    subset_dir = tmp_path / "subset"

    maxent._build_layer_subset(["a"], subset_dir, layers_dir=layers_dir)
    maxent._build_layer_subset(["a"], subset_dir, layers_dir=layers_dir)  # must not raise

    assert sorted(p.name for p in subset_dir.glob("*.asc")) == ["a.asc"]


def test_build_layer_subset_missing_source_raises(tmp_path):
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        maxent._build_layer_subset(["nope"], tmp_path / "subset", layers_dir=layers_dir)


# --------------------------------------------------------------------------
# _read_regularized_gain -- one scalar column out of a MaxEnt maxentResults.csv
# --------------------------------------------------------------------------

def test_read_regularized_gain_reads_named_species_row(tmp_path):
    df = pd.DataFrame({
        "Species": ["flood"],
        "#Training samples": [277],
        "Regularized training gain": [2.4799],
    })
    path = tmp_path / "maxentResults.csv"
    df.to_csv(path, index=False)
    assert maxent._read_regularized_gain(path) == pytest.approx(2.4799)


def test_read_regularized_gain_falls_back_when_species_name_differs(tmp_path):
    df = pd.DataFrame({
        "Species": ["not_flood"],
        "Regularized training gain": [1.234],
    })
    path = tmp_path / "maxentResults.csv"
    df.to_csv(path, index=False)
    assert maxent._read_regularized_gain(path, species="flood") == pytest.approx(1.234)


# --------------------------------------------------------------------------
# _bootstrap_importance -- tidy per-predictor mean/sd from the wide
# "<var> contribution"/"<var> permutation importance" bootstrap-replicate
# columns; must average ONLY the flood_<i> replicate rows, never MaxEnt's
# own "flood (average)" summary row (same pitfall apparent_auc's own
# regex already guards against).
# --------------------------------------------------------------------------

def test_bootstrap_importance_averages_replicates_and_excludes_average_row(tmp_path):
    df = pd.DataFrame({
        "Species": ["flood_0", "flood_1", "flood_2", "flood (average)"],
        "a contribution": [10.0, 20.0, 30.0, 999.0],
        "b contribution": [1.0, 1.0, 1.0, 999.0],
        "a permutation importance": [50.0, 60.0, 70.0, 999.0],
        "b permutation importance": [2.0, 2.0, 2.0, 999.0],
    })
    path = tmp_path / "maxentResults.csv"
    df.to_csv(path, index=False)

    result = maxent._bootstrap_importance(path, ["a", "b"]).set_index("predictor")

    assert set(result.index) == {"a", "b"}
    assert result.loc["a", "percent_contribution_mean"] == pytest.approx(20.0)
    assert result.loc["a", "permutation_importance_mean"] == pytest.approx(60.0)
    assert result.loc["b", "percent_contribution_sd"] == pytest.approx(0.0)  # constant column


def test_bootstrap_importance_falls_back_to_single_run_row(tmp_path):
    df = pd.DataFrame({
        "Species": ["flood"],
        "a contribution": [42.0],
        "a permutation importance": [17.0],
    })
    path = tmp_path / "maxentResults.csv"
    df.to_csv(path, index=False)

    result = maxent._bootstrap_importance(path, ["a"]).set_index("predictor")
    assert result.loc["a", "percent_contribution_mean"] == pytest.approx(42.0)
    assert result.loc["a", "percent_contribution_sd"] == 0.0
    assert result.loc["a", "permutation_importance_mean"] == pytest.approx(17.0)


def test_bootstrap_importance_missing_column_raises(tmp_path):
    df = pd.DataFrame({
        "Species": ["flood_0"],
        "a contribution": [10.0],
        "a permutation importance": [50.0],
    })
    path = tmp_path / "maxentResults.csv"
    df.to_csv(path, index=False)
    with pytest.raises(KeyError):
        maxent._bootstrap_importance(path, ["a", "b"])  # "b" columns absent


# --------------------------------------------------------------------------
# combine_variable_importance -- the join + sort logic, unit-tested on a
# small synthetic gains frame.
# --------------------------------------------------------------------------

def _toy_jk():
    return pd.DataFrame({
        "predictor": ["p1", "p2", "p3"],
        "gain_only": [0.10, 0.50, 0.05],
        "unique_contribution": [0.02, 0.40, 0.00],
    })


def _toy_boot():
    return pd.DataFrame({
        "predictor": ["p1", "p2", "p3"],
        "permutation_importance_mean": [10.0, 70.0, 5.0],
        "permutation_importance_sd": [1.0, 2.0, 0.5],
        "percent_contribution_mean": [15.0, 60.0, 8.0],
        "percent_contribution_sd": [1.0, 3.0, 0.5],
    })


def test_combine_variable_importance_sorts_by_unique_contribution_desc():
    combined = maxent.combine_variable_importance(_toy_jk(), _toy_boot())
    assert list(combined["predictor"]) == ["p2", "p1", "p3"]


def test_combine_variable_importance_has_expected_columns_without_bivariate():
    combined = maxent.combine_variable_importance(_toy_jk(), _toy_boot())
    assert list(combined.columns) == [
        "predictor", "gain_only", "unique_contribution",
        "permutation_importance_mean", "permutation_importance_sd",
        "percent_contribution_mean", "percent_contribution_sd",
    ]


def test_combine_variable_importance_joins_mw_rank_biserial_when_given():
    biv = pd.DataFrame({
        "predictor": ["p1", "p2", "p3"],
        "effect": [0.5, -0.3, 0.9],
        "effect_metric": ["rank-biserial r", "Cramer's V", "rank-biserial r"],
    })
    combined = maxent.combine_variable_importance(_toy_jk(), _toy_boot(), bivariate_df=biv)
    row = combined.set_index("predictor")
    assert row.loc["p1", "mw_rank_biserial"] == pytest.approx(0.5)
    assert row.loc["p3", "mw_rank_biserial"] == pytest.approx(0.9)
    # p2's bivariate metric is Cramer's V (categorical) -- must NEVER be
    # silently mislabeled as a rank-biserial r value.
    assert np.isnan(row.loc["p2", "mw_rank_biserial"])


def test_combine_variable_importance_without_bivariate_omits_the_column():
    combined = maxent.combine_variable_importance(_toy_jk(), _toy_boot())
    assert "mw_rank_biserial" not in combined.columns


def test_combine_variable_importance_raises_on_predictor_mismatch():
    """A predictor present in one source but not the other must be caught
    loudly, never silently dropped via an inner join."""
    boot = _toy_boot().iloc[:2].copy()  # drop p3
    with pytest.raises(AssertionError):
        maxent.combine_variable_importance(_toy_jk(), boot)


# ============================================================================
# Tier 2 -- real on-disk artifacts (the ~35-fit jackknife
# run + the final bootstrap model). No MaxEnt subprocess is invoked
# by these tests themselves -- variable_importance_combined() is pure CSV
# I/O + merge once jackknife_gains.csv exists on disk.
# ============================================================================

def _real_combined():
    return maxent.variable_importance_combined()


def test_real_combined_table_covers_all_17_modeling_predictors():
    modeling = set(maxent.modeling_layers())
    assert len(modeling) == 17
    df = _real_combined()
    assert set(df["predictor"]) == modeling


def test_real_combined_table_has_expected_columns():
    df = _real_combined()
    expected = [
        "predictor", "gain_only", "unique_contribution",
        "permutation_importance_mean", "permutation_importance_sd",
        "percent_contribution_mean", "percent_contribution_sd",
        "mw_rank_biserial",
    ]
    assert list(df.columns) == expected


def test_real_combined_table_values_are_finite_and_sane():
    df = _real_combined()
    assert np.isfinite(df["gain_only"]).all()
    assert np.isfinite(df["unique_contribution"]).all()
    # Small negative slack allowed -- MaxEnt's convergence is numerical, so
    # "removing a variable never helps" can show up as a tiny negative
    # rounding artifact rather than an exact >= 0.
    assert (df["unique_contribution"] > -0.01).all()
    assert df["permutation_importance_mean"].between(0, 100).all()
    assert df["percent_contribution_mean"].between(0, 100).all()
    # Percent contributions should each be non-negative and roughly sum to
    # 100 across predictors (MaxEnt's own convention) -- loose tolerance,
    # this is a sanity check, not a re-derivation of MaxEnt's internal math.
    assert df["percent_contribution_mean"].sum() == pytest.approx(100.0, abs=2.0)


def test_real_top_predictor_by_unique_contribution_is_pinned():
    """Pinned to the ACTUAL observed result (real ~35-fit jackknife run):
    ``distance_to_outfall`` IS #1 by
    unique_contribution (0.1353, next is available_water_storage at
    0.0800) -- the "infrastructure-controlled" thesis holds by THIS metric
    on the new 17-predictor/log-screened/TWI-retained model, matching the
    OLD 21-predictor model's jackknife winner. See the permutation-importance
    test directly below for where the two metrics diverge."""
    df = _real_combined().sort_values("unique_contribution", ascending=False)
    assert df.iloc[0]["predictor"] == "distance_to_outfall"


def test_real_top_predictor_by_permutation_importance_is_pinned():
    """Pinned to the ACTUAL observed result -- NOT forced to
    distance_to_outfall. By permutation importance, ``available_water_storage``
    is #1 (37.79 +/- 11.51) with ``distance_to_outfall`` a close #2
    (33.43 +/- 16.78, heavily overlapping SDs -- effectively a statistical
    tie between an infrastructure variable and a soil-property variable,
    not a clean loss for either). This is a genuine divergence from
    unique_contribution's own #1 (see the test directly above) -- exactly
    the "JK-unique vs. permutation-importance divergence flags redundancy"
    pattern the original notebook's own §7 interpretation notes describe,
    and the reason this is not a clean confirmation of the infrastructure
    narrative."""
    df = _real_combined().sort_values("permutation_importance_mean", ascending=False)
    # The exact #1 is bootstrap-UNSTABLE: available_water_storage (37.79+/-11.51)
    # and distance_to_outfall (33.43+/-16.78) are an effective statistical tie
    # (heavily overlapping SDs), and permutation importance is read from the
    # non-deterministic 10-replicate final maxentResults.csv (no MaxEnt CLI seed
    # -- see pipeline.maxent.fit_final's docstring), so jitter flips which of the
    # tied pair lands #1. The invariant this test guards -- and the scientific
    # point in the docstring above -- is that this tied soil/infrastructure pair
    # LEADS permutation importance, diverging from unique_contribution's own
    # clean #1 (distance_to_outfall, the test above); the exact winner within the
    # pair is not the claim.
    assert df.iloc[0]["predictor"] in {"available_water_storage", "distance_to_outfall"}

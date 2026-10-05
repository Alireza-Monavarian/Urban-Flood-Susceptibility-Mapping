"""Tests for pipeline.report -- the presentation layer (Tables
1/3/4 + core figures).

report.py is a THIN reader: every value asserted below must trace to an
already-produced artifact file (pipeline.maxent.VARIABLE_IMPORTANCE_CSV,
pipeline.sensitivity.SUMMARY_CSV) or to pipeline.hwm's own fast, seeded
parsing functions (raw_counts/build_presence) -- never a MaxEnt re-fit,
never a re-run of screening/sensitivity/robustness. No test here should
take more than a couple of seconds.

STALE-ARTIFACT GUARD: the pre-pipeline legacy CSV at
data/processed/sensitivity/sensitivity_summary.csv reports a DIFFERENT
(resubstitution-flavored) baseline AUC of 0.9842. The canonical CSV this
module must read is data/processed/maxent/sensitivity/sensitivity_summary.
csv (pipeline.sensitivity.SUMMARY_CSV), whose baseline is 0.9826.
test_table4_baseline_and_stormwater_bound_values below asserts the
canonical number AND explicitly asserts against the stale one, so a
future accidental re-point at the legacy path fails loudly.
"""
import pytest
from PIL import Image

from pipeline import evaluate, model_compare, report


# ============================================================================
# Table 1 -- flood-observation inventory.
# ============================================================================

def test_table1_raw_hw_is_595_and_eow_used():
    t1 = report.table1()
    assert t1.attrs["raw_hw"] == 595 and t1.attrs["eow_status"] == "used"


def test_table1_hw_and_eow_rows_trace_to_hwm():
    """Cross-checks table1() against the SAME two facts tests/test_hwm.py
    already pins independently (test_raw_hw_total_is_595 /
    test_presence_has_eow_and_286): raw HW=595, and after 30 m thinning
    279 HW + 7 EOW = 286 survive -- EOW points are retained, not dropped."""
    t1 = report.table1()
    hw_row = t1.loc[t1["Class"].str.contains("HW")].iloc[0]
    eow_row = t1.loc[t1["Class"].str.contains("EOW")].iloc[0]
    assert hw_row["Raw count"] == 595
    assert hw_row["Retained after 30 m thinning"] == 279
    assert eow_row["Raw count"] == 10
    assert eow_row["Retained after 30 m thinning"] == 7


def test_table1_total_row_sums_the_observation_classes():
    t1 = report.table1()
    total_row = t1.loc[t1["Class"] == "Total"].iloc[0]
    class_rows = t1.loc[t1["Class"] != "Total"]
    assert total_row["Raw count"] == class_rows["Raw count"].sum() == 605
    assert total_row["Retained after 30 m thinning"] == class_rows["Retained after 30 m thinning"].sum() == 286


# ============================================================================
# Table 3 -- variable importance (data/processed/maxent/
# variable_importance_combined.csv, via pipeline.maxent.VARIABLE_IMPORTANCE_CSV).
# ============================================================================

def test_table3_top_predictor_is_distance_to_outfall():
    """The headline result: distance_to_outfall ranks #1
    by gain_only/unique_contribution/percent_contribution alike."""
    t3 = report.table3()
    assert t3.iloc[0]["predictor"] == "distance_to_outfall"


def test_table3_has_the_required_metric_columns():
    t3 = report.table3()
    for col in ("predictor", "gain_only", "unique_contribution",
                "permutation_importance_mean", "permutation_importance_sd",
                "percent_contribution_mean", "percent_contribution_sd",
                "mw_rank_biserial"):
        assert col in t3.columns


def test_table3_is_sorted_top_predictor_first():
    t3 = report.table3()
    gains = t3["unique_contribution"].to_numpy()
    assert (gains[:-1] >= gains[1:]).all(), "table3() must be sorted so the top predictor is row 0"


# ============================================================================
# Table 4 -- ascertainment / sensitivity arms (pipeline.sensitivity.SUMMARY_CSV
# -- the canonical maxent/sensitivity/ file, never the stale sensitivity/ one).
# ============================================================================

def test_table4_has_stormwater_arm():
    assert any("stormwater" in r.lower() for r in report.table4()["Arm"])


def test_table4_outfall_distance_matched_arm_present_auc_near_0_929():
    """The NEW, proper (upper) ascertainment bound: real signal survives
    well above the 0.5 chance line even once distance_to_outfall is
    stripped out of the background's own distribution."""
    t4 = report.table4()
    row = t4.loc[t4["Arm"] == "bg_outfall_distance_matched"]
    assert len(row) == 1
    assert round(float(row.iloc[0]["AUC"]), 3) == 0.929


def test_table4_baseline_and_stormwater_bound_values():
    """Pins the two other canonical numbers straight off the real CSV
    (never fabricated) -- and explicitly guards against the STALE legacy
    file's 0.9842 ever silently becoming the baseline again."""
    t4 = report.table4()
    baseline = t4.loc[t4["Arm"] == "baseline"].iloc[0]
    sw = t4.loc[t4["Arm"] == "bg_target_group_stormwater_buffer"].iloc[0]

    assert baseline["AUC"] == pytest.approx(0.9826, abs=1e-3)
    assert baseline["AUC"] != pytest.approx(0.9842, abs=1e-4), (
        "table4() baseline must never reproduce the STALE legacy "
        "data/processed/sensitivity/sensitivity_summary.csv's resubstitution number"
    )
    assert sw["AUC"] == pytest.approx(0.9399, abs=1e-3)
    assert sw["Matched-baseline AUC"] == pytest.approx(0.9344, abs=1e-3)


def test_table4_required_columns_present():
    t4 = report.table4()
    for col in ("Arm", "Description", "AUC", "Matched-baseline AUC"):
        assert col in t4.columns


def test_table4_arm_order_is_sensible():
    """baseline first, then the ascertainment-bound background arms, then
    the remaining robustness arms (resolution/logistic/event-pooling)."""
    arms = list(report.table4()["Arm"])
    assert arms[0] == "baseline"
    assert arms.index("bg_target_group_stormwater_buffer") < arms.index("resolution_30m")
    assert arms.index("bg_outfall_distance_matched") < arms.index("output_format_logistic")
    assert arms.index("bg_outfall_distance_matched") < arms.index("event_pooling_top4")


def test_table4_includes_every_arm_from_the_canonical_csv():
    """No silent drop: every arm sensitivity.run() wrote to SUMMARY_CSV must
    surface somewhere in table4()."""
    from pipeline import sensitivity
    import pandas as pd
    on_disk = set(pd.read_csv(sensitivity.SUMMARY_CSV)["arm"])
    assert set(report.table4()["Arm"]) == on_disk


# ============================================================================
# figures() -- non-interactive backend; real PNGs written from the tables
# above. Content is never asserted, only existence + non-empty size.
# ============================================================================

def test_figures_returns_existing_nonempty_png_paths():
    paths = report.figures()
    assert len(paths) >= 2
    for p in paths:
        assert p.exists(), f"{p} was returned by figures() but does not exist"
        assert p.stat().st_size > 0, f"{p} exists but is empty"
        assert p.suffix == ".png"


def test_figures_writes_under_the_canonical_reports_figures_dir():
    from pipeline import config
    paths = report.figures()
    expected_dir = config.REPO / "reports" / "figures"
    for p in paths:
        assert p.parent == expected_dir


# ============================================================================
# Table 2 -- predictor stack (NEW; joins PREDICTOR_CATALOG with the live
# screen.BIVARIATE_REPORT_PATH / screen.REPORT_PATH artifacts).
# ============================================================================

def test_table2_has_expected_columns():
    t2 = report.table2()
    for col in ("Family", "Variable", "Source", "Type", "Effect (MW r / V)", "Sig", "Status"):
        assert col in t2.columns


def test_table2_status_distinguishes_retained_vs_dropped():
    """The canonical stack is 17 modeled predictors -- NOT the paper's
    stale '21 variables' framing."""
    t2 = report.table2()
    retained = t2.loc[t2["Status"] == "Retained (modeled)"]
    dropped = t2.loc[t2["Status"] != "Retained (modeled)"]
    assert len(retained) == 17
    assert len(dropped) == len(t2) - 17
    assert dropped["Status"].str.startswith("Dropped --").all()
    assert t2.attrs["n_modeled"] == 17


def test_table2_the_4_screening_dropped_predictors_show_dropped_with_reason():
    t2 = report.table2().set_index("Variable")
    for var in ("tri", "tpi", "flow_accumulation", "spi"):
        status = t2.loc[var, "Status"]
        assert status.startswith("Dropped --")
        assert "screening" in status.lower()
        assert status != "Dropped --"   # a real reason string was appended, not left blank


def test_table2_median_income_dropped_at_modeling_stage_not_screening():
    """median_income SURVIVES multicollinearity screening (it's in the live
    bivariate characterization, which only ever runs on the screened
    stack) but is dropped afterward for a structural ACS coverage gap."""
    t2 = report.table2().set_index("Variable")
    status = t2.loc["median_income", "Status"]
    assert status.startswith("Dropped --")
    assert "modeling" in status.lower()
    assert t2.loc["median_income", "Effect (MW r / V)"] != "N/A"


def test_table2_precip_dropped_not_acquired():
    t2 = report.table2().set_index("Variable")
    for var in ("precip_annual_mean", "precip_rx1day"):
        status = t2.loc[var, "Status"]
        assert "not acquired" in status.lower()
        assert t2.loc[var, "Effect (MW r / V)"] == "N/A"


def test_table2_distance_to_outfall_present_with_strong_negative_effect_and_sig():
    t2 = report.table2().set_index("Variable")
    row = t2.loc["distance_to_outfall"]
    assert row["Status"] == "Retained (modeled)"
    assert row["Sig"] == "**"
    effect = float(row["Effect (MW r / V)"])
    assert effect < -0.5


def test_table2_full_considered_stack_is_24_rows():
    """The paper's own 21 plus tri/tpi/median_income, which its table omits."""
    t2 = report.table2()
    assert len(t2) == 24
    assert t2.attrs["n_considered"] == 24


def test_table2_note_documents_the_stale_21_vs_canonical_17():
    t2 = report.table2()
    note = t2.attrs["note"].lower()
    assert "17" in note and "21" in note


# ============================================================================
# table3(top_n=...) -- new parameter; default stays the full 17 rows.
# ============================================================================

def test_table3_default_is_still_the_full_17_rows():
    t3 = report.table3()
    assert len(t3) == 17


def test_table3_top_n_returns_the_requested_prefix():
    full = report.table3()
    top10 = report.table3(top_n=10)
    assert len(top10) == 10
    assert list(top10["predictor"]) == list(full["predictor"])[:10]
    assert top10.iloc[0]["predictor"] == "distance_to_outfall"


# ============================================================================
# figures() -- full paper + nice-to-have suite, PNG(dpi=300)+PDF pairs.
# ============================================================================

_REQUIRED_PAPER_FIGURES = {
    "study_area_hwm", "susceptibility_maps", "jackknife_importance",
    "internal_validation_curves", "gfi_comparison", "sensitivity_summary",
}


def test_figures_produces_every_necessary_paper_figure_with_data():
    paths = report.figures()
    names = {p.stem for p in paths}
    assert _REQUIRED_PAPER_FIGURES <= names, (
        f"missing required paper figures: {_REQUIRED_PAPER_FIGURES - names}"
    )


def test_figures_writes_png_and_pdf_pair_for_every_figure():
    paths = report.figures()
    assert len(paths) >= len(_REQUIRED_PAPER_FIGURES)
    for p in paths:
        assert p.suffix == ".png"
        assert p.exists() and p.stat().st_size > 0
        pdf_path = p.with_suffix(".pdf")
        assert pdf_path.exists() and pdf_path.stat().st_size > 0, (
            f"{p} has no matching .pdf sibling"
        )


def test_figures_png_dpi_is_300():
    paths = report.figures()
    for p in paths:
        with Image.open(p) as im:
            dpi = im.info.get("dpi")
            assert dpi is not None, f"{p} has no DPI metadata"
            assert abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1, f"{p} dpi={dpi}, expected ~300"


# ============================================================================
# susceptibility_maps_comparison -- figure #7. READS model_compare.RASTER_DIR
# (populated upstream by model_compare.susceptibility_rasters(), NEVER by
# report.py itself -- report.py stays read-only) plus evaluate.EVAL_RASTER
# directly for MaxEnt's own panel. These tests supersede the OLD
# "permanently skipped" test from when this figure was a deliberate,
# unconditional data gap (see this module's docstring history) -- the gap
# is now closed, so the contract is "skip gracefully if the rasters are
# absent, produce a real 4-panel figure once they exist" instead.
# ============================================================================

def test_susceptibility_maps_comparison_gracefully_skips_when_rasters_absent(tmp_path, monkeypatch):
    """Environment-independent regression test for the skip-with-log path:
    monkeypatches model_compare.RASTER_DIR to an EMPTY tmp dir so this
    holds regardless of whether the real rasters happen to exist locally
    -- never the stale legacy PNG, never a raised error."""
    monkeypatch.setattr(model_compare, "RASTER_DIR", tmp_path / "no_rasters_here")
    assert report._fig_susceptibility_maps_comparison() is None


def test_susceptibility_maps_comparison_skip_prints_a_clear_log_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(model_compare, "RASTER_DIR", tmp_path / "no_rasters_here")
    report._fig_susceptibility_maps_comparison()
    captured = capsys.readouterr()
    assert "skipping susceptibility_maps_comparison" in captured.out


def test_figures_omits_the_4model_comparison_map_when_rasters_absent(tmp_path, monkeypatch, capsys):
    """figures() itself must never raise or fabricate this figure when its
    rasters are missing -- exercises the same graceful-skip contract
    through the public figures() list rather than the private builder."""
    monkeypatch.setattr(model_compare, "RASTER_DIR", tmp_path / "no_rasters_here")
    paths = report.figures()
    names = {p.stem for p in paths}
    assert "susceptibility_maps_comparison" not in names
    captured = capsys.readouterr()
    assert "skipping susceptibility_maps_comparison" in captured.out


# ----------------------------------------------------------------------
# Real-fixture: once model_compare.susceptibility_rasters() has actually
# been run (the notebook's own Stage-4 order), the figure IS produced.
# Skipped (not failed) if the real rasters aren't present locally.
# ----------------------------------------------------------------------

def _real_comparison_rasters_present():
    required = [model_compare.RASTER_DIR / f"{stem}.tif"
                for stem in ("logistic_regression", "random_forest", "xgboost")]
    return evaluate.EVAL_RASTER.exists() and all(p.exists() for p in required)


def test_susceptibility_maps_comparison_is_produced_when_rasters_exist():
    if not _real_comparison_rasters_present():
        pytest.skip("model_compare.susceptibility_rasters() has not been run locally yet")

    png_path = report._fig_susceptibility_maps_comparison()
    assert png_path is not None
    assert png_path.exists() and png_path.stat().st_size > 0
    pdf_path = png_path.with_suffix(".pdf")
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    with Image.open(png_path) as im:
        dpi = im.info.get("dpi")
        assert dpi is not None and abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1


def test_susceptibility_maps_comparison_appears_in_figures_when_rasters_exist():
    if not _real_comparison_rasters_present():
        pytest.skip("model_compare.susceptibility_rasters() has not been run locally yet")

    paths = report.figures()
    names = {p.stem for p in paths}
    assert "susceptibility_maps_comparison" in names


# ============================================================================
# tables() -- CSV + LaTeX for tables 1-4 (+ nice-to-haves when available).
# ============================================================================

def test_tables_returns_all_four_required_tables():
    result = report.tables()
    for name in ("table1", "table2", "table3", "table4"):
        assert name in result
        assert len(result[name]) > 0


def test_tables_writes_csv_and_tex_for_tables_1_through_4():
    from pipeline import config
    report.tables()
    tables_dir = config.REPO / "reports" / "tables"
    for name in ("table1", "table2", "table3", "table4"):
        csv_path = tables_dir / f"{name}.csv"
        tex_path = tables_dir / f"{name}.tex"
        assert csv_path.exists() and csv_path.stat().st_size > 0
        assert tex_path.exists() and tex_path.stat().st_size > 0
        assert "\\toprule" in tex_path.read_text()

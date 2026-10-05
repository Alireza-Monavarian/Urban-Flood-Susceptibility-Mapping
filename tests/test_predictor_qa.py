"""Tests for pipeline.predictors' QA gates -- valid_mask(),
qa_report() -- and pipeline.config.canonical_samples()'s reconciled wiring.

Deliberately reads the REAL on-disk 22-layer predictor stack
(data/processed/predictors/*.tif) and the REAL hwm.build_presence() 286
-point set, not a synthetic fixture: the whole point is to
VERIFY (not assume) the empirical 286->N valid-point count.

Empirical result: valid_mask().sum() == 279, not the originally
expected 278. This is isolated cleanly to median_income's own exclusive
-NaN count (6 points here vs an implied 5 in the source notebook's run) --
the 4-layer gNATSGO/curve_number soil cluster that excludes the other 7
points is unchanged and matches the notebook's own reported numbers
exactly (273/286 valid across all 22 raw layers). Nothing here is
hardcoded to force either 278 or 279 into being true independent of the
actual raster data -- valid_mask()/qa_report() compute both counts from
whatever is on disk.

UPDATE (canonical n=277): ``valid_mask()`` itself is UNCHANGED by this update and
still reports 279 (point-validity only). ``config.canonical_samples()``,
however, now ADDITIONALLY deduplicates those 279 valid points to one point
per DISTINCT 10 m modeling-grid pixel -- 2 pixels each hold 2 of the 279
points, so the canonical MODELING count is 277, not 279. This is a
SEPARATE, later reduction from the 279-vs-278 gap documented above (that
gap is about point VALIDITY; this one is about pixel UNIQUENESS on top of
validity) -- see the "config.canonical_samples() reconciliation" section
below for the pinned 277 tests.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import config, hwm, predictors, qa


# --------------------------------------------------------------------------
# valid_mask()
# --------------------------------------------------------------------------

def test_valid_mask_shape_and_dtype():
    mask = predictors.valid_mask()
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert len(mask) == 286
    assert list(mask.index) == list(range(286))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The original target: "
        "valid_mask().sum() was expected to be 278. Verified empirically "
        "against the real 22-layer on-disk stack + real "
        "hwm.build_presence(), "
        "it is 279 -- one point is recovered by dropping median_income "
        "that the source notebook's own run did not recover, most likely "
        "because acquire_census()'s re-pull (fresh ACS 2018-2022 "
        "5-yr estimates against current TIGER2023 block-group boundaries) "
        "has a 1-block-group-different missing-estimate footprint than "
        "whatever vintage/boundary produced the notebook's original "
        "median_income.tif. NOT hardcoded away -- see "
        "test_valid_presence_actual_count_is_279 below for the real, "
        "currently-true regression guard, and qa_report()['presence_points'] "
        "for the full accounting. strict=True: this loudly XPASS-fails, "
        "forcing a human to look, if the count ever reverts to 278 "
        "unnoticed. CLARIFICATION: this xfail is about valid_mask()'s point-VALIDITY count "
        "only (278 vs 279) and is UNCHANGED/unaffected by the separate, later "
        "277 pixel-uniqueness dedup config.canonical_samples() now applies on "
        "top of valid_mask()'s 279 -- see "
        "test_canonical_samples_dedups_279_valid_points_to_277_modeling_pixels "
        "below for that distinct guard."
    ),
)
def test_valid_presence_is_278():
    # The original 278 target, kept as a strict xfail.
    assert int(predictors.valid_mask().sum()) == 278


def test_valid_presence_actual_count_is_279():
    """The REAL, currently-true regression guard (see the xfail above for
    why this isn't 278 -- a documented, understood, one-point gap, not a
    bug in valid_mask()). If a future predictor-stack rebuild changes this
    number, THIS test should start failing -- that is the point: it pins
    down today's verified truth, not a hoped-for target.
    """
    assert int(predictors.valid_mask().sum()) == 279


def test_valid_mask_matches_qa_report_accounting():
    """valid_mask() and qa_report() must derive the SAME validity count --
    one source of truth (_validity_from_samples), not two independently
    -drifting computations.
    """
    n_valid = int(predictors.valid_mask().sum())
    report = predictors.qa_report()
    assert report["presence_points"]["valid_modeling_layers_21_excl_median_income"] == n_valid


def test_checkpoint_assert_counts_against_278_documents_the_gap():
    """The originally specified checkpoint --
    ``assert_counts(model_points=(278, valid_mask().sum()))`` -- run for
    real. It raises (278 != 279), which IS the correct, honest outcome
    given the empirical count; this test pins that down explicitly instead
    of quietly dropping the checkpoint, and would need deliberate updating
    (not silent deletion) if the underlying count is ever reconciled to
    278.
    """
    with pytest.raises(AssertionError, match="model_points"):
        qa.assert_counts(model_points=(278, int(predictors.valid_mask().sum())))


# --------------------------------------------------------------------------
# qa_report() -- alignment audit
# --------------------------------------------------------------------------

def test_qa_report_alignment_audit_all_22_layers_aligned():
    report = predictors.qa_report()
    align = report["alignment"]
    assert align["n_layers_checked"] == 22
    assert align["n_mismatched"] == 0
    assert align["reference_crs"] == "EPSG:26914"
    assert align["reference_shape"] == (1588, 2055)
    assert len(align["per_layer_nan_pct"]) == 22


# --------------------------------------------------------------------------
# qa_report() -- honest layer-count reconciliation
# --------------------------------------------------------------------------

def test_qa_report_layer_counts_reconcile_vs_notebook():
    counts = predictors.qa_report()["layer_counts"]
    assert (counts["raw"], counts["screened"], counts["modeled"]) == (22, 20, 19)
    assert (counts["notebook_raw"], counts["notebook_screened"], counts["notebook_modeled"]) == (24, 22, 21)
    # Canonical is exactly 2 below the notebook at every stage -- the 2
    # dropped-AT-ACQUISITION precip layers, never carried through
    # screening or modeling here (unlike the notebook, which kept them all
    # the way to its final 21-predictor model).
    assert counts["notebook_raw"] - counts["raw"] == 2
    assert counts["notebook_screened"] - counts["screened"] == 2
    assert counts["notebook_modeled"] - counts["modeled"] == 2
    assert set(counts["precip_layers_dropped_at_acquisition"]) == {"precip_annual_mean", "precip_rx1day"}
    assert counts["planned_multicollinearity_drops"] == ("tri", "tpi")
    assert counts["modeling_drops"] == ("median_income",)


def test_qa_report_layers_list_excludes_median_income_from_modeling_only():
    layers = predictors.qa_report()["layers"]
    assert len(layers["all_raw"]) == 22
    assert len(layers["modeling"]) == 21
    assert "median_income" in layers["all_raw"]
    assert "median_income" not in layers["modeling"]
    # tri/tpi are NOT yet dropped here -- that's the multicollinearity
    # screen, which runs later; this gate only removes median_income.
    assert "tri" in layers["modeling"] and "tpi" in layers["modeling"]


# --------------------------------------------------------------------------
# qa_report() -- per-layer exclusion-cause breakdown
# --------------------------------------------------------------------------

def test_qa_report_exclusion_causes_identify_median_income_and_soil_cluster():
    causes = predictors.qa_report()["exclusion_causes"]
    # The 4-layer gNATSGO/curve_number cluster always co-occurs (same 7 points).
    for layer in ("available_water_storage", "curve_number", "depth_to_restriction", "hydrologic_soil_group"):
        assert causes[layer] == 7
    assert causes["median_income"] == 6
    assert "out_of_bounds" not in causes  # all 286 points are in-bounds


def test_qa_report_presence_point_accounting_is_internally_consistent():
    pts = predictors.qa_report()["presence_points"]
    assert pts["total"] == 286
    assert pts["hw"] == 279
    assert pts["eow"] == 7
    assert pts["out_of_bounds"] == 0
    assert pts["valid_all_22_raw_layers"] == 273
    assert pts["excluded_any_of_22_raw_layers"] == 13
    assert pts["valid_modeling_layers_21_excl_median_income"] == 279
    assert pts["excluded_modeling_layers_21_excl_median_income"] == 7
    assert pts["recovered_by_dropping_median_income"] == 6
    assert pts["expected_modeling_valid_per_task_brief"] == 278
    assert pts["matches_expected_278"] is False


# --------------------------------------------------------------------------
# config.canonical_samples() reconciliation
# --------------------------------------------------------------------------

def test_canonical_samples_wired_to_real_valid_mask_and_presence():
    """``valid`` (point-validity) is 279; ``samples`` (canonical MODELING
    set, post pixel-dedup) is 277 -- two DIFFERENT numbers,
    not a typo (see this file's module docstring, "UPDATE")."""
    presence = hwm.build_presence()
    valid = predictors.valid_mask()
    samples = config.canonical_samples(presence, valid)
    assert list(samples.columns) == ["x", "y", "event", "type", "in_model"]
    assert samples["in_model"].all()
    assert int(valid.sum()) == 279
    assert len(samples) == 277
    assert list(samples.index) == list(range(len(samples)))


def test_canonical_samples_default_args_match_explicit_args():
    """Zero-arg convenience path (lazily imports hwm/predictors internally)
    must produce the identical frame as passing them explicitly -- both
    ultimately call the same deterministic hwm.build_presence()/
    predictors.valid_mask().
    """
    explicit = config.canonical_samples(hwm.build_presence(), predictors.valid_mask())
    default = config.canonical_samples()
    pd.testing.assert_frame_equal(explicit, default)


def test_canonical_samples_rejects_length_mismatch():
    presence = hwm.build_presence()
    with pytest.raises(AssertionError, match="drifted out of sync"):
        config.canonical_samples(presence, np.ones(len(presence) - 1, dtype=bool))


def test_canonical_samples_feeds_existing_split_and_loeo_folds():
    """Integration check: the reconciled canonical_samples() output must
    still work with the split()/loeo_folds() functions tests/test_split.py
    already covers with synthetic data -- same column/index contract
    (0..n-1 RangeIndex, x/y/event/type columns).
    """
    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == len(samples)
    folds = config.loeo_folds(samples)
    covered = sorted(i for idx in folds.values() for i in idx)
    assert covered == list(range(len(samples)))


# --------------------------------------------------------------------------
# config.canonical_samples() 277 pixel-unique dedup
# --------------------------------------------------------------------------

def test_canonical_samples_dedups_279_valid_points_to_277_modeling_pixels():
    """The 279 predictor-NaN-valid presence points (valid_mask(), UNCHANGED
    by this update) occupy only 277 DISTINCT 10 m modeling-grid pixels --
    exactly 2 pixels each hold 2 of the 279 points -- so
    config.canonical_samples() must return 277 rows, one per pixel, not
    279. Pinned as a real, empirically-verified regression guard (not a
    hoped-for target), same philosophy as
    test_valid_presence_actual_count_is_279 above.
    """
    assert int(predictors.valid_mask().sum()) == 279
    samples = config.canonical_samples()
    assert len(samples) == 277


def test_canonical_samples_are_pixel_unique_on_modeling_grid():
    """Every row config.canonical_samples() returns must map to a DISTINCT
    (row, col) pixel of the 10 m modeling grid -- the whole point of
    the dedup. Recomputes row/col independently here (via
    rasterio.transform.rowcol against dem.tif's own transform) rather than
    reaching into config.py's private helpers, so this verifies the
    CONTRACT from the outside rather than re-asserting the implementation's
    own internal bookkeeping.
    """
    import rioxarray
    from rasterio.transform import rowcol

    samples = config.canonical_samples()
    ref = rioxarray.open_rasterio(predictors.dem_path()).squeeze("band", drop=True)
    rows, cols = rowcol(ref.rio.transform(), samples["x"].to_numpy(), samples["y"].to_numpy())
    pixels = list(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist()))
    assert len(pixels) == 277
    assert len(set(pixels)) == 277  # every pixel distinct -- no duplicates survived


def test_canonical_samples_dedup_deterministic_across_calls():
    """Two independent calls (each re-deriving presence/valid_mask/pixel
    row-col from scratch) must produce the IDENTICAL 277-row frame, same
    row order included -- config.py's own docstring attributes this to
    df's row order already being fully deterministic upstream
    (hwm.build_presence()'s config.GLOBAL_SEED-seeded thinning) plus
    drop_duplicates(keep="first") introducing no additional randomness.
    """
    a = config.canonical_samples()
    b = config.canonical_samples()
    pd.testing.assert_frame_equal(a, b)


def test_build_samples_csv_uses_277_pixel_unique_canonical_set(tmp_path):
    """The dedup also rewires pipeline.maxent.build_samples_csv(): its
    default presence source is now config.canonical_samples() (277,
    pixel-unique), not hwm.build_presence() (286, undeduped) -- so MaxEnt's
    own historical same-cell collapsing becomes a no-op instead of the
    mechanism that determines n. Local import (pipeline.maxent) -- this
    file's own scope is predictors/config, not maxent; importing it only
    inside this one integration test keeps that boundary explicit.
    """
    from pipeline import maxent

    out_path = maxent.build_samples_csv(out_path=tmp_path / "samples.csv")
    written = pd.read_csv(out_path)
    assert len(written) == 277
    assert list(written.columns) == ["species", "X", "Y"]

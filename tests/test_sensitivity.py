"""Tests for pipeline.sensitivity -- the matched-baseline
sensitivity arms + the honest ascertainment-bias bound (see
pipeline/sensitivity.py's own module docstring for the full rationale).

Two tiers, mirroring test_robustness_folds.py's/test_model_families.py's
own established scoping precedent:

1. Pure-logic / synthetic-data tests for ``_points_fingerprint``,
   ``_make_eval_id``, ``_top_n_events``, ``_filter_points_by_mask`` -- fast,
   no MaxEnt subprocess, no real predictor rasters.
2. The real end-to-end run: ``sensitivity.run()`` -- 6 arms (2 of
   which fit two models each) plus a reused baseline, ~6 fresh MaxEnt fits
   total (EVAL_RASTER's own single train-only fit took ~6s cold; every
   fit here is the same size or smaller). Run ONCE per test session via a
   module-scoped fixture (cache-aware regardless -- ``_fit_train_only``'s
   own train_idx.csv provenance check skips any arm whose output already
   matches on a re-entrant call), then asserted against from multiple
   test functions. These PIN the ACTUAL observed AUCs -- never a value
   assumed in advance (see pipeline/sensitivity.py's own docstring for
   the ascertainment-bias narrative).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import sensitivity

# ============================================================================
# Tier 1 -- pure logic / synthetic data, no MaxEnt subprocess.
# ============================================================================

def test_points_fingerprint_is_order_independent():
    xs1, ys1 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    xs2, ys2 = [3.0, 1.0, 2.0], [6.0, 4.0, 5.0]   # same points, shuffled
    assert sensitivity._points_fingerprint(xs1, ys1) == sensitivity._points_fingerprint(xs2, ys2)


def test_points_fingerprint_differs_for_different_points():
    fp1 = sensitivity._points_fingerprint([1.0, 2.0], [1.0, 2.0])
    fp2 = sensitivity._points_fingerprint([1.0, 9.0], [1.0, 2.0])
    assert fp1 != fp2


def test_make_eval_id_matches_iff_both_test_and_background_match():
    test_pts = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    bg = pd.DataFrame({"x": [10.0, 20.0], "y": [10.0, 20.0]})
    bg_other = pd.DataFrame({"x": [99.0, 20.0], "y": [99.0, 20.0]})

    id1 = sensitivity._make_eval_id("standard83", test_pts, "random10k", bg)
    id2 = sensitivity._make_eval_id("standard83", test_pts, "random10k", bg)
    id3 = sensitivity._make_eval_id("standard83", test_pts, "random10k", bg_other)

    assert id1 == id2, "identical (test, background) pairs must produce the identical eval_id"
    assert id1 != id3, "a different background must change the eval_id"


def test_top_n_events_computed_fresh_never_hardcoded_list():
    samples = pd.DataFrame({"event": (["A"] * 10 + ["B"] * 8 + ["C"] * 5 + ["D"] * 2 + ["E"] * 1)})
    labels, counts = sensitivity._top_n_events(samples, n=3)
    assert labels == ["A", "B", "C"]
    assert counts == {"A": 10, "B": 8, "C": 5, "D": 2, "E": 1}


def test_top_n_events_breaks_ties_by_event_label():
    samples = pd.DataFrame({"event": (["Z"] * 5 + ["A"] * 5 + ["M"] * 1)})
    labels, _ = sensitivity._top_n_events(samples, n=2)
    assert labels == ["A", "Z"], "a count tie must break deterministically by event label, not insertion order"


def test_filter_points_by_mask_keeps_only_true_pixels():
    import rasterio
    transform = rasterio.transform.from_origin(0, 30, 10, 10)   # 10 m cells, origin (0, 30)
    mask = np.array([[True, False, True],
                      [False, False, False],
                      [True, True, False]])
    # Row 0: x in [0,10)->col0(True), [10,20)->col1(False), [20,30)->col2(True)
    points = pd.DataFrame({"x": [5.0, 15.0, 25.0, -100.0], "y": [25.0, 25.0, 25.0, 25.0]})
    kept = sensitivity._filter_points_by_mask(points, mask, transform)
    assert len(kept) == 2
    assert sorted(kept["x"].tolist()) == [5.0, 25.0]


# ============================================================================
# Tier 2 -- the real run (one sensitivity.run() call, module-scoped).
# ============================================================================

@pytest.fixture(scope="module")
def result():
    return sensitivity.run()


def test_baseline_shares_eval_set_with_arms(result):
    """The key test: baseline and its first (same-
    protocol) arm share the identical (test set, background) -- resolution_
    30m changes only the model's spatial resolution, never the standard 83
    test points or the random 10k background. output_format_logistic is the
    OTHER random-background arm sharing this exact protocol (outputformat
    only, no change to the evaluation universe) -- guarded here directly via
    its own baseline_eval_id field, the same way resolution_30m is."""
    b, arms = result
    assert b["eval_id"] == arms[0]["eval_id"]
    assert arms[0]["arm"] == "resolution_30m"
    assert arms[0]["baseline_eval_id"] == b["eval_id"], (
        "resolution_30m must be differenced against the OVERALL baseline's exact eval set"
    )

    ofl = next(a for a in arms if a["arm"] == "output_format_logistic")
    assert ofl["baseline_eval_id"] == b["eval_id"], (
        "output_format_logistic must be differenced against the OVERALL baseline's exact eval "
        "set (same standard 83 test points + same random 10k background; only outputformat "
        "differs) -- this is the required cross-check for the SECOND random-background arm, "
        "guarding the same invariant the resolution_30m check above guards for the first."
    )


def test_no_arm_uses_the_stale_notebook_baseline(result):
    """The critical fix: no arm's matched baseline is the old notebook's
    hardcoded, resubstitution-flavored 0.9842 -- every baseline_auc here is
    RECOMPUTED under that arm's own protocol."""
    b, arms = result
    assert b["auc"] != pytest.approx(0.9842, abs=1e-6)
    for a in arms:
        assert a["baseline_auc"] != pytest.approx(0.9842, abs=1e-6), (
            f"arm {a['arm']!r} must not be differenced against the stale 0.9842"
        )


def test_baseline_reproduces_pinned_oos_auc(result):
    """This module's baseline reuses evaluate.EVAL_RASTER + the standard
    config.split() test points + robustness._background_reference_raster's
    own background recipe -- the SAME model/protocol evaluate and robustness already
    pinned at 0.982577108433735 (tests/test_evaluate_oos.py,
    tests/test_robustness_folds.py). Reproducing it bit-for-bit here is the
    strongest possible confirmation this module is wired to the real
    canonical pipeline, not a parallel re-derivation."""
    b, _ = result
    assert b["auc"] == pytest.approx(0.982577108433735, abs=1e-6)


def test_stormwater_buffer_arm_is_labeled_a_lower_bound(result):
    _, arms = result
    sw = next(a for a in arms if a["arm"] == "bg_target_group_stormwater_buffer")
    assert sw["lower_bound"] is True
    assert sw["bound"] == "lower_bound"
    assert "lower bound" in sw["caveat"].lower()
    assert "outfall" in sw["caveat"].lower() and "line" in sw["caveat"].lower()


def test_outfall_distance_matched_arm_exists_and_reports_auc(result):
    _, arms = result
    om = next(a for a in arms if a["arm"] == "bg_outfall_distance_matched")
    assert isinstance(om["auc"], float) and 0.0 <= om["auc"] <= 1.0
    assert om["bound"] == "proper_upper_bound_on_ascertainment_bias"
    assert "measures" in om and len(om["measures"]) > 0
    assert om["n_presence"] > 0 and om["n_background"] > 0


def test_every_arm_has_a_matched_baseline_and_delta(result):
    b, arms = result
    assert set(arms[i]["arm"] for i in range(len(arms))) == {
        "resolution_30m", "bg_target_group_developed", "bg_target_group_stormwater_buffer",
        "bg_outfall_distance_matched", "output_format_logistic", "event_pooling_top4",
    }
    for a in arms:
        assert a["delta_auc"] == pytest.approx(a["auc"] - a["baseline_auc"], abs=1e-9)
        assert isinstance(a["eval_id"], str) and len(a["eval_id"]) > 0


def test_developed_and_stormwater_arms_share_eval_id_with_their_own_matched_baseline(result):
    """Internal consistency, DIRECTLY verified (not inferred from presence
    counts): the like-vs-like design means each arm's own matched baseline
    was scored on the SAME (filtered test presence, target background) the
    arm itself was -- pipeline.sensitivity._target_group_background_arm
    passes the identical filtered_test_pts/target_bg objects to BOTH _score
    calls (arm and baseline alike), and the arm dict now exposes exactly
    which eval set produced baseline_auc via its own baseline_eval_id field
    (see pipeline/sensitivity.py's module docstring). So for developed and
    stormwater-buffer, arm["eval_id"] == arm["baseline_eval_id"] must hold
    EXACTLY -- this is the assertion this test's name promises and the
    original body never made.

    bg_outfall_distance_matched is included but is
    deliberately NOT like-vs-like: it retrains nothing and compares the
    SAME model under two DIFFERENT backgrounds (random vs. distance-
    matched) -- that swap is this arm's entire scientific point. Its
    baseline_auc is the OVERALL baseline's number (scored on the standard
    random background), so its baseline_eval_id equals the OVERALL
    baseline's eval_id, NOT this arm's own (distance-matched-background)
    eval_id -- confirmed against the real cached run, where the two
    differ only in the background half of the id (bg=random10k vs.
    bg=outfall_distance_matched). Asserting arm["eval_id"] ==
    arm["baseline_eval_id"] here would be false and is deliberately NOT
    what is checked; instead this test verifies the invariant that IS true
    for this arm, the same one resolution_30m/output_format_logistic rely
    on (baseline_eval_id == the overall baseline's eval_id), plus the
    inequality that makes this arm's sensitivity comparison meaningful in
    the first place."""
    b, arms = result

    for name in ("bg_target_group_developed", "bg_target_group_stormwater_buffer"):
        arm = next(a for a in arms if a["arm"] == name)
        assert arm["eval_id"] == arm["baseline_eval_id"], (
            f"{name}: arm and its matched baseline must share the identical eval_id -- "
            "like-vs-like means the SAME filtered test presence + target background scored "
            "both the arm's own model and its matched baseline"
        )
        # Still-valid secondary checks from the original test.
        assert arm["n_presence_in_zone"] <= arm["n_presence_standard_total"]
        assert arm["n_presence_in_zone"] > 0, f"{name}: no test presence points fell inside the target zone"

    om = next(a for a in arms if a["arm"] == "bg_outfall_distance_matched")
    assert om["baseline_eval_id"] == b["eval_id"], (
        "bg_outfall_distance_matched's baseline_auc is the OVERALL baseline's number verbatim "
        "(no retraining) -- its baseline_eval_id must equal the overall baseline's own eval_id"
    )
    assert om["eval_id"] != om["baseline_eval_id"], (
        "bg_outfall_distance_matched deliberately swaps ONLY the background (random -> distance-"
        "matched) under the SAME model -- its own eval_id must differ from baseline_eval_id, or "
        "this arm would not be testing the background-sensitivity question it exists to answer"
    )


# ============================================================================
# Pinned real observed numbers (the ascertainment-bias narrative these
# numbers support is in pipeline/sensitivity.py's docstring).
# ============================================================================

def test_pinned_arm_aucs(result):
    _, arms = result
    by_name = {a["arm"]: a for a in arms}

    assert by_name["resolution_30m"]["auc"] == pytest.approx(PINNED["resolution_30m"], abs=5e-4)
    assert by_name["output_format_logistic"]["auc"] == pytest.approx(
        PINNED["output_format_logistic"], abs=5e-4)
    assert by_name["event_pooling_top4"]["auc"] == pytest.approx(PINNED["event_pooling_top4"], abs=5e-4)
    assert by_name["bg_target_group_developed"]["auc"] == pytest.approx(
        PINNED["bg_target_group_developed"], abs=5e-4)
    assert by_name["bg_target_group_stormwater_buffer"]["auc"] == pytest.approx(
        PINNED["bg_target_group_stormwater_buffer"], abs=5e-4)
    assert by_name["bg_outfall_distance_matched"]["auc"] == pytest.approx(
        PINNED["bg_outfall_distance_matched"], abs=5e-4)


# Pinned to the ACTUAL real run (2026-07-24, cold, 75.8s total
# wall time for all 6 arms + baseline reuse). Never forced toward an
# assumed value.
PINNED = {
    "resolution_30m": 0.983777516079355,
    "output_format_logistic": 0.9825771084337349,
    "event_pooling_top4": 0.9822136363636363,
    "bg_target_group_developed": 0.9515254545532044,
    "bg_target_group_stormwater_buffer": 0.9398912132516063,
    "bg_outfall_distance_matched": 0.9293995654750148,
}

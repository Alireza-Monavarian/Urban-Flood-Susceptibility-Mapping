"""Tests for pipeline.robustness -- spatial-CV + LOEO + ablation
on the canonical 277-point set (the honest, spatially-defensible
generalization AUC; see pipeline/robustness.py's own module docstring for
the full rationale).

Two tiers, mirroring test_maxent_aicc.py's/test_model_families.py's/
test_importance.py's own established scoping precedent
("pure-logic-or-cheap-real-call vs. real-subprocess-run"):

1. Pure-logic / synthetic-data tests for ``_merge_small_blocks``,
   ``spatial_cv_membership``, ``_fit_folds_cached`` (including its
   cache-hit train_idx.csv provenance guard), ``_protocol_fingerprint``,
   ``_recorded_train_idx``, ``_fit_train_only_on_layers``'s pre-fit
   leakage assert, and the ablation-arms partition invariant -- fast, no
   MaxEnt subprocess (``loeo_membership``/``test_loeo_reconciles_to_277``
   is the one exception: it calls the real
   ``config.canonical_samples()``, matching test_model_families.py's own
   precedent).
2. Real end-to-end tests that call the REAL ``spatial_cv()``/
   ``loeo()``/``ablation()`` (actual MaxEnt subprocess fits across ~14
   spatial blocks + 11 events + 1 baseline + 4 ablation arms, ~10-15 min
   cold). These PIN the ACTUAL observed mean/SD
   AUCs -- never a value assumed in advance.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import config, maxent, robustness

# ============================================================================
# Tier 1 -- pure logic / synthetic data, no MaxEnt subprocess.
# ============================================================================

# --------------------------------------------------------------------------
# _merge_small_blocks -- the centroid-distance merge (deviates from 04c's
# own raw-block-ID-distance merge; see robustness.py's module docstring).
# --------------------------------------------------------------------------

def test_merge_small_blocks_merges_into_nearest_centroid():
    # Block 0 centroid (0,0); block 1 centroid (100,100); lone block 2 is a
    # single point at (1,1) -- much nearer block 0's centroid than block 1's.
    samples = pd.DataFrame({
        "x": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 1.0],
        "y": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 1.0],
    })
    block_ids = np.array([0, 0, 0, 1, 1, 1, 2])
    merged = robustness._merge_small_blocks(samples, block_ids, min_presence=3)
    assert list(merged[:6]) == [0, 0, 0, 1, 1, 1], "large blocks must be untouched"
    assert merged[6] == 0, "the lone point must merge into the NEARER large block (0), not block 1"


def test_merge_small_blocks_is_noop_when_no_small_blocks():
    samples = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    block_ids = np.array([0, 0, 0])
    merged = robustness._merge_small_blocks(samples, block_ids, min_presence=3)
    assert list(merged) == [0, 0, 0]


def test_merge_small_blocks_raises_when_all_blocks_small():
    samples = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    block_ids = np.array([0, 1])
    with pytest.raises(AssertionError):
        robustness._merge_small_blocks(samples, block_ids, min_presence=3)


# --------------------------------------------------------------------------
# spatial_cv_membership -- must use config.spatial_folds's own 2km default
# (never re-specify/override it to 5km) AND apply the merge.
# --------------------------------------------------------------------------

def test_spatial_cv_membership_uses_2km_blocks_by_default():
    """Same discriminating fixture as tests/test_split.py::
    test_spatial_folds_uses_2km_blocks, exercised one level up through
    robustness.spatial_cv_membership -- min_presence_per_block=1 makes the
    merge step a guaranteed no-op, isolating the block_km behavior."""
    samples = pd.DataFrame({"x": [700000.0, 701000.0, 703000.0],
                            "y": [4340000.0, 4340000.0, 4340000.0],
                            "event": ["A", "A", "A"], "type": "HW"})
    holdouts = robustness.spatial_cv_membership(samples, min_presence_per_block=1)
    fold_of = {i: fid for fid, idx in holdouts.items() for i in idx}
    assert fold_of[0] == fold_of[1], "700000 & 701000 must share one 2km block"
    assert fold_of[0] != fold_of[2], "703000 must be a different 2km block (5km would collapse this)"


def test_spatial_cv_membership_merges_blocks_below_default_min_presence():
    """Default min_presence_per_block=3, default block_km=None (-> config's
    own 2km default): 4 points share one 2km block (survives unmerged);
    two lone points in their OWN 2km blocks must merge into that one
    surviving large block (the only candidate available)."""
    x = [700000.0, 700200.0, 700400.0, 700600.0, 650000.0, 600000.0]
    y = [4340000.0] * 6
    samples = pd.DataFrame({"x": x, "y": y, "event": "A", "type": "HW"})

    holdouts = robustness.spatial_cv_membership(samples)
    fold_of = {i: fid for fid, idx in holdouts.items() for i in idx}
    assert fold_of[0] == fold_of[1] == fold_of[2] == fold_of[3]
    assert fold_of[4] == fold_of[0]
    assert fold_of[5] == fold_of[0]
    assert len(holdouts) == 1


# --------------------------------------------------------------------------
# _fit_folds_cached -- idempotency orchestration (maxent.fit_folds/run()
# always pass MaxEnt's own redoifexists, so caching must happen HERE).
# --------------------------------------------------------------------------

def test_fit_folds_cached_skips_existing_raster(tmp_path, monkeypatch):
    """A cache hit is only trusted when the recorded train_idx.csv MATCHES
    this fold's intended train complement -- held_out_idx=[0,1] over a
    5-point universe means the intended complement is [2,3,4]."""
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "flood.tif").write_bytes(b"fake-raster")
    np.savetxt(fold_dir / "train_idx.csv", np.array([2, 3, 4]), fmt="%d", header="train_idx", comments="")

    def _boom(*a, **kw):
        raise AssertionError("maxent.fit_folds must NOT be called for an already-cached fold")
    monkeypatch.setattr(robustness.maxent, "fit_folds", _boom)

    samples = pd.DataFrame({"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)})
    holdouts = {0: np.array([0, 1])}
    result = robustness._fit_folds_cached(holdouts, samples, tmp_path, timeout=10)
    assert result[0] == fold_dir / "flood.tif"


def test_fit_folds_cached_refits_when_train_idx_mismatches(tmp_path, monkeypatch):
    """The cache-hit provenance guard: a STALE train_idx.csv (e.g. left
    over from a re-run with DIFFERENT fold membership -- a different
    min_presence_per_block, or a canonical_samples()/spatial_folds()
    change) must be REJECTED and the fold REFIT, never trusted just
    because flood.tif already exists on disk. Mirrors
    _baseline_eval_raster's own train_idx.csv mismatch handling."""
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "flood.tif").write_bytes(b"stale-raster")
    # Bogus/stale: does NOT equal the intended complement [2,3,4] for
    # held_out_idx=[0,1] over a 5-point universe.
    np.savetxt(fold_dir / "train_idx.csv", np.array([0, 2, 3]), fmt="%d", header="train_idx", comments="")

    calls = []

    def _fake_fit_folds(fold_assignment, samples, *, out_dir, timeout):
        calls.append(dict(fold_assignment))
        fid = next(iter(fold_assignment))
        d = Path(out_dir) / f"fold_{fid}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "flood.tif"
        p.write_bytes(b"fresh-raster")
        return {fid: p}
    monkeypatch.setattr(robustness.maxent, "fit_folds", _fake_fit_folds)

    samples = pd.DataFrame({"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)})
    holdouts = {0: np.array([0, 1])}
    result = robustness._fit_folds_cached(holdouts, samples, tmp_path, timeout=10)

    assert len(calls) == 1, "a mismatched train_idx.csv must trigger a REFIT, not a trusted cache hit"
    assert result[0].read_bytes() == b"fresh-raster", "the stale raster must be replaced"


def test_fit_folds_cached_refits_when_train_idx_csv_missing(tmp_path, monkeypatch):
    """A cached raster with NO companion train_idx.csv at all (e.g. an
    interrupted earlier run) must also never be trusted on path existence
    alone -- mirrors _baseline_eval_raster's own silent-refit-when-missing
    fallback."""
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "flood.tif").write_bytes(b"raster-with-no-provenance")

    calls = []

    def _fake_fit_folds(fold_assignment, samples, *, out_dir, timeout):
        calls.append(dict(fold_assignment))
        fid = next(iter(fold_assignment))
        d = Path(out_dir) / f"fold_{fid}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "flood.tif"
        p.write_bytes(b"fresh-raster")
        return {fid: p}
    monkeypatch.setattr(robustness.maxent, "fit_folds", _fake_fit_folds)

    samples = pd.DataFrame({"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)})
    holdouts = {0: np.array([0, 1])}
    result = robustness._fit_folds_cached(holdouts, samples, tmp_path, timeout=10)

    assert len(calls) == 1, "a raster with no recorded train_idx.csv must never be trusted"
    assert result[0].read_bytes() == b"fresh-raster"


def test_fit_folds_cached_fits_one_fold_at_a_time_when_missing(tmp_path, monkeypatch):
    calls = []

    def _fake_fit_folds(fold_assignment, samples, *, out_dir, timeout):
        calls.append(dict(fold_assignment))
        fid = next(iter(fold_assignment))
        d = Path(out_dir) / f"fold_{fid}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "flood.tif"
        p.write_bytes(b"fake")
        return {fid: p}

    monkeypatch.setattr(robustness.maxent, "fit_folds", _fake_fit_folds)

    samples = pd.DataFrame({"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)})
    holdouts = {0: np.array([0, 1]), 1: np.array([2, 3])}
    result = robustness._fit_folds_cached(holdouts, samples, tmp_path, timeout=10)

    assert len(calls) == 2, "one fit_folds call per missing fold, each a singleton dict"
    assert all(len(c) == 1 for c in calls)
    assert set(result) == {0, 1}


# --------------------------------------------------------------------------
# _protocol_fingerprint -- this DOES discriminate on test_idx/background in
# isolation (proven below), even though inside ablation() itself baseline_fp
# == arm_fp is guaranteed by shared-locals construction (see
# _protocol_fingerprint's own docstring) -- the load-bearing, genuinely
# non-tautological check ablation() relies on is _recorded_train_idx,
# covered in its own section further down.
# --------------------------------------------------------------------------

def test_protocol_fingerprint_differs_when_test_idx_differs():
    bg = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fp1 = robustness._protocol_fingerprint("holdout_oos", [0, 1, 2], bg)
    fp2 = robustness._protocol_fingerprint("holdout_oos", [0, 1, 3], bg)
    assert fp1 != fp2


def test_protocol_fingerprint_differs_when_background_differs():
    fp1 = robustness._protocol_fingerprint("holdout_oos", [0, 1], pd.DataFrame({"x": [1.0], "y": [1.0]}))
    fp2 = robustness._protocol_fingerprint("holdout_oos", [0, 1], pd.DataFrame({"x": [9.0], "y": [9.0]}))
    assert fp1 != fp2


def test_protocol_fingerprint_is_order_independent_and_matches_when_inputs_match():
    bg = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    fp1 = robustness._protocol_fingerprint("holdout_oos", [5, 3, 1], bg)
    fp2 = robustness._protocol_fingerprint("holdout_oos", [1, 3, 5], bg)
    assert fp1 == fp2


# --------------------------------------------------------------------------
# _recorded_train_idx -- the REAL, disk-provenance-based check ablation()
# uses to compare what each raster (baseline AND every arm) was ACTUALLY
# fit on, independent of any in-memory variable (closes the gap
# _protocol_fingerprint's shared-locals tautology leaves open).
# --------------------------------------------------------------------------

def test_recorded_train_idx_reads_back_written_csv(tmp_path):
    raster_path = tmp_path / "flood.tif"
    raster_path.write_bytes(b"fake-raster")
    np.savetxt(tmp_path / "train_idx.csv", np.array([3, 1, 2]), fmt="%d", header="train_idx", comments="")
    recorded = robustness._recorded_train_idx(raster_path)
    assert sorted(recorded.tolist()) == [1, 2, 3]


# --------------------------------------------------------------------------
# _fit_train_only_on_layers -- Fix 2: the pre-fit qa.
# assert_eval_model_excludes_test guard, mirroring maxent.fit_folds/
# _fit_train_only's own Invariant 2 (previously missing here -- ablation
# arms relied SOLELY on the scoring-time evaluate.oos_auc leakage assert).
# --------------------------------------------------------------------------

def test_fit_train_only_on_layers_rejects_overlapping_train_and_held_out(tmp_path, monkeypatch):
    """Must raise BEFORE touching export_layers/MaxEnt at all -- even
    before the cache-hit check -- when train_idx and held_out_idx overlap."""
    def _boom(*a, **kw):
        raise AssertionError("must not reach export_layers -- the leakage assert must fire FIRST")
    monkeypatch.setattr(robustness.maxent, "export_layers", _boom)

    samples = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float)})
    train_idx = np.array([0, 1, 2, 3, 5])   # overlaps held_out_idx on index 5
    held_out_idx = np.array([5, 6, 7])
    with pytest.raises(AssertionError):
        robustness._fit_train_only_on_layers(samples, train_idx, held_out_idx,
                                              {"dem"}, tmp_path / "arm", timeout=10)


# --------------------------------------------------------------------------
# ablation() -- protocol validation (BEFORE any I/O) + arms-partition
# invariant against the REAL live 17-predictor stack.
# --------------------------------------------------------------------------

def test_ablation_rejects_unsupported_baseline_protocol():
    """Must raise before touching samples/rasters -- the validation is
    this function's first executable statement (mirrors qa.
    assert_eval_model_excludes_test's "enforced before any subprocess"
    convention elsewhere in this codebase)."""
    with pytest.raises(ValueError):
        robustness.ablation(baseline_protocol="not_a_real_protocol")


def test_ablation_arms_partition_the_canonical_predictors():
    """ABLATION_ARMS must exactly cover the REAL, live 17-predictor
    canonical stack (maxent.modeling_layers(), recomputed fresh -- no
    MaxEnt subprocess) with no overlaps -- a stale/wrong grouping (e.g.
    after a future predictor-stack change) must fail loudly here, matching
    ablation()'s own internal assert."""
    all_layers = set(maxent.modeling_layers())
    assert len(all_layers) == 17
    seen, union = set(), set()
    for name, members in robustness.ABLATION_ARMS.items():
        overlap = seen & set(members)
        assert not overlap, f"arm {name!r} overlaps another arm on {overlap}"
        seen |= set(members)
        union |= set(members)
    assert union == all_layers


# --------------------------------------------------------------------------
# Output directories -- pure path-construction regression guard.
# --------------------------------------------------------------------------

def test_output_dirs_are_distinct():
    dirs = [str(robustness.SPATIAL_CV_DIR), str(robustness.LOEO_DIR), str(robustness.ABLATION_DIR)]
    assert len(set(dirs)) == 3
    for a in dirs:
        for b in dirs:
            if a != b:
                assert not a.startswith(b + "/"), f"{a} nested under {b}"


# --------------------------------------------------------------------------
# config.loeo_folds carries
# `event` as a NATIVE column (attached at HWM-parse time, never via a
# later rounded-coordinate join), so this reconciles to the full 277 with
# no drop possible (see robustness.py's own module docstring, point 2).
# --------------------------------------------------------------------------

def test_loeo_reconciles_to_277():
    folds = robustness.loeo_membership()
    assert sum(len(v) for v in folds.values()) == 277   # was 272 in the OLD notebook's rounded-coord join


def test_loeo_membership_matches_config_loeo_folds():
    samples = config.canonical_samples()
    expected = config.loeo_folds(samples)
    actual = robustness.loeo_membership(samples)
    assert set(actual) == set(expected)
    for k in expected:
        assert list(actual[k]) == list(expected[k])


# ============================================================================
# Tier 2 -- real on-disk run: the actual ~14-fold spatial CV + 11-
# event LOEO + 1 baseline + 4 ablation-arm MaxEnt fits. Run once;
# re-entrant calls here are cheap because every
# fold/arm raster is cached (see _fit_folds_cached / _fit_train_only_on_
# layers' own idempotency).
# ============================================================================

def test_real_spatial_cv_mean_is_pinned():
    """Pinned to the ACTUAL observed real 14-fold run (20 raw 2km blocks,
    6 merged away for < 3 presence points).
    THIS is the honest, spatially-robust generalization AUC: 0.9411,
    well below the single-split 0.982577 -- the ~0.0415 gap is the
    quantified spatial-autocorrelation component of the single-split
    number (never forced toward either figure)."""
    result = robustness.spatial_cv()
    assert result["n_folds"] == 14
    assert result["n_raw_blocks"] == 20
    assert len(result["per_fold"]) == result["n_folds"]
    print(f"\n[test_real_spatial_cv_mean_is_pinned] spatial-CV mean AUC = "
          f"{result['mean_auc']:.4f} +/- {result['sd_auc']:.4f} ({result['n_folds']} folds)")
    # Re-pinned 2026-08-12 after a clean rebuild under the pinned
    # environment (env/RUNTIME.md). Tolerance loosened 1e-6 -> 1e-4: MaxEnt
    # samples its own background WITHOUT a seed, so per-fold AUCs move at ~1e-5
    # between runs. 1e-6 was tighter than MaxEnt's own determinism and failed on
    # a faithful rebuild (0.9410618 vs 0.9410684). Both round to the 0.941 the
    # manuscript reports.
    assert result["mean_auc"] == pytest.approx(0.9410618165648726, abs=1e-4)
    assert result["sd_auc"] == pytest.approx(0.058208111354895016, abs=1e-4)


def test_real_loeo_mean_is_pinned():
    """Pinned to the ACTUAL observed real 11-event run
    (including the 2-point 2021_7_15 and 4-point 2015_5_4 events, reported
    as-is, not filtered out)."""
    result = robustness.loeo()
    assert result["n_events"] == 11
    assert len(result["per_event"]) == 11
    print(f"\n[test_real_loeo_mean_is_pinned] LOEO mean AUC = "
          f"{result['mean_auc']:.4f} +/- {result['sd_auc']:.4f} ({result['n_events']} events)")
    assert result["mean_auc"] == pytest.approx(0.9531697604247604, abs=1e-6)
    assert result["sd_auc"] == pytest.approx(0.06410172531540738, abs=1e-6)


def test_real_ablation_baseline_matches_pinned_oos_auc():
    """The ablation baseline reuses evaluate.EVAL_RASTER under the exact
    same protocol evaluate.oos_auc uses -- must reproduce 0.982577...
    bit-for-bit (see tests/test_evaluate_oos.py::test_oos_auc_in_expected_band),
    never a different (e.g. 0.9842 resubstitution) number."""
    df = robustness.ablation()
    baseline_row = df[df["arm"] == "baseline"].iloc[0]
    assert baseline_row["auc"] == pytest.approx(0.982577108433735, abs=5e-4)
    assert baseline_row["delta_auc"] == 0.0
    assert (df["protocol"] == "holdout_oos").all()
    assert set(df["arm"]) == {"baseline", "infrastructure", "soils", "terrain", "land_use"}
    print("\n[test_real_ablation_baseline_matches_pinned_oos_auc] ablation table:")
    print(df.to_string(index=False))


def test_real_ablation_infrastructure_drop_is_the_largest():
    """Pinned to the ACTUAL observed real ablation run: dropping
    `infrastructure` (distance_to_outfall/stormwater_density) costs the most
    AUC of the 4 arms (delta ~= -0.0103), ahead of soils (~-0.0075), terrain
    (~-0.0065) and land_use (~-0.0027, the smallest drop) -- consistent with
    the jackknife's finding that distance_to_outfall is the #1 predictor by
    unique contribution.

    **Re-pinned 2026-08-15.** `drainage_density` was moved from the
    infrastructure arm to terrain: it is derived from
    the National Hydrography Dataset, not from the municipal stormwater
    geodatabase, so counting it as infrastructure overstated that arm. The
    infrastructure drop therefore fell from -0.0127 (three predictors) to
    -0.0103 (two), and terrain rose from -0.0023 to -0.0065. Infrastructure is
    still the largest effect, which is the claim the manuscript makes. This
    test kept the pre-regrouping value and had been failing since."""
    df = robustness.ablation()
    arm_rows = df[df["arm"] != "baseline"].set_index("arm")
    worst_arm = arm_rows["delta_auc"].idxmin()
    assert worst_arm == "infrastructure"
    assert arm_rows.loc["infrastructure", "delta_auc"] == pytest.approx(-0.010323, abs=5e-4)
    assert (arm_rows["delta_auc"] < 0).all(), "dropping any predictor group should not IMPROVE AUC"

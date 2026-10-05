"""Spatial-CV + LOEO on the canonical set: the HONEST,
spatially-defensible generalization AUC (the number for the paper).

Background: the single random 70/30 split gives ``oos_auc =
0.982577`` (``evaluate.EVAL_RASTER`` sampled at the 83 held-out test
points -- genuinely out-of-sample w.r.t. that model's training set, but
STILL a random split. Flood high-water-mark points cluster tightly along
river corridors/street segments, so a random split puts test points a few
meters from spatially-correlated training points -- the model can get much
of its "generalization" credit from spatial autocorrelation rather than
genuine transferability. This module ports the original notebook
``04c_robustness_checks`` (spatial block CV + event-based LOEO + ablation) into the canonical
pipeline, with THREE deliberate fixes vs. that source:

1. **2 km blocks, not 5 km.** 04c's own Section 10 already found its
   original 5 km leave-one-block-out CV produced a degenerate 3-point fold
   (AUC=0.446) because the ~15x15 km AOI is too small for 5 km blocks to
   carve up cleanly -- it added a 2 km retry (Section 10) as the fix. This
   module goes straight to 2 km, which is ALSO ``config.spatial_folds``'s
   own locked, tested default (``tests/test_split.py::
   test_spatial_folds_uses_2km_blocks`` is the regression guard) -- this
   module never re-specifies ``block_km`` itself, it just inherits
   whatever that one canonical default is.

2. **No rounded-coordinate event join.** 04c's LOEO (Section 11a) attached
   event labels to presence points via a fragile string-key join
   (``x_utm.round(1).astype(str) + '_' + y_utm.round(1).astype(str)``)
   against a SEPARATE ``hwm_hw_thinned_30m.csv`` -- lossy in practice
   (dropped points meant a 278-point set reconciled to only 272). This pipeline's ``config.canonical_samples()`` carries an
   ``event`` column NATIVELY (populated at HWM-file-parse time by
   ``pipeline.hwm``, never re-attached via any coordinate join later), so
   ``config.loeo_folds(config.canonical_samples())`` -- what
   ``loeo_membership()``/``loeo()`` below call -- reconciles to the full
   **277** canonical points with NO join step to lose anything at all.

3. **Same-protocol ablation baseline.** 04c differenced every ablation
   ΔAUC against ``BASELINE_AUC = 0.9842`` -- a DIFFERENT model/config's
   resubstitution-flavored number (the very quantity that is unsafe to
   headline; see ``pipeline.evaluate``'s own module docstring).
   ``ablation()`` below instead computes its OWN baseline under the
   IDENTICAL evaluation protocol (canonical 194/83 ``config.split``, the
   SAME fixed background sample, ``evaluate.oos_auc`` scoring) every arm
   also uses -- reusing the existing ``evaluate.EVAL_RASTER``
   when its recorded train split matches (no need to refit the full-17
   model), so the baseline is the real ~0.9826 like-for-like number, never
   0.9842.

**Merge-logic deviation from 04c (spatial CV):** 04c merged small blocks
into their nearest neighbor by raw block-ID numeric distance (its own
``bx*100+by``/``bx*10000+by`` encoding made "nearest ID" a crude proxy for
"nearest in space"). ``config.spatial_folds()``'s block ids are arbitrary
``pandas.Categorical`` codes, not a spatially-ordered encoding, so that
trick does not apply here -- ``_merge_small_blocks`` instead merges by each
small block's own point-centroid EUCLIDEAN distance to each candidate
large block's centroid. Same effect (signal-poor blocks get folded into a
real spatial neighbor, single-hop, never chained through another small
block), correctly adapted to this pipeline's own block-id scheme.

**Scope vs. 04c:** 04c also ran LR/RF/XGBoost through the same spatial-CV/
LOEO harness. Those sklearn comparators belong to
``pipeline.model_compare`` (built against the identical matched test set);
the spatial-CV/LOEO core of this module is MaxEnt-only ("like-for-like
baselines" here means "the right REFERENCE number", not "re-run every model
family"). ``_loeo_xgboost_aucs`` and ``loeo_all_models`` add the comparators'
per-event LOEO AUCs only for the significance tests further down.

**Honest-reporting mandate:** every mean AUC this module returns is
reported AS COMPUTED -- never forced toward any expected band. The
spatial-CV mean and the LOEO mean are the honest, spatially-robust
generalization estimates; the gap between the single-split 0.9826
and the spatial-CV mean is the quantified spatial-autocorrelation
component of the apparent single-split performance -- a key paper result,
reported as-is regardless of its sign or size.

**Idempotency:** ``maxent.fit_folds``/``maxent.run`` always pass MaxEnt's
own ``redoifexists`` flag, so a direct call ALWAYS reruns the subprocess
even if its output already exists -- unlike ``aicc_grid``/``jackknife_gains``,
neither is idempotent on its own. Idempotency for this module's ~30 fits
(14ish spatial blocks + 11 events + 1 baseline + 4 ablation arms) is
orchestrated HERE instead: every fold/arm's raster path is checked for
existence BEFORE calling into ``maxent``, so a re-entrant call after a
partial/interrupted run only fits what's still missing.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import lil_matrix
from scipy.stats import chi2 as _chi2_dist, t as _t_dist
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from pipeline import config, evaluate, maxent, model_compare, qa

# ============================================================================
# Output locations. Raster-bearing fold/arm directories live under the
# maxent/ build tree (maxent.py's own FOLDS_DIR is explicitly earmarked for
# spatial-CV/LOEO significance testing in its docstring;
# ABLATION_DIR mirrors that same maxent/ tree for the reduced-predictor
# arms). Every .tif/.asc/.lambdas file anywhere is already covered by
# .gitignore's blanket rules; the lightweight per-fold/per-event/ablation
# CSV summaries go to a small dedicated data/processed/robustness/
# directory (mirroring 04c's own OUT_DIR naming).
# ============================================================================

SPATIAL_CV_DIR = maxent.FOLDS_DIR / "spatial_cv"
LOEO_DIR = maxent.FOLDS_DIR / "loeo"
ABLATION_DIR = maxent.OUT_DIR / "ablation"
OUT_DIR = config.DATA / "robustness"

# Blocks with fewer than this many PRESENCE points get merged into their
# nearest (by centroid distance) block with >= this many -- matches 04c's
# own MIN_PRES=3 (Section 2/10's rationale: too few points per fold to
# evaluate an AUC meaningfully).
MIN_PRESENCE_PER_BLOCK = 3


def _background_reference_raster():
    """The raster whose valid/NaN footprint defines the AOI for background
    sampling. Prefers ``evaluate.EVAL_RASTER`` (the existing
    train-only model) when present -- it shares dem.tif's own footprint
    EXACTLY (``maxent._asc_to_geotiff``'s ``.where(~ref.isnull())``
    masking, ``ref=dem.tif``, applied to every raster this pipeline
    produces), so background sampled from it reproduces the IDENTICAL draw
    the pinned out-of-sample AUC 0.982577108433735 used
    (``tests/test_evaluate_oos.py``'s ``real_eval_fixture``). Falls back to
    the predictor stack's own dem.tif reference
    (``maxent._reference_raster()``) if ``EVAL_RASTER`` doesn't exist yet
    in this environment -- same footprint either way (every fold/arm
    raster this module fits is ALSO masked against the same dem.tif via
    ``maxent._asc_to_geotiff``), so which one is used never changes the
    background draw.
    """
    if Path(evaluate.EVAL_RASTER).exists():
        return evaluate.EVAL_RASTER
    return maxent._reference_raster()


# ============================================================================
# Spatial CV -- leave-one-2km-block-out.
# ============================================================================

def _merge_small_blocks(samples, block_ids, min_presence):
    """Merge every block with < `min_presence` points into its nearest (by
    2-D point-centroid Euclidean distance) block among the pool that
    ALREADY had >= `min_presence` points before any merging began --
    single-hop: a small block never merges into another (formerly) small
    block, even one that ends up spatially closer post-merge, matching
    04c's own non-chained merge structure (``large_blocks`` computed once,
    used as a fixed target pool throughout).

    Deviates from 04c's own merge mechanics: 04c merged by raw block-ID
    numeric distance (meaningful only because ITS OWN block ids were a
    ``bx*100+by`` spatially-ordered encoding); ``config.spatial_folds()``'s
    codes are arbitrary ``pandas.Categorical`` labels, so this computes
    real centroid distance instead -- see this module's own docstring.

    Raises ``AssertionError`` if EVERY block is small (nothing to merge
    into) -- surfaced loudly rather than silently returning a meaningless
    single-fold assignment.

    Returns a NEW ndarray, same length/order as `block_ids`.
    """
    block_ids = np.asarray(block_ids)
    x = samples["x"].to_numpy()
    y = samples["y"].to_numpy()

    counts = pd.Series(block_ids).value_counts()
    small = counts[counts < min_presence].index.tolist()
    large = counts[counts >= min_presence].index.tolist()
    if not small:
        return block_ids.copy()
    assert large, (
        f"[robustness] _merge_small_blocks: every block has < {min_presence} "
        f"presence points -- nothing to merge into (n_blocks={len(counts)})"
    )

    centroids = {}
    for b in counts.index:
        m = block_ids == b
        centroids[b] = (float(x[m].mean()), float(y[m].mean()))

    remap = {b: b for b in large}
    for sb in small:
        cx, cy = centroids[sb]
        nearest = min(large, key=lambda lb: (centroids[lb][0] - cx) ** 2 + (centroids[lb][1] - cy) ** 2)
        remap[sb] = nearest

    return np.array([remap[b] for b in block_ids])


def spatial_cv_membership(samples=None, *, min_presence_per_block=MIN_PRESENCE_PER_BLOCK, block_km=None):
    """Fast (no MaxEnt) leave-one-block-out fold PLAN: ``config.
    spatial_folds()``'s own 2 km default (never overridden here unless
    `block_km` is given explicitly) + the small-block merge
    (``_merge_small_blocks``). Returns ``{fold_id: held_out_idx ndarray}``
    -- the same shape ``config.loeo_folds()`` returns, so both feed
    ``_fit_folds_cached``/scoring uniformly.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    raw_blocks = (config.spatial_folds(samples) if block_km is None
                  else config.spatial_folds(samples, block_km=block_km))
    merged = _merge_small_blocks(samples, raw_blocks, min_presence_per_block)
    return {int(fid): np.where(merged == fid)[0] for fid in sorted(set(merged.tolist()))}


def _fit_folds_cached(holdouts, samples, out_dir, timeout):
    """``{fold_id: held_out_idx}`` -> ``{fold_id: raster_path}``, fitting
    each fold's train-only model via ``maxent.fit_folds`` ONE FOLD AT A
    TIME (a singleton dict per call) so an already-built fold raster
    (``<out_dir>/fold_<id>/flood.tif``) can be reused across re-entrant
    calls without a redundant MaxEnt subprocess -- see this module's own
    docstring, "Idempotency". Calling ``fit_folds`` one fold at a time is
    semantically identical to calling it with every fold at once (each
    fold's train set is always "every OTHER point", never conditioned on
    which other folds happen to also be in the dict passed that call).

    **Cache-hit provenance check** (mirrors ``_baseline_eval_raster``'s own
    ``train_idx.csv`` check, below): a fold raster's existence on disk is
    NEVER trusted on its own -- ``<out_dir>/fold_<id>/train_idx.csv``
    (written by ``maxent._fit_train_only`` immediately after every fit) is
    read back and compared against THIS call's own freshly-recomputed
    intended complement (``np.setdiff1d(universe, held_out_idx)``). A
    re-run with DIFFERENT fold membership than whatever produced the
    cached raster (e.g. a different ``min_presence_per_block``, or any
    future change to ``canonical_samples()``/``spatial_folds()``/
    ``loeo_folds()``) would otherwise silently reuse a raster trained on a
    DIFFERENT complement -- possibly one that INCLUDES the current call's
    held-out points, a silent train/test leak the scoring-time
    ``evaluate.oos_auc`` assert cannot catch (it only checks the INTENDED
    split it's handed, never the raster's actual training set). On a
    mismatch this REFITS the fold from scratch rather than trusting the
    stale raster.
    """
    out_dir = Path(out_dir)
    universe = np.arange(len(samples))
    results = {}
    for fold_id in sorted(holdouts, key=str):
        held_out_idx = np.asarray(holdouts[fold_id])
        expected_train_idx = np.setdiff1d(universe, held_out_idx)
        fold_dir = out_dir / f"fold_{fold_id}"
        raster_path = fold_dir / "flood.tif"
        recorded_idx_path = fold_dir / "train_idx.csv"
        if raster_path.exists() and recorded_idx_path.exists():
            recorded = np.loadtxt(recorded_idx_path, skiprows=1, dtype=int).reshape(-1)
            if sorted(recorded.tolist()) == sorted(int(i) for i in expected_train_idx):
                print(f"[robustness] fold {fold_id}: cached ({raster_path})")
                results[fold_id] = raster_path
                continue
            print(f"[robustness] WARNING: {recorded_idx_path} does not match fold {fold_id}'s "
                  f"intended train complement -- refitting from scratch (stale cache, likely "
                  f"from a different fold membership)")
        one = maxent.fit_folds({fold_id: held_out_idx}, samples,
                                out_dir=out_dir, timeout=timeout)
        results[fold_id] = one[fold_id]
    return results


def spatial_cv(samples=None, *, min_presence_per_block=MIN_PRESENCE_PER_BLOCK, block_km=None,
               background_n=evaluate.DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED,
               out_dir=None, timeout=maxent.DEFAULT_TIMEOUT_S):
    """Leave-one-2km-block-out spatial CV on the canonical 277-point set:
    for each block (after merging any with < `min_presence_per_block`
    presence points into its nearest centroid-neighbor block), fits MaxEnt
    (``maxent.fit_folds``, ``FINAL_CONFIG``) on every OTHER block's
    presence, then scores that fold's own held-out presence points against
    a SHARED, seeded 10,000-point background sample
    (``evaluate.oos_auc``-style: presence-vs-background ``roc_auc_score``).

    **This mean is THE honest, spatially-robust generalization AUC** --
    random 70/30 splits (0.982577) let a model get credit for
    spatial autocorrelation between train/test points; leave-one-block-out
    removes that by construction (a held-out block's nearest surviving
    training point is always outside that same 2 km cell). Reported
    exactly as observed, never forced toward the single-split number or any
    assumed band.

    ``out_dir`` routes BOTH the per-fold raster tree AND this run's per-fold
    summary CSV. Default (``out_dir=None``): rasters under ``SPATIAL_CV_DIR``,
    summary CSV under the module ``OUT_DIR`` (``data/processed/robustness/``) --
    the canonical 2 km locations, unchanged (so the pinned 2 km tests and the
    2 km raster cache are untouched). A NON-default ``out_dir`` (e.g. the
    driver notebook's 5 km diagnostic) routes the summary CSV under ``out_dir``
    TOO, so a second call with a different block size can never clobber the
    canonical 2 km ``spatial_cv_per_fold.csv`` -- previously the CSV always went
    to the fixed ``OUT_DIR`` regardless of ``out_dir``, and the 5 km call
    silently overwrote the 2 km file.

    Returns ``{"per_fold": DataFrame(fold, n_test, auc), "mean_auc",
    "sd_auc", "n_folds", "n_raw_blocks", "min_presence_per_block"}``. Also
    written to ``<summary_dir>/spatial_cv_per_fold.csv`` (``summary_dir`` =
    ``OUT_DIR`` when ``out_dir is None``, else ``out_dir``).
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    raw_blocks = (config.spatial_folds(samples) if block_km is None
                  else config.spatial_folds(samples, block_km=block_km))
    n_raw_blocks = len(set(raw_blocks.tolist()))
    holdouts = spatial_cv_membership(samples, min_presence_per_block=min_presence_per_block, block_km=block_km)

    universe = np.arange(len(samples))
    # out_dir=None -> canonical 2 km locations (rasters under SPATIAL_CV_DIR,
    # summary CSV under the module OUT_DIR); a non-default out_dir routes BOTH
    # under it, so the notebook's 5 km call can never clobber the 2 km per-fold
    # CSV. See this function's docstring.
    fold_out_dir = SPATIAL_CV_DIR if out_dir is None else Path(out_dir)
    summary_dir = OUT_DIR if out_dir is None else Path(out_dir)
    rasters = _fit_folds_cached(holdouts, samples, fold_out_dir, timeout)
    background = evaluate.sample_background(_background_reference_raster(), n=background_n, seed=seed)

    rows = []
    for fid in sorted(holdouts):
        held_out_idx = holdouts[fid]
        train_idx = np.setdiff1d(universe, held_out_idx)
        test_pts = samples.iloc[held_out_idx][["x", "y"]]
        auc = evaluate.oos_auc(rasters[fid], test_pts, train_idx, held_out_idx, background)
        rows.append(dict(fold=fid, n_test=int(len(held_out_idx)), auc=auc))
        print(f"[robustness] spatial_cv fold {fid}: n_test={len(held_out_idx)} AUC={auc:.4f}")

    df = pd.DataFrame(rows)
    aucs = df["auc"].to_numpy()
    result = dict(
        per_fold=df,
        mean_auc=float(np.mean(aucs)),
        sd_auc=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        n_folds=int(len(aucs)),
        n_raw_blocks=n_raw_blocks,
        min_presence_per_block=min_presence_per_block,
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / "spatial_cv_per_fold.csv", index=False)
    print(f"[robustness] spatial_cv: mean AUC = {result['mean_auc']:.4f} +/- {result['sd_auc']:.4f} "
          f"({result['n_folds']} folds, merged from {n_raw_blocks} raw 2km blocks)")
    return result


# ============================================================================
# LOEO -- leave-one-event-out.
# ============================================================================

def loeo_membership(samples=None):
    """``config.loeo_folds()`` over the canonical set -- ``event`` is a
    NATIVE column on ``config.canonical_samples()`` (attached at HWM-file
    parse time by ``pipeline.hwm``, never via a later coordinate join), so
    this reconciles to the full 277 with no drop possible. See this
    module's own docstring, point 2.
    """
    if samples is None:
        samples = config.canonical_samples()
    return config.loeo_folds(samples)


def loeo(samples=None, *, background_n=evaluate.DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED,
         out_dir=LOEO_DIR, timeout=maxent.DEFAULT_TIMEOUT_S):
    """Leave-one-event-out CV on the canonical 277-point set: for each of
    the (currently 11) flood events, fits MaxEnt (``maxent.fit_folds``,
    ``FINAL_CONFIG``) on every OTHER event's presence, then scores that
    event's own held-out presence against the SAME shared, seeded
    background sample ``spatial_cv`` uses (``evaluate.oos_auc``-style).

    Tests whether the model predicts where a genuinely NEW, unseen flood
    event floods -- a stricter question than spatial block CV, and the one
    that directly answers "are HWMs event-specific observations rather
    than a universal susceptibility signal?" No event is dropped or
    filtered out for having few test points (unlike 04c's own >=5pt
    "valid" subset) -- every event's AUC is reported, however noisy a
    small event's holdout might be; `n_test` is carried alongside so a
    reader can see which folds are small.

    Returns ``{"per_event": DataFrame(event, n_test, auc), "mean_auc",
    "sd_auc", "n_events"}``. Also written to
    ``OUT_DIR/loeo_per_event.csv``.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    holdouts = loeo_membership(samples)
    total = sum(len(v) for v in holdouts.values())
    assert total == len(samples), (
        f"[robustness] loeo: fold membership sums to {total}, expected {len(samples)} "
        f"-- config.loeo_folds() should always reconcile exactly (event is a native "
        f"column, no coordinate join involved); investigate a data drift"
    )

    universe = np.arange(len(samples))
    fold_out_dir = Path(out_dir)
    rasters = _fit_folds_cached(holdouts, samples, fold_out_dir, timeout)
    background = evaluate.sample_background(_background_reference_raster(), n=background_n, seed=seed)

    rows = []
    for event in sorted(holdouts):
        held_out_idx = holdouts[event]
        train_idx = np.setdiff1d(universe, held_out_idx)
        test_pts = samples.iloc[held_out_idx][["x", "y"]]
        auc = evaluate.oos_auc(rasters[event], test_pts, train_idx, held_out_idx, background)
        rows.append(dict(event=event, n_test=int(len(held_out_idx)), auc=auc))
        print(f"[robustness] loeo {event}: n_test={len(held_out_idx)} AUC={auc:.4f}")

    df = pd.DataFrame(rows)
    aucs = df["auc"].to_numpy()
    result = dict(
        per_event=df,
        mean_auc=float(np.mean(aucs)),
        sd_auc=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        n_events=int(len(aucs)),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "loeo_per_event.csv", index=False)
    print(f"[robustness] loeo: mean AUC = {result['mean_auc']:.4f} +/- {result['sd_auc']:.4f} "
          f"({result['n_events']} events)")
    return result


# ============================================================================
# Ablation -- drop predictor GROUPS, refit + score under the SAME protocol
# the baseline uses (never 0.9842).
#
# The 4 arms partition the real, live 17-predictor canonical modeling stack
# (``maxent.modeling_layers()`` -- confirmed against the full ranked
# jackknife table) into
# domain groups generalizing 04c's own ad hoc STORMWATER/TERRAIN_ONLY sets
# (04c's 21-predictor stack had different/extra names -- flow_accumulation,
# spi, median_income -- none of which survive this project's screening/
# log-transform stages; see pipeline.maxent's own "Canonical modeling
# stack" docstring section):
#   - infrastructure (3): distance_to_outfall, stormwater_density,
#     drainage_density -- direct stormwater/drainage-network proxies
#     (04c's own STORMWATER group, minus the now-absent layers).
#   - soils (5): available_water_storage, depth_to_restriction, ksat,
#     hydrologic_soil_group, curve_number -- pedologic properties plus the
#     one composite runoff-potential index (SCS curve number) that's
#     conventionally soil-primary even though it also depends on land
#     cover.
#   - terrain (6): dem, slope, curvature, distance_to_streams, twi, hand --
#     pure topographic/hydrogeomorphic layers (04c's TERRAIN_ONLY set,
#     re-expressed on this stack; flow_accumulation/spi are gone, twi/hand
#     take over the topographic-wetness role).
#   - land_use (3): nlcd_landcover, impervious_pct, population_density --
#     land cover / urbanization / development proxies.
# This is an interpretive domain grouping (not derived from the data
# itself) -- `ablation()` asserts the union of all arms EXACTLY equals the
# live `maxent.modeling_layers()` set every call, so a future predictor-
# stack change is caught loudly rather than silently ablating a stale
# group.
# ============================================================================

# CORRECTED 2026-08-12. ``drainage_density`` is derived
# from the USGS NHD -- a NATURAL channel network -- and Table 2 of the manuscript
# files it under "Terrain and hydro-geomorphic". It was nevertheless grouped with
# the municipal stormwater predictors, so the reported "infrastructure" ablation
# loss was partly the loss of a natural-channel term. Moved to ``terrain``.
# Infrastructure is now exactly the two as-built municipal layers, which is what
# the paper's claim is actually about.
ABLATION_ARMS = {
    "infrastructure": frozenset({"distance_to_outfall", "stormwater_density"}),
    "soils": frozenset({"available_water_storage", "depth_to_restriction", "ksat",
                        "hydrologic_soil_group", "curve_number"}),
    "terrain": frozenset({"dem", "slope", "curvature", "distance_to_streams", "twi",
                          "hand", "drainage_density"}),
    "land_use": frozenset({"nlcd_landcover", "impervious_pct", "population_density"}),
}

# Only "holdout_oos" is implemented: the canonical 194/83 config.split()
# partition + one fixed seeded background sample + evaluate.oos_auc
# scoring -- exactly evaluate.oos_auc's own mechanism. ablation() raises ValueError
# for anything else rather than silently defaulting.
SUPPORTED_ABLATION_PROTOCOLS = ("holdout_oos",)


def _protocol_fingerprint(protocol_name, test_idx, background):
    """A cheap fingerprint of an AUC's evaluation protocol: which held-out
    indices and which background sample produced it. ``_protocol_
    fingerprint`` itself DOES discriminate on `test_idx`/`background` --
    proven directly (``tests/test_robustness_folds.py::
    test_protocol_fingerprint_differs_when_test_idx_differs`` /
    ``..._differs_when_background_differs``).

    Honest caveat about how ``ablation()`` uses this, below: baseline's and
    every arm's fingerprint are computed from the SAME `test_idx`/
    `background` locals, bound ONCE and never reassigned anywhere in that
    function -- so equality between them there is guaranteed BY
    CONSTRUCTION, not something the runtime assert can actually fail to
    hold today. It is kept as a tripwire against a future refactor that
    recomputes `test_idx`/`background` per-arm (it would fire immediately
    if it ever did), not as today's leakage guard. The genuinely
    load-bearing, non-tautological check ``ablation()`` uses is
    ``_recorded_train_idx``: it reads back what each raster (baseline AND
    every arm) was ACTUALLY fit on from its own ``train_idx.csv`` file,
    independent of any in-memory variable -- so a STALE cached
    ablation-arm raster (``_fit_train_only_on_layers``'s own cache check is
    path-existence-only) is caught by what it really trained on, not by
    what the current call merely intended.
    """
    test_key = tuple(sorted(int(i) for i in test_idx))
    bg_key = (
        int(len(background)),
        round(float(background["x"].iloc[0]), 3) if len(background) else None,
        round(float(background["y"].iloc[-1]), 3) if len(background) else None,
    )
    return (protocol_name, test_key, bg_key)


def _recorded_train_idx(raster_path):
    """The train_idx ACTUALLY recorded next to `raster_path` -- every
    train-only fit in this pipeline (``maxent._fit_train_only``, and this
    module's own ``_fit_train_only_on_layers``) writes a ``train_idx.csv``
    alongside its ``flood.tif`` immediately after fitting. Read back from
    disk rather than trusted from any in-memory variable, so a raster
    silently reused from a STALE cache (fit under a since-changed
    ``config.split()``/`samples`, past whatever ``_fit_train_only_on_
    layers``'s own path-existence-only cache check would notice) is caught
    by what it was ACTUALLY trained on.

    Returns a 1-D int ndarray.
    """
    recorded_idx_path = Path(raster_path).parent / "train_idx.csv"
    return np.loadtxt(recorded_idx_path, skiprows=1, dtype=int).reshape(-1)


def _baseline_eval_raster(samples, train_idx, *, timeout):
    """The ablation baseline's full-17-predictor, train-only raster.
    Reuses the existing ``evaluate.EVAL_RASTER`` if its recorded
    ``train_idx.csv`` matches the freshly recomputed `train_idx` exactly
    (``config.split()`` is deterministic/seeded, so this should always
    hold -- checked explicitly, never blindly trusted, so a silent data
    drift would be caught here instead of comparing every ablation arm
    against a stale baseline). Falls back to a fresh ``maxent.fit_eval``
    fit otherwise.
    """
    eval_raster = Path(evaluate.EVAL_RASTER)
    recorded_idx_path = eval_raster.parent / "train_idx.csv"
    if eval_raster.exists() and recorded_idx_path.exists():
        recorded = np.loadtxt(recorded_idx_path, skiprows=1, dtype=int).reshape(-1)
        if sorted(recorded.tolist()) == sorted(int(i) for i in train_idx):
            return eval_raster
        print(f"[robustness] WARNING: {recorded_idx_path} does not match the freshly "
              f"recomputed train split -- refitting the ablation baseline from scratch")
    return maxent.fit_eval(train_idx, samples, timeout=timeout)


def _fit_train_only_on_layers(samples, train_idx, held_out_idx, layers, out_dir, *, timeout):
    """Like ``maxent._fit_train_only`` but restricted to `layers` (a subset
    of the full 17-predictor canonical stack), via ``maxent.
    _build_layer_subset``'s symlink mechanism (the jackknife
    machinery -- reused here rather than reimplemented). Idempotent (unlike
    ``maxent._fit_train_only``/``maxent.run()``, which always pass
    MaxEnt's own ``redoifexists`` and would refit even when nothing
    changed): if ``<out_dir>/flood.tif`` already exists, returns it
    directly without invoking MaxEnt again.

    **Invariant 2, enforced before anything else (even the cache-hit
    check)**: ``qa.assert_eval_model_excludes_test(train_idx,
    held_out_idx)`` -- the same fit-time guard ``maxent.fit_folds``/
    ``maxent._fit_train_only`` enforce for every OTHER model this pipeline
    fits. Ablation arms previously relied SOLELY on the scoring-time
    ``evaluate.oos_auc`` leakage assert to catch an overlapping
    train/held-out pair; this closes that gap so a future caller passing a
    contaminated `train_idx`/`held_out_idx` is caught here, before any
    subprocess (or a cache-hit) is trusted -- never only after a full
    MaxEnt fit has already run.
    """
    qa.assert_eval_model_excludes_test(train_idx, held_out_idx)

    out_dir = Path(out_dir)
    raster_path = out_dir / "flood.tif"
    if raster_path.exists():
        return raster_path
    out_dir.mkdir(parents=True, exist_ok=True)

    maxent.export_layers()   # idempotent -- no-op if layers/*.asc already exported
    subset_dir = maxent._build_layer_subset(sorted(layers), out_dir / "_layers")

    train_idx = np.asarray(train_idx)
    train_points = samples.iloc[train_idx]
    samples_csv = maxent.build_samples_csv(out_path=out_dir / "train_samples.csv", presence=train_points)

    result = maxent.run(samples_csv, subset_dir, out_dir,
                         features=maxent.FINAL_CONFIG["features"], beta=maxent.FINAL_CONFIG["beta"],
                         replicates=0, outputformat="cloglog", timeout=timeout)
    if result["rc"] != 0:
        raise RuntimeError(
            f"[robustness] ablation arm fit failed (out_dir={out_dir}, rc={result['rc']}); "
            f"see {result['log_path']}"
        )

    asc_path = out_dir / "flood.asc"
    if not asc_path.exists():
        raise FileNotFoundError(f"[robustness] ablation arm fit: expected {asc_path} not found")

    raster_path = maxent._asc_to_geotiff(asc_path, maxent._reference_raster(), raster_path)
    np.savetxt(out_dir / "train_idx.csv", train_idx, fmt="%d", header="train_idx", comments="")
    return raster_path


SCREENED_DIR = config.DATA / "predictors_screened"
COND_DIR = OUT_DIR.parent / "maxent" / "conditional"
COND_NESTED_CSV = OUT_DIR / "conditional_nested_models.csv"
COND_CORR_CSV = OUT_DIR / "conditional_correlations.csv"

# The confounders to test: could outfall proximity simply be
# standing in for receiving-water position, low ground, or development?
#   distance_to_streams -- natural channel position (outfalls DISCHARGE to them)
#   dem, hand           -- low-lying ground
#   impervious_pct      -- urban development
# A distance-to-road / access-effort layer is NOT in the modeling stack; it is
# handled separately as a negative control (see the access proxy in
# pipeline.predictors), and the gap must be stated rather than quietly omitted.
CONFOUNDERS = ("distance_to_streams", "dem", "hand", "impervious_pct")
FOCAL = "distance_to_outfall"


def conditional_importance(*, focal=FOCAL, confounders=CONFOUNDERS, samples=None,
                           background_n=evaluate.DEFAULT_BACKGROUND_N,
                           seed=config.GLOBAL_SEED, out_dir=COND_DIR,
                           timeout=maxent.DEFAULT_TIMEOUT_S):
    """Does the focal predictor add information CONDITIONAL on its confounders?

    Marginal importance cannot
    distinguish a genuine drainage-network effect from receiving-water position,
    low ground, or development, because outfalls sit at all three.

    Two complementary reads, both on the standard protocol (canonical 194/83
    split, one fixed seeded background, ``evaluate.oos_auc``):

    * **Nested models** -- confounders alone, then + focal. The AUC gain is the
      focal predictor's *incremental* contribution given the confounders, which
      is the quantity that matters here. Also the reverse (focal alone,
      then + confounders) and the full-stack drop.
    * **Rank correlations** between focal and each confounder at the presence
      points and over the background, so collinearity is visible rather than
      inferred.

    Returns ``(nested_df, corr_df)`` and writes both to ``OUT_DIR``.
    """
    from scipy.stats import spearmanr

    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    all_layers = set(maxent.modeling_layers())
    assert focal in all_layers, focal
    confounders = [c for c in confounders if c in all_layers]

    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx][["x", "y"]]
    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)
    out_dir = Path(out_dir)

    def _auc(tag, members):
        members = sorted(set(members))
        raster = _fit_train_only_on_layers(samples, train_idx, test_idx, members,
                                           out_dir / tag, timeout=timeout)
        a = evaluate.oos_auc(raster, test_pts, train_idx, test_idx, background)
        print(f"[robustness]   {tag:26s} n={len(members):2d} AUC={a:.4f}")
        return a

    conf_only = _auc("confounders", confounders)
    conf_plus = _auc("confounders_plus_focal", list(confounders) + [focal])
    focal_only = _auc("focal_only", [focal])
    full = _auc("full", all_layers)
    full_minus = _auc("full_minus_focal", all_layers - {focal})

    nested = pd.DataFrame([
        dict(model="confounders only", predictors=",".join(sorted(confounders)),
             auc=conf_only, increment=float("nan"), question=""),
        dict(model="confounders + focal", predictors=",".join(sorted(confounders) + [focal]),
             auc=conf_plus, increment=conf_plus - conf_only,
             question=f"incremental value of {focal} GIVEN the confounders"),
        dict(model="focal only", predictors=focal, auc=focal_only,
             increment=float("nan"), question=""),
        dict(model="focal + confounders (same as above, reversed read)",
             predictors=",".join(sorted(confounders) + [focal]), auc=conf_plus,
             increment=conf_plus - focal_only,
             question="incremental value of the confounders GIVEN the focal"),
        dict(model="full stack", predictors=f"all {len(all_layers)}", auc=full,
             increment=float("nan"), question=""),
        dict(model="full minus focal", predictors=f"all except {focal}", auc=full_minus,
             increment=full_minus - full,
             question=f"cost of dropping {focal} from the full stack"),
    ])
    COND_NESTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    nested.to_csv(COND_NESTED_CSV, index=False)

    # collinearity, at presences and over the background
    pres = samples[["x", "y"]]
    rows = []
    for where, pts in (("presence", pres), ("background", background[["x", "y"]])):
        vals = {}
        for lyr in [focal] + list(confounders):
            da = evaluate._open_raster(SCREENED_DIR / f"{lyr}.tif")
            vals[lyr] = evaluate._sample_xy(da, pts["x"].to_numpy(), pts["y"].to_numpy())
        f = vals[focal]
        for c in confounders:
            m = np.isfinite(f) & np.isfinite(vals[c])
            rho = float(spearmanr(f[m], vals[c][m]).statistic) if m.sum() > 3 else float("nan")
            rows.append(dict(where=where, focal=focal, confounder=c, spearman_rho=rho,
                             n=int(m.sum())))
    corr = pd.DataFrame(rows)
    corr.to_csv(COND_CORR_CSV, index=False)
    print(f"[robustness] wrote {COND_NESTED_CSV.name} and {COND_CORR_CSV.name}")
    return nested, corr


ABLATION_LOEO_DIR = OUT_DIR.parent / "maxent" / "ablation_loeo"
ABLATION_LOEO_CSV = OUT_DIR / "ablation_loeo.csv"


def ablation_loeo(arms=None, *, samples=None, background_n=evaluate.DEFAULT_BACKGROUND_N,
                  seed=config.GLOBAL_SEED, out_dir=ABLATION_LOEO_DIR,
                  timeout=maxent.DEFAULT_TIMEOUT_S):
    """Predictor-group ablation evaluated under leave-one-event-out CV.

    Methods promises the ablation is evaluated "not only on
    the single 70/30 hold-out but also under leave-one-event-out
    cross-validation"; this supplies the LOEO half.

    Why it matters beyond completeness: a single 70/30 split can make a group
    look dispensable because one particular set of 83 test points happens to be
    predictable without it. Averaging over eleven event-folds — each withholding
    a whole flood event — is a much harder test of whether a predictor group
    carries information that generalises to an *unseen event*.

    Same arms as ``ablation()`` (with `drainage_density` in terrain, corrected
    2026-08-12), and the same per-fold protocol ``loeo()`` uses.

    Returns ``arm, dropped, n_predictors, loeo_mean_auc, loeo_sd, delta_vs_baseline``
    and writes ``OUT_DIR/ablation_loeo.csv``.
    """
    if arms is None:
        arms = ABLATION_ARMS
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    all_layers = set(maxent.modeling_layers())
    events = sorted(samples["event"].unique())
    universe = np.arange(len(samples))
    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)
    out_dir = Path(out_dir)

    def _loeo_mean(tag, kept):
        aucs = []
        for ev in events:
            held = np.flatnonzero((samples["event"] == ev).to_numpy())
            if len(held) == 0:
                continue
            train_idx = np.setdiff1d(universe, held)
            qa.assert_no_leakage(train_idx, held)
            raster = _fit_train_only_on_layers(samples, train_idx, held, sorted(kept),
                                               out_dir / tag / f"fold_{ev}", timeout=timeout)
            aucs.append(evaluate.oos_auc(raster, samples.iloc[held][["x", "y"]],
                                         train_idx, held, background))
        a = np.asarray(aucs, dtype="float64")
        print(f"[robustness] ablation-LOEO {tag:16s} n_pred={len(kept):2d} "
              f"mean={a.mean():.4f} +/- {a.std(ddof=0):.4f} ({len(a)} events)", flush=True)
        return a

    base = _loeo_mean("baseline", all_layers)
    rows = [dict(arm="baseline", dropped="", n_predictors=len(all_layers),
                 loeo_mean_auc=float(base.mean()), loeo_sd=float(base.std(ddof=0)),
                 delta_vs_baseline=0.0, n_events=len(base))]

    for arm_name, dropped in arms.items():
        kept = all_layers - set(dropped)
        a = _loeo_mean(arm_name, kept)
        rows.append(dict(arm=arm_name, dropped=",".join(sorted(dropped)),
                         n_predictors=len(kept), loeo_mean_auc=float(a.mean()),
                         loeo_sd=float(a.std(ddof=0)),
                         delta_vs_baseline=float(a.mean() - base.mean()),
                         n_events=len(a)))

    df = pd.DataFrame(rows)
    ABLATION_LOEO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_LOEO_CSV, index=False)
    print(f"[robustness] wrote {ABLATION_LOEO_CSV.name}")
    return df


NESTED_DIR = OUT_DIR.parent / "maxent" / "nested_tuning"
NESTED_CSV = OUT_DIR / "nested_tuning_cv.csv"


def _fit_at_config(samples, train_idx, out_dir, *, beta, features,
                   timeout=maxent.DEFAULT_TIMEOUT_S):
    """One deterministic MaxEnt fit on ``samples.iloc[train_idx]`` at an
    ARBITRARY (beta, features), returning the cloglog GeoTIFF path.

    ``maxent._fit_train_only`` hardcodes ``FINAL_CONFIG``; nested tuning needs
    whatever the inner grid selected for that fold.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / "flood.tif"
    if tif.exists():
        return tif
    train_pts = samples.iloc[np.asarray(train_idx)]
    samples_csv = maxent.build_samples_csv(out_path=out_dir / "train_samples.csv",
                                           presence=train_pts)
    res = maxent.run(samples_csv, maxent.LAYERS_DIR, out_dir,
                     features=features, beta=beta, replicates=0,
                     outputformat="cloglog", timeout=timeout)
    if res["rc"] != 0:
        raise RuntimeError(f"[robustness] _fit_at_config: MaxEnt rc={res['rc']} in {out_dir}")
    maxent._asc_to_geotiff(out_dir / "flood.asc", maxent._reference_raster(), tif)
    return tif


def nested_tuning_cv(scheme="loeo", *, samples=None, background_n=evaluate.DEFAULT_BACKGROUND_N,
                     seed=config.GLOBAL_SEED, out_dir=NESTED_DIR,
                     timeout=maxent.DEFAULT_TIMEOUT_S, block_km=2):
    """Cross-validation with the AICc grid re-run INSIDE each training fold.

    Why: ``aicc_grid()``
    was run once on all 277 presences, and the winning (beta, feature-class) was
    then used for the 70/30, spatial-block and LOEO evaluations. The withheld
    points therefore influenced model COMPLEXITY before they were scored.
    Refitting coefficients after withholding does not undo that -- the
    hyperparameter already saw them.

    Here each fold: (1) runs the full 15-candidate grid on its TRAINING points
    only, (2) selects by AICc within the fold, (3) fits at that configuration,
    (4) scores the held-out points. Nothing about the held-out set touches the
    selection.

    **Expect the AUC to fall relative to the published figure.** That is the
    honest number, and reporting it strengthens the paper's methodological
    position rather than weakening it.

    ``scheme`` is ``"loeo"`` or ``"spatial"``.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    out_dir = Path(out_dir) / scheme

    if scheme == "loeo":
        folds = {ev: np.flatnonzero((samples["event"] == ev).to_numpy())
                 for ev in sorted(samples["event"].unique())}
    elif scheme == "spatial":
        # MUST reuse spatial_cv_membership, which MERGES raw blocks with fewer
        # than MIN_PRESENCE_PER_BLOCK presences into their nearest large block
        # (20 raw blocks -> 14 folds). A first attempt (2026-08-13) used
        # config.spatial_folds() directly and produced ~20 unmerged folds,
        # including several with 1-2 presence points, whose mean is NOT
        # comparable to the published 0.941. Fold construction must match the
        # estimate it is being compared against.
        folds = spatial_cv_membership(samples, block_km=block_km)
    else:
        raise ValueError(f"[robustness] nested_tuning_cv: unknown scheme {scheme!r}")

    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)
    universe = np.arange(len(samples))
    rows = []
    for fold, held in folds.items():
        if len(held) == 0:
            continue
        train_idx = np.setdiff1d(universe, held)
        qa.assert_no_leakage(train_idx, held)
        fdir = out_dir / f"fold_{fold}"

        train_csv = maxent.build_samples_csv(out_path=fdir / "grid_samples.csv",
                                             presence=samples.iloc[train_idx])
        grid = maxent.aicc_grid(samples_csv=train_csv, layers_dir=maxent.LAYERS_DIR,
                                tuning_dir=fdir / "tuning", timeout=timeout)
        best = maxent.select_best(grid)

        raster = _fit_at_config(samples, train_idx, fdir / "fit",
                                beta=best["beta"], features=best["feature_classes"],
                                timeout=timeout)
        held_pts = samples.iloc[held][["x", "y"]]
        auc = evaluate.oos_auc(raster, held_pts, train_idx, held, background)
        rows.append(dict(scheme=scheme, fold=fold, n_test=len(held),
                         beta=best["beta"], features=best["feature_classes"],
                         k=best.get("k"), auc=auc))
        print(f"[robustness] nested {scheme} fold {str(fold):12s} n={len(held):3d} "
              f"beta={best['beta']:<5} fc={best['feature_classes']:<5} AUC={auc:.4f}", flush=True)

    df = pd.DataFrame(rows)
    path = NESTED_CSV.with_name(f"nested_tuning_{scheme}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[robustness] nested {scheme}: mean AUC = {df['auc'].mean():.4f} "
          f"+/- {df['auc'].std(ddof=0):.4f} over {len(df)} folds -> {path.name}")
    return df


SFHA_DIR = OUT_DIR.parent / "maxent" / "sfha_split"
SFHA_TAGS_CSV = OUT_DIR / "presence_sfha_tags.csv"
SFHA_SUMMARY_CSV = OUT_DIR / "sfha_stratified_summary.csv"


def tag_presence_by_sfha(samples=None, fema_gpkg=None):
    """Tag every presence point as inside / outside the FEMA regulatory
    floodplain (SFHA, the 1 % annual-chance zone).

    The eleven events pool riverine
    and pluvial flooding, and the paper's claim is about *stormwater-driven*
    urban flooding. Marks OUTSIDE the SFHA are the cleanest available proxy for
    the pluvial subset: the regulatory floodplain is where riverine flooding is
    mapped, so a mark outside it is unlikely to be river-driven.

    This is a spatial proxy for mechanism, not a mechanism classification.
    Rainfall and river-stage records would be needed for the latter; that
    remains open.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    fema_gpkg = fema_gpkg or evaluate.FEMA_GPKG

    gdf = gpd.read_file(fema_gpkg)
    sfha = gdf[gdf["FLD_ZONE"].astype(str).isin(evaluate.FEMA_SFHA_ZONES)]
    union = sfha.geometry.union_all() if hasattr(sfha.geometry, "union_all")         else sfha.geometry.unary_union

    pts = gpd.GeoDataFrame(
        samples.copy(),
        geometry=[Point(x, y) for x, y in zip(samples["x"], samples["y"])],
        crs=gdf.crs)
    pts["in_sfha"] = pts.geometry.within(union)

    out = pts.drop(columns="geometry")
    SFHA_TAGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SFHA_TAGS_CSV, index=False)
    n_in = int(out["in_sfha"].sum())
    print(f"[robustness] SFHA tagging: {n_in} inside / {len(out) - n_in} outside "
          f"({100 * (len(out) - n_in) / len(out):.0f} % outside the regulatory floodplain)")
    return out


def sfha_stratified_analysis(*, samples=None, timeout=maxent.DEFAULT_TIMEOUT_S,
                             background_n=evaluate.DEFAULT_BACKGROUND_N,
                             seed=config.GLOBAL_SEED, out_dir=SFHA_DIR):
    """Refit on the OUTSIDE-SFHA (pluvial-proxy) marks only and compare.

    If outfall proximity is a riverine artefact -- outfalls discharge to
    channels, so they sit in floodplains -- the association should weaken sharply
    once floodplain marks are removed. If it holds, the stormwater-driven reading
    is supported on the subset where it matters most.

    Fits the full stack and `distance_to_outfall` alone on the outside-SFHA
    subset, scored against a fresh background under the standard protocol.
    """
    tagged = tag_presence_by_sfha(samples=samples)
    outside = tagged[~tagged["in_sfha"]].reset_index(drop=True)
    if len(outside) < 40:
        raise ValueError(f"[robustness] sfha_stratified_analysis: only {len(outside)} "
                         f"outside-SFHA marks; too few to refit")

    all_layers = sorted(maxent.modeling_layers())
    tr, te = config.split(outside)
    test_pts = outside.iloc[te][["x", "y"]]
    bg = evaluate.sample_background(_background_reference_raster(), n=background_n, seed=seed)
    out_dir = Path(out_dir)

    rows = []
    for tag, members in (("full_stack", all_layers), ("outfall_only", ["distance_to_outfall"])):
        raster = _fit_train_only_on_layers(outside, tr, te, sorted(members),
                                           out_dir / tag, timeout=timeout)
        auc = evaluate.oos_auc(raster, test_pts, tr, te, bg)
        rows.append(dict(subset="outside_SFHA", model=tag, n_presence=len(outside),
                         n_train=len(tr), n_test=len(te), n_predictors=len(members),
                         auc=auc))
        print(f"[robustness] outside-SFHA {tag:14s} n={len(outside)} AUC={auc:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(SFHA_SUMMARY_CSV, index=False)
    print(f"[robustness] wrote {SFHA_SUMMARY_CSV.name}")
    return df


SINGLE_PRED_DIR = OUT_DIR.parent / "maxent" / "single_predictor"
SINGLE_PRED_CSV = OUT_DIR / "single_predictor_baselines.csv"

# Small named combinations: outfall distance alone, and outfall distance
# plus the leading topographic terms.
SINGLE_PRED_COMBOS = {
    "outfall+dem": ("distance_to_outfall", "dem"),
    "outfall+hand": ("distance_to_outfall", "hand"),
    "outfall+dem+hand": ("distance_to_outfall", "dem", "hand"),
    "outfall+aws": ("distance_to_outfall", "available_water_storage"),
}


def single_predictor_baselines(layers=None, combos=None, *, samples=None,
                               background_n=evaluate.DEFAULT_BACKGROUND_N,
                               seed=config.GLOBAL_SEED, out_dir=SINGLE_PRED_DIR,
                               timeout=maxent.DEFAULT_TIMEOUT_S):
    """One MaxEnt per single predictor, plus a few named combinations, scored
    under the baseline protocol.

    The question: how much performance does a model containing distance to
    outfall alone obtain? Without these baselines it is difficult to know
    whether the six-model exercise adds predictive information beyond one
    spatial proxy, and the question deserves a number rather than an argument. If
    one predictor nearly matches the full model, that is a finding worth
    reporting, not an embarrassment -- and it is far better to publish it than
    to have a reviewer compute it.

    Same protocol as ``ablation()`` and ``ablation_group_only()``: canonical
    194/83 split, one fixed seeded background, ``evaluate.oos_auc``. Directly
    comparable to both.

    Returns ``arm, predictors, n_predictors, auc, delta_vs_full, pct_of_full``
    and writes ``OUT_DIR/single_predictor_baselines.csv``.
    """
    if layers is None:
        layers = sorted(maxent.modeling_layers())
    if combos is None:
        combos = SINGLE_PRED_COMBOS
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    all_layers = set(maxent.modeling_layers())
    unknown = (set(layers) | {m for c in combos.values() for m in c}) - all_layers
    assert not unknown, f"[robustness] single_predictor_baselines: not modeling layers: {unknown}"

    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx][["x", "y"]]
    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)

    full_raster = _baseline_eval_raster(samples, train_idx, timeout=timeout)
    full_auc = evaluate.oos_auc(full_raster, test_pts, train_idx, test_idx, background)
    print(f"[robustness] single_predictor full model ({len(all_layers)}): AUC={full_auc:.4f}")

    out_dir = Path(out_dir)
    rows = [dict(arm="ALL", predictors=",".join(sorted(all_layers)),
                 n_predictors=len(all_layers), auc=full_auc,
                 delta_vs_full=0.0, pct_of_full=100.0)]

    def _score(tag, members):
        members = sorted(set(members))
        raster = _fit_train_only_on_layers(samples, train_idx, test_idx, members,
                                           out_dir / tag, timeout=timeout)
        auc = evaluate.oos_auc(raster, test_pts, train_idx, test_idx, background)
        rows.append(dict(arm=tag, predictors=",".join(members), n_predictors=len(members),
                         auc=auc, delta_vs_full=auc - full_auc,
                         pct_of_full=100.0 * auc / full_auc))
        print(f"[robustness]   {tag:34s} n={len(members)} AUC={auc:.4f}")

    for lyr in layers:
        _score(lyr, [lyr])
    for tag, members in combos.items():
        _score(tag, members)

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    SINGLE_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SINGLE_PRED_CSV, index=False)
    print(f"[robustness] wrote {SINGLE_PRED_CSV.name}")
    return df


def _unlink_stubbornly(path, *, tries=5, pause=0.4):
    """Delete a file that a lazily-opened raster may still hold.

    rioxarray/GDAL keep the file handle alive until the dataset is collected, and
    Windows refuses to unlink an open file. A collection plus a short retry clears
    it; POSIX unlinks on the first pass and never sleeps.
    """
    import gc
    import time

    for attempt in range(tries):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            gc.collect()
            time.sleep(pause * (attempt + 1))
    print(f"[robustness] WARNING: could not delete {path}")
    return False


TEMPORAL_DIR = config.DATA / "temporal_check"
TEMPORAL_CSV = OUT_DIR / "temporal_outfall_check.csv"
FIRST_EVENT_YEAR = 2007


def temporal_outfall_check(*, cutoff_year=FIRST_EVENT_YEAR, samples=None,
                           background_n=evaluate.DEFAULT_BACKGROUND_N,
                           seed=config.GLOBAL_SEED, out_dir=TEMPORAL_DIR,
                           timeout=maxent.DEFAULT_TIMEOUT_S):
    """Does the outfall association depend on outfalls built after the floods?

    The marks span 2007--2024 but
    every predictor is contemporary, so an outfall built in 2015 is credited
    with predicting a 2007 flood. This is not
    peripheral, because the affected layer is the strongest single predictor.

    A full event-by-event reconstruction is impossible: only ~29 % of outfalls
    carry a usable ``YEARBUILT``. What is possible is a one-shot lower bound --
    rebuild ``distance_to_outfall`` from **only** the outfalls demonstrably
    built on or before the first event, and refit a single-predictor model
    under the standard protocol.

    The check is conservative in two independent directions, and both belong in
    any sentence written from it: it discards the undated outfalls, many of
    which are certainly old, so the restricted network is far sparser than the
    true 2007 network; and a sparser network makes the predictor noisier, which
    biases the test *against* finding an association. A surviving association
    is therefore a lower bound, not an estimate.

    Same protocol as ``single_predictor_baselines()`` -- canonical 194/83 split,
    one fixed seeded background, ``evaluate.oos_auc`` -- so the ``outfall_ALL``
    row here must reproduce the ``distance_to_outfall`` row there.

    The restricted layer is staged into ``predictors_screened/`` and
    ``maxent/layers/`` only for the duration of the fit and removed afterwards:
    ``screen._load_stack()`` globs those directories, so a file left behind
    silently becomes an 18th predictor.
    """
    import shutil

    import geopandas as gpd
    import shapely
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    from pipeline import predictors as _pred

    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    name = f"distance_to_outfall_pre{cutoff_year}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- outfalls demonstrably older than the first event --------------------
    disp = gpd.read_file(_pred.GDB_PATH, layer="Stormwater_Discharge_Points").to_crs(_pred.TGT_CRS)
    outfalls = disp[disp["DISPTTYPE"] == "Outfall"]
    year = pd.to_numeric(outfalls["YEARBUILT"], errors="coerce")
    dated = (year > 1800) & (year < 2030)
    old = outfalls[dated & (year <= cutoff_year)]
    print(f"[robustness] outfalls {len(outfalls)}; dated {int(dated.sum())} "
          f"({100 * dated.mean():.0f} %); built <= {cutoff_year}: {len(old)}")
    assert len(old) >= 50, f"[robustness] only {len(old)} pre-{cutoff_year} outfalls -- too few to fit"

    # -- rebuild the distance raster on that subset --------------------------
    dem = _pred._load_cached("dem", _pred.OUT_DIR)
    transform = dem.rio.transform()
    burned = rasterize([(shapely.force_2d(g), 1) for g in old.geometry if g is not None],
                       out_shape=dem.shape, transform=transform, fill=0, dtype="uint8")
    dist = distance_transform_edt(burned == 0, sampling=abs(transform.a)).astype("float32")
    dist[~np.isfinite(dem.values)] = np.nan
    tif = out_dir / f"{name}.tif"
    dem.copy(data=dist).rio.write_crs(dem.rio.crs).rio.to_raster(tif)

    # -- fit both arms, same protocol ---------------------------------------
    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx][["x", "y"]]
    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)

    staged = config.DATA / "predictors_screened" / f"{name}.tif"
    asc = maxent.LAYERS_DIR / f"{name}.asc"
    n_asc_before = len(list(maxent.LAYERS_DIR.glob("*.asc")))
    shutil.copy(tif, staged)
    try:
        maxent.export_layers(layers=[name], pred_dir=staged.parent, out_dir=maxent.LAYERS_DIR)
        rows = []
        for tag, layer, n_out in ((f"outfall_pre{cutoff_year}_only", name, len(old)),
                                  ("outfall_ALL", "distance_to_outfall", len(outfalls))):
            raster = _fit_train_only_on_layers(samples, train_idx, test_idx, [layer],
                                               out_dir / tag, timeout=timeout)
            auc = evaluate.oos_auc(raster, test_pts, train_idx, test_idx, background)
            rows.append(dict(arm=tag, layer=layer, n_outfalls=n_out, auc=auc))
            print(f"[robustness]   {tag:26s} n_outfalls={n_out:5d} AUC={auc:.4f}")
    finally:
        # Each removal gets its own attempt: on Windows the reader that exported
        # the layer can still hold the GeoTIFF, and a PermissionError on the
        # first unlink would otherwise skip the second and strand the .asc --
        # which is exactly how a staged layer becomes an 18th predictor.
        for victim in (staged, asc):
            _unlink_stubbornly(victim)
        leaked = [p for p in (staged, asc) if p.exists()]
        n_after = len(list(maxent.LAYERS_DIR.glob("*.asc")))
        assert not leaked and n_after == n_asc_before, (
            f"[robustness] staged layer leaked: {[p.name for p in leaked]}; layers/ has "
            f"{n_after} asc, expected {n_asc_before}. A file left in either directory "
            f"silently joins the modeling stack -- delete it before running anything else.")

    df = pd.DataFrame(rows)
    restricted = df.loc[df["arm"].str.endswith("_only"), "auc"].iloc[0]
    published = df.loc[df["arm"] == "outfall_ALL", "auc"].iloc[0]
    df["delta_vs_all"] = df["auc"] - published
    TEMPORAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEMPORAL_CSV, index=False)
    print(f"[robustness] delta = {restricted - published:+.4f}; wrote {TEMPORAL_CSV.name}")
    return df


GROUP_ONLY_DIR = OUT_DIR.parent / "maxent" / "group_only"
GROUP_ONLY_CSV = OUT_DIR / "group_only_summary.csv"


def ablation_group_only(arms=None, *, samples=None,
                        background_n=evaluate.DEFAULT_BACKGROUND_N,
                        seed=config.GLOBAL_SEED, out_dir=GROUP_ONLY_DIR,
                        timeout=maxent.DEFAULT_TIMEOUT_S):
    """Fit MaxEnt on ONE predictor group at a time and score it under the
    baseline protocol.

    ``ablation()`` answers "what does
    removing this group cost?", which under collinearity can be near zero for a
    group that is perfectly informative on its own. It cannot answer "how much
    does this group know by itself?" -- and that is the question the paper's
    terrain claim actually turns on.

    The manuscript states in five places that terrain is uninformative, resting
    on a single externally-formulated geomorphic index (GFI, AUC 0.47). That
    conflates *terrain* with *one particular index of terrain*. A same-algorithm,
    same-folds, same-background terrain-only MaxEnt is the like-for-like
    comparison.

    Same protocol as ``ablation()`` throughout: the canonical 194/83 split, one
    fixed seeded background, ``evaluate.oos_auc`` scoring -- so these AUCs are
    directly comparable to the ablation baseline and to each other.

    Returns ``arm, kept, n_predictors, auc, delta_vs_baseline, protocol`` and
    writes ``OUT_DIR/group_only_summary.csv``.
    """
    if arms is None:
        arms = ABLATION_ARMS
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    all_layers = set(maxent.modeling_layers())
    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx][["x", "y"]]
    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)

    baseline_raster = _baseline_eval_raster(samples, train_idx, timeout=timeout)
    baseline_auc = evaluate.oos_auc(baseline_raster, test_pts, train_idx, test_idx,
                                    background)
    print(f"[robustness] group_only baseline (all {len(all_layers)}): "
          f"AUC={baseline_auc:.4f}")

    out_dir = Path(out_dir)
    rows = [dict(arm="all_predictors", kept=",".join(sorted(all_layers)),
                 n_predictors=len(all_layers), auc=baseline_auc,
                 delta_vs_baseline=0.0, protocol="holdout_oos")]

    for arm_name, members in arms.items():
        kept = sorted(set(members))
        raster = _fit_train_only_on_layers(samples, train_idx, test_idx, kept,
                                           out_dir / arm_name, timeout=timeout)
        auc = evaluate.oos_auc(raster, test_pts, train_idx, test_idx, background)
        rows.append(dict(arm=f"{arm_name}_only", kept=",".join(kept),
                         n_predictors=len(kept), auc=auc,
                         delta_vs_baseline=auc - baseline_auc,
                         protocol="holdout_oos"))
        print(f"[robustness] group_only {arm_name}_only ({len(kept)} predictors): "
              f"AUC={auc:.4f}")

    df = pd.DataFrame(rows)
    GROUP_ONLY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GROUP_ONLY_CSV, index=False)
    print(f"[robustness] wrote {GROUP_ONLY_CSV.name}")
    return df


def ablation(arms=None, *, baseline_protocol="holdout_oos", samples=None,
             background_n=evaluate.DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED,
             out_dir=ABLATION_DIR, timeout=maxent.DEFAULT_TIMEOUT_S):
    """Drop predictor GROUPS (``arms``, default ``ABLATION_ARMS``) one at a
    time, refit MaxEnt on the SAME 194-train-point complement restricted to
    the reduced layer set, and score EVERY arm -- and the full-17-predictor
    baseline -- under the IDENTICAL evaluation protocol: the same 83 test
    points (``config.split()``), the same fixed seeded background sample,
    the same ``evaluate.oos_auc`` scoring call. ``ΔAUC = arm_AUC -
    baseline_AUC`` is therefore always same-protocol, like-for-like --
    NEVER differenced against the old 0.9842 resubstitution-flavored
    number (see this module's own docstring, point 3).

    `baseline_protocol` must be one of ``SUPPORTED_ABLATION_PROTOCOLS``
    (checked FIRST, before any data/raster I/O) -- currently only
    ``"holdout_oos"``. Every row's `protocol` is fingerprinted
    (``_protocol_fingerprint``) against the SAME `test_idx`/`background`
    the baseline used -- a structural tripwire (guaranteed to hold today
    by shared-locals construction; see ``_protocol_fingerprint``'s own
    docstring) against a future refactor that recomputes either per-arm.
    The check that actually DOES carry real, non-tautological leakage
    protection is ``_recorded_train_idx``: every arm's raster's own
    recorded ``train_idx.csv`` is compared against the baseline raster's,
    so a stale/mismatched cached ablation-arm raster (fit under some
    earlier, different train split) is caught by what it was ACTUALLY
    fit on, not by what this call intended.

    Returns a DataFrame: ``arm, dropped, n_predictors, auc, delta_auc,
    protocol`` -- first row is the baseline (``arm="baseline"``,
    ``dropped=""``, ``delta_auc=0.0``). Also written to
    ``OUT_DIR/ablation_summary.csv``.
    """
    if baseline_protocol not in SUPPORTED_ABLATION_PROTOCOLS:
        raise ValueError(
            f"[robustness] ablation: unsupported baseline_protocol {baseline_protocol!r}; "
            f"supported: {SUPPORTED_ABLATION_PROTOCOLS}"
        )
    if arms is None:
        arms = ABLATION_ARMS
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    all_layers = set(maxent.modeling_layers())
    seen, arm_union = set(), set()
    for name, members in arms.items():
        members = set(members)
        overlap = seen & members
        assert not overlap, f"[robustness] ablation: arm {name!r} overlaps another arm on {overlap}"
        seen |= members
        arm_union |= members
    assert arm_union == all_layers, (
        f"[robustness] ablation: arms do not exactly partition the {len(all_layers)} "
        f"canonical modeling predictors -- missing={all_layers - arm_union}, "
        f"extra={arm_union - all_layers}"
    )

    train_idx, test_idx = config.split(samples)
    test_pts = samples.iloc[test_idx][["x", "y"]]
    background = evaluate.sample_background(_background_reference_raster(), n=background_n, seed=seed)

    baseline_raster = _baseline_eval_raster(samples, train_idx, timeout=timeout)
    baseline_auc = evaluate.oos_auc(baseline_raster, test_pts, train_idx, test_idx, background)
    baseline_fp = _protocol_fingerprint(baseline_protocol, test_idx, background)
    baseline_train_idx_recorded = _recorded_train_idx(baseline_raster)
    print(f"[robustness] ablation baseline (all {len(all_layers)} predictors): AUC={baseline_auc:.4f}")

    out_dir = Path(out_dir)
    rows = [dict(arm="baseline", dropped="", n_predictors=len(all_layers),
                 auc=baseline_auc, delta_auc=0.0, protocol=baseline_protocol)]

    for arm_name, dropped in arms.items():
        dropped = set(dropped)
        kept = sorted(all_layers - dropped)
        arm_raster = _fit_train_only_on_layers(samples, train_idx, test_idx, kept,
                                                out_dir / arm_name, timeout=timeout)
        arm_auc = evaluate.oos_auc(arm_raster, test_pts, train_idx, test_idx, background)

        # Structural tripwire only -- guaranteed to hold today because baseline_fp/arm_fp
        # are both derived from the SAME test_idx/background locals (bound once above,
        # never reassigned); would fire if a future refactor ever recomputed either per-arm.
        arm_fp = _protocol_fingerprint(baseline_protocol, test_idx, background)
        assert baseline_fp == arm_fp, (
            f"[robustness] ablation: arm {arm_name!r} was scored under a DIFFERENT protocol "
            f"than the baseline ({arm_fp} != {baseline_fp}) -- ΔAUC would be apples-to-oranges"
        )
        # The REAL, non-tautological check: what train set was arm_raster ACTUALLY fit on
        # (read back from its own train_idx.csv), independent of whether this call reused a
        # cached raster or fit a fresh one -- _fit_train_only_on_layers's own cache-hit check
        # is path-existence-only, so a stale arm raster (fit under an earlier, different
        # train_idx) would otherwise be scored and differenced against the baseline silently.
        arm_train_idx_recorded = _recorded_train_idx(arm_raster)
        assert sorted(arm_train_idx_recorded.tolist()) == sorted(baseline_train_idx_recorded.tolist()), (
            f"[robustness] ablation: arm {arm_name!r}'s raster ({arm_raster}) was ACTUALLY fit "
            f"on a DIFFERENT train set than the baseline raster ({baseline_raster}) recorded -- "
            f"likely a stale cached ablation-arm raster left over from an earlier "
            f"config.split()/samples; ΔAUC would be apples-to-oranges"
        )
        delta = arm_auc - baseline_auc
        rows.append(dict(arm=arm_name, dropped=",".join(sorted(dropped)), n_predictors=len(kept),
                          auc=arm_auc, delta_auc=delta, protocol=baseline_protocol))
        print(f"[robustness] ablation {arm_name} (dropped {sorted(dropped)}): "
              f"n_predictors={len(kept)} AUC={arm_auc:.4f} (delta={delta:+.4f})")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    return df


# ============================================================================
# Matched-threshold stats (AUPRC/Brier/bootstrap-CI/McNemar),
# Moran's I on residuals, and the corrected significance test. Ports
# the original notebook 04c_robustness_checks, Sections 15-18 (AUPRC, calibration/Brier,
# bootstrap CI, Wilcoxon+McNemar) and Section 13 (Moran's I), with TWO
# deliberate fixes vs. that source:
#
# 1. **THE THRESHOLD FIX.** 04c's Section 18b McNemar test (and its Section
#    17 bootstrap-CI confusion-matrix metrics) binarized every model's
#    continuous score at a single FIXED 0.5 cutoff
#    (``pred1 = (d1['p'] >= 0.5)``) -- meaningless for MaxEnt/Logistic
#    Regression, whose own MaxSS operating thresholds sit at ~0.04-0.08 (see
#    model_compare's metrics table), and
#    different again for the four tree-ensemble comparators (~0.37-0.80).
#    Every binarization below (McNemar's own b/c discordant-pair counts)
#    uses EACH model's OWN MaxSS threshold instead
#    (``_binarization_thresholds()``, never 0.5 for any model -- regression-
#    guarded by ``tests/test_stats.py::test_no_fixed_half_threshold``).
#
# 2. **THE SIGNIFICANCE FIX.** 04c's Section 18a used a Wilcoxon signed-rank
#    test on LOEO fold AUCs across only 8 valid (>=5pt) events -- with n=8
#    paired differences all pointing the same direction, the two-sided
#    Wilcoxon p-value hits its OWN resolution floor (p=0.0078, the smallest
#    attainable two-sided Wilcoxon p at n=8) REGARDLESS of the actual effect
#    size, so "significant" there is an artifact of too few folds, not
#    evidence of a well-quantified gap. ``loeo_significance_test()`` below
#    REPLACES this with the Nadeau-Bengio (2003) corrected resampled
#    t-test -- the standard correction for exactly this situation (comparing
#    two models across CV folds whose training sets overlap, violating the
#    naive paired t-test's independence assumption and understating its true
#    variance) -- reported alongside a plain effect size (Cohen's d) and a
#    naive CI for transparency, never a bare Wilcoxon p. There is no
#    ``wilcoxon`` import or call anywhere in this module (regression-guarded
#    by ``tests/test_stats.py::test_significance_test_is_not_wilcoxon_at_p_floor``).
#
# Reuses this module's OWN ``loeo()``/``loeo_membership()`` (cached on
# disk) for MaxEnt's per-event AUCs, and the
# ``model_compare.py`` building blocks (``build_background``/
# ``_presence_features``/the five ``_fit_*`` helpers) for both the matched-
# test-set scores (``_matched_test_scores``) and a NEW per-event XGBoost
# refit (``_loeo_xgboost_aucs``) needed to give the significance test a
# genuine SECOND model's AUC on the identical 11 folds. ``model_compare.
# run()`` itself only returns AGGREGATED per-model metrics (auc, tss, ...),
# never the raw ``(y_true, y_score)`` arrays AUPRC/Brier/bootstrap-CI/
# McNemar all need, and it never touches LOEO at all -- so this module
# cannot just call it a second time for either need. ``_matched_test_scores``
# therefore duplicates ``run()``'s own ~15-line X_train/X_test assembly
# (unavoidable without restructuring model_compare.py) but reuses every one of ``run()``'s own tested fitting/feature
# functions directly -- never a re-derivation of the fitting math itself --
# and ``tests/test_stats.py::test_matched_scores_agree_with_model_compare_run``
# cross-checks the two produce the SAME per-model AUC, so a future drift
# between this module's copy and ``model_compare.run()``'s own assembly
# logic is caught immediately rather than silently diverging.
#
# **No new MaxEnt fits**: MaxEnt is always scored by
# resampling its EXISTING ``evaluate.EVAL_RASTER`` (the
# train-only raster) -- never refit. The 5 ML comparators (fast, seconds
# each) ARE refit here (``_matched_test_scores``) because ``model_compare.
# run()`` does not expose raw scores; XGBoost is ALSO refit per-LOEO-event
# (``_loeo_xgboost_aucs``) purely for the significance test's second model.
# ============================================================================

_MATCHED_SCORES_CACHE = {}


def _ml_model_scores(model, X_train, y_train, X_test, y_test):
    """``(y_true, y_score, threshold)`` for one already-fitted presence-
    background classifier -- the MaxSS threshold is Youden's J on the
    model's OWN TRAIN predictions (``evaluate.maxss_threshold_from_roc``),
    identical to ``model_compare._score_ml_model``'s own threshold source
    (never derived from test)."""
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    threshold = evaluate.maxss_threshold_from_roc(y_train, prob_train)
    return dict(y_true=np.asarray(y_test), y_score=np.asarray(prob_test), threshold=float(threshold))


def _matched_test_scores(background_n=model_compare.DEFAULT_BACKGROUND_N,
                          seed=config.GLOBAL_SEED, force_refresh=False):
    """Per-model ``{y_true, y_score, threshold, x, y}`` on model_compare's
    IDENTICAL matched test set (83 canonical test-presence points + the
    matching 30% test partition of a presence-excluded 10,000-pixel
    background pool) for all 6 models -- MaxEnt (``evaluate.EVAL_RASTER``,
    NOT refit) plus the 5 ML comparators (``model_compare``'s own fitting
    recipe, refit here to capture raw scores ``model_compare.run()`` does
    not expose). In-memory-cached (keyed on ``(background_n, seed)``) so
    the several stats functions below that each need this (AUPRC/Brier,
    bootstrap CI, McNemar, ``_binarization_thresholds``, Moran's I) do not
    each independently pay the ~5 ML model fits again within one process;
    pass ``force_refresh=True`` to bypass.

    Returns ``{model_name: {"y_true", "y_score", "threshold", "x", "y"}}``
    -- ``x``/``y`` are the SAME coordinate arrays (matched test point
    order: test-presence then test-background) for every model, enabling
    ``morans_i_residuals`` to consume any one model's entry directly.
    """
    key = (background_n, seed)
    if not force_refresh and key in _MATCHED_SCORES_CACHE:
        return _MATCHED_SCORES_CACHE[key]

    layers = maxent.modeling_layers()
    canonical = config.canonical_samples()
    train_idx, test_idx = config.split(canonical)
    qa.assert_no_leakage(train_idx, test_idx)

    pres_feats = model_compare._presence_features(canonical, layers=layers)
    background = model_compare.build_background(n=background_n, seed=seed, layers=layers, presence=canonical)
    bg_feats = background[list(layers)].copy()
    for c in model_compare.CATEGORICAL:
        bg_feats[c] = bg_feats[c].astype(int)

    y_pres = np.ones(len(pres_feats), dtype=int)
    y_bg = np.zeros(len(bg_feats), dtype=int)

    bg_train_idx, bg_test_idx = config.split(background)
    qa.assert_no_leakage(bg_train_idx, bg_test_idx)

    X_train = pd.concat(
        [pres_feats.iloc[train_idx].reset_index(drop=True),
         bg_feats.iloc[bg_train_idx].reset_index(drop=True)],
        ignore_index=True,
    )
    y_train = np.concatenate([y_pres[train_idx], y_bg[bg_train_idx]])

    X_test = pd.concat(
        [pres_feats.iloc[test_idx].reset_index(drop=True),
         bg_feats.iloc[bg_test_idx].reset_index(drop=True)],
        ignore_index=True,
    )
    y_test = np.concatenate([y_pres[test_idx], y_bg[bg_test_idx]])

    x_test = np.concatenate([canonical.iloc[test_idx]["x"].to_numpy(),
                              background.iloc[bg_test_idx]["x"].to_numpy()])
    y_coord_test = np.concatenate([canonical.iloc[test_idx]["y"].to_numpy(),
                                    background.iloc[bg_test_idx]["y"].to_numpy()])

    # THE CRITICAL FIX is model_compare's, reused unchanged here: the boosting
    # models' eval_set comes ONLY from _inner_split(X_train, y_train) --
    # X_test/y_test are never touched until each already-fitted model's own
    # predict_proba call inside _ml_model_scores, below.
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = model_compare._inner_split(X_train, y_train, seed=seed)

    scores = {}

    lr = model_compare._fit_logistic_regression(X_train, y_train, layers)
    scores["Logistic Regression"] = _ml_model_scores(lr, X_train, y_train, X_test, y_test)

    rf = model_compare._fit_random_forest(X_train, y_train)
    scores["Random Forest"] = _ml_model_scores(rf, X_train, y_train, X_test, y_test)

    xgb_model = model_compare._fit_xgboost(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner)
    scores["XGBoost"] = _ml_model_scores(xgb_model, X_train, y_train, X_test, y_test)

    lgb_model = model_compare._fit_lightgbm(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner)
    scores["LightGBM"] = _ml_model_scores(lgb_model, X_train, y_train, X_test, y_test)

    cat_model = model_compare._fit_catboost(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner)
    scores["CatBoost"] = _ml_model_scores(cat_model, X_train, y_train, X_test, y_test)

    for d in scores.values():
        d["x"] = x_test
        d["y"] = y_coord_test

    # MaxEnt -- NOT refit; reuses the existing train-only eval
    # raster, sampled at the SAME matched test points (never FINAL_RASTER).
    qa.assert_eval_model_excludes_test(train_idx, test_idx)
    test_pts = canonical.iloc[test_idx][["x", "y"]]
    bg_test_pts = background.iloc[bg_test_idx][["x", "y"]]
    y_true_mx, y_score_mx = evaluate.oos_scores(evaluate.EVAL_RASTER, test_pts, bg_test_pts)
    assert len(y_true_mx) == len(y_test), (
        f"[robustness] _matched_test_scores: MaxEnt's oos_scores dropped "
        f"{len(y_test) - len(y_true_mx)} off-raster point(s) that the ML "
        f"comparators' feature extraction did NOT drop -- the coordinate "
        f"arrays shared across models would misalign; investigate before "
        f"trusting Moran's I / McNemar for MaxEnt"
    )
    threshold_mx = evaluate.maxss_threshold_from_results()
    scores["MaxEnt (reference)"] = dict(y_true=y_true_mx, y_score=y_score_mx, threshold=float(threshold_mx),
                                         x=x_test.copy(), y=y_coord_test.copy())

    _MATCHED_SCORES_CACHE[key] = scores
    return scores


def _binarization_thresholds(background_n=model_compare.DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED):
    """THE THRESHOLD FIX, exposed directly: ``{model_name: MaxSS
    threshold}`` for all 6 models on the matched test set -- every
    binarization this module performs (McNemar's own discordant-pair
    counts) reads from exactly this dict, NEVER a fixed 0.5. Regression-
    guarded by ``tests/test_stats.py::test_no_fixed_half_threshold``: none of these six thresholds is
    anywhere near 0.5 in practice (MaxEnt/Logistic Regression sit at
    ~0.04-0.08; the four tree-ensemble comparators at ~0.37-0.80 -- see
    model_compare's metrics table)."""
    scores = _matched_test_scores(background_n=background_n, seed=seed)
    return {model: d["threshold"] for model, d in scores.items()}


# ----------------------------------------------------------------------
# AUPRC + Brier (imbalance-appropriate metrics; ports 04c Sections 15-16).
# ----------------------------------------------------------------------

def imbalance_metrics(scores=None, **kwargs):
    """AUPRC (``average_precision_score`` -- area under the precision-
    recall curve, the imbalance-appropriate companion to AUC-ROC under
    this dataset's ~1:36 presence:background ratio) + Brier score
    (``brier_score_loss`` -- calibration: mean squared error between each
    predicted probability and its {0,1} observed outcome, lower is
    better-calibrated) for all 6 models on the matched test set. Ports
    the original notebook's Sections 15-16.

    Returns a DataFrame (model, auprc, brier, threshold) and writes it to
    ``OUT_DIR/imbalance_metrics.csv``.
    """
    if scores is None:
        scores = _matched_test_scores(**kwargs)
    rows = []
    for model, d in scores.items():
        auprc = float(average_precision_score(d["y_true"], d["y_score"]))
        brier = float(brier_score_loss(d["y_true"], d["y_score"]))
        rows.append(dict(model=model, auprc=auprc, brier=brier, threshold=d["threshold"]))
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "imbalance_metrics.csv", index=False)
    return df


# ----------------------------------------------------------------------
# Bootstrap CI on AUC (ports 04c Section 17).
# ----------------------------------------------------------------------

def bootstrap_auc_ci(y_true, y_score, n_boot=1000, ci=0.95, seed=config.GLOBAL_SEED):
    """95% (default) empirical bootstrap CI on AUC-ROC: resample the test
    set WITH replacement `n_boot` times (default 1000, matching
    the original notebook's Section 17), recompute
    ``roc_auc_score`` each time, skipping any resample that happens to draw
    only one class (an AUC is undefined there -- matches 04c's own
    ``if len(np.unique(y_b[idx])) < 2: continue`` guard).

    Returns a dict: point_auc (the real, non-bootstrapped AUC), mean/sd of
    the bootstrap distribution, ci_lo/ci_hi (percentile CI), n_boot_used
    (may be < `n_boot` if any resample was skipped), n_boot_requested,
    ci_level.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, y_score[idx]))
    aucs = np.asarray(aucs)

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return dict(
        point_auc=float(roc_auc_score(y_true, y_score)),
        mean_boot_auc=float(aucs.mean()),
        sd_boot_auc=float(aucs.std(ddof=1)),
        ci_lo=float(np.percentile(aucs, lo_pct)),
        ci_hi=float(np.percentile(aucs, hi_pct)),
        n_boot_used=int(len(aucs)),
        n_boot_requested=int(n_boot),
        ci_level=float(ci),
    )


def bootstrap_auc_ci_all_models(scores=None, n_boot=1000, ci=0.95, seed=config.GLOBAL_SEED, **kwargs):
    """``bootstrap_auc_ci`` applied to all 6 models on the matched test
    set. Returns a DataFrame (model + every ``bootstrap_auc_ci`` key) and
    writes it to ``OUT_DIR/bootstrap_auc_ci.csv``."""
    if scores is None:
        scores = _matched_test_scores(**kwargs)
    rows = []
    for model, d in scores.items():
        r = bootstrap_auc_ci(d["y_true"], d["y_score"], n_boot=n_boot, ci=ci, seed=seed)
        rows.append(dict(model=model, **r))
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "bootstrap_auc_ci.csv", index=False)
    return df


# ----------------------------------------------------------------------
# McNemar's test -- THE THRESHOLD FIX (ports 04c Section 18b).
# ----------------------------------------------------------------------

def mcnemar_pairs(scores=None, **kwargs):
    """Pairwise McNemar's test (continuity-corrected chi-square on the
    discordant-pair counts b/c of a 2x2 contingency table) between every
    pair of the 6 models on the matched test set. THE THRESHOLD FIX vs.
    the original notebook's Section 18b: each model's binary
    prediction is thresholded at its OWN MaxSS operating point
    (``_binarization_thresholds``), never a fixed 0.5.

    Returns a DataFrame (pair, threshold_a, threshold_b, b, c, chi2,
    p_value, note) and writes it to ``OUT_DIR/mcnemar_pairs.csv``.
    """
    if scores is None:
        scores = _matched_test_scores(**kwargs)
    names = list(scores.keys())

    y_ref = np.asarray(scores[names[0]]["y_true"])
    for name in names[1:]:
        assert np.array_equal(y_ref, np.asarray(scores[name]["y_true"])), (
            f"[robustness] mcnemar_pairs: {name}'s y_true differs from "
            f"{names[0]}'s -- McNemar requires every model scored on the "
            f"IDENTICAL matched test set"
        )

    rows = []
    for a, b_name in itertools.combinations(names, 2):
        sa, sb = scores[a], scores[b_name]
        pred_a = (np.asarray(sa["y_score"]) >= sa["threshold"]).astype(int)   # THE FIX: own threshold
        pred_b = (np.asarray(sb["y_score"]) >= sb["threshold"]).astype(int)   # THE FIX: own threshold
        ok_a = pred_a == y_ref
        ok_b = pred_b == y_ref
        b_count = int(np.sum(ok_a & ~ok_b))
        c_count = int(np.sum(~ok_a & ok_b))

        if b_count + c_count == 0:
            rows.append(dict(pair=f"{a} vs {b_name}", threshold_a=sa["threshold"], threshold_b=sb["threshold"],
                              b=b_count, c=c_count, chi2=float("nan"), p_value=float("nan"),
                              note="identical predictions at own thresholds"))
            continue

        chi_sq = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        p_value = float(_chi2_dist.sf(chi_sq, df=1))
        rows.append(dict(pair=f"{a} vs {b_name}", threshold_a=sa["threshold"], threshold_b=sb["threshold"],
                          b=b_count, c=c_count, chi2=float(chi_sq), p_value=p_value, note=""))

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "mcnemar_pairs.csv", index=False)
    return df


# ----------------------------------------------------------------------
# Moran's I on test-set residuals (ports 04c Section 13).
# ----------------------------------------------------------------------

def morans_i_residuals(scores=None, model="MaxEnt (reference)", k=8, n_perm=999,
                        seed=config.GLOBAL_SEED, **kwargs):
    """Moran's I on MaxEnt's test-set residuals (residual = predicted -
    observed = ``y_score - y_true``) -- ports the original notebook's
    Section 13 verbatim in arithmetic: k=8 nearest-neighbour ROW-
    STANDARDISED spatial weights (each of a point's k nearest neighbours
    gets weight 1/k), a 999-permutation (default) significance test
    (shuffle the residuals, hold the spatial weights fixed, recompute I
    each time), on the COMBINED presence (y=1) + background (y=0) matched
    test points.

    Tests whether MaxEnt's prediction errors are spatially random (good --
    the model has captured the spatial structure in the data) or clustered
    (a significant positive I -- evidence of a missing spatially-structured
    predictor or unmodelled non-stationarity: the model systematically
    over- or under-predicts in particular geographic zones). NOTE: with
    background outnumbering presence ~36:1 in the matched test set, this
    number is heavily influenced by the spatial smoothness of the
    predicted surface itself over the (much larger) background sample, not
    only by presence-point errors -- read the observed value with that in
    mind.

    Returns a dict: model, n, k, observed_I, expected_I, p_value, n_perm,
    verdict ("clustered" if p < 0.05 else "spatially_random"). Also written
    to ``OUT_DIR/morans_i_residuals.csv``.
    """
    if scores is None:
        scores = _matched_test_scores(**kwargs)
    d = scores[model]
    y_true = np.asarray(d["y_true"], dtype="float64")
    y_score = np.asarray(d["y_score"], dtype="float64")
    x = np.asarray(d["x"], dtype="float64")
    y_coord = np.asarray(d["y"], dtype="float64")

    residuals = y_score - y_true   # predicted - observed
    n = len(residuals)

    coords = np.column_stack([x, y_coord])
    _, idx_nn = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)

    w = lil_matrix((n, n), dtype=float)
    for i, nbrs in enumerate(idx_nn[:, 1:]):   # column 0 is each point itself
        for j in nbrs:
            w[i, j] = 1.0 / k
    w = w.tocsr()

    z = residuals - residuals.mean()
    wz = w.dot(z)
    s0 = float(w.sum())
    zz = float(z.dot(z))
    observed_i = float((n / s0) * (z.dot(wz) / zz))
    expected_i = -1.0 / (n - 1)

    rng = np.random.default_rng(seed)
    i_perm = np.empty(n_perm)
    for i in range(n_perm):
        zp = rng.permutation(z)
        i_perm[i] = float((n / s0) * (zp.dot(w.dot(zp)) / zz))
    p_value = float((np.sum(i_perm >= observed_i) + 1) / (n_perm + 1))

    verdict = "clustered" if p_value < 0.05 else "spatially_random"
    result = dict(model=model, n=int(n), k=int(k), observed_I=observed_i, expected_I=expected_i,
                  p_value=p_value, n_perm=int(n_perm), verdict=verdict)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUT_DIR / "morans_i_residuals.csv", index=False)
    print(f"[robustness] Moran's I on {model} test residuals (n={n}): "
          f"observed={observed_i:+.4f} expected={expected_i:+.4f} p={p_value:.4f} -> {verdict}")
    return result


# ============================================================================
# Spatial-CV buffering: how far does dependence actually reach, how close do
# the existing folds let training points sit, and does enforcing a gap matter?
# ============================================================================

CORRELOGRAM_CSV = OUT_DIR / "residual_correlogram.csv"
CV_GEOMETRY_CSV = OUT_DIR / "spatial_cv_geometry.csv"
BUFFERED_CV_CSV = OUT_DIR / "spatial_cv_buffered.csv"
BUFFERED_CV_DIR = OUT_DIR.parent / "maxent" / "spatial_cv_buffered"


def residual_correlogram(*, scores=None, model="MaxEnt (reference)", band_m=250.0,
                         max_dist_m=5000.0, n_perm=299, seed=config.GLOBAL_SEED, **kwargs):
    """Moran's I of the test residuals in distance bands -- the empirical
    autocorrelation range, rather than an assumed one.

    Added 2026-08-15. Leave-one-block-out CV is only honest if the blocks are
    larger than the distance over which the data are dependent; asserting that
    without measuring it is exactly the gap a reviewer should press on. This
    reports where dependence actually dies out.

    ``morans_i_residuals`` uses k=8 nearest neighbours, which answers "are the
    errors clustered at all". Distance-band weights answer the different
    question the CV design needs: *at what separation does the clustering
    stop.* The reported range is the lower edge of the first band whose I is
    not significantly positive.

    Returns a DataFrame of one row per band and writes ``residual_correlogram.csv``.
    """
    if scores is None:
        scores = _matched_test_scores(**kwargs)
    d = scores[model]
    resid = np.asarray(d["y_score"], dtype="float64") - np.asarray(d["y_true"], dtype="float64")
    coords = np.column_stack([np.asarray(d["x"], dtype="float64"),
                              np.asarray(d["y"], dtype="float64")])
    n = len(resid)
    z = resid - resid.mean()
    zz = float(z.dot(z))
    rng = np.random.default_rng(seed)

    # One pairwise distance matrix reused by every band. n is a few thousand at
    # most here, so this is cheaper than rebuilding neighbours band by band.
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)

    edges = np.arange(0.0, max_dist_m + band_m, band_m)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (dist > lo) & (dist <= hi)
        n_pairs = int(mask.sum())
        if n_pairs < 30:
            rows.append(dict(lo_m=lo, hi_m=hi, n_pairs=n_pairs, morans_i=np.nan,
                             p_value=np.nan, significant=False))
            continue
        w = mask.astype("float64")
        rs = w.sum(axis=1, keepdims=True)
        w = np.divide(w, rs, out=np.zeros_like(w), where=rs > 0)   # row-standardised
        s0 = float(w.sum())
        obs = float((n / s0) * (z.dot(w.dot(z)) / zz))
        perm = np.empty(n_perm)
        for i in range(n_perm):
            zp = rng.permutation(z)
            perm[i] = float((n / s0) * (zp.dot(w.dot(zp)) / zz))
        p = float((np.sum(perm >= obs) + 1) / (n_perm + 1))
        rows.append(dict(lo_m=lo, hi_m=hi, n_pairs=n_pairs, morans_i=obs,
                         p_value=p, significant=bool(p < 0.05)))
        print(f"[robustness]   {lo:6.0f}-{hi:6.0f} m  pairs={n_pairs:8d}  I={obs:+.4f}  p={p:.3f}")

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["morans_i"])
    nonsig = scored[~scored["significant"]]
    rng_m = float(nonsig["lo_m"].iloc[0]) if len(nonsig) else float("nan")
    df.attrs["autocorrelation_range_m"] = rng_m
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORRELOGRAM_CSV, index=False)
    print(f"[robustness] autocorrelation range ~ {rng_m:.0f} m "
          f"(first band with no significant positive I); wrote {CORRELOGRAM_CSV.name}")
    return df


def spatial_cv_geometry(samples=None, **kwargs):
    """How close does the nearest training point actually sit to each held-out
    point, fold by fold?

    Leave-one-block-out guarantees a held-out point's neighbours are outside its
    own 2 km cell, but says nothing about a point sitting just inside the block
    edge -- its nearest training point can be metres away. Reporting the minimum
    and median per fold makes the design's real separation visible instead of
    implied.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    holdouts = spatial_cv_membership(samples, **kwargs)
    xy = samples[["x", "y"]].to_numpy(dtype="float64")
    universe = np.arange(len(samples))

    rows = []
    for fid in sorted(holdouts):
        held = np.asarray(holdouts[fid])
        train = np.setdiff1d(universe, held)
        d = np.linalg.norm(xy[held][:, None, :] - xy[train][None, :, :], axis=-1)
        nearest = d.min(axis=1)
        rows.append(dict(fold=fid, n_test=len(held), n_train=len(train),
                         min_train_test_m=float(nearest.min()),
                         median_train_test_m=float(np.median(nearest)),
                         max_train_test_m=float(nearest.max())))
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CV_GEOMETRY_CSV, index=False)
    print(f"[robustness] spatial CV geometry: overall min train-test distance "
          f"{df['min_train_test_m'].min():.0f} m, median of fold medians "
          f"{df['median_train_test_m'].median():.0f} m; wrote {CV_GEOMETRY_CSV.name}")
    return df


def spatial_cv_buffered(buffers_m=(0.0, 500.0, 1000.0, 2000.0), *, samples=None,
                        background_n=evaluate.DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED,
                        out_dir=BUFFERED_CV_DIR, timeout=maxent.DEFAULT_TIMEOUT_S,
                        min_train=40):
    """Leave-one-block-out CV with a dead zone around each held-out block.

    For each buffer, any training point within ``buffer_m`` of a held-out point
    is dropped before fitting. ``buffer_m=0`` reproduces the standard blocked
    estimate and is the control -- it must land on the published mean, or the
    comparison is measuring something else.

    A buffer removes real training data as well as dependent training data, so a
    decline is expected and is not by itself evidence of leakage: the question is
    whether the estimate falls off a cliff (the blocked number was optimistic) or
    erodes gently (blocking was already doing its job). Folds left with fewer
    than ``min_train`` training points are dropped and counted, because an AUC
    from a starved fit is noise, not evidence.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    holdouts = spatial_cv_membership(samples)
    xy = samples[["x", "y"]].to_numpy(dtype="float64")
    universe = np.arange(len(samples))
    layers = sorted(maxent.modeling_layers())
    background = evaluate.sample_background(_background_reference_raster(),
                                            n=background_n, seed=seed)
    out_dir = Path(out_dir)

    rows = []
    for buf in buffers_m:
        fold_aucs, dropped, excluded_total = [], 0, 0
        for fid in sorted(holdouts):
            held = np.asarray(holdouts[fid])
            train = np.setdiff1d(universe, held)
            if buf > 0:
                d = np.linalg.norm(xy[train][:, None, :] - xy[held][None, :, :], axis=-1)
                keep = d.min(axis=1) > buf
                excluded_total += int((~keep).sum())
                train = train[keep]
            if len(train) < min_train:
                dropped += 1
                continue
            raster = _fit_train_only_on_layers(samples, train, held, layers,
                                               out_dir / f"buf{int(buf)}" / f"fold{fid}",
                                               timeout=timeout)
            fold_aucs.append(evaluate.oos_auc(raster, samples.iloc[held][["x", "y"]],
                                              train, held, background))
        aucs = np.asarray(fold_aucs, dtype="float64")
        rows.append(dict(buffer_m=buf, n_folds_scored=len(aucs), n_folds_dropped=dropped,
                         mean_train_points_excluded=excluded_total / max(len(holdouts), 1),
                         mean_auc=float(aucs.mean()) if len(aucs) else np.nan,
                         sd_auc=float(aucs.std(ddof=1)) if len(aucs) > 1 else np.nan))
        print(f"[robustness] buffer {buf:6.0f} m: mean AUC={rows[-1]['mean_auc']:.4f} "
              f"({len(aucs)} folds scored, {dropped} dropped)")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BUFFERED_CV_CSV, index=False)
    print(f"[robustness] wrote {BUFFERED_CV_CSV.name}")
    return df


# ----------------------------------------------------------------------
# THE SIGNIFICANCE FIX -- Nadeau-Bengio corrected resampled t-test,
# replacing 04c Section 18a's Wilcoxon-at-its-own-p-floor test.
# ----------------------------------------------------------------------

def nadeau_bengio_corrected_t(diffs, n_train, n_test):
    """The Nadeau-Bengio (2003) corrected resampled t-test: like a paired
    t-test on `diffs` (model A's per-fold metric minus model B's, one per
    CV fold), but replaces the naive ``1/k`` variance term with
    ``1/k + n_test/n_train`` -- the correction that accounts for CV folds'
    training sets overlapping (they are NOT independent samples the way a
    naive paired t-test assumes), which otherwise understates the true
    variance of the mean difference and inflates significance. THE
    SIGNIFICANCE FIX this module exists to make: replaces
    the original notebook's Section 18a Wilcoxon signed-rank test,
    which (at n=8 folds, all-same-direction differences) hits its own
    p-value RESOLUTION FLOOR (p=0.0078, the smallest attainable two-sided
    Wilcoxon p at n=8) regardless of effect size -- a "significant" claim
    manufactured by too few folds, not evidence of a well-quantified gap.

    `n_train`/`n_test` are the (mean, if folds vary in size -- as LOEO's
    11 event-folds do, from n_test=2 to n_test=90) per-fold train/test
    PRESENCE counts of the shared fold structure both models were scored
    on (never each model's own internal train-row count, which for a
    presence-background classifier also includes thousands of background
    rows MaxEnt's own protocol never uses at fit time at all -- the
    correction is about fold-overlap structure, which is defined by
    presence-point membership here, identically for both models).

    Returns a dict: method="nadeau_bengio_corrected_t", mean_diff, sd_diff,
    t_stat, df, k, n_train, n_test, p_value (two-sided, Student's t with
    `k-1` degrees of freedom).
    """
    diffs = np.asarray(diffs, dtype="float64")
    k = len(diffs)
    assert k >= 2, f"[robustness] nadeau_bengio_corrected_t: need >=2 paired folds, got {k}"

    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    correction = 1.0 / k + float(n_test) / float(n_train)
    se = float(np.sqrt(correction) * sd_d)

    if se > 0:
        t_stat = mean_d / se
    else:
        t_stat = 0.0 if mean_d == 0 else (float("inf") if mean_d > 0 else float("-inf"))
    df = k - 1
    p_value = float(2 * (1 - _t_dist.cdf(abs(t_stat), df))) if np.isfinite(t_stat) else 0.0

    return dict(method="nadeau_bengio_corrected_t", mean_diff=mean_d, sd_diff=sd_d,
                t_stat=float(t_stat), df=int(df), k=int(k),
                n_train=float(n_train), n_test=float(n_test), p_value=p_value)


def _loeo_xgboost_aucs(samples=None, background_n=model_compare.DEFAULT_BACKGROUND_N,
                        seed=config.GLOBAL_SEED, bg_test_multiplier=50):
    """XGBoost's own per-event AUC on the SAME 11-event leave-one-event-out
    fold structure ``loeo()``/``loeo_membership()`` use for MaxEnt -- built
    so ``loeo_significance_test()`` has a genuine SECOND model's per-fold
    array to pair against MaxEnt's, on the IDENTICAL folds (never a
    different partition). Ports the original notebook's Section
    11b `loeo_sklearn` (XGBoost arm): one-shot fit (no early stopping --
    unlike ``model_compare._fit_xgboost``, which is deliberately
    reserved for the ONE matched-test-set comparison; a fresh, simpler
    inner-validation-free recipe here avoids needing an inner split PER
    LOEO FOLD too), `scale_pos_weight` for class imbalance, background test
    points drawn fresh per event (`n_test_presence * bg_test_multiplier`,
    capped at the background pool size -- 04c's own sizing rule, since
    background carries no "event" label to hold out by).

    Reuses ``model_compare.build_background``/``_presence_features``/
    ``CATEGORICAL`` (the canonical 17-layer feature stack), never the old
    21-predictor stack 04c's own source used.

    CAVEAT (ported from 04c's own Section 18 text, still true here): MaxEnt
    and this XGBoost arm score against slightly DIFFERENT per-event
    background samples (MaxEnt's own ``loeo()`` reuses ``evaluate.
    sample_background``'s un-filtered draw; this function needs a
    presence-excluded, LABELED background for a discriminative fit) -- so
    ``loeo_significance_test()`` below is an approximate, not perfectly
    matched, paired comparison.

    Returns a DataFrame: event, n_test, auc -- same columns as ``loeo()``'s
    own `per_event`, so the two can be merged directly on `event`.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    layers = maxent.modeling_layers()

    pres_feats = model_compare._presence_features(samples, layers=layers)
    background = model_compare.build_background(n=background_n, seed=seed, layers=layers, presence=samples)
    bg_feats = background[list(layers)].copy()
    for c in model_compare.CATEGORICAL:
        bg_feats[c] = bg_feats[c].astype(int)

    holdouts = loeo_membership(samples)
    rng = np.random.default_rng(seed)
    universe_bg = np.arange(len(background))

    rows = []
    for event in sorted(holdouts):
        held_out_idx = holdouts[event]
        train_idx = np.setdiff1d(np.arange(len(samples)), held_out_idx)

        n_bg_test = min(len(held_out_idx) * bg_test_multiplier, len(background))
        bg_test_idx = rng.choice(universe_bg, size=n_bg_test, replace=False)
        bg_train_idx = np.setdiff1d(universe_bg, bg_test_idx)

        X_tr = pd.concat([pres_feats.iloc[train_idx].reset_index(drop=True),
                          bg_feats.iloc[bg_train_idx].reset_index(drop=True)], ignore_index=True)
        y_tr = np.concatenate([np.ones(len(train_idx)), np.zeros(len(bg_train_idx))])
        X_te = pd.concat([pres_feats.iloc[held_out_idx].reset_index(drop=True),
                          bg_feats.iloc[bg_test_idx].reset_index(drop=True)], ignore_index=True)
        y_te = np.concatenate([np.ones(len(held_out_idx)), np.zeros(len(bg_test_idx))])

        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos, random_state=seed,
            n_jobs=-1, verbosity=0, eval_metric="logloss",
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, prob))
        rows.append(dict(event=event, n_test=int(len(held_out_idx)), auc=auc))
        print(f"[robustness] loeo XGBoost {event}: n_test={len(held_out_idx)} AUC={auc:.4f}")

    return pd.DataFrame(rows)


LOEO_ALL_MODELS_CSV = OUT_DIR / "loeo_all_models.csv"
OMNIBUS_CSV = OUT_DIR / "omnibus_model_comparison.csv"
OMNIBUS_PAIRS_CSV = OUT_DIR / "omnibus_pairwise_nb_holm.csv"


def loeo_all_models(samples=None, background_n=model_compare.DEFAULT_BACKGROUND_N,
                    seed=config.GLOBAL_SEED, bg_test_multiplier=50):
    """Per-event LOEO AUC for all five ML comparators on the identical folds.

    Generalises ``_loeo_xgboost_aucs``, which
    built this for XGBoost alone so the pairwise significance test had a second
    model; an omnibus comparison needs every model on the same folds.

    MaxEnt is NOT refit here -- ``loeo()`` already produces its per-event AUCs
    from the MaxEnt rasters, and ``omnibus_model_comparison()`` merges them in.

    Same caveat as ``_loeo_xgboost_aucs``: MaxEnt scores against ``loeo()``'s own
    background draw while these five use a presence-excluded labelled background,
    so the MaxEnt column is an approximate rather than perfectly matched pairing.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)
    layers = maxent.modeling_layers()
    pres_feats = model_compare._presence_features(samples, layers=layers)
    background = model_compare.build_background(n=background_n, seed=seed,
                                                layers=layers, presence=samples)
    bg_feats = background[list(layers)].copy()
    for c in model_compare.CATEGORICAL:
        bg_feats[c] = bg_feats[c].astype(int)

    holdouts = loeo_membership(samples)
    rng = np.random.default_rng(seed)
    universe_bg = np.arange(len(background))
    rows = []
    for event in sorted(holdouts):
        held = holdouts[event]
        train_idx = np.setdiff1d(np.arange(len(samples)), held)
        n_bg_test = min(len(held) * bg_test_multiplier, len(background))
        bg_test_idx = rng.choice(universe_bg, size=n_bg_test, replace=False)
        bg_train_idx = np.setdiff1d(universe_bg, bg_test_idx)
        X_tr = pd.concat([pres_feats.iloc[train_idx].reset_index(drop=True),
                          bg_feats.iloc[bg_train_idx].reset_index(drop=True)],
                         ignore_index=True)
        y_tr = np.concatenate([np.ones(len(train_idx)), np.zeros(len(bg_train_idx))])
        X_te = pd.concat([pres_feats.iloc[held].reset_index(drop=True),
                          bg_feats.iloc[bg_test_idx].reset_index(drop=True)],
                         ignore_index=True)
        y_te = np.concatenate([np.ones(len(held)), np.zeros(len(bg_test_idx))])
        X_tr_i, X_val_i, y_tr_i, y_val_i = model_compare._inner_split(X_tr, y_tr, seed=seed)

        fitted = {
            "Logistic Regression": model_compare._fit_logistic_regression(X_tr, y_tr, layers),
            "Random Forest": model_compare._fit_random_forest(X_tr, y_tr),
            "XGBoost": model_compare._fit_xgboost(X_tr_i, y_tr_i, X_val_i, y_val_i),
            "LightGBM": model_compare._fit_lightgbm(X_tr_i, y_tr_i, X_val_i, y_val_i),
            "CatBoost": model_compare._fit_catboost(X_tr_i, y_tr_i, X_val_i, y_val_i),
        }
        for name, m in fitted.items():
            auc = float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
            rows.append(dict(event=event, n_test=int(len(held)), model=name, auc=auc))
        print(f"[robustness] loeo_all_models {event}: n_test={len(held)} "
              f"({len(fitted)} models)", flush=True)

    df = pd.DataFrame(rows)
    LOEO_ALL_MODELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOEO_ALL_MODELS_CSV, index=False)
    print(f"[robustness] wrote {LOEO_ALL_MODELS_CSV.name}")
    return df


def omnibus_model_comparison(samples=None, alpha=0.05, **kwargs):
    """Are the six models distinguishable? Two answers, and they differ.

    A single pairwise test (MaxEnt vs XGBoost) cannot show that the six
    models are "statistically indistinguishable": one non-significant
    comparison cannot license a blanket claim about six models. This adds
    the omnibus tests.

    Reports both:

    1. **Friedman** -- the conventional omnibus for k models over n blocks. It
       assumes INDEPENDENT blocks, which LOEO folds are not: consecutive folds
       share roughly 90 percent of their training data. It is therefore
       anti-conservative here -- precisely the reason
       ``nadeau_bengio_corrected_t`` replaced a bare Wilcoxon for the pairwise
       test. Reported for completeness, flagged, never used alone.

    2. **All 15 pairwise Nadeau-Bengio corrected t-tests, Holm-adjusted** -- the
       principled answer, consistent with the correction the paper already
       justifies for overlapping training sets.

    Returns ``(omnibus_dict, pairwise_df)``.
    """
    from itertools import combinations
    from scipy.stats import friedmanchisquare

    ml = loeo_all_models(samples=samples, **kwargs)
    me_res = loeo(samples=samples)
    me = me_res["per_event"][["event", "n_test", "auc"]].copy()
    me["model"] = "MaxEnt"
    long = pd.concat([ml, me[["event", "n_test", "model", "auc"]]], ignore_index=True)

    wide = long.pivot(index="event", columns="model", values="auc").dropna()
    models = sorted(wide.columns)
    n_test = long.groupby("event")["n_test"].first().reindex(wide.index)
    n_test_mean = float(n_test.mean())
    n_train_mean = float(len(config.canonical_samples()) - n_test_mean)

    stat, p = friedmanchisquare(*[wide[m].to_numpy() for m in models])
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    omnibus = dict(
        test="friedman", k_models=len(models), n_folds=int(len(wide)),
        chi2=float(stat), p_value=float(p),
        caveat=("LOEO folds share most of their training data, so Friedman's "
                "independence assumption fails and this p-value is "
                "anti-conservative; prefer the pairwise NB+Holm result"),
        **{f"meanrank_{m}": float(ranks[m]) for m in models})

    rows = []
    for a, b in combinations(models, 2):
        d = (wide[a] - wide[b]).to_numpy()
        nb = nadeau_bengio_corrected_t(d, n_train=n_train_mean, n_test=n_test_mean)
        rows.append(dict(model_a=a, model_b=b, mean_diff=nb["mean_diff"],
                         t_stat=nb["t_stat"], p_raw=nb["p_value"]))
    pw = pd.DataFrame(rows)

    # Holm step-down across the 15 pairs
    raw = pw["p_raw"].to_numpy()
    order = np.argsort(raw)
    m_tests = len(raw)
    adj = np.empty(m_tests)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m_tests - rank) * raw[i])
        adj[i] = min(1.0, running)
    pw["p_holm"] = adj
    pw["significant_holm"] = pw["p_holm"] < alpha
    pw = pw.sort_values("p_raw").reset_index(drop=True)

    pd.DataFrame([omnibus]).to_csv(OMNIBUS_CSV, index=False)
    pw.to_csv(OMNIBUS_PAIRS_CSV, index=False)
    print(f"[robustness] Friedman chi2={stat:.3f} p={p:.4g} over {len(wide)} folds "
          f"-- ANTI-CONSERVATIVE, folds are not independent")
    print(f"[robustness] pairwise NB+Holm: {int(pw['significant_holm'].sum())} of "
          f"{m_tests} pairs significant at alpha={alpha}")
    print(f"[robustness] wrote {OMNIBUS_CSV.name} and {OMNIBUS_PAIRS_CSV.name}")
    return omnibus, pw


def loeo_significance_test(samples=None, second_model="XGBoost",
                            background_n=model_compare.DEFAULT_BACKGROUND_N,
                            seed=config.GLOBAL_SEED):
    """THE SIGNIFICANCE FIX: is MaxEnt's AUC really different from the
    strongest ML comparator's (XGBoost, the AUC leader in model_compare), or is the
    gap within CV noise? Pairs MaxEnt's own per-event LOEO AUCs (``loeo()``,
    cached) against XGBoost's per-event LOEO AUCs
    (``_loeo_xgboost_aucs``, new -- the SAME 11-event fold structure) and
    runs the Nadeau-Bengio corrected resampled t-test on the 11 paired
    differences -- REPLACES the original notebook's Section 18a
    Wilcoxon-at-its-own-p-floor test (see ``nadeau_bengio_corrected_t``'s
    own docstring).

    Also reports a plain effect size (Cohen's d on the paired differences)
    and an (uncorrected, naive) t-based CI on the mean difference alongside
    the NB test, for transparency -- never as a SUBSTITUTE significance
    claim, and never a Wilcoxon call anywhere in this function.

    Returns a dict (method="nadeau_bengio_corrected_t", the per-event
    arrays, mean_diff, cohens_d, naive_ci_lo/hi, t_stat, df, p_value,
    verdict="real_difference"/"within_noise", ...) and writes a one-row
    summary to ``OUT_DIR/loeo_significance_test.csv`` plus a per-event
    breakdown to ``OUT_DIR/loeo_significance_per_event.csv``.
    """
    if samples is None:
        samples = config.canonical_samples()
    samples = samples.reset_index(drop=True)

    maxent_df = loeo(samples)["per_event"]
    second_df = _loeo_xgboost_aucs(samples, background_n=background_n, seed=seed)

    key = f"auc_{second_model.lower()}"
    n_test_key = f"n_test_{second_model.lower()}"
    merged = maxent_df.merge(second_df, on="event", suffixes=("_maxent", f"_{second_model.lower()}"))
    assert len(merged) == len(maxent_df) == len(second_df), (
        f"[robustness] loeo_significance_test: event sets differ between "
        f"MaxEnt ({len(maxent_df)} events) and {second_model} "
        f"({len(second_df)} events) -- both must use the IDENTICAL "
        f"loeo_membership() fold structure"
    )
    assert (merged["n_test_maxent"] == merged[n_test_key]).all(), (
        f"[robustness] loeo_significance_test: MaxEnt and {second_model} "
        f"disagree on n_test for at least one event -- they must be scored "
        f"on the IDENTICAL held-out presence points per fold"
    )

    diffs = (merged["auc_maxent"] - merged[key]).to_numpy()
    n_test_mean = float(merged["n_test_maxent"].mean())
    n_train_mean = float(len(samples) - n_test_mean)

    nb = nadeau_bengio_corrected_t(diffs, n_train=n_train_mean, n_test=n_test_mean)

    k = len(diffs)
    sd_d = float(diffs.std(ddof=1))
    cohens_d = float(diffs.mean() / sd_d) if sd_d > 0 else float("nan")
    se_naive = sd_d / np.sqrt(k)
    t_crit = float(_t_dist.ppf(0.975, k - 1))
    naive_ci_lo = float(diffs.mean() - t_crit * se_naive)
    naive_ci_hi = float(diffs.mean() + t_crit * se_naive)
    verdict = "real_difference" if nb["p_value"] < 0.05 else "within_noise"

    result = dict(
        second_model=second_model,
        events=merged["event"].tolist(),
        maxent_aucs=merged["auc_maxent"].tolist(),
        second_model_aucs=merged[key].tolist(),
        cohens_d=cohens_d,
        naive_ci_lo=naive_ci_lo, naive_ci_hi=naive_ci_hi,
        verdict=verdict,
        **nb,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_row = {k_: v for k_, v in result.items()
                   if k_ not in ("events", "maxent_aucs", "second_model_aucs")}
    pd.DataFrame([summary_row]).to_csv(OUT_DIR / "loeo_significance_test.csv", index=False)
    pd.DataFrame({
        "event": merged["event"],
        "n_test": merged["n_test_maxent"],
        "auc_maxent": merged["auc_maxent"],
        key: merged[key],
        "diff": diffs,
    }).to_csv(OUT_DIR / "loeo_significance_per_event.csv", index=False)

    print(f"[robustness] LOEO significance (MaxEnt vs {second_model}, Nadeau-Bengio corrected t, "
          f"k={nb['k']} folds): mean_diff={nb['mean_diff']:+.4f} cohens_d={cohens_d:+.3f} "
          f"t={nb['t_stat']:.3f} df={nb['df']} p={nb['p_value']:.4f} -> {verdict}")
    return result

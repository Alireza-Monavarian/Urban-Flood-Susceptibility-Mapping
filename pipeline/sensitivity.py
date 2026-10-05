"""Matched-baseline sensitivity arms + an honest
ascertainment-bias bound.

Ports the four arms of the original notebook ``06_sensitivity`` (background sampling x2,
spatial resolution, event pooling -- its "log transform" arm is now MOOT,
see below) onto the canonical pipeline's locked config (``maxent.
FINAL_CONFIG`` = beta=0.5/LQ, ``maxent.modeling_layers()`` = the 17-layer
log-scaled stack, ``config.canonical_samples()`` = 277 points), adds a NEW
required arm (output format), and fixes the source's central bug.

**THE CRITICAL FIX vs. source.** The source notebook differenced every
arm's AUC against a single hardcoded ``BASELINE_AUC = 0.9842`` -- a
DIFFERENT model/config's number (beta=1/L, 21 raw predictors, resubstitution-
flavored: the source's own "test" split was scored against a model that had
already trained on those same 83 points -- exactly the resubstitution bug
``evaluate.oos_auc`` exists to fix; see that module's docstring). Comparing
a target-group-background arm's AUC (a genuinely different evaluation
protocol -- different background, sometimes a different, filtered test-
presence subset) against that ONE frozen number is an apples-to-oranges
"mismatched protocol" bug, not a sensitivity analysis. Every arm below
instead carries its own ``eval_id`` (a content hash of the exact test-
presence + background point sets actually scored) and its own
``baseline_auc`` / ``delta_auc``, RECOMPUTED under the arm's own protocol:
  - Arms that do not change the evaluation universe (resolution, output
    format) share the OVERALL baseline's ``eval_id`` verbatim (literally the
    same ``test_pts``/``background`` objects) -- their ``baseline_auc`` IS
    the overall baseline's ``auc``.
  - The two background-sampling arms (target-group NLCD-developed,
    stormwater-buffer) retrain the model (masked layers restrict where
    MaxEnt samples ITS OWN training background from) but compare against a
    baseline that reuses the STANDARD (unmasked) train-only model SCORED ON
    THE SAME (developed-only / buffer-only) test background + filtered test
    presence -- like-vs-like.
  - Event pooling changes the presence universe outright (top-N events
    only), so it gets its OWN from-scratch baseline: an all-events model
    trained on every canonical point EXCEPT this arm's own held-out top-N
    test points, scored on those SAME held-out points.
  - The NEW outfall-distance-matched arm retrains NOTHING -- it reuses the
    exact standard baseline model and asks a different background-SAMPLING
    question (see below); its ``baseline_auc`` is the overall baseline's
    ``auc`` (same model, random vs. distance-matched background).

Every arm dict ALSO carries a ``baseline_eval_id``: the ``eval_id`` of the
EXACT (test, background) pair its own ``baseline_auc`` was scored against
(never just a label -- a content hash, like ``eval_id`` itself). For
``resolution_30m``/``output_format_logistic`` and ``bg_outfall_distance_
matched``, ``baseline_auc`` is the overall baseline's number verbatim, so
``baseline_eval_id`` is the overall baseline's ``eval_id`` -- for the first
two this happens to EQUAL the arm's own ``eval_id`` (same protocol
throughout); for outfall-matched it deliberately does NOT (same model,
different background -- see that arm's own docstring). For the developed/
stormwater/event-pooling arms, ``baseline_auc`` is a fresh recompute on the
SAME points as the arm's own ``auc``, so ``baseline_eval_id`` there always
EQUALS the arm's own ``eval_id`` -- the like-vs-like guarantee, now directly
machine-checkable per arm rather than inferred from reading the source.

**The ascertainment-bias story (background-sampling arms).** HWM presence
points are surveyed near infrastructure, not drawn at random from the AOI --
a random background makes any model that scores urban/infrastructure-
adjacent pixels highly look artificially good. Three of the six arms below
target this directly, forming a bound sandwich:
  - ``bg_target_group_developed`` -- background restricted to NLCD
    developed pixels (21-24), matching where the city plausibly surveys.
  - ``bg_target_group_stormwater_buffer`` -- background restricted to a
    200 m buffer of the stormwater pipe/channel network --
    **explicitly labeled a LOWER bound** (``lower_bound=True``,
    ``bound="lower_bound"``): ``distance_to_outfall`` is distance to
    discrete outfall POINTS, but this arm buffers stormwater LINES, so
    within the buffer the outfall-proximity axis still separates presence
    from background -- it reduces, but does not remove, the ascertainment
    bias.
  - ``bg_outfall_distance_matched`` (NEW, the proper bound) -- background
    STRATIFIED-RESAMPLED (quantile bins of ``distance_to_outfall``,
    proportional allocation) so its distance-to-outfall distribution
    matches the 277 presence points' own. This isolates whatever
    discrimination is NOT attributable to outfall proximity: if AUC
    collapses toward 0.5, the standard model's skill was largely
    ascertainment-driven; if it stays well above 0.5, real signal survives.
    The observed numbers are reported as they come out -- never forced
    toward either outcome here.

**Log-transform arm -- MOOT, not ported.** The source's 3rd arm applied
log1p to ``flow_accumulation``/``spi`` as a variation around a raw baseline.
In THIS canonical pipeline, log-scaling those two layers is already the BASE
decision (``predictors.apply_modeling_transform``, the log-transform
adoption) -- and both layers are additionally SCREENED OUT of the 17-layer
modeling stack entirely (Stage 1 Pearson / Stage 2 VIF -- see
``pipeline.maxent``'s own "Canonical modeling stack" docstring section).
There is no raw-vs-log axis left to test around this base; running one would
silently reintroduce a layer the screen already removed. Not implemented,
by design -- see the module-level ``LOG_TRANSFORM_ARM_NOTE`` string for the
one-line version of this same point, surfaced in the report.

**Why ``_score``/``_open_scoring_raster`` exist instead of reusing
``evaluate.oos_auc`` directly.** Several arms here score a raw MaxEnt
``flood.asc`` output (30 m resolution, target-group-masked layers) rather
than a converted GeoTIFF. ``evaluate._open_raster`` opens a path WITHOUT
``masked=True`` -- safe for this pipeline's own GeoTIFFs (already NaN-valued
on disk, no encoding trick needed) but WRONG for a raw ``.asc``, whose
``-9999`` NODATA sentinel is a literal small numeric value until decoded.
``_open_scoring_raster`` below always opens with ``masked=True`` (correct
for ``.asc`` AND ``.tif`` alike) and assigns this project's modeling CRS if
absent (``.asc`` carries no CRS; ``.tif`` already has one, so this is a
no-op there) -- then reuses ``evaluate._presence_background_scores`` (the
SAME NaN-drop + presence/background assembly the rest of the pipeline uses)
so the actual scoring arithmetic is never duplicated, only the raster-open
step differs, for the documented reason above.

**Reused verbatim, never re-derived:** ``maxent.run``/``maxent.
build_samples_csv``/``maxent.export_layers``/``maxent.modeling_layers``/
``maxent.FINAL_CONFIG``/``maxent._ascii_header`` (the .asc header format),
``config.canonical_samples``/``config.split``, ``evaluate.
sample_background``/``evaluate._presence_background_scores``, ``qa.
assert_eval_model_excludes_test`` (asserted before every fit AND every
score, per this codebase's Invariant 2 discipline), and ``robustness.
_baseline_eval_raster``/``robustness._background_reference_raster`` (so the
overall baseline reuses the SAME train-only raster and background footprint
the evaluation model, the out-of-sample AUC and spatial CV already use -- this module's own baseline AUC is
therefore expected to reproduce the pinned 0.982577108433735 bit-for-bit,
never a fresh, potentially-diverging re-derivation of "the" baseline).

**Outputs (build artifacts under ``data/processed/maxent/sensitivity/`` --
covered by the existing bare ``maxent/`` .gitignore rule at ANY depth, so
nothing this module writes needs a new .gitignore entry):** per-arm
``.asc``/``.lambdas``/``maxentResults.csv``/``train_idx.csv`` fit
directories, plus ``sensitivity_summary.csv`` (arm/description/auc/
baseline_auc/delta_auc/eval_id).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
from rasterio.enums import Resampling as RS
from rasterio.features import rasterize
from rasterio.transform import rowcol as _rowcol, xy as _xy
from sklearn.metrics import roc_auc_score

from pipeline import config, evaluate, maxent, predictors, qa, robustness, screen

MODELING_CRS = "EPSG:26914"

SENS_DIR = maxent.OUT_DIR / "sensitivity"
SUMMARY_CSV = SENS_DIR / "sensitivity_summary.csv"

N_BACKGROUND = evaluate.DEFAULT_BACKGROUND_N          # 10,000 -- "random 10k valid pixels"
TOP_N_EVENTS = 4
SW_BUFFER_M = 200                                     # stormwater-network buffer
N_OUTFALL_BINS = 10
NLCD_DEVELOPED = (21, 22, 23, 24)
RESOLUTION_M = 30

LOG_TRANSFORM_ARM_NOTE = (
    "MOOT, not implemented: log-scaling flow_accumulation/spi is already the BASE "
    "decision in this canonical pipeline (predictors.apply_modeling_transform), and "
    "both layers are screened OUT of the 17-layer modeling stack entirely (Stage 1 "
    "Pearson / Stage 2 VIF) -- there is no raw-vs-log axis left to test around this "
    "base without silently reintroducing a layer the screen already removed."
)


# ============================================================================
# Shared scoring primitives.
# ============================================================================

def _open_scoring_raster(raster):
    """Open ``raster`` (a path to a ``.asc``/``.tif``, or an already-opened
    DataArray) with ``masked=True`` -- correct NaN-decoding for a raw MaxEnt
    ``.asc`` NODATA sentinel (unlike ``evaluate._open_raster``, which omits
    ``masked=True`` -- safe only for this pipeline's own NaN-valued
    GeoTIFFs, not for a raw ``.asc``). Assigns ``MODELING_CRS`` if the
    opened array has none (``.asc`` carries no CRS; a ``.tif`` already does,
    so this is a no-op there).
    """
    if isinstance(raster, (str, Path)):
        da = rioxarray.open_rasterio(raster, masked=True).squeeze("band", drop=True)
        if da.rio.crs is None:
            da = da.rio.write_crs(MODELING_CRS)
        return da
    return raster


def _score(raster, points, background, train_idx=None, test_idx=None):
    """presence-vs-background AUC (``evaluate.oos_auc``-style: leakage
    assert FIRST, then score) -- reuses ``evaluate._presence_background_
    scores`` for the actual NaN-drop/assembly (never a second, independently
    -written copy of that logic), only the raster-open step
    (``_open_scoring_raster``) differs from ``evaluate.oos_auc`` itself, for
    the reason this module's docstring documents.

    ``train_idx``/``test_idx`` are optional (event-pooling's baseline4 arm
    and the background-sampling arms all have a well-defined pair; pass
    both or neither -- never just one). Returns ``(auc, n_presence,
    n_background)``.
    """
    if train_idx is not None and test_idx is not None:
        qa.assert_eval_model_excludes_test(train_idx, test_idx)
    da = _open_scoring_raster(raster)
    pres_scores, bg_scores = evaluate._presence_background_scores(da, points, background)
    y_true = np.concatenate([np.ones(len(pres_scores)), np.zeros(len(bg_scores))])
    y_score = np.concatenate([pres_scores, bg_scores])
    return float(roc_auc_score(y_true, y_score)), int(len(pres_scores)), int(len(bg_scores))


def _points_fingerprint(xs, ys):
    """Order-independent content hash of a point set -- ``eval_id``'s
    building block. Rounds to mm precision (plenty for a 10 m grid) so
    float round-trip noise can never spuriously change an identity that
    should hold.
    """
    arr = np.round(np.column_stack([np.asarray(xs, dtype="float64"),
                                     np.asarray(ys, dtype="float64")]), 3)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    import hashlib
    return hashlib.sha256(arr[order].tobytes()).hexdigest()[:12]


def _make_eval_id(test_tag, test_pts, bg_tag, background):
    """A single string identifying the EXACT (test-presence-set,
    background-set) pair an AUC was computed against -- two dicts share an
    ``eval_id`` if and only if they were scored on the identical point sets
    (content-hashed, not just labeled), which is what makes a ΔAUC
    apples-to-apples. See this module's docstring, "THE CRITICAL FIX".
    """
    return (f"test={test_tag}:{len(test_pts)}:{_points_fingerprint(test_pts['x'], test_pts['y'])}"
            f"|bg={bg_tag}:{len(background)}:{_points_fingerprint(background['x'], background['y'])}")


def _filter_points_by_mask(points, mask, transform):
    """The subset of ``points`` (x/y columns) whose pixel (under
    ``transform``) is True in the boolean array ``mask`` -- out-of-bounds
    points are dropped, mirroring the source notebook's own "test presences
    inside buffer (evaluated)" accounting. Returns a NEW DataFrame, reset to
    a clean RangeIndex.
    """
    rows, cols = _rowcol(transform, points["x"].to_numpy(), points["y"].to_numpy())
    rows, cols = np.asarray(rows), np.asarray(cols)
    H, W = mask.shape
    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    keep = np.zeros(len(points), dtype=bool)
    keep[in_bounds] = mask[rows[in_bounds], cols[in_bounds]]
    return points.iloc[keep].reset_index(drop=True)


def _read_n_training(out_dir):
    """MaxEnt's own reported ``#Training samples`` for a single (non-
    bootstrap) fit -- how many of the points handed to it actually survived
    its internal NODATA/same-cell drop (relevant for the masked-layer
    background arms, where a training point outside the target zone is
    silently dropped by MaxEnt itself, exactly as in the source notebook).
    """
    df = pd.read_csv(Path(out_dir) / "maxentResults.csv")
    row = df[df["Species"].astype(str) == "flood"]
    row = row.iloc[0] if not row.empty else df.iloc[0]
    return int(row["#Training samples"])


# ============================================================================
# Idempotent single-fit helper -- shared by every arm that retrains.
# ============================================================================

def _fit_train_only(train_idx, held_out_idx, out_dir, *, layers_dir, samples,
                     outputformat="cloglog", timeout=maxent.DEFAULT_TIMEOUT_S):
    """A single deterministic MaxEnt fit (``replicates=0``, ``FINAL_CONFIG``)
    on ``samples.iloc[train_idx]`` against ``layers_dir``'s environmental
    layers, writing ``<out_dir>/flood.asc`` directly (no GeoTIFF conversion
    -- this module's own ``_score``/``_open_scoring_raster`` score the
    ``.asc`` directly). A generalized ``maxent._fit_train_only`` (which
    hardcodes ``LAYERS_DIR``/``cloglog``) -- mirrors how ``robustness.
    _fit_train_only_on_layers`` already generalizes the SAME function for
    the ablation module's own reduced-layer-subset needs; the same
    "write a small parameterized variant rather than modify the shared one"
    precedent applies here for a variable ``layers_dir``/``outputformat``.

    **Invariant 2, enforced BEFORE anything else (even the cache-hit
    check)**: ``qa.assert_eval_model_excludes_test(train_idx, held_out_idx)``.

    Idempotent: if ``<out_dir>/flood.asc`` exists AND its companion
    ``train_idx.csv`` matches ``train_idx`` exactly, the cached ``.asc`` is
    returned without invoking MaxEnt again (this module's "cache-aware, skip
    an arm whose output exists" requirement) -- a MISMATCHED or missing
    ``train_idx.csv`` triggers a refit, never a silently-stale reuse
    (mirrors ``robustness._fit_folds_cached``'s own provenance guard).

    Returns the ``flood.asc`` Path.
    """
    qa.assert_eval_model_excludes_test(train_idx, held_out_idx)

    out_dir = Path(out_dir)
    train_idx = np.asarray(train_idx)
    asc_path = out_dir / "flood.asc"
    idx_path = out_dir / "train_idx.csv"

    if asc_path.exists() and idx_path.exists():
        recorded = np.loadtxt(idx_path, skiprows=1, dtype=int).reshape(-1)
        if sorted(recorded.tolist()) == sorted(int(i) for i in train_idx):
            print(f"[sensitivity] {out_dir}: cached ({asc_path})")
            return asc_path
        print(f"[sensitivity] WARNING: {idx_path} does not match the intended train "
              f"set -- refitting from scratch (stale cache)")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_points = samples.iloc[train_idx]
    samples_csv = maxent.build_samples_csv(out_path=out_dir / "train_samples.csv", presence=train_points)

    result = maxent.run(samples_csv, layers_dir, out_dir,
                         features=maxent.FINAL_CONFIG["features"], beta=maxent.FINAL_CONFIG["beta"],
                         replicates=0, outputformat=outputformat, timeout=timeout)
    if result["rc"] != 0:
        raise RuntimeError(f"[sensitivity] fit failed (out_dir={out_dir}, rc={result['rc']}); "
                            f"see {result['log_path']}")
    if not asc_path.exists():
        raise FileNotFoundError(f"[sensitivity] expected {asc_path} not found")

    np.savetxt(idx_path, train_idx, fmt="%d", header="train_idx", comments="")
    return asc_path


# ============================================================================
# Layer-export variants.
# ============================================================================

def _export_masked_layers(mask, out_dir, layers=None, pred_dir=None, nodata=maxent.NODATA):
    """Export the 17 canonical layers to ``.asc`` (mirrors ``maxent.
    export_layers``'s own recipe: raw ``.tif`` -> ``predictors.
    apply_modeling_transform`` -> NaN-fill), with every pixel OUTSIDE
    ``mask`` ALSO forced to NODATA -- MaxEnt's background sampling (and any
    training point landing outside ``mask``) is then automatically
    restricted to the target zone, with no ``biasfile`` parameter (which has
    a serialization bug in MaxEnt 3.4.4 -- see this module's docstring and
    the source notebook's own Section 1 note).

    Idempotent per file (an existing ``.asc`` is left untouched), mirroring
    ``export_layers()``'s own per-file cache check.
    """
    if layers is None:
        layers = maxent.modeling_layers()
    if pred_dir is None:
        pred_dir = screen.SCREENED_DIR
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask, dtype=bool)

    files = [pred_dir / f"{name}.tif" for name in layers]
    ref = rioxarray.open_rasterio(files[0]).squeeze("band", drop=True)
    header = maxent._ascii_header(ref, nodata)

    for f in files:
        out_path = out_dir / f"{f.stem}.asc"
        if out_path.exists():
            continue
        da = rioxarray.open_rasterio(f).squeeze("band", drop=True)
        arr = da.values.astype("float64")
        arr = predictors.apply_modeling_transform(f.stem, arr)
        arr = np.where(mask, arr, np.nan)
        arr = np.nan_to_num(arr, nan=nodata)
        if f.stem in maxent.CATEGORICAL:
            arr = np.where(arr == nodata, nodata, np.round(arr)).astype("int64")
            fmt = "%d"
        else:
            fmt = "%.6g"
        with open(out_path, "w") as fh:
            fh.write(header)
            np.savetxt(fh, arr, fmt=fmt)
    return out_dir


def _export_resolution_layers(out_dir, resolution_m=RESOLUTION_M, layers=None, pred_dir=None,
                               nodata=maxent.NODATA):
    """Resample the 17 canonical layers to ``resolution_m`` (bilinear for
    continuous, nearest-neighbour for the two categorical layers -- mirrors
    the source notebook's own resampling choice) via ``rioxarray``'s own
    ``reproject`` (not a hand-rolled affine/out_shape read), then export as
    ``.asc`` the SAME way ``_export_masked_layers``/``maxent.export_layers``
    do (``apply_modeling_transform`` -> NaN-fill -> categorical/continuous
    format dispatch). ``apply_modeling_transform`` is applied AFTER
    resampling here (bilinear-then-transform, not transform-then-bilinear --
    order is inert for the current 17-layer stack, since neither
    ``flow_accumulation`` nor ``spi`` survives screening into it; flagged
    here for a future stack where that might not hold).

    Idempotent per file. Returns ``out_dir``.
    """
    if layers is None:
        layers = maxent.modeling_layers()
    if pred_dir is None:
        pred_dir = screen.SCREENED_DIR
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = None
    for name in layers:
        out_path = out_dir / f"{name}.asc"
        if out_path.exists():
            continue
        da = rioxarray.open_rasterio(pred_dir / f"{name}.tif", masked=True).squeeze("band", drop=True)
        resamp = RS.nearest if name in maxent.CATEGORICAL else RS.bilinear
        da_r = da.rio.reproject(da.rio.crs, resolution=(resolution_m, resolution_m), resampling=resamp)
        if header is None:
            header = maxent._ascii_header(da_r, nodata)

        arr = da_r.values.astype("float64")
        arr = predictors.apply_modeling_transform(name, arr)
        arr = np.nan_to_num(arr, nan=nodata)
        if name in maxent.CATEGORICAL:
            arr = np.where(arr == nodata, nodata, np.round(arr)).astype("int64")
            fmt = "%d"
        else:
            fmt = "%.6g"
        with open(out_path, "w") as fh:
            fh.write(header)
            np.savetxt(fh, arr, fmt=fmt)
    return out_dir


# ============================================================================
# Background variants.
# ============================================================================

def _developed_mask(ref_da):
    """NLCD developed pixels (classes 21-24), ANDed with ``ref_da``'s own
    valid (finite) AOI footprint."""
    nlcd = rioxarray.open_rasterio(screen.SCREENED_DIR / "nlcd_landcover.tif",
                                    masked=True).squeeze("band", drop=True)
    valid = np.isfinite(ref_da.values)
    return valid & np.isin(nlcd.values, NLCD_DEVELOPED)


def _stormwater_buffer_mask(ref_da, buffer_m=SW_BUFFER_M):
    """A ``buffer_m``-meter buffer around the stormwater pipe/channel
    network (``data/processed/infrastructure/stormwater_lines.gpkg``),
    rasterized onto ``ref_da``'s grid, ANDed with its valid footprint."""
    lines = gpd.read_file(predictors.INFRA_DIR / "stormwater_lines.gpkg").to_crs(ref_da.rio.crs)
    buf = lines.buffer(buffer_m).union_all()
    rast = rasterize([(buf, 1)], out_shape=ref_da.rio.shape, transform=ref_da.rio.transform(),
                      fill=0, dtype="uint8").astype(bool)
    return np.isfinite(ref_da.values) & rast


def _mask_background(mask, ref_da, n=N_BACKGROUND, seed=config.GLOBAL_SEED):
    """A random background sample restricted to ``mask`` -- builds an
    in-memory DataArray whose only finite pixels are ``mask``'s True cells,
    then reuses ``evaluate.sample_background`` verbatim (never a second,
    independently-written random-valid-pixel draw)."""
    target_da = ref_da.copy(data=np.where(np.asarray(mask), 1.0, np.nan).astype("float32"))
    return evaluate.sample_background(target_da, n=n, seed=seed)


def _distance_matched_background(presence, dist_da, ref_da, n=N_BACKGROUND,
                                  n_bins=N_OUTFALL_BINS, seed=config.GLOBAL_SEED):
    """Background stratified-resampled so its ``distance_to_outfall``
    distribution matches ``presence``'s own: quantile-bin the presence
    points' distance-to-outfall values (``n_bins`` bins), then draw ``n``
    background points from the AOI's valid pixels, ALLOCATED PROPORTIONALLY
    to each bin's share of the presence histogram (without replacement
    within a bin; a bin whose valid-pixel pool is smaller than its
    allocation is simply exhausted, not over-drawn).

    This is the honest ascertainment-bias UPPER bound (see this module's
    docstring): outfall proximity can no longer separate presence from
    background BY CONSTRUCTION, so whatever discrimination survives is
    attributable to something else.
    """
    pres_dist = evaluate._sample_xy(dist_da, presence["x"].to_numpy(), presence["y"].to_numpy())
    pres_dist = pres_dist[np.isfinite(pres_dist)]

    edges = np.unique(np.quantile(pres_dist, np.linspace(0, 1, n_bins + 1)))
    n_bins_eff = len(edges) - 1
    pres_counts, _ = np.histogram(pres_dist, bins=edges)
    pres_frac = pres_counts / pres_counts.sum()

    dist_arr = dist_da.values
    valid = np.isfinite(dist_arr) & np.isfinite(ref_da.values)
    rows, cols = np.where(valid)
    dvals = dist_arr[rows, cols]
    bin_idx = np.digitize(dvals, edges[1:-1], right=False)

    rng = np.random.default_rng(seed)
    picks = []
    for b in range(n_bins_eff):
        pool = np.flatnonzero(bin_idx == b)
        want = int(round(n * pres_frac[b]))
        if want == 0 or len(pool) == 0:
            continue
        want = min(want, len(pool))
        picks.append(rng.choice(pool, size=want, replace=False))
    picks = np.concatenate(picks) if picks else np.array([], dtype=int)

    transform = dist_da.rio.transform()
    xs, ys = _xy(transform, rows[picks], cols[picks])
    return pd.DataFrame({"x": np.asarray(xs, dtype="float64"), "y": np.asarray(ys, dtype="float64")})


# ============================================================================
# Event-pooling helper.
# ============================================================================

def _top_n_events(samples, n=TOP_N_EVENTS):
    """The top-``n`` events by ACTUAL point count in ``samples`` -- computed
    FRESH every call, never hardcoded to the source notebook's specific
    named list (``2007_5_24``/``2010_6_10``/``2018_9_3``/``2019_5_18``),
    which does not exactly match this project's own canonical event sizes
    (the source's list was sized for a 278-point, differently-thinned
    presence set -- see ``pipeline.robustness``'s own module docstring,
    point 2, for the analogous "never trust the old notebook's specific
    numbers" precedent). Ties broken by event label (ascending) for full
    determinism regardless of pandas' own internal ``value_counts()`` tie
    order.

    Returns ``(labels, counts_dict)`` -- ``labels`` a list of the top-``n``
    event codes, ``counts_dict`` ``{label: n_points}`` for every event (not
    just the top-n), for reporting.
    """
    counts = samples["event"].value_counts()
    ranked = pd.DataFrame({"event": counts.index, "n": counts.to_numpy()}) \
        .sort_values(["n", "event"], ascending=[False, True])
    top_labels = ranked["event"].head(n).tolist()
    return top_labels, dict(zip(ranked["event"], ranked["n"]))


# ============================================================================
# Baseline + arms.
# ============================================================================

def _baseline(canonical, train_idx, test_idx, background_n, seed, timeout):
    """The overall baseline: FINAL_CONFIG train-only model (reused from
    ``evaluate.EVAL_RASTER`` (from ``maxent.fit_eval()``) if its recorded train split
    matches, via ``robustness._baseline_eval_raster`` -- never re-fit for
    nothing), scored at the standard 83 test points against a random
    ``background_n``-point background (``robustness.
    _background_reference_raster``'s own footprint, ``evaluate.
    sample_background`` -- the SAME recipe ``evaluate.oos_auc`` and spatial CV already use). This
    is expected to reproduce the pinned 0.982577108433735 bit-for-bit (see
    ``tests/test_evaluate_oos.py``/``tests/test_robustness_folds.py``).

    Returns ``(baseline_dict, baseline_raster_path, test_pts, background)``
    -- the latter three threaded into every arm that shares this exact
    protocol.
    """
    baseline_raster = robustness._baseline_eval_raster(canonical, train_idx, timeout=timeout)
    test_pts = canonical.iloc[test_idx][["x", "y"]]
    ref = robustness._background_reference_raster()
    background = evaluate.sample_background(ref, n=background_n, seed=seed)

    auc, n_p, n_b = _score(baseline_raster, test_pts, background, train_idx, test_idx)
    eval_id = _make_eval_id("standard83", test_pts, "random10k", background)

    b = dict(
        arm="baseline",
        description=(
            "Random 10k-valid-pixel background, 10 m resolution, cloglog output, all events pooled -- "
            "FINAL_CONFIG (beta=0.5, LQ) train-only model (194/83 canonical split), the SAME model/"
            "protocol the honest out-of-sample AUC uses."
        ),
        auc=auc, baseline_auc=auc, delta_auc=0.0, eval_id=eval_id, baseline_eval_id=eval_id,
        n_presence=n_p, n_background=n_b,
    )
    return b, baseline_raster, test_pts, background


def _arm_resolution_30m(canonical, train_idx, test_idx, baseline, test_pts, background, timeout):
    layers_dir = _export_resolution_layers(SENS_DIR / "resolution_30m" / "layers")
    asc = _fit_train_only(train_idx, test_idx, SENS_DIR / "resolution_30m" / "fit",
                           layers_dir=layers_dir, samples=canonical, timeout=timeout)
    auc, n_p, n_b = _score(asc, test_pts, background, train_idx, test_idx)
    eval_id = _make_eval_id("standard83", test_pts, "random10k", background)
    return dict(
        arm="resolution_30m",
        description=(
            f"Predictor stack resampled to {RESOLUTION_M} m (bilinear continuous / nearest-neighbour "
            "categorical), re-exported and refit at FINAL_CONFIG; scored at the SAME standard 83 test "
            "points + random 10k background as the baseline (only spatial resolution changes)."
        ),
        auc=auc, baseline_auc=baseline["auc"], delta_auc=auc - baseline["auc"],
        eval_id=eval_id, baseline_eval_id=baseline["eval_id"], n_presence=n_p, n_background=n_b,
        n_train_used=_read_n_training(SENS_DIR / "resolution_30m" / "fit"),
        resolution_m=RESOLUTION_M,
    )


def _arm_output_format_logistic(canonical, train_idx, test_idx, baseline, test_pts, background, timeout):
    out_dir = SENS_DIR / "output_format_logistic"
    asc = _fit_train_only(train_idx, test_idx, out_dir, layers_dir=maxent.LAYERS_DIR,
                           samples=canonical, outputformat="logistic", timeout=timeout)
    auc, n_p, n_b = _score(asc, test_pts, background, train_idx, test_idx)
    eval_id = _make_eval_id("standard83", test_pts, "random10k", background)
    return dict(
        arm="output_format_logistic",
        description=(
            "Same FINAL_CONFIG train-only model/points; outputformat=logistic instead of cloglog. Both "
            "are strictly monotonic transforms of the same linear predictor, so AUC (rank-based) is "
            "expected to be numerically identical to the baseline's cloglog AUC."
        ),
        auc=auc, baseline_auc=baseline["auc"], delta_auc=auc - baseline["auc"],
        eval_id=eval_id, baseline_eval_id=baseline["eval_id"], n_presence=n_p, n_background=n_b,
        n_train_used=_read_n_training(out_dir), output_format="logistic",
    )


def _arm_event_pooling(canonical, background, timeout, top_n=TOP_N_EVENTS):
    top_labels, all_counts = _top_n_events(canonical, top_n)
    top_mask = canonical["event"].isin(top_labels).to_numpy()
    top_global_idx = np.flatnonzero(top_mask)
    top_samples_local = canonical.iloc[top_global_idx].reset_index(drop=True)

    train4_local, test4_local = config.split(top_samples_local)
    test4_global = top_global_idx[test4_local]
    train4_global = top_global_idx[train4_local]
    baseline4_train_global = np.setdiff1d(np.arange(len(canonical)), test4_global)

    test4_pts = canonical.iloc[test4_global][["x", "y"]]

    arm_dir = SENS_DIR / "event_pooling_top4" / "arm"
    arm_asc = _fit_train_only(train4_global, test4_global, arm_dir,
                               layers_dir=maxent.LAYERS_DIR, samples=canonical, timeout=timeout)
    arm_auc, n_p, n_b = _score(arm_asc, test4_pts, background, train4_global, test4_global)

    base_dir = SENS_DIR / "event_pooling_top4" / "baseline"
    baseline_asc = _fit_train_only(baseline4_train_global, test4_global, base_dir,
                                    layers_dir=maxent.LAYERS_DIR, samples=canonical, timeout=timeout)
    baseline_auc, _, _ = _score(baseline_asc, test4_pts, background, baseline4_train_global, test4_global)

    eval_id = _make_eval_id("top4_test", test4_pts, "random10k", background)
    return dict(
        arm="event_pooling_top4",
        description=(
            f"Top-{top_n} largest events by ACTUAL point count in the current canonical 277-point set "
            f"({ {k: int(v) for k, v in all_counts.items()} }) -- computed fresh, NOT hardcoded to the "
            "source notebook's 2007/2010/2018/2019 list (which does not exactly match this project's own "
            "event sizes). Arm trains on top-N-events-only presence; matched baseline trains on ALL "
            "events EXCEPT this arm's own held-out test points, so both score the IDENTICAL held-out "
            "top-N-event test points + the standard random background."
        ),
        auc=arm_auc, baseline_auc=baseline_auc, delta_auc=arm_auc - baseline_auc,
        eval_id=eval_id, baseline_eval_id=eval_id, n_presence=n_p, n_background=n_b,
        events_included=top_labels, n_events_total=len(all_counts),
        n_train_arm=int(len(train4_global)), n_train_baseline=int(len(baseline4_train_global)),
        n_test=int(len(test4_global)),
        n_train_used_arm=_read_n_training(arm_dir), n_train_used_baseline=_read_n_training(base_dir),
    )


MATCHED_REFIT_CSV = config.DATA / "robustness" / "matched_background_refit.csv"


def matched_background_refit(*, canonical=None, timeout=maxent.DEFAULT_TIMEOUT_S,
                             n_bins=N_OUTFALL_BINS, seed=config.GLOBAL_SEED):
    """REFIT MaxEnt on the outfall-distance-matched background, then ask whether
    the bias-corrected model still ranks outfall proximity highly.

    The existing ``bg_outfall_distance_matched`` arm "retrains NOTHING" -- it reuses
    the standard fit and swaps only the *evaluation* background. That measures
    how discrimination responds to the evaluation contrast; it cannot answer the
    question this function asks:

        does a model TRAINED against a survey-mimicking background still assign
        large importance to distance to outfall?

    Mechanics: draw the distance-matched background, rasterise it (plus the
    presence pixels) to a mask, export layers masked to it, and refit train-only
    -- so MaxEnt's own training background is drawn from the matched set rather
    than from the whole AOI. This reuses the same masking machinery the
    developed / stormwater target-group arms already use.

    Returns a one-row-per-predictor DataFrame of the refit model's permutation
    importance alongside the standard model's, plus AUCs, and writes it to
    ``data/processed/robustness/matched_background_refit.csv``.

    **This does not eliminate ascertainment bias.** Matching the *marginal*
    distance-to-outfall distribution leaves every other access-related structure
    (roads, rights-of-way, inspection priorities, damage reports) uncontrolled,
    and the stack contains no road/access layer at all.
    """
    if canonical is None:
        canonical = config.canonical_samples()
    canonical = canonical.reset_index(drop=True)
    train_idx, test_idx = config.split(canonical)
    test_pts_all = canonical.iloc[test_idx][["x", "y"]]

    ref = maxent._reference_raster()
    transform = ref.rio.transform()
    dist_da = evaluate._open_raster(config.DATA / "predictors_screened" /
                                   "distance_to_outfall.tif")

    matched_bg = _distance_matched_background(canonical, dist_da, ref,
                                              n=N_BACKGROUND, n_bins=n_bins, seed=seed)

    # mask = the matched background pixels PLUS every presence pixel (MaxEnt must
    # see its presences as valid); everything else becomes nodata, so MaxEnt's
    # own background draw is confined to the matched set.
    mask = np.zeros(ref.shape, dtype=bool)
    for pts in (matched_bg[["x", "y"]], canonical[["x", "y"]]):
        rows, cols = _rowcol(transform, pts["x"].to_numpy(), pts["y"].to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = ((rows >= 0) & (rows < ref.shape[0]) & (cols >= 0) & (cols < ref.shape[1]))
        mask[rows[ok], cols[ok]] = True
    print(f"[sensitivity] matched-background mask: {int(mask.sum()):,} valid pixels "
          f"(of {int(np.isfinite(ref.values).sum()):,})")

    layers_dir = _export_masked_layers(mask, SENS_DIR / "bg_matched_refit" / "layers")
    fit_dir = SENS_DIR / "bg_matched_refit" / "fit"
    arm_asc = _fit_train_only(train_idx, test_idx, fit_dir, layers_dir=layers_dir,
                              samples=canonical, timeout=timeout)

    filtered_test = _filter_points_by_mask(test_pts_all, mask, transform)
    arm_auc, _, _ = _score(arm_asc, filtered_test, matched_bg, train_idx, test_idx)
    baseline_raster = robustness._baseline_eval_raster(canonical, train_idx, timeout=timeout)
    base_auc, _, _ = _score(baseline_raster, filtered_test, matched_bg, train_idx, test_idx)
    print(f"[sensitivity] matched-background REFIT AUC={arm_auc:.4f}  "
          f"(standard model on the same contrast: {base_auc:.4f})")

    def _perm(results_csv):
        df = pd.read_csv(results_csv)
        row = df.iloc[0]
        out = {}
        for col in df.columns:
            if col.endswith(" permutation importance"):
                out[col[: -len(" permutation importance")]] = float(row[col])
        return out

    refit_imp = _perm(Path(fit_dir) / "maxentResults.csv")
    std_imp = _perm(Path(baseline_raster).parent / "maxentResults.csv")

    preds = sorted(set(refit_imp) | set(std_imp))
    df = pd.DataFrame([
        dict(predictor=p,
             perm_importance_standard=std_imp.get(p, float("nan")),
             perm_importance_matched_refit=refit_imp.get(p, float("nan")))
        for p in preds
    ]).sort_values("perm_importance_matched_refit", ascending=False).reset_index(drop=True)
    df.attrs["auc_matched_refit"] = arm_auc
    df.attrs["auc_standard_same_contrast"] = base_auc
    df["auc_matched_refit"] = arm_auc
    df["auc_standard_same_contrast"] = base_auc

    MATCHED_REFIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MATCHED_REFIT_CSV, index=False)
    print(f"[sensitivity] wrote {MATCHED_REFIT_CSV.name}")
    return df


def _target_group_background_arm(name, description, mask, canonical, train_idx, test_idx,
                                  test_pts_all, timeout, extra_fields=None):
    """Shared mechanics behind the developed/stormwater arms: retrain on
    layers masked to ``mask``, evaluate against a like-vs-like matched
    baseline (the STANDARD, unmasked train-only model scored on the SAME
    restricted test background + filtered test presence).

    ``baseline_auc`` is scored via the SAME ``filtered_test_pts``/
    ``target_bg`` objects as ``arm_auc`` (see the two ``_score`` calls
    below) -- so the returned ``baseline_eval_id`` is set to this arm's own
    ``eval_id`` verbatim: the like-vs-like guarantee, made directly
    machine-checkable (``d["eval_id"] == d["baseline_eval_id"]``)."""
    ref = maxent._reference_raster()
    transform = ref.rio.transform()

    filtered_test_pts = _filter_points_by_mask(test_pts_all, mask, transform)
    target_bg = _mask_background(mask, ref, n=N_BACKGROUND, seed=config.GLOBAL_SEED)

    layers_dir = _export_masked_layers(mask, SENS_DIR / name / "layers")
    fit_dir = SENS_DIR / name / "fit"
    arm_asc = _fit_train_only(train_idx, test_idx, fit_dir, layers_dir=layers_dir,
                               samples=canonical, timeout=timeout)
    arm_auc, n_p, n_b = _score(arm_asc, filtered_test_pts, target_bg, train_idx, test_idx)

    baseline_raster = robustness._baseline_eval_raster(canonical, train_idx, timeout=timeout)
    baseline_auc, _, _ = _score(baseline_raster, filtered_test_pts, target_bg, train_idx, test_idx)

    eval_id = _make_eval_id(f"{name}_test", filtered_test_pts, f"{name}_bg", target_bg)
    n_valid = int(np.isfinite(ref.values).sum())
    n_target = int(mask.sum())

    d = dict(
        arm=name, description=description,
        auc=arm_auc, baseline_auc=baseline_auc, delta_auc=arm_auc - baseline_auc,
        eval_id=eval_id, baseline_eval_id=eval_id, n_presence=n_p, n_background=n_b,
        n_presence_in_zone=len(filtered_test_pts), n_presence_standard_total=len(test_pts_all),
        pct_aoi=100.0 * n_target / n_valid,
        n_train_used=_read_n_training(fit_dir),
    )
    if extra_fields:
        d.update(extra_fields)
    return d


def _arm_background_developed(canonical, train_idx, test_idx, test_pts_all, timeout):
    ref = maxent._reference_raster()
    mask = _developed_mask(ref)
    return _target_group_background_arm(
        "bg_target_group_developed",
        "Target-group background restricted to NLCD developed pixels (classes 21-24) via layer masking "
        "(MaxEnt 3.4.4's biasfile parameter has a serialization bug -- masking is the equivalent fix; "
        "see module docstring). Matched baseline reuses the standard train-only model scored on the SAME "
        "developed-only test background + filtered test presence.",
        mask, canonical, train_idx, test_idx, test_pts_all, timeout,
    )


def _arm_background_stormwater(canonical, train_idx, test_idx, test_pts_all, timeout):
    ref = maxent._reference_raster()
    mask = _stormwater_buffer_mask(ref, buffer_m=SW_BUFFER_M)
    caveat = (
        "LOWER BOUND on ascertainment bias, not a removal of it: distance_to_outfall is distance to "
        "discrete outfall POINTS, but this arm buffers stormwater LINES (pipes/channels). Within the "
        "buffer, the outfall-proximity axis can still separate presence from background (a pixel can be "
        "near the stormwater network yet far from any outfall, or vice versa), so this arm REDUCES but "
        "does NOT REMOVE the structural ascertainment bias. Treat delta_auc here as a lower "
        "bound on how much of the standard model's AUC reflects ascertainment -- see "
        "bg_outfall_distance_matched for the proper (upper-bound) isolation of signal beyond outfall "
        "proximity."
    )
    return _target_group_background_arm(
        "bg_target_group_stormwater_buffer",
        f"Target-group background restricted to a {SW_BUFFER_M} m buffer around the stormwater pipe/"
        "channel network; matched baseline reuses the standard train-only model scored on "
        "the SAME buffer-only test background + filtered test presence. " + caveat,
        mask, canonical, train_idx, test_idx, test_pts_all, timeout,
        extra_fields=dict(lower_bound=True, bound="lower_bound", caveat=caveat),
    )


def _arm_background_outfall_matched(canonical, baseline_raster, test_pts_all, baseline_auc,
                                     baseline_eval_id, train_idx, test_idx, timeout):
    """THE NEW, proper ascertainment-bias bound (see module docstring). No
    retraining -- reuses the standard baseline model verbatim; only the
    EVALUATION background changes (stratified-resampled to match presence's
    own distance_to_outfall distribution).

    Unlike the developed/stormwater arms, this one is deliberately NOT
    like-vs-like: ``baseline_auc`` (and the ``baseline_eval_id`` caller
    passes in, verbatim the OVERALL baseline's own ``eval_id``) was scored
    against the STANDARD random background, not the distance-matched
    ``background`` built below -- the whole point of this arm is comparing
    the SAME model under two DIFFERENT backgrounds (random vs. distance-
    matched), so this arm's own ``eval_id`` is expected to DIFFER from
    ``baseline_eval_id`` (same test points, different background -- see
    ``_make_eval_id``'s ``bg_tag``: ``"random10k"`` vs.
    ``"outfall_distance_matched"``)."""
    ref = maxent._reference_raster()
    dist_da = rioxarray.open_rasterio(screen.SCREENED_DIR / "distance_to_outfall.tif",
                                       masked=True).squeeze("band", drop=True)
    background = _distance_matched_background(canonical, dist_da, ref, n=N_BACKGROUND,
                                                seed=config.GLOBAL_SEED)
    auc, n_p, n_b = _score(baseline_raster, test_pts_all, background, train_idx, test_idx)
    eval_id = _make_eval_id("standard83", test_pts_all, "outfall_distance_matched", background)

    return dict(
        arm="bg_outfall_distance_matched",
        description=(
            "Same FINAL_CONFIG train-only model as the baseline (no retraining); background stratified-"
            "resampled (10 quantile bins, proportional allocation) so its distance_to_outfall histogram "
            "matches the 277 canonical presence points' own."
        ),
        auc=auc, baseline_auc=baseline_auc, delta_auc=auc - baseline_auc,
        eval_id=eval_id, baseline_eval_id=baseline_eval_id, n_presence=n_p, n_background=n_b,
        bound="proper_upper_bound_on_ascertainment_bias",
        measures=(
            "Isolates model discrimination NOT attributable to outfall proximity: background is drawn so "
            "it has the SAME distance_to_outfall distribution as the presence points, removing outfall-"
            "proximity as an axis that can separate presence from background. If AUC collapses toward "
            "0.5 here, the standard model's apparent skill was largely ascertainment-driven (surveyed-"
            "near-infrastructure bias); if AUC stays well above 0.5, real signal beyond outfall proximity "
            "survives."
        ),
    )


# ============================================================================
# Orchestrator.
# ============================================================================

def _write_summary(baseline, arms):
    rows = [dict(arm=baseline["arm"], description=baseline["description"], auc=baseline["auc"],
                 baseline_auc=baseline["baseline_auc"], delta_auc=baseline["delta_auc"],
                 eval_id=baseline["eval_id"])]
    for a in arms:
        rows.append(dict(arm=a["arm"], description=a.get("description", ""), auc=a["auc"],
                          baseline_auc=a["baseline_auc"], delta_auc=a["delta_auc"], eval_id=a["eval_id"]))
    df = pd.DataFrame(rows)
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    return df


def run(*, background_n=N_BACKGROUND, seed=config.GLOBAL_SEED, top_n_events=TOP_N_EVENTS,
        timeout=maxent.DEFAULT_TIMEOUT_S):
    """Run every sensitivity arm and return ``(baseline_dict, arms_list)``.

    ``arms_list`` order: ``resolution_30m`` (shares the baseline's own
    ``eval_id`` -- the required cross-check), the three background-sampling
    arms (developed / stormwater-buffer LOWER bound / outfall-distance-
    matched PROPER bound), ``output_format_logistic``, then
    ``event_pooling_top4``.

    Cache-aware throughout (``_fit_train_only``'s own ``train_idx.csv``
    provenance check): a re-entrant call after a partial/interrupted run
    only fits whatever is still missing.
    """
    maxent.export_layers()
    canonical = config.canonical_samples()
    train_idx, test_idx = config.split(canonical)

    baseline, baseline_raster, test_pts, background = _baseline(
        canonical, train_idx, test_idx, background_n, seed, timeout)

    arms = [
        _arm_resolution_30m(canonical, train_idx, test_idx, baseline, test_pts, background, timeout),
        _arm_background_developed(canonical, train_idx, test_idx, test_pts, timeout),
        _arm_background_stormwater(canonical, train_idx, test_idx, test_pts, timeout),
        _arm_background_outfall_matched(canonical, baseline_raster, test_pts, baseline["auc"],
                                         baseline["eval_id"], train_idx, test_idx, timeout),
        _arm_output_format_logistic(canonical, train_idx, test_idx, baseline, test_pts, background, timeout),
        _arm_event_pooling(canonical, background, timeout, top_n=top_n_events),
    ]

    _write_summary(baseline, arms)
    return baseline, arms

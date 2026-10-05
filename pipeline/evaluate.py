"""The HONEST out-of-sample AUC, never a resubstitution number silently
standing in for it.

Background: the original notebook pipeline (notebook ``05_validation``)
sampled the PUBLISHED (full-277-point) MaxEnt raster at 83 nominally
"held-out" test points and reported the resulting AUC (0.9842) as the
headline validation number -- but that raster HAD been trained on those
same 83 points (they are a subset of the 277 canonical presence points
the final model fits on), so 0.9842 was resubstitution accuracy wearing a
held-out-validation costume. The true held-out AUC (sampling a model that
had genuinely never seen the 83 test points) was 0.9684.

``pipeline.maxent`` provides the fix's PREREQUISITE: two distinct
model families fit at the same locked configuration (``FINAL_CONFIG``,
beta=0.5/LQ) but on deliberately different point sets --
  - ``fit_final``      -- ALL 277 canonical points -> the PUBLISHED map
    (``maxent.FINAL_DIR / "cloglog" / "flood_avg.tif"``, this module's
    ``FINAL_RASTER``). Its own self-reported "Training AUC"
    (``maxent.apparent_auc()`` / this module's ``apparent_auc``) is
    resubstitution and must NEVER be reported as the headline.
  - ``fit_eval``        -- the 194 TRAIN-split points only -> a raster
    that has never seen the 83 held-out TEST points
    (``maxent.HOLDOUT_MODEL_DIR / "flood.tif"``, this module's
    ``EVAL_RASTER``).

THIS module is the fix itself: ``oos_auc`` samples ``EVAL_RASTER`` (never
``FINAL_RASTER``) at the TEST points for the headline number, asserting
``qa.assert_eval_model_excludes_test`` BEFORE any scoring so a caller can
never silently reconstruct that resubstitution bug by accident. Everything else here
(``threshold_metrics``, ``success_prediction_rate``, ``classed_map``)
composes around that one honest evaluation set.

**Module map:**
  - ``sample_background``           -- seeded random AOI valid-pixel draw.
  - ``oos_auc``                     -- THE headline: honest holdout AUC.
  - ``oos_scores``                  -- the (y_true, y_score) pair behind
                                        it, exposed so threshold_metrics/
                                        internal_validation_table never
                                        re-derive a second, possibly-
                                        drifting evaluation set.
  - ``apparent_auc``                -- re-export of ``maxent.apparent_auc``
                                        (resubstitution, NOT the headline).
  - ``threshold_metrics``           -- ported from the original notebook.
  - ``maxss_threshold_from_results``/``p10_threshold_from_results``/
    ``maxss_threshold_from_roc``    -- threshold sourcing for the table.
  - ``internal_validation_table``   -- MaxSS/P10 rows, composed from the
                                        above (convenience, not a second
                                        implementation).
  - ``success_prediction_rate``     -- the two missing field-standard SDM
                                        curves (success rate = fraction of
                                        TEST presence captured; prediction
                                        rate = fraction of the AOI's
                                        valid pixels flagged susceptible),
                                        both against ``EVAL_RASTER`` so
                                        this figure never mixes the
                                        full/final model into what must
                                        stay an honest, held-out view.
  - ``classed_map``                 -- a density-classed (default
                                        quantile, optionally Jenks natural
                                        breaks) susceptibility map from
                                        the FINAL/published raster -- a
                                        cartographic product of the model
                                        actually published, not a held-out
                                        evaluation artifact, so it
                                        deliberately uses ``FINAL_RASTER``
                                        (no test-point label is involved,
                                        hence no train/test leakage concern here).
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")   # headless-safe (no DISPLAY needed in CI/pytest)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from rasterio.features import rasterize
from rasterio.transform import xy as _rasterio_xy
from sklearn.metrics import roc_auc_score, roc_curve

from pipeline import config, maxent, predictors, qa

EVAL_RASTER = maxent.HOLDOUT_MODEL_DIR / "flood.tif"
EVAL_RESULTS_CSV = maxent.HOLDOUT_MODEL_DIR / "maxentResults.csv"
FINAL_RASTER = maxent.FINAL_DIR / "cloglog" / "flood_avg.tif"
VALIDATION_DIR = config.DATA / "validation"

MAXSS_COL = "Maximum training sensitivity plus specificity Cloglog threshold"
P10_COL = "10 percentile training presence Cloglog threshold"

DEFAULT_CLASS_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]

DEFAULT_BACKGROUND_N = 10_000


# ============================================================================
# Raster I/O + point sampling -- thin wrappers. Sampling itself reuses
# ``predictors._sample_at_points`` (the ONE rowcol/floor convention this
# whole pipeline shares -- see config.py's ``_modeling_grid_rowcol``
# docstring on why that function is never reimplemented a third time), so a
# point's assigned pixel here is guaranteed identical to what every other
# stage of this pipeline would assign it.
# ============================================================================

def _open_raster(raster):
    """Accept either a path-like (opened fresh via rioxarray) or an
    already-open xarray DataArray (as tests hand in for small synthetic
    rasters) -- returns a 2-D, band-squeezed DataArray either way."""
    if isinstance(raster, (str, Path)):
        return rioxarray.open_rasterio(raster).squeeze("band", drop=True)
    return raster


def _sample_xy(raster_da, x, y):
    """Sample one raster DataArray at point coordinates -- a 1-D float
    ndarray, NaN wherever the point falls outside the raster extent
    (never a raised error, matching ``predictors._sample_at_points``'s own
    contract)."""
    values_df, _ = predictors._sample_at_points({"score": raster_da}, np.asarray(x), np.asarray(y))
    return values_df["score"].to_numpy()


def sample_background(raster, n=DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED):
    """A random AOI valid-pixel sample, seeded (default ``config.
    GLOBAL_SEED``) so repeated calls with the same seed reproduce the
    IDENTICAL point set (and hence identical downstream AUC).

    Drawn from `raster`'s OWN non-NaN footprint rather than reloading the
    full predictor stack (the original notebook's approach): every
    MaxEnt output raster this pipeline produces is already masked to the
    canonical AOI footprint (``maxent._asc_to_geotiff``'s ``da.where(~ref.
    isnull())`` against ``dem.tif``), so `raster`'s own finite pixels ARE
    the AOI validity mask -- no second file dependency needed.

    Returns a DataFrame with ``x``, ``y`` columns (pixel-center
    coordinates, matching ``config.canonical_samples()``'s own column
    convention so the result can be fed directly wherever a presence-style
    point table is expected).
    """
    da = _open_raster(raster)
    valid = np.isfinite(da.values)
    rows, cols = np.where(valid)

    rng = np.random.default_rng(seed)
    n = min(n, len(rows))
    pick = rng.choice(len(rows), size=n, replace=False)
    r, c = rows[pick], cols[pick]

    transform = da.rio.transform()
    xs, ys = _rasterio_xy(transform, r, c)
    return pd.DataFrame({"x": np.asarray(xs, dtype="float64"), "y": np.asarray(ys, dtype="float64")})


def _presence_background_scores(eval_raster, points, background):
    """Sample `eval_raster` at `points` (x/y columns) and at `background`
    (x/y columns), dropping any out-of-bounds/NaN samples from each side
    independently (matches the original notebook's own defensive
    ``np.isfinite`` filtering). Returns ``(presence_scores,
    background_scores)``, both 1-D float ndarrays.

    Factored out so ``oos_auc`` and ``oos_scores`` share EXACTLY one
    sampling path -- the headline AUC and the y_true/y_score pair fed to
    ``threshold_metrics``/``success_prediction_rate`` can never silently
    diverge from each other.
    """
    raster_da = _open_raster(eval_raster)
    presence_scores = _sample_xy(raster_da, points["x"].to_numpy(), points["y"].to_numpy())
    bg_scores = _sample_xy(raster_da, background["x"].to_numpy(), background["y"].to_numpy())

    presence_scores = presence_scores[np.isfinite(presence_scores)]
    bg_scores = bg_scores[np.isfinite(bg_scores)]
    assert len(presence_scores) > 0, "[evaluate] no valid presence scores -- points off the AOI?"
    assert len(bg_scores) > 0, "[evaluate] no valid background scores -- points off the AOI?"
    return presence_scores, bg_scores


def oos_scores(eval_raster, test_pts, background):
    """The ``(y_true, y_score)`` pair behind ``oos_auc`` -- exposed
    separately so a caller building the threshold-metrics table or the
    success/prediction-rate curves scores EXACTLY the same evaluation set
    the headline AUC uses, rather than re-sampling independently."""
    presence_scores, bg_scores = _presence_background_scores(eval_raster, test_pts, background)
    y_true = np.concatenate([np.ones(len(presence_scores)), np.zeros(len(bg_scores))])
    y_score = np.concatenate([presence_scores, bg_scores])
    return y_true, y_score


# ============================================================================
# THE TRAIN/TEST SEPARATION GUARD
# ============================================================================

def oos_auc(eval_raster, test_pts, train_idx, test_idx, background):
    """THE headline number: the honest out-of-sample AUC.

    Samples `eval_raster` -- which MUST be the TRAIN-ONLY evaluation
    raster (``EVAL_RASTER`` / ``maxent.HOLDOUT_MODEL_DIR / "flood.tif"``,
    fit on the 194 TRAIN points only), NEVER the published/full-model
    raster (``FINAL_RASTER``) -- at the TEST presence points
    (`test_pts`, an x/y point table) and at `background` (an x/y point
    table, e.g. from ``sample_background``), then computes
    ``roc_auc_score`` over presence-vs-background labels.

    **Invariant 2, enforced FIRST, before touching `eval_raster`/
    `test_pts`/`background` at all**: ``qa.
    assert_eval_model_excludes_test(train_idx, test_idx)`` -- if the
    model's own recorded training index overlaps the points this call is
    about to score it on, this raises ``AssertionError`` immediately.
    This is the fix in one line: a resubstitution AUC is never even
    computed, let alone reported, for an overlapping train/test pair.

    Returns a Python ``float`` -- the honest, out-of-sample AUC.
    """
    qa.assert_eval_model_excludes_test(train_idx, test_idx)

    y_true, y_score = oos_scores(eval_raster, test_pts, background)
    return float(roc_auc_score(y_true, y_score))


# Re-export: maxent.apparent_auc already computes the ONLY
# resubstitution AUC this pipeline reports (mean +/- sd of the FINAL/
# full-277-point bootstrap replicates' own self-reported "Training AUC")
# -- reusing it here (rather than a second, independent re-derivation)
# means there is exactly one implementation of "apparent AUC" in this
# codebase, callable from either module. See maxent.apparent_auc's own
# docstring for the full resubstitution background this name is chosen to avoid
# repeating ("apparent_auc", never "auc"/"headline_auc").
apparent_auc = maxent.apparent_auc


# ============================================================================
# Threshold-based metrics -- ported from the original notebook's
# `threshold_metrics` (Sec. 2) verbatim in arithmetic (tp/tn/fp/fn,
# sensitivity, specificity, precision, TSS, F1, Cohen's Kappa, HSS), pure
# numpy, zero raster/file dependency.
# ============================================================================

def threshold_metrics(y_true, y_score, threshold, label=""):
    """Confusion-matrix-derived metrics at one operating threshold.

    Direct port of the original notebook's ``threshold_metrics`` --
    same formulas, same guarded division-by-zero defaults (0. rather than
    NaN/raising when a denominator is 0, e.g. a threshold above every
    score leaves tp=fp=0 so precision/F1 fall back to 0.). Only prints a
    verbose block when `label` is truthy (matches the source notebook's
    own behavior of staying silent for the ~200-point threshold-curve
    sweep and verbose only for the 2-3 headline thresholds).

    Returns a dict: threshold, label, tp, tn, fp, fn, sensitivity,
    specificity, precision, tss, f1, kappa, hss, pred_frac.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.
    tss = sens + spec - 1
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.

    po = (tp + tn) / (tp + tn + fp + fn)
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (tp + tn + fp + fn) ** 2
    kap = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.

    hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / hss_denom if hss_denom > 0 else 0.

    pred_frac = float((y_pred == 1).mean())

    if label:
        print(f"Threshold: {threshold:.4f} ({label})  "
              f"TP={tp} FP={fp} TN={tn} FN={fn}  "
              f"sens={sens:.4f} spec={spec:.4f} tss={tss:.4f} "
              f"f1={f1:.4f} kappa={kap:.4f} hss={hss:.4f}")

    return dict(threshold=threshold, label=label, tp=tp, tn=tn, fp=fp, fn=fn,
                sensitivity=sens, specificity=spec, precision=ppv,
                tss=tss, f1=f1, kappa=kap, hss=hss, pred_frac=pred_frac)


def _read_results_threshold(results_csv, column):
    """One row's value of `column` from a MaxEnt ``maxentResults.csv`` --
    the ``"flood"`` species row if present (the non-bootstrapped eval fit,
    ``replicates=0``), else the last row (defensive fallback, mirrors
    ``apparent_auc``'s own species-row-then-fallback pattern)."""
    df = pd.read_csv(results_csv)
    row = df[df["Species"].astype(str) == "flood"]
    if row.empty:
        row = df.iloc[[-1]]
    return float(row[column].iloc[0])


def maxss_threshold_from_results(results_csv=EVAL_RESULTS_CSV):
    """The eval model's OWN "Maximum training sensitivity plus specificity
    Cloglog threshold" -- MaxEnt's standard SDM operating threshold,
    computed from the EVAL (194-point, single-run) fit's own training
    data. Matches the original notebook's ``THRESH_FULL`` in spirit (that
    notebook needed a bootstrap-mean vs full-row distinction because its
    ``maxentResults.csv`` had 10 replicate rows; this project's
    ``holdout_model/maxentResults.csv`` is always a single non-bootstrapped
    run, so there is only one row to read)."""
    return _read_results_threshold(results_csv, MAXSS_COL)


def p10_threshold_from_results(results_csv=EVAL_RESULTS_CSV):
    """The eval model's own "10 percentile training presence Cloglog
    threshold" -- there is no ROC-based equivalent to fall back to (unlike
    MaxSS/Youden's J, a percentile-of-training-presence rule is not a
    sensitivity/specificity trade-off point), so this always reads the
    file."""
    return _read_results_threshold(results_csv, P10_COL)


def maxss_threshold_from_roc(y_true, y_score):
    """Empirical MaxSS-equivalent threshold from the TEST ROC curve
    itself: Youden's J = argmax(tpr - fpr), algebraically identical to
    maximizing TSS = sens + spec - 1 (since spec = 1 - fpr). A cross-check/
    fallback for ``maxss_threshold_from_results`` -- computed directly from the held-out
    evaluation set rather than the training data, so it can legitimately
    differ from the file-sourced threshold above."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def internal_validation_table(eval_raster, test_pts, background,
                               results_csv=EVAL_RESULTS_CSV):
    """The MaxSS + 10th-percentile rows of ``threshold_metrics``, scored
    against the SAME honest evaluation set ``oos_auc`` uses (``oos_scores``
    -- never a second, independently-resampled evaluation set). A
    convenience composition of the primitives above (not a new scoring
    path): the report's threshold-metrics table is exactly this
    DataFrame.
    """
    y_true, y_score = oos_scores(eval_raster, test_pts, background)

    maxss_t = maxss_threshold_from_results(results_csv)
    p10_t = p10_threshold_from_results(results_csv)

    rows = [
        threshold_metrics(y_true, y_score, maxss_t, label="MaxSS"),
        threshold_metrics(y_true, y_score, p10_t, label="10th percentile"),
    ]
    return pd.DataFrame(rows)


# ============================================================================
# success_prediction_rate -- the two missing field-standard SDM figures.
# Both curves are computed against EVAL_RASTER-consistent inputs (the
# raster's OWN full valid-pixel population for "prediction rate", the SAME
# test-presence scores oos_auc uses for "success rate") -- never the
# FINAL/full-model raster, so this figure stays inside the honest,
# held-out frame established above.
# ============================================================================

def _success_prediction_rate_arrays(valid_vals, presence_scores, thresholds):
    """Pure-numpy core (no raster I/O -- unit-testable against small
    synthetic arrays directly):
      success_rate(t)    = fraction of `presence_scores` >= t
      prediction_rate(t) = fraction of `valid_vals` (the ENTIRE AOI's
                            valid-pixel population, not a sample) >= t
    `valid_vals` is sorted once; each threshold's prediction_rate is then
    an O(log n) ``searchsorted`` lookup rather than a fresh O(n) compare
    -- cheap either way at a few million pixels, but this keeps a dense
    threshold sweep instant.
    """
    thresholds = np.asarray(thresholds, dtype="float64")
    presence_scores = np.asarray(presence_scores, dtype="float64")

    sorted_valid = np.sort(np.asarray(valid_vals, dtype="float64"))
    n_valid = len(sorted_valid)
    n_pres = len(presence_scores)

    success_rate = np.array([
        float(np.mean(presence_scores >= t)) if n_pres else np.nan for t in thresholds
    ])

    idx = np.searchsorted(sorted_valid, thresholds, side="left")
    prediction_rate = (n_valid - idx) / n_valid if n_valid else np.full(len(thresholds), np.nan)

    return success_rate, prediction_rate


def _plot_success_prediction_rate(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(df["threshold"], df["success_rate"], color="#1565C0", linewidth=1.8, label="Success rate (test presence captured)")
    ax.plot(df["threshold"], df["prediction_rate"], color="#E53935", linewidth=1.8, label="Prediction rate (AOI area flagged)")
    ax.set_xlabel("Susceptibility threshold")
    ax.set_ylabel("Fraction")
    ax.set_title("(a) Success / prediction rate vs. threshold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(df["prediction_rate"], df["success_rate"], color="#2E7D32", linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="random")
    ax.set_xlabel("Prediction rate (cumulative fraction of AOI area)")
    ax.set_ylabel("Success rate (cumulative fraction of test presence captured)")
    ax.set_title("(b) Combined success/prediction-rate curve")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Success-rate / prediction-rate curves -- honest holdout (eval raster)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def success_prediction_rate(eval_raster, test_pts, thresholds=None, out_path=None):
    """The two missing field-standard SDM curves (Chung & Fabbri 2003's
    success/prediction-rate framing, computed here against a genuinely
    held-out point set + raster rather than training data):
      - success rate:    fraction of TEST presence points (`test_pts`)
                         scoring >= threshold.
      - prediction rate: fraction of the EVAL raster's ENTIRE valid-AOI
                         pixel population scoring >= threshold ("how much
                         of the map you would have had to flag as
                         susceptible to hit that capture rate").

    Both against `eval_raster` (default caller should pass ``EVAL_RASTER``)
    -- deliberately never ``FINAL_RASTER``, so this figure cannot
    reintroduce the train/test leakage the rest of this module prevents.

    `thresholds` defaults to a 200-point grid spanning the raster's own
    actual valid-pixel value range (mirrors the original notebook's
    ``np.linspace(0.01, 0.99, 200)`` in spirit, but derived from the real
    range rather than a hardcoded presumed [0,1] span).

    Returns a DataFrame (threshold, success_rate, prediction_rate) and
    writes a two-panel PNG to `out_path` (default ``VALIDATION_DIR /
    "success_prediction_rate.png"``).
    """
    raster_da = _open_raster(eval_raster)
    arr = raster_da.values
    valid_vals = arr[np.isfinite(arr)]

    presence_scores = _sample_xy(raster_da, test_pts["x"].to_numpy(), test_pts["y"].to_numpy())
    presence_scores = presence_scores[np.isfinite(presence_scores)]

    if thresholds is None:
        lo, hi = float(valid_vals.min()), float(valid_vals.max())
        thresholds = np.linspace(lo, hi, 200)

    success_rate, prediction_rate = _success_prediction_rate_arrays(valid_vals, presence_scores, thresholds)
    df = pd.DataFrame({"threshold": thresholds, "success_rate": success_rate,
                        "prediction_rate": prediction_rate})

    out_path = Path(out_path) if out_path is not None else VALIDATION_DIR / "success_prediction_rate.png"
    _plot_success_prediction_rate(df, out_path)

    return df


# ============================================================================
# classed_map -- a density-classed susceptibility map from the FINAL
# (published, all-277-point) raster. This is a cartographic product of the
# model actually published, not a held-out-evaluation artifact -- no test-
# point label is involved, so there is no train/test leakage concern in using
# FINAL_RASTER here (unlike oos_auc/success_prediction_rate, which must
# never touch it).
# ============================================================================

def _digitize(vals, breaks, n_classes):
    """Assign each of `vals` to a class index in ``0..n_classes-1`` given
    `breaks` (an ``n_classes+1`` sorted bin-edge array) -- the ONE
    searchsorted/clip formula ``_classify_values`` (counting) and
    ``classed_map`` (painting the 2-D class map) both call, so it is
    never written out twice and cannot silently diverge between the two
    use sites."""
    idx = np.searchsorted(breaks, vals, side="right") - 1
    return np.clip(idx, 0, n_classes - 1)


def _classify_values(vals, n_classes=5, method="quantile"):
    """Pure-numpy classification core (unit-testable against a small
    synthetic array directly, no raster I/O):
      - "quantile" (default, always available -- pure numpy): `n_classes`
        equal-COUNT bins (``np.quantile`` linear interpolation).
      - "natural_breaks": Jenks natural breaks (equal within-class
        variance minimization) via a LAZY ``mapclassify`` import --
        `mapclassify` is not a hard dependency in environment.yml, so this
        path is only exercised if explicitly requested.

    Returns ``(breaks, counts_per_class)``: `breaks` an ``n_classes+1``
    ndarray of bin edges (forced to exactly `vals`' own min/max at the two
    ends, avoiding float round-off excluding the extremes), `counts_per_
    class` an ``n_classes``-length int ndarray of how many `vals` fall in
    each class.
    """
    vals = np.asarray(vals, dtype="float64")

    if method == "quantile":
        breaks = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    elif method == "natural_breaks":
        import mapclassify
        nb = mapclassify.NaturalBreaks(vals, k=n_classes)
        breaks = np.concatenate([[vals.min()], np.asarray(nb.bins, dtype="float64")])
    else:
        raise ValueError(f"[evaluate] _classify_values: unknown method {method!r}")

    breaks = np.asarray(breaks, dtype="float64")
    breaks[0] = vals.min()
    breaks[-1] = vals.max()

    class_idx = _digitize(vals, breaks, n_classes)
    counts_per_class = np.bincount(class_idx, minlength=n_classes)

    return breaks, counts_per_class


def _plot_classed_map(class_idx, labels, breaks, out_path):
    n_classes = len(labels)
    cmap = plt.get_cmap("YlOrRd", n_classes)

    fig, ax = plt.subplots(figsize=(8, 8))
    masked = np.ma.masked_less(class_idx, 0)   # -1 sentinel = outside AOI
    im = ax.imshow(masked, cmap=cmap, vmin=-0.5, vmax=n_classes - 0.5)
    ax.set_title("Density-classed flood susceptibility (published map)", fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, ticks=range(n_classes), shrink=0.8)
    cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def classed_map(final_raster=FINAL_RASTER, n_classes=5, method="quantile", labels=None, out_path=None):
    """A density-classed susceptibility map -- the standard SDM/
    landslide-and-flood-susceptibility-mapping cartographic product
    (default 5 classes: Very Low..Very High), binned from the FINAL
    (published, all-277-point) raster's valid AOI pixels.

    `method`: "quantile" (default, equal pixel COUNT per class, pure
    numpy) or "natural_breaks" (Jenks, lazy `mapclassify` import -- see
    `_classify_values`).

    Returns ``(breaks, class_counts)`` -- `breaks` the `n_classes`+1 bin
    edges, `class_counts` a ``{label: pixel_count}`` dict. Writes a PNG to
    `out_path` (default ``VALIDATION_DIR / "classed_susceptibility_map.
    png"``).
    """
    if labels is None:
        labels = DEFAULT_CLASS_LABELS[:n_classes] if n_classes <= len(DEFAULT_CLASS_LABELS) \
            else [f"Class {i + 1}" for i in range(n_classes)]
    assert len(labels) == n_classes, (
        f"[evaluate] classed_map: labels length {len(labels)} != n_classes {n_classes}"
    )

    da = _open_raster(final_raster)
    arr = da.values.astype("float64")
    valid = np.isfinite(arr)
    vals = arr[valid]

    breaks, counts_per_class = _classify_values(vals, n_classes=n_classes, method=method)

    class_idx = np.full(arr.shape, -1, dtype="int16")
    class_idx[valid] = _digitize(vals, breaks, n_classes)

    class_counts = {labels[i]: int(counts_per_class[i]) for i in range(n_classes)}

    out_path = Path(out_path) if out_path is not None else VALIDATION_DIR / "classed_susceptibility_map.png"
    _plot_classed_map(class_idx, labels, breaks, out_path)

    return breaks, class_counts


# ============================================================================
# External comparisons (DIRECTIONAL CONSISTENCY, never "validated against")
# + the GFI terrain-only baseline recomputed on the FIXED (WhiteboxTools)
# flow_accumulation.
#
# Ports the original notebook's Sec.3 (FEMA), Sec.5 (GFI), Sec.6 (NWS AHPS
# FIM), Sec.7 (Sentinel-1 SAR) -- with two changes over the source notebook:
#
#   1. FRAMING: every dict this section returns carries
#      ``kind=DIRECTIONAL_KIND`` ("directional_consistency", never
#      "validation"), an explicit "NOT independent validation" disclaimer
#      (``DIRECTIONAL_FRAMING``), and a shared urban-pluvial ``scope_caveat``
#      -- the SAME string on FEMA, NWS, and S1 alike. This mirrors
#      ``screen.bivariate()``'s own ``kind``/``not_validation``/``framing``
#      convention -- this pipeline's established pattern for exactly this
#      kind of caveat. The source notebook scoped these comparisons
#      INCONSISTENTLY ("urban-pluvial not riverine" to excuse low FEMA
#      agreement, then "riverine-focused" to excuse low Sentinel-1
#      agreement) -- reusing one literal caveat string everywhere is the
#      fix, not a paraphrase per product.
#
#   2. GFI ON THE FIXED ACCUMULATION: ``gfi_baseline()`` reads
#      ``data/processed/predictors/flow_accumulation.tif`` as it stands
#      today -- the WhiteboxTools, flats-resolving accumulation (max ~5.5M
#      cells; the OLD ``xarray-spatial`` chain capped at 2,451). The source
#      notebook built its GFI "terrain-only baseline" from that capped
#      raster, so its claim that pure terrain indices are uninformative
#      (AUC ~= 0.50) partly rested on a routing bug, not hydrology. This
#      recomputes the IDENTICAL formula (``predictors.gfi``, unchanged) on
#      the FIXED input and reports whatever AUC actually comes out -- see
#      ``gfi_baseline``'s own docstring. Neither forced toward "still
#      uninformative" nor toward "now discriminates".
#
# All three external products (FEMA NFHL, NWS AHPS FIM, Sentinel-1 SAR) are
# COMPARISON layers only -- never predictors, never independent ground truth
# for this presence-only model (see DIRECTIONAL_FRAMING below).
# ============================================================================

FEMA_GPKG = config.REPO / "data" / "raw" / "fema" / "flood_hazard_zones.gpkg"
NWS_DIR = config.REPO / "data" / "raw" / "nws"
NWS_CATEGORIES = ("low", "near", "minor", "moderate", "major")

FLOW_ACCUMULATION_TIF = config.DATA / "predictors" / "flow_accumulation.tif"
SLOPE_TIF = config.DATA / "predictors" / "slope.tif"
FINAL_RESULTS_CSV = maxent.FINAL_DIR / "cloglog" / "maxentResults.csv"

# FEMA Special Flood Hazard Area zone codes (1% AEP) -- matches the original
# notebook's own SFHA_ZONES set exactly (that set literally listed
# 'AE'/'AO' twice; a frozenset already dedupes, so this is the identical 7
# codes, not a behavior change).
FEMA_SFHA_ZONES = frozenset({"A", "AE", "AH", "AO", "VE", "A99", "AR"})

DIRECTIONAL_KIND = "directional_consistency"

DIRECTIONAL_FRAMING = (
    "Spatial agreement (Jaccard/IoU, hit rate, precision, F1) between the "
    "thresholded MaxEnt susceptibility map and an external flood product "
    "(FEMA NFHL flood hazard zones, NWS AHPS Flood Inundation Maps, or "
    "Sentinel-1 SAR change detection). This is directional consistency, "
    "explicitly NOT independent validation: none of these three products "
    "is ground truth for this presence-only model, none is used as a "
    "predictor, and agreement or disagreement with them is never reported "
    "as a validation claim -- only as directional support for (or against) "
    "the map's plausibility."
)

URBAN_PLUVIAL_SCOPE_CAVEAT = (
    "Scope caveat (applied identically to FEMA, NWS, and Sentinel-1): our "
    "HWM presence set includes urban pluvial (stormwater) flooding that "
    "these riverine-oriented external products do not represent. "
    "Disagreement away from mapped stream channels is expected under this "
    "scope difference and is not, by itself, evidence the susceptibility "
    "map is wrong -- the same caveat applies to every external product "
    "compared here, never a different excuse per product."
)


# ----------------------------------------------------------------------
# Spatial agreement primitive -- ported (identical arithmetic) from the
# original notebook's `spatial_overlap`.
# ----------------------------------------------------------------------

def spatial_overlap(pred_binary, ref_binary, valid_mask=None):
    """Confusion-matrix-derived Jaccard/IoU + hit-rate (recall) + precision
    + F1 between two binary masks -- direct port of the original notebook's
    `spatial_overlap` (same tp/tn/fp/fn/jaccard/hit_rate/precision/f1
    formulas). `valid_mask` restricts the comparison to a footprint (e.g.
    a raster's own finite-AOI mask) -- defaults to "everywhere" if omitted
    (only useful for a small hand-computed toy where every cell counts).

    Returns a dict: jaccard, hit_rate, precision, f1 (floats, NaN wherever
    the denominator is 0 -- matches the original notebook's own `np.nan`
    guard here, deliberately NOT `threshold_metrics`' 0.0 guard above,
    since a 0 denominator here means "reference/prediction footprint is
    empty at this operating point", a different situation from
    `threshold_metrics`' "no pixel predicted positive") plus the four raw
    confusion counts tp/tn/fp/fn (Python ints).
    """
    p = np.asarray(pred_binary, dtype=bool)
    r = np.asarray(ref_binary, dtype=bool)
    v = np.ones_like(p, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)

    tp = int(np.sum(p & r & v))
    tn = int(np.sum(~p & ~r & v))
    fp = int(np.sum(p & ~r & v))
    fn = int(np.sum(~p & r & v))

    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

    return dict(jaccard=jaccard, hit_rate=hit_rate, precision=precision, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def _rasterize_binary(gdf, ref_da):
    """Burn a GeoDataFrame's geometries onto `ref_da`'s grid as a boolean
    mask -- generalizes the original notebook's `rasterize_fema`/NWS
    per-category rasterization (`rasterize([(geom, 1) for geom in
    gdf.geometry], ...)`) to any reference DataArray + any external layer.
    Empty/`None` `gdf` returns an all-False mask of `ref_da`'s shape rather
    than raising -- a product with zero features inside the AOI is a valid
    (if uninteresting) comparison state, not an error.
    """
    shape = ref_da.rio.shape
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=bool)

    if gdf.crs is not None and ref_da.rio.crs is not None:
        assert gdf.crs.to_epsg() == ref_da.rio.crs.to_epsg(), (
            f"[evaluate] _rasterize_binary: CRS mismatch -- gdf={gdf.crs} "
            f"vs raster={ref_da.rio.crs}"
        )

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros(shape, dtype=bool)

    arr = rasterize(shapes, out_shape=shape, transform=ref_da.rio.transform(),
                     fill=0, dtype="uint8", all_touched=False)
    return arr.astype(bool)


# ----------------------------------------------------------------------
# Threshold sourcing for the FINAL (published) model -- the bootstrap-mean
# MaxSS matching the original notebook's own THRESH_MAXSS (`mx_res_df[
# maxss_col].iloc[:-1].mean()`), but selected the SAME way `apparent_auc()`
# selects its bootstrap rows (`Species` matching `flood_<int>`, explicitly
# excluding the trailing "flood (average)" summary row) rather than a
# positional `iloc[:-1]` -- robust to row order, matching this module's
# existing convention.
# ----------------------------------------------------------------------

def final_maxss_threshold(results_csv=FINAL_RESULTS_CSV):
    """The published (FINAL, all-277-point, 10-replicate bootstrap) model's
    own MaxSS operating threshold -- mean of the 10 bootstrap replicates'
    "Maximum training sensitivity plus specificity Cloglog threshold"
    (real on-disk value: ~0.1396). This is the threshold
    `external_comparisons`/`fema_comparison`/`nws_comparison`/
    `sentinel1_comparison` binarize `FINAL_RASTER` at by default -- the
    "thresholded MaxEnt map (final, at MaxSS)".
    """
    df = pd.read_csv(results_csv)
    reps = df[df["Species"].astype(str).str.match(r"^flood_\d+$")]
    if reps.empty:
        reps = df[df["Species"].astype(str) == "flood"]
    return float(reps[MAXSS_COL].astype(float).mean())


# ----------------------------------------------------------------------
# GFI terrain-only baseline, on the fixed flow accumulation.
# ----------------------------------------------------------------------

DEFAULT_GFI_CELL_AREA = 100.0   # 10 m x 10 m pixels -- matches
                                # predictors.flow_derivatives()'s own cell_area.


def gfi_baseline(flow_accumulation_tif=FLOW_ACCUMULATION_TIF, slope_tif=SLOPE_TIF,
                  presence=None, n_background=DEFAULT_BACKGROUND_N,
                  seed=config.GLOBAL_SEED, cell_area=DEFAULT_GFI_CELL_AREA):
    """The Geomorphic Flood Index (Samela et al. 2017),
    ``GFI = ln((flow_accumulation * cell_area) / tan(slope))``, recomputed
    on the FIXED (WhiteboxTools, flats-resolving) accumulation --
    NEVER the old `xarray-spatial` accumulation, which was capped at
    2,451 cells (river-scale flow never reached).

    Uses `predictors.gfi` verbatim (the flow-routing step's own formula,
    unchanged) -- this function's only job is to feed it the CURRENT on-disk
    `flow_accumulation.tif`/`slope.tif` (whatever the flow-routing step last
    wrote) and
    score the result, never to re-derive the formula a second time.
    `slope_tif` is stored in DEGREES on disk (`xrspatial.slope`'s own
    convention -- see `predictors._slope_rad`'s docstring) and is converted
    to radians here before calling `predictors.gfi` (which expects
    `slope_rad`).

    Presence defaults to `config.canonical_samples()` (the 277-point
    canonical modeling set); background is a fresh
    `sample_background` draw from the GFI surface's OWN valid footprint,
    seeded (default `config.GLOBAL_SEED`) for determinism -- matching every
    other background-sampling call in this module.

    **This is a terrain-ONLY baseline** -- no MaxEnt model is fit or
    touched here, so there is no train/test leakage risk to guard
    against (GFI is a closed-form function of two raster layers, evaluated
    directly at presence/background points -- it never sees a train/test
    split at all).

    Returns a dict:
      - `auc`             -- the honest headline number, reported as
                             observed and never forced toward either
                             verdict on whether terrain alone can
                             discriminate.
      - `n_presence`/`n_background` -- valid, finite-score counts actually
                             scored (after dropping any off-AOI/NaN draws).
      - `gfi_min`/`gfi_max` -- the recomputed GFI surface's own range.
      - `flow_accumulation_max` -- the gate value itself: must be
                             > 50,000 for this to genuinely be the FIXED
                             raster (the old broken chain topped out at
                             2,451) -- see
                             `tests/test_routing.py::test_real_aoi_reaches_
                             river_scale`, the same gate this function's
                             own test re-uses.
    """
    acc_da = _open_raster(flow_accumulation_tif)
    slope_da = _open_raster(slope_tif)
    assert acc_da.shape == slope_da.shape, (
        f"[evaluate] gfi_baseline: flow_accumulation shape {acc_da.shape} != "
        f"slope shape {slope_da.shape}"
    )

    slope_rad = np.deg2rad(slope_da.values)
    gfi_arr = np.asarray(predictors.gfi(acc_da.values, slope_rad, cell_area=cell_area),
                          dtype="float32")
    gfi_da = acc_da.copy(data=gfi_arr)

    if presence is None:
        presence = config.canonical_samples()

    background = sample_background(gfi_da, n=n_background, seed=seed)

    pres_scores = _sample_xy(gfi_da, presence["x"].to_numpy(), presence["y"].to_numpy())
    bg_scores = _sample_xy(gfi_da, background["x"].to_numpy(), background["y"].to_numpy())
    pres_scores = pres_scores[np.isfinite(pres_scores)]
    bg_scores = bg_scores[np.isfinite(bg_scores)]
    assert len(pres_scores) > 0, "[evaluate] gfi_baseline: no valid presence GFI scores"
    assert len(bg_scores) > 0, "[evaluate] gfi_baseline: no valid background GFI scores"

    y_true = np.concatenate([np.ones(len(pres_scores)), np.zeros(len(bg_scores))])
    y_score = np.concatenate([pres_scores, bg_scores])
    auc = float(roc_auc_score(y_true, y_score))

    return {
        "auc": auc,
        "n_presence": int(len(pres_scores)),
        "n_background": int(len(bg_scores)),
        "gfi_min": float(np.nanmin(gfi_arr)),
        "gfi_max": float(np.nanmax(gfi_arr)),
        "flow_accumulation_max": float(np.nanmax(acc_da.values)),
    }


# ----------------------------------------------------------------------
# FEMA NFHL -- directional spatial agreement (1% and 0.2% AEP zones).
# ----------------------------------------------------------------------

def fema_comparison(final_raster=FINAL_RASTER, fema_gpkg=FEMA_GPKG, threshold=None):
    """Spatial agreement between the thresholded FINAL susceptibility map
    (MaxSS, `final_maxss_threshold()` by default) and FEMA's National
    Flood Hazard Layer zones -- ported from the original notebook's
    Sec.3-4. FEMA is a COMPARISON layer, never a predictor and never
    independent ground truth for this model (see `DIRECTIONAL_FRAMING`).
    `fema_gpkg` (default `FEMA_GPKG`) is the already-clipped/reprojected
    (EPSG:26914, matching this project's modeling grid) local cache -- this
    function never re-downloads or re-clips the FEMA FIRM package itself.

    Two reference masks, matching the source notebook's own `zone_class`
    convention: `sfha_1pct` (Zone A/AE/AH/AO/VE/A99/AR -- the 1% Annual
    Exceedance Probability Special Flood Hazard Area) and `zone_x_0_2pct`
    (SFHA union Zone X shaded, i.e. `ZONE_SUBTY` containing "0.2 PCT" --
    the 0.2% AEP zone, matching the notebook's `fema_arr >= 1`).

    Returns a dict: `threshold`, `sfha_1pct`/`zone_x_0_2pct` (each a
    `spatial_overlap` result dict), `kind`, `framing`, `scope_caveat`.
    """
    raster_da = _open_raster(final_raster)
    arr = raster_da.values
    valid = np.isfinite(arr)

    if threshold is None:
        threshold = final_maxss_threshold()
    pred_bin = (arr >= threshold) & valid

    gdf = gpd.read_file(fema_gpkg)
    is_sfha = gdf["FLD_ZONE"].astype(str).isin(FEMA_SFHA_ZONES)
    if "ZONE_SUBTY" in gdf.columns:
        is_x02_shaded = (gdf["FLD_ZONE"].astype(str) == "X") & \
            gdf["ZONE_SUBTY"].astype(str).str.upper().str.contains("0.2 PCT", na=False)
    else:
        is_x02_shaded = pd.Series(False, index=gdf.index)

    sfha_bin = _rasterize_binary(gdf[is_sfha], raster_da)
    x02_bin = _rasterize_binary(gdf[is_sfha | is_x02_shaded], raster_da)

    return {
        "threshold": threshold,
        "sfha_1pct": spatial_overlap(pred_bin, sfha_bin, valid),
        "zone_x_0_2pct": spatial_overlap(pred_bin, x02_bin, valid),
        "kind": DIRECTIONAL_KIND,
        "framing": DIRECTIONAL_FRAMING,
        "scope_caveat": URBAN_PLUVIAL_SCOPE_CAVEAT,
    }


# ----------------------------------------------------------------------
# NWS AHPS Flood Inundation Maps -- directional spatial agreement.
# ----------------------------------------------------------------------

def nws_comparison(final_raster=FINAL_RASTER, nws_dir=NWS_DIR,
                    categories=NWS_CATEGORIES, threshold=None):
    """Spatial agreement between the thresholded FINAL susceptibility map
    and each NWS AHPS Flood Inundation Map category (Low/Near/Minor/
    Moderate/Major -- Wildcat Creek at Scenic Drive, gauge MWCK1) -- ported
    from the original notebook's Sec.6. Cached locally at
    `data/raw/nws/fim_<category>.gpkg` (already reprojected to this
    project's modeling CRS, EPSG:26914 -- see `_rasterize_binary`'s own CRS
    assertion); this function reads that cache only, it never re-fetches
    from the NWS REST service (that fetch/cache step already ran once and
    is out of scope to repeat here).

    Returns a dict: `threshold`, `categories` (`{category: spatial_overlap
    result dict}`, or `{"status": "missing_file", "path": ...}` for any
    category whose cache is absent -- never a raised error, matching the
    notebook's own per-category tolerance for a failed fetch), `kind`,
    `framing`, `scope_caveat`.
    """
    raster_da = _open_raster(final_raster)
    arr = raster_da.values
    valid = np.isfinite(arr)

    if threshold is None:
        threshold = final_maxss_threshold()
    pred_bin = (arr >= threshold) & valid

    nws_dir = Path(nws_dir)
    results = {}
    for cat in categories:
        gpkg_path = nws_dir / f"fim_{cat}.gpkg"
        if not gpkg_path.exists():
            results[cat] = {"status": "missing_file", "path": str(gpkg_path)}
            continue
        gdf = gpd.read_file(gpkg_path)
        ref_bin = _rasterize_binary(gdf, raster_da)
        results[cat] = spatial_overlap(pred_bin, ref_bin, valid)

    return {
        "threshold": threshold,
        "categories": results,
        "kind": DIRECTIONAL_KIND,
        "framing": DIRECTIONAL_FRAMING,
        "scope_caveat": URBAN_PLUVIAL_SCOPE_CAVEAT,
    }


# ----------------------------------------------------------------------
# Sentinel-1 SAR change detection -- ATTEMPTED (real Planetary Computer
# fetch), gracefully SKIPPED (never raised) if the scenes/network are
# unavailable. Ported from the original notebook's Sec.7.
# ----------------------------------------------------------------------

S1_PRE_ID = "S1A_IW_GRDH_1SDV_20190510T002100_20190510T002125_027158_030FB5_rtc"
S1_POST_ID = "S1A_IW_GRDH_1SDV_20190522T002101_20190522T002126_027333_031530_rtc"
S1_FLOOD_THRESH_DB = -2.0
S1_MIN_CLUSTER_PX = 5
S1_PERMANENT_WATER_DB = -17.0
S1_BUFFER_M = 1500.0


def _s1_load_vv_db(item, ref_da):
    """Windowed COG read of one Sentinel-1 RTC scene's VV band (Planetary
    Computer `sentinel-1-rtc` collection), reprojected onto `ref_da`'s
    exact grid, in dB -- ported from the original notebook's `load_vv_db`.
    Reads only the AOI-overlapping window from the remote COG (no
    full-scene download). Local imports (`rasterio`, `pyproj`,
    `rasterio.warp`/`.windows`) -- this whole S1 path is optional/
    best-effort (see `sentinel1_comparison`), so nothing above it in this
    module pays for these imports at load time.
    """
    import rasterio
    from pyproj import Transformer as ProjTransformer
    from rasterio.warp import reproject as rio_reproject, Resampling
    from rasterio.windows import Window, from_bounds as window_from_bounds

    ref_crs = ref_da.rio.crs
    ref_transform = ref_da.rio.transform()
    ref_h, ref_w = ref_da.rio.shape
    left, bottom, right, top = ref_da.rio.bounds()

    href = item.assets["vv"].href
    with rasterio.open(href) as src:
        scene_crs = src.crs
        tr = ProjTransformer.from_crs(ref_crs, scene_crs, always_xy=True)
        xs = [left, right, left, right]
        ys = [bottom, bottom, top, top]
        sxs, sys_ = tr.transform(xs, ys)
        sc_minx, sc_maxx = min(sxs) - S1_BUFFER_M, max(sxs) + S1_BUFFER_M
        sc_miny, sc_maxy = min(sys_) - S1_BUFFER_M, max(sys_) + S1_BUFFER_M

        win = window_from_bounds(sc_minx, sc_miny, sc_maxx, sc_maxy, src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        arr_win = src.read(1, window=win, out_dtype="float32")
        win_transform = src.window_transform(win)

    dst_arr = np.zeros((ref_h, ref_w), dtype="float32")
    rio_reproject(
        source=arr_win, destination=dst_arr,
        src_transform=win_transform, src_crs=scene_crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        resampling=Resampling.bilinear, src_nodata=0,
    )
    dst_arr[dst_arr <= 0] = np.nan
    return 10 * np.log10(dst_arr)


def _s1_flood_mask(ref_da):
    """STAC search (Planetary Computer, `sentinel-1-rtc`) for the two
    known track-136 scenes bracketing the 2019-05-19 Wildcat Creek event
    (pre: 2019-05-10, post: 2019-05-22) + VV change detection -- ported
    from the original notebook's Sec.7 (change threshold < -2 dB, >=5 px
    connected-cluster despeckle filter, pre-event VV < -17 dB permanent-
    water exclusion). Raises on ANY failure (network, missing scene,
    reprojection error, ...) -- `sentinel1_comparison` (the only caller) is
    responsible for catching broadly and degrading to a graceful skip;
    this helper itself never swallows an error, so a real failure is
    always visible to whichever caller wants to see it.

    Returns `(flood_mask, valid_mask)`, two boolean ndarrays shaped like
    `ref_da`.
    """
    import pystac_client
    import planetary_computer
    from scipy import ndimage

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = {it.id: it for it in catalog.search(
        collections=["sentinel-1-rtc"], ids=[S1_PRE_ID, S1_POST_ID],
    ).items()}
    if len(items) < 2:
        raise RuntimeError(
            f"[evaluate] S1: only {len(items)}/2 expected scenes found on Planetary Computer"
        )

    vv_pre_db = _s1_load_vv_db(items[S1_PRE_ID], ref_da)
    vv_post_db = _s1_load_vv_db(items[S1_POST_ID], ref_da)

    delta_db = vv_post_db - vv_pre_db
    raw_flood = (delta_db < S1_FLOOD_THRESH_DB) & np.isfinite(delta_db)

    labeled, n_feat = ndimage.label(raw_flood)
    cluster_sizes = ndimage.sum(raw_flood, labeled, range(1, n_feat + 1))
    keep_labels = np.where(np.asarray(cluster_sizes) >= S1_MIN_CLUSTER_PX)[0] + 1
    cleaned = np.isin(labeled, keep_labels)

    permanent_water = (vv_pre_db < S1_PERMANENT_WATER_DB) & np.isfinite(vv_pre_db)
    flood_mask = cleaned & ~permanent_water
    valid = np.isfinite(delta_db)
    return flood_mask, valid


def sentinel1_comparison(final_raster=FINAL_RASTER, threshold=None):
    """Sentinel-1 SAR change-detection comparison for the 2019-05-22
    post-flood acquisition -- ATTEMPTED for real (a genuine Planetary
    Computer STAC search + windowed COG read + reprojection), never
    blocking the pipeline: any failure (network unreachable, scene(s) no
    longer on the catalog, timeout, missing optional dependency, ...) is
    caught broadly and returned as a `{"status": "skipped", "reason": ...}`
    dict rather than raised.

    On success, returns `{"status": "ok", "threshold", "event":
    "2019-05-22", jaccard/hit_rate/precision/f1/tp/tn/fp/fn, "kind",
    "framing", "scope_caveat"}`. On skip, returns `{"status": "skipped",
    "reason": <str>, "kind", "framing", "scope_caveat"}` -- the framing/
    caveat fields are present either way so a caller folding this into
    `external_comparisons()`'s combined metadata never has to special-case
    a missing key.
    """
    raster_da = _open_raster(final_raster)
    arr = raster_da.values

    if threshold is None:
        threshold = final_maxss_threshold()

    try:
        flood_mask, s1_valid = _s1_flood_mask(raster_da)
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
            "kind": DIRECTIONAL_KIND,
            "framing": DIRECTIONAL_FRAMING,
            "scope_caveat": URBAN_PLUVIAL_SCOPE_CAVEAT,
        }

    pred_bin = (arr >= threshold) & s1_valid
    metrics = spatial_overlap(pred_bin, flood_mask, s1_valid)

    return {
        "status": "ok",
        "threshold": threshold,
        "event": "2019-05-22",
        **metrics,
        "kind": DIRECTIONAL_KIND,
        "framing": DIRECTIONAL_FRAMING,
        "scope_caveat": URBAN_PLUVIAL_SCOPE_CAVEAT,
    }


# ----------------------------------------------------------------------
# Capture vs. area -- the comparison metric that IS interpretable between
# products with different definitions.
#
# `spatial_overlap` above answers "how much do these two footprints coincide",
# which is hard to read when the two layers represent different quantities (a
# regulatory 1%-AEP riverine envelope vs. a pluvial susceptibility ranking):
# a low Jaccard is equally consistent with a poor model and with two products
# that legitimately delineate different things.
#
# This section scores every layer on ONE common, interpretable pair of
# quantities instead: how much of the study area it flags, and how many
# independently surveyed high-water marks fall inside it. That is the design
# Mobley et al. (2019) use for the Hurricane Harvey MaxEnt study -- validate on
# independent occurrence data, and treat the regulatory product as a comparison
# whose coverage is measured on the same yardstick rather than as ground truth.
# ----------------------------------------------------------------------

def _mask_capture(mask, points, ref_da, valid):
    """Fraction of `valid` AOI cells that `mask` flags, and how many `points`
    (an x/y frame) fall inside it. Points are located with the SAME
    rowcol/floor convention every other sampler in this pipeline uses."""
    import numpy as _np
    from rasterio.transform import rowcol as _rowcol

    area_frac = float((mask & valid).sum() / valid.sum())
    rows, cols = _rowcol(ref_da.rio.transform(),
                         points["x"].to_numpy(), points["y"].to_numpy())
    rows, cols = _np.asarray(rows), _np.asarray(cols)
    inb = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    hit = _np.zeros(len(points), dtype=bool)
    hit[inb] = mask[rows[inb], cols[inb]]
    return area_frac, int(hit.sum()), int(len(points))


def capture_vs_area(final_raster=None, fema_gpkg=FEMA_GPKG,
                     points=None, thresholds=None, out_csv=None,
                     results_csv=None):
    """Score the susceptibility map and the FEMA flood-hazard zones on one
    common yardstick: **area flagged** vs. **surveyed high-water marks
    captured**.

    Both quantities are interpretable on their own and comparable between
    layers that mean different things, which Jaccard is not. The ratio
    (capture per unit area) expresses how efficiently each layer concentrates
    observed flooding.

    **Both halves of the comparison are kept out of sample.** `points` defaults
    to the HELD-OUT test partition of the canonical presence set, and
    `final_raster` defaults to ``EVAL_RASTER`` -- the TRAIN-ONLY model -- not
    the published all-277 surface. Using the published raster here would
    reproduce the resubstitution error this module exists to prevent: it was fitted on the
    very marks whose capture is being counted, so its capture rate would be
    resubstitution while FEMA's is not, and the comparison would flatter the
    model by construction. Pass ``FINAL_RASTER`` with
    ``points=config.canonical_samples()`` if the published surface's in-sample
    coverage is wanted, and label it as such.

    `thresholds` is ``{label: value}`` for the susceptibility raster; defaults
    to the published model's MaxSS and 10th-percentile operating points.

    Returns a DataFrame (source, area_fraction, n_captured, n_points,
    capture_fraction, capture_per_area) and writes it to `out_csv`
    (default ``VALIDATION_DIR/capture_vs_area.csv``).

    NOTE the interpretation limit: the NFHL is a regulatory 1 %-annual-
    -exceedance riverine product with a different purpose, definition and
    epoch. A lower capture rate is a statement about what it represents, not
    a deficiency of the NFHL, and the marks are themselves surveyed where
    crews have access.
    """
    import geopandas as _gpd

    if final_raster is None:
        final_raster = EVAL_RASTER            # train-only; see the docstring
    if results_csv is None:
        results_csv = EVAL_RESULTS_CSV if Path(final_raster) == Path(EVAL_RASTER) \
            else FINAL_RESULTS_CSV

    ref = _open_raster(final_raster)
    arr = ref.values
    valid = np.isfinite(arr)

    if points is None:
        samples = config.canonical_samples()
        _train_idx, test_idx = config.split(samples)
        points = samples.iloc[test_idx][["x", "y"]]
    if thresholds is None:
        thresholds = {
            "This model (MaxSS)": maxss_threshold_from_results(results_csv),
            "This model (10th percentile)": p10_threshold_from_results(results_csv),
        }

    rows = []
    gdf = _gpd.read_file(fema_gpkg)
    is_sfha = gdf["FLD_ZONE"].astype(str).isin(FEMA_SFHA_ZONES)
    if "ZONE_SUBTY" in gdf.columns:
        is_x02 = (gdf["FLD_ZONE"].astype(str) == "X") & \
            gdf["ZONE_SUBTY"].astype(str).str.upper().str.contains("0.2 PCT", na=False)
    else:
        is_x02 = pd.Series(False, index=gdf.index)

    for label, subset in [("FEMA SFHA (1% AEP)", gdf[is_sfha]),
                          ("FEMA incl. Zone X (0.2% AEP)", gdf[is_sfha | is_x02])]:
        mask = _rasterize_binary(subset, ref)
        a, n, tot = _mask_capture(mask, points, ref, valid)
        rows.append(dict(source=label, area_fraction=a, n_captured=n, n_points=tot))

    for label, thr in thresholds.items():
        mask = (arr >= thr) & valid
        a, n, tot = _mask_capture(mask, points, ref, valid)
        rows.append(dict(source=f"{label}, thr={thr:.4f}", area_fraction=a,
                          n_captured=n, n_points=tot))

    df = pd.DataFrame(rows)
    df["capture_fraction"] = df["n_captured"] / df["n_points"]
    df["capture_per_area"] = df["capture_fraction"] / df["area_fraction"]

    out_csv = Path(out_csv) if out_csv is not None else VALIDATION_DIR / "capture_vs_area.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    for r in df.itertuples():
        print(f"[evaluate] {r.source:<38} area={r.area_fraction:6.1%}  "
              f"captured={r.n_captured:>3}/{r.n_points} ({r.capture_fraction:5.1%})  "
              f"per-area={r.capture_per_area:5.2f}")
    return df


# ----------------------------------------------------------------------
# external_comparisons -- the entry point: composes FEMA +
# NWS + Sentinel-1 into ONE result, all three sharing the SAME threshold
# (FINAL model's own MaxSS) and the SAME directional-consistency framing.
# ----------------------------------------------------------------------

def internal_validation_metrics(eval_raster=None, test_pts=None, background=None,
                                out_path=None):
    """``internal_validation_table`` for the canonical held-out set, persisted.

    Writes ``validation/internal_validation_metrics.csv`` so a clean rebuild
    regenerates it.

    Defaults to the TRAIN-ONLY ``EVAL_RASTER`` and the canonical test split --
    never ``FINAL_RASTER``, which the training points contributed to.
    """
    from pipeline import config

    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    if eval_raster is None:
        eval_raster = EVAL_RASTER
    if test_pts is None:
        test_pts = samples.iloc[test_idx][["x", "y"]]
    if background is None:
        background = sample_background(eval_raster, n=DEFAULT_BACKGROUND_N,
                                       seed=config.GLOBAL_SEED)
    df = internal_validation_table(eval_raster, test_pts, background)
    out_path = Path(out_path) if out_path is not None else         VALIDATION_DIR / "internal_validation_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[evaluate] wrote {out_path.name} ({len(df)} threshold rows)")
    return df


def gfi_comparison_sweep(final_raster=None, out_path=None, n_steps=40,
                         cell_area=DEFAULT_GFI_CELL_AREA):
    """Hit rate vs. flagged area for the terrain-only GFI and for MaxEnt, swept
    across thresholds and scored on the same held-out marks.

    Writes ``validation/gfi_comparison.csv``, which
    ``report._fig_gfi_comparison``'s second panel reads.

    The GFI surface is rebuilt exactly as ``gfi_baseline`` builds it -- same
    formula, same on-disk accumulation and slope -- so the two functions cannot
    diverge. Both surfaces are swept over their own percentile range, because
    the operationally meaningful axis is *area flagged*: a raw threshold is not
    comparable across two differently-scaled surfaces.
    """
    acc_da = _open_raster(FLOW_ACCUMULATION_TIF)
    slope_da = _open_raster(SLOPE_TIF)
    slope_rad = np.deg2rad(slope_da.values)
    gfi_arr = np.asarray(predictors.gfi(acc_da.values, slope_rad, cell_area=cell_area),
                         dtype="float32")

    samples = config.canonical_samples()
    _train_idx, test_idx = config.split(samples)
    pts = samples.iloc[test_idx][["x", "y"]]
    xs, ys = pts["x"].to_numpy(), pts["y"].to_numpy()

    maxent_path = Path(final_raster) if final_raster is not None else EVAL_RASTER
    surfaces = {
        "GFI": (gfi_arr, _sample_xy(acc_da.copy(data=gfi_arr), xs, ys)),
    }
    if maxent_path.exists():
        me_da = _open_raster(maxent_path)
        surfaces["MaxEnt"] = (me_da.values, _sample_xy(me_da, xs, ys))
    else:
        print(f"[evaluate] gfi_comparison_sweep: {maxent_path} missing, GFI only")

    rows = []
    for model, (surface, at_pts) in surfaces.items():
        finite = surface[np.isfinite(surface)]
        at_pts = at_pts[np.isfinite(at_pts)]
        for q in np.linspace(1, 99, n_steps):
            thr = float(np.percentile(finite, 100 - q))
            rows.append(dict(
                model=model,
                threshold=thr,
                pred_area_pct=100.0 * float((finite >= thr).mean()),
                hit_rate=float((at_pts >= thr).mean()) if len(at_pts) else float("nan"),
                n_points=int(len(at_pts)),
            ))
    df = pd.DataFrame(rows)
    out_path = Path(out_path) if out_path is not None else         VALIDATION_DIR / "gfi_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[evaluate] wrote {out_path.name} ({len(df)} rows, "
          f"{df['model'].nunique()} surfaces)")
    return df


def external_comparisons(final_raster=FINAL_RASTER, threshold=None):
    """All external directional-consistency comparisons for the PUBLISHED
    susceptibility map (`final_raster`, default `FINAL_RASTER`), thresholded
    once at its own MaxSS (`final_maxss_threshold()` unless `threshold` is
    given explicitly) and shared across FEMA/NWS/S1 so all three compare
    against the IDENTICAL binary map -- never a different threshold per
    product.

    Returns `{"threshold", "fema": fema_comparison(...), "nws":
    nws_comparison(...), "s1": sentinel1_comparison(...), "kind",
    "framing", "scope_caveat"}`. See each sub-function's own docstring for
    its result shape; see `DIRECTIONAL_FRAMING`/`URBAN_PLUVIAL_SCOPE_CAVEAT`
    for the caveats repeated at every level (top-level AND inside each
    product's own sub-dict) so a caller inspecting any one piece in
    isolation still sees the full disclaimer, not just the top-level
    wrapper.
    """
    if threshold is None:
        threshold = final_maxss_threshold()

    result = {
        "threshold": threshold,
        "fema": fema_comparison(final_raster, threshold=threshold),
        "nws": nws_comparison(final_raster, threshold=threshold),
        "s1": sentinel1_comparison(final_raster, threshold=threshold),
        "kind": DIRECTIONAL_KIND,
        "framing": DIRECTIONAL_FRAMING,
        "scope_caveat": URBAN_PLUVIAL_SCOPE_CAVEAT,
    }

    # Persist one CSV per product, as a THRESHOLD SWEEP with the label column
    # report._table_external_directional_comparison() groups on
    # ("fema_zone" / "nws_category" / none).
    METRIC_KEYS = ("jaccard", "hit_rate", "precision", "f1", "tp", "fp", "tn", "fn")
    sweep = sorted({0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, round(threshold, 5)})

    def _rows_for(fn, label_col, unpack):
        rows = []
        for thr in sweep:
            payload = fn(final_raster, threshold=thr)
            for label, metrics in unpack(payload):
                if not isinstance(metrics, dict):
                    continue
                row = {k: metrics.get(k) for k in METRIC_KEYS}
                row["threshold"] = thr
                if label_col:
                    row[label_col] = label
                rows.append(row)
        return rows

    specs = (
        ("fema", fema_comparison, "fema_zone",
         lambda d: [("SFHA_1pct", d.get("sfha_1pct")),
                    ("ZoneX02pct", d.get("zone_x_0_2pct"))]),
        ("nws", nws_comparison, "nws_category",
         lambda d: list((d.get("categories") or {}).items())),
        ("s1", sentinel1_comparison, None, lambda d: [(None, d)]),
    )
    for name, fn, label_col, unpack in specs:
        try:
            df = pd.DataFrame(_rows_for(fn, label_col, unpack))
        except Exception as exc:
            print(f"[evaluate] external_comparisons: {name} sweep failed "
                  f"({type(exc).__name__}: {exc}); not written")
            continue
        if df.empty or df["jaccard"].isna().all():
            print(f"[evaluate] external_comparisons: {name} produced no comparable "
                  f"rows (product may be unavailable); not written")
            continue
        path = VALIDATION_DIR / f"{name}_spatial_comparison.csv"
        df.to_csv(path, index=False)
        print(f"[evaluate] wrote {path.name} ({len(df)} rows over "
              f"{len(sweep)} thresholds)")

    return result

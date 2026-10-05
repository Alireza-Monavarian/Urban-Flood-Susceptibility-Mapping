"""MaxEnt run wrapper + AICc-based tuning grid search.

Ports the original notebook ``04_maxent``'s §1 (export predictor stack + presence
samples to MaxEnt's native ASCII-grid format), §2 (AICc helper functions --
already validated by the source notebook against a hand-computed toy), and
§3 (the tuning grid, 15 candidates as of the log-transform adoption --
originally 12; see "Log-transform adoption" below) into an importable
pipeline stage.

Fix vs. source: Java/jar resolution is
**OS-agnostic** -- every MaxEnt invocation below calls
``config.java_exe()``/``config.maxent_jar()`` (env override, then
``shutil.which("java")`` on PATH, then a ``tools/`` fallback -- see
``pipeline/config.py``) instead of the source notebook's hardcoded Windows
path (``tools/jdk-17.0.19+10/bin/java.exe``), which does not exist on macOS
or any Linux CI runner. The command line itself also
switches from ``-jar <maxent.jar>`` to ``-cp <maxent.jar> density.MaxEnt``
(the jar's own ``META-INF/MANIFEST.MF`` confirms ``Main-Class:
density.MaxEnt``).

**Canonical modeling stack (17 layers, not the source's 21):** the source
notebook modeled the 21-layer set (22 raw minus ``median_income``). This
project's multicollinearity screen (``pipeline.screen``) drops
``tri``/``tpi`` (Pearson |r| >= 0.8 against ``slope``/``curvature`` -- the
source's own (pre-D8-fix) hydrology never triggered a re-screen against
this pair) and, since the log-transform adoption ("Log-transform
adoption" above), ALSO ``spi`` (Stage 1: log10(``flow_accumulation``) vs
log1p(``spi``) now correlates r=0.81, crossing the 0.8 threshold the raw
values' r=0.79 stayed just under) and ``flow_accumulation`` (Stage 2: VIF,
once ``flow_accumulation`` shares a log-like scale with ``twi``'s own
``ln(flow_accumulation/cell_width * tan(slope))`` definition) -- 4 drops total, none of it hardcoded here; see ``screen.py``'s
module docstring and ``tests/test_screen.py``'s ``EXPECTED_KEPT``.

**TWI retention:** of the ``flow_accumulation``/``twi`` pair Stage 2 must resolve, the
project owner's standing preference is to KEEP ``twi`` (the canonical
topographic wetness index in flood-susceptibility literature -- it
integrates slope directly) and drop ``flow_accumulation`` -- the reverse of
which specific layer this Stage-2 cascade used to remove. ``screen.
_choose_vif_drop`` implements the swap (consulting ``screen.
_KNOWN_PAIR_RATIONALE``'s ``flow_accumulation``/``twi`` entry); this
module's ``modeling_layers()`` needed NO code change at all -- it already
gets its ``kept`` list fresh from ``screen.multicollinearity()`` every
call, so the new 17-layer stack (``twi`` present, ``flow_accumulation``
absent) flows through automatically.

``modeling_layers()`` below gets that 18-name ``kept`` list FRESH from
``screen.multicollinearity()`` every call (never hardcodes it) and drops
``median_income`` from it, producing the 17-layer canonical modeling
stack. Because the predictor
count and valid presence-point count (``config.canonical_samples()`` ->
277 pixel-unique, not the source's 278 -- see "Canonical n=277" below and
``pipeline/config.py``) both differ from the source notebook's run,
**the AICc grid winner in this project may legitimately differ from the
source's recorded ``beta=1, L``** -- this module computes AICc fresh from
whatever the real MaxEnt subprocess actually returns for the real canonical
stack, never fabricating or forcing a particular winner (see
``aicc_grid()``/``select_best()`` below).

**Canonical n=277, pixel-unique:** ``config.canonical_samples()`` returns 277 rows, not 279 --
of the 279 predictor-NaN-valid presence points (``predictors.valid_mask()``,
UNCHANGED), 2 pairs land in the same 10 m modeling-grid pixel as each other,
so only 277 pixels are actually distinct. ``config.canonical_samples()``
now deduplicates to exactly one point per pixel (deterministic --
``config.py``'s own docstring), so ``config.split()``/``spatial_folds()``/
``loeo_folds()`` and this module's ``build_samples_csv()`` (below) all
agree on n=277 -- MaxEnt's own historical same-cell collapsing (which is
what silently produced n=277 in every prior grid run here, from a
286-point or 279-point input) becomes a no-op rather than the mechanism
that determines ``n``.

**Log-transform adoption:** an ad-hoc experiment found the
canonical grid's raw-stack winner (``beta=3, LQHP``, 27 hinges) was largely
a SCALING ARTIFACT of ``flow_accumulation``/``spi`` each spanning 6+ orders
of magnitude. ``export_layers()`` below now passes every layer through
``pipeline.predictors.apply_modeling_transform()`` (log10 for
``flow_accumulation``, log1p for ``spi``, identity elsewhere) before
writing its ``.asc`` grid, and ``modeling_layers()``'s ``screen.
multicollinearity()`` call picks up the SAME transform automatically
(``pipeline.screen`` applies it internally for Stage 1/2 -- see that
module's docstring) -- so the ``kept`` list, the ``.asc`` grids, and the
AICc grid below are all computed on one consistent, log-scaled hydrology
block, never a mix of transformed and raw views of the same two layers.
``BETAS`` also gained a 5th candidate (``0.25``) as a diagnostic check that
the new winner isn't merely sitting at the low edge of the original grid.

**Inputs:** ``data/processed/predictors_screened/*.tif`` (the screen's 18
screened layers, post log-transform-adoption re-screen + TWI retention) +
``config.canonical_samples()`` (the 277-row, predictor-NaN-gated AND
modeling-grid-pixel-deduplicated presence set -- see ``build_samples_csv()``
below and ``config.canonical_samples()``'s docstring; a REVERSAL of
this module's previous behavior, which fed MaxEnt the full undeduped
286-row ``hwm.build_presence()`` set and relied on MaxEnt's own NODATA/
same-cell dropping to arrive at n=277 implicitly).

**Outputs (build artifacts under ``data/processed/maxent/`` -- gitignored,
see ``.gitignore``):**
``layers/*.asc`` (ASCII grids), ``samples.csv`` (MaxEnt presence file),
``tuning/b<beta>_<fc>/`` (one dir per grid candidate: ``flood.lambdas``,
``flood_samplePredictions.csv``, ``run.log``, plus MaxEnt's own
``maxentResults.csv`` etc.), ``tuning/aicc_results.csv`` (the scored grid).

**Final vs. evaluation model families (§4 below), the fix for the
resubstitution-AUC bug:** two DISTINCT model families are fit at
one LOCKED configuration (``FINAL_CONFIG`` = beta=0.5, LQ -- an owner
decision, NOT the raw AICc grid's numerical winner; see ``FINAL_CONFIG``'s
own docstring for the full rationale) but on DIFFERENT point sets.
``fit_final`` trains on ALL 277 canonical points (``FINAL_DIR`` =
``data/processed/maxent/final/`` -- the published map; its own
resubstitution AUC is exposed via ``apparent_auc()``, never the headline
number). ``fit_eval`` trains on a train-SPLIT ONLY (``HOLDOUT_MODEL_DIR``
= ``data/processed/maxent/holdout_model/``) and ``fit_folds`` trains one
model per spatial-CV/LOEO fold on that fold's complement (``FOLDS_DIR`` =
``data/processed/maxent/folds/``) -- both assert ``qa.
assert_eval_model_excludes_test`` BEFORE any subprocess runs, so a model
that will be sampled at held-out points for an honest out-of-sample AUC
(``evaluate.oos_auc``) can never have trained on those same points.

**Jackknife + permutation importance (§5 below):** combines
three per-predictor importance metrics into one table.
``jackknife_gains()`` runs ~2N+1 (N=17 -> 35) non-bootstrap MaxEnt fits at
``FINAL_CONFIG`` -- the full 17-predictor model once (``gain_all``), each
16-predictor leave-one-out model (``gain_without``, via
``_build_layer_subset``'s symlinked reduced layer directories -- no raster
copies), and each single-predictor model (``gain_only``) -- reading
``Regularized training gain`` from each run's own ``maxentResults.csv``;
``unique_contribution = gain_all - gain_without`` isolates non-redundant
information. ``variable_importance_combined()`` then joins that against
``fit_final``'s bootstrap ``maxentResults.csv`` (permutation importance +
percent contribution, mean +/- sd across the 10 replicates) and optionally
``screen.bivariate()``'s CSV (MW rank-biserial r, continuous predictors only).
**The actual #1 predictor is whatever this real ~35-fit run produces --
never hardcoded to any name in advance.**
"""
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray

from pipeline import config, predictors, qa, screen

OUT_DIR = config.DATA / "maxent"
LAYERS_DIR = OUT_DIR / "layers"
TUNING_DIR = OUT_DIR / "tuning"
SAMPLES_PATH = OUT_DIR / "samples.csv"

# Safe sentinel: no real predictor value in this stack is anywhere near
# -9999 (matches the source notebook's own choice).
NODATA = -9999

# The two nominal-class-code layers MaxEnt must be told to treat as
# categorical via `togglelayertype=` at run time (MaxEnt's default is
# continuous for every environmental layer) -- same set `pipeline.screen`
# already excludes from Pearson/VIF for the identical reason.
CATEGORICAL = screen.CATEGORICAL

# The 4 non-feature rows every MaxEnt `.lambdas` file appends after its real
# feature rows -- normalizer/background-count/entropy bookkeeping, not
# fitted feature coefficients, so they must never contribute to `k`.
LAMBDAS_METADATA_ROWS = {
    "linearPredictorNormalizer", "densityNormalizer",
    "numBackgroundPoints", "entropy",
}

# The 15-candidate AICc tuning grid: 5 regularization multipliers x 3
# feature-class combinations. `autofeature=false` is always passed at run
# time so these flags are used exactly as given (MaxEnt's own
# sample-size-based automatic feature selection never overrides them).
# `0.25` was added for the log-transform adoption as a diagnostic: confirms the
# beta=0.5/LQ winner found on the log-transformed stack isn't merely
# sitting at the low edge of the original 0.5/1/2/3 grid.
BETAS = (0.25, 0.5, 1, 2, 3)
FEATURE_CLASSES = {
    "L":    dict(linear=True, quadratic=False, product=False, threshold=False, hinge=False),
    "LQ":   dict(linear=True, quadratic=True,  product=False, threshold=False, hinge=False),
    "LQHP": dict(linear=True, quadratic=True,  product=True,  threshold=False, hinge=True),
}

DEFAULT_MAXIMUM_BACKGROUND = 10_000
DEFAULT_TIMEOUT_S = 600

# Layers acquired for the EQUITY OVERLAY only -- never candidate predictors.
#
# ``median_income`` is a measure of social vulnerability and adaptive capacity,
# not a physical or infrastructural control on where water goes, and the study
# design reserves it for the equity analysis (``pipeline.equity``). Excluding it
# here is therefore a scope decision taken in advance, NOT a screening outcome:
# it is never entered into the multicollinearity screen, never reported as a
# dropped predictor, and never shown in the predictor-stack figure.
#
# Note the deliberate asymmetry with ``population_density``, which IS a
# predictor: population density is used as a proxy for urbanisation and
# development intensity -- a physical driver of runoff generation and of the
# drainage network's spatial configuration -- whereas income has no comparable
# mechanistic link to flood generation. ``pipeline.equity`` discloses that
# overlap explicitly via its ``coupled_with_predictor`` column, so the one
# variable that is both a predictor and an equity indicator is flagged as such
# rather than left implicit.
EQUITY_ONLY_LAYERS = ("median_income",)


# ============================================================================
# §1 -- Export the modeling predictor stack + presence samples to MaxEnt's
# native ASCII-grid / CSV formats.
# ============================================================================

def modeling_layers():
    """The 17-layer canonical MaxEnt modeling stack: ``pipeline.screen.
    multicollinearity()``'s 18-name screened ``kept`` list (16 continuous +
    ``hydrologic_soil_group`` + ``nlcd_landcover``), minus the
    ``EQUITY_ONLY_LAYERS`` (``median_income``), which is acquired for the
    equity overlay and was never a candidate predictor -- see that constant's
    own comment. 18 (not the pre-log-transform 20) because
    the log-transform adoption's re-screen additionally drops ``spi``/
    ``flow_accumulation`` (TWI retained -- see this module's docstring,
    "Canonical modeling stack" / "TWI retention").

    Recomputes the real two-stage screen fresh every call
    (``copy_rasters=False`` -- no raster-copy side effect, ~a few seconds
    of correlation/VIF compute over the real 22-layer stack) rather than
    trusting ``predictors_screened/``'s on-disk file list, so this can never
    silently drift from whatever ``pipeline.screen``'s real screen says is
    ``kept`` today. Sorted for a deterministic, reproducible layer order.
    """
    kept, _report = screen.multicollinearity(screen._load_stack(), copy_rasters=False)
    return sorted(name for name in kept if name not in EQUITY_ONLY_LAYERS)


def _ascii_header(ref, nodata=NODATA):
    """Build the shared 6-line ESRI ASCII grid header from a reference
    raster's affine transform -- every exported layer uses this SAME header
    (MaxEnt requires all environmental layers to share one grid geometry).
    """
    transform = ref.rio.transform()
    nrows, ncols = ref.shape
    xllcorner = transform.c
    yllcorner = transform.f + transform.e * nrows  # transform.e is negative (north-up)
    cellsize = transform.a
    return (
        f"ncols {ncols}\n"
        f"nrows {nrows}\n"
        f"xllcorner {xllcorner}\n"
        f"yllcorner {yllcorner}\n"
        f"cellsize {cellsize}\n"
        f"NODATA_value {nodata}\n"
    )


def export_layers(layers=None, pred_dir=None, out_dir=LAYERS_DIR, nodata=NODATA):
    """Export each modeling layer to an ESRI ASCII grid (``.asc``) under
    ``out_dir``, one file per layer, all sharing one header (``ncols``/
    ``nrows``/``xllcorner``/``yllcorner``/``cellsize``/``NODATA_value``).
    Every layer is first passed through ``pipeline.predictors.
    apply_modeling_transform()`` (log10 for ``flow_accumulation``, log1p
    for ``spi``, identity for every other layer -- log-transform adoption)
    BEFORE the NaN
    nodata cells are flattened to the ``.asc`` sentinel below, so the two
    power-law-skewed hydrology layers reach MaxEnt already log-scaled, the
    same transformed values ``pipeline.screen.multicollinearity()`` now
    screens on. NaN cells are then written as ``NODATA_value`` (default
    -9999); the two ``CATEGORICAL`` layers are written as integers
    (MaxEnt's ``togglelayertype=`` expects whole-number class codes), every
    other layer as ``%.6g`` floats. Idempotent: an ``.asc`` that already
    exists under ``out_dir`` is left untouched, not regenerated.

    ``layers`` defaults to ``modeling_layers()`` (the live 17-name
    canonical stack); ``pred_dir`` defaults to ``screen.SCREENED_DIR``
    (``data/processed/predictors_screened`` -- the screen's output,
    which already contains every name ``modeling_layers()`` can return,
    ``median_income`` included, since screening only ever ADDS a drop on
    top of what modeling then also removes).

    Returns the list of ``.asc`` Paths written (or already present), in the
    same (sorted) order as ``layers``.
    """
    if layers is None:
        layers = modeling_layers()
    if pred_dir is None:
        pred_dir = screen.SCREENED_DIR
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [pred_dir / f"{name}.tif" for name in layers]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"[maxent] export_layers: missing predictor raster(s) under {pred_dir}: {missing}"
        )

    ref = rioxarray.open_rasterio(files[0]).squeeze("band", drop=True)
    header = _ascii_header(ref, nodata)

    written = []
    for f in files:
        out_path = out_dir / f"{f.stem}.asc"
        written.append(out_path)
        if out_path.exists():
            continue
        t0 = time.time()
        da = rioxarray.open_rasterio(f).squeeze("band", drop=True)
        arr = da.values.astype("float64")
        arr = predictors.apply_modeling_transform(f.stem, arr)
        arr = np.nan_to_num(arr, nan=nodata)
        if f.stem in CATEGORICAL:
            arr = np.where(arr == nodata, nodata, np.round(arr)).astype("int64")
            fmt = "%d"
        else:
            fmt = "%.6g"
        with open(out_path, "w") as fh:
            fh.write(header)
            np.savetxt(fh, arr, fmt=fmt)
        print(f"[maxent] exported {f.stem}: {time.time() - t0:.1f}s -> {out_path.name} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return written


def build_samples_csv(out_path=SAMPLES_PATH, presence=None, species="flood"):
    """Write MaxEnt's presence samples file (``species,X,Y``) from the
    canonical presence points -- ``presence`` defaults to
    ``config.canonical_samples()`` (the 277-row, predictor-NaN-gated AND
    modeling-grid-pixel-deduplicated set -- see that function's docstring).

    UPDATE (canonical n=277): this is a deliberate REVERSAL of the previous design,
    which defaulted to ``hwm.build_presence()``'s full 286-row thinned set
    (matching the source notebook's own ``pd.concat([hw, eow])``) and
    relied on MaxEnt dropping any point landing on a NODATA cell in any
    environmental layer, or sharing a grid cell with another point, at run
    time. That worked, but left ``n`` an IMPLICIT consequence of MaxEnt's
    own internal handling rather than a value this pipeline states and
    controls -- empirically it always converged to 277 (279 predictor
    -valid points, 2 pixels each holding 2 of them), but nothing upstream
    said so. Pre-filtering to ``config.canonical_samples()`` here makes
    MaxEnt's own same-cell/NODATA dropping a NO-OP (every point handed to
    it is already guaranteed valid and pixel-unique), so ``n=277`` is now
    guaranteed BEFORE the MaxEnt subprocess ever runs, not discovered
    after it -- and matches ``config.split()``/``spatial_folds()``/
    ``loeo_folds()``, which already operate on this same 277-row set.

    Returns the written path.
    """
    if presence is None:
        presence = config.canonical_samples()
    df = pd.DataFrame({
        "species": species,
        "X": presence["x"].to_numpy(dtype=float),
        "Y": presence["y"].to_numpy(dtype=float),
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[maxent] wrote {len(df)} presence points -> {out_path}")
    return out_path


# ============================================================================
# §2 -- AICc helpers (Warren & Seifert 2011 / ENMeval's aic.maxent),
# validated against a hand-computed toy in tests/test_maxent_aicc.py.
# ============================================================================

def _aicc(loglik, k, n):
    """AICc = 2k - 2*loglik + 2k(k+1)/(n-k-1) (Warren & Seifert 2011).

    ``k`` = number of feature parameters with non-zero lambda (see
    ``read_lambdas_k``); ``n`` = number of presence points MaxEnt actually
    used; ``loglik`` = sum of log(raw prediction) at those ``n`` points.
    Returns +inf if ``n - k - 1 <= 0`` (too few points for the parameter
    count to define a finite small-sample correction) rather than raising
    or dividing by a non-positive denominator -- a candidate this
    over-parameterized is simply worse than any finite-AICc candidate, so
    +inf sorts it last exactly as intended.
    """
    aic = 2 * k - 2 * loglik
    denom = n - k - 1
    if denom <= 0:
        return float("inf")
    return aic + (2 * k * (k + 1)) / denom


def read_lambdas(lambdas_path):
    """Parse a MaxEnt ``.lambdas`` file into its fitted feature coefficients.

    Returns ``{"linear": {var: (lam, mn, mx)}, "quadratic": {var: (lam, mn, mx)},
    "categorical": {var: {class_code: lam}}, "meta": {...}}``.

    MaxEnt writes one row per feature as ``name, lambda, min, max``, where
    ``min``/``max`` are the range it used to rescale that feature to [0, 1]
    (values outside are clamped). Feature names are the bare variable for a
    linear term, ``var^2`` for a quadratic term, and ``(var=code)`` for a
    one-hot indicator on a categorical layer. The 4 rows in
    ``LAMBDAS_METADATA_ROWS`` are bookkeeping, not features.
    """
    out = {"linear": {}, "quadratic": {}, "categorical": {}, "meta": {}}
    for line in Path(lambdas_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        if name in LAMBDAS_METADATA_ROWS:
            out["meta"][name] = float(parts[1])
            continue
        lam = float(parts[1])
        if name.startswith("(") and name.endswith(")") and "=" in name:
            var, code = name[1:-1].split("=", 1)
            out["categorical"].setdefault(var, {})[float(code)] = lam
        elif name.endswith("^2"):
            out["quadratic"][name[:-2]] = (lam, float(parts[2]), float(parts[3]))
        elif "*" in name or "'" in name or "`" in name or name.startswith("I("):
            # product / hinge / threshold features. FINAL_CONFIG fits linear +
            # quadratic only, so these must not appear; if the configuration
            # ever changes, fail loudly rather than silently dropping terms.
            raise ValueError(
                f"[maxent] read_lambdas: non-additive feature {name!r} in "
                f"{lambdas_path}; per-predictor decomposition is not valid"
            )
        else:
            out["linear"][name] = (lam, float(parts[2]), float(parts[3]))
    return out


def _clamp01(v, mn, mx):
    if mx == mn:
        return np.zeros(len(v))
    return np.clip((v - mn) / (mx - mn), 0.0, 1.0)


def exact_shap(X, lambdas_path, background=None):
    """Exact Shapley values for a MaxEnt fit, per predictor, in the space of
    the linear predictor (the exponent of MaxEnt's raw output).

    This is EXACT, not an approximation, and needs no sampling: at
    ``FINAL_CONFIG`` MaxEnt fits linear, quadratic and categorical-indicator
    features only, so the linear predictor is additive across predictors with
    no interaction terms (``read_lambdas`` raises if any appear). For an
    additive model the Shapley value of predictor *p* collapses to its own
    centred contribution,

        phi_p(x) = c_p(x) - E_background[c_p(X)],

    with no independence assumption required -- the usual caveat about
    correlated features applies to interaction terms, and here there are none.

    ``X`` is a DataFrame of raw (unscaled) predictor values; ``background``
    defaults to ``X`` itself, which centres attributions on the explained set.
    Returns a DataFrame of contributions aligned to ``X``'s columns.

    NOTE the attributions are in the linear-predictor space, not cloglog
    probability: cloglog is a monotone but non-linear transform, and applying
    it destroys the additivity that makes the decomposition exact. Magnitudes
    are therefore only comparable to other models after normalising within a
    model (a share), and directions via a rank statistic -- both of which are
    unit-free.
    """
    lam = read_lambdas(lambdas_path)
    bg = X if background is None else background

    def contrib(frame):
        out = pd.DataFrame(0.0, index=frame.index, columns=list(frame.columns))
        for var in frame.columns:
            v = frame[var].to_numpy(dtype="float64")
            if var in lam["categorical"]:
                table = lam["categorical"][var]
                # classes absent from the file were dropped by regularisation:
                # their coefficient is exactly 0, which .get supplies
                out[var] = np.array([table.get(float(c), 0.0) for c in v])
                continue
            total = np.zeros(len(v))
            if var in lam["linear"]:
                a, mn, mx = lam["linear"][var]
                total = total + a * _clamp01(v, mn, mx)
            if var in lam["quadratic"]:
                a, mn, mx = lam["quadratic"][var]
                total = total + a * _clamp01(v ** 2, mn, mx)
            out[var] = total
        return out

    c_x = contrib(X)
    baseline = contrib(bg).mean(axis=0)
    return c_x - baseline


def read_lambdas_k(lambdas_path):
    """Number of feature lambdas with a non-zero coefficient in a MaxEnt
    ``.lambdas`` file. Excludes the 4 metadata rows MaxEnt always appends
    last -- ``linearPredictorNormalizer``, ``densityNormalizer``,
    ``numBackgroundPoints``, ``entropy`` -- which are normalization/
    bookkeeping constants, not fitted feature coefficients, even though
    they too carry a non-zero value in column ``a``.
    """
    rows = pd.read_csv(lambdas_path, header=None, names=["name", "a", "b", "c"])
    features = rows[~rows["name"].isin(LAMBDAS_METADATA_ROWS)]
    return int((features["a"] != 0).sum())


def aicc_from_raw(k, raw_at_presence):
    """AICc from a feature-parameter count ``k`` and the vector of raw
    MaxEnt predictions at the ``n`` presence points MaxEnt actually used
    (``outputformat=raw``'s ``Raw prediction`` column in
    ``*_samplePredictions.csv``) -- ``loglik = sum(log(raw))``, then
    ``_aicc``. Accepts any array-like (ndarray, list, pandas Series).
    """
    raw = np.asarray(raw_at_presence, dtype=float)
    n = len(raw)
    loglik = float(np.sum(np.log(raw)))
    return _aicc(loglik, k, n)


# ============================================================================
# MaxEnt subprocess wrapper
# ============================================================================

def _categorical_layers_present(layers_dir):
    """Which of ``screen.CATEGORICAL``'s two nominal-class layers
    (``hydrologic_soil_group``, ``nlcd_landcover``) actually have an
    ``.asc`` file present in ``layers_dir`` -- ``run()``'s
    ``togglelayertype=`` flags are computed from this rather than
    unconditionally emitted for both names, because the leave-one-out/
    single-variable jackknife runs (``jackknife_gains()`` below) use REDUCED
    layer subsets, down to as few as 1 layer, that may omit one or both
    categorical layers. Behavior-preserving for every OTHER call site
    (``aicc_grid``/``fit_final``/``fit_eval``/``fit_folds`` all pass the
    full ``LAYERS_DIR``, which always contains both, so their emitted args
    are byte-for-byte unchanged by this generalization).

    Returns a sorted list (0, 1, or 2 names).
    """
    layers_dir = Path(layers_dir)
    return sorted(name for name in CATEGORICAL if (layers_dir / f"{name}.asc").exists())


def run(samples_csv, layers_dir, out_dir, *, features, beta, replicates=0,
        outputformat="raw", testsamplesfile=None,
        maximumbackground=DEFAULT_MAXIMUM_BACKGROUND,
        responsecurves=False,
        timeout=DEFAULT_TIMEOUT_S):
    """Run MaxEnt (``density.MaxEnt``) once via subprocess, OS-agnostic Java
    (``config.java_exe()``/``config.maxent_jar()`` -- see this module's
    docstring, "Fix vs. source"): invokes
    ``[config.java_exe(), '-mx2g', '-cp', config.maxent_jar(), 'density.MaxEnt', ...]``,
    never a hardcoded ``java``/jar path.

    Always headless (``visible=false autorun redoifexists nowarnings
    autofeature=false``), ``togglelayertype=<name>`` for whichever of
    ``screen.CATEGORICAL``'s two layers ``_categorical_layers_present``
    finds in ``layers_dir`` (both, for every full-stack call site --
    see that helper's docstring), ``betamultiplier=<beta>``, the
    ``features`` flags, and ``maximumbackground=<maximumbackground>``.
    ``outputformat`` defaults to ``"raw"`` (needed for the AICc
    log-likelihood this module computes from ``*_samplePredictions.csv`` --
    every ``aicc_grid()`` call site relies on this default and never
    overrides it, so the tuning grid's behavior is unchanged by this
    parameter's addition) -- ``fit_final``/``fit_eval``/
    ``fit_folds`` (§4 below -- the final cloglog/logistic bootstrap model
    family, a later stage than the tuning grid) pass
    ``outputformat="cloglog"``/``"logistic"`` explicitly instead.
    stdout/stderr are redirected to ``<out_dir>/run.log``; the subprocess is
    bounded by ``timeout`` seconds (default 600s) so a hung MaxEnt run can't
    block the caller forever -- a timeout is reported as ``rc=-1`` in the
    return value (and noted in the log), not raised, so a caller iterating
    over a grid can record the failure and move on to the next candidate.

    Parameters
    ----------
    samples_csv : path to a MaxEnt presence samples CSV (``species,X,Y`` --
        see ``build_samples_csv``).
    layers_dir : path to the directory of ``.asc`` environmental layers
        (see ``export_layers``).
    out_dir : this run's own output directory (created if missing) --
        MaxEnt writes ``<species>.lambdas``, ``<species>_samplePredictions.csv``,
        ``maxentResults.csv``, etc. here; ``run.log`` is added alongside them.
    features : either one of ``FEATURE_CLASSES``' 5-key flag dicts
        (``linear``/``quadratic``/``product``/``threshold``/``hinge`` ->
        bool) or one of its string keys (``'L'``/``'LQ'``/``'LQHP'``) as a
        convenience -- resolved via ``FEATURE_CLASSES[features]`` first.
    beta : regularization multiplier (``betamultiplier=``).
    replicates : if > 0, adds ``replicates=<n> replicatetype=bootstrap``
        (matching the source notebook's final-model usage); if 0 (default),
        a single deterministic run with neither flag -- what the tuning
        grid uses (a bootstrapped run has no single well-defined ``k``/
        ``n``/raw-prediction set to score by AICc).
    outputformat : ``outputformat=`` (default ``"raw"`` -- the tuning
        grid's requirement; pass ``"cloglog"``/``"logistic"`` for a
        published/eval susceptibility raster -- see §4's ``FINAL_CONFIG``
        callers below).
    testsamplesfile : optional path to a held-out samples CSV -- passed
        through as ``testsamplesfile=<path>`` if given (used by a later
        holdout-AUC step, not the AICc grid itself).
    maximumbackground : ``maximumbackground=`` (default 10,000, MaxEnt's
        own default -- the "random background" arm; the target-group/
        stratified sensitivity arms live in ``pipeline.sensitivity``).
    timeout : per-run subprocess timeout in seconds (default 600).

    Returns
    -------
    dict: ``{rc, out_dir, lambdas_path, preds_path, log_path}`` -- ``rc`` is
    the subprocess return code (``-1`` on timeout, never raised);
    ``lambdas_path``/``preds_path`` are ``<out_dir>/<species>.lambdas`` /
    ``<out_dir>/<species>_samplePredictions.csv`` (the species label is read
    from ``samples_csv`` itself, not hardcoded to ``"flood"``, so this
    wrapper stays correct if a caller ever passes a differently-labeled
    samples file) -- present as *paths* regardless of whether MaxEnt
    actually wrote them; callers must check ``.exists()``/``rc`` before
    trusting their contents, exactly as ``aicc_grid`` does below.
    """
    if isinstance(features, str):
        features = FEATURE_CLASSES[features]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    species = pd.read_csv(samples_csv, nrows=1)["species"].iloc[0]

    args = [
        config.java_exe(), "-mx2g", "-cp", config.maxent_jar(), "density.MaxEnt",
        f"environmentallayers={Path(layers_dir).resolve()}",
        f"samplesfile={Path(samples_csv).resolve()}",
        f"outputdirectory={out_dir.resolve()}",
        *(f"togglelayertype={name}" for name in _categorical_layers_present(layers_dir)),
        "visible=false", "autorun", "redoifexists", "nowarnings",
        f"outputformat={outputformat}",
        "autofeature=false",
        f"betamultiplier={beta}",
        f"maximumbackground={maximumbackground}",
    ]
    for flag, on in features.items():
        args.append(f"{flag}={'true' if on else 'false'}")
    if replicates:
        args += [f"replicates={replicates}", "replicatetype=bootstrap"]
    if responsecurves:
        # Emits plots/<species>_<variable>.dat (the curve data) alongside the
        # PNGs, which is what response_curves() below parses. Off by default:
        # every other call site fits many models and does not need them, and
        # the extra per-variable passes are pure overhead there.
        # writeplotdata is required as well: responsecurves alone renders the
        # PNGs, but only writeplotdata emits the plots/*.dat curve data that
        # response_curves() parses.
        args += ["responsecurves=true", "writeplotdata=true"]
    if testsamplesfile is not None:
        args.append(f"testsamplesfile={Path(testsamplesfile).resolve()}")

    log_path = out_dir / "run.log"
    with open(log_path, "w") as fh:
        fh.write(f"[maxent] args: {args}\n\n")
        fh.flush()
        try:
            result = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[maxent] TIMEOUT after {timeout}s -- process killed\n")
            rc = -1

    return {
        "rc": rc,
        "out_dir": out_dir,
        "lambdas_path": out_dir / f"{species}.lambdas",
        "preds_path": out_dir / f"{species}_samplePredictions.csv",
        "log_path": log_path,
    }


# ============================================================================
# §3 -- Tuning grid search (AICc)
# ============================================================================

def aicc_grid(samples_csv=SAMPLES_PATH, layers_dir=LAYERS_DIR, tuning_dir=TUNING_DIR,
              betas=BETAS, feature_classes=None,
              maximumbackground=DEFAULT_MAXIMUM_BACKGROUND, timeout=DEFAULT_TIMEOUT_S):
    """The 15-candidate AICc tuning grid: ``beta`` in ``betas`` (default
    ``BETAS`` = 0.25/0.5/1/2/3) x feature-class combination in
    ``feature_classes`` (default ``FEATURE_CLASSES`` = L/LQ/LQHP) -- 15
    candidates by default (originally 12, before the log-transform
    adoption added ``beta=0.25`` -- see ``BETAS``'s own comment).
    Each candidate is run via ``run()`` (``outputformat=raw``, a single
    deterministic fit, no bootstrap) and scored by AICc computed from its
    own ``.lambdas`` (``read_lambdas_k`` -> ``k``) and
    ``*_samplePredictions.csv`` (``Raw prediction`` column -> ``n``,
    ``aicc_from_raw`` -> AICc).

    **Idempotent**: a candidate whose ``<out_dir>/<species>.lambdas`` AND
    ``..._samplePredictions.csv`` already exist is scored from the existing
    files directly, without re-invoking the MaxEnt subprocess -- so a
    partially completed grid (e.g. interrupted mid-run) resumes instead of
    restarting every candidate from scratch.

    A candidate that fails (nonzero/timeout ``rc``, or MaxEnt not writing
    one of the two output files) gets ``AICc=inf`` and ``status='failed'``
    -- explicitly recorded, not silently dropped from the returned grid, so
    a caller can see exactly which candidate(s) (if any) failed and why
    (``run.log`` under that candidate's own ``out_dir``).

    Returns a DataFrame with columns ``beta``, ``feature_classes``, ``k``,
    ``n``, ``AICc``, ``status``, plus ``delta_AICc`` (``AICc`` minus the
    grid's own minimum), sorted by ``AICc`` ascending. Also written to
    ``<tuning_dir>/aicc_results.csv``.
    """
    if feature_classes is None:
        feature_classes = FEATURE_CLASSES

    tuning_dir = Path(tuning_dir)
    species = pd.read_csv(samples_csv, nrows=1)["species"].iloc[0]

    rows = []
    for beta in betas:
        for fc_name, fc_flags in feature_classes.items():
            tag = f"b{beta}_{fc_name}"
            run_dir = tuning_dir / tag
            lambdas_path = run_dir / f"{species}.lambdas"
            preds_path = run_dir / f"{species}_samplePredictions.csv"

            if lambdas_path.exists() and preds_path.exists():
                print(f"[maxent] {tag}: cached (lambdas + samplePredictions already present)")
                rc, elapsed = 0, 0.0
            else:
                t0 = time.time()
                result = run(samples_csv, layers_dir, run_dir, features=fc_flags, beta=beta,
                              maximumbackground=maximumbackground, timeout=timeout)
                elapsed = time.time() - t0
                rc = result["rc"]

            if rc != 0 or not lambdas_path.exists() or not preds_path.exists():
                print(f"[maxent] {tag}: FAILED (rc={rc}, {elapsed:.0f}s)")
                rows.append(dict(beta=beta, feature_classes=fc_name, k=None, n=None,
                                  AICc=np.inf, status="failed"))
                continue

            k = read_lambdas_k(lambdas_path)
            raw = pd.read_csv(preds_path)["Raw prediction"].to_numpy(dtype=float)
            n = len(raw)
            aicc = aicc_from_raw(k, raw)
            print(f"[maxent] {tag}: k={k} n={n} AICc={aicc:.2f} ({elapsed:.0f}s)")
            rows.append(dict(beta=beta, feature_classes=fc_name, k=k, n=n, AICc=aicc, status="ok"))

    df = pd.DataFrame(rows)
    df["delta_AICc"] = df["AICc"] - df["AICc"].min()
    df = df.sort_values("AICc").reset_index(drop=True)

    tuning_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tuning_dir / "aicc_results.csv", index=False)
    return df


def select_best(grid):
    """The lowest-AICc row of an ``aicc_grid()`` DataFrame, as a dict --
    restricted to ``status == 'ok'`` rows (a ``'failed'`` candidate's
    ``AICc`` is always ``inf`` by construction, so it would never win on
    value alone, but this filters on ``status`` explicitly rather than
    relying on that invariant holding for every possible caller-built
    grid). Raises ``ValueError`` if every candidate failed.
    """
    ok = grid[grid["status"] == "ok"]
    if len(ok) == 0:
        raise ValueError("[maxent] select_best: no successful grid candidates (all failed)")
    return ok.sort_values("AICc").iloc[0].to_dict()


# ============================================================================
# §4 -- Final vs. evaluation model families (the correctness fix behind
# the resubstitution-AUC bug: a full-model raster sampled at its
# own training points reported 0.9842 AUC as "the" accuracy number, when
# the true held-out AUC was 0.9684). Two DISTINCT model families are fit
# at the SAME locked configuration (``FINAL_CONFIG``) but on DIFFERENT
# point sets:
#   - ``fit_final``  -- ALL 277 canonical points -> the PUBLISHED map. Its
#     own self-reported "Training AUC" (``apparent_auc()`` below) is
#     resubstitution/in-sample and must NEVER be reported as the headline
#     accuracy number.
#   - ``fit_eval``/``fit_folds`` -- TRAIN-SPLIT ONLY -> a raster that has
#     never seen the held-out point(s) it will later be sampled at
#     (the honest out-of-sample AUC). ``qa.
#     assert_eval_model_excludes_test`` is asserted BEFORE any subprocess
#     runs, never after -- a caller that manages to construct an
#     overlapping train/held-out pair gets an ``AssertionError`` before a
#     single MaxEnt process starts, never a silently-contaminated raster.
# ============================================================================

FINAL_DIR = OUT_DIR / "final"
HOLDOUT_MODEL_DIR = OUT_DIR / "holdout_model"
FOLDS_DIR = OUT_DIR / "folds"

# **LOCKED** (owner decision) -- the final
# published model's regularization multiplier and feature classes. This is
# deliberately NOT the raw AICc grid's numerical winner: ``aicc_grid()``/
# ``select_best()`` on the real 277-point/17-layer canonical stack picked
# **beta=0.25, LQ** (AICc=3794.93 -- see ``data/processed/maxent/tuning/
# aicc_results.csv``), but that winner sits at ``BETAS``' own low edge
# (0.25/0.5/1/2/3) with AICc *monotonically decreasing* as beta shrinks
# toward it -- the classic signature of the criterion still wanting to go
# lower still (more overfitting risk, not less), not a genuine interior
# minimum. The raw runner-up (**beta=3, LQHP**, Delta_AICc=1.245) is
# worse evidence of a stable choice, not better: k=60 fitted feature
# parameters for a 277-point presence sample is implausibly complex --
# also a strong overfitting signal despite its low AICc. **beta=0.5, LQ**
# (k=41, Delta_AICc=2.075 -- at the edge of, but still inside, the
# conventional Delta<2 "substantial support" band) is chosen instead
# because it is an INTERIOR, stable optimum (both neighbors in the
# pre-0.25 four-point grid, 0.25 and 1.0, score worse -- see
# finalize-config-report.md's beta=0.5 interior-minimum discussion) and is
# the standard regularization floor most SDM practice starts from --
# preferred over a boundary/edge candidate on statistical-equivalence-plus-
# stability grounds, not because it scored lowest on the grid. This dict
# is the single source of truth for every function below; nothing in this
# module re-derives beta/features from the tuning grid at fit time.
FINAL_CONFIG = {"beta": 0.5, "features": "LQ"}

# 10-replicate bootstrap (matches the original notebook's §4-5 recipe
# exactly): per-replicate rasters + ``flood_avg.asc``
# (mean -- the published map) + ``flood_stddev.asc`` (uncertainty across
# replicates). cloglog is MaxEnt's modern recommended output scale
# (primary/published); logistic is retained for comparison -- both are
# strictly monotonic transforms of the same linear predictor (identical
# point-ranking/AUC), so only cloglog is treated as "primary" and returned
# by ``fit_final``.
FINAL_REPLICATES = 10
FINAL_FORMATS = ("cloglog", "logistic")


def _reference_raster(pred_dir=None):
    """The alignment reference used to convert a MaxEnt ``.asc`` output
    grid back to a GeoTIFF -- ``dem.tif`` from the screened predictor
    stack, mirroring the original notebook's §5 (``ref = rioxarray.
    open_rasterio(PRED_DIR / 'dem.tif')``). Any ONE of the modeling layers
    would work equally (``predictors._alignment_audit`` already guarantees
    every layer shares one CRS/shape/transform), but ``dem`` specifically
    is used because ``modeling_layers()`` never drops it (see ``screen``'s
    ``EXPECTED_KEPT``), so this call can never fail on a missing file for
    whatever the live screened stack currently is.
    """
    if pred_dir is None:
        pred_dir = screen.SCREENED_DIR
    return rioxarray.open_rasterio(Path(pred_dir) / "dem.tif").squeeze("band", drop=True)


def _asc_to_geotiff(asc_path, ref, out_path, nodata=NODATA):
    """Convert one MaxEnt ``.asc`` output grid to a GeoTIFF sharing
    ``ref``'s CRS/transform/shape -- the same recipe as the original
    notebook's §5 ``asc_to_raster`` + ``.rio.to_raster``: the
    ``.asc``'s own NODATA sentinel is mapped to NaN, the array is wrapped
    as a copy of ``ref`` (inheriting its georeferencing), then re-masked
    by ``ref``'s OWN null footprint (belt-and-suspenders -- if MaxEnt ever
    wrote a numeric value outside the AOI, that pixel is still forced back
    to NaN rather than leaking a bogus value onto the published map).
    Returns ``out_path`` (parent directory created if missing).
    """
    arr = np.loadtxt(asc_path, skiprows=6)
    arr = np.where(arr == nodata, np.nan, arr).astype("float32")
    da = ref.copy(data=arr)
    da = da.where(~ref.isnull())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(out_path)
    return out_path


def apparent_auc(out_dir=FINAL_DIR, fmt="cloglog"):
    """The final (all-277-point) model's OWN self-reported "Training AUC"
    from ``<out_dir>/<fmt>/maxentResults.csv`` -- mean +/- sd across the
    bootstrap replicate rows (``Species`` matching ``flood_<int>``; falls
    back to the single ``"flood"`` row if the run was not bootstrapped,
    e.g. a ``replicates=0`` debug call).

    Deliberately named ``apparent_auc``, never ``auc``/``headline_auc``:
    it is resubstitution accuracy -- the model scored on the SAME 277
    points it was fit on -- not a generalization estimate. This is exactly
    the quantity behind the original resubstitution bug: a resubstitution AUC
    (0.9842) silently stood in for the headline number where the true
    held-out AUC was 0.9684. The honest out-of-sample metric is
    ``evaluate.oos_auc``, computed by sampling ``fit_eval``'s train-only
    raster at the TEST points -- never computed in this module.

    Returns ``{"mean": float, "sd": float, "n_replicates": int}``.
    """
    path = Path(out_dir) / fmt / "maxentResults.csv"
    df = pd.read_csv(path)
    reps = df[df["Species"].astype(str).str.match(r"^flood_\d+$")]
    if reps.empty:
        reps = df[df["Species"].astype(str) == "flood"]
    aucs = reps["Training AUC"].astype(float)
    return {
        "mean": float(aucs.mean()),
        "sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "n_replicates": int(len(aucs)),
    }


def fit_final(samples=None, *, out_dir=FINAL_DIR, replicates=FINAL_REPLICATES,
              formats=FINAL_FORMATS, timeout=DEFAULT_TIMEOUT_S):
    """Fit the PUBLISHED susceptibility model: ``FINAL_CONFIG`` (beta=0.5,
    LQ) on ALL of ``samples`` (default ``config.canonical_samples()`` --
    the full 277-point canonical set), ``replicates=10
    replicatetype=bootstrap``, once per entry in ``formats`` (default
    cloglog + logistic). Mirrors the original notebook's §4-5 exactly (same
    recipe, OS-agnostic Java/jar via ``run()``).

    For each format, converts MaxEnt's own bootstrap ``flood_avg.asc``
    (mean across replicates) AND ``flood_stddev.asc`` (uncertainty) to
    GeoTIFFs (``_asc_to_geotiff``, matching the predictor grid's CRS/
    transform) under ``<out_dir>/<fmt>/``.

    Returns the PRIMARY map path -- ``<out_dir>/cloglog/flood_avg.tif``,
    the mean of ``replicates`` bootstrap replicates trained on all 277
    points. Its accompanying ``maxentResults.csv`` "Training AUC" is
    resubstitution, NOT a generalization estimate -- read it via
    ``apparent_auc()``, never treat it as the headline number (see that
    function's docstring and this module's §4 header comment for the
    background).

    **Non-deterministic ensemble -- statistical, not bit, reproducibility.**
    The 10-replicate bootstrap resamples the presence set WITHOUT a fixed
    seed: MaxEnt's CLI exposes no ``randomseed`` hook and ``run()`` passes
    none, so ``flood_avg.tif`` is NOT bit-reproducible run-to-run -- two fresh
    fits yield slightly different rasters (different md5s). The jitter is small
    and every downstream CONCLUSION is invariant: the equity survivor set
    (``bg_pop_density_mean`` only), every Holm/FDR significance verdict, and
    the headline AUC bands -- the last computed from the DETERMINISTIC
    ``replicates=0`` evaluation model (``fit_eval``/``fit_folds`` ->
    ``evaluate.EVAL_RASTER``), never from THIS raster. This is precisely why
    the equity tests/gates that read this map assert invariants + loose bands
    rather than exact values, and why the driver notebook REUSES an existing
    ``flood_avg.tif`` instead of regenerating it on every run (regeneration
    would perturb the equity zonal stats for no scientific gain). Only
    ``equity.run()`` and the susceptibility map/figures depend on this
    non-deterministic raster; the AUC/robustness/significance stack does not.
    """
    if samples is None:
        samples = config.canonical_samples()

    export_layers()
    out_dir = Path(out_dir)
    samples_csv = build_samples_csv(out_path=out_dir / "samples.csv", presence=samples)
    ref = _reference_raster()

    primary_path = None
    for fmt in formats:
        fmt_dir = out_dir / fmt
        result = run(samples_csv, LAYERS_DIR, fmt_dir,
                     features=FINAL_CONFIG["features"], beta=FINAL_CONFIG["beta"],
                     replicates=replicates, outputformat=fmt, timeout=timeout)
        if result["rc"] != 0:
            raise RuntimeError(
                f"[maxent] fit_final: MaxEnt failed (fmt={fmt}, rc={result['rc']}); "
                f"see {result['log_path']}"
            )

        avg_asc = fmt_dir / "flood_avg.asc"
        if not avg_asc.exists():
            raise FileNotFoundError(f"[maxent] fit_final: expected {avg_asc} not found")
        avg_tif = _asc_to_geotiff(avg_asc, ref, fmt_dir / "flood_avg.tif")

        std_asc = fmt_dir / "flood_stddev.asc"
        if std_asc.exists():
            _asc_to_geotiff(std_asc, ref, fmt_dir / "flood_stddev.tif")

        if fmt == "cloglog":
            primary_path = avg_tif

    return primary_path


def _fit_train_only(samples, train_idx, out_dir, *, timeout=DEFAULT_TIMEOUT_S):
    """Shared mechanics behind ``fit_eval``/``fit_folds``: a SINGLE
    deterministic MaxEnt fit (no bootstrap -- this raster exists to be
    sampled at held-out points for an honest AUC, not to be published, so
    per-replicate uncertainty quantification is out of scope here) at
    ``FINAL_CONFIG`` on ``samples.iloc[train_idx]`` ONLY. Converts the
    resulting cloglog ``flood.asc`` to a GeoTIFF and records the exact
    ``train_idx`` used (``<out_dir>/train_idx.csv``) so a later caller
    (``evaluate`` or ``robustness``) can verify provenance without re-deriving the split.

    Returns the GeoTIFF path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_layers()
    train_idx = np.asarray(train_idx)
    train_points = samples.iloc[train_idx]
    samples_csv = build_samples_csv(out_path=out_dir / "train_samples.csv", presence=train_points)

    result = run(samples_csv, LAYERS_DIR, out_dir,
                 features=FINAL_CONFIG["features"], beta=FINAL_CONFIG["beta"],
                 replicates=0, outputformat="cloglog", timeout=timeout)
    if result["rc"] != 0:
        raise RuntimeError(
            f"[maxent] train-only fit failed (out_dir={out_dir}, rc={result['rc']}); "
            f"see {result['log_path']}"
        )

    asc_path = out_dir / "flood.asc"
    if not asc_path.exists():
        raise FileNotFoundError(f"[maxent] train-only fit: expected {asc_path} not found")

    raster_path = _asc_to_geotiff(asc_path, _reference_raster(), out_dir / "flood.tif")
    np.savetxt(out_dir / "train_idx.csv", train_idx, fmt="%d", header="train_idx", comments="")
    return raster_path


def fit_eval(train_idx, samples=None, *, out_dir=HOLDOUT_MODEL_DIR,
             timeout=DEFAULT_TIMEOUT_S, _record_only=False):
    """Fit the EVALUATION model: ``FINAL_CONFIG`` on ``samples.iloc[
    train_idx]`` ONLY -- never the full 277-point set -- writing the
    train-only susceptibility raster to ``<out_dir>/flood.tif``
    (``holdout_model/`` by default). This is the raster ``evaluate.oos_auc`` samples
    at the TEST points for the honest out-of-sample AUC (never the
    ``fit_final`` raster -- that was exactly the resubstitution bug).

    **Invariant 2, enforced BEFORE any fit**:
    ``samples`` defaults to ``config.canonical_samples()``, and the
    canonical held-out partition is recomputed fresh via ``config.
    split(samples)`` (the SAME deterministic, ``GLOBAL_SEED``-seeded split
    every other caller in this pipeline uses) -- ``qa.
    assert_eval_model_excludes_test(train_idx, held_out_idx)`` is asserted
    against THAT recomputation, not merely trusted from the caller, so a
    future refactor that accidentally feeds this function the full
    277-point set (or any other ``train_idx`` overlapping the real test
    split) raises immediately instead of silently training an "eval"
    model on its own held-out points.

    ``_record_only=True`` returns ``train_idx`` (as an ndarray)
    immediately after the invariant check -- no directory, no samples
    CSV, no subprocess -- fast enough for a pytest invariant test
    (``tests/test_model_families.py::test_eval_model_never_sees_test``).
    """
    if samples is None:
        samples = config.canonical_samples()
    train_idx = np.asarray(train_idx)

    _, held_out_idx = config.split(samples)
    qa.assert_eval_model_excludes_test(train_idx, held_out_idx)

    if _record_only:
        return train_idx

    return _fit_train_only(samples, train_idx, out_dir, timeout=timeout)


def _fold_holdouts(fold_assignment, n):
    """Normalize a fold assignment into ``{fold_id: held_out_idx_ndarray}``
    -- accepts either a per-sample fold-code array (``config.
    spatial_folds()``'s return: one integer code per sample, length ``n``)
    or a ``{fold_id: held_out_idx}`` dict (``config.loeo_folds()``'s
    return, keyed by event label) -- ``robustness`` (spatial-CV/LOEO) calls
    ``fit_folds`` with either shape, so this dispatches on which one it
    got (``hasattr(..., "items")``) rather than requiring the caller to
    pre-convert. ``n`` (the samples' full length -- matching ``config.
    split()``/``loeo_folds()``'s own "0..n-1 RangeIndex" convention)
    defines the complement universe for the array form; for the dict form
    ``n`` is trusted as given (``loeo_folds()`` already covers every point
    exactly once -- see ``test_loeo_covers_every_point_once``).

    Returns ``(holdouts, universe)`` -- ``universe`` is ``np.arange(n)``.
    """
    if hasattr(fold_assignment, "items"):
        holdouts = {fid: np.asarray(idx) for fid, idx in fold_assignment.items()}
    else:
        arr = np.asarray(fold_assignment)
        assert len(arr) == n, (
            f"[maxent] fold_assignment length {len(arr)} != samples length {n}"
        )
        holdouts = {fid: np.where(arr == fid)[0] for fid in np.unique(arr)}
    return holdouts, np.arange(n)


def fit_folds(fold_assignment, samples=None, *, out_dir=FOLDS_DIR,
              timeout=DEFAULT_TIMEOUT_S, _record_only=False):
    """Per-fold train-only models: for every fold in ``fold_assignment``
    (``config.spatial_folds()``'s per-sample array OR ``config.
    loeo_folds()``'s ``{event: idx}`` dict -- see ``_fold_holdouts``), fits
    ``FINAL_CONFIG`` on the COMPLEMENT of that fold's held-out indices
    (every other point in ``samples``), writing to ``<out_dir>/
    fold_<id>/``. Used by ``robustness``'s spatial-CV/LOEO significance testing.

    **Invariant 2, enforced per fold before its fit**: for each
    ``fold_id``, ``qa.assert_eval_model_excludes_test(train_idx,
    held_out_idx)`` is asserted (``train_idx`` = every OTHER fold's
    indices) before that fold's MaxEnt subprocess ever runs.

    ``_record_only=True`` skips the MaxEnt subprocess entirely and returns
    ``{fold_id: train_idx}`` (ndarrays) instead of ``{fold_id: raster_
    path}`` -- fast, for a synthetic invariant test that needs neither a
    real predictor stack nor the MaxEnt binary (only ``len(samples)`` is
    used before the early return -- a bare-length placeholder frame is
    enough).

    Returns ``{fold_id: raster_path}`` (or ``{fold_id: train_idx}`` under
    ``_record_only``).
    """
    if samples is None:
        samples = config.canonical_samples()
    holdouts, universe = _fold_holdouts(fold_assignment, n=len(samples))

    results = {}
    for fold_id, held_out_idx in holdouts.items():
        train_idx = np.setdiff1d(universe, held_out_idx)
        qa.assert_eval_model_excludes_test(train_idx, held_out_idx)

        if _record_only:
            results[fold_id] = train_idx
            continue

        results[fold_id] = _fit_train_only(
            samples, train_idx, Path(out_dir) / f"fold_{fold_id}", timeout=timeout
        )

    return results


# ============================================================================
# §5 -- jackknife unique contribution + permutation importance +
# percent contribution (+ optional MW rank-biserial join), combined into one
# per-predictor table.
#
# Ports the original notebook's §7 ("Variable Importance: Jackknife +
# Permutation + Bivariate") -- but that source cell only COMBINES a
# pre-existing ``jackknife_gains.csv``; the LOO-fit generation itself is
# reconstructed here from MaxEnt's own documented semantics (§7's markdown
# table: "21 leave-one-out MaxEnt runs" / "21 single-variable MaxEnt runs"),
# since the source notebook conversion under review does not carry that
# generation cell.
#
# Three metrics, three sources -- never fabricated, always read from a real
# MaxEnt/Task-6.2 output:
#   1. Jackknife unique contribution = gain_all - gain_without, from
#      ``jackknife_gains()``'s own ~2N+1 leave-one-out/single-variable
#      MaxEnt fits at ``FINAL_CONFIG`` (all 277 points, non-bootstrap,
#      ``outputformat=raw``) -- for THIS project's 17-predictor canonical
#      stack, N=17 so 2*17+1 = 35 fits (the OLD 21-predictor stack needed
#      43; see this module's top docstring, "Canonical modeling stack").
#   2. Permutation importance + percent contribution (mean +/- sd across
#      the 10 bootstrap replicates), read directly from ``fit_final``'s
#      ``FINAL_DIR / "cloglog" / "maxentResults.csv"`` -- no new MaxEnt run.
#   3. MW rank-biserial r, an OPTIONAL join from ``screen.bivariate()``'s
#      ``predictor_bivariate_characterization.csv`` if present on disk --
#      only for rows whose ``effect_metric`` really is "rank-biserial r"
#      (continuous predictors); the two categorical predictors' Cramer's V
#      is a DIFFERENT statistic on a different scale ([0,1] vs [-1,1]) and
#      is deliberately left NaN here rather than silently merged into the
#      same numeric column under a false shared label.
#
# CRITICAL (task instructions): the actual #1 predictor by unique
# contribution / permutation importance is whatever the real ~35-fit run
# below actually produces -- this module never hardcodes or forces
# ``distance_to_outfall`` (or any other name) to rank first. The OLD
# 21-predictor/no-log-transform/no-TWI model's jackknife happened to crown
# ``distance_to_outfall``; THIS model's screened stack differs (17
# predictors, log10/log1p-transformed hydrology, TWI retained instead of
# flow_accumulation -- see this module's top docstring), so the ranking may
# legitimately differ.
# ============================================================================

JACKKNIFE_DIR = OUT_DIR / "jackknife"
JACKKNIFE_GAINS_CSV = OUT_DIR / "jackknife_gains.csv"
VARIABLE_IMPORTANCE_CSV = OUT_DIR / "variable_importance_combined.csv"
BIVARIATE_CSV = config.DATA / "predictor_bivariate_characterization.csv"


def _build_layer_subset(names, subset_dir, layers_dir=LAYERS_DIR):
    """Build (idempotently) a directory of SYMLINKS to ``{name}.asc`` for
    each ``name`` in ``names``, sourced from the already-exported
    ``layers_dir`` (default ``LAYERS_DIR``, §1's 17-layer canonical export)
    -- never copies the (tens-of-MB) ``.asc`` grids, so building 35 reduced
    subset directories costs ~zero extra disk, only inode/symlink overhead.

    Idempotent: if ``subset_dir`` already contains exactly ``{f"{n}.asc"
    for n in names}`` (no more, no fewer), it is left untouched. Otherwise
    any stale ``.asc`` entries not in ``names`` are removed and any missing
    ones are (re-)symlinked -- so a subset directory can be safely reused
    across repeated calls (e.g. an interrupted-and-resumed jackknife run)
    without accumulating leftovers from a differently-named prior use of
    the same directory.

    Raises ``FileNotFoundError`` if ``layers_dir / f"{name}.asc"`` is
    missing for any requested ``name`` (i.e. ``export_layers()`` has not
    yet produced it) -- never silently skips a requested layer.

    Returns ``subset_dir`` (created if missing).
    """
    subset_dir = Path(subset_dir)
    layers_dir = Path(layers_dir)
    expected = {f"{n}.asc" for n in names}

    existing = {p.name for p in subset_dir.glob("*.asc")} if subset_dir.exists() else set()
    if existing == expected:
        return subset_dir

    subset_dir.mkdir(parents=True, exist_ok=True)
    for p in subset_dir.glob("*.asc"):
        if p.name not in expected:
            p.unlink()

    for name in names:
        target = layers_dir / f"{name}.asc"
        if not target.exists():
            raise FileNotFoundError(
                f"[maxent] _build_layer_subset: missing source layer {target} "
                f"-- run export_layers() first"
            )
        link = subset_dir / f"{name}.asc"
        if not link.exists():
            link.symlink_to(target.resolve())

    return subset_dir


def _read_regularized_gain(maxent_results_csv, species="flood"):
    """The ``Regularized training gain`` scalar from a single (non-bootstrap)
    MaxEnt run's ``maxentResults.csv`` -- selects the row matching
    ``species`` if present, else falls back to the first row (mirrors
    ``apparent_auc``'s own species-row-then-fallback pattern).
    """
    df = pd.read_csv(maxent_results_csv)
    match = df[df["Species"].astype(str) == species]
    row = match.iloc[0] if not match.empty else df.iloc[0]
    return float(row["Regularized training gain"])


def jackknife_gains(layers=None, samples_csv=None, *, out_dir=JACKKNIFE_DIR,
                     gains_csv=JACKKNIFE_GAINS_CSV, beta=None, features=None,
                     maximumbackground=DEFAULT_MAXIMUM_BACKGROUND,
                     timeout=DEFAULT_TIMEOUT_S):
    """The ~2N+1-fit jackknife: for the N (17) canonical modeling
    predictors, fits MaxEnt at ``FINAL_CONFIG`` (all 277 canonical points,
    ``replicates=0``, ``outputformat=raw``) on:
      - **all** N layers (once) -> ``gain_all``;
      - **without** each layer (N fits, the other N-1 layers) -> ``gain_without``;
      - **only** each layer (N fits, that layer alone) -> ``gain_only``.
    ``unique_contribution = gain_all - gain_without`` per predictor -- the
    regularized training gain lost when that predictor is removed from the
    full model (isolates non-redundant information; a predictor whose
    information is fully captured by OTHER correlated predictors in the
    stack has ``unique_contribution`` near zero even if its ``gain_only``
    or bootstrap percent-contribution is high -- see this section's header
    comment).

    Each of the N+1 reduced/single-variable layer subsets is built via
    ``_build_layer_subset`` (symlinks into ``LAYERS_DIR``, no raster
    copies). ``beta``/``features`` default to ``FINAL_CONFIG``'s locked
    values (never re-derived from the AICc grid here) -- the SAME
    regularization/feature-class settings as the published model, varying
    only the predictor SET, matching the source notebook's own recipe
    ("Full model gain = 2.4446 (n=278 presence points, beta=1, linear
    features)" -- this project's own gain_all differs because both the
    predictor stack and beta/features differ; see this module's top
    docstring).

    **Idempotent**: a candidate whose ``<out_dir>/<tag>/maxentResults.csv``
    already exists is scored from that existing file directly, without
    re-invoking the MaxEnt subprocess -- an interrupted-and-resumed
    jackknife run skips every already-completed fit (mirrors
    ``aicc_grid()``'s own idempotency).

    Returns a DataFrame with columns ``predictor``, ``gain_all``,
    ``gain_without``, ``gain_only``, ``unique_contribution``, ``n_all``,
    ``n_without``, ``n_only`` (the ``#Training samples`` MaxEnt actually
    used in each of the three fits -- a sanity check that removing/isolating
    a layer never changes ``n`` away from 277, since every canonical point
    is already valid against the FULL screened stack, a superset of any
    reduced subset used here; a value that DOES differ is surfaced, not
    silently accepted). Also written to ``gains_csv`` (default
    ``data/processed/maxent/jackknife_gains.csv``).
    """
    if layers is None:
        layers = modeling_layers()
    if beta is None:
        beta = FINAL_CONFIG["beta"]
    if features is None:
        features = FINAL_CONFIG["features"]

    export_layers(layers)
    if samples_csv is None:
        samples_csv = build_samples_csv(out_path=SAMPLES_PATH)

    out_dir = Path(out_dir)

    def _fit(tag, names):
        run_dir = out_dir / tag
        results_csv = run_dir / "maxentResults.csv"
        if results_csv.exists():
            print(f"[maxent] jackknife {tag}: cached")
        else:
            subset_dir = _build_layer_subset(names, run_dir / "_layers")
            t0 = time.time()
            result = run(samples_csv, subset_dir, run_dir, features=features, beta=beta,
                         replicates=0, outputformat="raw",
                         maximumbackground=maximumbackground, timeout=timeout)
            elapsed = time.time() - t0
            if result["rc"] != 0 or not results_csv.exists():
                raise RuntimeError(
                    f"[maxent] jackknife {tag}: MaxEnt failed (rc={result['rc']}); "
                    f"see {result['log_path']}"
                )
            print(f"[maxent] jackknife {tag}: {len(names)} layer(s), {elapsed:.0f}s")
        df = pd.read_csv(results_csv)
        n = int(df["#Training samples"].iloc[0])
        return _read_regularized_gain(results_csv), n

    gain_all, n_all = _fit("all", layers)
    print(f"[maxent] jackknife gain_all={gain_all:.4f} (n={n_all}, {len(layers)} predictors)")

    rows = []
    for name in layers:
        gain_without, n_without = _fit(f"without_{name}", [n for n in layers if n != name])
        gain_only, n_only = _fit(f"only_{name}", [name])
        rows.append(dict(
            predictor=name,
            gain_all=gain_all,
            gain_without=gain_without,
            gain_only=gain_only,
            unique_contribution=gain_all - gain_without,
            n_all=n_all, n_without=n_without, n_only=n_only,
        ))

    df = pd.DataFrame(rows).sort_values("unique_contribution", ascending=False).reset_index(drop=True)
    gains_csv = Path(gains_csv)
    gains_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(gains_csv, index=False)
    print(f"[maxent] wrote {len(df)} predictors' jackknife gains -> {gains_csv}")
    return df


def _bootstrap_importance(maxent_results_csv, predictors):
    """Tidy per-predictor ``percent_contribution``/``permutation_importance``
    (mean +/- sd) from a bootstrap-replicate ``maxentResults.csv`` (Task
    6.2's ``FINAL_DIR / "cloglog" / "maxentResults.csv"`` by default, via
    ``variable_importance_combined`` below) -- averages ONLY the
    ``flood_<i>`` replicate rows (``apparent_auc``'s own regex), explicitly
    excluding MaxEnt's own ``"flood (average)"`` summary row; falls back to
    a single ``"flood"`` row if the file was not a bootstrap run.

    Raises ``KeyError`` if either wide column (``f"{p} contribution"`` /
    ``f"{p} permutation importance"``) is missing for any requested
    predictor -- never silently produces NaN for a predictor that should
    have been in the model.

    Returns a DataFrame: ``predictor``, ``percent_contribution_mean``,
    ``percent_contribution_sd``, ``permutation_importance_mean``,
    ``permutation_importance_sd``.
    """
    df = pd.read_csv(maxent_results_csv)
    reps = df[df["Species"].astype(str).str.match(r"^flood_\d+$")]
    if reps.empty:
        reps = df[df["Species"].astype(str) == "flood"]

    rows = []
    for p in predictors:
        contrib_col = f"{p} contribution"
        perm_col = f"{p} permutation importance"
        missing = [c for c in (contrib_col, perm_col) if c not in reps.columns]
        if missing:
            raise KeyError(
                f"[maxent] _bootstrap_importance: predictor {p!r} missing column(s) "
                f"{missing} in {maxent_results_csv}"
            )
        contrib = reps[contrib_col].astype(float)
        perm = reps[perm_col].astype(float)
        rows.append(dict(
            predictor=p,
            percent_contribution_mean=float(contrib.mean()),
            percent_contribution_sd=float(contrib.std(ddof=1)) if len(contrib) > 1 else 0.0,
            permutation_importance_mean=float(perm.mean()),
            permutation_importance_sd=float(perm.std(ddof=1)) if len(perm) > 1 else 0.0,
        ))
    return pd.DataFrame(rows)


def combine_variable_importance(jackknife_df, bootstrap_importance_df, bivariate_df=None):
    """Join jackknife (``gain_only``/``unique_contribution``) with bootstrap
    (``permutation_importance_mean``/``_sd``, ``percent_contribution_mean``/
    ``_sd``) on ``predictor``, sorted by ``unique_contribution`` descending
    (the source notebook's own sort key -- the metric that isolates
    non-redundant information, as opposed to ``gain_only``/percent
    contribution which can be inflated by a predictor's correlated stand-ins
    -- see this section's header comment).

    An outer join with a post-hoc completeness assertion (rather than an
    inner join) -- a predictor present in only ONE of the two sources is a
    real bug (e.g. a failed jackknife fit for that predictor) and must be
    surfaced loudly, never silently dropped from the combined table.

    ``bivariate_df`` is an OPTIONAL third source (``screen.bivariate()``'s
    ``predictor_bivariate_characterization.csv``) -- if given, a
    ``mw_rank_biserial`` column is left-joined in, populated ONLY for rows
    whose ``effect_metric`` is literally ``"rank-biserial r"`` (continuous
    predictors); a categorical predictor's Cramer's V is a different
    statistic on a different scale and is left NaN rather than merged in
    under the same numeric column. If ``bivariate_df`` is ``None``, the
    ``mw_rank_biserial`` column is omitted entirely (not filled with NaN
    for every row) -- this table's other two metrics stand alone.

    Returns the combined DataFrame.
    """
    df = jackknife_df[["predictor", "gain_only", "unique_contribution"]].merge(
        bootstrap_importance_df[[
            "predictor", "permutation_importance_mean", "permutation_importance_sd",
            "percent_contribution_mean", "percent_contribution_sd",
        ]],
        on="predictor", how="outer", validate="one_to_one",
    )
    incomplete = df[df.isna().any(axis=1)]
    assert incomplete.empty, (
        f"[maxent] combine_variable_importance: predictor(s) present in only one of "
        f"the jackknife/bootstrap sources: {sorted(incomplete['predictor'])}"
    )

    if bivariate_df is not None:
        biv = bivariate_df.copy()
        biv["mw_rank_biserial"] = np.where(
            biv["effect_metric"] == "rank-biserial r", biv["effect"], np.nan
        )
        df = df.merge(biv[["predictor", "mw_rank_biserial"]], on="predictor", how="left")

    return df.sort_values("unique_contribution", ascending=False).reset_index(drop=True)


def variable_importance_combined(jackknife_csv=JACKKNIFE_GAINS_CSV,
                                  bootstrap_results_csv=None,
                                  bivariate_csv=BIVARIATE_CSV,
                                  out_csv=VARIABLE_IMPORTANCE_CSV):
    """Orchestrates the combined variable-importance table: reads
    ``jackknife_csv`` (default ``jackknife_gains()``'s own output) and
    ``bootstrap_results_csv`` (default ``fit_final``'s
    ``FINAL_DIR / "cloglog" / "maxentResults.csv"``), joins via
    ``_bootstrap_importance`` + ``combine_variable_importance``, optionally
    joining ``bivariate_csv`` (``screen.bivariate()``'s output) if it exists on disk --
    silently omitted (not an error) if absent.

    Pure CSV I/O + merge -- no MaxEnt subprocess is invoked here (that is
    entirely ``jackknife_gains()``'s job, run once beforehand); safe to call
    repeatedly, including from pytest.

    Writes the combined table to ``out_csv`` (default
    ``data/processed/maxent/variable_importance_combined.csv``) and returns
    it as a DataFrame.
    """
    if bootstrap_results_csv is None:
        bootstrap_results_csv = FINAL_DIR / "cloglog" / "maxentResults.csv"

    jk = pd.read_csv(jackknife_csv)
    boot = _bootstrap_importance(bootstrap_results_csv, jk["predictor"].tolist())
    biv = pd.read_csv(bivariate_csv) if Path(bivariate_csv).exists() else None

    df = combine_variable_importance(jk, boot, bivariate_df=biv)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[maxent] wrote combined variable importance ({len(df)} predictors) -> {out_csv}")
    return df


# ============================================================================
# Response curves -- the shape and direction of each predictor's effect.
#
# Jackknife gain and permutation importance both answer "how much does this
# predictor matter"; neither answers "in which direction, and with what shape".
# For the interpretability claim this study rests on, that second question is
# the one a reader actually needs, and MaxEnt answers it natively rather than
# through a post-hoc attribution method.
#
# The curves are produced by a SINGLE DETERMINISTIC fit at FINAL_CONFIG on all
# canonical points -- the same configuration as the published ensemble but with
# replicates=0, so the curves are exactly reproducible even though the
# published map (a 10-replicate bootstrap with no CLI seed) is not.
# ============================================================================

RESPONSE_DIR = OUT_DIR / "response_curves"


def response_curves(out_dir=RESPONSE_DIR, samples=None, timeout=DEFAULT_TIMEOUT_S,
                     force=False):
    """Fit once at ``FINAL_CONFIG`` with ``responsecurves=true`` and parse the
    resulting curve data into a tidy frame.

    MaxEnt writes one ``plots/<species>_<variable>.dat`` per predictor, each a
    two-column ``x,y`` sampling of the fitted response holding the other
    predictors at their means. Returns a DataFrame with columns ``predictor``,
    ``x``, ``y`` (long form, all predictors stacked), and caches the fit so
    repeated calls do not re-invoke MaxEnt unless ``force``.

    NOTE these are the MARGINAL curves (``<species>_<variable>.dat``), which
    hold other predictors at their mean. MaxEnt also writes ``_only`` variants
    fitted on that predictor alone; the marginal form is the one that describes
    the fitted model and is what is reported here.
    """
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    if force or not plots.exists() or not any(plots.glob("*.dat")):
        out_dir.mkdir(parents=True, exist_ok=True)
        export_layers()
        if samples is None:
            samples = config.canonical_samples()
        samples_csv = build_samples_csv(out_path=out_dir / "samples.csv", presence=samples)
        result = run(samples_csv, LAYERS_DIR, out_dir,
                     features=FINAL_CONFIG["features"], beta=FINAL_CONFIG["beta"],
                     replicates=0, outputformat="cloglog",
                     responsecurves=True, timeout=timeout)
        if result["rc"] != 0:
            raise RuntimeError(
                f"[maxent] response_curves: MaxEnt failed (rc={result['rc']}); "
                f"see {result['log_path']}"
            )
    dats = sorted(plots.glob("*.dat"))
    if not dats:
        raise FileNotFoundError(f"[maxent] response_curves: no .dat files under {plots}")

    species = "flood"
    rows = []
    for d in dats:
        stem = d.stem
        if stem.endswith("_only") or not stem.startswith(species + "_"):
            continue
        predictor = stem[len(species) + 1:]
        try:
            curve = pd.read_csv(d)
        except Exception:
            continue
        cols = list(curve.columns)
        if len(cols) < 2:
            continue
        sub = curve.rename(columns={cols[-2]: "x", cols[-1]: "y"})[["x", "y"]].copy()
        sub.insert(0, "predictor", predictor)
        rows.append(sub)

    if not rows:
        raise FileNotFoundError(
            f"[maxent] response_curves: parsed no marginal curves from {len(dats)} .dat files"
        )
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "response_curves.csv", index=False)
    print(f"[maxent] response_curves: {df['predictor'].nunique()} predictors "
          f"-> {out_dir / 'response_curves.csv'}")
    return df

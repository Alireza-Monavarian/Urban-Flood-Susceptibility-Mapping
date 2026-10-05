"""Two-stage multicollinearity screen (Pearson |r| >= 0.8, then VIF > 10)
over the predictor stack, plus a proper categorical-association statistic
(Cramer's V) for the two nominal layers the Pearson/VIF screen cannot use.

Ports the original notebook ``03_multicollinearity`` into an importable pipeline stage.
Faithful to the source in every way that affects the result: RAW
(non-log-transformed) values, the same 200,000-pixel random subsample at
``random_state=42``, the same two-stage Pearson-then-VIF procedure, and the
same >=0.8 / >10 thresholds.

UPDATE (log-transform adoption): Stage 1 Pearson / Stage 2 VIF
(``multicollinearity()``) no longer assess ``flow_accumulation``/``spi`` on
raw physical units. An experiment
found the canonical MaxEnt AICc grid's raw-stack winner was largely a
SCALING ARTIFACT: ``flow_accumulation`` (1 -> ~5.5M) and ``spi`` (0 ->
~9.5e7) each span 6+ orders of magnitude, too wide for any single
linear/Pearson relationship to fit meaningfully. ``multicollinearity()``
now runs Stage 1/2 on a COPY of ``stack`` with every continuous column
passed through ``pipeline.predictors.apply_modeling_transform()`` (log10
for ``flow_accumulation``, log1p for ``spi``, identity for every other
layer) -- collinearity is assessed on the same values the model (MaxEnt)
actually uses. ``_load_stack()`` below is UNCHANGED (still raw, still
faithful to the source's own sampling cell -- see its own docstring): only
the correlation/VIF computation itself sees the transformed copy. The
categorical-association report (``_categorical_association_report``) is
also unaffected -- ``cramers_v``'s quantile-binning is invariant to any
monotone transform, the same reasoning ``bivariate()``'s Mann-Whitney/
rank-biserial relies on (see that function's docstring).

UPDATE 2 (TWI retention): the log-transform adoption above (Stage 1) drops ``spi``
(paired against ``flow_accumulation``, now r=0.81 >= 0.8) and previously
(Stage 2) went on to drop ``twi`` too, once ``flow_accumulation`` shared
TWI's own log-like scale closely enough to push its VIF over 10 -- leaving
``flow_accumulation`` as the sole survivor of the three collinear
accumulation-derivatives. The project owner's standing preference is the
opposite: TWI is the canonical topographic wetness index in
flood-susceptibility literature (it integrates slope directly into a single
index), so ``flow_accumulation`` should be the one Stage 2 removes, not
``twi``. ``_KNOWN_PAIR_RATIONALE`` now includes a
``frozenset({"flow_accumulation", "twi"})`` entry documenting this, and
``_choose_vif_drop`` (Stage 2's per-iteration column-removal choice, used by
``_stage2_vif``) consults it: when the mechanically-highest-VIF column is
TWI and ``flow_accumulation`` is still present, ``flow_accumulation`` is
removed instead. Stage 1 itself is UNCHANGED (still drops ``spi``, keeps
``flow_accumulation`` past Stage 1) -- the swap happens entirely at Stage 2,
because the ``flow_accumulation``/``twi`` pair is a VIF-only effect (their
own pairwise Pearson r=0.768 never crosses the Stage-1 threshold). Net
result: the drop SET is unchanged in size (4 continuous layers: ``tri``,
``tpi``, plus now ``flow_accumulation`` and ``spi`` -- not ``spi`` and
``twi``) and kept/modeled counts are unchanged (18/17) -- only WHICH of
``flow_accumulation``/``twi`` survives is different. See
``_choose_vif_drop``'s own docstring for the mechanics.

Fix vs. source ("fabricated categorical-Pearson r"): the source
notebook's Stage 1 markdown quotes Pearson "r" values between
``curve_number`` (continuous) and the two CATEGORICAL layers --
``hydrologic_soil_group`` (r=0.48) and ``nlcd_landcover`` (r=-0.14) -- as
though they came from its own ``corr`` matrix. They could not have: that
matrix is built from ``df_sample[continuous_cols]``, which the same
notebook explicitly excludes those two layers from three paragraphs
earlier ("Pearson correlation and VIF assume continuous/ordinal-numeric
relationships, and these are nominal class codes"). Pearson correlation
against a nominal class-code column is not even a well-defined operation --
the quoted numbers are inconsistent with the notebook's own code. This
module (a) keeps categorical layers out of the Pearson/VIF computation and
the returned ``report`` entirely (matching the source's actual CODE, not
its markdown prose -- ``"nlcd_landcover" in report.index`` and
``"hydrologic_soil_group" in report.index`` are both always False), and
(b) adds ``cramers_v()`` -- the statistic actually defined for a
categorical-categorical or categorical-continuous pair -- and uses it to
save a genuine categorical-association table to its own CSV
(``CATEGORICAL_REPORT_PATH``), kept separate from the Pearson ``report``.

IMPORTANT -- this module does NOT hardcode a drop set. Stage 1/Stage 2 run
for real against whatever 22-layer stack is on disk (via the modeling
transform described in the "UPDATE" paragraph above). The two terrain
pairs the source notebook found (``slope``/``tri`` r~0.98, ``curvature``/
``tpi`` r~0.97) were the only Stage-1 hits on the RAW (pre-transform)
stack first screened after the D8-routing fix
(``flow_accumulation``/``spi`` was the closest non-terrain pair at that
time, r~0.79, just under threshold). Log-transforming ``flow_accumulation``/``spi``
changes what Stage 1 actually computes for that pair; this docstring
deliberately does not restate a specific current number here (that would
just be a second place for it to go stale) -- run ``multicollinearity()``
for the actual, current drop set on the transformed stack. If a future predictor rebuild (or transform change)
alters the on-disk stack, this screen's ``kept``/``dropped`` output
changes with it.

Inputs: ``data/processed/predictors/*.tif`` (written by ``pipeline.predictors``).
Outputs (build artifacts -- not committed):
  - ``data/processed/multicollinearity_report.csv`` (Pearson corr matrix +
    VIF + dropped flag, continuous layers only);
  - ``data/processed/categorical_association_report.csv`` (Cramer's V of
    each categorical layer against every other layer);
  - ``data/processed/predictors_screened/*.tif`` (the surviving layers,
    copied from ``data/processed/predictors/``).

Also provides ``bivariate()`` (ports the original notebook
``02c_predictor_bivariate``): a presence-vs-background predictor
CHARACTERIZATION -- Mann-Whitney U + rank-biserial r for continuous
predictors, chi-square + this module's own ``cramers_v`` for categorical
predictors, Bonferroni-corrected over however many predictors are actually
tested. Explicitly labeled non-validation throughout (``bivariate``'s
docstring, ``df.attrs``, and its output CSV's filename) -- see that
function's docstring for the full framing rationale.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from pipeline import config, predictors

PRED_DIR = config.DATA / "predictors"
SCREENED_DIR = config.DATA / "predictors_screened"
REPORT_PATH = config.DATA / "multicollinearity_report.csv"
CATEGORICAL_REPORT_PATH = config.DATA / "categorical_association_report.csv"
BIVARIATE_REPORT_PATH = config.DATA / "predictor_bivariate_characterization.csv"

# The same two nominal-class-code layers the original notebooks acquire
# (02_predictors) and exclude from Pearson/VIF (03_multicollinearity) -- kept in
# the final predictor set regardless, used as categorical features
# directly (e.g. in MaxEnt).
CATEGORICAL = ("hydrologic_soil_group", "nlcd_landcover")

SAMPLE_N = 200_000
CORR_THRESHOLD = 0.8
VIF_THRESHOLD = 10.0

# Bivariate characterization (ports the original notebook 02c_predictor_bivariate)
# -- background sample size, matching 02c's own `size=10000`.
BACKGROUND_N = 10_000

# Known-collinear-pair tie-break rationale -- "keep the standard/canonical
# variable, drop its redundant partner". The first two entries are the
# source notebook's own documented Stage-1 pairs (its
# Stage-1 markdown table) -- consulted by ``_choose_drop`` ONLY if the pair
# is actually flagged by the real Stage-1 Pearson computation below (never
# assumed to fire). Any Stage-1-flagged pair not listed here falls back to a
# generic, pair-agnostic rule -- see ``_choose_drop``'s else branch.
#
# The third entry (``flow_accumulation``/``twi``) is DIFFERENT in kind: it is
# never a Stage-1 hit (their pairwise Pearson r=0.768 stays under the 0.8
# threshold even post log-transform -- only the multi-predictor VIF sees the
# real collinearity)
# and is instead consulted by ``_choose_vif_drop`` (Stage 2, below) -- see
# that function's docstring for how the SAME "keep the standard one"
# principle applies at the VIF stage, and why TWI -- not
# flow_accumulation -- is the one kept.
_KNOWN_PAIR_RATIONALE = {
    frozenset({"slope", "tri"}): (
        "slope",
        "at 10 m resolution TRI is almost a rescaled slope here; slope is "
        "the standard hydrologic predictor and feeds directly into "
        "TWI/SPI/curve_number elsewhere in the stack",
    ),
    frozenset({"curvature", "tpi"}): (
        "curvature",
        "curvature is the more direct measure of flow convergence/"
        "divergence (concave = water-accumulating) than TPI's relative "
        "elevation position",
    ),
    frozenset({"flow_accumulation", "twi"}): (
        "twi",
        "among the collinear post-log accumulation-derivatives "
        "(flow_accumulation, spi, twi), retain TWI as the standard "
        "topographic wetness index used in flood-susceptibility studies -- "
        "it integrates slope directly into a single index, unlike the bare "
        "accumulation layers -- and drop flow_accumulation (here) and SPI "
        "(Stage 1 -- see the flow_accumulation/spi entry in decisions "
        "above) as redundant",
    ),
}


def _load_stack(pred_dir=PRED_DIR, sample_n=SAMPLE_N, random_state=config.GLOBAL_SEED):
    """Load every ``*.tif`` under ``pred_dir``, flatten each to one pixel
    per row, keep only pixels valid (non-NaN) across EVERY layer, then draw
    a reproducible random subsample of ``sample_n`` valid pixels. Faithful
    port of the original notebook's load+sample cell: raw
    (un-transformed) values, ``df.dropna()`` requires ALL columns --
    including the 2 categorical layers -- to be non-NaN, and
    ``random_state=42`` matches the source exactly.

    Returns the FULL stack (continuous AND categorical columns) as one
    DataFrame -- ``multicollinearity()`` is the one that splits them.
    """
    files = sorted(Path(pred_dir).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"[screen] no predictor rasters found under {pred_dir}")

    arrays = {}
    for f in files:
        da = rioxarray.open_rasterio(f).squeeze("band", drop=True)
        arrays[f.stem] = da.values

    columns = list(arrays.keys())
    stacked = np.stack([arrays[name].ravel() for name in columns], axis=1)
    df = pd.DataFrame(stacked, columns=columns)

    valid = df.dropna()
    n = min(sample_n, len(valid))
    return valid.sample(n=n, random_state=random_state)


def _choose_drop(a, b, corr):
    """Pick which of a Stage-1-flagged pair ``(a, b)`` to drop.

    Uses the source notebook's own documented rationale for the two
    SPECIFIC pairs it names (``_KNOWN_PAIR_RATIONALE``) when ``{a, b}``
    matches one of them. For any other pair -- i.e. one the source never
    analyzed, such as a new >=0.8 hit the corrected hydrology might have
    introduced -- falls back to a standard, pair-agnostic multicollinearity
    heuristic: drop whichever variable is MORE redundant with the rest of
    the continuous stack (larger mean |r| against every other continuous
    column). This is the same "drop the more broadly correlated variable"
    default rule R's ``caret::findCorrelation`` uses, so an unanticipated
    pair is still resolved by a principled, citable method rather than an
    arbitrary/alphabetical choice.
    """
    key = frozenset({a, b})
    if key in _KNOWN_PAIR_RATIONALE:
        keep, _why = _KNOWN_PAIR_RATIONALE[key]
        return b if keep == a else a

    others = [c for c in corr.columns if c not in (a, b)]
    mean_abs_a = corr.loc[a, others].abs().mean()
    mean_abs_b = corr.loc[b, others].abs().mean()
    if mean_abs_a != mean_abs_b:
        return a if mean_abs_a > mean_abs_b else b
    return max(a, b)  # deterministic last-resort tie-break; not expected to fire


def _apply_modeling_transforms(stack, continuous_cols):
    """Return a COPY of ``stack`` with ``pipeline.predictors.
    apply_modeling_transform()`` applied to every column in
    ``continuous_cols`` -- log10 for ``flow_accumulation``, log1p for
    ``spi``, identity (unchanged, just cast to float64) for every other
    continuous column. ``stack`` itself is left untouched; the categorical
    columns (not in ``continuous_cols``) are also left untouched, copied
    over verbatim.

    Exists so ``multicollinearity()`` can assess Stage 1 Pearson / Stage 2
    VIF on the same values MaxEnt actually models (module docstring,
    "UPDATE" -- log-transform adoption) via ONE call site, rather than
    duplicating the per-column transform-dispatch logic that already lives
    in ``pipeline.predictors.apply_modeling_transform``.
    """
    transformed = stack.copy()
    for col in continuous_cols:
        transformed[col] = predictors.apply_modeling_transform(col, stack[col].to_numpy())
    return transformed


def _stage1_pearson(stack, continuous_cols, threshold=CORR_THRESHOLD):
    """Stage 1: pairwise Pearson |r| >= ``threshold`` -> drop one of each
    pair. Mirrors the original notebook's cells 3-4 ALGORITHM
    exactly -- ``method='pearson'`` over whatever numeric values ``stack``
    holds; this function itself applies no transform of its own.
    ``multicollinearity()`` (the only caller) now passes it the MODELING-
    transformed stack (``pipeline.predictors.apply_modeling_transform()`` --
    log10(``flow_accumulation``), log1p(``spi``), identity elsewhere), not
    raw physical units, per the log-transform adoption (module docstring,
    "UPDATE").

    Flagged pairs are resolved strongest-|r|-first, skipping any pair whose
    partner is already dropped (so if three-plus variables are mutually
    over-threshold, one drop can resolve more than one pair without
    double-processing).

    Returns ``(corr, dropped, decisions)``: ``corr`` is the full Pearson
    matrix over ``continuous_cols`` (survivors and drops alike); ``dropped``
    is the ordered list of column names Stage 1 removes; ``decisions`` is
    ``[(a, b, r, dropped_name), ...]`` for the report/audit trail.
    """
    corr = stack[continuous_cols].corr(method="pearson")

    pairs = []
    for i, a in enumerate(continuous_cols):
        for b in continuous_cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append((a, b, r))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)

    dropped = []
    decisions = []
    for a, b, r in pairs:
        if a in dropped or b in dropped:
            continue
        d = _choose_drop(a, b, corr)
        dropped.append(d)
        decisions.append((a, b, r, d))

    return corr, dropped, decisions


def _choose_vif_drop(vifs, working_cols):
    """Pick which column ``_stage2_vif`` removes THIS iteration.

    Defaults to the mechanical rule every prior version of this function
    used unconditionally -- ``vifs.index[0]``, the single highest-VIF
    column (``vifs`` is already sorted descending by the caller) -- and
    still returns exactly that for the overwhelming majority of columns/
    iterations. The ONE exception: if that top column is the documented
    "keep" side of a ``_KNOWN_PAIR_RATIONALE`` entry (the same module-level
    table Stage 1's ``_choose_drop`` already consults) AND that entry's
    OTHER member is ALSO still present in ``working_cols``, drop the OTHER
    member instead -- the known-collinear partner actually driving the top
    column's inflated VIF -- preserving the standard/canonical predictor.
    This is the identical "keep-the-standard-one" principle ``_choose_drop``
    already applies at Stage 1 (e.g. keep ``slope``, drop ``tri``), simply
    extended to Stage 2: the ``flow_accumulation``/``twi`` pair (log-
    transform adoption cascade) never crosses the Stage-1 Pearson threshold
    on its own (r=0.768 < 0.8) -- it only shows up as a multi-predictor VIF
    effect once ``flow_accumulation`` shares TWI's own log-like scale, so
    Stage 1 never gets a chance to apply this tie-break and Stage 2 must.

    This can never fire for the two PRE-EXISTING ``_KNOWN_PAIR_RATIONALE``
    entries (``slope``/``tri``, ``curvature``/``tpi``): both ``tri`` and
    ``tpi`` are always removed at Stage 1 (their Pearson r against
    slope/curvature is far over 0.8), so neither can still be a member of
    ``working_cols`` by the time Stage 2 runs -- the ``partner in
    working_cols`` guard below is always False for those two entries in
    practice (see
    test_choose_vif_drop_never_fires_for_pre_existing_stage1_only_pairs).

    Parameters
    ----------
    vifs : the current iteration's VIF ``Series``, sorted descending
        (as built by ``_stage2_vif``'s own loop) -- ``vifs.index[0]`` is
        the mechanical top.
    working_cols : the current iteration's surviving column list (same
        object ``_stage2_vif`` is about to call ``.remove()`` on) -- used
        only to check whether a known partner is still present.

    Returns
    -------
    The column name to remove -- either ``vifs.index[0]`` (default) or its
    known partner (override case above).
    """
    top = vifs.index[0]
    for pair, (keep, _why) in _KNOWN_PAIR_RATIONALE.items():
        if keep == top:
            partner = next(iter(pair - {top}))
            if partner in working_cols:
                return partner
    return top


def _stage2_vif(stack, cols, threshold=VIF_THRESHOLD):
    """Stage 2: iteratively compute VIF over ``cols``, drop one column if
    the highest VIF exceeds ``threshold``, repeat until every remaining VIF
    <= ``threshold``. Mirrors the original notebook's cell 5
    ALGORITHM (iterative highest-VIF removal) -- WHICH column is removed
    each iteration is ``_choose_vif_drop``'s job (usually, but not always,
    the literal highest -- see that function's docstring for the one
    documented override).

    Returns ``(working_cols, history)``: ``working_cols`` is the surviving
    column list; ``history`` is the list of per-iteration VIF ``Series``
    (index = that iteration's working columns) -- its LAST entry is the
    final VIF table, whose ``iloc[0]`` is guaranteed <= ``threshold``.
    """
    working_cols = list(cols)
    history = []
    while True:
        X = add_constant(stack[working_cols])
        vifs = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
            index=working_cols,
        ).sort_values(ascending=False)
        history.append(vifs)
        if vifs.iloc[0] <= threshold:
            break
        working_cols.remove(_choose_vif_drop(vifs, working_cols))
    return working_cols, history


def cramers_v(a, b, bins=10, *, a_nominal=False, b_nominal=False):
    """Cramer's V association between ``a`` and ``b`` -- the statistic
    actually defined for a categorical-categorical or categorical-
    continuous pair (unlike Pearson r, which assumes both variables are at
    least ordinal-numeric; see this module's docstring, "Fix vs source").
    Returns a value in [0, 1]: 0 = independent, 1 = perfect association.

    Each side is handled independently, per its own ``*_nominal`` flag:
      - NOMINAL (``a_nominal=True`` / ``b_nominal=True``): used as-is, at
        its NATIVE values, regardless of how many distinct values it has.
        The caller must set this explicitly for any side it KNOWS is a
        categorical class-code column (e.g. the 15 NLCD legend codes in
        ``nlcd_landcover``) -- cardinality alone cannot tell "15 nominal
        classes" apart from "15 unlucky quantile bins of a continuous
        variable," and guessing from cardinality used to silently
        rebin any nominal layer with more than ``bins`` codes by raw
        code VALUE, discarding its true category structure (the bug this
        parameter fixes).
      - continuous (the default, ``*_nominal=False``, unchanged behavior):
        discretized into ``bins`` equal-frequency (quantile) bins if it has
        more than ``bins`` unique values, so a contingency table is
        well-defined; used as-is if it already has <= ``bins`` unique
        values. Binning a genuinely CONTINUOUS partner into quantiles for
        the contingency table is correct and is exactly what this branch
        is for -- only a side actually known to be nominal must opt out of
        it via the flag above.

    Pairwise NaNs (in either input) are dropped before building the table.

    Uses the classic (uncorrected) formula
    ``V = sqrt(chi2 / (n * (min(r, k) - 1)))``, where ``chi2`` is the
    table's Pearson chi-square statistic (``scipy.stats.chi2_contingency``,
    called with ``correction=False`` so Yates' continuity correction --
    applied by default on 2x2 tables -- can never silently diverge from the
    classic formula this docstring claims) and ``r``/``k`` are its
    row/column counts -- not Bergsma's bias-corrected variant, since the
    goal here is to characterize association strength for a supplementary
    report, not to compare across wildly different sample sizes or table
    shapes.

    A thin wrapper around ``_cramers_v_stats`` (below), which builds the
    exact same table/statistic but also returns the chi-square value and
    its p-value -- needed by ``bivariate()``'s categorical rows,
    so that caller can get its test's p-value from the SAME computation
    this function already does, instead of a second, independently-built
    contingency table.
    """
    v, _chi2, _p, _table = _cramers_v_stats(a, b, bins=bins, a_nominal=a_nominal, b_nominal=b_nominal)
    return v


def _cramers_v_stats(a, b, bins=10, *, a_nominal=False, b_nominal=False):
    """Shared core behind ``cramers_v()``: builds the same (optionally
    nominal-aware, see ``cramers_v``'s docstring) contingency table and
    returns ``(v, chi2, p, table)`` instead of just ``v``.

    Exists so a caller that needs the chi-square test's p-value alongside
    the effect size -- ``bivariate()``'s categorical rows, which report a
    Bonferroni-corrected ``p_raw`` for EVERY predictor including the two
    categorical ones -- gets it from this EXACT table/statistic, not a
    second, separately-built contingency table that could silently diverge
    from what ``cramers_v()`` itself reports (the whole point: reuse the
    already-fixed cramers_v, not a fresh rebinning path).
    """
    a = pd.Series(np.asarray(a).ravel())
    b = pd.Series(np.asarray(b).ravel())
    ok = a.notna().to_numpy() & b.notna().to_numpy()
    a, b = a[ok], b[ok]

    def _discretize(s, nominal):
        if nominal:
            return s
        if s.nunique() > bins:
            return pd.qcut(s, q=bins, duplicates="drop")
        return s

    table = pd.crosstab(_discretize(a, a_nominal), _discretize(b, b_nominal))
    chi2, p, _dof, _expected = chi2_contingency(table, correction=False)
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    v = float("nan") if denom <= 0 else float(np.sqrt(chi2 / denom))
    return v, float(chi2), float(p), table


def _categorical_association_report(stack, columns, categorical):
    """Cramer's V of each ``categorical`` layer against every OTHER column
    in ``stack`` (continuous columns and the other categorical layer
    alike) -- the genuine replacement for the source notebook's fabricated
    "Pearson r" claims against categorical layers (module docstring).

    Every ``cramers_v`` call here declares its nominal side(s) explicitly
    instead of letting cardinality guess: the row's own ``cat`` layer is
    always passed with ``a_nominal=True`` (native class codes, never
    quantile-rebinned no matter how many distinct codes it has -- e.g.
    ``nlcd_landcover``'s 15 NLCD legend codes), and ``other`` is ALSO
    passed ``b_nominal=True`` whenever it is itself one of the
    ``categorical`` layers (the ``nlcd_landcover`` x
    ``hydrologic_soil_group`` cell) -- a genuinely continuous ``other``
    still gets the correct quantile-bin treatment via the default
    ``b_nominal=False``. This is what guarantees ``nlcd_landcover`` and
    ``hydrologic_soil_group`` are never silently quantile-rebinned by raw
    code value, whether compared against each other or against any
    continuous layer (the bug ``cramers_v``'s
    cardinality-only heuristic used to cause).

    Square-ish DataFrame: index = ``categorical`` (sorted), columns = every
    other column name, values = ``cramers_v``. A categorical layer's
    association with itself is left as NaN (undefined/not meaningful here,
    not fudged to 1.0 -- that would conflate "identical to itself" with
    genuine cross-variable association).
    """
    cats = sorted(categorical)
    rows = {}
    for cat in cats:
        row = {}
        for other in columns:
            if other == cat:
                row[other] = np.nan
            else:
                row[other] = cramers_v(
                    stack[cat], stack[other],
                    a_nominal=True, b_nominal=other in categorical,
                )
        rows[cat] = row
    return pd.DataFrame(rows).T


def multicollinearity(stack, categorical=CATEGORICAL, corr_threshold=CORR_THRESHOLD,
                       vif_threshold=VIF_THRESHOLD, pred_dir=PRED_DIR,
                       screened_dir=SCREENED_DIR, report_path=REPORT_PATH,
                       categorical_report_path=CATEGORICAL_REPORT_PATH,
                       copy_rasters=True):
    """Two-stage multicollinearity screen over ``stack`` (as returned by
    ``_load_stack()``).

    Before Stage 1/Stage 2 run, every CONTINUOUS column is passed through
    ``pipeline.predictors.apply_modeling_transform()`` (log10 for
    ``flow_accumulation``, log1p for ``spi``, identity for every other
    layer) to build an internal, MODELING-transformed copy -- collinearity
    is assessed on the same values the model (MaxEnt) actually uses, not
    raw physical units spanning 6+ orders of magnitude for those two
    hydrology layers (log-transform adoption; module docstring, "UPDATE").
    ``stack`` itself is never mutated by this. ``_categorical_association_
    report`` (below) still runs on the untransformed ``stack`` -- Cramer's
    V's quantile-binning is invariant to any monotone transform, so the two
    would produce identical numbers either way (see this module's
    docstring).

    Stage 1 drops one of each Pearson |r| >= ``corr_threshold`` pair among
    the CONTINUOUS predictors (``_stage1_pearson``); Stage 2 iteratively
    drops the highest-VIF continuous predictor while any remains >
    ``vif_threshold`` (``_stage2_vif``). The ``categorical`` layers never
    enter either stage -- Pearson/VIF are undefined for nominal class codes
    -- and are re-attached to ``kept`` unconditionally at the end.

    Returns ``(kept, report)``:
      - ``kept``: sorted list of surviving predictor names -- the
        continuous Stage-1/2 survivors plus the ``categorical`` layers.
      - ``report``: the Pearson correlation matrix over ALL continuous
        columns on the MODELING-transformed values described above
        (survivors and drops alike, so the r that caused a drop stays
        visible), with two added columns -- ``VIF`` (NaN for any Stage-1
        drop, since those never reached Stage 2) and ``dropped`` (bool,
        True for every Stage-1 or Stage-2 drop). ``report.index``/
        ``report.columns`` never include a categorical layer -- see this
        module's docstring, "Fix vs source".

    Side effects (mirrors the original notebook's final cell; pass
    ``copy_rasters=False`` for a fast, in-memory-only call that skips the
    raster copy):
      - writes ``report`` to ``report_path``;
      - writes the categorical-association table (``cramers_v`` of each
        ``categorical`` layer against every other layer) to
        ``categorical_report_path`` -- see module docstring, never merged
        into ``report`` itself;
      - if ``copy_rasters``, clears ``screened_dir`` of any stale
        ``*.tif`` and copies the ``kept`` rasters into it from
        ``pred_dir``.
    """
    columns = list(stack.columns)
    continuous_cols = [c for c in columns if c not in categorical]

    modeling_stack = _apply_modeling_transforms(stack, continuous_cols)

    corr, stage1_drop, _decisions = _stage1_pearson(modeling_stack, continuous_cols, corr_threshold)
    stage1_cols = [c for c in continuous_cols if c not in stage1_drop]

    working_cols, vif_history = _stage2_vif(modeling_stack, stage1_cols, vif_threshold)
    stage2_drop = [c for c in stage1_cols if c not in working_cols]

    dropped_continuous = stage1_drop + stage2_drop
    kept = sorted(working_cols + list(categorical))

    # vif_history[-1]'s index is already `working_cols` (the final
    # survivors); reindexing onto the FULL `continuous_cols` here makes the
    # NaN-for-every-dropped-column fill explicit rather than relying on
    # pandas' implicit index alignment on the `report["VIF"] = ...`
    # assignment below (same result either way).
    final_vif = vif_history[-1].reindex(continuous_cols)

    report = corr.copy()
    report["VIF"] = final_vif
    report["dropped"] = report.index.isin(dropped_continuous)

    cat_report = _categorical_association_report(stack, columns, categorical)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path)

    categorical_report_path = Path(categorical_report_path)
    categorical_report_path.parent.mkdir(parents=True, exist_ok=True)
    cat_report.to_csv(categorical_report_path)

    if copy_rasters:
        screened_dir = Path(screened_dir)
        screened_dir.mkdir(parents=True, exist_ok=True)
        for f in screened_dir.glob("*.tif"):
            f.unlink()
        for name in kept:
            shutil.copy(Path(pred_dir) / f"{name}.tif", screened_dir / f"{name}.tif")

    return kept, report


# ============================================================================
# Bivariate characterization -- ports the original notebook
# 02c_predictor_bivariate. NOT part of the multicollinearity screen
# above; a separate, purely descriptive presence-vs-background contrast run
# AFTER it, over whatever stack `multicollinearity()` kept (02c's own
# `PRED_DIR` already pointed at `predictors_screened`, not the raw stack --
# see `bivariate`'s docstring).
#
# Framing (required -- the explicit "this is NOT validation" label):
# `bivariate()` characterizes how presence
# points statistically differ from a random background sample. It is NOT a
# measure of model predictive performance, is not computed from any fitted
# model, and must never be cited as a validation metric -- see
# `BIVARIATE_FRAMING`/the `df.attrs` it's stored under below.
# ============================================================================

BIVARIATE_KIND = "characterization"
BIVARIATE_FRAMING = (
    "Descriptive presence-vs-background statistical contrast (Mann-Whitney U "
    "/ rank-biserial r for continuous predictors, chi-square / Cramer's V "
    "for categorical predictors), Bonferroni-corrected across every "
    "predictor actually tested. This is NOT model validation, NOT a measure "
    "of predictive skill, and NOT a susceptibility estimate -- it only "
    "describes whether/how each predictor's marginal distribution differs "
    "between the observed presence points and a random background sample. "
    "See pipeline.screen.bivariate's docstring."
)


def _load_arrays_2d(pred_dir):
    """Load every ``*.tif`` under ``pred_dir`` as a 2-D ``(y, x)`` float64
    numpy array (nodata -> NaN), plus the affine transform shared by the
    whole aligned stack. Mirrors 02c's own ``ref``/``arrays``/``valid_mask``
    cells: every layer already shares one grid (``pipeline.predictors``'s
    ``reproject_match`` alignment guarantee), so the FIRST file's transform
    is authoritative for all.

    Returns ``(arrays, transform, valid_mask)``: ``arrays`` maps name -> 2-D
    ndarray; ``valid_mask`` is True where EVERY layer is finite (used only
    to draw the background sample -- presence-side finiteness is checked
    per-predictor inside ``bivariate()``, matching 02c's own per-predictor
    ``np.isfinite`` filter rather than requiring simultaneous validity
    across every layer).
    """
    files = sorted(Path(pred_dir).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"[screen] bivariate: no predictor rasters found under {pred_dir}")

    ref = rioxarray.open_rasterio(files[0]).squeeze("band", drop=True)
    transform = ref.rio.transform()

    arrays = {}
    valid_mask = None
    for f in files:
        da = rioxarray.open_rasterio(f).squeeze("band", drop=True)
        arr = da.values.astype("float64")
        nd = da.rio.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        arrays[f.stem] = arr
        finite = np.isfinite(arr)
        valid_mask = finite if valid_mask is None else (valid_mask & finite)

    return arrays, transform, valid_mask


def _xy_to_rowcol(x, y, transform):
    """Invert an axis-aligned, north-up affine ``transform`` to map
    projected x/y coordinates onto integer ``(row, col)`` pixel indices.
    Ported verbatim from 02c's own ``xy_to_rowcol`` -- floor-based, matching
    how rasterio/GDAL define pixel membership (a coordinate exactly on a
    cell boundary belongs to the cell whose extent starts there).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    col = np.floor((x - transform.c) / transform.a).astype(int)
    row = np.floor((y - transform.f) / transform.e).astype(int)
    return row, col


def _draw_background(valid_mask, n=BACKGROUND_N, random_state=config.GLOBAL_SEED):
    """Draw ``n`` random valid-pixel ``(row, col)`` locations without
    replacement from ``valid_mask`` (True where every layer is finite),
    seeded via ``np.random.default_rng(random_state)``. Ported verbatim
    from 02c's own background-sample cell (``np.random.default_rng(42)``,
    ``size=10_000``, ``replace=False``) -- the presence-vs-background
    contrast ``bivariate()`` characterizes depends entirely on this sample,
    so determinism here (same ``random_state`` -> identical ``(row, col)``
    draw, every call) is load-bearing for the whole characterization being
    reproducible.
    """
    rng = np.random.default_rng(random_state)
    valid_mask = np.asarray(valid_mask)
    vi = np.flatnonzero(valid_mask.ravel())
    n = min(n, len(vi))
    bg_flat = rng.choice(vi, size=n, replace=False)
    return np.unravel_index(bg_flat, valid_mask.shape)


def _mwu_rank_biserial(presence_vals, background_vals):
    """Two-sided Mann-Whitney U + rank-biserial r = (2*U/(n1*n2)) - 1.

    Positive r => presence distribution shifted HIGHER than background;
    negative r => shifted LOWER (``direction``, below, restates this same
    sign as a label). Pulled out of ``bivariate()``'s continuous-predictor
    branch into its own pure helper specifically so this SIGN CONVENTION --
    which sets the ``direction`` of every continuous predictor's row in the
    paper-facing bivariate table -- can be pinned against a hand-computable
    ground truth (tests/test_bivariate.py) independently of the real
    on-disk predictor stack. A bare ``r in [-1, 1]`` / ``direction in
    {"higher", "lower"}`` check (this module's original test coverage)
    would still pass even if the formula were silently inverted (e.g.
    ``r = 1 - 2*U/(n1*n2)``, a real alternative convention in the
    literature) -- only a known-sign synthetic case catches that.

    Parameters
    ----------
    presence_vals, background_vals : 1-D array-likes of finite floats --
        the two samples compared, in that order (``presence_vals`` is
        ``mannwhitneyu``'s first sample, i.e. its own "n1").

    Returns
    -------
    ``(u, p, r, direction)``: ``u`` is the Mann-Whitney U statistic, ``p``
    its two-sided p-value, ``r`` the rank-biserial effect size (in
    [-1, 1]), and ``direction`` (``"higher"``/``"lower"``) -- ``r``'s own
    sign restated as a label.
    """
    u, p = mannwhitneyu(presence_vals, background_vals, alternative="two-sided")
    r = (2 * u / (len(presence_vals) * len(background_vals))) - 1
    direction = "higher" if r > 0 else "lower"
    return u, p, r, direction


def bivariate(presence, background=None, *, pred_dir=SCREENED_DIR,
              categorical=CATEGORICAL, n_background=BACKGROUND_N,
              random_state=config.GLOBAL_SEED, report_path=BIVARIATE_REPORT_PATH,
              save_csv=True):
    """Presence-vs-background bivariate CHARACTERIZATION -- NOT validation,
    see "Framing" below -- of every predictor raster under ``pred_dir``.

    Faithful port of the original notebook ``02c_predictor_bivariate``: continuous
    predictors get a two-sided Mann-Whitney U test plus rank-biserial
    ``r = (2*U / (n1*n2)) - 1`` (positive => presence values skew HIGHER
    than background, matching the source's own comment); categorical
    predictors (``categorical``, default ``CATEGORICAL`` =
    ``hydrologic_soil_group``/``nlcd_landcover``) get a chi-square test of
    independence between presence/background GROUP MEMBERSHIP and the
    predictor's own native category codes, plus Cramer's V as the effect
    size.

    The categorical branch is computed via THIS module's ``cramers_v``/
    ``_cramers_v_stats`` -- the nominal-aware fix -- with BOTH
    sides passed ``*_nominal=True``: the group-membership side (binary
    presence/background) is trivially nominal, and the predictor's own
    codes side must be nominal explicitly so a >10-class layer like
    ``nlcd_landcover`` is never silently quantile-rebinned by raw code
    value (see ``cramers_v``'s docstring). This is a deliberate departure
    from 02c, which hand-rolled its own contingency-table/chi-square call
    for this branch instead of reusing a shared, already-audited helper --
    the numbers are unchanged (same contingency table -- a 2 x k cross-tab
    of group membership by category code -- and the same classic
    Cramer's-V formula), only the code path is now the single audited
    implementation instead of a second, parallel one.

    Every predictor's raw p-value (``p_raw``) is Bonferroni-corrected over
    ``n_tests = len(df)`` -- the ACTUAL number of predictors this call
    tested, derived from whatever is really on disk under ``pred_dir``,
    never a hardcoded count (see test_screen.py's own ``EXPECTED_KEPT``)
    -- ``bivariate()`` reports whatever that real number is via
    ``df.attrs["n_tests"]``/the returned frame's own length.

    Framing (required, not optional -- the explicit "this is NOT validation"
    label): this function
    characterizes how presence points statistically differ from a random
    background sample -- a descriptive summary for methods/results
    narrative. It is NOT a measure of model predictive performance, is not
    computed from any fitted model, and must never be cited as a
    validation metric. The returned DataFrame encodes this three ways so no
    downstream consumer can lose the distinction by accident:
    ``df.attrs["kind"] == "characterization"``,
    ``df.attrs["not_validation"] is True``, and ``df.attrs["framing"]``
    (the full caveat string, ``BIVARIATE_FRAMING`` above) -- plus this
    docstring and the output CSV's own filename
    (``predictor_bivariate_characterization.csv``, NOT ``..._stats.csv`` as
    in 02c -- deliberately renamed so even a bare ``ls data/processed/``
    can't mistake this for a validation artifact).

    Parameters
    ----------
    presence : DataFrame with numeric ``x``/``y`` columns (projected-CRS
        coordinates matching ``pred_dir``'s grid) -- e.g.
        ``hwm.build_presence()`` or ``data/processed/canonical_presence.csv``
        (the canonical 286-point HW+EOW presence set; deliberately NOT
        ``config.canonical_samples()``'s smaller, predictor-NaN-gated
        subset -- ``bivariate()`` mirrors 02c's own per-predictor
        ``np.isfinite`` drop instead, so each predictor's test uses
        whatever presence points are valid for THAT layer, not only those
        valid across every layer simultaneously).
    background : optional pre-drawn ``(row_idx, col_idx)`` pair (as
        returned by ``_draw_background``) to use verbatim instead of
        drawing a fresh sample -- lets a caller/test reuse or inspect a
        specific draw. If None (default, matches 02c's own behavior), a
        fresh sample of ``n_background`` valid pixels is drawn via
        ``_draw_background`` with ``random_state`` (default
        ``config.GLOBAL_SEED`` = 42).
    pred_dir : directory of ``*.tif`` predictors to characterize. Defaults
        to ``SCREENED_DIR`` (``data/processed/predictors_screened``) --
        matching 02c's own ``PRED_DIR``, which already pointed at the
        SCREENED stack, not the raw 22-layer one -- so ``bivariate()``
        characterizes exactly the predictors that survive
        ``multicollinearity()``, whatever that number empirically is
        (currently 20; see the ``n_tests`` note above).
    categorical : names of predictors to route through the chi-square/
        Cramer's V branch instead of Mann-Whitney/rank-biserial. Defaults
        to ``CATEGORICAL`` (``hydrologic_soil_group``, ``nlcd_landcover``).
    n_background : number of background pixels to draw when ``background``
        is None. Defaults to ``BACKGROUND_N`` (10,000), matching 02c's own
        ``size=10000``.
    random_state : seed passed to ``_draw_background`` when ``background``
        is None. Defaults to ``config.GLOBAL_SEED`` (42) -- same seed in,
        same background draw out, every call (see ``_draw_background``'s
        own docstring).
    report_path : destination CSV path for the ``save_csv`` side effect.
        Defaults to ``BIVARIATE_REPORT_PATH``.
    save_csv : whether to write the returned frame to ``report_path`` as a
        side effect. Defaults to True.

    Returns
    -------
    DataFrame, one row per predictor actually found under ``pred_dir``,
    columns: ``predictor``, ``type`` (``continuous``/``categorical``),
    ``test`` (``Mann-Whitney U``/``chi-square``), ``stat``, ``p_raw``,
    ``effect``, ``effect_metric`` (``rank-biserial r``/``Cramer's V``),
    ``n_pres``, ``n_bg``, ``direction`` (``higher``/``lower`` for
    continuous, ``n/a`` for categorical), ``p_bonferroni``, ``significant``
    (``p_bonferroni < 0.05``) -- sorted by ``p_bonferroni`` ascending,
    matching 02c. ``df.attrs`` carries the characterization/non-validation
    framing described above plus ``n_tests``.

    Side effect: writes the same frame to ``report_path`` (default
    ``BIVARIATE_REPORT_PATH``) if ``save_csv`` (default True) -- a build
    artifact, not committed (mirrors ``multicollinearity``'s own CSV
    outputs).
    """
    arrays, transform, valid_mask = _load_arrays_2d(pred_dir)
    nrows, ncols = valid_mask.shape

    pres_row, pres_col = _xy_to_rowcol(presence["x"].to_numpy(), presence["y"].to_numpy(), transform)
    in_bounds = (pres_row >= 0) & (pres_row < nrows) & (pres_col >= 0) & (pres_col < ncols)
    n_out = int((~in_bounds).sum())
    if n_out:
        print(f"[screen] bivariate: {n_out}/{len(in_bounds)} presence points fall outside "
              f"the predictor grid -- excluded")
    pres_row, pres_col = pres_row[in_bounds], pres_col[in_bounds]

    if background is None:
        bg_row, bg_col = _draw_background(valid_mask, n=n_background, random_state=random_state)
    else:
        bg_row, bg_col = background

    rows = []
    for name, arr in arrays.items():
        pv = arr[pres_row, pres_col]
        bv = arr[bg_row, bg_col]
        pv = pv[np.isfinite(pv)]
        bv = bv[np.isfinite(bv)]

        if name in categorical:
            group = np.concatenate([np.ones(len(pv), dtype=int), np.zeros(len(bv), dtype=int)])
            codes = np.concatenate([pv, bv]).astype(int)
            v, chi2, p, _table = _cramers_v_stats(group, codes, a_nominal=True, b_nominal=True)
            rows.append(dict(predictor=name, type="categorical", test="chi-square",
                              stat=chi2, p_raw=p, effect=v, effect_metric="Cramer's V",
                              n_pres=len(pv), n_bg=len(bv), direction="n/a"))
        else:
            u, p, r, direction = _mwu_rank_biserial(pv, bv)
            rows.append(dict(predictor=name, type="continuous", test="Mann-Whitney U",
                              stat=u, p_raw=p, effect=r, effect_metric="rank-biserial r",
                              n_pres=len(pv), n_bg=len(bv),
                              direction=direction))

    df = pd.DataFrame(rows)
    n_tests = len(df)
    df["p_bonferroni"] = (df["p_raw"] * n_tests).clip(upper=1.0)
    df["significant"] = df["p_bonferroni"] < 0.05
    df = df.sort_values("p_bonferroni").reset_index(drop=True)

    df.attrs["kind"] = BIVARIATE_KIND
    df.attrs["not_validation"] = True
    df.attrs["framing"] = BIVARIATE_FRAMING
    df.attrs["n_tests"] = n_tests

    if save_csv:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)

    return df

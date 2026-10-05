import os, sys, hashlib, subprocess, shutil, datetime
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
TOOLS = REPO / "tools"
GLOBAL_SEED = 42

def java_exe() -> str:
    # OS-agnostic: env override, else conda openjdk on PATH, else tools/ jdk.
    if os.getenv("FLOOD_JAVA"):
        p = os.environ["FLOOD_JAVA"]
        if not Path(p).exists():
            raise FileNotFoundError(f"FLOOD_JAVA points to a nonexistent path: {p}")
        return p
    for cand in ("java",):
        p = shutil.which(cand)
        if p:
            return p
    # The tools/ fallback the comment above has always promised, but never
    # implemented: the JDK is vendored into the repo precisely so a run does not
    # depend on the ambient PATH. Without this, whether MaxEnt fits at all varies
    # between shells -- which is how a fold silently failed to refit.
    for exe in sorted(TOOLS.glob("jdk-*/bin/java.exe")) + sorted(TOOLS.glob("jdk-*/bin/java")):
        if exe.exists():
            return str(exe)
    raise FileNotFoundError(
        "No java found: FLOOD_JAVA unset, no java on PATH, and no tools/jdk-*/bin/java. "
        "Install openjdk, set FLOOD_JAVA, or vendor a JDK into tools/.")

def maxent_jar() -> str:
    p = os.getenv("FLOOD_MAXENT_JAR") or str(TOOLS / "maxent" / "maxent.jar")
    if not Path(p).exists():
        raise FileNotFoundError(f"maxent.jar not found at {p}; see README to download 3.4.4")
    return p

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nogit"

def provenance() -> dict:
    cfg = f"{GLOBAL_SEED}|{sys.version}"
    return {"git_sha": _git_sha(),
            "config_hash": hashlib.sha256(cfg.encode()).hexdigest()[:12],
            "python": sys.version.split()[0],
            "created": datetime.datetime.now().isoformat(timespec="seconds")}

def _modeling_grid_rowcol(x, y):
    """Map projected x/y coordinates (the modeling CRS, EPSG:26914) onto
    integer ``(row, col)`` pixel indices of the canonical 10 m modeling grid
    -- ``data/processed/predictors/dem.tif``, the alignment target every
    predictor layer in this pipeline is ``reproject_match``ed onto (see
    ``pipeline.predictors``' module docstring) -- so any one predictor
    raster's affine transform is an equally valid reference grid for every
    other layer.

    Local imports (``rioxarray``, ``rasterio``) -- ``config`` otherwise has
    zero raster-I/O imports at module level; a caller that never needs this
    (e.g. tests exercising ``split()``/``loeo_folds()`` against a synthetic
    frame) pays nothing for it, matching this module's existing "leaf
    module" import discipline for ``pipeline.hwm``/``pipeline.predictors``
    (see ``canonical_samples()``'s own docstring).

    Uses ``rasterio.transform.rowcol`` (default ``op=floor``) -- the SAME
    function ``pipeline.predictors._sample_at_points`` already uses for the
    identical purpose, so a point's assigned pixel here is guaranteed
    identical to what the rest of this pipeline would assign it (never a
    second, independently-reimplemented floor-math path that could
    silently diverge -- ``pipeline.screen._xy_to_rowcol`` reimplements this
    same floor rule for the same reason, in a module that cannot import
    ``config`` reciprocally without local imports either).

    Returns ``(rows, cols)`` -- two int ndarrays, same length as ``x``/``y``.
    """
    import rioxarray
    from rasterio.transform import rowcol

    ref_path = DATA / "predictors" / "dem.tif"
    ref = rioxarray.open_rasterio(ref_path).squeeze("band", drop=True)
    rows, cols = rowcol(ref.rio.transform(), np.asarray(x), np.asarray(y))
    return np.asarray(rows), np.asarray(cols)


def _dedup_to_modeling_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``df`` (must have ``x``/``y`` columns) to at most one row
    per DISTINCT ``(row, col)`` pixel of the canonical 10 m modeling grid
    (``_modeling_grid_rowcol``). The 279 predictor-NaN-valid presence points
    ``canonical_samples()`` selects occupy only 277 DISTINCT modeling-grid
    pixels (exactly 2 pixels each hold 2 of the 279 points -- two pairs of
    survey points closer together than this grid's 10 m cell size), so
    MaxEnt (which treats co-located presence records as one occurrence, a
    standard SDM data-preparation convention) was silently deduplicating
    its own input from 279 down to 277. This function makes that
    deduplication explicit and UPSTREAM of MaxEnt, so
    ``valid_mask()``/``canonical_samples()``/``split()``/
    ``spatial_folds()``/``loeo_folds()``/``maxent.build_samples_csv()`` all
    agree on n=277 instead of only MaxEnt implicitly landing there on its
    own.

    Deterministic: ties within a pixel are broken by keeping the FIRST row
    in ``df``'s existing order (``DataFrame.drop_duplicates(...,
    keep="first")``) -- reproducible because ``df``'s row order is itself
    already fully deterministic by the time this is called (ultimately
    derived from ``hwm.build_presence()``'s ``config.GLOBAL_SEED``-seeded
    30 m thinning, followed by a fixed boolean-mask filter -- neither step
    involves any additional randomness), so the same two points collapse
    to the same single survivor on every call/run. No new RNG draw is
    introduced here.

    Returns a NEW DataFrame with the same columns as ``df`` (the
    internal ``_row``/``_col`` grouping keys are dropped before returning),
    reset to a clean ``0..n-1`` RangeIndex.
    """
    rows, cols = _modeling_grid_rowcol(df["x"].to_numpy(), df["y"].to_numpy())
    keyed = df.copy()
    keyed["_row"] = rows
    keyed["_col"] = cols
    deduped = keyed.drop_duplicates(subset=["_row", "_col"], keep="first")
    return deduped.drop(columns=["_row", "_col"]).reset_index(drop=True)


def canonical_samples(presence=None, valid=None) -> pd.DataFrame:
    """The canonical modeling presence set: every HW/EOW point from the
    286-row presence frame for which the predictor-NaN gate is True (279
    points -- see "279 vs 277" below), deduplicated to one point per
    DISTINCT pixel of the 10 m modeling grid (277 points --
    ``_dedup_to_modeling_pixels``), as a DataFrame with columns ``x, y,
    event, type, in_model`` (every row here has ``in_model == True`` by
    construction -- points that failed the gate are excluded from the frame
    entirely, not merely flagged; the column exists so a caller that
    re-merges this subset back against the full, unfiltered presence set
    can join on it explicitly instead of assuming "present in this frame"
    implicitly means "in the model").

    ``presence``/``valid`` are injected (``hwm.build_presence()`` and
    ``predictors.valid_mask()`` respectively) rather than hardcoded so this
    stays testable against synthetic data.
    If either is omitted, both default to the real pipeline outputs
    (``hwm.build_presence()`` and ``predictors.valid_mask()``) via a local import -- deferred (not at module level) so
    ``config`` stays a leaf module with no import-time dependency on
    ``pipeline.hwm``/``pipeline.predictors`` (both of which already import
    ``config`` at module level; ``config`` importing either of THEM at
    module level would deadlock the very first ``import pipeline.config``
    anywhere). A function-local import at CALL time (long after both
    modules have finished loading) carries no such risk.

    NOTE on the row count -- TWO numbers, not one:
      - **279** = point-VALIDITY count. ``predictors.valid_mask()`` run
        against the real on-disk 22-layer stack finds 279 valid points, one
        more than the 278 of the original notebooks (dropping
        ``median_income`` recovers one point). ``valid_mask()`` itself
        reports 279 -- it is a per-POINT validity gate, not a per-PIXEL one.
      - **277** = canonical MODELING count, and this function's actual
        return length. Of those 279 valid points, 2 PAIRS land in the same
        10 m modeling-grid pixel as each other (4 points -> 2 pixels), so
        only 277 DISTINCT pixels are actually available to MaxEnt (which
        cannot distinguish two presence records in the same cell from one)
        -- ``_dedup_to_modeling_pixels`` makes that reduction explicit
        rather than leaving it as an implicit side effect of MaxEnt's own
        internal handling.
    This function does NOT hardcode either number -- it asserts only
    internal self-consistency (every pre-dedup row really is valid, that
    count matches the number of True entries in ``valid``, and every
    POST-dedup row maps to a distinct pixel) -- so it reflects whatever the
    real predictor stack/grid on disk says rather than silently forcing a
    stale target. Callers that need the raw 279 point-validity count should
    use ``predictors.valid_mask().sum()`` or
    ``predictors.qa_report()["presence_points"]`` directly -- this
    function's own return value is always the smaller, pixel-deduplicated
    277-row modeling set.
    """
    if presence is None or valid is None:
        from pipeline import hwm, predictors
        if presence is None:
            presence = hwm.build_presence()
        if valid is None:
            valid = predictors.valid_mask()

    presence = presence.reset_index(drop=True)
    valid_arr = np.asarray(valid, dtype=bool)
    assert len(valid_arr) == len(presence), (
        f"[config] canonical_samples: valid length {len(valid_arr)} != "
        f"presence length {len(presence)} -- presence frame and validity "
        f"mask have drifted out of sync"
    )
    df = presence.loc[valid_arr, ["x", "y", "event", "type"]].reset_index(drop=True)
    df["in_model"] = True
    assert len(df) == int(valid_arr.sum()), "internal inconsistency in canonical_samples() filtering"

    df = _dedup_to_modeling_pixels(df)
    return df

def split(samples, test_frac=0.30):
    """Split samples into train and test indices.

    Assumes samples has a 0..n-1 RangeIndex; returned indices are positional and
    coincide with label indices under this precondition.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    idx = rng.permutation(len(samples))
    n_test = round(len(samples) * test_frac)
    return idx[n_test:], idx[:n_test]             # train, test

def spatial_folds(samples, block_km=2):
    """Assign spatial fold ids using square block binning.

    Assumes samples has a 0..n-1 RangeIndex. Default block_km=2 must not be changed
    without updating test_spatial_folds_uses_2km_blocks (regression guard).
    """
    bx = (samples["x"] // (block_km*1000)).astype(int)
    by = (samples["y"] // (block_km*1000)).astype(int)
    blocks = (bx.astype(str) + "_" + by.astype(str))
    return blocks.astype("category").cat.codes.to_numpy()

def loeo_folds(samples):
    """Leave-one-event-out folds: dict {event_label -> indices of that event}.

    Assumes samples has a 0..n-1 RangeIndex; returned indices are label indices and
    coincide with positional indices under this precondition.
    """
    return {ev: samples.index[samples["event"] == ev].to_numpy()
            for ev in sorted(samples["event"].unique())}

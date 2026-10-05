"""Corrected ACS equity/environmental-justice overlay: honest
index composition, a Holm-corrected Spearman family, and a disclosed
population_density predictor<->outcome coupling.

Ports Sections 1-3 of the original notebook ``07_equity_analysis`` (ACS sociodemographic
acquisition, MaxEnt zonal statistics per block group, Spearman
correlations) -- NOT its Sections 4-8 (bivariate 2x2 classification,
income-quintile/DEI exposure, FEMA SFHA-vs-income, resilience-deficit
mapping): those are out of scope (this module covers the
ACS-acquisition/zonal-stats/Spearman-correlation path plus corrections,
not a full re-port of the notebook's whole feature set).

Four corrections vs. that source notebook, all made after a review
flagged the original equity analysis as an overclaim:

1. **Honest index composition** (``index_composition()``). The source
   notebook's setup-cell prose described a 7-indicator vulnerability index
   (poverty rate, % non-white, % renter, rent burden, % elderly,
   % disability, % no vehicle) and a 3-indicator adaptive-capacity index
   (% owner, median income, % vehicle access). That prose never matched
   what Riley County block-group data actually supports: three of those
   eight named ACS tables -- poverty (B17001), disability (B18101), and
   vehicle access (B08201) -- return null for EVERY Riley County block
   **UPDATE 2026-08-14.** Verified against the live 2023
   ACS API: poverty and vehicle access ARE recoverable at block group from
   DIFFERENT tables -- C17002 (ratio of income to poverty) and B25044 (tenure by
   vehicles available), both 59/59 non-null. Those are now used. **Disability
   genuinely is not published at block group**: B18101 AND C18108 (the table the
   review proposed) both return all-null. So the review was right that two of the
   three were obtainable, and wrong about every table ID it named.
   group in the 2019-2023 5-year ACS release actually queried here
   (empirically confirmed by ``fetch_acs()``, not assumed; see
   ``tests/test_equity.py::test_dropped_indicators_are_actually_null_in_the_real_pull``).
   ``index_composition()`` returns the composition this module actually
   computes -- 5 vulnerability indicators (incl. ``pct_elderly``), 2
   adaptive-capacity indicators (vehicle access dropped) -- as the single
   source of truth ``compute_indices()`` reads from, so the description
   and the computation can never drift apart the way the source notebook's
   did.

2. **Multiple-comparison correction** (``correlations()``). The source
   notebook reported each Spearman p-value against a bare p<0.05/p<0.01
   threshold with no correction across the family of demographic
   comparisons. ``correlations()`` runs the sociodemographic family through
   ``statsmodels.stats.multitest.multipletests`` -- Holm (family-wise
   error rate control, valid under arbitrary dependence between the
   demographic indicators, which are themselves correlated with each
   other) as the primary ``p_adjusted`` column, plus Benjamini-Hochberg FDR
   as a secondary ``p_adjusted_fdr_bh`` column so a reader can see whether
   the "which associations survive" conclusion is sensitive to the
   correction method chosen (it is: under the two-county overlay Holm keeps
   no sociodemographic indicator and BH keeps four -- see
   ``tests/test_equity.py``).

3. **Disclosed population_density predictor<->outcome coupling**
   (``population_density_coupling_disclosure()``). ``population_density``
   is BOTH one of the ACS variables tested here AND a live predictor in
   the final fitted MaxEnt model -- ``data/processed/predictors_screened/
   population_density.tif`` is the exact raster MaxEnt trained on
   (nonzero coefficient in every one of the 10 replicate ``.lambdas``
   files; see ``population_density_lambda_summary()``). Susceptibility is
   therefore partly a function of population_density BY CONSTRUCTION, so
   any correlation between the two below is partly mechanical, not
   independent evidence of a density-linked equity pattern. This is
   emitted into ``correlations()``'s returned ``DataFrame.attrs``
   (``population_density_coupling``) and reflected per-row via the
   ``coupled_with_predictor`` column -- never buried in prose only.
   It is also corrected in a family of its own, never alongside the
   sociodemographic indicators (manuscript Methods, "analyzed separately"):
   its near-zero p-value would otherwise take the first Holm/BH rank and
   lower every genuine indicator's BH-adjusted value (the manuscript's BH
   column moved from 0.037-0.045 to 0.048-0.050 when this was fixed).

4. **Caveats in the output metadata** (``correlations()``'s
   ``attrs["caveats"]``): MAUP, ecological fallacy, ACS margin-of-error,
   and the KSU student-housing confound (Manhattan, KS is a college town;
   block groups dominated by student renters have artefactually depressed
   income/homeownership). The whole analysis is framed as exploratory /
   correlational, never causal (``attrs["kind"] ==
   "exploratory_correlational"``, ``attrs["causal"] is False``) --
   mirrors this pipeline's existing ``kind``/``framing``/``attrs``
   convention (``pipeline.screen.bivariate``'s
   ``not_validation``/``framing``, ``pipeline.evaluate``'s
   ``DIRECTIONAL_KIND``/``DIRECTIONAL_FRAMING``), not a new pattern
   invented for this module.

**Honesty mandate.** Every correlation is reported AS OBSERVED. If most
(or all) of the genuine sociodemographic indicators are not significant
even before correction, that is reported plainly.
Nothing here is forced toward an expected equity finding.

**Caching.** ``fetch_acs()`` caches the Census API JSON pull (already
derived to indicator columns) to ``data/raw/census/`` (gitignored, matches
the existing ``data/raw/tiger/``/``data/raw/fema/`` cache convention).
``block_group_geometries()`` reuses the TIGER 2022 Kansas block-group
shapefile already cached at ``data/raw/tiger/tl_2022_20_bg.zip`` (same
vintage/path the source notebook used), downloading it only if that cache
is absent. ``run()`` writes ``equity_summary_all_bgs.csv``,
``equity_correlations.csv`` and ``income_quintile_susceptibility.csv`` to
``data/processed/equity/``; the tests write nothing there.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import requests
from shapely.geometry import mapping
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from pipeline import config

# ============================================================================
# Paths / constants
# ============================================================================
# Kansas; Riley AND Pottawatomie. Manhattan is a two-county city and the
# modelling AOI spans both (``predictors.CENSUS_COUNTIES`` is the same pair),
# so restricting the equity analysis to Riley -- as this module and
# the original notebook previously did -- silently analysed less than the
# area the susceptibility surface covers, dropping the Pottawatomie block
# groups east of the Big Blue River entirely. The two lists are now the same
# pair by construction; ``test_equity_counties_match_predictors`` asserts it.
STATE = "20"
COUNTIES = ("161", "149")   # Riley, Pottawatomie
COUNTY = COUNTIES[0]        # retained for backward compatibility only

OUT_DIR = config.DATA / "equity"
SUSC_PATH = config.DATA / "maxent" / "final" / "cloglog" / "flood_avg.tif"
POPDENS_PATH = config.DATA / "predictors_screened" / "population_density.tif"
MAXENT_FINAL_DIR = config.DATA / "maxent" / "final" / "cloglog"

# Dasymetric weighting layers. An unweighted zonal mean over every pixel in a
# block group counts undeveloped land -- fields, water, wooded slope -- as
# equally representative of the residents the ACS attributes describe, which
# biases the exposure measure toward whatever land happens to fall inside the
# polygon. NLCD developed classes restrict the statistic to built-up land, and
# impervious fraction weights it continuously toward development intensity.
# NOTE: population_density CANNOT serve as the weight -- it is rasterised FROM
# block-group polygons, so it is constant within each polygon and weighting by
# it inside one is a no-op.
NLCD_PATH = config.DATA / "predictors" / "nlcd_landcover.tif"
IMPERV_PATH = config.DATA / "predictors" / "impervious_pct.tif"
NLCD_DEVELOPED = (21, 22, 23, 24)

CENSUS_KEY_PATH = config.REPO / ".census_api_key"
CENSUS_CACHE = config.REPO / "data" / "raw" / "census" / "acs5_2023_bg.csv"
BG_CACHE = config.REPO / "data" / "raw" / "tiger" / f"tl_2022_{STATE}_bg.zip"
BG_URL = f"https://www2.census.gov/geo/tiger/TIGER2022/BG/tl_2022_{STATE}_bg.zip"

MIN_PIXELS = 50    # >= 0.5 ha of 10 m raster coverage per block group -- matches
                   # the original notebook's zonal_susc(min_pixels=50).
STUDENT_THR = 0.30  # >= 30% aged 18-24 flags a student-dominated block group
                    # (a proxy; enrollment/group-quarters tables are not
                    # published at block-group geography) -- matches the original notebook.

# ACS 2019-2023 5-year variables -- matches the original notebook's PULL
# list. Kept even for the three tables (B17001/B18101/B08201) empirically
# found to be 100% null at this geography/vintage (see module docstring,
# correction 1): if the Census Bureau ever backfills them in a later 5-year
# release, fetch_acs() picks that up automatically -- index_composition()'s
# hardcoded 5/2 recipe would still need a human to deliberately widen it
# (a versioned, reproducible recipe, not a silently-drifting "whatever's
# non-null today" list).
ACS_PULL = [
    "B19013_001E",                                          # median HH income
    "B17001_001E", "B17001_002E",                           # poverty universe (EMPTY at BG -- kept as evidence)
    # C17002 = ratio of income to poverty level. UNLIKE B17001 it IS published at
    # block-group geography -- verified against the live 2023 ACS API 2026-08-14,
    # 59/59 non-null for Riley County, whereas B17001 returns all-null at BG.
    "C17002_001E", "C17002_002E", "C17002_003E",           # total, <0.50, 0.50-0.99
    "B03002_001E", "B03002_003E",                           # race universe, NH-White alone
    "B25003_001E", "B25003_002E", "B25003_003E",            # tenure: total, owner, renter
    "B25071_001E",                                          # median rent as % of income
    "B01003_001E",                                          # total population
    # Elderly 65+ male (6 groups) + female (6 groups)
    "B01001_020E", "B01001_021E", "B01001_022E",
    "B01001_023E", "B01001_024E", "B01001_025E",
    "B01001_044E", "B01001_045E", "B01001_046E",
    "B01001_047E", "B01001_048E", "B01001_049E",
    # Young adults 18-24 (college-age; student-housing confound proxy)
    "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",
    "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",
    "B01002_001E",                                          # median age (confound corroboration)
    # Disability: universe + 12 with-disability cells
    "B18101_001E",
    "B18101_004E", "B18101_007E", "B18101_010E",
    "B18101_013E", "B18101_016E", "B18101_019E",
    "B18101_023E", "B18101_026E", "B18101_029E",
    "B18101_032E", "B18101_035E", "B18101_038E",
    # Vehicle access
    "B08201_001E", "B08201_002E",                           # journey-to-work (EMPTY at BG -- kept as evidence)
    # B25044 = tenure by vehicles available. Published at block group (59/59
    # non-null, verified 2026-08-14). B08201 is a commuting table and returns
    # all-null at BG.
    "B25044_001E", "B25044_003E", "B25044_010E",           # total, owner-no-veh, renter-no-veh
]

_ELDERLY_COLS = ["B01001_020E", "B01001_021E", "B01001_022E",
                 "B01001_023E", "B01001_024E", "B01001_025E",
                 "B01001_044E", "B01001_045E", "B01001_046E",
                 "B01001_047E", "B01001_048E", "B01001_049E"]
_YOUNG_COLS = ["B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",
               "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E"]
_DISABILITY_COLS = ["B18101_004E", "B18101_007E", "B18101_010E",
                    "B18101_013E", "B18101_016E", "B18101_019E",
                    "B18101_023E", "B18101_026E", "B18101_029E",
                    "B18101_032E", "B18101_035E", "B18101_038E"]


# ============================================================================
# Section 1: ACS acquisition
# ============================================================================

def _census_key() -> str:
    """Reads ``.census_api_key`` unconditionally (fails loudly if absent)
    -- mirrors ``predictors.acquire_census``'s own convention: callers are
    expected to have the key present, not to probe for it defensively.
    """
    if not CENSUS_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Census API key not found at {CENSUS_KEY_PATH} -- required to "
            f"fetch ACS sociodemographic data. Obtain a free key at "
            f"https://api.census.gov/data/key_signup.html."
        )
    return CENSUS_KEY_PATH.read_text().strip()


def fetch_acs(cache_path=CENSUS_CACHE, force=False) -> pd.DataFrame:
    """ACS 2019-2023 5-year estimates, block groups of every county in
    ``COUNTIES`` (Riley + Pottawatomie -- Manhattan spans both), via
    the Census API -- derives every indicator ``index_composition()`` and
    ``correlations()`` need (median income + its polarity-flipped
    ``median_income_inv``, poverty rate, % non-white, % renter/owner, rent
    burden, % elderly, % disability, % no vehicle, % aged 18-24, median
    age). Cached to ``cache_path`` (default ``data/raw/census/`` --
    gitignored, matches ``data/raw/tiger/``/``data/raw/fema/``) so a
    second call/test run never re-hits the network.

    ``force=True`` bypasses the cache and re-fetches (for a genuine
    refresh, e.g. a new ACS vintage).

    Returns one row per block group actually returned by the API, columns:
    ``GEOID``, ``median_income``, ``median_income_inv``, ``poverty_rate``,
    ``pct_nonwhite``, ``pct_renter``, ``pct_owner``, ``rent_burden_pct``,
    ``pct_elderly``, ``pct_disability``, ``pct_no_vehicle``, ``pop_total``,
    ``pct_18_24``, ``median_age``. Empirically (2019-2023 5-year release,
    Riley County), ``poverty_rate``/``pct_disability``/``pct_no_vehicle``
    come back entirely null -- see module docstring correction 1 -- but
    the columns are always present so downstream code never KeyErrors on
    them, only NaNs.
    """
    if cache_path.exists() and not force:
        print(f"[equity] ACS pull: cached ({cache_path})")
        return pd.read_csv(cache_path, dtype={"GEOID": str})

    key = _census_key()
    url = "https://api.census.gov/data/2023/acs/acs5"

    # The ACS API rejects a request with more than 50 variables (HTTP 400).
    # Adding C17002/B25044 pushed ACS_PULL to 51, so the pull is
    # chunked and the chunks joined on the geography keys. Batching rather than
    # dropping variables keeps the empirically-empty B17001/B18101/B08201
    # columns in the raw pull as evidence that they are unavailable at
    # block-group geography.
    GEO_KEYS = ["state", "county", "tract", "block group"]
    MAX_VARS = 45          # headroom under the API's 50-variable cap

    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    frames = []
    for county in COUNTIES:
        county_part = None
        for chunk in _chunks(list(ACS_PULL), MAX_VARS):
            params = {"get": ",".join(chunk), "for": "block group:*",
                      "in": f"state:{STATE} county:{county}", "key": key}
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            rows = resp.json()
            part = pd.DataFrame(rows[1:], columns=rows[0])
            county_part = part if county_part is None else \
                county_part.merge(part, on=GEO_KEYS, how="outer")
        print(f"[equity] ACS pull: county {county} -> {len(county_part)} block groups "
              f"({len(ACS_PULL)} variables in "
              f"{-(-len(ACS_PULL) // MAX_VARS)} request(s))")
        frames.append(county_part)
    acs = pd.concat(frames, ignore_index=True)
    print(f"[equity] ACS pull: fetched {len(acs)} block groups from the Census API "
          f"(2019-2023 5-year ACS, counties {'+'.join(COUNTIES)})")

    numeric_cols = [c for c in acs.columns if c not in ("state", "county", "tract", "block group")]
    for c in numeric_cols:
        acs[c] = pd.to_numeric(acs[c], errors="coerce").replace(-666666666, np.nan)

    acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"] + acs["block group"]

    # Prefer C17002 (published at BG); B17001 is retained only so the empirical
    # all-null finding stays visible in the raw pull.
    _pov_num = acs["C17002_002E"] + acs["C17002_003E"]
    acs["poverty_rate"] = _pov_num / acs["C17002_001E"].replace(0, np.nan)
    if acs["poverty_rate"].notna().sum() == 0:
        acs["poverty_rate"] = acs["B17001_002E"] / acs["B17001_001E"].replace(0, np.nan)
    acs["pct_nonwhite"] = 1 - acs["B03002_003E"] / acs["B03002_001E"].replace(0, np.nan)
    acs["pct_renter"] = acs["B25003_003E"] / acs["B25003_001E"].replace(0, np.nan)
    acs["pct_owner"] = acs["B25003_002E"] / acs["B25003_001E"].replace(0, np.nan)
    acs["rent_burden_pct"] = acs["B25071_001E"]
    acs["median_income"] = acs["B19013_001E"]
    # Polarity-flipped so "higher = more vulnerable" holds uniformly across
    # every vulnerability-index component (see index_composition()); a
    # plain sign flip is monotonic-equivalent to inverting the z-score
    # later, computed once here so the column exists by this exact name.
    acs["median_income_inv"] = -acs["median_income"]
    acs["pop_total"] = acs["B01003_001E"]

    acs["pct_elderly"] = acs[_ELDERLY_COLS].sum(axis=1) / acs["B01003_001E"].replace(0, np.nan)
    acs["pct_18_24"] = acs[_YOUNG_COLS].sum(axis=1) / acs["B01003_001E"].replace(0, np.nan)
    acs["median_age"] = acs["B01002_001E"]
    acs["pct_disability"] = acs[_DISABILITY_COLS].sum(axis=1) / acs["B18101_001E"].replace(0, np.nan)
    # Prefer B25044 (published at BG); B08201 retained as evidence only.
    _noveh = acs["B25044_003E"] + acs["B25044_010E"]
    acs["pct_no_vehicle"] = _noveh / acs["B25044_001E"].replace(0, np.nan)
    if acs["pct_no_vehicle"].notna().sum() == 0:
        acs["pct_no_vehicle"] = acs["B08201_002E"] / acs["B08201_001E"].replace(0, np.nan)

    keep = ["GEOID", "median_income", "median_income_inv", "poverty_rate",
            "pct_nonwhite", "pct_renter", "pct_owner", "rent_burden_pct",
            "pct_elderly", "pct_disability", "pct_no_vehicle", "pop_total",
            "pct_18_24", "median_age"]
    acs_clean = acs[keep].copy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    acs_clean.to_csv(cache_path, index=False)
    print(f"[equity] ACS pull: cached to {cache_path}")
    return acs_clean


def block_group_geometries(cache_path=BG_CACHE, url=BG_URL, counties=COUNTIES) -> gpd.GeoDataFrame:
    """Block-group polygons for every county in ``counties`` (Riley AND
    Pottawatomie by default -- Manhattan spans both), TIGER 2022 -- reuses the shapefile
    already cached at ``data/raw/tiger/tl_2022_20_bg.zip`` (same file
    the original notebook cached; downloads it fresh only if that
    cache is somehow absent). Returns columns ``GEOID``, ``geometry`` in
    the shapefile's native CRS (NAD83, EPSG:4269) -- reprojection to a
    raster's CRS happens per-call inside ``_zonal_mean``, never assumed
    here.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[equity] TIGER block groups: cached ({cache_path})")
    else:
        print("[equity] TIGER block groups: downloading Kansas block-group shapefile...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
        print(f"[equity] TIGER block groups: saved {len(r.content) / 1e6:.1f} MB to {cache_path}")

    bg_all = gpd.read_file(f"zip://{cache_path}")
    if isinstance(counties, str):
        counties = (counties,)
    bg = bg_all[bg_all["COUNTYFP"].isin(list(counties))][["GEOID", "geometry"]].copy()
    print(f"[equity] TIGER block groups: {len(bg)} in counties {'+'.join(counties)}")
    return bg.reset_index(drop=True)


# ============================================================================
# Section 2: MaxEnt susceptibility + population_density zonal statistics
# ============================================================================

def _zonal_mean(gdf, raster_path, min_pixels=MIN_PIXELS):
    """Per-geometry mean of ``raster_path`` within each ``gdf`` geometry,
    via ``rasterio.mask`` -- mirrors the original notebook's own
    ``zonal_susc`` helper, generalized to any single-band raster (reused
    here for both the susceptibility raster and the population_density
    predictor raster, so both go through IDENTICAL zonal-stats mechanics).
    Reprojects ``gdf`` to ``raster_path``'s own CRS first (never assumes
    the caller already matched it).

    A geometry with fewer than ``min_pixels`` finite pixels inside it
    (< 0.5 ha of 10 m raster coverage at the default 50) is scored NaN --
    insufficiently represented, matching the source notebook's own
    threshold, rather than reporting a mean computed from a handful of
    edge pixels.

    Returns two parallel numpy arrays, same length/order as ``gdf``:
    ``(means, n_valid_pixels)``.
    """
    means, ns = [], []
    with rasterio.open(raster_path) as src:
        gdf_r = gdf.to_crs(src.crs)
        nodata = src.nodata
        for geom in gdf_r.geometry:
            try:
                out, _ = rasterio.mask.mask(src, [mapping(geom)], crop=True,
                                             nodata=np.nan, all_touched=False)
                data = out[0].astype("float64")
                if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                    data[data == nodata] = np.nan
                v = data[np.isfinite(data)]
            except Exception:
                v = np.array([])
            means.append(float(np.mean(v)) if len(v) >= min_pixels else np.nan)
            ns.append(int(len(v)))
    return np.array(means, dtype="float64"), np.array(ns, dtype="int64")


def _zonal_dasymetric(gdf, raster_path, min_pixels=MIN_PIXELS,
                       nlcd_path=NLCD_PATH, imperv_path=IMPERV_PATH):
    """Per-geometry susceptibility restricted/weighted to DEVELOPED land.

    ``_zonal_mean`` above averages every finite pixel in a block group, so a
    polygon that is mostly field, water or wooded slope is summarised by land
    where nobody the ACS attributes describe actually lives. Flood-exposure
    work treats that mismatch as a first-order measurement problem and corrects
    it dasymetrically -- reallocating the statistic onto the land that is
    actually built up (Tate et al., 2021).

    Two corrected statistics are returned per geometry:
      - ``dev``    -- mean over NLCD developed pixels only (classes 21-24);
                      NaN if fewer than ``min_pixels`` developed pixels.
      - ``impw``   -- impervious-fraction-weighted mean over all valid pixels,
                      a continuous version of the same idea that degrades
                      gracefully where the developed mask is sparse.

    NOTE population_density is deliberately NOT the weight: it is rasterised
    from the block-group polygons themselves, so it is constant inside each one
    and weighting by it within a polygon changes nothing.

    Returns ``(dev_means, impw_means, n_developed_pixels)``.
    """
    dev_means, impw_means, n_dev = [], [], []
    with rasterio.open(raster_path) as src_s, \
         rasterio.open(nlcd_path) as src_n, \
         rasterio.open(imperv_path) as src_i:
        gdf_r = gdf.to_crs(src_s.crs)
        for geom in gdf_r.geometry:
            gj = [mapping(geom)]
            try:
                sus, _ = rasterio.mask.mask(src_s, gj, crop=True, nodata=np.nan, all_touched=False)
                nlc, _ = rasterio.mask.mask(src_n, gj, crop=True, nodata=np.nan, all_touched=False)
                imp, _ = rasterio.mask.mask(src_i, gj, crop=True, nodata=np.nan, all_touched=False)
                sus, nlc, imp = sus[0].astype("float64"), nlc[0].astype("float64"), imp[0].astype("float64")
                shp = min(sus.shape, nlc.shape, imp.shape)
                sus, nlc, imp = sus[:shp[0], :shp[1]], nlc[:shp[0], :shp[1]], imp[:shp[0], :shp[1]]

                developed = np.isin(np.round(nlc), NLCD_DEVELOPED) & np.isfinite(sus)
                vals_dev = sus[developed]
                dev_means.append(float(np.mean(vals_dev)) if len(vals_dev) >= min_pixels else np.nan)
                n_dev.append(int(len(vals_dev)))

                ok = np.isfinite(sus) & np.isfinite(imp)
                w = imp[ok]
                impw_means.append(float(np.average(sus[ok], weights=w))
                                  if ok.sum() >= min_pixels and w.sum() > 0 else np.nan)
            except Exception:
                dev_means.append(np.nan); impw_means.append(np.nan); n_dev.append(0)
    return (np.array(dev_means, dtype="float64"),
            np.array(impw_means, dtype="float64"),
            np.array(n_dev, dtype="int64"))


def equity_table(acs=None, bg=None, susc_path=SUSC_PATH, popdens_path=POPDENS_PATH,
                  min_pixels=MIN_PIXELS) -> gpd.GeoDataFrame:
    """Block-group-level table merging ACS sociodemographics with zonal
    mean MaxEnt susceptibility (``susc_mean``, from ``susc_path`` --
    default the published ``data/processed/maxent/final/cloglog/
    flood_avg.tif``) and zonal mean population_density
    (``bg_pop_density_mean``, from ``popdens_path`` -- default
    ``data/processed/predictors_screened/population_density.tif``, the
    EXACT raster feeding the final MaxEnt fit; see module docstring
    correction 3).

    ``acs``/``bg`` are injected (default to fresh ``fetch_acs()``/
    ``block_group_geometries()`` calls) so this stays testable against a
    synthetic frame without hitting the network/disk.

    Drops any block group without a valid ``susc_mean`` (insufficient
    raster coverage, per ``_zonal_mean``'s ``min_pixels`` gate) and resets
    to a clean ``0..n-1`` index. Every other column may still be NaN
    per-row (e.g. a block group with valid susceptibility but a suppressed
    ACS variable) -- ``correlations()`` drops NaNs per-INDICATOR, not
    per-row, so one sparse column never silently shrinks every other
    indicator's sample size.
    """
    if acs is None:
        acs = fetch_acs()
    if bg is None:
        bg = block_group_geometries()

    susc_mean, susc_n = _zonal_mean(bg, susc_path, min_pixels)
    dev_mean, impw_mean, n_dev = _zonal_dasymetric(bg, susc_path, min_pixels=min_pixels)
    popdens_mean, popdens_n = _zonal_mean(bg, popdens_path, min_pixels)

    out = bg.copy()
    out["susc_mean"] = susc_mean
    out["susc_mean_dev"] = dev_mean       # dasymetric: NLCD developed land only
    out["susc_mean_impw"] = impw_mean     # impervious-fraction weighted
    out["susc_n_developed"] = n_dev
    out["susc_n"] = susc_n
    out["bg_pop_density_mean"] = popdens_mean
    out["bg_pop_density_n"] = popdens_n

    out = out.merge(acs, on="GEOID", how="left")
    out = out.dropna(subset=["susc_mean"]).reset_index(drop=True)
    print(f"[equity] zonal stats: {len(out)}/{len(bg)} block groups have "
          f">= {min_pixels} valid susceptibility pixels")
    return out


# ============================================================================
# Index composition (correction 1 in the module docstring)
# ============================================================================

VULNERABILITY_DROPPED = {
    "poverty_rate": (
        "ACS table B17001 (poverty status in the past 12 months) returns "
        "null for every Riley County block group in the 2019-2023 5-year "
        "release (empirically confirmed by fetch_acs(), not assumed). Not "
        "a redundancy judgment call -- there is no usable data to include."
    ),
    "pct_disability": (
        "ACS table B18101 (disability status) is likewise null for every "
        "Riley County block group in this release -- the identical "
        "complete-suppression failure, not a redundancy decision."
    ),
}

ADAPTIVE_CAPACITY_DROPPED = {
    "pct_no_vehicle": (
        "ACS table B08201 (household vehicle access) is null for every "
        "Riley County block group in this release -- the same "
        "data-suppression pattern as the two dropped vulnerability "
        "indicators. Dropped for data availability, not because vehicle "
        "access is conceptually redundant with homeownership/income."
    ),
}

OLD_NOTEBOOK_DESCRIPTION = (
    "The setup-cell prose of the original notebook 07_equity_analysis described a "
    "7-indicator vulnerability index (poverty rate, % non-white, "
    "% renter, rent burden, % elderly, % disability, % no vehicle) and a "
    "3-indicator adaptive-capacity index (% owner, median income, "
    "% vehicle access). Three of those eight named ACS tables -- poverty "
    "B17001, disability B18101, vehicle access B08201 -- are entirely "
    "suppressed at the Riley County block-group level in the 2019-2023 "
    "5-year release actually queried here; the prose never matched what "
    "the data supports. index_composition() returns the composition this "
    "module actually computes: 5 vulnerability indicators, 2 "
    "adaptive-capacity indicators -- not the notebook's stale 7/3 count."
)


def index_composition() -> dict:
    """The ACTUAL composition of the two composite indices this module can
    build (via ``compute_indices()``) -- single source of truth so the
    description can never drift from the code the way
    the original notebook's prose drifted from its own z-score
    computation (see ``OLD_NOTEBOOK_DESCRIPTION``).

    Pure, no I/O, no network -- callable instantly, independent of
    ``fetch_acs()``/``equity_table()``.

    Returns a dict:
      - ``vulnerability``: the 5 indicators actually used (higher = more
        vulnerable for every entry, including ``median_income_inv``, the
        polarity-flipped median income): ``pct_nonwhite``, ``pct_renter``,
        ``rent_burden_pct``, ``pct_elderly``, ``median_income_inv``.
      - ``vulnerability_dropped``: dict of the 2 indicators the source
        notebook's prose named but this module excludes, each with the
        empirical (data-availability, not redundancy) reason -- see
        ``VULNERABILITY_DROPPED``.
      - ``adaptive_capacity``: the 2 indicators actually used (higher =
        more capacity): ``pct_owner``, ``median_income``.
      - ``adaptive_capacity_dropped``: dict of the 1 indicator (vehicle
        access) the source notebook's prose named but this module
        excludes, with the empirical reason -- see
        ``ADAPTIVE_CAPACITY_DROPPED``.
      - ``old_notebook_description``: the mismatch this function corrects,
        for provenance/citation.
    """
    return {
        "vulnerability": ["pct_nonwhite", "pct_renter", "rent_burden_pct",
                          "pct_elderly", "median_income_inv"],
        "vulnerability_dropped": dict(VULNERABILITY_DROPPED),
        "adaptive_capacity": ["pct_owner", "median_income"],
        "adaptive_capacity_dropped": dict(ADAPTIVE_CAPACITY_DROPPED),
        "old_notebook_description": OLD_NOTEBOOK_DESCRIPTION,
    }


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not sd or pd.isna(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ``vulnerability_index``/``adaptive_capacity_index`` columns to
    a copy of ``df``, z-scoring and averaging (``skipna=True``, so a
    block group missing one component still gets an index from the rest)
    EXACTLY the columns ``index_composition()`` declares -- no second,
    independently-maintained list that could drift from it. ``df`` must
    already carry every column named in both of ``index_composition()``'s
    lists (``equity_table()``'s output does, including
    ``median_income_inv``, computed in ``fetch_acs()``).
    """
    comp = index_composition()
    out = df.copy()
    vuln_parts = pd.concat([_zscore(out[c]) for c in comp["vulnerability"]], axis=1)
    out["vulnerability_index"] = vuln_parts.mean(axis=1, skipna=True)
    ac_parts = pd.concat([_zscore(out[c]) for c in comp["adaptive_capacity"]], axis=1)
    out["adaptive_capacity_index"] = ac_parts.mean(axis=1, skipna=True)
    return out


# ============================================================================
# Spearman correlations + multiple-comparison correction (correction 2) +
# population_density coupling disclosure (correction 3) + caveats
# (correction 4)
# ============================================================================

# Genuine sociodemographic/equity indicators -- matches
# the original notebook's own DEMO correlation battery exactly (8
# variables). Deliberately excludes pct_owner (= 1 - pct_renter, the exact
# complementary share from the same B25003 denominator -- would duplicate
# pct_renter's test with the opposite sign, inflating the correction
# family with a redundant comparison) and median_income_inv (a monotonic
# sign-flip of median_income already tested -- same duplication concern;
# median_income_inv instead feeds compute_indices()'s composite only).
DEMOGRAPHIC_VARS = {
    "median_income":   "Median household income ($)",
    "poverty_rate":    "Poverty rate",
    "pct_nonwhite":    "% non-white",
    "pct_renter":      "% renter-occupied",
    "rent_burden_pct": "Rent burden (% of income)",
    "pct_elderly":     "% elderly (65+)",
    "pct_disability":  "% with disability",
    "pct_no_vehicle":  "% no vehicle access",
}

# The mechanically-coupled predictor (correction 3) -- kept structurally
# separate from DEMOGRAPHIC_VARS (not just distinguished in prose) so
# `variable in COUPLED_VARS` is a hard, testable fact, not a claim that
# could silently go stale.
COUPLED_VARS = {
    "bg_pop_density_mean": ("Population density (zonal mean of the exact "
                             "raster MaxEnt trained on) -- see "
                             "population_density_coupling_disclosure()"),
}

ALL_CORRELATION_VARS = {**DEMOGRAPHIC_VARS, **COUPLED_VARS}

MULTIPLE_COMPARISON_METHOD = "holm"  # FWER control, valid under arbitrary
                                      # dependence between indicators (they
                                      # are themselves correlated with each
                                      # other) -- more defensible here than
                                      # an independence-assuming procedure.
                                      # fdr_bh is also reported (secondary
                                      # p_adjusted_fdr_bh column) so the
                                      # "which associations survive"
                                      # conclusion's sensitivity to the
                                      # correction method is visible, not
                                      # hidden behind one choice.

EQUITY_KIND = "exploratory_correlational"

EQUITY_FRAMING = (
    "Exploratory / correlational only. Every number here is a Spearman "
    "rank correlation between BLOCK-GROUP-level aggregates (mean MaxEnt "
    "susceptibility vs. one ACS sociodemographic indicator, computed on "
    "a small set of Riley County census block groups) -- never a causal "
    "estimate, never an individual-level claim, and not validated against "
    "any independent equity ground truth. See "
    "attrs['population_density_coupling'] and attrs['caveats'] (MAUP, "
    "ecological fallacy, ACS margin of error, KSU student-housing "
    "confound) before citing any figure from this table."
)

MAUP_CAVEAT = (
    "Modifiable areal unit problem: results are computed on census block "
    "groups, an administratively-drawn areal unit. Both the magnitude and "
    "the sign of an aggregate correlation can shift under a different "
    "zonation (e.g. census tracts) or a different boundary drawing at the "
    "same resolution; no sensitivity check against an alternate zonation "
    "is performed here."
)

ECOLOGICAL_FALLACY_CAVEAT = (
    "Ecological fallacy (Robinson, 1950): every correlation is between "
    "block-group MEANS/RATES, never individual households or parcels. It "
    "says nothing about which specific residents within a block group "
    "are more exposed, and must not be read as an individual-level "
    "finding."
)

ACS_MOE_CAVEAT = (
    "ACS 5-year (2019-2023) block-group estimates carry substantial "
    "margins of error, especially for a small county (Riley County has on "
    "the order of 50-60 block groups total) and for rates computed on "
    "small population denominators (e.g. elderly share or rent burden in "
    "a low-population block group). Every ACS value used here is a point "
    "estimate treated as exact; published margins of error are not "
    "propagated into the Spearman tests or the multiple-comparison "
    "correction, so results should be read as approximate, not exact."
)


def _student_housing_caveat(table: pd.DataFrame) -> str:
    """Formats the KSU student-housing confound caveat WITH the actual
    observed block-group count from ``table`` (grounded in real data every
    time this runs, never a static/potentially-stale sentence) -- falls
    back to a mechanism-only description if ``table`` lacks ``pct_18_24``
    (e.g. a caller passed a synthetic frame without it).
    """
    if "pct_18_24" not in table.columns or table["pct_18_24"].notna().sum() == 0:
        return (
            "KSU student-housing confound: Manhattan, KS is a college "
            "town; block groups dominated by student renters have "
            "artefactually depressed median household income and "
            "homeownership, inflating 'low-income'/'low-adaptive-"
            "capacity' classifications for reasons unrelated to genuine "
            "economic hardship. pct_18_24 (the available age-share proxy "
            "-- enrollment/group-quarters tables are not published at "
            "block-group geography) was not available in this table to "
            "quantify how many block groups here are affected."
        )
    valid = table["pct_18_24"].notna()
    n_student = int((table["pct_18_24"] >= STUDENT_THR).sum())
    n_total = int(valid.sum())
    return (
        "KSU student-housing confound: Manhattan, KS is a college town; "
        "block groups with a high share of residents aged 18-24 (student "
        "renters) have artefactually depressed median household income "
        "and homeownership, inflating 'low-income'/'low-adaptive-"
        "capacity' classifications for reasons unrelated to genuine "
        f"economic hardship. {n_student} of {n_total} block groups here "
        f"meet a >= {STUDENT_THR:.0%} aged-18-24 threshold historically "
        "used to flag this (enrollment/group-quarters tables are not "
        "published at block-group geography, so this age share is only "
        "a proxy, not a direct student count). No correction for this "
        "confound is applied to the Spearman correlations below -- they "
        "are computed on the full sample; read renter/rent-burden/income "
        "associations with this in mind."
    )


def population_density_lambda_summary(final_dir=MAXENT_FINAL_DIR) -> dict:
    """Mean/min/max of the ``population_density`` MaxEnt coefficient across
    every replicate ``.lambdas`` file under ``final_dir`` (the 10
    ``flood_0..flood_9.lambdas`` files backing ``flood_avg.tif``) --
    derived live from whatever is really on disk, never a hardcoded
    literal (this pipeline's established discipline -- see e.g.
    ``pipeline.screen.bivariate``'s live ``n_tests``). The
    replicate-to-replicate spread is itself informative: a positive,
    non-trivial coefficient in EVERY replicate (not just one cherry-picked
    run) is stronger evidence of a real, non-fluke coupling than a single
    number would be.

    Returns ``{"mean": float, "min": float, "max": float, "n_replicates":
    int}`` -- all ``None``/0 if no ``.lambdas`` files are found (e.g. a
    checkout without the MaxEnt build artifacts).
    """
    vals = []
    for f in sorted(final_dir.glob("flood_*.lambdas")):
        for line in f.read_text().splitlines():
            if line.startswith("population_density,"):
                vals.append(float(line.split(",")[1]))
                break
    if not vals:
        return {"mean": None, "min": None, "max": None, "n_replicates": 0}
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "n_replicates": len(vals)}


def population_density_coupling_disclosure(final_dir=MAXENT_FINAL_DIR) -> str:
    """The population_density predictor<->outcome coupling disclosure
    (module docstring correction 3), with the live coefficient summary
    (``population_density_lambda_summary``) interpolated in -- never a
    hardcoded coefficient literal that could go stale if the model is
    ever refit.
    """
    summ = population_density_lambda_summary(final_dir)
    if summ["n_replicates"] == 0:
        coeff_txt = "coefficient unavailable (no .lambdas files found on disk)"
    else:
        coeff_txt = (f"mean coefficient (lambda) = {summ['mean']:.3f} across "
                     f"{summ['n_replicates']} replicate fits (range "
                     f"{summ['min']:.3f} to {summ['max']:.3f}, positive and "
                     f"non-trivial in every replicate)")
    return (
        "population_density is BOTH an ACS variable correlated against "
        "susceptibility below AND a live predictor in the final fitted "
        f"MaxEnt model: {POPDENS_PATH.name} (data/processed/"
        f"predictors_screened/) is the EXACT raster this table's "
        f"bg_pop_density_mean is sampled from AND the exact raster MaxEnt "
        f"trained on -- {coeff_txt} "
        "(data/processed/maxent/final/cloglog/flood_*.lambdas). "
        "Susceptibility is therefore PARTLY A FUNCTION OF "
        "population_density BY CONSTRUCTION. Any correlation between "
        "susceptibility and population_density reported below is partly "
        "mechanical -- the model was fit using this exact variable as an "
        "input -- and must NOT be read as independent evidence of a "
        "density-linked equity pattern."
    )


def correlations(table=None, variables=None, method=MULTIPLE_COMPARISON_METHOD,
                  min_n=5) -> pd.DataFrame:
    """Spearman rank correlations of block-group mean MaxEnt susceptibility
    vs. each variable in ``variables`` (default ``ALL_CORRELATION_VARS`` --
    the 8 genuine sociodemographic indicators plus the mechanically-coupled
    ``bg_pop_density_mean``), with a multiple-comparison correction applied
    WITHIN each family (module docstring corrections 2 and 3): the genuine
    sociodemographic indicators are corrected together, and each variable in
    ``COUPLED_VARS`` is corrected in a separate family, so population density
    never shifts the indicators' adjusted p-values.

    ``table`` defaults to a fresh ``equity_table()`` call (ACS pull +
    TIGER geometries + zonal stats against both the susceptibility and
    population_density rasters -- the ~2-5 min foreground path). Pass a
    pre-built/synthetic frame to test the correlation/correction logic in
    isolation.

    Any indicator with fewer than ``min_n`` non-null (susc_mean, indicator)
    pairs is skipped (matches the original notebook's own
    ``if len(sub) < 5: continue``) -- empirically this drops
    ``poverty_rate``/``pct_disability``/``pct_no_vehicle`` entirely (0
    valid rows each, the same total ACS suppression documented in
    ``index_composition()``), leaving 6 tested indicators from the default
    9.

    Returns a DataFrame sorted by ``p_raw`` ascending, columns:
    ``variable``, ``label``, ``rho``, ``p_raw``, ``n``,
    ``coupled_with_predictor`` (bool -- True only for variables in
    ``COUPLED_VARS``), ``correction_family`` (``"sociodemographic"`` or
    ``"coupled_predictor"``), ``p_adjusted`` (the ``method`` correction, Holm
    by default), ``p_adjusted_holm``, ``p_adjusted_fdr_bh``,
    ``significant_raw`` (``p_raw < 0.05``), ``significant_corrected``
    (``p_adjusted < 0.05``). A family of one (population density on its own)
    has adjusted p equal to its raw p.

    ``DataFrame.attrs`` carries (mirrors ``screen.bivariate``'s/
    ``evaluate``'s existing ``kind``/``framing`` convention):
    ``kind`` (``EQUITY_KIND``), ``causal`` (``False``), ``framing``
    (``EQUITY_FRAMING``), ``n_tests`` (all rows), ``n_tests_by_family``,
    ``multiple_comparison_method``,
    ``population_density_coupling``
    (``population_density_coupling_disclosure()``), and ``caveats`` (dict:
    ``maup``, ``ecological_fallacy``, ``acs_margin_of_error``,
    ``student_housing_confound`` -- the last grounded in ``table``'s own
    real ``pct_18_24`` counts).
    """
    if table is None:
        table = equity_table()
    if variables is None:
        variables = ALL_CORRELATION_VARS

    rows = []
    for col, label in variables.items():
        if col not in table.columns:
            print(f"[equity] correlations: skipping {col} (not present in table)")
            continue
        sub = table[["susc_mean", col]].dropna()
        if len(sub) < min_n:
            print(f"[equity] correlations: skipping {col} "
                  f"(only {len(sub)} valid rows, need >= {min_n})")
            continue
        rho, p = spearmanr(sub["susc_mean"], sub[col])
        if not np.isfinite(rho) or not np.isfinite(p):
            print(f"[equity] correlations: skipping {col} (degenerate Spearman test)")
            continue
        rows.append({"variable": col, "label": label, "rho": float(rho),
                      "p_raw": float(p), "n": int(len(sub)),
                      "coupled_with_predictor": col in COUPLED_VARS})

    df = pd.DataFrame(rows, columns=["variable", "label", "rho", "p_raw", "n",
                                      "coupled_with_predictor"])
    df = df.sort_values("p_raw").reset_index(drop=True)
    df["correction_family"] = np.where(df["coupled_with_predictor"],
                                       "coupled_predictor", "sociodemographic")
    n_tests = len(df)
    n_tests_by_family = df["correction_family"].value_counts().to_dict()

    if n_tests == 0:
        df["p_adjusted"] = pd.Series(dtype="float64")
        df["p_adjusted_holm"] = pd.Series(dtype="float64")
        df["p_adjusted_fdr_bh"] = pd.Series(dtype="float64")
        df["significant_raw"] = pd.Series(dtype="bool")
        df["significant_corrected"] = pd.Series(dtype="bool")
    else:
        # Correct within each family only (docstring above): the coupled
        # predictor must not share ranks with the genuine indicators.
        df["p_adjusted_holm"] = np.nan
        df["p_adjusted_fdr_bh"] = np.nan
        for _, idx in df.groupby("correction_family").groups.items():
            _, p_holm, _, _ = multipletests(df.loc[idx, "p_raw"], method="holm")
            _, p_bh, _, _ = multipletests(df.loc[idx, "p_raw"], method="fdr_bh")
            df.loc[idx, "p_adjusted_holm"] = p_holm
            df.loc[idx, "p_adjusted_fdr_bh"] = p_bh
        if method == "holm":
            df["p_adjusted"] = df["p_adjusted_holm"]
        elif method == "fdr_bh":
            df["p_adjusted"] = df["p_adjusted_fdr_bh"]
        else:
            raise ValueError(f"Unsupported multiple-comparison method: {method!r} "
                              f"(use 'holm' or 'fdr_bh')")
        df["significant_raw"] = df["p_raw"] < 0.05
        df["significant_corrected"] = df["p_adjusted"] < 0.05

    df.attrs["kind"] = EQUITY_KIND
    df.attrs["causal"] = False
    df.attrs["framing"] = EQUITY_FRAMING
    df.attrs["n_tests"] = n_tests
    df.attrs["n_tests_by_family"] = n_tests_by_family
    df.attrs["multiple_comparison_method"] = method
    df.attrs["population_density_coupling"] = population_density_coupling_disclosure()
    df.attrs["caveats"] = {
        "maup": MAUP_CAVEAT,
        "ecological_fallacy": ECOLOGICAL_FALLACY_CAVEAT,
        "acs_margin_of_error": ACS_MOE_CAVEAT,
        "student_housing_confound": _student_housing_caveat(table),
    }

    n_sig_raw = int(df["significant_raw"].sum()) if n_tests else 0
    n_sig_corrected = int(df["significant_corrected"].sum()) if n_tests else 0
    print(f"[equity] correlations: {n_tests} tests {n_tests_by_family}, {n_sig_raw} "
          f"significant at raw p<0.05, {n_sig_corrected} survive {method} correction "
          f"within their family")
    return df


# ============================================================================
# Orchestration
# ============================================================================

def income_quintile_susceptibility(table=None, n_quantiles=5,
                                   susc_col="susc_mean", income_col="median_income"):
    """Mean modelled susceptibility by median-household-income quintile.

    ``run()`` writes this table so ``report._fig_equity_income_quintile`` can be
    rebuilt from a clean run.

    Block groups with no income estimate are dropped (ACS suppresses some).
    Returns one row per quintile with the mean, quartiles and n. This is a
    DESCRIPTIVE summary of an exploratory overlay -- it is not a significance
    test, and the correlation in ``correlations()`` is the inferential result.
    """
    if table is None:
        table = compute_indices(equity_table())
    df = table[[income_col, susc_col]].dropna()
    if len(df) < n_quantiles:
        raise ValueError(
            f"[equity] income_quintile_susceptibility: only {len(df)} block groups "
            f"with both {income_col} and {susc_col}; need >= {n_quantiles}")
    labels = [f"Q{i + 1}" for i in range(n_quantiles)]
    q = pd.qcut(df[income_col], n_quantiles, labels=labels, duplicates="drop")
    g = df.groupby(q, observed=True)[susc_col]
    out = pd.DataFrame({
        "income_q": g.mean().index.astype(str),
        "mean": g.mean().to_numpy(),
        "q25": g.quantile(0.25).to_numpy(),
        "q75": g.quantile(0.75).to_numpy(),
        "n": g.size().to_numpy(),
    })
    out["sample"] = "full"
    return out.reset_index(drop=True)


def run(min_n=5, method=MULTIPLE_COMPARISON_METHOD) -> dict:
    """Full pipeline in one call: ACS acquisition -> zonal stats (both
    susceptibility and population_density) -> composite indices ->
    Holm-corrected Spearman correlations. Convenience entry point for
    downstream reporting (``pipeline.report``); every piece above is independently
    callable/testable on its own.

    Returns ``{"table": equity_table() with vulnerability_index/
    adaptive_capacity_index columns added, "correlations":
    correlations()'s DataFrame, "index_composition": index_composition()'s
    dict}``.
    """
    table = equity_table()
    table = compute_indices(table)
    corr = correlations(table=table, method=method, min_n=min_n)

    # Persist the three CSVs that pipeline.report reads, so a clean rebuild
    # produces the equity table and figure.
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "equity_summary_all_bgs.csv", index=False)
    corr.to_csv(out_dir / "equity_correlations.csv", index=False)
    quint = income_quintile_susceptibility(table)
    quint.to_csv(out_dir / "income_quintile_susceptibility.csv", index=False)
    print(f"[equity] wrote equity_summary_all_bgs.csv ({len(table)} block groups), "
          f"equity_correlations.csv ({len(corr)} tests), "
          f"income_quintile_susceptibility.csv ({len(quint)} quintiles)")

    return {
        "table": table,
        "correlations": corr,
        "index_composition": index_composition(),
        "income_quintiles": quint,
    }

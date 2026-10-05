"""Non-hydrology predictor-raster acquisition and alignment.

Ports the original notebook ``02_predictors.ipynb`` into an importable
pipeline stage. Acquires
every predictor layer that does NOT require flow routing, aligns each to
the 10 m NAD83/UTM 14N (EPSG:26914) AOI grid via ``reproject_match`` against
the DEM, and writes ``data/processed/predictors/<name>.tif``.

Layers acquired here:
  - ``dem``                                          -- USGS 3DEP, py3dep (keyless)
  - ``slope``, ``curvature``, ``tri``, ``tpi``        -- local terrain derivatives
  - ``distance_to_streams``, ``drainage_density``     -- NHD High-Resolution flowlines, pynhd
  - ``ksat``                                          -- POLARIS topsoil Ksat
  - ``available_water_storage``, ``depth_to_restriction``  -- gNATSGO / Planetary Computer
  - ``nlcd_landcover``, ``impervious_pct``            -- NLCD 2021, MRLC (pygeohydro)
  - ``hydrologic_soil_group``                         -- gNATSGO mukey + NRCS Soil Data Access
  - ``curve_number``                                  -- local TR-55 lookup (HSG x NLCD), no network
  - ``stormwater_density``, ``distance_to_outfall``   -- City of Manhattan stormwater GDB, local, no network
  - ``population_density``, ``median_income``         -- ACS 5-yr + TIGER block groups (gated, see below)

Changes vs. the notebook source -- three deliberate scope cuts:
  (a) the two gridMET precipitation layers (``precip_annual_mean``,
      ``precip_rx1day``) are DROPPED entirely -- not acquired, not written.
      Locked project decision (documented limitation), not a bug fix;
  (b) the D8 flow-routing chain (``flow_accumulation``, ``twi``, ``spi``,
      ``hand``) is acquired separately from ``acquire_all()`` --
      ``flow_derivatives()`` (bottom of this module) replaces the notebook's
      ``xarray-spatial`` D8 chain (``fill_d8``/``flow_direction_d8``/
      ``flow_accumulation_d8``, which cannot resolve flats and capped
      real-AOI flow accumulation at 2451 cells -- never reaching river
      scale) with WhiteboxTools (flats-resolving) routed on a DEM buffered
      ~5 km beyond the AOI, then clipped/``reproject_match``ed back onto
      the AOI grid. ``acquire_all()`` itself is intentionally left
      unchanged -- the driver notebook calls ``flow_derivatives()`` as its
      own step;
  (c) Census ACS (``population_density``, ``median_income``) is
      encapsulated in ``acquire_census()`` but ``acquire_all()`` only calls
      it when ``(config.REPO / ".census_api_key").exists()`` -- census is
      acquired whenever that key file is present on the machine running the
      pipeline, and skipped (no error) otherwise, since api.census.gov now
      requires a key. Adding or removing the key file alone toggles census
      acquisition on/off for the next ``acquire_all()`` run; no code changes
      needed either way.

...and two substitutions FORCED by the environment actually available for
this port (discovered while implementing, not scope choices -- the pinned
``flood-ch1`` env requires python=3.11 for the rest of the stack, e.g.
numba/datashader/whitebox compatibility):
  (d) ``tri``/``tpi``: ``xarray-spatial``'s own ``tri``/``tpi`` functions
      only exist from 0.6 onward (under ``xrspatial.terrain_metrics``), and
      0.6+ requires python>=3.12 -- incompatible with this project's pinned
      python=3.11 (confirmed: conda-forge has no python-3.11 build past
      0.5.3, which has neither function anywhere in its source tree).
      ``_tri``/``_tpi`` below reimplement the exact kernels of the newer
      library's ``terrain_metrics._tri_cpu``/``_tpi_cpu`` (verified against
      xarray-spatial's GitHub source: 3x3 window, Riley et al. 1999 TRI /
      Weiss 2001 TPI, edges -> NaN), vectorized with numpy instead of numba;
  (e) ``ksat``: the installed ``pygeohydro==0.17.1`` predates
      ``soil_polaris`` (added in 0.19.3); upgrading pygeohydro pulls in
      numpy>=2 across the whole shared conda env (breaking the numba/numpy1
      ABI the rest of the pinned stack -- xarray-spatial, other completed
      pipeline stages -- assumes), so that upgrade was rejected as too
      broad a blast radius for this port. ``_read_polaris_layer`` below
      reimplements ``soil_polaris`` directly against the public POLARIS VRT
      endpoint it wraps (verified against pygeohydro's 0.19.4 source),
      producing the identical quantity (topsoil Ksat, log10(cm/hr), mean of
      the 0-5/5-15/15-30 cm layers) with zero environment changes.

Every ``acquire_*`` function is idempotent/cache-aware: if the output
``.tif``(s) already exist under ``data/processed/predictors/``, they are
loaded from disk instead of re-fetched, so re-running after a downstream
fix doesn't re-download the whole stack. ``acquire_all()`` treats the AOI
and DEM as hard prerequisites (everything else aligns to the DEM grid, so
their failure is unrecoverable and propagates immediately); beyond that,
each remaining source's failure is caught, reported by name, and the rest
still proceed. Before returning, ``acquire_all()`` then checks the
resulting stack against the canonical ``EXPECTED_NON_CENSUS`` names (plus
``CENSUS_LAYERS`` if a census key is present) and raises, naming every
missing layer alongside the recorded per-source failures, if any are
absent -- for ANY reason, not only an explicit source failure, since a
failed prerequisite can silently skip a dependent layer (e.g.
``curve_number`` after a failed ``nlcd``/``hydrologic_soil_group``)
without that skip itself being recorded as a failure. A caller only ever
sees a complete stack or an exception, never a silently partial one.
"""
import tempfile
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pygeohydro as gh
import py3dep
import pynhd
import rasterio
import requests
import rioxarray  # noqa: F401 -- registers the .rio accessor
import shapely
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from scipy.ndimage import distance_transform_edt, uniform_filter
from shapely.geometry import box
from xrspatial import curvature, slope

from pipeline import config

RAW_DIR = config.REPO / "data" / "raw"
OUT_DIR = config.DATA / "predictors"
AOI_DIR = config.DATA / "aoi"
NHD_DIR = config.DATA / "nhd"
INFRA_DIR = config.DATA / "infrastructure"
GDB_PATH = RAW_DIR / "infrastructure" / "extracted" / "commondata" / "mhk_stormwater_export.gdb"

TGT_CRS = "EPSG:26914"


# --------------------------------------------------------------------------
# Cache / save helpers
# --------------------------------------------------------------------------

def _load_cached(name, out_dir=OUT_DIR):
    """Return the cached layer from ``out_dir/<name>.tif`` if present, else
    None. Loaded 2-D (band dim squeezed away) with nodata masked to NaN.
    """
    path = out_dir / f"{name}.tif"
    if not path.exists():
        return None
    arr = rioxarray.open_rasterio(path, masked=True)
    if "band" in arr.dims:
        arr = arr.squeeze("band", drop=True)
    arr.name = name
    print(f"[predictors] cache hit: {name:<24} <- {path}")
    return arr


def _save(arr, name, dem, out_dir=OUT_DIR):
    """Force-align to the DEM grid's CRS/transform (paranoia against tiny
    floating-point drift from independent reproject_match calls -- mirrors
    the original notebook's own ``_save`` helper), assert the shapes truly match, force NaN
    as the array's nodata value, write to ``<out_dir>/<name>.tif``, and
    return the saved array.

    The ``write_nodata(np.nan, encoded=True)`` call closes a real bug found
    in ``flow_derivatives``: ``twi``/``spi`` are built via
    ``acc.copy(data=...)`` from an array read back (``masked=True``) from a
    WhiteboxTools raster whose on-disk nodata TAG is ``-32768`` (even where
    no cell actually holds that value) -- rioxarray decodes that into
    ``encoded_nodata``/``encoding['_FillValue']=-32768``, and ``.copy()``
    carries that stale encoding over to the new array even though its
    ``.rio.nodata`` correctly reports NaN. Without overriding it here,
    ``to_raster()`` used that inherited ``-32768`` encoding to re-encode the
    array's genuine (mathematical) NaN cells back to ``-32768`` on write --
    silently, since nothing else in the array looked wrong. ``encoded=True``
    is required, not optional: plain ``write_nodata(np.nan)`` only sets
    ``attrs['_FillValue']`` and leaves the stale ``encoding['_FillValue']``
    in place, which then makes ``to_raster()`` raise
    (``ValueError: Key '_FillValue' already exists in attrs...``) instead of
    fixing anything (verified empirically against this project's pinned
    rioxarray==0.19.0/xarray==2026.7.0) -- ``encoded=True`` is what actually
    clears the stale encoding, per rioxarray's own documented "mask with
    nodata" round-trip pattern. Applying this uniformly to every layer this
    module saves (all of which already use NaN as their in-memory nodata
    convention -- see e.g. ``acquire_nlcd``/``acquire_hsg``/
    ``_read_polaris_layer``'s own ``write_nodata(np.nan)`` calls, unaffected
    by adding this) closes the hole for good instead of only at the one
    call site that happened to trip it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assert arr.rio.shape == dem.rio.shape, (
        f"[predictors] {name}: shape {arr.rio.shape} != dem shape {dem.rio.shape} "
        "-- reproject_match produced a misaligned grid"
    )
    arr = arr.rio.write_crs(dem.rio.crs)
    arr = arr.rio.write_transform(dem.rio.transform())
    arr = arr.rio.write_nodata(np.nan, encoded=True)
    arr.name = name
    path = out_dir / f"{name}.tif"
    arr.rio.to_raster(path)
    lo, hi = float(arr.min()), float(arr.max())
    print(f"[predictors] {name:<24} range: {lo:>10.3f} to {hi:<10.3f}  "
          f"saved: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return arr


def _ensure_2d(arr):
    """Guarantee ``arr.values`` is 2-D ``(y, x)`` by squeezing away a
    size-1 ``band`` dim if present (mirrors ``_load_cached``'s own squeeze
    convention), falling back to squeezing any other leftover singleton
    dim. Idempotent -- a no-op if ``arr`` is already 2-D.

    Guards against mixing layers that came straight from an in-memory
    ``acquire_*`` return (which can still carry a singleton band dim, i.e.
    shape ``(1, y, x)``) with layers that round-tripped through
    ``_load_cached`` (already squeezed to ``(y, x)``): a boolean mask built
    across a 2-D and a 3-D array (e.g. ``~np.isnan(a) & ~np.isnan(b)``)
    silently broadcasts to 3-D, which then can't index a genuinely-2-D
    array (``IndexError: too many indices for array``) -- see
    ``acquire_curve_number``, the one place this project mixes multiple
    independently-sourced arrays via raw ``.values`` indexing.
    """
    if "band" in arr.dims:
        arr = arr.squeeze("band", drop=True)
    elif arr.values.ndim > 2:
        arr = arr.squeeze(drop=True)
    return arr


def _all_cached(cached):
    """True iff every value in a ``{name: DataArray-or-None}`` cache-lookup
    dict (as built by the ``{n: _load_cached(n, out_dir) for n in names}``
    idiom used throughout this module) is present.

    Deliberately checks ``v is not None`` rather than truthiness of ``v``
    itself: multi-name ``acquire_*`` functions used to write
    ``all(cached.values())`` directly, which calls ``bool()`` on each
    xarray DataArray in turn. ``bool()`` of a DataArray is only defined for
    single-element arrays -- for the full-grid rasters here it raises
    ``ValueError: The truth value of an array with more than one element is
    ambiguous``. Because ``all()`` short-circuits on the *first* item, this
    stayed hidden as long as at least one of the two names was still
    uncached (``None`` sorts first, falsy, no ``bool()`` call needed) --
    i.e. on a first-ever run -- and only surfaced once every name in a
    group was already cached on disk (both values real arrays -> ``bool()``
    called on the first one -> crash), breaking the very re-run/idempotency
    contract the caching is there for.
    """
    return all(v is not None for v in cached.values())


# --------------------------------------------------------------------------
# Area of interest -- Manhattan, KS city limits (Census TIGER Places)
# buffered by 2 km. Mirrors cell 2 of the original notebook.
# --------------------------------------------------------------------------

def _aoi():
    """Compute (or load cached) the AOI bounding box and its representations
    needed downstream. Cached via ``data/processed/aoi/aoi_bbox.gpkg`` so a
    re-run doesn't re-download the national TIGER Places shapefile.

    Returns (aoi_bbox, aoi_gdf, aoi_4326_geom, aoi_4326_gdf):
      aoi_bbox      -- (minx, miny, maxx, maxy) tuple in TGT_CRS
      aoi_gdf       -- 1-row GeoDataFrame of the bbox rectangle, TGT_CRS
      aoi_4326_geom -- aoi_gdf's geometry reprojected to EPSG:4326 (for
                       pygeohydro calls that require geo_crs=4326)
      aoi_4326_gdf  -- aoi_gdf reprojected to EPSG:4326, GeoDataFrame form
                       (for gh.nlcd_bygeom)
    """
    AOI_DIR.mkdir(parents=True, exist_ok=True)
    bbox_path = AOI_DIR / "aoi_bbox.gpkg"

    if bbox_path.exists():
        aoi_gdf = gpd.read_file(bbox_path)
        print(f"[predictors] cache hit: aoi <- {bbox_path}")
    else:
        places_url = "https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_20_place.zip"
        places = gpd.read_file(places_url)
        city_limits = places[places["NAME"] == "Manhattan"].to_crs(TGT_CRS)
        city_limits.to_file(AOI_DIR / "manhattan_city_limits.gpkg", driver="GPKG")

        aoi_buffered = city_limits.buffer(2000)
        minx, miny, maxx, maxy = aoi_buffered.total_bounds
        aoi_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=TGT_CRS)
        aoi_gdf.to_file(bbox_path, driver="GPKG")
        print(f"[predictors] city limits bounds (UTM 14N): {city_limits.total_bounds}")
        print(f"[predictors] AOI bbox (city limits + 2km buffer, UTM 14N): {(minx, miny, maxx, maxy)}")

    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    aoi_bbox = (minx, miny, maxx, maxy)
    print(f"[predictors] AOI size: {(maxx - minx) / 1000:.1f} km x {(maxy - miny) / 1000:.1f} km")

    aoi_4326_gdf = aoi_gdf.to_crs(4326)
    aoi_4326_geom = aoi_4326_gdf.geometry.iloc[0]
    return aoi_bbox, aoi_gdf, aoi_4326_geom, aoi_4326_gdf


# --------------------------------------------------------------------------
# DEM -- USGS 3DEP (10 m). Mirrors cell 3 of the original notebook.
# --------------------------------------------------------------------------

def acquire_dem(aoi_bbox, out_dir=OUT_DIR):
    """USGS 3DEP DEM, 10 m, via py3dep (keyless)."""
    cached = _load_cached("dem", out_dir)
    if cached is not None:
        return cached

    dem = py3dep.get_dem(aoi_bbox, resolution=10, crs=TGT_CRS)
    dem = dem.rio.reproject(TGT_CRS, resolution=(10, 10))
    dem = dem.rio.clip_box(*aoi_bbox)
    dem.name = "dem"
    dem = dem.rio.write_crs(TGT_CRS)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dem.tif"
    dem.rio.to_raster(path)
    print(f"[predictors] dem                      shape={dem.shape} "
          f"res={dem.rio.resolution()} range={float(dem.min()):.1f}-{float(dem.max()):.1f} m "
          f"saved: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return dem


# --------------------------------------------------------------------------
# Local terrain derivatives -- slope, curvature, TRI, TPI. Mirrors cell 4 of
# the original notebook. TWI/SPI/HAND (flow-routing-based) are built by
# flow_derivatives() below.
# --------------------------------------------------------------------------

def _tri(dem: xr.DataArray) -> xr.DataArray:
    """Terrain Ruggedness Index (Riley et al. 1999): sqrt(sum of squared
    elevation differences to the 8-connected neighbors), 3x3 window, edges
    -> NaN. See module docstring, deviation (d), for why this is a local
    reimplementation rather than a library call.
    """
    z = dem.values.astype("float64")
    pad = np.pad(z, 1, mode="constant", constant_values=np.nan)
    total = np.zeros_like(z)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = pad[1 + dy: 1 + dy + z.shape[0], 1 + dx: 1 + dx + z.shape[1]]
            diff = neighbor - z
            total += diff * diff
    result = dem.copy(data=np.sqrt(total).astype("float32"))
    result.name = "tri"
    return result


def _tpi(dem: xr.DataArray) -> xr.DataArray:
    """Topographic Position Index (Weiss 2001): center elevation minus the
    mean elevation of its 8-connected neighbors, 3x3 window, edges -> NaN.
    See module docstring, deviation (d).
    """
    z = dem.values.astype("float64")
    pad = np.pad(z, 1, mode="constant", constant_values=np.nan)
    total = np.zeros_like(z)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = pad[1 + dy: 1 + dy + z.shape[0], 1 + dx: 1 + dx + z.shape[1]]
            total += neighbor
    result = dem.copy(data=(z - total / 8.0).astype("float32"))
    result.name = "tpi"
    return result


TERRAIN_FUNCS = {"slope": slope, "curvature": curvature, "tri": _tri, "tpi": _tpi}


def acquire_terrain(dem, out_dir=OUT_DIR):
    """Local terrain derivatives -- simple neighborhood ops on the 10 m DEM,
    no flow routing. Each is cached/computed independently.

    ``dem`` is normalized to 2-D via ``_ensure_2d`` up front, mirroring the
    same guard in ``acquire_curve_number`` -- see ``_ensure_2d``'s docstring
    for why (band-dim inconsistency between in-memory and cache-loaded
    arrays). Safe/no-op here today since ``acquire_all()`` always passes a
    fresh, already-2-D ``dem``, but keeps the invariant explicit and holds
    if that ever changes.
    """
    dem = _ensure_2d(dem)
    out = {}
    for name, func in TERRAIN_FUNCS.items():
        cached = _load_cached(name, out_dir)
        if cached is not None:
            out[name] = cached
            continue
        result = func(dem)
        out[name] = _save(result, name, dem, out_dir)
    return out


# --------------------------------------------------------------------------
# NHD High-Resolution flowlines -- distance to streams, drainage density.
# Mirrors cell 6 of the original notebook.
# --------------------------------------------------------------------------

NHD_RETRY_BACKOFFS_S = (5, 15, 45)  # exponential backoff, seconds between attempts


def _retry(fn, *, attempts=3, backoffs=NHD_RETRY_BACKOFFS_S, label="operation"):
    """Call ``fn()`` (a zero-arg callable), retrying on any exception up to
    ``attempts`` times with exponential backoff (default 5s/15s/45s between
    attempts). Re-raises the final exception if every attempt fails.

    Hardens against transient national-map / hydro.nationalmap.gov service
    outages (observed here: a ``ServiceError`` from the NHD flowline_hr
    MapServer endpoint that turned out to be transient -- the flowline_mr
    endpoint and, on retest minutes later, flowline_hr itself both
    responded normally) so a one-off blip doesn't fail the whole
    ``acquire_all()`` build.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                wait = backoffs[min(attempt - 1, len(backoffs) - 1)]
                print(f"[predictors] {label}: attempt {attempt}/{attempts} failed "
                      f"({type(exc).__name__}: {exc}); retrying in {wait}s")
                time.sleep(wait)
    raise last_exc


def acquire_nhd(aoi_bbox, dem, out_dir=OUT_DIR):
    """NHD HR flowlines (pynhd) -> distance_to_streams, drainage_density.
    The ``bygeom`` fetch is wrapped in ``_retry`` -- the national-map
    flowline_hr MapServer endpoint has shown transient ``ServiceError``
    outages that self-heal within minutes.
    """
    names = ("distance_to_streams", "drainage_density")
    cached = {n: _load_cached(n, out_dir) for n in names}
    if _all_cached(cached):
        return cached

    NHD_DIR.mkdir(parents=True, exist_ok=True)
    nhd = pynhd.NHD("flowline_hr")
    flowlines = _retry(lambda: nhd.bygeom(aoi_bbox, geo_crs=TGT_CRS),
                        label="NHD flowline_hr.bygeom")
    flowlines = flowlines.to_crs(TGT_CRS)
    flowlines.to_file(NHD_DIR / "flowlines_hr.gpkg", driver="GPKG")
    print(f"[predictors] NHD flowlines: {len(flowlines)} features")

    transform = dem.rio.transform()
    out_shape = dem.shape
    stream_mask = rasterize(
        [(geom, 1) for geom in flowlines.geometry if geom is not None],
        out_shape=out_shape, transform=transform, fill=0, dtype="uint8",
    )
    cell_size = abs(transform.a)

    # --- Distance to streams (m) ---
    dist = distance_transform_edt(stream_mask == 0) * cell_size
    dist_arr = dem.copy(data=dist.astype("float32"))
    dist_arr = dist_arr.where(~dem.isnull())

    # --- Drainage density (km stream / km^2) within 500 m-radius window ---
    win_cells = int(round(500 / cell_size)) * 2 + 1
    stream_len_km = stream_mask.astype("float64") * (cell_size / 1000)
    window_sum_km = uniform_filter(stream_len_km, size=win_cells, mode="constant") * (win_cells ** 2)
    window_area_km2 = (win_cells * cell_size / 1000) ** 2
    density = window_sum_km / window_area_km2
    density_arr = dem.copy(data=density.astype("float32"))
    density_arr = density_arr.where(~dem.isnull())

    return {
        "distance_to_streams": _save(dist_arr, "distance_to_streams", dem, out_dir),
        "drainage_density": _save(density_arr, "drainage_density", dem, out_dir),
    }


# --------------------------------------------------------------------------
# Soils -- Ksat (POLARIS), available water storage + depth to restriction
# (gNATSGO). Mirrors cell 7 of the original notebook.
# --------------------------------------------------------------------------

POLARIS_BASE_URL = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt"
# cm-depth-band suffixes used in the POLARIS VRT filenames, keyed by the
# notebook-style layer suffix (ksat_5 -> "5" -> band "0_5", etc.)
POLARIS_DEPTH_KEY = {"5": "0_5", "15": "5_15", "30": "15_30"}


def _read_polaris_layer(prop, depth_key, aoi_4326_geom):
    """Read one POLARIS property/depth VRT, clipped to the AOI. Reimplements
    pygeohydro>=0.19.3's ``soil_polaris()`` directly against the public
    POLARIS VRT endpoint it wraps -- see module docstring, deviation (e).
    """
    depth = POLARIS_DEPTH_KEY[depth_key]
    url = f"{POLARIS_BASE_URL}/{prop}_mean_{depth}.vrt"
    da = rioxarray.open_rasterio(url).squeeze(drop=True)
    minx, miny, maxx, maxy = aoi_4326_geom.bounds
    da = da.rio.clip_box(minx, miny, maxx, maxy, crs=4326)
    da = da.rio.clip([aoi_4326_geom], crs=4326)
    da = da.where(da != da.rio.nodata)
    da = da.rio.write_nodata(np.nan)
    return da


def acquire_ksat(aoi_4326_geom, dem, out_dir=OUT_DIR):
    """Topsoil Ksat (POLARIS, 0-30 cm mean, log10(cm/hr))."""
    cached = _load_cached("ksat", out_dir)
    if cached is not None:
        return {"ksat": cached}

    layers = [_read_polaris_layer("ksat", d, aoi_4326_geom) for d in ("5", "15", "30")]
    ksat = xr.concat(layers, dim="depth").mean(dim="depth")
    ksat = ksat.rio.write_crs(4326)
    ksat_10m = ksat.rio.reproject_match(dem, resampling=Resampling.bilinear)
    ksat_10m.name = "ksat"
    return {"ksat": _save(ksat_10m, "ksat", dem, out_dir)}


def acquire_gnatsgo_soils(aoi_4326_geom, dem, out_dir=OUT_DIR):
    """Available water storage (aws0_100) + depth to restrictive layer
    (tk0_100a) via gNATSGO / Microsoft Planetary Computer.
    """
    names = ("available_water_storage", "depth_to_restriction")
    cached = {n: _load_cached(n, out_dir) for n in names}
    if _all_cached(cached):
        return cached

    gnatsgo = gh.soil_gnatsgo(["aws0_100", "tk0_100a"], aoi_4326_geom, crs=4326)

    aws_10m = gnatsgo["aws0_100"].rio.reproject_match(dem, resampling=Resampling.bilinear)
    aws_10m.name = "available_water_storage"

    tk_10m = gnatsgo["tk0_100a"].rio.reproject_match(dem, resampling=Resampling.bilinear)
    tk_10m.name = "depth_to_restriction"

    return {
        "available_water_storage": _save(aws_10m, "available_water_storage", dem, out_dir),
        "depth_to_restriction": _save(tk_10m, "depth_to_restriction", dem, out_dir),
    }


# --------------------------------------------------------------------------
# Land cover -- NLCD 2021 (land cover class, % impervious). Mirrors cell 8
# of the original notebook.
# --------------------------------------------------------------------------

def acquire_nlcd(aoi_4326_gdf, dem, out_dir=OUT_DIR):
    """NLCD 2021 land cover (categorical, nearest-neighbor) + % impervious
    (continuous, bilinear), via pygeohydro's MRLC web service query.
    """
    names = ("nlcd_landcover", "impervious_pct")
    cached = {n: _load_cached(n, out_dir) for n in names}
    if _all_cached(cached):
        return cached

    nlcd = gh.nlcd_bygeom(aoi_4326_gdf, resolution=30, years={"cover": [2021], "impervious": [2021]}, crs=4326)
    nlcd_ds = next(iter(nlcd.values()))

    # --- Land cover (categorical, nearest-neighbor) ---
    cover = nlcd_ds["cover_2021"].astype("float32")
    cover = cover.where(cover != 127)  # 127 = fill value at AOI edges, not a valid NLCD class
    cover = cover.rio.write_crs(nlcd_ds.rio.crs)
    cover = cover.rio.write_nodata(np.nan)  # overwrite inherited nodata=127, or reproject refills NaN with 127
    cover_10m = cover.rio.reproject_match(dem, resampling=Resampling.nearest)
    cover_10m.name = "nlcd_landcover"

    # --- Impervious surface % (continuous, bilinear) ---
    imperv = nlcd_ds["impervious_2021"].rio.write_crs(nlcd_ds.rio.crs)
    imperv_10m = imperv.rio.reproject_match(dem, resampling=Resampling.bilinear)
    imperv_10m.name = "impervious_pct"

    return {
        "nlcd_landcover": _save(cover_10m, "nlcd_landcover", dem, out_dir),
        "impervious_pct": _save(imperv_10m, "impervious_pct", dem, out_dir),
    }


# --------------------------------------------------------------------------
# Hydrologic soil group & curve number (SDA + NLCD). Mirrors notebook cell 11.
# --------------------------------------------------------------------------

# A=1, B=2, C=3, D=4; dual classes (e.g. "A/D") use the undrained (second) letter
HSG_CODE = {"A": 1, "B": 2, "C": 3, "D": 4}


def acquire_hsg(aoi_4326_geom, dem, out_dir=OUT_DIR):
    """NRCS hydrologic soil group (A/B/C/D -> 1-4): dominant component's
    hydgrp per map unit via Soil Data Access, joined to the gNATSGO mukey
    raster (Microsoft Planetary Computer). Resampled nearest-neighbor.
    """
    cached = _load_cached("hydrologic_soil_group", out_dir)
    if cached is not None:
        return {"hydrologic_soil_group": cached}

    mukey_ds = gh.soil_gnatsgo("mukey", aoi_4326_geom, crs=4326)
    mukey_da = mukey_ds["mukey"]

    mukeys = np.unique(mukey_da.values)
    mukeys = mukeys[~np.isnan(mukeys)].astype(int)

    mukey_list = ",".join(str(m) for m in mukeys)
    sql = f"SELECT mukey, comppct_r, hydgrp FROM component WHERE mukey IN ({mukey_list})"
    resp = requests.post("https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest",
                          json={"query": sql, "format": "JSON"}, timeout=60)
    resp.raise_for_status()
    comp = pd.DataFrame(resp.json()["Table"], columns=["mukey", "comppct_r", "hydgrp"])
    comp["comppct_r"] = comp["comppct_r"].astype(float)
    comp = comp.dropna(subset=["hydgrp"])
    dominant = comp.loc[comp.groupby("mukey")["comppct_r"].idxmax()].set_index("mukey")["hydgrp"]

    def _hydgrp_to_code(mukey_val):
        if np.isnan(mukey_val):
            return np.nan
        letter = dominant.get(str(int(mukey_val)))
        if letter is None:
            return np.nan
        letter = letter.split("/")[-1]
        return HSG_CODE.get(letter, np.nan)

    hsg_arr = np.vectorize(_hydgrp_to_code)(mukey_da.values).astype("float32")
    hsg = mukey_da.copy(data=hsg_arr)
    hsg = hsg.rio.write_crs(mukey_da.rio.crs)
    hsg = hsg.rio.write_nodata(np.nan)
    hsg_10m = hsg.rio.reproject_match(dem, resampling=Resampling.nearest)
    hsg_10m.name = "hydrologic_soil_group"

    return {"hydrologic_soil_group": _save(hsg_10m, "hydrologic_soil_group", dem, out_dir)}


# CN by [A, B, C, D], good hydrologic condition (USDA NRCS TR-55 Table 2-2)
CN_TABLE = {
    11: [100, 100, 100, 100],  # open water
    31: [77, 86, 91, 94],      # barren land (fallow, bare soil)
    41: [30, 55, 70, 77],      # deciduous forest
    42: [30, 55, 70, 77],      # evergreen forest
    43: [30, 55, 70, 77],      # mixed forest
    51: [30, 48, 65, 73],      # dwarf scrub
    52: [30, 48, 65, 73],      # shrub/scrub (brush, good condition)
    71: [30, 58, 71, 78],      # grassland/herbaceous (meadow, continuous grass)
    72: [30, 58, 71, 78],      # sedge/herbaceous
    73: [30, 58, 71, 78],      # lichens
    74: [30, 58, 71, 78],      # moss
    81: [39, 61, 74, 80],      # pasture/hay (good condition)
    82: [67, 78, 85, 89],      # cultivated crops (row crops, straight row, good condition)
    90: [30, 55, 70, 77],      # woody wetlands -> treated as forest
    95: [30, 58, 71, 78],      # emergent herbaceous wetlands -> treated as meadow
}
# Developed classes: composite of "open space, good condition" pervious CN and CN=98 for the impervious fraction
OPEN_SPACE_CN = {1: 39, 2: 61, 3: 74, 4: 80}
DEVELOPED_CLASSES = {21, 22, 23, 24}


def acquire_curve_number(dem, hsg, nlcd_cover, imperv_pct, out_dir=OUT_DIR):
    """NRCS-TR55 runoff curve number from hydrologic_soil_group x
    nlcd_landcover (good hydrologic condition); developed classes are a
    composite of pervious "open space, good condition" CN and CN=98 for the
    impervious fraction. Open water fixed at CN=100. Purely local -- no
    network calls; requires hsg + nlcd_cover + imperv_pct already acquired.

    ``dem``/``hsg``/``nlcd_cover``/``imperv_pct`` are each normalized to
    2-D via ``_ensure_2d`` before use -- see that helper's docstring for
    why (band-dim inconsistency between in-memory and cache-loaded arrays).
    """
    cached = _load_cached("curve_number", out_dir)
    if cached is not None:
        return {"curve_number": cached}

    # Normalize every input to 2-D (y, x) before any mask/index logic
    # below. Layers pulled straight from acquire_all()'s in-memory stack
    # can still carry a singleton band dim (3-D, e.g. (1, y, x)); layers
    # reloaded from cache are already squeezed (2-D). Mixing the two in
    # `~np.isnan(nlcd_arr) & ~np.isnan(hsg_arr_10m)` below silently
    # broadcasts to 3-D, which then can't index the (2-D) array --
    # "IndexError: too many indices for array: array is 2-dimensional, but
    # 3 were indexed".
    dem = _ensure_2d(dem)
    hsg = _ensure_2d(hsg)
    nlcd_cover = _ensure_2d(nlcd_cover)
    imperv_pct = _ensure_2d(imperv_pct)

    shapes = {"dem": dem.values.shape, "hydrologic_soil_group": hsg.values.shape,
              "nlcd_landcover": nlcd_cover.values.shape, "impervious_pct": imperv_pct.values.shape}
    assert len(set(shapes.values())) == 1, (
        f"[predictors] curve_number: input layer shapes disagree after 2-D normalization: {shapes}"
    )

    nlcd_arr = nlcd_cover.values
    hsg_arr_10m = hsg.values
    imp_arr = imperv_pct.values

    cn_arr = np.full(nlcd_arr.shape, np.nan, dtype="float32")

    valid = ~np.isnan(nlcd_arr) & ~np.isnan(hsg_arr_10m)
    nlcd_v = nlcd_arr[valid].astype(int)
    hsg_v = hsg_arr_10m[valid].astype(int)
    imp_v = imp_arr[valid]
    cn_v = np.full(nlcd_v.shape, np.nan, dtype="float32")

    dev_mask = np.isin(nlcd_v, list(DEVELOPED_CLASSES))
    open_cn = np.array([OPEN_SPACE_CN[h] for h in hsg_v[dev_mask]], dtype="float32")
    cn_v[dev_mask] = open_cn * (1 - imp_v[dev_mask] / 100) + 98 * (imp_v[dev_mask] / 100)

    for cls, cn_by_hsg in CN_TABLE.items():
        for h_code, cn_val in zip([1, 2, 3, 4], cn_by_hsg):
            m = (~dev_mask) & (nlcd_v == cls) & (hsg_v == h_code)
            cn_v[m] = cn_val

    cn_arr[valid] = cn_v
    # Open water is CN=100 regardless of soil (covers pixels where HSG is unavailable for water mukeys)
    cn_arr[nlcd_arr == 11] = 100

    cn = nlcd_cover.copy(data=cn_arr)
    cn.name = "curve_number"
    return {"curve_number": _save(cn, "curve_number", dem, out_dir)}


# --------------------------------------------------------------------------
# Stormwater infrastructure -- City of Manhattan Public Works (local GDB,
# no network I/O). Mirrors notebook cell 12.
# --------------------------------------------------------------------------

def acquire_stormwater(dem, out_dir=OUT_DIR):
    """Stormwater network (pipes + open channels) -> stormwater_density;
    discharge outfalls (DISPTTYPE == 'Outfall') -> distance_to_outfall.
    """
    names = ("stormwater_density", "distance_to_outfall")
    cached = {n: _load_cached(n, out_dir) for n in names}
    if _all_cached(cached):
        return cached

    INFRA_DIR.mkdir(parents=True, exist_ok=True)

    pipes = gpd.read_file(GDB_PATH, layer="Stormwater_Pipes").to_crs(TGT_CRS)
    channels = gpd.read_file(GDB_PATH, layer="Stormwater_Channels").to_crs(TGT_CRS)
    discharge = gpd.read_file(GDB_PATH, layer="Stormwater_Discharge_Points").to_crs(TGT_CRS)
    outfalls = discharge[discharge["DISPTTYPE"] == "Outfall"][["geometry"]].copy()

    stormwater_lines = gpd.GeoDataFrame(
        pd.concat([pipes[["geometry"]], channels[["geometry"]]], ignore_index=True), crs=TGT_CRS,
    )
    stormwater_lines["geometry"] = shapely.force_2d(stormwater_lines.geometry.values)
    stormwater_lines.to_file(INFRA_DIR / "stormwater_lines.gpkg", driver="GPKG")

    outfalls["geometry"] = shapely.force_2d(outfalls.geometry.values)
    outfalls.to_file(INFRA_DIR / "stormwater_outfalls.gpkg", driver="GPKG")

    print(f"[predictors] stormwater lines (pipes + channels): {len(stormwater_lines)} features")
    print(f"[predictors] stormwater outfalls: {len(outfalls)} features")

    transform = dem.rio.transform()
    out_shape = dem.shape
    cell_size = abs(transform.a)

    # --- Stormwater density (km pipe+channel / km^2) within 500 m-radius window ---
    stormwater_mask = rasterize(
        [(geom, 1) for geom in stormwater_lines.geometry if geom is not None],
        out_shape=out_shape, transform=transform, fill=0, dtype="uint8",
    )
    win_cells = int(round(500 / cell_size)) * 2 + 1
    stormwater_len_km = stormwater_mask.astype("float64") * (cell_size / 1000)
    window_sum_km = uniform_filter(stormwater_len_km, size=win_cells, mode="constant") * (win_cells ** 2)
    window_area_km2 = (win_cells * cell_size / 1000) ** 2
    density = window_sum_km / window_area_km2
    density_arr = dem.copy(data=density.astype("float32"))
    density_arr = density_arr.where(~dem.isnull())

    # --- Distance to nearest stormwater outfall (m) ---
    outfall_mask = rasterize(
        [(geom, 1) for geom in outfalls.geometry if geom is not None],
        out_shape=out_shape, transform=transform, fill=0, dtype="uint8",
    )
    outfall_dist = distance_transform_edt(outfall_mask == 0) * cell_size
    outfall_dist_arr = dem.copy(data=outfall_dist.astype("float32"))
    outfall_dist_arr = outfall_dist_arr.where(~dem.isnull())

    return {
        "stormwater_density": _save(density_arr, "stormwater_density", dem, out_dir),
        "distance_to_outfall": _save(outfall_dist_arr, "distance_to_outfall", dem, out_dir),
    }


# --------------------------------------------------------------------------
# Census ACS -- population density, median household income. Mirrors notebook
# cell 10. DEFERRED: acquire_all() only calls this when a census API key is
# present (see module docstring, change (c)).
# --------------------------------------------------------------------------

CENSUS_COUNTIES = ["161", "149"]  # Riley, Pottawatomie


def acquire_census(dem, aoi_gdf, out_dir=OUT_DIR):
    """ACS 2018-2022 5-year estimates (block-group level, Riley +
    Pottawatomie counties) -> population_density, median_income. Requires
    ``(config.REPO / '.census_api_key')`` to exist -- callers (acquire_all)
    are expected to check that before calling; this function reads the key
    unconditionally so it fails loudly if called without one.
    """
    names = ("population_density", "median_income")
    cached = {n: _load_cached(n, out_dir) for n in names}
    if _all_cached(cached):
        return cached

    census_key = (config.REPO / ".census_api_key").read_text().strip()

    acs_records = []
    for county in CENSUS_COUNTIES:
        params = {
            "get": "B01003_001E,B19013_001E",
            "for": "block group:*",
            "in": f"state:20 county:{county}",
            "key": census_key,
        }
        r = requests.get("https://api.census.gov/data/2022/acs/acs5", params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        acs_records.extend(rows[1:])

    acs_cols = ["B01003_001E", "B19013_001E", "state", "county", "tract", "block group"]
    acs = pd.DataFrame(acs_records, columns=acs_cols)
    acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"] + acs["block group"]
    acs["population"] = acs["B01003_001E"].astype("int64")
    acs["median_income"] = acs["B19013_001E"].astype("int64")
    acs.loc[acs["median_income"] == -666666666, "median_income"] = np.nan

    bg = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_20_bg.zip")
    bg = bg[bg["COUNTYFP"].isin(CENSUS_COUNTIES)].to_crs(TGT_CRS)
    bg = bg.merge(acs[["GEOID", "population", "median_income"]], on="GEOID")
    bg["population_density"] = bg["population"] / (bg["ALAND"] / 1e6)  # people per km^2

    bg_aoi = bg.clip(aoi_gdf)

    out_shape = dem.shape
    transform = dem.rio.transform()

    result = {}
    for varname, col in [("population_density", "population_density"), ("median_income", "median_income")]:
        shapes = [(geom, val) for geom, val in zip(bg_aoi.geometry, bg_aoi[col]) if not np.isnan(val)]
        arr = rasterize(shapes, out_shape=out_shape, transform=transform, fill=np.nan, dtype="float32")
        layer = dem.copy(data=arr)
        layer = layer.where(~dem.isnull())
        layer.name = varname
        result[varname] = _save(layer, varname, dem, out_dir)
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

# Canonical layer names ``acquire_all()`` must return -- the single source
# of truth for its completeness gate below, and imported verbatim by
# tests/test_predictors_align.py so the expected-layer list isn't
# maintained in two places. ``EXPECTED_NON_CENSUS`` is unconditional (every
# run, key or no key); ``CENSUS_LAYERS`` is only expected when a census API
# key is present (see module docstring, change (c)).
EXPECTED_NON_CENSUS = (
    "dem", "slope", "curvature", "tri", "tpi",
    "distance_to_streams", "drainage_density",
    "ksat", "available_water_storage", "depth_to_restriction",
    "hydrologic_soil_group", "curve_number",
    "nlcd_landcover", "impervious_pct",
    "distance_to_outfall", "stormwater_density",
)
CENSUS_LAYERS = ("population_density", "median_income")


def acquire_all():
    """Acquire every non-hydrology predictor layer, align each to the 10 m
    EPSG:26914 AOI grid (reproject_match to the DEM), and write
    ``data/processed/predictors/<name>.tif``. See module docstring for the
    full scope, deviations, caching, and robustness contract.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aoi_bbox, aoi_gdf, aoi_4326_geom, aoi_4326_gdf = _aoi()
    # DEM is a hard prerequisite -- every other layer aligns to its grid via
    # reproject_match, so there is nothing else that CAN be produced if this
    # fails. Let the failure propagate immediately rather than "report and
    # continue" (there is no "rest" to acquire without it).
    dem = acquire_dem(aoi_bbox)

    stack = {"dem": dem}
    failures = {}

    def _try(label, fn, *args, **kwargs):
        try:
            stack.update(fn(*args, **kwargs))
            return True
        except Exception as exc:
            print(f"[predictors] FAILED source '{label}': {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            failures[label] = exc
            return False

    _try("terrain", acquire_terrain, dem)
    _try("nhd", acquire_nhd, aoi_bbox, dem)
    _try("ksat", acquire_ksat, aoi_4326_geom, dem)
    _try("gnatsgo_soils", acquire_gnatsgo_soils, aoi_4326_geom, dem)
    _try("nlcd", acquire_nlcd, aoi_4326_gdf, dem)
    _try("hydrologic_soil_group", acquire_hsg, aoi_4326_geom, dem)

    cn_prereqs = ("hydrologic_soil_group", "nlcd_landcover", "impervious_pct")
    missing_cn_prereqs = [p for p in cn_prereqs if p not in stack]
    if missing_cn_prereqs:
        print(f"[predictors] SKIP curve_number: missing prerequisite layer(s) {missing_cn_prereqs}")
    else:
        _try("curve_number", acquire_curve_number, dem, stack["hydrologic_soil_group"],
             stack["nlcd_landcover"], stack["impervious_pct"])

    _try("stormwater", acquire_stormwater, dem)

    census_key_path = config.REPO / ".census_api_key"
    if census_key_path.exists():
        _try("census", acquire_census, dem, aoi_gdf)
    else:
        print("[predictors] SKIP census: no .census_api_key present")

    # Completeness gate -- acquire_all() must never hand a downstream caller
    # (e.g. flow routing or modeling) a silently partial stack. This SUBSUMES the
    # previous "raise only if >= 2 independent sources failed" guard: a
    # single failure can drop more than one layer (e.g. a failed 'nlcd'
    # takes 'nlcd_landcover'/'impervious_pct' AND causes curve_number to be
    # skipped via missing_cn_prereqs above, without ever touching
    # `failures`), so counting `failures` alone is not a reliable proxy for
    # stack completeness. Checking the actual stack contents against the
    # canonical name lists is -- any gap, from any cause, now fails fast
    # here instead of returning quietly.
    missing = [n for n in EXPECTED_NON_CENSUS if n not in stack]
    if census_key_path.exists():
        missing += [n for n in CENSUS_LAYERS if n not in stack]

    if missing:
        raise RuntimeError(
            f"[predictors] BLOCKED: acquire_all() produced an incomplete stack -- "
            f"missing layer(s) {missing}; recorded source failures: {failures}"
        )

    return stack


# --------------------------------------------------------------------------
# Flow routing (D8) -- flow_accumulation, TWI, SPI, HAND, GFI. Fixes a
# routing bug: the notebook's `xarray-spatial` D8 chain (`fill_d8` -> `flow_direction_d8`
# -> `flow_accumulation_d8`) cannot resolve flats and capped real-AOI flow
# accumulation at 2451 cells -- never reaching river scale, so TWI/SPI/HAND
# (and any GFI derived from them) were wrong. Replaced here with
# WhiteboxTools (flats-resolving: `breach_depressions_least_cost` then
# `d8_flow_accumulation`), run on the DEM buffered ~5 km beyond the AOI (to
# capture upstream contributing area the AOI-only grid would truncate at its
# own edge) and clipped/`reproject_match`ed back onto the AOI DEM grid
# afterward, so the outputs align pixel-exactly with the rest of the
# predictor stack. Deliberately separate from `acquire_all()` -- see module
# docstring, deviation (b).
# --------------------------------------------------------------------------

HYDRO_DIR = config.DATA / "hydrology"


def dem_path(out_dir=OUT_DIR):
    """Path to the AOI DEM ``acquire_dem()`` already wrote --
    the alignment target every hydrology derivative below
    ``reproject_match``es onto. A function (not a constant) so callers/
    tests reference the canonical location without hardcoding it.
    """
    return out_dir / "dem.tif"


def _wbt(working_dir):
    """Construct a ``WhiteboxTools`` instance rooted at ``working_dir``
    (created if missing). Local import -- ``whitebox`` is only needed for
    hydrology routing, not the rest of this module.
    """
    import whitebox

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(working_dir))
    return wbt


def _check_wbt(exit_code, label, output_path):
    """Raise if a WhiteboxTools call failed. The ``whitebox`` package's
    ``run_tool`` dispatcher returns 0/1/2 for success/error/cancelled but
    does NOT raise a Python exception on a tool-level error -- it only
    prints to its callback -- so a nonzero exit code (or a 0 that still
    didn't write the promised output) must be checked explicitly, or a
    failed routing step would silently propagate as a missing/stale file
    instead of a loud failure.
    """
    if exit_code != 0:
        raise RuntimeError(f"[predictors] WhiteboxTools '{label}' failed (exit code {exit_code})")
    if not Path(output_path).exists():
        raise RuntimeError(
            f"[predictors] WhiteboxTools '{label}' returned success (exit code 0) "
            f"but did not write {output_path}"
        )


def _route_d8(dem_tif_path, working_dir, breach_dist=100, wbt=None):
    """Breach depressions (least-cost) then compute D8 flow accumulation
    (cell count) for the DEM at ``dem_tif_path``, via WhiteboxTools -- the
    flats-resolving router that replaces the old xarray-spatial D8 chain
    (see the section comment above). Shared by ``flow_derivatives`` (the real,
    buffered AOI grid) and ``_d8_accumulation_synthetic`` (the flats
    regression test below) so the test exercises the exact same routing
    calls production uses, rather than a parallel reimplementation that
    could silently diverge.

    Cache-aware per intermediate file (mirrors the rest of this module):
    an existing ``dem_breached.tif``/``acc.tif`` under ``working_dir`` is
    reused rather than recomputed. Returns ``(breached_path, acc_path)``,
    both written into ``working_dir``.
    """
    working_dir = Path(working_dir)
    wbt = wbt if wbt is not None else _wbt(working_dir)
    breached_path = working_dir / "dem_breached.tif"
    acc_path = working_dir / "acc.tif"

    if breached_path.exists():
        print(f"[predictors] cache hit: dem_breached <- {breached_path}")
    else:
        rc = wbt.breach_depressions_least_cost(str(dem_tif_path), str(breached_path), dist=breach_dist)
        _check_wbt(rc, "breach_depressions_least_cost", breached_path)

    if acc_path.exists():
        print(f"[predictors] cache hit: acc <- {acc_path}")
    else:
        rc = wbt.d8_flow_accumulation(str(breached_path), str(acc_path), out_type="cells")
        _check_wbt(rc, "d8_flow_accumulation", acc_path)

    return breached_path, acc_path


def _d8_accumulation_synthetic(out_dir=None):
    """Build a 100x100 synthetic DEM -- tilted south (high north -> low
    south), with a 40-row flat PLATEAU band across the middle (rows
    30-69, pinned to one constant elevation, no north-south gradient at
    all), and a V-shaped valley cross-section (elevation rises 1 m per
    column moving away from the two center columns) present at every row,
    including through the plateau -- then route it through the exact same
    breach+accumulate WhiteboxTools chain ``flow_derivatives`` uses below,
    and return the resulting accumulation grid (plain ndarray).

    Regression guard for the capped-accumulation bug: within the plateau, the two center columns
    are perfectly flat in BOTH directions at once (zero north-south
    gradient from the plateau, zero east-west gradient because they sit
    at the valley's tied minimum) -- a genuine 2-D flat that a router
    without flat-resolution logic can leave without a defined flow
    direction. The old `xarray-spatial` D8 chain could not resolve flats
    like this and capped real-AOI flow accumulation at 2451 cells, never
    reaching river scale (see section docstring). Verified empirically
    (during development) that this exact DEM converges
    ~84.5% of the grid (8450/10000 cells) into the southern outlet under
    the WhiteboxTools chain used here -- comfortably past the >1000 gate.
    """
    n = 100
    rows = np.arange(n).reshape(-1, 1).astype("float64")
    cols = np.arange(n).reshape(1, -1).astype("float64")

    # Regional tilt, flattened into one constant-elevation plateau for
    # rows 30-69 (a 40-row genuine flat -- no row-to-row gradient there).
    tilt = np.where((rows >= 30) & (rows < 70), float(n - 30), n - rows)
    # V-shaped valley cross-section, present at every row (incl. the
    # plateau): elevation rises 1 m per column away from the tied-minimum
    # center columns -- the convergent channel every column drains toward.
    lateral = np.abs(cols - (n - 1) / 2.0)
    z = (tilt + lateral).astype("float32")

    out_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="wbt_synthetic_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_tif = out_dir / "dem_synthetic.tif"
    transform = rasterio.transform.from_origin(0, n * 10, 10, 10)
    profile = {
        "driver": "GTiff", "height": n, "width": n, "count": 1, "dtype": "float32",
        "crs": TGT_CRS, "transform": transform, "nodata": -9999.0,
    }
    with rasterio.open(dem_tif, "w", **profile) as dst:
        dst.write(z, 1)

    _breached_path, acc_path = _route_d8(dem_tif, out_dir)
    with rasterio.open(acc_path) as src:
        acc = src.read(1)
    return acc


def _aoi_bounds(dem_path):
    """(minx, miny, maxx, maxy) of the raster at ``dem_path``, in its own
    CRS -- the clip/alignment target used below, tied to whatever grid the
    caller actually passed rather than a separately cached AOI bbox that
    could drift a pixel from it.
    """
    da = rioxarray.open_rasterio(dem_path)
    return tuple(da.rio.bounds())


def _buffer_dem(dem_path, buffer_km, hydro_dir=HYDRO_DIR):
    """Fetch (or load cached) a DEM covering ``dem_path``'s AOI expanded by
    ``buffer_km`` beyond every edge, on the same 10 m EPSG:26914 convention
    as ``acquire_dem()`` -- captures upstream contributing area so D8
    routing on the AOI isn't truncated at the AOI's own edge.
    WhiteboxTools operates on file paths, so this writes (rather than
    returns in-memory) ``<hydro_dir>/dem_buffered.tif``.
    """
    hydro_dir = Path(hydro_dir)
    hydro_dir.mkdir(parents=True, exist_ok=True)
    out_path = hydro_dir / "dem_buffered.tif"
    if out_path.exists():
        print(f"[predictors] cache hit: dem_buffered <- {out_path}")
        return out_path

    minx, miny, maxx, maxy = _aoi_bounds(dem_path)
    buf = buffer_km * 1000.0
    buffered_bbox = (minx - buf, miny - buf, maxx + buf, maxy + buf)

    dem_buf = py3dep.get_dem(buffered_bbox, resolution=10, crs=TGT_CRS)
    dem_buf = dem_buf.rio.reproject(TGT_CRS, resolution=(10, 10))
    dem_buf = dem_buf.rio.clip_box(*buffered_bbox)
    dem_buf = dem_buf.rio.write_crs(TGT_CRS)
    dem_buf.rio.to_raster(out_path)
    print(f"[predictors] dem_buffered (+{buffer_km} km) shape={dem_buf.shape} "
          f"bounds={tuple(round(b) for b in buffered_bbox)} "
          f"saved: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def _streams(dem_buf_path, hydro_dir=HYDRO_DIR):
    """Rasterize the NHD HR flowlines (``data/processed/nhd/
    flowlines_hr.gpkg``) onto the buffered routing grid at
    ``dem_buf_path``, for WhiteboxTools' ``elevation_above_stream``.
    Cached at ``<hydro_dir>/streams.tif``.

    LIMITATION: those flowlines were fetched for the un-buffered AOI bbox
    (by ``acquire_nhd``), so they do not cover the full
    ``buffer_km`` ring around the AOI -- cells strictly inside the
    buffer-only margin have no local stream target and will get an
    over-large HAND value. This is acceptable: HAND
    is only clipped/used/saved for the AOI itself, and every AOI cell is
    still within the extent the original (unbuffered) NHD fetch covered,
    so it still reaches a real AOI-covering stream.
    """
    hydro_dir = Path(hydro_dir)
    out_path = hydro_dir / "streams.tif"
    if out_path.exists():
        print(f"[predictors] cache hit: streams <- {out_path}")
        return out_path

    grid = rioxarray.open_rasterio(dem_buf_path)
    grid = _ensure_2d(grid)
    flowlines = gpd.read_file(NHD_DIR / "flowlines_hr.gpkg")

    stream_mask = rasterize(
        [(geom, 1) for geom in flowlines.geometry if geom is not None],
        out_shape=grid.shape, transform=grid.rio.transform(), fill=0, dtype="uint8",
    )
    streams = grid.copy(data=stream_mask)
    streams = streams.rio.write_crs(grid.rio.crs)
    streams = streams.rio.write_nodata(0)
    streams.rio.to_raster(out_path)
    print(f"[predictors] streams (rasterized NHD onto routing grid): "
          f"{int(stream_mask.sum())} stream cells -> {out_path.name}")
    return out_path


def _slope_rad(dem_path):
    """Slope in radians (``xrspatial.slope`` returns degrees) on the grid
    at ``dem_path`` -- the denominator surface for TWI/SPI/GFI below.
    """
    dem = rioxarray.open_rasterio(dem_path, masked=True)
    dem = _ensure_2d(dem)
    return np.deg2rad(slope(dem))


def flow_derivatives(dem_path, buffer_km=5, out_dir=OUT_DIR, hydro_dir=HYDRO_DIR):
    """D8 ``flow_accumulation``/TWI/SPI/HAND on ``dem_path``'s AOI grid,
    routed on a DEM buffered ``buffer_km`` beyond the AOI (via
    WhiteboxTools) so accumulation reflects real upstream contributing
    area instead of being truncated at the AOI edge -- fixes the capped
    accumulation (old xarray-spatial D8 max: 2451 cells; see section comment). All four
    outputs are ``reproject_match``ed onto ``dem_path``'s exact grid
    before returning, so they align pixel-exactly with the rest of the
    predictor stack, and are saved to ``<out_dir>/<name>.tif``
    (cache-aware: an existing complete set of the four output files
    short-circuits the whole routing run, mirroring every other
    ``acquire_*`` function in this module).
    """
    names = ("flow_accumulation", "twi", "spi", "hand")
    cached = {n: _load_cached(n, out_dir) for n in names}
    if _all_cached(cached):
        return cached

    hydro_dir = Path(hydro_dir)
    wbt = _wbt(hydro_dir)

    dem_buf_path = _buffer_dem(dem_path, buffer_km, hydro_dir)
    breached_path, acc_path = _route_d8(dem_buf_path, hydro_dir, wbt=wbt)

    ptr_path = hydro_dir / "ptr.tif"
    if ptr_path.exists():
        print(f"[predictors] cache hit: ptr <- {ptr_path}")
    else:
        rc = wbt.d8_pointer(str(breached_path), str(ptr_path))
        _check_wbt(rc, "d8_pointer", ptr_path)

    streams_path = _streams(dem_buf_path, hydro_dir)

    hand_path = hydro_dir / "hand.tif"
    if hand_path.exists():
        print(f"[predictors] cache hit: hand <- {hand_path}")
    else:
        rc = wbt.elevation_above_stream(str(breached_path), str(streams_path), str(hand_path))
        _check_wbt(rc, "elevation_above_stream", hand_path)

    aoi_bounds = _aoi_bounds(dem_path)
    dem_target = rioxarray.open_rasterio(dem_path, masked=True)
    dem_target = _ensure_2d(dem_target)

    acc = rioxarray.open_rasterio(acc_path, masked=True)
    acc = _ensure_2d(acc).rio.clip_box(*aoi_bounds)
    acc = acc.rio.reproject_match(dem_target)
    acc.name = "flow_accumulation"
    print(f"[predictors] flow_accumulation (post-clip/match) max={float(np.nanmax(acc.values)):.1f} cells "
          f"(gate: old xarray-spatial max was 2451; must be > 50,000)")

    hand = rioxarray.open_rasterio(hand_path, masked=True)
    hand = _ensure_2d(hand).rio.clip_box(*aoi_bounds)
    hand = hand.rio.reproject_match(dem_target)
    # WhiteboxTools' elevation_above_stream writes float64; every other
    # layer in the predictor stack (incl. the other three routing outputs)
    # is float32 -- cast down for consistency (values are 0-123 m here, so
    # float32's ~7 significant digits lose nothing that matters).
    hand = hand.astype("float32")
    hand.name = "hand"

    slope_rad = _slope_rad(dem_path)
    cell_area = 100.0  # 10 m x 10 m cells

    with np.errstate(divide="ignore", invalid="ignore"):
        tan_slope = np.tan(slope_rad.values)
        tan_slope_safe = np.clip(tan_slope, 1e-4, None)
        twi_arr = np.log((acc.values * cell_area) / tan_slope_safe).astype("float32")
        spi_arr = (acc.values * cell_area * tan_slope).astype("float32")

    # Explicitly mask twi/spi to NaN wherever slope is NaN (the undefined-
    # slope edge cells xrspatial.slope() leaves at the grid boundary -- see
    # _slope_rad). This is already true mathematically (log/multiplication
    # of a NaN tan-slope stays NaN), but pinning it down here -- on the
    # plain ndarrays, matching this function's existing style, rather than
    # via a DataArray-level `.where()` against a separately-sourced slope
    # array whose coordinates could carry the same tiny reproject_match
    # floating-point drift `_save()` itself guards against -- guarantees
    # twi/spi's NaN-mask matches slope's exactly (nodata consistency across
    # the predictor stack: valid_mask() NaN-drops presence-point samples, so a
    # non-NaN cell here where slope is NaN would silently survive that
    # drop instead of being excluded like every other layer's edge cells).
    slope_nan_mask = np.isnan(slope_rad.values)
    twi_arr[slope_nan_mask] = np.nan
    spi_arr[slope_nan_mask] = np.nan

    twi = acc.copy(data=twi_arr)
    twi.name = "twi"
    spi = acc.copy(data=spi_arr)
    spi.name = "spi"

    return {
        "flow_accumulation": _save(acc, "flow_accumulation", dem_target, out_dir),
        "twi": _save(twi, "twi", dem_target, out_dir),
        "spi": _save(spi, "spi", dem_target, out_dir),
        "hand": _save(hand, "hand", dem_target, out_dir),
    }


def gfi(acc, slope_rad, cell_area=100.0):
    """Geomorphic Flood Index (Samela et al. 2017): ``ln(contributing area
    / tan(slope))``, on the FIXED (WhiteboxTools) accumulation -- the same
    formula as ``twi`` above, exposed standalone so later stages (e.g. the
    external-GFI comparison) can recompute it from any accumulation/slope
    pair without re-running the whole routing chain. Works on either plain
    ndarrays or xarray DataArrays (``np.tan``/``.clip``/``np.log`` all
    dispatch the same way for both).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_slope_safe = np.tan(slope_rad).clip(1e-4)
        return np.log((acc * cell_area) / tan_slope_safe)


# --------------------------------------------------------------------------
# Predictor QA -- alignment audit, HWM-point validity gate, and honest
# layer-count reconciliation. Ports notebook 02b_predictor_qa's alignment
# audit (SS1), HWM-point extraction (SS3), and "presence points excluded by
# predictor NaN" accounting (SS4) into live, re-runnable pipeline gates
# instead of one-off notebook cells that only ever printed a report. SS2
# (the visual map grid) is out of scope here -- the notebook itself frames
# it as "a verification pass, not a pipeline stage".
# --------------------------------------------------------------------------

# The 4 WhiteboxTools flow-routing outputs `flow_derivatives()` writes
# -- named here as a module constant (mirrors the local `names`
# tuple inside `flow_derivatives()` itself) purely so QA can enumerate the
# full raw stack without importing a name list out of a function body.
FLOW_DERIVATIVE_LAYERS = ("flow_accumulation", "twi", "spi", "hand")

# Every raw predictor layer this canonical pipeline actually has on disk --
# EXPECTED_NON_CENSUS (16) + CENSUS_LAYERS (2) + FLOW_DERIVATIVE_LAYERS (4)
# = 22. This is 2 FEWER than the original notebook's own "24 raw
# layers": the notebook's 24 included 2 gridMET precipitation layers
# (precip_annual_mean, precip_rx1day) that this canonical rebuild DROPPED
# entirely at acquisition (module docstring, deviation (a)) rather than at
# a later screening stage -- they were never written to
# data/processed/predictors/ at all, so they are simply absent from
# ALL_RAW_LAYERS, not filtered out of it.
ALL_RAW_LAYERS = EXPECTED_NON_CENSUS + CENSUS_LAYERS + FLOW_DERIVATIVE_LAYERS

# Layers the multicollinearity screen is known, from the source notebook's
# own 03_multicollinearity screen, to drop: tri/tpi correlate too highly
# with slope/curvature/dem. Named here as an asserted fact of the design,
# NOT computed by a live screen (pipeline/screen.py runs that) --
# qa_report()'s "screened"/"modeled" counts below are a stated forward
# reconciliation, not a recomputation.
PLANNED_MULTICOLLINEARITY_DROPS = ("tri", "tpi")

# The one additional layer dropped after screening, before the final
# model: some ACS block groups in the AOI have no median-income estimate
# at all (see module docstring, change (c) / acquire_census()), which is a
# structural data gap rather than an edge/nodata artifact like every other
# layer's NaNs -- accounted for separately/explicitly rather than folded
# into the same blanket NaN gate as the rest of the stack.
MODELING_DROPS = ("median_income",)

# The layers valid_mask() requires to be non-NaN: every raw layer EXCEPT
# median_income. This is NOT the eventual post-multicollinearity modeling
# set (that would additionally drop tri/tpi -- 19 layers, see
# qa_report()'s "modeled" count) -- this gate runs before the screen, so it
# intentionally still includes tri/tpi (whose edge-only NaN never actually
# lands on a presence point) and every other not-yet-screened layer.
MODELING_LAYERS = tuple(n for n in ALL_RAW_LAYERS if n not in MODELING_DROPS)

# Log-transform adoption: an ad-hoc experiment confirmed
# the raw-stack AICc winner (beta=3, LQHP, 27 hinges) was a SCALING
# ARTIFACT, not real ecological nonlinearity -- ``flow_accumulation`` spans
# 1 -> ~5.5M cells and ``spi`` spans 0 -> ~9.5e7, each 6+ orders of
# magnitude, so no single linear coefficient (in a MaxEnt feature, or in a
# Pearson correlation) can fit them, and MaxEnt/the screen compensate with
# hinges/inflated-looking correlation elsewhere instead. Log-scaling just
# these two layers let a plain beta=0.5/LQ model win outright (0 hinges,
# AICc 3782 <= the raw winner's 3787). ``MODELING_LOG_TRANSFORMS`` and
# ``apply_modeling_transform`` below are the single source of truth for
# this transform -- ``pipeline.screen`` (Pearson/VIF continuous matrix) and
# ``pipeline.maxent`` (``export_layers()``'s ``.asc`` export) both call
# THIS function rather than each re-implementing the log math, so the two
# consumers can never silently diverge on which layers get transformed or
# how.
MODELING_LOG_TRANSFORMS = {"flow_accumulation": "log10", "spi": "log1p"}


def apply_modeling_transform(name, arr):
    """Log-scale the two power-law-skewed hydrology predictors for
    MODELING (the ``pipeline.screen`` Pearson/VIF continuous matrix and the
    ``pipeline.maxent`` ``.asc`` export) -- ``flow_accumulation`` ->
    ``log10`` (this stack's min is 1, so every finite cell is a safe,
    strictly-positive input), ``spi`` -> ``log1p`` (has exact-zero cells at
    hydrologically flat pixels -- plain ``log`` would hit ``log(0) =
    -inf``; ``log1p(0) = 0`` is well-defined). Every layer NOT a key in
    ``MODELING_LOG_TRANSFORMS`` (i.e. every layer except these two) is
    returned unchanged (as float64) -- identity, not a no-op error.

    Applied only to FINITE cells; any NaN in ``arr`` (this module's nodata
    convention -- see ``_save``'s docstring) is left as NaN, never passed
    through ``log10``/``log1p``, so nodata cells stay nodata rather than
    silently becoming a second, transform-derived NaN (which numpy would
    produce anyway for ``log(nan)`` -- explicit here instead of incidental).

    This is a LOAD-TIME modeling transform ONLY -- it never writes to disk.
    The raw ``.tif`` files under ``data/processed/predictors/`` (and
    ``data/processed/predictors_screened/``) stay physical/untouched:
    ``gfi()`` and any provenance/reporting code need the real flow-
    accumulation cell counts, not a log-scaled proxy. Every caller that
    needs the model-space value applies this function to an in-memory
    array it already loaded, rather than reading a persisted transformed
    copy from anywhere on disk.

    Parameters
    ----------
    name : predictor layer name (e.g. ``"flow_accumulation"``, ``"slope"``)
        -- looked up in ``MODELING_LOG_TRANSFORMS``; any name not a key
        there passes ``arr`` through unchanged.
    arr : array-like (any shape ``np.asarray`` accepts) of the layer's
        in-memory values, NaN nodata already decoded -- matching every
        other function in this module's convention.

    Returns
    -------
    ndarray, same shape as ``arr``, dtype float64: the transformed values
    at finite cells with NaN nodata cells preserved, or ``arr`` cast to
    float64 unchanged if ``name`` is not a ``MODELING_LOG_TRANSFORMS`` key.
    """
    arr = np.asarray(arr, dtype="float64")
    transform = MODELING_LOG_TRANSFORMS.get(name)
    if transform is None:
        return arr

    out = arr.copy()
    finite = np.isfinite(arr)
    if transform == "log10":
        out[finite] = np.log10(arr[finite])
    elif transform == "log1p":
        out[finite] = np.log1p(arr[finite])
    else:
        raise ValueError(
            f"[predictors] apply_modeling_transform: unknown transform "
            f"{transform!r} configured for layer {name!r}"
        )
    return out


def _load_all_raw_layers(out_dir=OUT_DIR, layer_names=ALL_RAW_LAYERS):
    """Load every canonical raw predictor layer (2-D, NaN nodata) from
    ``out_dir``, keyed by name, in ``layer_names`` order. Raises
    ``FileNotFoundError`` naming every missing layer if the stack is
    incomplete -- ``valid_mask()``/``qa_report()`` need the FULL stack to
    be honest about which layer causes which exclusion; silently skipping
    a missing layer would hide its NaNs as false "valid" pixels instead of
    the exclusion (or outright gap) they actually represent.
    """
    arrays = {}
    missing = []
    for name in layer_names:
        arr = _load_cached(name, out_dir)
        if arr is None:
            missing.append(name)
            continue
        arrays[name] = _ensure_2d(arr)
    if missing:
        raise FileNotFoundError(
            f"[predictors] QA: missing raw layer file(s) under {out_dir}: {missing}"
        )
    return arrays


def _alignment_audit(arrays: dict) -> pd.DataFrame:
    """Mirrors the original notebook 02b_predictor_qa, SS1: every layer must share CRS,
    grid shape, and affine transform, or a value at row/col (i, j) in one
    layer would not correspond to the same ground location in another --
    silently corrupting every per-point sample taken below. The notebook
    only ever PRINTED its mismatch count for a human to notice; this
    asserts (raises) on any mismatch instead, since a live pipeline gate
    must fail loudly rather than hand a misaligned stack to
    ``valid_mask()`` or downstream modeling.

    Returns the per-layer audit table (layer/crs/shape/resolution/
    transform/nan_pct) for ``qa_report()`` -- computed once by the caller
    and reused, not recomputed independently by every consumer.
    """
    rows = []
    for name, da in arrays.items():
        rows.append({
            "layer": name,
            "crs": str(da.rio.crs),
            "shape": da.rio.shape,
            "resolution": da.rio.resolution(),
            "transform": tuple(da.rio.transform())[:6],
            "nan_pct": float(np.isnan(da.values).mean()) * 100,
        })
    audit = pd.DataFrame(rows)
    ref_crs = audit.loc[0, "crs"]
    ref_shape = audit.loc[0, "shape"]
    ref_transform = audit.loc[0, "transform"]
    mismatched = [
        r.layer for r in audit.itertuples()
        if not (r.crs == ref_crs and r.shape == ref_shape and r.transform == ref_transform)
    ]
    assert not mismatched, (
        f"[predictors] QA alignment audit FAILED -- layer(s) misaligned vs reference "
        f"(crs={ref_crs}, shape={ref_shape}, transform={ref_transform}): {mismatched}"
    )
    return audit


def _sample_at_points(arrays: dict, xs: np.ndarray, ys: np.ndarray):
    """Sample every layer in ``arrays`` at point coordinates ``(xs, ys)``,
    nearest-pixel (``rasterio.transform.rowcol``, default ``op=floor`` --
    the pixel containing the point, same as the original notebook's
    own unqualified ``rowcol(transform, xs, ys)`` call), against the
    transform of the FIRST layer in ``arrays`` -- safe because
    ``_alignment_audit`` has already asserted every layer shares one
    transform/shape, so any one of them is an equally valid reference.

    Returns ``(values_df, in_bounds)``: ``values_df`` has one row per
    point and one column per layer name (NaN wherever the point falls
    outside the raster extent, an explicit value rather than a raised
    error, so a caller can still assemble a full-width table); ``in_bounds``
    is a boolean ndarray, True where the point's row/col fell inside the
    grid at all (kept separate from per-layer NaN so an out-of-extent point
    is distinguishable from an in-extent NoData pixel).
    """
    ref = next(iter(arrays.values()))
    transform = ref.rio.transform()
    shape = ref.rio.shape

    rows_idx, cols_idx = rowcol(transform, xs, ys)
    rows_idx = np.asarray(rows_idx)
    cols_idx = np.asarray(cols_idx)
    in_bounds = (rows_idx >= 0) & (rows_idx < shape[0]) & (cols_idx >= 0) & (cols_idx < shape[1])

    data = {}
    for name, da in arrays.items():
        vals = np.full(xs.shape[0], np.nan, dtype="float64")
        vals[in_bounds] = da.values[rows_idx[in_bounds], cols_idx[in_bounds]]
        data[name] = vals
    return pd.DataFrame(data), in_bounds


def _validity_from_samples(values_df: pd.DataFrame, in_bounds: np.ndarray, layers) -> np.ndarray:
    """The one shared validity formula ``valid_mask()`` and ``qa_report()``
    both use (over whichever ``layers`` subset each needs) -- True where a
    point is in-bounds AND non-NaN across every named layer. Factored out
    so the two callers can never quietly diverge in what "valid" means.
    """
    return in_bounds & ~values_df[list(layers)].isna().any(axis=1).to_numpy()


def _presence_points():
    """The canonical 286 HW+EOW presence points
    (``hwm.build_presence()``). Local import: ``pipeline.hwm`` already
    imports ``pipeline.config`` at module level, so importing ``hwm`` here
    at module level too would be safe today, but a local import costs
    nothing and keeps this module's only dependency on ``hwm`` confined to
    the two QA entry points that actually need presence-point data (every
    other function in this module -- the acquire_*/flow_derivatives family
    -- has no reason to import it at all).
    """
    from pipeline import hwm
    return hwm.build_presence()


def valid_mask(out_dir=OUT_DIR) -> pd.Series:
    """Boolean Series (length 286, ``RangeIndex`` 0..285, one entry per
    canonical HW+EOW presence point in ``hwm.build_presence()`` row order)
    -- True where that point samples a non-NaN value (nearest pixel) across
    EVERY layer in ``MODELING_LAYERS`` (all 22 raw predictors except
    ``median_income``). Mirrors the original notebook 02b_predictor_qa's SS3/SS4
    exactly, generalized from a one-off notebook report into a live,
    re-runnable gate that downstream modeling code can call directly.

    The empirical result of this call is 279, not the 278 of the original
    notebooks -- verified against the real on-disk stack, not
    hardcoded either way. See ``qa_report()['presence_points']`` for the
    full accounting.
    """
    presence = _presence_points()
    arrays = _load_all_raw_layers(out_dir)
    _alignment_audit(arrays)
    values_df, in_bounds = _sample_at_points(arrays, presence["x"].to_numpy(), presence["y"].to_numpy())
    valid = _validity_from_samples(values_df, in_bounds, MODELING_LAYERS)
    return pd.Series(valid, index=presence.index, name="valid")


# ---------------------------------------------------------------------------
# Access proxy -- distance to road (TIGER/Line). NOT a modeling predictor.
# ---------------------------------------------------------------------------
# The presence set is surveyed by
# a municipal crew, so proximity to roads is a proxy for where surveying is
# CHEAP, independent of where flooding is likely. Without it, no analysis in
# this project can separate "floods near outfalls" from "crews survey near
# outfalls" (see also sensitivity.matched_background_refit).
#
# This layer is a NEGATIVE CONTROL and a confounder for conditional-importance
# tests. It is deliberately NOT added to maxent.modeling_layers(): adding an
# access proxy as a predictor would bake the sampling process into the
# susceptibility surface itself.
#
# CRITICAL: rasterised onto the EXISTING canonical grid read from
# the reference raster. Nothing here re-acquires or re-derives the grid.
ACCESS_DIR = config.DATA / "access"     # NOT predictors/ -- see below
TIGER_ROADS_URL = ("https://www2.census.gov/geo/tiger/TIGER2022/ROADS/"
                   "tl_2022_{fips}_roads.zip")
ROAD_COUNTIES = ("20161", "20149")          # Riley, Pottawatomie -- matches equity.COUNTIES
ACCESS_LAYER = "distance_to_road"


def acquire_distance_to_road(out_dir=None, counties=ROAD_COUNTIES, force=False):
    """Euclidean distance (m) to the nearest TIGER/Line road centreline.

    Returns the path to ``distance_to_road.tif``, written on the canonical grid.
    Cached: returns immediately if the file already exists and ``force`` is False.
    """
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt
    from rasterio.features import rasterize as _rasterize

    # MUST default to ACCESS_DIR, never OUT_DIR. screen._load_stack() globs
    # "*.tif" in predictors/, so a raster written there is silently picked up as
    # an 18th MODELING predictor -- which is exactly what happened on the first
    # attempt (2026-08-13) and was caught only by asserting len(modeling_layers())
    # == 17. An access proxy must never enter the susceptibility model: it would
    # bake the sampling process into the prediction.
    out_dir = Path(out_dir) if out_dir is not None else ACCESS_DIR
    path = out_dir / f"{ACCESS_LAYER}.tif"
    if path.exists() and not force:
        print(f"[predictors] cache hit: {ACCESS_LAYER:<24} <- {path}")
        return path

    ref = _load_cached("dem", OUT_DIR)   # grid comes from predictors/, output does not
    if ref is None:
        raise FileNotFoundError(
            "[predictors] acquire_distance_to_road: dem.tif absent; the canonical "
            "grid is read from it and must NOT be re-derived")

    frames = []
    for fips in counties:
        url = TIGER_ROADS_URL.format(fips=fips)
        print(f"[predictors] roads: reading {url}")
        frames.append(gpd.read_file(url))
    roads = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    roads = roads.to_crs(ref.rio.crs)
    print(f"[predictors] roads: {len(roads):,} centreline features")

    transform = ref.rio.transform()
    shape = (ref.sizes["y"], ref.sizes["x"])
    burned = _rasterize(((g, 1) for g in roads.geometry if g is not None),
                        out_shape=shape, transform=transform, fill=0, dtype="uint8")
    res = abs(transform.a)
    dist = distance_transform_edt(burned == 0, sampling=res).astype("float32")
    dist[~np.isfinite(ref.values)] = np.nan

    da = ref.copy(data=dist)
    da.name = ACCESS_LAYER
    da = da.rio.write_crs(ref.rio.crs)
    out_dir.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path)
    finite = dist[np.isfinite(dist)]
    print(f"[predictors] {ACCESS_LAYER}: min={finite.min():.1f} m "
          f"median={np.median(finite):.1f} m max={finite.max():.1f} m -> {path}")
    return path


def qa_report(out_dir=OUT_DIR) -> dict:
    """The live predictor QA gate, as one dict: the alignment audit (all 22
    raw layers share CRS/shape/transform -- asserted, not merely reported),
    the honest raw/screened/modeled layer-count reconciliation against the
    source notebook's own 24/22/21 accounting, and the per-layer HWM-point
    NaN-exclusion breakdown (which layer(s) cause each of the 286 presence
    points to be excluded, and how many `median_income` alone recovers).

    Reuses one load + one point-sample of the 22-layer stack internally
    (does not call ``valid_mask()`` a second time) so the alignment audit,
    the layer-count reconciliation, and the exclusion-cause breakdown all
    come from the exact same in-memory sample -- never three independently
    -reloaded, potentially-drifting views of the same 286 points.
    """
    presence = _presence_points()
    arrays = _load_all_raw_layers(out_dir)
    audit = _alignment_audit(arrays)
    values_df, in_bounds = _sample_at_points(arrays, presence["x"].to_numpy(), presence["y"].to_numpy())

    valid_all22 = _validity_from_samples(values_df, in_bounds, ALL_RAW_LAYERS)
    valid_modeling = _validity_from_samples(values_df, in_bounds, MODELING_LAYERS)

    # Per-layer exclusion causes over the FULL 22-layer stack (matching the
    # notebook's own "24 raw layers" framing), mirroring the notebook's SS4
    # Counter breakdown -- computed here (not restricted to MODELING_LAYERS)
    # so median_income's own exclusive contribution is visible even though
    # valid_mask() itself doesn't penalize a point for it.
    nan_mask_df = values_df[list(ALL_RAW_LAYERS)].isna()
    layer_exclusion_counts = {name: 0 for name in ALL_RAW_LAYERS}
    n_out_of_bounds = 0
    for i in range(len(presence)):
        if not in_bounds[i]:
            n_out_of_bounds += 1
            continue
        for name in nan_mask_df.columns[nan_mask_df.iloc[i]]:
            layer_exclusion_counts[name] += 1
    layer_exclusion_counts = {k: v for k, v in layer_exclusion_counts.items() if v > 0}
    if n_out_of_bounds:
        layer_exclusion_counts["out_of_bounds"] = n_out_of_bounds

    n_raw = len(ALL_RAW_LAYERS)
    n_screened = n_raw - len(PLANNED_MULTICOLLINEARITY_DROPS)
    n_modeled = n_screened - len(MODELING_DROPS)

    excluded_all22 = int((~valid_all22).sum())
    excluded_modeling = int((~valid_modeling).sum())

    return {
        "layers": {
            "all_raw": list(ALL_RAW_LAYERS),
            "modeling": list(MODELING_LAYERS),
        },
        "alignment": {
            "reference_crs": str(audit.loc[0, "crs"]),
            "reference_shape": tuple(audit.loc[0, "shape"]),
            "reference_transform": tuple(audit.loc[0, "transform"]),
            "n_layers_checked": len(audit),
            "n_mismatched": 0,  # _alignment_audit() already asserted this -- reaching here means 0
            "per_layer_nan_pct": dict(zip(audit["layer"], audit["nan_pct"])),
        },
        "layer_counts": {
            "raw": n_raw,
            "screened": n_screened,
            "modeled": n_modeled,
            "notebook_raw": 24,
            "notebook_screened": 22,
            "notebook_modeled": 21,
            "precip_layers_dropped_at_acquisition": ("precip_annual_mean", "precip_rx1day"),
            "planned_multicollinearity_drops": PLANNED_MULTICOLLINEARITY_DROPS,
            "modeling_drops": MODELING_DROPS,
            "note": (
                "Canonical raw/screened/modeled counts (22/20/19) are each exactly 2 "
                "below the notebook's own 24/22/21: the 2 gridMET precipitation layers "
                "the notebook carried all the way through to its 21-predictor model are, "
                "in this canonical rebuild, dropped entirely at acquisition (never "
                "written to data/processed/predictors/) rather than surviving to the "
                "modeling stage -- see predictors.py module docstring, deviation (a). "
                "'screened'/'modeled' here are a STATED forward reconciliation of "
                "already-decided drops (tri/tpi at multicollinearity, per the notebook's "
                "own 03_multicollinearity screen; median_income before the final model), "
                "not a live recomputation (pipeline/screen.py runs the live screen)."
            ),
        },
        "presence_points": {
            "total": len(presence),
            "hw": int((presence["type"] == "HW").sum()),
            "eow": int((presence["type"] == "EOW").sum()),
            "out_of_bounds": n_out_of_bounds,
            "excluded_any_of_22_raw_layers": excluded_all22,
            "valid_all_22_raw_layers": len(presence) - excluded_all22,
            "excluded_modeling_layers_21_excl_median_income": excluded_modeling,
            "valid_modeling_layers_21_excl_median_income": len(presence) - excluded_modeling,
            "recovered_by_dropping_median_income": excluded_all22 - excluded_modeling,
            "expected_modeling_valid_per_task_brief": 278,
            "matches_expected_278": (len(presence) - excluded_modeling) == 278,
        },
        "exclusion_causes": layer_exclusion_counts,
    }

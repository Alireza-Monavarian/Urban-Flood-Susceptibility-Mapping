"""HWM (high-water-mark) survey parsing, seeded thinning, and canonical
presence-point assembly.

Ports the original notebook ``01_hwm_processing.ipynb`` into an importable
pipeline stage. Parses every
City of Manhattan flood-event survey file (several ASCII dialects plus Carlson
SurvCE/SurvPC ``.RW5`` raw data and two point shapefiles), reprojects
everything from KS State Plane North (EPSG:3419, US survey feet) to the
modeling CRS (NAD83 / UTM Zone 14N, EPSG:26914), and applies deterministic
30 m spatial thinning (Luan et al. 2025 protocol) separately for high-water-mark
(HW) and edge-of-water (EOW) points.

Changes vs. the notebook source (only these four):
  (a) thinning takes a REQUIRED ``seed`` (``thin_30m(df, seed)``) instead of a
      silently-defaulted one, so every call site is explicit;
  (b) EOW points are written into ``canonical_presence.csv`` alongside HW
      (the notebook only ever wrote a thinned-HW CSV; EOW only got a ``.gpkg``);
  (c) shapefile-derived HW rows carry ``provenance="file_level_HW"`` (the whole
      file was surveyed/exported as HW with no per-point description to filter
      on) vs. ``"point_level_HW"``/``"point_level_EOW"`` for text/RW5 rows that
      passed a per-point description filter;
  (d) ``raw_counts()`` exposes the true pre-thinning totals, computed from the
      actual parse of the manifest below -- never hardcoded.

Out of scope for this module (not part of the interfaces below): the
``.gpkg`` exports and the folium verification map the notebook also produced.
Those are notebook/reporting concerns, not inputs any downstream pipeline
stage consumes; a later reporting step can regenerate them from
``build_presence()`` if wanted.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from pipeline import config

HWM_DIR = config.REPO / "data" / "raw" / "hwm"

# KS State Plane North 1983 (NAD83(NSRS2007)), US Survey Feet -- confirmed by
# the 2011 and 2018 shapefiles' own .prj files.
SRC_CRS = "EPSG:3419"
# NAD83 / UTM Zone 14N -- modeling CRS.
TGT_CRS = "EPSG:26914"

# Generous sanity bbox in TGT_CRS meters, used to catch gross CRS/column-order
# errors (e.g. a wrong UTM zone) -- not a tight AOI clip. Manhattan KS itself
# sits well inside this (easting ~680,000-710,000 m, northing ~4,330,000-
# 4,360,000 m).
BBOX = dict(minx=640_000, maxx=760_000, miny=4_300_000, maxy=4_400_000)

KEEP_COLS = ["pt_id", "event", "type", "description", "elevation_ft",
             "source_file", "provenance", "geometry"]


# --------------------------------------------------------------------------
# Per-file-format parsers -- one per survey-file dialect identified during
# inventory. Ported verbatim from the original notebook.
# --------------------------------------------------------------------------

def _row(parts, desc, ptype, event, fname):
    return {
        "pt_id":        parts[0].strip(),
        "northing_ft":  float(parts[1]),
        "easting_ft":   float(parts[2]),
        "elevation_ft": float(parts[3]),
        "description":  desc,
        "type":         ptype,
        "event":        event,
        "source_file":  fname,
    }


def parse_A(filepath, event):
    """Pattern A -- standard format, optional header.
    Keep description in {HW, B HW, WH, WH*}.
    """
    rows = []
    fname = Path(filepath).name
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("PT") or line.startswith("["):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            desc = parts[4].strip()
            desc_up = desc.upper()
            if desc_up in ("HW", "B HW") or desc_up.startswith("WH"):
                try:
                    rows.append(_row(parts, desc, "HW", event, fname))
                except ValueError:
                    pass
    return rows


def parse_B(filepath, event):
    """Pattern B -- GPS extended format (Leica).
    Keep HW / B HW with STATUS:FIXED only; drop FLOAT.
    """
    rows = []
    fname = Path(filepath).name
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("PT"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            desc = parts[4].strip().upper()
            if desc not in ("HW", "B HW"):
                continue
            status_field = next((p for p in parts if p.strip().upper().startswith("STATUS:")), "")
            if "FIXED" not in status_field.upper():
                continue
            try:
                rows.append(_row(parts, parts[4].strip(), "HW", event, fname))
            except ValueError:
                pass
    return rows


def parse_D(filepath, event):
    """Pattern D -- header-less with // comment lines.
    Keep description == HW.
    """
    rows = []
    fname = Path(filepath).name
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            desc = parts[4].strip()
            if desc.upper() == "HW":
                try:
                    rows.append(_row(parts, desc, "HW", event, fname))
                except ValueError:
                    pass
    return rows


def parse_H(filepath, event):
    """Pattern H -- compound 'EL / HIGH WATER' descriptions.
    Keep any row whose description contains HIGH WATER.
    """
    rows = []
    fname = Path(filepath).name
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            desc = parts[4].strip()
            if "HIGH WATER" in desc.upper():
                try:
                    rows.append(_row(parts, desc, "HW", event, fname))
                except ValueError:
                    pass
    return rows


def parse_EOW(filepath, event):
    """Patterns E/G -- edge-of-water points.
    Keep EOW and EOA, flagged as type='EOW' for separate sensitivity analysis.
    """
    rows = []
    fname = Path(filepath).name
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            desc = parts[4].strip()
            base = desc.split("/")[0].strip().upper()
            if base in ("EOW", "EOA"):
                try:
                    rows.append(_row(parts, desc, "EOW", event, fname))
                except ValueError:
                    pass
    return rows


def parse_RW5(filepath, event):
    """Pattern RW5 -- Carlson SurvCE/SurvPC raw data files (2015, 2016, 2021, 2024 events).
    Coordinates (KS State Plane North NAD83, feet) and quality flags live in
    commented '--GS,PN...' lines and the STATUS line that follows within the
    same point block. Keep HW/B HW/WH*/HIGH WATER/HIGHWATER (type='HW') and
    EOW/EOA (type='EOW') with STATUS:FIXED (or FIXED+).
    """
    rows = []
    fname = Path(filepath).name
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("--GS,PN"):
            continue
        parts = line.split(",")
        pt_id = parts[1].replace("PN", "").strip()
        desc = parts[-1].lstrip("-").strip()
        desc_up = desc.upper()

        if desc_up in ("HW", "B HW") or desc_up.startswith("WH") or "HIGH WATER" in desc_up or desc_up == "HIGHWATER":
            ptype = "HW"
        elif desc_up.split("/")[0].strip() in ("EOW", "EOA"):
            ptype = "EOW"
        else:
            continue

        try:
            northing = float(next(p for p in parts if p.startswith("N ")).split()[1])
            easting  = float(next(p for p in parts if p.startswith("E ")).split()[1])
            elev     = float(next(p for p in parts if p.startswith("EL")).replace("EL", ""))
        except (StopIteration, ValueError):
            continue

        # STATUS line follows within the same point block (1-4 lines later)
        status_ok = False
        for nxt in lines[i + 1:i + 6]:
            nxt = nxt.strip()
            if nxt.startswith("GPS,") or nxt.startswith("--GS,PN") or nxt.startswith("BP,"):
                break
            if "STATUS:" in nxt:
                status_ok = "FIXED" in nxt
                break
        if not status_ok:
            continue

        rows.append({
            "pt_id":        pt_id,
            "northing_ft":  northing,
            "easting_ft":   easting,
            "elevation_ft": elev,
            "description":  desc,
            "type":         ptype,
            "event":        event,
            "source_file":  fname,
        })
    return rows


PARSERS = {"A": parse_A, "B": parse_B, "D": parse_D, "H": parse_H, "EOW": parse_EOW, "RW5": parse_RW5}


def parse_event(path, fmt) -> pd.DataFrame:
    """Dispatch to the per-format parser selected by ``fmt`` (a key of
    ``PARSERS``) and return the parsed rows as a DataFrame.

    ``event`` is inferred from the immediate parent directory name, since raw
    HWM files live at ``data/raw/hwm/<event>/<event>/<file>`` -- callers don't
    need to pass it separately. ``build_presence()``'s manifest loop re-tags
    the authoritative event from the manifest tuple regardless, so this
    inference only matters when calling ``parse_event`` standalone.
    """
    path = Path(path)
    event = path.parent.name
    rows = PARSERS[fmt](path, event)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Event manifest -- text/RW5 files, one entry per file used. The comments
# below record why each skipped file is skipped.
# --------------------------------------------------------------------------

MANIFEST = [
    ("2007_5_24", "2007_5_24/2007_5_24/flood HW PTS 24May2007.txt",           "A"),
    ("2010_6_10", "2010_6_10/2010_6_10/FLOOD EVENT LEICA.txt",                 "B"),
    ("2010_6_10", "2010_6_10/2010_6_10/HIGH WATER 2010.txt",                   "B"),
    # FLOOD HW PTS 16June2010.txt / Wildcat Creek event June 2010.txt are combined
    # re-exports of LEICA (1000-series) + HIGH WATER 2010 (10000-series) -- skipped
    # to avoid duplicates now that both source surveys are read directly above.
    ("2014_6_9",  "2014_6_9/2014_6_9/Flood Hw Pts 9June2014.txt",              "A"),
    ("2015_5_4",  "2015_5_4/2015_5_4/MAY4-2015 HIGH WATER.RW5",                "RW5"),
    ("2016_4_24", "2016_4_24/2016_4_24/HIGHWATER042416.RW5",                   "RW5"),
    ("2018_9_3",  "2018_9_3/2018_9_3/FLOOD 932018 Combo.txt",                  "A"),
    ("2018_9_3",  "2018_9_3/2018_9_3/2018 HIGH WATER.txt",                     "H"),
    # FLOOD HW PTS 3Sept2018.txt covered by Combo except for 2 extra HW points
    # (1019, 4001) -- added separately below. FLOOD 932018.txt covered by Combo.
    ("2019_5_18", "2019_5_18/2019_5_18/Flood HW Pts 8May2019.txt",             "A"),
    ("2019_8_30", "2019_8_30/2019_8_30/08_29_2019 STORM.txt",                  "EOW"),
    ("2021_7_15", "2021_7_15/2021_7_15/07152021 FLOOD EVENT.rw5",              "RW5"),
    ("2024_5_19", "2024_5_19/2024_5_19/HIGHWATER 05192024.rw5",                "RW5"),
]

# Two HW points (1019, 4001) present in FLOOD HW PTS 3Sept2018.txt but absent
# from FLOOD 932018 Combo.txt.
EXTRA_2018_IDS = {"1019", "4001"}
EXTRA_2018_PATH = "2018_9_3/2018_9_3/FLOOD HW PTS 3Sept2018.txt"


def _parse_all_text_sources() -> pd.DataFrame:
    """Parse every MANIFEST entry plus the special 2018 extra points, tag each
    row with its event and a point_level_HW/point_level_EOW provenance flag,
    and concatenate. Mirrors cell 3 of the original notebook.
    """
    parts = []
    for event, rel_path, fmt in MANIFEST:
        fpath = HWM_DIR / rel_path
        part = parse_event(fpath, fmt)
        print(f"{event}  {fpath.name:<45} [{fmt}]  -> {len(part)} pts")
        if len(part):
            part["event"] = event
            parts.append(part)

    extra_path = HWM_DIR / EXTRA_2018_PATH
    extra = parse_event(extra_path, "A")
    extra = extra[extra["pt_id"].isin(EXTRA_2018_IDS)].copy()
    print(f"2018_9_3  {extra_path.name:<45} [A, extra pts]  -> {len(extra)} pts")
    if len(extra):
        extra["event"] = "2018_9_3"
        parts.append(extra)

    df_txt = pd.concat(parts, ignore_index=True, sort=False)
    df_txt["provenance"] = df_txt["type"].map({"HW": "point_level_HW", "EOW": "point_level_EOW"})
    return df_txt


# --------------------------------------------------------------------------
# Shapefile ingestion -- 2011 and 2018 events have georeferenced shapefiles
# with no per-point description field, so the whole file is treated as HW
# (provenance="file_level_HW").
# --------------------------------------------------------------------------

SHAPEFILE_MANIFEST = [
    ("2011_6_2", "2011_6_2/2011_6_2/Wildcreek 6 2 2011 flood_11_UNKNOWN.shp", "2011 Wildcreek",   SRC_CRS),
    ("2018_9_3", "2018_9_3/2018_9_3/HW from pts.shp",                        "2018 HW from pts", None),
    ("2018_9_3", "2018_9_3/2018_9_3/huntersHW.shp",                          "2018 huntersHW",   None),
]


def shp_to_rows(path, event, label, assumed_crs=None):
    """Convert a point shapefile to the standard row format.
    Only uses Point geometries -- polygon/line layers are skipped with a warning.
    If CRS is missing, assumed_crs is used (must be provided explicitly).
    Rows are flagged provenance="file_level_HW": these shapefiles carry no
    per-point description field, so (unlike the text/RW5 parsers) there is no
    per-point HW/EOW filter to apply -- the whole file is HW by survey design.
    """
    gdf = gpd.read_file(path)
    print(f"  [{label}] CRS={gdf.crs}  geom_types={gdf.geometry.geom_type.unique().tolist()}  rows={len(gdf)}")
    print(f"  [{label}] columns={list(gdf.columns)}")

    # Keep only Point geometries
    point_mask = gdf.geometry.geom_type == "Point"
    if not point_mask.any():
        print(f"  [{label}] no Point geometries -- skipping (likely a polygon boundary layer)")
        return []
    gdf = gdf[point_mask].copy()

    # Assign CRS if missing
    if gdf.crs is None:
        if assumed_crs is None:
            print(f"  [{label}] no CRS and no assumed_crs -- skipping")
            return []
        print(f"  [{label}] no CRS -- assigning {assumed_crs}")
        gdf = gdf.set_crs(assumed_crs)

    # Reproject to UTM 14N
    gdf = gdf.to_crs(TGT_CRS)

    # Bbox sanity check -- drop points that land outside Manhattan KS area
    ok = ((gdf.geometry.x >= BBOX["minx"]) & (gdf.geometry.x <= BBOX["maxx"]) &
          (gdf.geometry.y >= BBOX["miny"]) & (gdf.geometry.y <= BBOX["maxy"]))
    n_bad = (~ok).sum()
    if n_bad > 0:
        print(f"  [{label}] {n_bad} points outside Manhattan KS bbox -- possible CRS mismatch, dropping")
    gdf = gdf[ok].copy()

    rows = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        rows.append({
            "pt_id":        str(i),
            "northing_ft":  np.nan,
            "easting_ft":   np.nan,
            "elevation_ft": np.nan,
            "description":  "HW",
            "type":         "HW",
            "event":        event,
            "source_file":  Path(path).name,
            "provenance":   "file_level_HW",
            "geometry":     geom,
        })
    print(f"  [{label}] -> {len(rows)} usable points")
    return rows


def _parse_all_shapefile_sources() -> pd.DataFrame:
    """Parse the 2011 + 2018 point shapefiles into rows. Mirrors cells 4-5 of the original notebook."""
    rows = []
    for event, rel_path, label, assumed_crs in SHAPEFILE_MANIFEST:
        rows.extend(shp_to_rows(HWM_DIR / rel_path, event, label, assumed_crs=assumed_crs))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Combine, reproject, sanity-check
# --------------------------------------------------------------------------

def _txt_to_gdf(df_txt: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build point geometries from northing_ft/easting_ft (KS State Plane
    North, US survey feet) and reproject to the modeling CRS.
    """
    valid = df_txt.dropna(subset=["easting_ft", "northing_ft"]).copy()
    geom = [Point(e, n) for e, n in zip(valid["easting_ft"], valid["northing_ft"])]
    gdf_src = gpd.GeoDataFrame(valid, geometry=geom, crs=SRC_CRS)
    gdf = gdf_src.to_crs(TGT_CRS)
    return gdf[[c for c in KEEP_COLS if c in gdf.columns]]


def _raw_combined() -> gpd.GeoDataFrame:
    """All parsed HW+EOW points, reprojected to TGT_CRS, pre-thinning.

    Shared by raw_counts() and build_presence() so the two can never drift
    apart -- both tally off exactly this frame.
    """
    df_txt = _parse_all_text_sources()
    df_shp = _parse_all_shapefile_sources()

    gdf_txt = _txt_to_gdf(df_txt)
    if len(df_shp) > 0:
        gdf_shp = gpd.GeoDataFrame(df_shp, geometry="geometry", crs=TGT_CRS)
        gdf_shp = gdf_shp[[c for c in KEEP_COLS if c in gdf_shp.columns]]
        gdf_all = pd.concat([gdf_txt, gdf_shp], ignore_index=True)
    else:
        gdf_all = gdf_txt
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs=TGT_CRS)

    # Ensure only Point geometries remain before accessing .x / .y
    non_point = gdf_all[gdf_all.geometry.geom_type != "Point"]
    if len(non_point) > 0:
        print(f"dropping {len(non_point)} non-Point geometries: "
              f"{non_point.geometry.geom_type.value_counts().to_dict()}")
        gdf_all = gdf_all[gdf_all.geometry.geom_type == "Point"].copy()

    # AOI sanity check -- points outside this bbox indicate a CRS or
    # column-order error upstream; fail loudly rather than silently include
    # bad points in the presence set.
    outliers = gdf_all[
        (gdf_all.geometry.x < BBOX["minx"]) | (gdf_all.geometry.x > BBOX["maxx"]) |
        (gdf_all.geometry.y < BBOX["miny"]) | (gdf_all.geometry.y > BBOX["maxy"])
    ]
    assert len(outliers) == 0, (
        f"{len(outliers)} HWM points fall outside the Manhattan KS sanity bbox "
        f"{BBOX} -- likely CRS/column-order error: "
        f"{outliers[['event', 'source_file']].to_dict('records')}"
    )

    return gdf_all


# --------------------------------------------------------------------------
# Spatial thinning -- 30 m grid (Luan et al. 2025 protocol)
# --------------------------------------------------------------------------

def thin_30m(df: gpd.GeoDataFrame, seed: int, cell_size_m: float = 30) -> gpd.GeoDataFrame:
    """Thin a point GeoDataFrame to at most one point per `cell_size_m` grid
    cell, retaining one point per cell at random.

    `seed` is REQUIRED (no default) so every call site is explicit about
    reproducibility. A fresh `np.random.default_rng(seed)` is created inside
    this call, so the result is fully determined by (df, seed) regardless of
    prior calls or global RNG state.
    """
    rng = np.random.default_rng(seed)
    xs = df.geometry.x.values
    ys = df.geometry.y.values
    cell_x = np.floor(xs / cell_size_m).astype(int)
    cell_y = np.floor(ys / cell_size_m).astype(int)
    cell_id = list(zip(cell_x, cell_y))

    df = df.copy()
    df["_cell"] = cell_id

    # Shuffle then drop duplicates -- equivalent to random retention per cell
    idx = np.arange(len(df))
    rng.shuffle(idx)
    df_shuffled = df.iloc[idx].reset_index(drop=True)
    df_thinned = df_shuffled.drop_duplicates(subset="_cell").drop(columns="_cell")
    return df_thinned


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def raw_counts() -> dict:
    """Pre-thinning HW/EOW totals, computed from the actual parse of the
    manifest (never hardcoded).
    """
    gdf_all = _raw_combined()
    hw_raw = int((gdf_all["type"] == "HW").sum())
    eow_raw = int((gdf_all["type"] == "EOW").sum())
    return {"HW_raw": hw_raw, "EOW_raw": eow_raw, "total_raw": hw_raw + eow_raw}


def build_presence() -> gpd.GeoDataFrame:
    """Parse every manifest entry, combine HW+EOW (event-tagged), thin HW and
    EOW SEPARATELY at 30 m with config.GLOBAL_SEED, write
    data/processed/canonical_presence.csv (HW+EOW both included), and return
    the thinned, combined GeoDataFrame.
    """
    gdf_all = _raw_combined()

    gdf_hw = gdf_all[gdf_all["type"] == "HW"].copy()
    gdf_eow = gdf_all[gdf_all["type"] == "EOW"].copy()

    hw_thin = thin_30m(gdf_hw, seed=config.GLOBAL_SEED)
    eow_thin = thin_30m(gdf_eow, seed=config.GLOBAL_SEED)

    print(f"HW  points: {len(gdf_hw):>4} raw -> {len(hw_thin):>4} after 30m thinning")
    print(f"EOW points: {len(gdf_eow):>4} raw -> {len(eow_thin):>4} after 30m thinning")

    combined = pd.concat([hw_thin, eow_thin], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=TGT_CRS)
    combined["x"] = combined.geometry.x
    combined["y"] = combined.geometry.y

    config.DATA.mkdir(parents=True, exist_ok=True)
    csv_cols = ["pt_id", "event", "type", "description", "elevation_ft",
                "source_file", "provenance", "x", "y"]
    combined[csv_cols].to_csv(config.DATA / "canonical_presence.csv", index=False)

    return combined

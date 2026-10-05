"""Tests for pipeline.evaluate's external comparisons
(FEMA/NWS/Sentinel-1, DIRECTIONAL CONSISTENCY, never "validated against")
and the GFI terrain-only baseline recomputed on the FIXED (WhiteboxTools)
flow_accumulation.

Two tiers, mirroring test_evaluate_oos.py's own convention:
  1. Pure-logic/synthetic tests (fast, no real raster/vector I/O) --
     `spatial_overlap`/`_rasterize_binary` against hand-computed toys, plus
     the directional-consistency framing constants themselves (zero file
     dependency at all -- these must pass on a bare checkout).
  2. Real-fixture tests against the actual on-disk predictor/raw-data
     artifacts (data/processed/predictors/flow_accumulation.tif, data/raw/
     fema/flood_hazard_zones.gpkg, data/raw/nws/fim_*.gpkg) -- gitignored
     build/raw artifacts, so each skips (rather than hard-failing) if not
     present locally. The Sentinel-1 tests are a real, live network
     attempt (Planetary Computer) -- they assert the function degrades
     gracefully rather than mocking the network away (attempt it if
     reachable; if unavailable, skip with a clear note).
"""
import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from pipeline import config, evaluate, predictors


def _fake_raster(values, cell=10.0, crs="EPSG:26914"):
    """Minimal synthetic georeferenced raster (rioxarray DataArray) --
    same recipe as test_evaluate_oos.py's own `_fake_raster` (duplicated
    here rather than imported across test modules -- this repo's test
    modules do not import each other)."""
    values = np.asarray(values, dtype="float64")
    da = xr.DataArray(values, dims=("y", "x"))
    transform = from_origin(0, values.shape[0] * cell, cell, cell)
    da = da.rio.write_transform(transform)
    da = da.rio.write_crs(crs)
    return da


# ==========================================================================
# 1. spatial_overlap -- hand-computed 3x3 toy (cell-by-cell, worked by
#    hand):
#      pred = [[1,1,0],
#              [0,1,0],
#              [0,0,1]]
#      ref  = [[1,0,0],
#              [0,1,1],
#              [0,0,1]]
#    (0,0): pred1 ref1 -> TP   (0,1): pred1 ref0 -> FP   (0,2): pred0 ref0 -> TN
#    (1,0): pred0 ref0 -> TN   (1,1): pred1 ref1 -> TP   (1,2): pred0 ref1 -> FN
#    (2,0): pred0 ref0 -> TN   (2,1): pred0 ref0 -> TN   (2,2): pred1 ref1 -> TP
#    => TP=3 FP=1 FN=1 TN=4
#      jaccard   = 3/(3+1+1) = 0.6
#      hit_rate  = 3/(3+1)   = 0.75   (recall)
#      precision = 3/(3+1)   = 0.75
#      f1        = 2*3/(2*3+1+1) = 6/8 = 0.75
# ==========================================================================

PRED_TOY = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
REF_TOY = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)


def test_spatial_overlap_hand_computed_counts():
    m = evaluate.spatial_overlap(PRED_TOY, REF_TOY)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 4)


def test_spatial_overlap_hand_computed_scores():
    m = evaluate.spatial_overlap(PRED_TOY, REF_TOY)
    assert m["jaccard"] == pytest.approx(0.6)
    assert m["hit_rate"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)


def test_spatial_overlap_respects_valid_mask():
    """Marking the (0,0) TP cell invalid must remove it from every count --
    proves `valid_mask` genuinely restricts the comparison rather than
    being accepted and silently ignored."""
    valid = np.ones_like(PRED_TOY, dtype=bool)
    valid[0, 0] = False
    m = evaluate.spatial_overlap(PRED_TOY, REF_TOY, valid_mask=valid)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)


def test_spatial_overlap_empty_masks_give_nan_not_crash():
    pred = np.zeros((3, 3), dtype=bool)
    ref = np.zeros((3, 3), dtype=bool)
    m = evaluate.spatial_overlap(pred, ref)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 0)
    assert np.isnan(m["jaccard"]) and np.isnan(m["hit_rate"]) and np.isnan(m["precision"])


# ==========================================================================
# 2. _rasterize_binary -- tiny synthetic GeoDataFrame + synthetic reference
#    grid, zero dependency on the real FEMA/NWS files.
# ==========================================================================

def test_rasterize_binary_none_returns_all_false():
    ref = _fake_raster(np.zeros((4, 4)))
    mask = evaluate._rasterize_binary(None, ref)
    assert mask.shape == (4, 4) and not mask.any()


def test_rasterize_binary_empty_gdf_returns_all_false():
    ref = _fake_raster(np.zeros((5, 5)))
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:26914")
    mask = evaluate._rasterize_binary(empty, ref)
    assert mask.shape == (5, 5) and not mask.any()


def test_rasterize_binary_burns_a_polygon():
    # 10x10 grid, 10 m cells -> 100x100 m AOI, top-left origin (0, 100).
    ref = _fake_raster(np.zeros((10, 10)), cell=10.0)
    poly = box(0, 70, 30, 100)   # the top-left 30x30 m corner
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:26914")
    mask = evaluate._rasterize_binary(gdf, ref)
    assert mask.shape == (10, 10)
    assert mask[1, 1] == True     # cell center (15,85) -- clearly inside
    assert mask[9, 9] == False    # cell center (95,5)  -- opposite corner, outside
    assert mask.sum() > 0


# ==========================================================================
# 3. Directional-consistency framing -- the core framing
#    requirement, checkable with ZERO real-file dependency: the emitted
#    "kind"/"framing"/"scope_caveat" constants themselves.
# ==========================================================================

# Phrases that would constitute an AFFIRMATIVE validation claim -- the
# thing the framing prohibits. The DISCLAIMING form ("this is NOT ...
# validation") is expected and required (matches screen.py's own
# BIVARIATE_FRAMING convention) -- only these bare/affirmative
# phrasings must never appear.
FORBIDDEN_VALIDATION_CLAIMS = (
    "validated against", "validates the model", "model validation:",
    "this validates", "validation of the model",
)


def test_directional_kind_constant():
    assert evaluate.DIRECTIONAL_KIND == "directional_consistency"


def test_framing_disclaims_validation_explicitly():
    """Matches this codebase's OWN established convention (screen.py's
    BIVARIATE_FRAMING / bivariate()'s not_validation pattern) for exactly
    this kind of caveat: "validation" is expected to appear, but only
    inside an explicit "NOT ... validation" disclaimer, mirroring
    test_bivariate.py's own `test_labeled_characterization_not_validation`."""
    framing = evaluate.DIRECTIONAL_FRAMING.lower()
    assert "not" in framing and "validation" in framing
    for phrase in FORBIDDEN_VALIDATION_CLAIMS:
        assert phrase not in framing


def test_scope_caveat_mentions_urban_pluvial_and_all_three_products():
    caveat = evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT.lower()
    assert "urban" in caveat and ("pluvial" in caveat or "stormwater" in caveat)
    assert "fema" in caveat and "nws" in caveat and "sentinel" in caveat


def test_final_maxss_threshold_reads_real_final_results():
    if not evaluate.FINAL_RESULTS_CSV.exists():
        pytest.skip(f"{evaluate.FINAL_RESULTS_CSV} not built locally (run the MaxEnt fits first)")
    t = evaluate.final_maxss_threshold()
    assert 0.0 < t < 1.0
    # final_maxss_threshold() reads the FINAL (10-replicate bootstrap) model's
    # maxentResults.csv, regenerated non-deterministically run-to-run (MaxEnt
    # exposes no CLI seed -- see pipeline.maxent.fit_final's docstring), so the
    # MaxSS threshold jitters (observed 0.13961, then 0.12355). Band-pin around
    # the observed neighborhood rather than the old exact abs=1e-4 value.
    assert t == pytest.approx(0.13, abs=0.05)


# ==========================================================================
# 4. gfi_baseline on the fixed accumulation. Real-fixture: skips if the
#    on-disk predictor stack is absent.
# ==========================================================================

@pytest.fixture(scope="module")
def real_gfi_result():
    if not evaluate.FLOW_ACCUMULATION_TIF.exists():
        pytest.skip(f"{evaluate.FLOW_ACCUMULATION_TIF} not built locally (run predictors.flow_derivatives() first)")
    return evaluate.gfi_baseline()


def test_gfi_baseline_reads_the_fixed_accumulation(real_gfi_result):
    """The accumulation gate itself: flow_accumulation.tif's own max must be well
    past river scale (> 50,000) -- the OLD xarray-spatial chain topped out
    at 2,451 cells (see tests/test_routing.py::test_real_aoi_reaches_
    river_scale, the SAME gate value/rationale)."""
    assert real_gfi_result["flow_accumulation_max"] > 50_000, (
        "gfi_baseline read a flow_accumulation.tif that never reaches "
        "river scale -- this would be the OLD capped raster, not the "
        "FIXED WhiteboxTools one"
    )


def test_gfi_baseline_auc_pinned_value(real_gfi_result):
    """Pins the ACTUAL, honestly-observed GFI AUC on the FIXED D8 routing:
    0.4696 -- essentially unchanged from the OLD (broken-routing) value of
    ~0.4997. The "terrain can't discriminate" claim HOLDS even after fixing
    the routing bug -- it was NOT an artifact of the routing failure.

    This is a REGRESSION PIN, not a target: if a future predictor-stack
    change legitimately moves this number, update this assertion to the
    new honestly-observed value -- never loosen it just to make a stale
    number pass.
    """
    print(f"\n[test_gfi_baseline_auc_pinned_value] GFI AUC on FIXED D8 = "
          f"{real_gfi_result['auc']:.6f} (old broken-routing value was ~0.4997)")
    assert real_gfi_result["auc"] == pytest.approx(0.4695583032490975, abs=5e-4)
    assert real_gfi_result["n_presence"] == 277


def test_gfi_baseline_matches_predictors_gfi_formula_on_twi():
    """gfi_baseline's GFI surface must be (numerically) identical to the
    already on-disk twi.tif (flow_derivatives' output): GFI and
    TWI are the IDENTICAL formula (`ln(acc*cell_area / tan(slope))`,
    predictors.gfi's own docstring) -- this is an independent correctness
    cross-check for the GFI re-check itself (proves gfi_baseline is really
    reading/combining the fixed inputs correctly, not a coincidence),
    never a claim about model performance.
    """
    if not (evaluate.FLOW_ACCUMULATION_TIF.exists() and evaluate.SLOPE_TIF.exists()):
        pytest.skip("predictor stack not built locally (run predictors.flow_derivatives() first)")
    twi_path = evaluate.FLOW_ACCUMULATION_TIF.parent / "twi.tif"
    if not twi_path.exists():
        pytest.skip(f"{twi_path} not built locally (run predictors.flow_derivatives() first)")

    acc_da = evaluate._open_raster(evaluate.FLOW_ACCUMULATION_TIF)
    slope_da = evaluate._open_raster(evaluate.SLOPE_TIF)
    gfi_arr = np.asarray(predictors.gfi(acc_da.values, np.deg2rad(slope_da.values), cell_area=100.0))

    twi_da = evaluate._open_raster(twi_path)
    both_finite = np.isfinite(gfi_arr) & np.isfinite(twi_da.values)
    assert both_finite.sum() > 0
    np.testing.assert_allclose(gfi_arr[both_finite], twi_da.values[both_finite], atol=1e-3)


def test_gfi_baseline_is_deterministic_for_same_seed():
    if not evaluate.FLOW_ACCUMULATION_TIF.exists():
        pytest.skip(f"{evaluate.FLOW_ACCUMULATION_TIF} not built locally (run predictors.flow_derivatives() first)")
    r1 = evaluate.gfi_baseline(seed=config.GLOBAL_SEED)
    r2 = evaluate.gfi_baseline(seed=config.GLOBAL_SEED)
    assert r1["auc"] == r2["auc"]


# ==========================================================================
# 5. fema_comparison / nws_comparison -- real-fixture, local cached files
#    under data/raw/ (gitignored, but already present on this machine).
# ==========================================================================

def test_fema_comparison_real_data():
    if not evaluate.FEMA_GPKG.exists():
        pytest.skip(f"{evaluate.FEMA_GPKG} not present locally")
    result = evaluate.fema_comparison()
    for zone_key in ("sfha_1pct", "zone_x_0_2pct"):
        m = result[zone_key]
        assert 0.0 <= m["jaccard"] <= 1.0
        assert 0.0 <= m["hit_rate"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
    assert result["kind"] == "directional_consistency"
    assert result["scope_caveat"] == evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT


def test_nws_comparison_real_data():
    if not evaluate.NWS_DIR.exists():
        pytest.skip(f"{evaluate.NWS_DIR} not present locally")
    result = evaluate.nws_comparison()
    seen_ok = False
    for cat in evaluate.NWS_CATEGORIES:
        m = result["categories"].get(cat)
        assert m is not None, f"missing category {cat!r} in result"
        if "status" in m:
            continue   # tolerated missing-cache case, matches source notebook
        seen_ok = True
        assert 0.0 <= m["jaccard"] <= 1.0
        assert 0.0 <= m["hit_rate"] <= 1.0
    assert seen_ok, "no NWS category actually scored -- all cache files missing?"
    assert result["scope_caveat"] == evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT


def test_fema_and_nws_share_the_same_threshold_and_caveat():
    """external_comparisons composes both under ONE threshold/caveat --
    this proves the two sub-functions never silently drift to different
    operating points when called at their own defaults."""
    if not (evaluate.FEMA_GPKG.exists() and evaluate.NWS_DIR.exists()):
        pytest.skip("FEMA/NWS local data not present")
    fema = evaluate.fema_comparison()
    nws = evaluate.nws_comparison()
    assert fema["threshold"] == nws["threshold"]
    assert fema["scope_caveat"] == nws["scope_caveat"] == evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT


# ==========================================================================
# 6. sentinel1_comparison -- ATTEMPTED for real (live Planetary Computer
#    network call); must NEVER raise regardless of whether the fetch
#    succeeds -- attempt it if reachable; if unavailable, skip with a
#    clear note, never block the pipeline on S1.
# ==========================================================================

def test_sentinel1_comparison_never_raises_and_is_well_formed():
    result = evaluate.sentinel1_comparison()   # real attempt, no mocking
    assert result["status"] in ("ok", "skipped")
    assert result["kind"] == "directional_consistency"
    assert result["scope_caveat"] == evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT
    if result["status"] == "ok":
        assert 0.0 <= result["jaccard"] <= 1.0
        assert 0.0 <= result["hit_rate"] <= 1.0
        print(f"\n[test_sentinel1_comparison] S1 attempt SUCCEEDED: "
              f"jaccard={result['jaccard']:.4f} hit_rate={result['hit_rate']:.4f} "
              f"precision={result['precision']:.4f}")
    else:
        assert result["reason"]
        print(f"\n[test_sentinel1_comparison] S1 attempt SKIPPED: {result['reason']}")


# ==========================================================================
# 7. external_comparisons -- THE key test: the emitted metadata,
#    walked recursively over every nested string, contains NO "validated"/
#    "validation-against" CLAIM anywhere -- checked structurally over the
#    whole tree (top-level framing AND each product's own nested framing/
#    reason strings), not just one field.
# ==========================================================================

def _all_strings(obj):
    """Recursively yield every string value nested anywhere inside `obj`
    (dict keys+values, list/tuple items, or a bare string)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _all_strings(k)
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _all_strings(item)


@pytest.fixture(scope="module")
def real_external_result():
    if not evaluate.FINAL_RASTER.exists():
        pytest.skip(f"{evaluate.FINAL_RASTER} not built locally (run the MaxEnt fits first)")
    return evaluate.external_comparisons()   # real FEMA+NWS+S1 attempt, once


def test_external_comparisons_metadata_has_kind_directional_consistency(real_external_result):
    result = real_external_result
    assert result["kind"] == "directional_consistency"
    assert result["fema"]["kind"] == "directional_consistency"
    assert result["nws"]["kind"] == "directional_consistency"
    assert result["s1"]["kind"] == "directional_consistency"


def test_external_comparisons_framing_and_caveat_identical_across_products(real_external_result):
    """ONE literal framing/caveat string reused everywhere -- never a
    different paraphrase per product (the source notebook's contradictory
    scoping was exactly this failure mode)."""
    result = real_external_result
    for key in ("fema", "nws", "s1"):
        assert result[key]["framing"] == evaluate.DIRECTIONAL_FRAMING
        assert result[key]["scope_caveat"] == evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT


def test_external_comparisons_metadata_never_claims_validation(real_external_result):
    """THE key test: grep the ENTIRE emitted metadata dict
    (every nested string, not just the top level) for a forbidden
    affirmative "validated"/"validation-against" claim."""
    blob = " ".join(_all_strings(real_external_result)).lower()

    for phrase in FORBIDDEN_VALIDATION_CLAIMS:
        assert phrase not in blob, f"forbidden validation claim found in metadata: {phrase!r}"

    # The word IS expected to appear -- but only inside the explicit "NOT
    # ... validation" disclaimer (proves the assertion above isn't
    # vacuously true because "validation" never appears at all).
    assert "validation" in blob
    assert "not" in blob
    assert "directional" in blob


def test_external_comparisons_shares_one_threshold_across_all_three_products(real_external_result):
    result = real_external_result
    assert result["fema"]["threshold"] == result["threshold"]
    assert result["nws"]["threshold"] == result["threshold"]
    if result["s1"]["status"] == "ok":
        assert result["s1"]["threshold"] == result["threshold"]


# ============================================================================
# capture_vs_area -- the interpretable common-yardstick comparison, and the
# leakage guard that keeps it honest.
# ============================================================================

def test_capture_vs_area_defaults_to_the_train_only_raster():
    """THE regression guard. Scoring the PUBLISHED all-277 raster at held-out
    marks would count captures for a surface fitted on those very marks --
    resubstitution on one side of a comparison whose other side (FEMA) is not
    fitted at all. The default must therefore be the train-only eval raster."""
    import inspect
    sig = inspect.signature(evaluate.capture_vs_area)
    assert sig.parameters["final_raster"].default is None, (
        "final_raster must default to None so the function can resolve it to "
        "EVAL_RASTER; a literal FINAL_RASTER default reintroduces the train/test leak"
    )
    src = inspect.getsource(evaluate.capture_vs_area)
    assert "final_raster = EVAL_RASTER" in src


def test_mask_capture_hand_computed():
    """4x4 grid, all valid; a mask covering the left half; three points, two of
    them in the masked half."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from rasterio.transform import from_origin

    transform = from_origin(0, 40, 10, 10)          # 10 m cells, origin top-left
    ref = xr.DataArray(np.zeros((4, 4), dtype="float32"),
                       dims=("y", "x"),
                       coords={"y": [35, 25, 15, 5], "x": [5, 15, 25, 35]})
    ref = ref.rio.write_transform(transform).rio.write_crs("EPSG:26914")

    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True                               # left half -> 50% of area
    valid = np.ones((4, 4), dtype=bool)

    pts = pd.DataFrame({"x": [5.0, 15.0, 35.0], "y": [35.0, 25.0, 5.0]})
    area, n_hit, n_tot = evaluate._mask_capture(mask, pts, ref, valid)
    assert area == pytest.approx(0.5)
    assert (n_hit, n_tot) == (2, 3)


def test_capture_vs_area_real(tmp_path):
    if not evaluate.EVAL_RASTER.exists():
        pytest.skip(f"{evaluate.EVAL_RASTER} not built locally")
    if not evaluate.FEMA_GPKG.exists():
        pytest.skip(f"{evaluate.FEMA_GPKG} not present locally")

    df = evaluate.capture_vs_area(out_csv=tmp_path / "capture.csv")
    assert {"source", "area_fraction", "n_captured", "n_points",
            "capture_fraction", "capture_per_area"} <= set(df.columns)
    assert (df["area_fraction"].between(0, 1)).all()
    assert (df["capture_fraction"].between(0, 1)).all()
    # the two derived columns must be internally consistent
    assert np.allclose(df["capture_fraction"], df["n_captured"] / df["n_points"])
    assert np.allclose(df["capture_per_area"], df["capture_fraction"] / df["area_fraction"])
    # every row scores the SAME points, or the comparison is not like-for-like
    assert df["n_points"].nunique() == 1
    assert (tmp_path / "capture.csv").exists()

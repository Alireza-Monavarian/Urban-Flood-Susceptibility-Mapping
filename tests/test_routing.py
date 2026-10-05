import numpy as np
import rioxarray

from pipeline import predictors


def test_accumulation_grows_downslope(tmp_path):
    """Synthetic 100x100 DEM: a 40-row flat plateau band (zero north-south
    gradient) plus a V-shaped valley cross-section present at every row,
    including through the plateau -- so the two center columns inside the
    plateau are a genuine 2-D flat (see
    predictors._d8_accumulation_synthetic's docstring for the exact
    construction). Routed through the same WhiteboxTools
    breach_depressions_least_cost -> d8_flow_accumulation chain
    flow_derivatives() uses below, not a parallel reimplementation.

    Regression guard for the capped-accumulation bug: the old `xarray-spatial` D8 chain
    (`fill_d8`/`flow_direction_d8`/`flow_accumulation_d8`) could not
    resolve flats like this and capped real-AOI flow accumulation at 2451
    cells, never reaching river scale. If this router is flats-safe, flow
    still converges into the valley and keeps accumulating past the flat
    band on its way to the southern outlet -- far more than a "few cells".
    """
    acc = predictors._d8_accumulation_synthetic(tmp_path)
    assert acc.max() > 1000, f"router failed on flats: max acc {acc.max()}"


def test_real_aoi_reaches_river_scale():
    """The flow-routing correctness gate itself: on the real Manhattan, KS AOI, D8
    flow accumulation -- routed via WhiteboxTools on a DEM buffered 5 km
    beyond the AOI, then clipped/reproject_matched back onto the AOI grid
    -- must reach river scale. The old xarray-spatial D8 chain capped out
    at 2451 cells (never reaching river scale) because it could not
    resolve flats over the real terrain.
    """
    acc = predictors.flow_derivatives(predictors.dem_path())["flow_accumulation"]
    assert float(np.nanmax(acc.values)) > 50_000, \
        "flow accumulation never reaches river scale -- routing still broken (old max was 2451)"


def test_routing_outputs_use_nan_nodata_not_sentinel():
    """Regression guard for the twi/spi nodata-sentinel-leak bug: ``_save()``
    used to write genuine in-memory NaN cells in ``twi``/``spi`` back out as
    WhiteboxTools' ``-32768`` nodata sentinel, because ``twi``/``spi`` (built
    via ``acc.copy(data=...)``) inherited a stale ``-32768``-tagged nodata
    *encoding* from ``acc`` that ``to_raster()`` used to re-encode the
    array's real NaN cells on write -- see ``predictors._save``'s docstring
    for the exact mechanism and why the fix needs ``write_nodata(...,
    encoded=True)``, not a plain ``write_nodata(np.nan)``.

    Reads the SAVED FILES back unmasked (raw stored values), not
    ``flow_derivatives()``'s in-memory return value or a ``masked=True``
    re-read: a masked read auto-decodes a ``-32768``-tagged file back to
    NaN regardless of whether the underlying write bug is present, so it
    would pass even against the original buggy output -- only an unmasked
    read can tell a genuinely-NaN cell apart from a finite ``-32768`` that
    merely happens to decode the same way.

    Also checks (via a ``masked=True`` re-read, where NaN nodata decodes
    normally either way) that twi/spi's NaN-mask is pixel-identical to
    slope's -- nodata consistency across the predictor stack, since Task
    3.1 NaN-drops presence-point samples and a non-NaN cell here would
    silently survive that drop while every other layer's equivalent edge
    cell would not.

    Calls ``flow_derivatives()`` itself (cache-aware: reuses the files
    ``test_real_aoi_reaches_river_scale`` above already produced in this
    run rather than re-triggering WhiteboxTools) so this exercises the real
    production path, not a parallel reimplementation.
    """
    predictors.flow_derivatives(predictors.dem_path())
    out_dir = predictors.dem_path().parent

    for name in ("flow_accumulation", "twi", "spi", "hand"):
        raw = rioxarray.open_rasterio(out_dir / f"{name}.tif", masked=False)
        n_sentinel = int(np.sum(raw.values == -32768))
        assert n_sentinel == 0, (
            f"{name}.tif: {n_sentinel} cell(s) store the -32768 sentinel as "
            "a raw finite value instead of NaN nodata"
        )

    slope = rioxarray.open_rasterio(out_dir / "slope.tif", masked=True)
    slope_nan = np.isnan(slope.values)

    for name in ("twi", "spi"):
        layer = rioxarray.open_rasterio(out_dir / f"{name}.tif", masked=True)
        layer_nan = np.isnan(layer.values)
        assert int(layer_nan.sum()) == int(slope_nan.sum()) == 7287, (
            f"{name}.tif: {int(layer_nan.sum())} NaN cells vs slope.tif's "
            f"{int(slope_nan.sum())} (expected 7287 each -- the known "
            "undefined-slope edge-cell count for this AOI/DEM)"
        )
        assert np.array_equal(layer_nan, slope_nan), (
            f"{name}.tif: NaN mask is not pixel-identical to slope.tif's "
            "NaN mask -- nodata pattern should match exactly"
        )

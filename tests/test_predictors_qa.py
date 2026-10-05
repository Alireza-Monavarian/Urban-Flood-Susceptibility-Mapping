

def test_access_proxy_never_enters_the_modeling_stack():
    """The distance-to-road access proxy must NOT be a modeling predictor.

    screen._load_stack() globs "*.tif" under data/processed/predictors/, so any
    raster written there is silently picked up as a predictor. On 2026-08-13 the
    access proxy was written there and became an 18th modeling layer; it was
    caught only by asserting the count. An access proxy in the model would bake
    the survey process into the susceptibility surface -- the exact confounding
    the layer exists to MEASURE.
    """
    from pipeline import maxent, predictors

    layers = maxent.modeling_layers()
    assert len(layers) == 17, f"modeling stack is {len(layers)}, expected 17: {sorted(layers)}"
    assert predictors.ACCESS_LAYER not in layers
    assert predictors.ACCESS_DIR != predictors.OUT_DIR, \
        "the access proxy must not be written into the globbed predictors/ directory"

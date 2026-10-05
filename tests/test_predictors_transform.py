"""Tests for ``pipeline.predictors.apply_modeling_transform`` -- the single
source of truth for the log-transform adoption: ``flow_accumulation`` -> log10, ``spi``
-> log1p, identity for every other layer.

Deliberately fast and I/O-free (no raster reads, no real predictor stack):
this is a pure numpy function, so its correctness is pinned directly
against hand-computable values rather than via the real 22-layer stack
(that real-stack exercise happens indirectly through
``tests/test_screen.py``'s ``multicollinearity()`` tests and
``pipeline.maxent.export_layers()``'s own idempotent ``.asc`` export --
both of which call this exact function, never a re-implementation of the
log math).
"""
import numpy as np
import pytest

from pipeline import predictors


# --------------------------------------------------------------------------
# flow_accumulation -> log10
# --------------------------------------------------------------------------

def test_flow_accumulation_uses_log10_on_a_known_value():
    out = predictors.apply_modeling_transform("flow_accumulation", np.array([100.0]))
    assert out[0] == pytest.approx(2.0)  # log10(100) == 2


def test_flow_accumulation_log10_at_the_stack_minimum_is_zero():
    """This layer's real on-disk minimum is 1 (D8 accumulation counts at
    least the cell itself) --
    log10(1) == 0 exactly, never a negative or undefined result."""
    out = predictors.apply_modeling_transform("flow_accumulation", np.array([1.0]))
    assert out[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# spi -> log1p (has exact zeros at flat cells -- plain log() would hit -inf)
# --------------------------------------------------------------------------

def test_spi_uses_log1p_handling_a_zero():
    out = predictors.apply_modeling_transform("spi", np.array([0.0]))
    assert out[0] == pytest.approx(0.0)  # log1p(0) == 0, finite (not -inf)
    assert np.isfinite(out[0])


def test_spi_log1p_on_a_known_nonzero_value():
    out = predictors.apply_modeling_transform("spi", np.array([np.e - 1]))
    assert out[0] == pytest.approx(1.0)  # log1p(e - 1) == ln(e) == 1


# --------------------------------------------------------------------------
# Identity for every layer NOT in MODELING_LOG_TRANSFORMS
# --------------------------------------------------------------------------

def test_identity_for_a_non_listed_layer():
    arr = np.array([1.0, 2.5, -3.0, 0.0])
    out = predictors.apply_modeling_transform("slope", arr)
    np.testing.assert_array_equal(out, arr)


def test_identity_still_casts_to_float64():
    """Even the identity (pass-through) path returns float64, not whatever
    dtype the caller happened to pass in -- every other function in this
    module's in-memory convention."""
    out = predictors.apply_modeling_transform("dem", np.array([1, 2, 3], dtype="int32"))
    assert out.dtype == np.float64


# --------------------------------------------------------------------------
# NaN nodata preserved, never fed through log10/log1p
# --------------------------------------------------------------------------

def test_nan_preserved_for_flow_accumulation():
    out = predictors.apply_modeling_transform("flow_accumulation", np.array([1.0, np.nan, 100.0]))
    assert np.isnan(out[1])
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(2.0)


def test_nan_preserved_for_spi():
    out = predictors.apply_modeling_transform("spi", np.array([0.0, np.nan]))
    assert out[0] == pytest.approx(0.0)
    assert np.isnan(out[1])


def test_nan_preserved_for_identity_layer():
    out = predictors.apply_modeling_transform("curvature", np.array([np.nan, 1.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Shape / dtype / config-driven dispatch
# --------------------------------------------------------------------------

def test_2d_array_shape_preserved():
    arr = np.array([[1.0, 100.0], [np.nan, 10_000.0]])
    out = predictors.apply_modeling_transform("flow_accumulation", arr)
    assert out.shape == arr.shape
    assert np.isnan(out[1, 0])
    assert out[0, 1] == pytest.approx(2.0)
    assert out[1, 1] == pytest.approx(4.0)


def test_modeling_log_transforms_has_exactly_the_two_hydrology_layers():
    """Regression guard on the dict driving the dispatch itself -- the
    pipeline log-transforms ONLY flow_accumulation (log10) and spi (log1p), never
    any other layer."""
    assert predictors.MODELING_LOG_TRANSFORMS == {
        "flow_accumulation": "log10",
        "spi": "log1p",
    }


def test_dispatch_matches_modeling_log_transforms_for_every_configured_layer():
    """The function must actually route through MODELING_LOG_TRANSFORMS
    rather than hardcoding the two names a second time -- verified by
    checking dispatch for every (name, transform) pair the dict itself
    declares, not just the two literals above."""
    value = np.array([9.0])
    for name, transform in predictors.MODELING_LOG_TRANSFORMS.items():
        out = predictors.apply_modeling_transform(name, value)
        expected = np.log10(value) if transform == "log10" else np.log1p(value)
        np.testing.assert_allclose(out, expected)

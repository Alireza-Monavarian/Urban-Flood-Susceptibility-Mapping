import pytest

from pipeline import config, predictors, qa
from pipeline.predictors import CENSUS_LAYERS, EXPECTED_NON_CENSUS


def test_stack_aligned_and_no_precip():
    # Full real acquisition (network-heavy). Census is acquired only if
    # .census_api_key is present on this machine -- handled below.
    stack = predictors.acquire_all()

    assert "precip_annual_mean" not in stack and "precip_rx1day" not in stack

    # Completeness: a partial acquisition (e.g. a source silently dropped
    # from the stack) must fail loudly here rather than pass silently just
    # because alignment/precip-absence still hold. EXPECTED_NON_CENSUS /
    # CENSUS_LAYERS are imported from pipeline.predictors -- the same
    # canonical lists acquire_all() itself now enforces internally -- so
    # there is one source of truth, not two.
    missing = [name for name in EXPECTED_NON_CENSUS if name not in stack]
    assert not missing, f"acquire_all() stack is missing non-census layer(s): {missing}"

    if (config.REPO / ".census_api_key").exists():
        missing_census = [name for name in CENSUS_LAYERS if name not in stack]
        assert not missing_census, f"acquire_all() stack is missing census layer(s): {missing_census}"

    qa.assert_aligned(list(stack.values()))


def test_acquire_all_raises_on_incomplete_stack(monkeypatch):
    """Synthetic/no-network: every layer still loads from its on-disk cache
    (no re-acquisition), except that ``acquire_stormwater`` is monkeypatched
    to drop one of its two names from the dict it returns, without raising
    -- exactly the "silently incomplete, `failures` never incremented"
    failure mode the completeness gate in acquire_all() exists to catch
    (the real-world trigger was a failed prerequisite silently skipping
    curve_number; this reproduces the same *shape* of bug -- a name absent
    from the stack with no corresponding entry in `failures` -- via the
    cheapest source to intercept without network I/O).
    """
    real_acquire_stormwater = predictors.acquire_stormwater

    def _fake_stormwater(dem, out_dir=predictors.OUT_DIR):
        result = real_acquire_stormwater(dem, out_dir)
        result.pop("distance_to_outfall", None)
        return result

    monkeypatch.setattr(predictors, "acquire_stormwater", _fake_stormwater)

    with pytest.raises(RuntimeError, match="distance_to_outfall"):
        predictors.acquire_all()

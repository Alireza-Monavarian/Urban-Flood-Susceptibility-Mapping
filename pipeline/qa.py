import numpy as np


def assert_counts(**pairs):
    for name, (expected, actual) in pairs.items():
        assert expected == actual, f"[QA count] {name}: expected {expected}, got {actual}"


def assert_aligned(rasters):
    ref = rasters[0].rio
    for r in rasters[1:]:
        assert r.rio.crs == ref.crs and r.rio.shape == ref.shape and \
               r.rio.transform() == ref.transform(), "[QA align] CRS/shape/transform mismatch"


def assert_no_leakage(train_idx, test_idx):
    inter = set(np.asarray(train_idx).tolist()) & set(np.asarray(test_idx).tolist())
    assert not inter, f"[QA leakage] train∩test = {sorted(inter)[:5]}…"


def assert_eval_model_excludes_test(model_train_idx, test_idx):
    """Invariant 2: a model used for out-of-sample eval must not have trained on the eval points."""
    assert_no_leakage(model_train_idx, test_idx)

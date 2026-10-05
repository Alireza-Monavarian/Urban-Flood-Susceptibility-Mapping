import numpy as np, pytest
from pipeline import qa


def test_no_leakage_passes_on_disjoint():
    qa.assert_no_leakage(np.array([0, 1, 2]), np.array([3, 4]))       # no raise


def test_no_leakage_catches_overlap():
    with pytest.raises(AssertionError):
        qa.assert_no_leakage(np.array([0, 1, 2, 3]), np.array([3, 4]))


def test_eval_model_excludes_test():
    with pytest.raises(AssertionError):
        qa.assert_eval_model_excludes_test(model_train_idx=np.array([0, 1, 2, 3]),
                                           test_idx=np.array([2]))


def test_assert_counts_passes_and_catches_mismatch():
    qa.assert_counts(model_points=(278, 278))     # equal -> no raise
    with pytest.raises(AssertionError):
        qa.assert_counts(hw_raw=(595, 594))       # mismatch -> raises

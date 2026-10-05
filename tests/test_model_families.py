"""Tests for pipeline.maxent's final-vs-evaluation model-family split
(invariant 2, the correctness fix behind the resubstitution-AUC bug: a
full-model raster sampled at its own training points had silently stood
in for the headline accuracy number, 0.9842 in-sample vs. a true 0.9684
holdout).

Deliberately fast and MaxEnt-subprocess-free -- mirrors test_maxent_aicc.py's
own scoping note for ``aicc_grid()``: every test here exercises either a
pure-logic helper (``_fold_holdouts``, ``apparent_auc``'s CSV parsing) or
a ``_record_only=True`` path (``fit_eval``/``fit_folds``), never a real
MaxEnt/Java subprocess or predictor-raster I/O beyond the ONE real
``config.canonical_samples()`` call the key test
needs (to verify ``fit_eval``'s internal split recomputation agrees with
an external one -- see ``test_eval_model_never_sees_test``). The REAL
fits (``fit_final``'s 10-replicate bootstrap + one ``fit_eval`` train-only
run) are run once via a standalone script, not
asserted here as pytest -- same rationale test_maxent_aicc.py gives for
``aicc_grid()`` itself.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import config, maxent, qa


# --------------------------------------------------------------------------
# The key test, using the REAL canonical
# 277-point set -- config.split()'s determinism across two INDEPENDENT
# calls (one here, one inside fit_eval's own default) is exactly what's
# under test, not just a single split reused twice.
# --------------------------------------------------------------------------

def test_eval_model_never_sees_test():
    s = config.canonical_samples()
    tr, te = config.split(s)
    used = maxent.fit_eval(tr, _record_only=True)      # returns training idx it used
    qa.assert_eval_model_excludes_test(used, te)        # must NOT raise
    assert list(used) == list(tr)
    assert set(used).isdisjoint(set(te))


def test_fit_eval_record_only_raises_on_a_contaminated_train_idx():
    """A train_idx that (incorrectly) includes a held-out test point must
    be caught by the invariant assertion, even under _record_only=True --
    proves the check is a real guard (fires on bad input), not a
    tautology that can never trigger."""
    s = config.canonical_samples()
    tr, te = config.split(s)
    contaminated = np.concatenate([tr, te[:1]])         # leaks 1 test point
    with pytest.raises(AssertionError):
        maxent.fit_eval(contaminated, samples=s, _record_only=True)


# --------------------------------------------------------------------------
# FINAL_CONFIG -- locked to the owner's choice, NOT the grid's raw AICc
# winner (beta=0.25, LQ -- the grid's own low edge; see FINAL_CONFIG's
# module-level rationale in pipeline/maxent.py).
# --------------------------------------------------------------------------

def test_final_config_is_locked_to_beta_half_lq():
    assert maxent.FINAL_CONFIG == {"beta": 0.5, "features": "LQ"}


def test_final_replicates_and_formats_match_source_recipe():
    assert maxent.FINAL_REPLICATES == 10
    assert maxent.FINAL_FORMATS == ("cloglog", "logistic")


# --------------------------------------------------------------------------
# Final vs. eval (vs. folds) output directories must never collide -- a
# pure path-construction regression guard (no I/O).
# --------------------------------------------------------------------------

def test_final_and_eval_and_folds_output_dirs_are_distinct():
    dirs = [str(maxent.FINAL_DIR), str(maxent.HOLDOUT_MODEL_DIR), str(maxent.FOLDS_DIR)]
    assert len(set(dirs)) == 3
    for a in dirs:
        for b in dirs:
            if a != b:
                assert not a.startswith(b + "/"), f"{a} nested under {b}"


# --------------------------------------------------------------------------
# fit_folds -- pure-logic invariant coverage via a synthetic fold
# assignment; _record_only=True never touches a real predictor stack or
# the MaxEnt binary (only len(samples) matters before the early return).
# --------------------------------------------------------------------------

def test_fit_folds_complement_never_leaks_a_folds_own_points():
    n = 10
    fold_assignment = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    samples = pd.DataFrame({"x": np.arange(n), "y": np.arange(n)})   # length-only stand-in

    used = maxent.fit_folds(fold_assignment, samples=samples, _record_only=True)

    assert set(used.keys()) == {0, 1, 2}
    for fold_id, train_idx in used.items():
        held_out = np.where(fold_assignment == fold_id)[0]
        qa.assert_eval_model_excludes_test(train_idx, held_out)      # must NOT raise
        assert sorted(np.concatenate([train_idx, held_out]).tolist()) == list(range(n))


def test_fit_folds_accepts_loeo_style_dict():
    """config.loeo_folds() returns {event: held_out_idx}, not a per-sample
    array -- fit_folds must accept this shape too (robustness calls it
    with real loeo_folds() output for the LOEO significance test)."""
    n = 6
    fold_dict = {"E1": np.array([0, 1]), "E2": np.array([2, 3]), "E3": np.array([4, 5])}
    samples = pd.DataFrame({"x": np.arange(n), "y": np.arange(n)})

    used = maxent.fit_folds(fold_dict, samples=samples, _record_only=True)

    assert set(used.keys()) == {"E1", "E2", "E3"}
    for fold_id, train_idx in used.items():
        held_out = fold_dict[fold_id]
        qa.assert_eval_model_excludes_test(train_idx, held_out)      # must NOT raise
        assert sorted(np.concatenate([train_idx, held_out]).tolist()) == list(range(n))


def test_fold_holdouts_from_array_form():
    arr = np.array([0, 0, 1, 1, 2])
    holdouts, universe = maxent._fold_holdouts(arr, n=5)
    assert set(holdouts) == {0, 1, 2}
    assert holdouts[0].tolist() == [0, 1]
    assert holdouts[1].tolist() == [2, 3]
    assert holdouts[2].tolist() == [4]
    assert universe.tolist() == [0, 1, 2, 3, 4]


def test_fold_holdouts_from_dict_form_passes_through():
    d = {"a": np.array([0, 2]), "b": np.array([1, 3, 4])}
    holdouts, universe = maxent._fold_holdouts(d, n=5)
    assert holdouts["a"].tolist() == [0, 2]
    assert holdouts["b"].tolist() == [1, 3, 4]
    assert universe.tolist() == [0, 1, 2, 3, 4]


def test_fold_holdouts_array_length_mismatch_raises():
    with pytest.raises(AssertionError):
        maxent._fold_holdouts(np.array([0, 0, 1]), n=10)


# --------------------------------------------------------------------------
# apparent_auc() -- reads MaxEnt's own "Training AUC" column from a
# maxentResults.csv; pure CSV I/O over a tiny synthetic file, no real
# MaxEnt subprocess.
# --------------------------------------------------------------------------

def test_apparent_auc_averages_bootstrap_replicate_rows(tmp_path):
    fmt_dir = tmp_path / "cloglog"
    fmt_dir.mkdir()
    df = pd.DataFrame({
        "Species": [f"flood_{i}" for i in range(4)],
        "Training AUC": [0.96, 0.97, 0.95, 0.98],
    })
    df.to_csv(fmt_dir / "maxentResults.csv", index=False)

    result = maxent.apparent_auc(out_dir=tmp_path, fmt="cloglog")
    assert result["n_replicates"] == 4
    assert result["mean"] == pytest.approx(0.965)
    assert result["sd"] > 0


def test_apparent_auc_falls_back_to_single_run_row(tmp_path):
    fmt_dir = tmp_path / "cloglog"
    fmt_dir.mkdir()
    df = pd.DataFrame({"Species": ["flood"], "Training AUC": [0.97]})
    df.to_csv(fmt_dir / "maxentResults.csv", index=False)

    result = maxent.apparent_auc(out_dir=tmp_path, fmt="cloglog")
    assert result["n_replicates"] == 1
    assert result["mean"] == pytest.approx(0.97)
    assert result["sd"] == 0.0

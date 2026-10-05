import numpy as np, pandas as pd
from pipeline import config

def _synthetic(n=278):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x": rng.uniform(697000,712000,n), "y": rng.uniform(4337000,4345000,n),
                         "event": rng.choice(list("ABCDEFGH"), n), "type":"HW"})

def test_split_disjoint_and_deterministic():
    s = _synthetic()
    tr1, te1 = config.split(s); tr2, te2 = config.split(s)
    assert set(tr1).isdisjoint(te1)
    assert list(te1) == list(te2)                 # deterministic
    assert len(tr1) + len(te1) == len(s)

def test_loeo_covers_every_point_once():
    s = _synthetic()
    folds = config.loeo_folds(s)
    covered = sorted(i for idx in folds.values() for i in idx)
    assert covered == list(range(len(s)))         # EOW included, nothing dropped

def test_spatial_folds_uses_2km_blocks():
    # x-coords chosen so binning differs between 2km (correct) and 5km (the old bug):
    #   700000//2000=350, 701000//2000=350 (same 2km block),
    #   703000//2000=351 (different 2km block) — but ALL three share one 5km block (//5000=140).
    s = pd.DataFrame({"x": [700000.0, 701000.0, 703000.0],
                      "y": [4340000.0, 4340000.0, 4340000.0],
                      "event": ["A", "A", "A"], "type": "HW"})
    folds = config.spatial_folds(s)                 # default block_km=2
    assert folds[0] == folds[1], "700000 & 701000 must share one 2km block"
    assert folds[0] != folds[2], "703000 must be a different 2km block (would collapse at 5km — regression guard)"

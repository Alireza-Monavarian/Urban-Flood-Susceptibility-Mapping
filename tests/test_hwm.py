from pipeline import hwm


def test_thin_is_deterministic():
    df = hwm.build_presence()
    a = hwm.thin_30m(df, seed=42); b = hwm.thin_30m(df, seed=42)
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))


def test_raw_hw_total_is_595():
    assert hwm.raw_counts()["HW_raw"] == 595


def test_presence_has_eow_and_286():
    df = hwm.build_presence()
    assert (df["type"] == "EOW").sum() == 7 and len(df) == 286

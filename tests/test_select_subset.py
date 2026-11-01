import numpy as np
import pandas as pd

from brainengage.data.select_subset import balance


def test_balance_even_when_supply_allows():
    df = pd.DataFrame({"id": [f"v{i}" for i in range(90)], "duration_s": np.r_[np.full(30, 10.0), np.full(30, 30.0), np.full(30, 50.0)]})
    out = balance(df, 30, [5, 20, 40, 61], seed=1)
    assert len(out) == 30 and out.duration_bin.value_counts().tolist() == [10, 10, 10]


def test_balance_fills_from_other_bins_when_short():
    df = pd.DataFrame({"id": [f"v{i}" for i in range(50)], "duration_s": np.r_[np.full(45, 10.0), np.full(5, 50.0)]})
    out = balance(df, 30, [5, 20, 40, 61], seed=1)
    assert len(out) == 30 and (out.duration_bin == 2).sum() == 5


def test_balance_is_deterministic():
    df = pd.DataFrame({"id": [f"v{i}" for i in range(60)], "duration_s": np.random.default_rng(0).uniform(5, 60, 60)})
    a = balance(df, 20, [5, 20, 40, 61], seed=7).id.tolist()
    b = balance(df, 20, [5, 20, 40, 61], seed=7).id.tolist()
    assert a == b

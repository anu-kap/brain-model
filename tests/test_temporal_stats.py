import numpy as np
import pytest

from brainengage.features.temporal_stats import ALL_STATS, stat_names, temporal_stats


def test_shapes_and_order():
    ts = np.random.default_rng(0).standard_normal((12, 3))
    v = temporal_stats(ts, early_window_s=3)
    assert v.shape == (3 * len(ALL_STATS),)
    names = stat_names(["a", "b", "c"])
    assert names[0] == "a__mean" and names[len(ALL_STATS)] == "b__mean"


def test_early_window_is_first_rows_from_onset():
    ts = np.zeros((10, 1))
    ts[:3, 0] = [1, 2, 3]
    v = temporal_stats(ts, early_window_s=3)
    d = dict(zip(stat_names(["x"]), v))
    assert d["x__early"] == pytest.approx(2.0)   # mean of rows 0..2 (stimulus seconds 0-2)
    assert d["x__peak"] == 3 and d["x__time_to_peak"] == 2


def test_slope_sign_and_single_timestep():
    up = np.arange(10, dtype=float)[:, None]
    d = dict(zip(stat_names(["x"]), temporal_stats(up)))
    assert d["x__slope"] == pytest.approx(1.0)
    one = temporal_stats(np.array([[5.0]]))
    assert np.isfinite(one).all() and dict(zip(stat_names(["x"]), one))["x__slope"] == 0

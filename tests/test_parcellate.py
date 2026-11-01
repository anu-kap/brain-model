import numpy as np

from brainengage.config import resolve
from brainengage.features.parcellate import N_HEMI, N_VERT, parcel_timeseries, yeo7_vertex_labels


def test_yeo7_labels_cover_both_hemispheres():
    lab = yeo7_vertex_labels(resolve("data/tribe_cache/atlas"))
    assert lab.shape == (N_VERT,)
    assert set(np.unique(lab)) <= set(range(8))
    for hemi in (lab[:N_HEMI], lab[N_HEMI:]):
        assert all((hemi == k).sum() > 300 for k in range(1, 8)), "every network present in each hemisphere"


def test_parcel_timeseries_means_and_gfp():
    lab = np.zeros(N_VERT, int)
    lab[:100] = 1
    lab[100:200] = 2
    preds = np.zeros((4, N_VERT), np.float32)
    preds[:, :100] = 2.0
    preds[:, 100:200] = -1.0
    ts = parcel_timeseries(preds, lab, 2)
    assert ts.shape == (4, 3)
    assert np.allclose(ts[:, 0], 2.0) and np.allclose(ts[:, 1], -1.0)
    assert np.allclose(ts[:, 2], np.sqrt((100 * 4 + 100 * 1) / N_VERT))

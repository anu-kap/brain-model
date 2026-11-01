"""Map fsaverage5 vertices (left 10242 then right 10242, as TRIBE outputs them)
to cortical networks and aggregate time series per network.

Primary: Yeo 2011 7-network atlas, native fsaverage5 .annot files from the Yeo lab
(CBIG GitHub). Vertex order matches TRIBE output (left hemisphere first).
Exploratory: HCP-MMP1 via tribev2.utils (needs mne, GPU env only).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_HEMI = 10242
N_VERT = 2 * N_HEMI
YEO7_NAMES = ["visual", "somatomotor", "dorsal_attn", "ventral_attn", "limbic", "frontoparietal", "default"]


YEO_ANNOT_URL = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
                 "Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/fsaverage5/label/"
                 "{hemi}.Yeo2011_7Networks_N1000.annot")


def yeo7_vertex_labels(cache_dir: Path) -> np.ndarray:
    """int array (20484,) with 0 = medial wall and 1..7 = Yeo networks, in TRIBE vertex
    order (left 10242 then right 10242). Uses the Yeo lab's native fsaverage5 .annot files
    (no volume-to-surface approximation)."""
    import nibabel.freesurfer as fs
    import requests

    cache_dir = Path(cache_dir)
    f = cache_dir / "yeo7_fsaverage5_labels.npy"
    if f.exists():
        return np.load(f)
    annot_dir = cache_dir / "yeo_fsaverage5"
    annot_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for hemi in ("lh", "rh"):
        af = annot_dir / f"{hemi}.Yeo2011_7Networks_N1000.annot"
        if not af.exists():
            r = requests.get(YEO_ANNOT_URL.format(hemi=hemi), timeout=60)
            r.raise_for_status()
            af.write_bytes(r.content)
        lab, _ctab, names = fs.read_annot(str(af))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        assert names[0].endswith("Medial_Wall") and names[1:] == [f"7Networks_{i}" for i in range(1, 8)], names
        lab = np.asarray(lab).astype(int)
        lab[lab < 0] = 0
        assert lab.shape == (N_HEMI,), lab.shape
        labels.append(lab)
    out = np.concatenate(labels)
    log.info("Yeo-7 on fsaverage5: vertices per label %s", np.bincount(out, minlength=8).tolist())
    np.save(f, out)
    return out


def hcp_vertex_labels() -> tuple[np.ndarray, list[str]]:
    """HCP-MMP1 bilateral parcels via tribev2.utils (GPU env only). Returns (labels, names)."""
    from tribev2.utils import get_hcp_labels  # lazy: needs mne + tribev2

    label_to_vertices = get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both")
    names = sorted(label_to_vertices)
    out = np.zeros(N_VERT, dtype=int)
    for i, name in enumerate(names, start=1):
        out[label_to_vertices[name]] = i
    return out, names


def parcel_timeseries(preds: np.ndarray, labels: np.ndarray, n_labels: int, add_gfp: bool = True) -> np.ndarray:
    """preds (T, 20484) -> (T, n_labels [+1]) mean per label (label 0 ignored), plus
    global field power (RMS over all vertices) as the last channel."""
    preds = np.asarray(preds, dtype=np.float32)
    if preds.shape[1] != labels.shape[0]:
        raise ValueError(f"vertex mismatch {preds.shape} vs {labels.shape}")
    cols = []
    for k in range(1, n_labels + 1):
        m = labels == k
        cols.append(preds[:, m].mean(1) if m.any() else np.zeros(preds.shape[0], np.float32))
    if add_gfp:
        cols.append(np.sqrt((preds.astype(np.float64) ** 2).mean(1)).astype(np.float32))
    return np.stack(cols, axis=1)

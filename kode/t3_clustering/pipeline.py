"""Tugas 3: antarmuka clustering yang dipakai bersama (bagian Orang A → dipakai Orang B).

fit_predict(df, seed, method, k, cfg) → (labels, model, X)
predict(X, model) → labels
"""
import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from .preprocess import build_features, DEFAULT_WEIGHTS
from ..common.paths import KODE, SEED

N_SAMPLE_AGG = 15000   # metode O(n²) dijalankan di sampel, sisanya di-assign


def load_config():
    p = KODE / "configs" / "t3_terpilih.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"use_noise": False, "weights": "kelompok", "method": "kmeans", "k": 8}


def features(df, cfg):
    return build_features(df, use_noise=cfg.get("use_noise", False),
                          weights=DEFAULT_WEIGHTS if cfg.get("weights", "kelompok") == "kelompok" else None)


def fit_method(X, method, k, seed=SEED):
    Xv = X.values if hasattr(X, "values") else X
    if method == "kmeans":
        m = KMeans(k, n_init=5, random_state=seed).fit(Xv)
        return m.labels_, m
    if method == "gmm":
        rng = np.random.RandomState(seed); idx = rng.choice(len(Xv), min(40000, len(Xv)), replace=False)
        m = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, n_init=2, random_state=seed).fit(Xv[idx])
        return m.predict(Xv), m
    if method in ("agglomerative", "hdbscan"):
        rng = np.random.RandomState(seed); idx = rng.choice(len(Xv), min(N_SAMPLE_AGG, len(Xv)), replace=False)
        if method == "agglomerative":
            lab_s = AgglomerativeClustering(k, linkage="ward").fit_predict(Xv[idx])
            m = NearestCentroid().fit(Xv[idx], lab_s)
            return m.predict(Xv), m
        lab_s = HDBSCAN(min_cluster_size=150, min_samples=20).fit_predict(Xv[idx])
        ok = lab_s >= 0
        m = KNeighborsClassifier(15).fit(Xv[idx][ok], lab_s[ok])
        lab = m.predict(Xv)
        # titik yang jauh dari semua cluster dianggap noise (-1), seperti HDBSCAN
        d, _ = m.kneighbors(Xv, n_neighbors=1)
        thr = np.quantile(m.kneighbors(Xv[idx][ok], n_neighbors=2)[0][:, 1], 0.99)
        lab[d[:, 0] > thr] = -1
        m.noise_rate_sample_ = float((~ok).mean())
        return lab, m
    raise ValueError(method)


def fit_predict(df, seed=SEED, method=None, k=None, cfg=None):
    cfg = cfg or load_config()
    X, _, meta = features(df, cfg)
    lab, m = fit_method(X, method or cfg["method"], k or cfg["k"], seed)
    return lab, m, X


def predict(X, model):
    Xv = X.values if hasattr(X, "values") else X
    return model.predict(Xv)

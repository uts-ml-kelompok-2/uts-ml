"""Tugas 3 (g): pelanggan yang tidak cocok dengan segmen mana pun → -1.

Prinsip konservatif: anomali referensi tidak dihitung di ARI, tetapi pelanggan normal yang kita beri -1
menurunkan ARI. Jadi -1 hanya untuk kasus yang jelas: perilaku mustahil/tidak wajar + cluster berprofil bot."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from ..common.paths import SEED


def domain_rule(raw: pd.DataFrame) -> np.ndarray:
    """Perilaku yang tidak mungkin/tidak wajar untuk manusia: > 100 sesi/minggu (±14 sesi per hari),
    rasio konversi > 1 (pesanan lebih banyak dari kunjungan), belanja 30 hari > belanja 90 hari."""
    return ((raw.sesi_per_minggu > 100) | (raw.rasio_konversi > 1) | (raw.belanja_30h > raw.belanja_90h + 1)).values


def suspicious_clusters(labels, flag, thr=0.5):
    """Cluster yang mayoritas anggotanya (> thr) terkena aturan domain = cluster anomali (mis. bot)."""
    s = pd.Series(flag).groupby(labels).mean()
    return [int(c) for c, v in s.items() if v > thr]


def compare_methods(X, raw, labels, model, rate=0.02):
    """Bandingkan beberapa detektor anomali (untuk laporan)."""
    Xv = X.values
    out = {}
    out["aturan domain"] = domain_rule(raw)
    sus = suspicious_clusters(labels, out["aturan domain"])
    out["cluster berprofil bot"] = np.isin(labels, sus)
    iso = IsolationForest(n_estimators=200, contamination=rate, random_state=SEED).fit(Xv)
    out[f"Isolation Forest (top {rate*100:.0f}%)"] = iso.predict(Xv) == -1
    idx = np.random.RandomState(SEED).choice(len(Xv), min(30000, len(Xv)), replace=False)
    g = GaussianMixture(int(labels.max()) + 1, covariance_type="diag", reg_covar=1e-3, random_state=SEED).fit(Xv[idx])
    ll = g.score_samples(Xv); out[f"log-likelihood GMM terendah {rate*100:.0f}%"] = ll < np.quantile(ll, rate)
    if hasattr(model, "cluster_centers_"):
        d = np.linalg.norm(Xv - model.cluster_centers_[labels], axis=1)
        out[f"jarak ke centroid tertinggi {rate*100:.0f}%"] = d > np.quantile(d, 1 - rate)
    return out, sus

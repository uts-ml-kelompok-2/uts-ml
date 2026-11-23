"""Tugas 1 (e): kalibrasi batas atas P90 dengan Conformalized Quantile Regression (CQR) satu sisi."""
import numpy as np
import pandas as pd


class CQRUpper:
    """Skor konformitas s = y − P90. Koreksi = kuantil-(1−α) dari s pada set kalibrasi (dengan koreksi sampel hingga).
    P90 baru = P90 + koreksi. Mode 'mondrian' menghitung koreksi terpisah per grup (mis. per layanan)."""

    def __init__(self, alpha=0.10, group_col=None):
        self.alpha, self.group_col = alpha, group_col

    def _q(self, s):
        s = np.sort(np.asarray(s)); n = len(s)
        k = min(n - 1, int(np.ceil((n + 1) * (1 - self.alpha))) - 1)
        return float(s[max(k, 0)])

    def fit(self, y, p90, groups=None):
        y = np.asarray(y, float); ok = ~np.isnan(y)
        s = y[ok] - np.asarray(p90)[ok]
        self.global_ = self._q(s)
        self.group_ = {}
        if self.group_col is not None and groups is not None:
            g = np.asarray(groups)[ok]
            for k in np.unique(g):
                if (g == k).sum() >= 200:
                    self.group_[k] = self._q(s[g == k])
        return self

    def adjust(self, p90, groups=None):
        p90 = np.asarray(p90, float)
        if not self.group_ or groups is None:
            return p90 + self.global_
        adj = pd.Series(np.asarray(groups)).map(self.group_).fillna(self.global_).values
        return p90 + adj


MONTHS = [f"2025-{m:02d}" for m in range(1, 9)]


def pooled_scores(d, cfg, last_month: str, first_train_end: str = "2025-02"):
    """Kumpulkan skor konformitas out-of-time dari beberapa jendela bergulir.
    Untuk setiap bulan akhir latih m (mulai Feb): latih model P90 pada Jan..m, prediksi 2 bulan berikutnya
    (horizon sama dengan test: Sep–Okt diprediksi dari data s.d. Agu). Hanya jendela yang selesai <= last_month.
    Mengembalikan DataFrame: y, p90, layanan, bulan_prediksi, horizon."""
    import lightgbm as lgb
    from .common_t1 import make_xy_cfg
    from .models import LGB_BASE
    from ..common.splits import month_mask
    rows = []
    ends = [m for m in MONTHS if first_train_end <= m]
    for m in ends:
        i = MONTHS.index(m)
        if i + 2 >= len(MONTHS) or MONTHS[i + 2] > last_month:
            continue
        nxt = MONTHS[i + 1:i + 3]
        a = month_mask(d.tgl_pickup, "2025-01", m); b = month_mask(d.tgl_pickup, nxt[0], nxt[-1])
        Xa, Xb, _ = make_xy_cfg(d[a], d[b], cfg)
        ok = d.loc[a, "y"].notna().values
        m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **{**LGB_BASE, **cfg["params"]}).fit(Xa[ok], d.loc[a, "y"].values[ok])
        db = d[b]
        rows.append(pd.DataFrame({"y": db["y_eval"].values, "p90": m90.predict(Xb), "layanan": db["layanan"].values,
                                  "bulan": db["bulan"].values, "akhir_latih": m}))
    return pd.concat(rows, ignore_index=True)

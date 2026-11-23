"""Uji anti-kebocoran fitur perilaku Tugas 2:
fitur transaksi sebelum waktu T harus IDENTIK, baik dihitung dengan seluruh data maupun hanya dengan data s.d. T.
Kalau ada fitur yang memakai informasi masa depan, hasilnya akan berbeda."""
import numpy as np
import pandas as pd
from kode.t2_fraud.features import build, ALL_BEHAV as BEHAV_COLS


def _toy(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    t0 = pd.Timestamp("2025-01-01")
    return pd.DataFrame({
        "id_transaksi": [f"T{i:06d}" for i in range(n)],
        "waktu": t0 + pd.to_timedelta(np.sort(rng.randint(0, 200 * 24 * 60, n)), unit="m"),
        "id_pengguna": rng.choice([f"U{i}" for i in range(150)], n),
        "id_merchant": rng.choice([f"M{i}" for i in range(40)], n),
        "kategori_merchant": rng.choice(list("abcd"), n),
        "nominal": rng.lognormal(11, 1, n).round(-2), "jam": rng.randint(0, 24, n),
        "is_fraud": (rng.rand(n) < 0.05).astype(float)})


def test_no_future_leak():
    d = _toy()
    T = d["waktu"].quantile(0.6)
    full = build(d, d.iloc[:0], prior=0.05).set_index("id_transaksi")
    trunc = build(d[d.waktu <= T], d.iloc[:0], prior=0.05).set_index("id_transaksi")
    ids = trunc.index
    for c in BEHAV_COLS:
        a, b = full.loc[ids, c].values, trunc.loc[ids, c].values
        assert np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)), f"fitur {c} memakai informasi masa depan"


def test_future_labels_do_not_change_past():
    d = _toy(seed=1)
    T = d["waktu"].quantile(0.5)
    d2 = d.copy(); d2.loc[d2.waktu > T, "is_fraud"] = 1.0       # ubah semua label masa depan
    a = build(d, d.iloc[:0], prior=0.05).set_index("id_transaksi")
    b = build(d2, d2.iloc[:0], prior=0.05).set_index("id_transaksi")
    ids = d.loc[d.waktu <= T, "id_transaksi"]
    for c in ["m_fraud_rate_matang", "u_fraud_rate_matang", "k_fraud_rate_matang"]:
        assert np.allclose(a.loc[ids, c], b.loc[ids, c]), f"{c} terpengaruh label masa depan"


def test_manual_recount():
    d = _toy(seed=2)
    f = build(d, d.iloc[:0], prior=0.05).set_index("id_transaksi")
    for i in [100, 1500, 2900]:
        row = d.iloc[i]
        prev = d[(d.id_pengguna == row.id_pengguna) & (d.waktu < row.waktu)]
        assert f.loc[row.id_transaksi, "u_n_trx_prev"] >= len(prev)  # >= karena transaksi pada detik yang sama
        win = prev[prev.waktu > row.waktu - pd.Timedelta("24h")]
        assert abs(f.loc[row.id_transaksi, "u_trx_24j_sendiri"] - len(win)) <= 1

"""Tugas 2 (c): fitur perilaku pengguna & merchant dari riwayat SEBELUM transaksi.

Aturan anti-kebocoran:
1. Train dan test digabung lalu diurutkan menurut waktu. Fitur transaksi t hanya memakai baris dengan waktu < t
   (shift / rolling yang tidak memasukkan baris saat ini). Transaksi test boleh memakai riwayat transaksi train.
2. Fitur berbasis LABEL (tingkat fraud historis) hanya memakai label yang sudah 'matang': transaksi yang terjadi
   > LABEL_LAG_HARI sebelum t. Label test tidak ada, jadi otomatis tidak terpakai.
"""
import numpy as np
import pandas as pd

LABEL_LAG_HARI = 30
PRIOR_STRENGTH = 50


def rolling_count(df, key, window):
    d = df[["waktu", key]].copy()
    d["_pos"] = np.arange(len(d))
    d = d.sort_values([key, "waktu", "_pos"], kind="stable")
    d["_one"] = 1.0
    r = d.groupby(key, sort=False).rolling(window, on="waktu")["_one"].sum().values
    out = np.empty(len(d)); out[d["_pos"].values] = r - 1.0      # kurangi transaksi ini sendiri
    return out


def _expanding_prev(df, key, col):
    g = df.groupby(key, sort=False)[col]
    n = g.cumcount().values.astype(float)
    csum = g.cumsum().values - df[col].values
    csq = (df[col] ** 2).groupby(df[key], sort=False).cumsum().values - df[col].values ** 2
    mean = np.where(n > 0, csum / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, (csq - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    return n, mean, np.sqrt(np.clip(var, 0, None))


def _mature_rate(df, key, label_col, prior):
    """Tingkat fraud historis per `key` dari label yang sudah matang (waktu transaksi < t − LABEL_LAG_HARI)."""
    lab = df.loc[df[label_col].notna(), ["waktu", key, label_col]].copy()
    lab["t_tersedia"] = lab["waktu"] + pd.Timedelta(days=LABEL_LAG_HARI)
    lab = lab.sort_values("t_tersedia")
    lab["cum_f"] = lab.groupby(key)[label_col].cumsum()
    lab["cum_n"] = lab.groupby(key).cumcount() + 1
    left = df[["waktu", key]].copy(); left["_pos"] = np.arange(len(df))
    left = left.sort_values("waktu")
    m = pd.merge_asof(left, lab[["t_tersedia", key, "cum_f", "cum_n"]], left_on="waktu", right_on="t_tersedia",
                      by=key, direction="backward", allow_exact_matches=False)
    f = m["cum_f"].fillna(0).values; n = m["cum_n"].fillna(0).values
    rate = (f + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH)
    out_rate = np.empty(len(df)); out_n = np.empty(len(df)); out_f = np.empty(len(df))
    out_rate[m["_pos"].values] = rate; out_n[m["_pos"].values] = n; out_f[m["_pos"].values] = f
    return out_rate, out_n, out_f


def build(train: pd.DataFrame, test: pd.DataFrame, label_col="is_fraud", prior: float | None = None) -> pd.DataFrame:
    """Kembalikan DataFrame gabungan (train+test, urut waktu) dengan fitur perilaku. Kolom 'set' = train/test."""
    a = train.copy(); a["set"] = "train"
    b = test.copy(); b["set"] = "test"; b[label_col] = np.nan
    df = pd.concat([a, b], ignore_index=True).sort_values(["waktu", "id_transaksi"], kind="stable").reset_index(drop=True)
    prior = float(train[label_col].mean()) if prior is None else prior
    df["log_nom"] = np.log1p(df["nominal"])

    # --- pengguna
    n, mean, std = _expanding_prev(df, "id_pengguna", "log_nom")
    df["u_n_trx_prev"] = n
    df["u_nom_mean_prev"] = np.expm1(mean)
    df["u_nom_z"] = (df["log_nom"] - mean) / (std + 0.25)   # z-score log-nominal vs riwayat pengguna
    df["u_nom_rasio_maks"] = df["nominal"] / (df.groupby("id_pengguna")["nominal"].transform(lambda s: s.shift().cummax()) + 1.0)
    first_pair = ~df.duplicated(["id_pengguna", "id_merchant"])
    df["u_merchant_baru"] = first_pair.astype(int)
    df["u_n_merchant_prev"] = first_pair.astype(int).groupby(df["id_pengguna"]).cumsum().values - df["u_merchant_baru"].values
    first_cat = ~df.duplicated(["id_pengguna", "kategori_merchant"])
    df["u_kategori_baru"] = first_cat.astype(int)
    df["u_trx_1j_sendiri"] = rolling_count(df, "id_pengguna", "1h")
    df["u_trx_24j_sendiri"] = rolling_count(df, "id_pengguna", "24h")
    df["u_jeda_menit_sendiri"] = df.groupby("id_pengguna")["waktu"].diff().dt.total_seconds().values / 60.0
    df["u_malam"] = df["jam"].isin([0, 1, 2, 3, 4]).astype(int)
    df["u_rasio_malam_prev"] = (df.groupby("id_pengguna")["u_malam"].cumsum().values - df["u_malam"].values) / np.maximum(n, 1)

    # --- merchant
    n_m, mean_m, _ = _expanding_prev(df, "id_merchant", "nominal")
    df["m_n_trx_prev"] = n_m
    df["m_nom_rasio"] = df["nominal"] / (mean_m + 1.0)
    df["m_trx_24j"] = rolling_count(df, "id_merchant", "24h")
    df["m_trx_7h"] = rolling_count(df, "id_merchant", "7D")
    df["m_pengguna_baru"] = (~df.duplicated(["id_merchant", "id_pengguna"])).astype(int)

    # --- tingkat fraud historis dari label matang
    for key, nm in [("id_merchant", "m"), ("id_pengguna", "u"), ("kategori_merchant", "k")]:
        r, nn, ff = _mature_rate(df, key, label_col, prior)
        df[f"{nm}_fraud_rate_matang"] = r
        if nm != "k":
            df[f"{nm}_n_fraud_matang"] = ff
    return df


BEHAV_COLS = ["u_nom_mean_prev", "u_nom_z", "u_nom_rasio_maks", "u_merchant_baru", "u_kategori_baru",
              "u_trx_1j_sendiri", "u_trx_24j_sendiri", "u_jeda_menit_sendiri", "u_rasio_malam_prev",
              "m_nom_rasio", "m_trx_24j", "m_trx_7h", "m_pengguna_baru", "m_fraud_rate_matang", "u_fraud_rate_matang"]
# Fitur kumulatif sejak awal data: nilainya naik terus mengikuti waktu sehingga menjadi penanda waktu
# (terdeteksi lewat adversarial validation di experiments.py butir e). Dihitung tapi TIDAK dipakai model.
DRIFT_COLS = ["u_n_trx_prev", "u_n_merchant_prev", "m_n_trx_prev", "m_n_fraud_matang", "u_n_fraud_matang", "k_fraud_rate_matang"]
ALL_BEHAV = BEHAV_COLS + DRIFT_COLS

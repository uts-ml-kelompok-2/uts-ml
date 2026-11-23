"""Tugas 3 (a): preprocessing data pelanggan untuk clustering.

- Variabel skewed (uang, hitungan, recency, aktivitas) → log1p.
- Data komposisional (16 porsi kategori, 4 porsi jam; jumlahnya = 1) → transformasi CLR (centered log-ratio)
  dengan epsilon kecil. Alasan: porsi saling bergantung (naik satu = turun yang lain), jarak Euclidean pada porsi
  mentah tidak bermakna; CLR memetakan simpleks ke ruang Euclidean.
- Kategorikal: kota_tier (ordinal 1–3, dipakai apa adanya), perangkat (one-hot).
- Missing rata2_keranjang: struktural (frekuensi_90h = 0) → 0 + indikator tidak_aktif_90h.
- Outlier: winsorize persentil 0.5–99.5 lalu RobustScaler. Skor outlier & inkonsistensi disimpan di meta untuk tahap anomali.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from ..common.paths import data_file, QUICK, SEED
from ..common.io_utils import read_cache, cache_parquet

KAT = ["porsi_fashion", "porsi_elektronik", "porsi_kecantikan", "porsi_rumah_tangga", "porsi_makanan_minuman", "porsi_ibu_bayi",
       "porsi_olahraga", "porsi_hobi", "porsi_otomotif", "porsi_kesehatan", "porsi_buku_alat_tulis", "porsi_voucher_digital",
       "porsi_gadget_aksesoris", "porsi_perabot", "porsi_pertanian", "porsi_lainnya"]
JAM = ["porsi_jam_pagi", "porsi_jam_siang", "porsi_jam_sore", "porsi_jam_malam"]
SKEW = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan", "retur_90h", "ulasan_90h",
        "sesi_per_minggu", "durasi_sesi_menit", "jumlah_alamat", "wishlist", "klik_iklan_30h", "belanja_30h", "frekuensi_30h",
        "lama_member_bulan"]
PERSEN = ["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod"]
LINEAR = ["rating_beri_rata2", "rasio_konversi", "umur", "kota_tier"]
NOISE = [f"n{i:02d}" for i in range(1, 16)]
EPS = 1e-3

GROUPS = {"RFM & nilai": ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan", "belanja_30h", "frekuensi_30h", "tidak_aktif_90h"],
          "kategori (CLR)": [f"clr_{c}" for c in KAT], "waktu belanja (CLR)": [f"clr_{c}" for c in JAM],
          "promo & pembayaran": PERSEN, "perilaku pasca-beli": ["retur_90h", "rating_beri_rata2", "ulasan_90h"],
          "aktivitas aplikasi": ["sesi_per_minggu", "durasi_sesi_menit", "rasio_konversi", "wishlist", "klik_iklan_30h"],
          "profil": ["lama_member_bulan", "umur", "kota_tier", "jumlah_alamat", "perangkat_android", "perangkat_ios", "perangkat_web"],
          "atribut CRM anonim (n01–n15)": NOISE}


def load_raw():
    df = read_cache("t3_raw")
    if df is None:
        df = pd.read_csv(data_file("clustering.csv.gz"))
        cache_parquet(df, "t3_raw")
    if QUICK:
        df = df.sample(frac=0.25, random_state=SEED).reset_index(drop=True)
    return df


def clr(P: pd.DataFrame) -> pd.DataFrame:
    L = np.log(P.values + EPS)
    return pd.DataFrame(L - L.mean(axis=1, keepdims=True), columns=[f"clr_{c}" for c in P.columns], index=P.index)


def build_features(df: pd.DataFrame, use_noise: bool = False, weights: dict | None = None):
    """Kembalikan (X, nama_fitur, meta). X sudah diskalakan; `weights` = bobot per kelompok (GROUPS)."""
    d = df.copy()
    meta = pd.DataFrame(index=d.index)
    meta["tidak_aktif_90h"] = (d["frekuensi_90h"] == 0).astype(int)
    meta["inkonsisten_konversi"] = (d["rasio_konversi"] > 1).astype(int)
    meta["inkonsisten_belanja"] = (d["belanja_30h"] > d["belanja_90h"] + 1).astype(int)
    d["tidak_aktif_90h"] = meta["tidak_aktif_90h"]
    d["rata2_keranjang"] = d["rata2_keranjang"].fillna(0)
    d["rasio_konversi"] = d["rasio_konversi"].clip(upper=1.0)
    for c in SKEW:
        d[c] = np.log1p(d[c].clip(lower=0))
    feats = pd.concat([d[SKEW + PERSEN + LINEAR + ["tidak_aktif_90h"]], clr(d[KAT]), clr(d[JAM]),
                       pd.get_dummies(d["perangkat"], prefix="perangkat").astype(float)], axis=1)
    if use_noise:
        feats = pd.concat([feats, d[NOISE].apply(lambda s: np.log1p(s.clip(lower=0)) if s.skew() > 3 else s)], axis=1)
    # skor outlier mentah (sebelum winsorize): jumlah fitur yang berada di luar persentil 0.5–99.5
    lo, hi = feats.quantile(0.005), feats.quantile(0.995)
    meta["n_fitur_ekstrem"] = ((feats < lo) | (feats > hi)).sum(axis=1)
    feats = feats.clip(lower=lo, upper=hi, axis=1)
    X = RobustScaler(quantile_range=(10, 90)).fit_transform(feats)
    X = pd.DataFrame(X, columns=feats.columns, index=feats.index)
    # bobot kelompok: tiap kelompok berkontribusi seimbang ke jarak (bobot / sqrt(jumlah kolom))
    if weights is not None:
        for g, cols in GROUPS.items():
            cc = [c for c in cols if c in X.columns]
            if cc:
                X[cc] = X[cc] * (weights.get(g, 1.0) / np.sqrt(len(cc)))
    return X.astype("float32"), list(X.columns), meta


DEFAULT_WEIGHTS = {g: 1.0 for g in GROUPS}

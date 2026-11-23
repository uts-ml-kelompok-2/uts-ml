"""Tugas 1: muat dan bersihkan data regresi waktu kirim."""
import numpy as np
import pandas as pd
from ..common.paths import data_file, QUICK, SEED
from ..common.io_utils import read_cache, cache_parquet

CAT_COLS = ["layanan", "mitra_kurir", "hub_asal", "kota_asal", "pulau_tujuan", "kota_tujuan"]
TARGET = "durasi_jam"
# Fitur yang TIDAK dipakai, alasan lengkap di audit.py dan configs/t1.yaml
LEAK_COLS = ["jumlah_scan"]       # bertambah selama pengiriman (tidak tersedia saat pickup)
TIME_PROXY_COLS = ["f30"]         # naik lurus mengikuti waktu → hanya penanda waktu, tidak bisa diekstrapolasi
DROP_ALWAYS = ["id_paket", "tgl_pickup", TARGET, "bulan"]
UNIT_ERROR_THRESHOLD = 500        # durasi > 500 jam dianggap tercatat dalam menit


def load_raw():
    tr, te = read_cache("t1_train_raw"), read_cache("t1_test_raw")
    if tr is None:
        tr = pd.read_csv(data_file("regresi_train.csv.gz"))
        te = pd.read_csv(data_file("regresi_test.csv.gz"))
        cache_parquet(tr, "t1_train_raw"); cache_parquet(te, "t1_test_raw")
    for d in (tr, te):
        d["tgl_pickup"] = pd.to_datetime(d["tgl_pickup"])
        d["bulan"] = d["tgl_pickup"].dt.to_period("M").astype(str)
    if QUICK:
        tr = tr.sample(frac=0.2, random_state=SEED).sort_values("tgl_pickup").reset_index(drop=True)
    return tr, te


def fix_target(y: pd.Series, mode: str = "bagi60") -> pd.Series:
    """Perlakuan durasi ekstrem (> 500 jam). mode: bagi60 | buang (NaN) | biarkan."""
    y = y.astype(float).copy()
    bad = y > UNIT_ERROR_THRESHOLD
    if mode == "bagi60":
        y[bad] = y[bad] / 60.0
    elif mode == "buang":
        y[bad] = np.nan
    return y


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur turunan yang tersedia saat pickup."""
    d = df.copy()
    d["hari_minggu"] = d["tgl_pickup"].dt.dayofweek
    d["jarak_tidak_valid"] = ((d["jarak_km"] >= 9999) | (d["jarak_km"] <= 0)).astype(int)
    d.loc[d["jarak_tidak_valid"] == 1, "jarak_km"] = np.nan
    d["berat_hilang"] = d["berat_kg"].isna().astype(int)
    d["densitas"] = d["berat_kg"] / d["volume_liter"].clip(lower=0.1)
    d["nilai_per_item"] = d["nilai_barang"] / d["jumlah_item"].clip(lower=1)
    d["rute"] = d["kota_asal"].astype(str) + ">" + d["kota_tujuan"].astype(str)
    return d


def feature_columns(df: pd.DataFrame, drop_extra=(), keep=()) -> list[str]:
    """Kolom fitur. `keep` dipakai HANYA di audit untuk mengukur dampak memasukkan fitur bocor."""
    drop = (set(DROP_ALWAYS) | set(LEAK_COLS) | set(TIME_PROXY_COLS) | set(drop_extra) | {"rute"}) - set(keep)
    return [c for c in df.columns if c not in drop]

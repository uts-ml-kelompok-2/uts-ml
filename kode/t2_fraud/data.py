"""Tugas 2: muat data transaksi."""
import pandas as pd
from ..common.paths import data_file, QUICK, SEED
from ..common.io_utils import read_cache, cache_parquet

CAT_COLS = ["kategori_merchant", "kanal", "tipe_perangkat", "email_domain"]
ID_COLS = ["id_transaksi", "id_pengguna", "id_merchant"]
TARGET = "is_fraud"
LEAK_COLS = ["flag_review_manual"]   # review manual terjadi SETELAH transaksi dicurigai → tidak tersedia saat transaksi
TIME_PROXY_COLS = ["v15"]            # naik lurus mengikuti waktu → hanya penanda waktu
REVIEW_COST = 25_000


def load_raw():
    tr, te = read_cache("t2_train_raw"), read_cache("t2_test_raw")
    if tr is None:
        tr = pd.read_csv(data_file("klasifikasi_train.csv.gz"))
        te = pd.read_csv(data_file("klasifikasi_test.csv.gz"))
        cache_parquet(tr, "t2_train_raw"); cache_parquet(te, "t2_test_raw")
    for d in (tr, te):
        d["waktu"] = pd.to_datetime(d["waktu"])
        d["bulan"] = d["waktu"].dt.to_period("M").astype(str)
    if QUICK:
        # sampel per PENGGUNA (bukan per baris) supaya riwayat perilaku tetap utuh
        users = tr["id_pengguna"].drop_duplicates().sample(frac=0.25, random_state=SEED)
        tr = tr[tr["id_pengguna"].isin(users)]
        te = te[te["id_pengguna"].isin(set(users)) | (te.index % 4 == 0)]
    tr = tr.sort_values(["waktu", "id_transaksi"]).reset_index(drop=True)
    te = te.sort_values(["waktu", "id_transaksi"]).reset_index(drop=True)
    return tr, te

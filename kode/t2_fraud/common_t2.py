"""Helper Tugas 2: fold waktu (validasi hanya di periode label matang) + cache fitur."""
import numpy as np
import pandas as pd
from .data import load_raw
from .features import build
from ..common.io_utils import read_cache, cache_parquet
from ..common.splits import month_mask
from ..common.paths import QUICK

# latih → kalibrasi (1 bulan) → validasi (2 bulan). Semua periode validasi labelnya sudah matang (< Agu).
FOLDS = [
    {"nama": "F1", "latih": ("2025-01", "2025-03"), "kal": ("2025-04", "2025-04"), "val": ("2025-05", "2025-06")},
    {"nama": "F2", "latih": ("2025-01", "2025-04"), "kal": ("2025-05", "2025-05"), "val": ("2025-06", "2025-07")},
]


def get_features():
    name = "t2_fitur_quick" if QUICK else "t2_fitur"
    df = read_cache(name)
    if df is None:
        tr, te = load_raw()
        df = build(tr, te)
        cache_parquet(df, name)
    df["waktu"] = pd.to_datetime(df["waktu"])
    return df


def masks(df, fold):
    tr = df["set"].eq("train")
    return (tr & month_mask(df.waktu, *fold["latih"]), tr & month_mask(df.waktu, *fold["kal"]), tr & month_mask(df.waktu, *fold["val"]))


def weekly_rate(df):
    t = df[df.set == "train"].set_index("waktu")["is_fraud"]
    return t.resample("W-SUN").agg(["mean", "size", "sum"])


def estimate_mature_cutoff(df, ratio=0.8):
    """Tanggal mulai label dianggap belum matang: minggu pertama (dari belakang berturut-turut) yang tingkat fraudnya
    < ratio × median tingkat fraud mingguan Jan–Jul."""
    w = weekly_rate(df)
    base = w.loc[w.index < "2025-08-01", "mean"].median()
    low = w["mean"] < ratio * base
    cut = None
    for ts in reversed(w.index):
        if low.loc[ts]:
            cut = ts
        else:
            break
    if cut is None:
        return None, base
    return (cut - pd.Timedelta(days=6)).normalize(), base

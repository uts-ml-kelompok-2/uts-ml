"""Helper bersama Tugas 1: susun X/y per fold dan definisi fold waktu."""
import numpy as np
from .data import add_features, feature_columns, fix_target
from .models import CategoryEncoder
from ..common.splits import month_mask

# Fold validasi waktu: latih s.d. bulan M, validasi 2 bulan berikutnya (meniru prediksi Sep–Okt dari data s.d. Agu)
TIME_FOLDS = [
    {"nama": "F1", "latih": ("2025-01", "2025-04"), "val": ("2025-05", "2025-06")},
    {"nama": "F2", "latih": ("2025-01", "2025-06"), "val": ("2025-07", "2025-08")},
]


def prepare(train_raw, target_mode="bagi60"):
    d = add_features(train_raw)
    d["y"] = fix_target(d["durasi_jam"], target_mode)
    d["y_eval"] = fix_target(d["durasi_jam"], "bagi60")   # label evaluasi: kesalahan satuan sudah dikoreksi
    return d


def fold_masks(d, fold):
    return month_mask(d["tgl_pickup"], *fold["latih"]), month_mask(d["tgl_pickup"], *fold["val"])


def make_xy(tr, va, high_card="te", drop_extra=(), min_count=30, keep=(), unknown_frac=0.05):
    cols = feature_columns(tr, drop_extra=list(drop_extra) + ["y", "y_eval"], keep=keep)
    enc = CategoryEncoder(high_card=high_card, min_count=min_count, unknown_frac=unknown_frac).fit(tr[cols], tr["y"].fillna(tr["y"].median()))
    Xtr = enc.transform(tr[cols], y=tr["y"].fillna(tr["y"].median()), oof=True)
    Xva = enc.transform(va[cols])
    return Xtr, Xva, enc


def load_config():
    import json
    from ..common.paths import KODE
    p = KODE / "configs" / "t1_terpilih.json"
    if p.exists():
        return json.loads(p.read_text())
    from .models import LGB_BASE
    return {"params": {}, "high_card": "te", "kalibrasi": "CQR global", "unknown_frac": 0.05, "drop_extra": []}


def make_xy_cfg(tr, va, cfg):
    return make_xy(tr, va, high_card=cfg["high_card"], unknown_frac=cfg.get("unknown_frac", 0.05),
                   drop_extra=cfg.get("drop_extra", []))

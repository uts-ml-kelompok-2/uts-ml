"""Validasi format 3 file prediksi sesuai soal (nama kolom, jumlah baris, ID, rentang nilai)."""
import sys
import numpy as np
import pandas as pd
from .paths import ROOT, data_file

SPEC = {
    "prediksi_regresi.csv": (["id_paket", "p50", "p90"], "regresi_test.csv.gz", "id_paket"),
    "prediksi_klasifikasi.csv": (["id_transaksi", "prob", "tandai"], "klasifikasi_test.csv.gz", "id_transaksi"),
    "cluster.csv": (["id_pelanggan", "cluster"], "clustering.csv.gz", "id_pelanggan"),
}


def check(strict=True) -> bool:
    ok_all = True
    for fn, (cols, ref, key) in SPEC.items():
        p = ROOT / fn
        errs = []
        if not p.exists():
            print(f"[GAGAL] {fn}: file tidak ada"); ok_all = False; continue
        d = pd.read_csv(p)
        if list(d.columns) != cols:
            errs.append(f"kolom {list(d.columns)} ≠ {cols}")
        ids = pd.read_csv(data_file(ref), usecols=[key])[key]
        if len(d) != len(ids):
            errs.append(f"jumlah baris {len(d)} ≠ {len(ids)}")
        if d[key].duplicated().any():
            errs.append("ada ID duplikat")
        if set(d[key]) != set(ids):
            errs.append(f"ID tidak cocok dengan {ref} (hilang {len(set(ids) - set(d[key]))}, asing {len(set(d[key]) - set(ids))})")
        if d.isna().any().any():
            errs.append("ada nilai kosong")
        if fn == "prediksi_regresi.csv" and not errs:
            if (d.p90 < d.p50).any(): errs.append("ada p90 < p50")
            if (d.p50 < 0).any(): errs.append("ada p50 negatif")
        if fn == "prediksi_klasifikasi.csv" and not errs:
            if not d.prob.between(0, 1).all(): errs.append("prob di luar [0,1]")
            if not d.tandai.isin([0, 1]).all(): errs.append("tandai bukan 0/1")
        if fn == "cluster.csv" and not errs:
            if not np.issubdtype(d.cluster.dtype, np.integer): errs.append("cluster bukan integer")
        if errs:
            ok_all = False
            print(f"[GAGAL] {fn}: " + "; ".join(errs))
        else:
            print(f"[LULUS] {fn}: {len(d)} baris")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if check() else 1)

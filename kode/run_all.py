"""Satu perintah untuk mereproduksi SEMUA angka, gambar, tabel, dan file prediksi.

  python kode/run_all.py --task all --mode train        # jalankan semua eksperimen + model final (±1 jam)
  python kode/run_all.py --task t2  --mode reproduce    # muat model tersimpan, hitung ulang angka kunci & CSV (< 5 menit)
  python kode/run_all.py --task all --mode train --quick  # uji cepat pada subset data
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2", "t3", "all"], default="all")
    ap.add_argument("--mode", choices=["train", "reproduce"], default="train")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        os.environ["UTS_QUICK"] = "1"
    from kode.common.seed import set_seed
    from kode.common.paths import SEED
    set_seed(SEED)
    tasks = ["t1", "t2", "t3"] if a.task == "all" else [a.task]
    durasi = {}
    for t in tasks:
        t0 = time.time()
        print(f"\n================ {t.upper()} ({a.mode}) ================", flush=True)
        if t == "t1":
            from kode.t1_regresi import audit, experiments, analysis, final
            if a.mode == "train":
                audit.run(); experiments.run(); analysis.run(); final.train()
            else:
                final.reproduce()
        elif t == "t2":
            from kode.t2_fraud import audit, experiments, analysis, final
            if a.mode == "train":
                audit.run(); experiments.run(); analysis.run(); final.train()
            else:
                final.reproduce()
        elif t == "t3":
            from kode.t3_clustering import feature_select, select_k, stability, final
            if a.mode == "train":
                feature_select.run(); select_k.run(); stability.run(); final.train()
            else:
                final.reproduce()
        durasi[t] = round((time.time() - t0) / 60, 1)
    from kode.common import make_tables, validate_submission
    if a.mode == "train":
        make_tables.run()
    print("\n================ VALIDASI FILE PREDIKSI ================")
    ok = validate_submission.check()
    if a.quick:
        print("(mode --quick: jumlah baris memang tidak lengkap, validasi format saja yang relevan)")
    print("\nDurasi (menit):", durasi)
    return 0 if (ok or a.quick) else 1


if __name__ == "__main__":
    sys.exit(main())

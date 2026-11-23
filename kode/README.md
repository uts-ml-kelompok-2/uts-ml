# UTS Machine Learning — Kode

Tiga tugas: regresi waktu kirim (P50/P90), deteksi fraud (probabilitas + keputusan berbasis biaya), dan segmentasi pelanggan.
Semua model klasik (scikit-learn, LightGBM). Tidak ada data eksternal.

## Struktur
```
kode/
├── run_all.py               ← satu perintah untuk semua angka, gambar, tabel, dan file prediksi
├── requirements.txt
├── common/                  ← paths, metrik (pinball, coverage, PR-AUC, biaya, ARI), split waktu, I/O hasil,
│                              validate_submission.py, make_tables.py (→ outputs/tables/angka.tex)
├── t1_regresi/              ← data, audit, experiments (validasi/model/encoding/CQR), analysis (SHAP), final
├── t2_fraud/                ← data, features (perilaku tanpa kebocoran), audit, experiments, analysis, final
├── t3_clustering/           ← preprocess, feature_select, pipeline, select_k, stability, anomalies, profiles, final
├── tests/                   ← tes metrik + tes anti-kebocoran fitur Tugas 2
├── configs/                 ← konfigurasi terpilih (ditulis otomatis oleh experiments/select_k)
└── outputs/
    ├── results/*.json       ← SEMUA angka laporan (+ versi library, seed, waktu)
    ├── figures/*.png  tables/*.tex (+ angka.tex)  models/*.joblib
    └── cache/ logs/         ← tidak di-commit
```

## Data
Taruh 6 file dataset di `data/` (root repo) atau set `UTS_DATA_DIR`:
`kamus_data.md, regresi_train.csv.gz, regresi_test.csv.gz, klasifikasi_train.csv.gz, klasifikasi_test.csv.gz, clustering.csv.gz`.
Data tidak di-commit (lihat `.gitignore`).

## Menjalankan
```bash
pip install -r kode/requirements.txt
python -m pytest -q kode/tests                            # tes metrik + anti-kebocoran
python kode/run_all.py --task all --mode train            # SEMUA eksperimen + model final + 3 CSV (±45–75 menit, CPU 2 core)
python kode/run_all.py --task t2 --mode reproduce         # reproduksi cepat dari model tersimpan (< 5 menit per tugas)
python kode/run_all.py --task all --mode train --quick    # uji cepat pada subset data (±10 menit)
```
Di Google Colab cukup jalankan `UTS_ML_Colab.ipynb` (Runtime → Run all).

| Tahap | Perkiraan waktu (2 core) |
|---|---|
| Tugas 1 (train) | ±25 menit |
| Tugas 2 (train) | ±15 menit |
| Tugas 3 (train) | ±10 menit |
| Reproduce per tugas | ±1–3 menit |

## Reproducibility
- Seed global `42` (`UTS_SEED`); semua split, sampling, dan model memakai seed ini.
- Versi library tercatat di setiap `outputs/results/*.json` (kunci `versions`).
- Angka di laporan dibaca dari `outputs/tables/angka.tex` (dibuat `make_tables.py` dari `results/*.json`). Tidak ada angka yang diketik manual.

## Peta angka laporan → file hasil
| Bagian laporan | File hasil |
|---|---|
| T1 (a) audit | `results/t1_audit.json`, `tables/t1_audit_*.tex` |
| T1 (b)–(e) | `results/t1_experimen.json`, `tables/t1_validasi.tex`, `t1_model.tex`, `t1_tuning.tex`, `t1_encoding.tex`, `t1_mitra_baru.tex`, `t1_kalibrasi*.tex` |
| T1 (f) | `results/t1_analisis.json`, `tables/t1_breakdown_*.tex`, `t1_shap.tex` |
| T1 final | `results/t1_final.json`, `prediksi_regresi.csv` |
| T2 (a) | `results/t2_audit.json` |
| T2 (b)–(f) | `results/t2_experimen.json`, `tables/t2_validasi.tex`, `t2_label_lag.tex`, `t2_fitur_perilaku.tex`, `t2_imbalance.tex`, `t2_adversarial.tex`, `t2_adaptasi.tex`, `t2_kalibrasi_biaya.tex` |
| T2 (g) + final | `results/t2_analisis.json`, `results/t2_final.json`, `prediksi_klasifikasi.csv` |
| T3 (a)–(b) | `results/t3_seleksi_fitur.json`, `tables/t3_bukti_n.tex`, `t3_konfigurasi_fitur.tex`, `t3_kepentingan_kelompok.tex` |
| T3 (c)–(e) | `results/t3_metode_k.json`, `results/t3_stabilitas.json` |
| T3 (f)–(g) + final | `results/t3_final.json`, `tables/t3_profil.tex`, `t3_aksi.tex`, `t3_anomali.tex`, `cluster.csv` |

Keputusan dan alasannya: `DECISIONS.md`. Bank soal ujian lisan: `LISAN.md`.

## Penggunaan AI
Kode disusun dengan bantuan Claude (Anthropic) dan diverifikasi kelompok melalui tes otomatis, perbandingan skema validasi,
reproduksi ulang, dan pemeriksaan setiap tabel/gambar. Detail per anggota ada di `kontribusi.md`.

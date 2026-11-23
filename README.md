# UTS Machine Learning — Kelompok 2

**Estimasi Waktu Kirim, Deteksi Fraud, dan Segmentasi Pelanggan**
S1 Teknik Komputer, Telkom University · November 2026

| Nama | NIM | GitHub |
|---|---|---|
| Hadi Nugroho | 101032330235 | [@happhify](https://github.com/happhify) |
| Aryo Zaki Abdillah | 101032300060 | [@Ry-Gith](https://github.com/Ry-Gith) |

**Laporan lengkap:** [`laporan.pdf`](laporan.pdf) · **Catatan keputusan:** [`DECISIONS.md`](DECISIONS.md) · **Pembagian kerja:** [`kontribusi.md`](kontribusi.md)

---

## Ringkasan hasil

| Tugas | Target | Model terpilih | Hasil validasi (fold waktu F2) | File prediksi |
|---|---|---|---|---|
| 1. Regresi waktu kirim | Durasi kirim P50 & P90 (jam) | LightGBM quantile + kalibrasi P90 (CQR gabungan lintas bulan) | Pinball P50 **6,80** · pinball P90 **4,32** · coverage P90 **0,924** (target 0,90) | `prediksi_regresi.csv` |
| 2. Deteksi fraud | Probabilitas fraud + keputusan review | LightGBM + fitur perilaku tanpa kebocoran, kalibrasi sigmoid, aturan biaya | PR-AUC **0,774** · biaya **Rp67 juta** per 2 bulan (vs Rp832 juta bila tidak ada review) | `prediksi_klasifikasi.csv` |
| 3. Segmentasi pelanggan | Cluster tanpa label + anomali (−1) | K-Means k = 8 pada fitur terpilih berbobot per kelompok | Stabilitas ARI rata-rata **0,985** (20 run) · 7 segmen + **2,12%** anomali | `cluster.csv` |

Semua model adalah model klasik (scikit-learn, LightGBM). Validasi Tugas 1 dan 2 berbasis **waktu**, karena data uji (September–Oktober 2025) berada di masa depan data latih.

---

## Tugas 1 — Estimasi waktu kirim (P50 dan P90)

**Data:** 250.000 paket latih (Januari–Agustus 2025), 60.000 paket uji (September–Oktober 2025).

**Masalah data yang ditemukan dan ditangani**
- `jumlah_scan` **bocor**: rata-rata 9,7 di data latih vs 1,5 di data uji. Scan bertambah selama perjalanan, jadi tidak tersedia saat paket diambil → dibuang.
- `f30` hanya penanda waktu (naik lurus per bulan) → dibuang.
- 397 paket (0,16%) berdurasi > 500 jam karena **salah satuan** (menit, bukan jam) → dibagi 60.
- `jarak_km` bernilai sentinel 9999 dan 0 → diubah jadi *missing* + indikator.
- Mitra kurir **K13** hanya ada di data uji → disimulasikan dengan mengeluarkan satu mitra dari data latih.

**Pendekatan:** LightGBM quantile (α = 0,5 dan 0,9) dibandingkan dengan baseline kuantil per layanan × pulau (pinball P50 10,21) dan Ridge (7,42). Target encoding kota out-of-fold. P90 dikalibrasi dengan *conformalized quantile regression* yang skornya digabung dari beberapa bulan, karena kalibrasi dari satu bulan tidak stabil.

**Analisis error:** error terbesar ada di layanan *hemat*, tujuan **Papua** (pinball P50 21,3 jam), dan mitra **K07** (coverage P90 hanya 0,81). Fitur terpenting menurut SHAP: `layanan`, `jarak_km`, dan kota tujuan.

## Tugas 2 — Deteksi fraud

**Masalah data yang ditemukan dan ditangani**
- `flag_review_manual` **bocor**: tingkat fraud ±30% bila 1 vs ±0,1% bila 0, karena review terjadi *setelah* transaksi dicurigai → dibuang (PR-AUC turun dari 0,917 ke 0,781, angka yang jujur).
- **Label belum matang**: tingkat fraud mingguan stabil ±0,6% lalu anjlok ke ±0,09% di minggu-minggu terakhir, karena fraud belum terkonfirmasi saat data ditarik → model final hanya dilatih sampai batas label matang (11 Agustus 2025).
- Fitur kumulatif membuat data latih dan uji mudah dibedakan (AUC adversarial 0,985) → dibuang (turun ke 0,873).

**Pendekatan:** fitur perilaku dihitung hanya dari transaksi **sebelum** waktu transaksi, dijaga oleh 3 tes otomatis anti-kebocoran. Fitur ini menaikkan PR-AUC dari 0,763 ke 0,787. Penanganan imbalance (class weight, undersampling, SMOTE) dibandingkan; yang terbaik justru tanpa penanganan + kalibrasi sigmoid.

**Aturan keputusan:** tandai transaksi bila `probabilitas × nominal > Rp25.000` (biaya satu review). Biaya validasi per 2 bulan:

| Strategi | Biaya |
|---|---|
| Aturan biaya (dipakai) | **Rp69,1 juta** |
| Threshold tunggal | Rp81,3 juta |
| Tidak menandai apa pun | Rp832,5 juta |
| Menandai semua | Rp2.567,7 juta |

Di data uji, 1.723 transaksi (1,62%) ditandai untuk direview.

## Tugas 3 — Segmentasi pelanggan

**Temuan penting:** kolom `n01`–`n15` adalah **distraktor**. Kamus data hanya menyebut n01–n10, korelasinya dengan kolom lain ≈ 0, dan kolom ini membentuk cluster rapi sendiri (silhouette 0,67) yang tidak ada hubungannya dengan perilaku pelanggan (ARI ≈ 0) → dibuang.

**Pendekatan:** log1p untuk variabel miring, transformasi CLR untuk porsi kategori dan jam belanja (data komposisional), RobustScaler, dan bobot `1/√(jumlah kolom)` per kelompok fitur supaya tidak ada kelompok yang mendominasi. Silhouette datar dan BIC terus turun, jadi k dipilih sebagai **k terbesar yang stabil** (ARI antar run ≥ 0,95) → k = 8. K-Means dibandingkan dengan GMM, Agglomerative, dan HDBSCAN.

**Segmen dan aksi marketing**

| Segmen | Nama | Aksi |
|---|---|---|
| 0 | Pemburu Voucher & Flash Sale | Voucher dengan minimum belanja sedikit di atas rata-rata keranjang |
| 1 | Belanja Pagi & Pengguna Android | Promo pagi dan pengingat keranjang sebelum jam kerja |
| 2 | Mulai Pasif | Win-back dengan voucher terbatas waktu |
| 3 | Sering Retur & Pengguna Paylater | Panduan ukuran, ulasan, dan rekomendasi yang lebih akurat |
| 4 | Pengguna iOS & Usia Matang | Produk premium di aplikasi iOS |
| 5 | Pelanggan Loyal Bernilai Tinggi | Cross-sell produk pelengkap |
| 6 | Pelanggan Tidur | Reaktivasi: voucher comeback + pengingat kategori favorit |
| −1 | Anomali | Perilaku tidak wajar (mis. > 100 sesi per minggu dengan konversi hampir nol) — 2.546 pelanggan |

---

## Struktur repo

```
.
├── kode/
│   ├── run_all.py            ← satu perintah untuk semua angka, gambar, tabel, dan file prediksi
│   ├── requirements.txt
│   ├── common/               ← path, metrik, split waktu, I/O, validasi format, generator tabel LaTeX
│   ├── t1_regresi/           ← audit, eksperimen (validasi, model, encoding, kalibrasi), analisis SHAP, final
│   ├── t2_fraud/             ← fitur perilaku, audit, eksperimen, analisis, final
│   ├── t3_clustering/        ← preprocessing, seleksi fitur, pemilihan k, stabilitas, anomali, profil, final
│   ├── tests/                ← tes metrik + tes anti-kebocoran fitur Tugas 2
│   ├── configs/              ← konfigurasi terpilih (ditulis otomatis)
│   └── outputs/
│       ├── results/*.json    ← semua angka laporan + versi library + seed
│       ├── figures/  tables/ ← gambar dan tabel LaTeX (angka.tex dibuat otomatis)
│       └── models/           ← model final tersimpan
├── laporan/                  ← sumber LaTeX laporan
├── laporan.pdf
├── prediksi_regresi.csv      ← id_paket, p50, p90                (60.000 baris)
├── prediksi_klasifikasi.csv  ← id_transaksi, prob, tandai        (106.674 baris)
├── cluster.csv               ← id_pelanggan, cluster             (120.000 baris)
├── DECISIONS.md              ← setiap keputusan, buktinya, dan file hasilnya
└── kontribusi.md             ← pembagian kerja dan penggunaan AI
```

## Menjalankan ulang

1. Taruh file dataset di folder `data/` (tidak ikut di repo): `kamus_data.md`, `regresi_train.csv.gz`, `regresi_test.csv.gz`, `klasifikasi_train.csv.gz`, `klasifikasi_test.csv.gz`, `clustering.csv.gz`.
2. Jalankan:

```bash
pip install -r kode/requirements.txt
python -m pytest -q kode/tests                      # tes metrik + anti-kebocoran
python kode/run_all.py --task all --mode train      # semua eksperimen + model final + 3 CSV (±45–75 menit, CPU)
python kode/run_all.py --task all --mode reproduce  # reproduksi cepat dari model tersimpan (beberapa menit)
python kode/run_all.py --task all --mode train --quick   # uji cepat pada subset data
```

Bisa juga per tugas: `--task t1`, `t2`, atau `t3`.

## Reproducibility

- Seed global **42** untuk semua split, sampling, dan model.
- Setiap angka di laporan dibaca otomatis dari `kode/outputs/results/*.json` melalui `kode/outputs/tables/angka.tex`; **tidak ada angka yang diketik manual**.
- Versi library yang dipakai tercatat di setiap file hasil (kunci `versions`).
- Format ketiga file prediksi diperiksa otomatis oleh `kode/common/validate_submission.py` (kolom, jumlah baris, rentang nilai).

## Penggunaan AI

Kode disusun dengan bantuan Claude (Anthropic) dan diverifikasi kelompok melalui tes otomatis, perbandingan validasi waktu vs random CV, reproduksi ulang dari model tersimpan, dan pemeriksaan setiap tabel serta gambar. Rincian per anggota ada di [`kontribusi.md`](kontribusi.md).

# Catatan Keputusan (DECISIONS.md)

Setiap keputusan penting beserta bukti, alternatif yang ditolak, dan file hasil yang memuat angkanya.
Angka persis selalu dibaca dari `kode/outputs/results/*.json` (bisa sedikit berbeda antar-mesin).

## Umum
- **Hanya model klasik** (scikit-learn, LightGBM), sesuai soal. Seed tetap `42` (`UTS_SEED`).
- **Validasi berbasis waktu** untuk Tugas 1 dan 2, karena test adalah Sep–Okt 2025 (masa depan dari data latih). Random CV dilaporkan hanya sebagai pembanding.
- **Satu perintah reproduksi**: `python kode/run_all.py --task all --mode train`. Untuk reproduksi live saat ujian: `--mode reproduce` (memuat model tersimpan, < 5 menit).

## Tugas 1 — Regresi waktu kirim

| Keputusan | Bukti | File hasil |
|---|---|---|
| Buang `jumlah_scan` | Rata-rata 9,7 di train vs 1,5 di test (KS ±0,92). Scan bertambah selama pengiriman, jadi tidak tersedia saat pickup. Dengan fitur ini validasi tampak lebih baik, tetapi jika nilainya diganti seperti di test, pinball P50 dan coverage P90 runtuh. | `t1_audit.json`, `t1_audit_dampak.tex`, `t1_audit_bocor.png` |
| Buang `f30` | Median naik lurus per bulan (Jan→Okt), jadi hanya penanda waktu. Nilai test di luar rentang train, dan dampaknya ke pinball ≈ 0. | `t1_audit.json` |
| Durasi > 500 jam dibagi 60 | ±0,16% baris. Nilai ÷ 60 jatuh tepat di distribusi normal (median ±61 jam), jadi kesalahan satuan menit. Dibandingkan dengan "buang" dan "biarkan" pada validasi waktu. | `t1_audit_target.tex`, `t1_audit_durasi.png` |
| `jarak_km` 9999 / 0 → NaN + indikator | Nilai sentinel (±0,18% dan ±0,28%), juga ada di test. | `t1_audit.json` |
| Fold waktu F1 (latih Jan–Apr → val Mei–Jun) dan F2 (latih Jan–Jun → val Jul–Agu) | Meniru horizon test: 2 bulan setelah data latih. | `t1_validasi.tex` |
| LightGBM quantile (α=0,5 dan 0,9) sebagai model utama | Dibandingkan dengan baseline kuantil per (layanan × pulau) dan Ridge log-target. Tuning 5 konfigurasi hanya di fold waktu. | `t1_model.tex`, `t1_tuning.tex` |
| Encoding kota (TE out-of-fold / frekuensi / native) | Dipilih otomatis dengan pinball fold F2. | `t1_encoding.tex` |
| Mitra kurir baru (K13 hanya di test, ±3,9% baris) | Simulasi: satu mitra berporsi serupa dikeluarkan dari data latih. Strategi dipilih dengan pinball terbobot porsi test. | `t1_mitra_baru.tex` |
| Kalibrasi P90 | Kalibrasi dari satu bulan ternyata tidak stabil (kebablasan di fold yang memuat Maret, kurang di fold lain). Karena itu diuji juga CQR **gabungan lintas bulan** (skor out-of-time dari beberapa jendela "latih s.d. bulan m → prediksi 2 bulan berikutnya"). Coverage tanpa kalibrasi juga menurun dari fold ke fold (F1 ±0,91 → F2 ±0,87), jadi risiko kurang di Sep–Okt. Aturan: nilai hanya pada fold dengan ≥ 2 jendela (F2); di antara varian yang pinball P90-nya tidak lebih buruk dari tanpa kalibrasi (toleransi 1%), pilih coverage terdekat 0,90. | `t1_kalibrasi.tex`, `t1_kalibrasi_layanan.png` |
| Model final | Konfigurasi terpilih dilatih ulang pada Jan–Agu. Kalibrator dari jendela s.d. Agu. `p90 = max(p90, p50)`. | `t1_final.json`, `configs/t1_terpilih.json` |

## Tugas 2 — Fraud

| Keputusan | Bukti | File hasil |
|---|---|---|
| Buang `flag_review_manual` | Tingkat fraud ±30% jika 1 vs ±0,1% jika 0. ±80% fraud sudah pernah direview. Tingkat review di test lebih rendah (review belum terjadi). Review terjadi setelah/karena transaksi dicurigai, jadi tidak tersedia saat transaksi. PR-AUC melonjak jika dipakai (bocor). | `t2_audit.json`, `t2_audit_dampak.tex` |
| Buang `v15` | Naik lurus per bulan, jadi penanda waktu. | `t2_audit.json` |
| Label belum matang | Tingkat fraud mingguan stabil ±0,6% lalu turun tajam mulai ±11 Agu (minggu terakhir ±0,09%), karena fraud belum terkonfirmasi saat data ditarik 31 Agu. Validasi hanya di periode matang (Mei–Jul). Model final dilatih s.d. batas label matang (diestimasi otomatis). Simulasi label tersembunyi di fold F2 dipakai untuk memilih strategi. | `t2_audit_label.png`, `t2_label_lag.tex` |
| Fitur perilaku | Dihitung dari transaksi **sebelum** waktu transaksi (train+test digabung, urut waktu). Tingkat fraud historis memakai label > 30 hari sebelumnya saja. Diuji dengan 3 tes otomatis (`kode/tests/test_t2_features.py`). | `t2_fitur_perilaku.tex` |
| Buang fitur kumulatif (jumlah transaksi sejak Jan, dsb.) | Adversarial validation (Jun–Agu vs test): fitur kumulatif membuat train dan test hampir sempurna terbedakan (AUC ±0,99), karena nilainya naik terus mengikuti waktu. Setelah dibuang, AUC adversarial turun, dengan PR-AUC hampir sama. | `t2_adversarial.tex` |
| Imbalance | Dibandingkan: tanpa penanganan, scale_pos_weight, undersampling 1:20, SMOTE 1:10, plus threshold tuning. Semua dikalibrasi isotonic sebelum aturan biaya supaya adil. Pilihan: biaya terendah di validasi. | `t2_imbalance.tex` |
| Adaptasi waktu | Semua data vs 3 bulan terakhir vs bobot waktu (paruh 60 hari). Dipilih dengan PR-AUC fold F2. | `t2_adaptasi.tex` |
| Kalibrasi | Tanpa / sigmoid / isotonic pada 1 bulan kalibrasi (out-of-time). Dipilih dengan biaya validasi. | `t2_kalibrasi_biaya.tex`, `t2_reliability.png` |
| Aturan keputusan | Tandai jika `p × nominal > 25.000`. Turunan: biaya jika lolos = `p × nominal` (ekspektasi), biaya review = 25.000. Tandai kalau ekspektasi kerugian > biaya review. Dibandingkan dengan threshold tunggal, tandai semua, dan tidak menandai. | `t2_kalibrasi_biaya.tex` |

## Tugas 3 — Segmentasi

| Keputusan | Bukti | File hasil |
|---|---|---|
| Buang `n01`–`n15` | Kamus hanya menyebut n01–n10 tetapi data berisi 15 kolom. Korelasi dengan semua kolom lain ≈ 0. Kolom-kolom ini membentuk cluster sangat rapi di antara mereka sendiri (silhouette ±0,67) tetapi ARI dengan cluster perilaku/profil ≈ 0. Ini *distractor*: kalau dimasukkan, cluster akhir tertarik ke struktur n. | `t3_bukti_n.tex`, `t3_konfigurasi_fitur.tex` |
| Transformasi | log1p untuk variabel skewed. **CLR** untuk porsi kategori dan porsi jam (data komposisional). One-hot perangkat. `rata2_keranjang` NA = struktural (frekuensi 0) → 0 + indikator. Winsorize 0,5–99,5 lalu RobustScaler. | `kode/t3_clustering/preprocess.py` |
| Bobot per kelompok fitur | Tiap kelompok (RFM, kategori, waktu, promo, pasca-beli, aktivitas, profil) diberi bobot `1/√(jumlah kolom)` supaya 16 kolom kategori tidak mendominasi jarak. | `t3_konfigurasi_fitur.tex`, `t3_kepentingan_kelompok.tex` |
| Pemilihan k | Silhouette datar (±0,09) dan BIC terus turun, jadi keduanya tidak informatif. Aturan: **k terbesar yang stabil** (ARI antar run subsampling ≥ 0,95, k ≥ 3). | `t3_kriteria_k.tex`, `t3_kriteria_k.png` |
| Metode | K-Means vs GMM vs Agglomerative (Ward, sampel 15 ribu) vs HDBSCAN. Dipilih yang paling stabil antar seed. | `t3_metode.tex` |
| Anomali (-1) | Aturan domain (sesi > 100/minggu, rasio konversi > 1, belanja 30 hari > 90 hari) ∪ cluster yang mayoritas anggotanya melanggar aturan itu (profil bot). Konservatif, karena pelanggan normal yang diberi -1 menurunkan ARI. | `t3_anomali.tex` |
| Nama & aksi | Dibuat dari fitur pembeda (z-score segmen), lalu disempurnakan di laporan. | `t3_aksi.tex`, `t3_profil_heatmap.png` |

## Penggunaan AI
Kode disusun dengan bantuan Claude (Anthropic), lalu diverifikasi kelompok dengan:
- tes otomatis metrik dan anti-kebocoran;
- membandingkan validasi waktu dengan random CV;
- menjalankan ulang `--mode reproduce`;
- membaca setiap tabel/gambar.

Catat detailnya di `kontribusi.md`.

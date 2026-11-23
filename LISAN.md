# Bank Soal Ujian Lisan — UTS Machine Learning

Format ujian: 5 menit reproduksi live (satu anggota dipilih dosen), lalu 15 menit tanya jawab tentang tugas **mana saja** ke anggota **mana saja**.
Angka persis selalu ada di `kode/outputs/results/*.json` dan `kode/outputs/tables/angka.tex`. Hafalkan polanya, jangan cuma angkanya.

## Reproduksi live (latih minimal 2×)
```bash
python kode/run_all.py --task t1 --mode reproduce   # pinball & coverage fold F2 + tulis ulang prediksi_regresi.csv
python kode/run_all.py --task t2 --mode reproduce   # PR-AUC & biaya fold F2 + tulis ulang prediksi_klasifikasi.csv
python kode/run_all.py --task t3 --mode reproduce   # ARI(cluster.csv tersimpan, hasil ulang) harus 1.0
```
Yang muncul cocokkan dengan tabel di laporan. Model tersimpan di `kode/outputs/models/`.

---

## Tugas 1 — Regresi waktu kirim

1. **Kenapa `jumlah_scan` dibuang padahal paling berkorelasi dengan durasi?**
   Scan dicatat selama perjalanan paket. Paket yang lama di jalan punya scan lebih banyak, jadi itu akibat, bukan penyebab, dan tidak tersedia saat pickup. Buktinya: rata-rata di test ±1,5 vs ±9,7 di train. Model yang memakainya tampak bagus di validasi, tetapi saat nilainya diganti seperti di test, pinball dan coverage runtuh (`t1_audit_dampak`).
2. **Kenapa validasi waktu, bukan random CV?**
   Test adalah Sep–Okt (masa depan). Random CV mencampur bulan, sehingga model "melihat" pola musiman/tren dari bulan yang sama, dan hasilnya terlalu optimis. Kami tunjukkan perbandingannya di `t1_validasi`.
3. **Apa itu pinball loss? Kenapa untuk P90 angkanya lebih kecil dari P50?**
   `max(α(y−q), (α−1)(y−q))`. Untuk α = 0,9, prediksi terlalu rendah dihukum 0,9 per jam dan terlalu tinggi 0,1 per jam. Skalanya berbeda per α, jadi angka P50 dan P90 tidak dibandingkan langsung.
4. **Apa itu coverage P90 dan kenapa perlu kalibrasi?**
   Coverage adalah proporsi durasi aktual ≤ P90; idealnya 0,90. Model quantile tidak menjamin coverage, apalagi jika distribusi bergeser. CQR menambah koreksi berdasarkan skor `y − P90` di data out-of-time.
5. **Kenapa CQR gabungan lintas bulan, bukan satu bulan kalibrasi?**
   Kalibrasi satu bulan sangat dipengaruhi bulan itu: bulan yang memuat lonjakan Maret membuat koreksi kebablasan, bulan lain kurang. Menggabungkan skor dari beberapa jendela "latih s.d. m → prediksi 2 bulan" meniru horizon test dan lebih stabil (`t1_kalibrasi`).
6. **Bagaimana menangani mitra kurir K13 yang hanya ada di test?**
   Simulasi: satu mitra berporsi mirip dikeluarkan dari data latih, lalu diuji strategi `__OTHER__`, kategori dropout, atau tanpa fitur mitra (`t1_mitra_baru`). Kategori yang tidak dikenal dipetakan ke `__OTHER__`.
7. **Target encoding: kenapa out-of-fold?**
   Kalau dihitung dari seluruh data latih, setiap baris ikut "melihat" targetnya sendiri, sehingga model over-percaya pada kota jarang. Out-of-fold mencegah itu, dan smoothing menarik kota jarang ke rata-rata global.
8. **Durasi 25.452 jam itu apa?**
   ±0,16% baris > 500 jam. Jika ÷ 60, distribusinya sama dengan data normal, jadi itu kesalahan satuan (menit). Kami membandingkan 3 perlakuan (`t1_audit_target`).
9. **Kenapa `f30` dibuang?**
   Median per bulan naik lurus dari Jan ke Okt, jadi itu penanda waktu, bukan sifat paket. Pohon keputusan tidak bisa mengekstrapolasi nilai di luar rentang latih (Sep–Okt).
10. **Baca SHAP summary: apa artinya?**
    Setiap titik adalah satu paket. Posisi horizontal = kontribusi fitur ke prediksi P50 (jam), warna = nilai fitur. Fitur teratas ada di `t1_shap` (biasanya jarak dan layanan).
11. **Bagaimana kalau 30% paket test dari mitra baru?**
    Strategi `__OTHER__`/dropout jadi lebih penting, dan coverage harus dicek per mitra. CQR per grup bisa memperlebar interval untuk grup tak dikenal.

## Tugas 2 — Fraud

1. **Kenapa `flag_review_manual` dibuang?**
   Review dilakukan setelah transaksi dicurigai atau dilaporkan. ±80% fraud pernah direview dan tingkat fraud saat review ±30%, sehingga label hampir "dibocorkan". Di test tingkat review lebih rendah karena review belum terjadi. Saat skor dipakai (sebelum transaksi diproses), informasi ini belum ada.
2. **Apa maksud label belum matang dan bagaimana menanganinya?**
   `is_fraud` = terkonfirmasi saat data ditarik 31 Agu. Fraud di akhir Agustus belum sempat dilaporkan, jadi terlabel 0, dan tingkat fraud mingguan anjlok mulai ±11 Agu. Penanganannya: validasi hanya di bulan matang, model final dilatih s.d. batas matang, dan fitur historis memakai label > 30 hari saja.
3. **Kenapa PR-AUC, bukan akurasi atau ROC-AUC?**
   Fraud ±0,65%. Akurasi 99,3% bisa dicapai dengan menebak "bukan fraud" semua. ROC-AUC terlihat tinggi karena true negative sangat banyak. PR-AUC fokus pada kualitas peringkat di kelas positif.
4. **Turunkan aturan `p × nominal > 25.000`.**
   Jika ditandai: biaya 25.000 (fraud tertahan tanpa kerugian). Jika tidak: ekspektasi kerugian `p × nominal`. Tandai jika kerugian harapan > biaya review. Syaratnya `p` terkalibrasi.
5. **Kenapa perlu kalibrasi?**
   Undersampling, class weight, dan SMOTE mengubah proporsi kelas, sehingga probabilitas jadi terlalu tinggi. Aturan biaya butuh probabilitas yang sebenarnya. Kalibrasi isotonic/sigmoid dilakukan di bulan terpisah (out-of-time).
6. **Bagaimana memastikan fitur perilaku tidak bocor?**
   Data diurutkan menurut waktu, hanya baris sebelumnya yang dipakai (shift/rolling tanpa baris saat ini). Ada 3 tes otomatis: fitur tidak berubah jika data masa depan dihapus, tidak berubah jika label masa depan diubah, dan hitung ulang manual cocok.
7. **Apa itu adversarial validation dan apa temuannya?**
   Kami melatih model untuk membedakan train vs test. Kalau bisa, berarti distribusi berbeda, dan fitur terpentingnya menunjukkan penyebabnya. Fitur kumulatif sejak Jan (jumlah transaksi, tingkat fraud kategori kumulatif) membuat AUC ±0,99 karena naik mengikuti waktu. Fitur itu dibuang.
8. **Kenapa tidak menandai semua transaksi saja?**
   Biayanya 25.000 × jumlah transaksi (ratusan juta rupiah per 2 bulan), jauh di atas total nominal fraud. Tidak menandai sama sekali = semua kerugian fraud. Aturan kami jauh lebih murah dari keduanya (`t2_kalibrasi_biaya`).
9. **Pola fraud apa yang ditemukan?**
   Lihat `t2_pola_fraud`: pengambilalihan akun (ganti password + perangkat baru), akses IP luar negeri, kartu tidak hadir + jarak jauh, card testing. Recall model berbeda per pola, jadi sebutkan pola dengan recall terendah sebagai keterbatasan.
10. **Bagaimana kalau biaya review naik ke Rp 100.000?**
    Ambang `p × nominal > 100.000`, sehingga lebih sedikit yang ditandai: hanya transaksi bernominal besar atau berprobabilitas tinggi. Model tidak perlu dilatih ulang.

## Tugas 3 — Segmentasi

1. **Kenapa n01–n15 dibuang padahal clusternya paling rapi?**
   Justru itu jebakannya. n01–n15 membentuk cluster kuat (silhouette ±0,67) tetapi tidak berhubungan dengan perilaku, profil, perangkat, atau kota (ARI ≈ 0, korelasi ≈ 0). Kamus hanya menyebut n01–n10. Segmentasi untuk marketing harus bisa ditindaklanjuti, sedangkan cluster dari kolom anonim acak tidak.
2. **Apa itu CLR dan kenapa dipakai?**
   Porsi kategori jumlahnya 1, sehingga saling bergantung. Jarak Euclidean pada porsi mentah tidak bermakna. CLR: `log(x_i) − rata-rata log(x)`, yang memetakan simpleks ke ruang Euclidean.
3. **Kenapa memilih k dengan stabilitas, bukan silhouette?**
   Silhouette hampir datar (±0,09 untuk k = 3–12) dan BIC terus turun, jadi keduanya tidak bisa memilih. Stabilitas (ARI antar run subsampling) menunjukkan partisi yang muncul konsisten. Aturannya: k terbesar dengan ARI ≥ 0,95.
4. **Apa arti ARI?**
   Adjusted Rand Index: kesepakatan dua pelabelan, dikoreksi untuk kebetulan. 1 = identik, 0 = seperti acak. Nomor cluster tidak berpengaruh.
5. **Bagaimana menentukan -1?**
   Aturan domain (perilaku tidak mungkin) ditambah cluster yang mayoritas anggotanya melanggar aturan itu (profil bot: ratusan sesi per minggu dengan konversi ≈ 0). Kami sengaja konservatif karena -1 yang salah menurunkan ARI.
6. **Kenapa pelanggan tidak aktif jadi segmen sendiri, bukan anomali?**
   Jumlahnya besar (±5,6%) dan perilakunya konsisten (frekuensi 0), dan itu segmen marketing yang jelas (reaktivasi). Anomali adalah kasus yang tidak cocok dengan segmen mana pun.
7. **Kenapa K-Means, bukan HDBSCAN atau GMM?**
   K-Means paling stabil antar seed (`t3_metode`). HDBSCAN hanya menemukan 2 kelompok kepadatan (aktif vs tidak aktif) karena struktur lainnya tidak terpisah oleh celah kepadatan. GMM kurang stabil.
8. **Sebutkan 2 segmen dan aksinya.**
   Ambil dari `t3_aksi` dan `t3_profil`, misalnya "Sering Retur & Paylater" → panduan ukuran + rekomendasi lebih akurat; "Pelanggan Tidur" → voucher comeback.
9. **Kenapa bobot per kelompok fitur?**
   Tanpa bobot, 16 kolom porsi kategori menyumbang jauh lebih banyak ke jarak dibanding 4 kolom porsi jam. Bobot `1/√n` membuat setiap kelompok berkontribusi seimbang.

## Umum
- **Penggunaan AI:** sebutkan bahwa kode disusun dengan bantuan Claude, lalu jelaskan cara kalian memverifikasinya (tes, perbandingan validasi, reproduksi, membaca hasil). Ini tertulis di `kontribusi.md`.
- **Reproducibility:** seed 42; versi library tercatat di setiap `results/*.json`; satu perintah `run_all.py`.

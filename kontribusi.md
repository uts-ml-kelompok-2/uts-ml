# Kontribusi Anggota — UTS Machine Learning

> DRAF — lengkapi bagian `[...]` sebelum push. Isi file ini harus disetujui semua anggota.

## Anggota dan peran

| Anggota | NIM | Akun GitHub | Peran utama | Pull request |
|---|---|---|---|---|
| [Hadi Nugroho] | [101032330235] | [happhify] | Tugas 1 (regresi), Tugas 3 bagian 1 (preprocessing, seleksi fitur, metode, jumlah cluster), integrasi `run_all.py` | [link PR] |
| [Aryo Zaki Abdillah] | [1010323300060] | [Ry-Gith] | Tugas 2 (fraud), Tugas 3 bagian 2 (stabilitas, profil & aksi, anomali), laporan | [link PR] |

Review silang: PR milik A direview B, PR milik B direview A.

## Penggunaan AI

| Tool | Bagian yang dibantu | Cara kelompok memverifikasi |
|---|---|---|
| Claude (Anthropic) | Penyusunan sebagian besar kode di `kode/` (pipeline T1–T3, tes, `run_all.py`), kerangka laporan, `DECISIONS.md`, `LISAN.md` | (1) tes otomatis `pytest kode/tests` (metrik + 3 tes anti-kebocoran fitur T2); (2) membandingkan validasi waktu vs random CV; (3) menjalankan ulang `run_all.py --mode reproduce` dan mencocokkan angka; (4) memeriksa setiap tabel/gambar dan keputusan di `DECISIONS.md`; (5) [tambahkan pemeriksaan lain yang benar-benar dilakukan] |
| [tool lain, jika ada] | [...] | [...] |

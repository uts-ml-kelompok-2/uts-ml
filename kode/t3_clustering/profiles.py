"""Tugas 3 (f): profil, nama segmen, dan aksi marketing.
Nama & aksi dibuat otomatis dari fitur yang paling membedakan segmen (z-score rata-rata segmen vs populasi),
lalu disempurnakan manual di laporan berdasarkan tabel profil."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .preprocess import KAT, JAM

FRASA = {  # fitur (arah positif) → (frasa nama, aksi marketing)
    "persen_paylater": ("Pengguna Paylater", "promo cicilan 0% dan kenaikan limit paylater untuk pembelian berikutnya"),
    "persen_flash_sale": ("Pemburu Flash Sale", "notifikasi flash sale yang dipersonalisasi berdasarkan kategori favorit"),
    "persen_voucher": ("Pemburu Voucher", "voucher bertarget dengan minimum belanja sedikit di atas rata-rata keranjangnya"),
    "persen_cod": ("Setia COD", "insentif (cashback) untuk beralih ke pembayaran digital"),
    "clr_porsi_jam_malam": ("Belanja Malam", "kampanye push notification dan live shopping pukul 20.00–23.00"),
    "clr_porsi_jam_pagi": ("Belanja Pagi", "promo pagi hari dan pengingat keranjang sebelum jam kerja"),
    "clr_porsi_jam_siang": ("Belanja Siang", "promo jam makan siang"),
    "clr_porsi_jam_sore": ("Belanja Sore", "promo pulang kerja"),
    "belanja_90h": ("Belanja Besar", "program loyalitas/VIP dan akses awal produk baru"),
    "rata2_keranjang": ("Keranjang Besar", "bundling produk premium dan gratis ongkir tanpa minimum"),
    "frekuensi_90h": ("Sering Belanja", "membership berlangganan (gratis ongkir bulanan)"),
    "recency_hari": ("Mulai Pasif", "kampanye win-back dengan voucher terbatas waktu"),
    "retur_90h": ("Sering Retur", "panduan ukuran/ulasan produk dan rekomendasi yang lebih akurat"),
    "ulasan_90h": ("Rajin Mengulas", "program reviewer/poin ulasan"),
    "sesi_per_minggu": ("Aktif di Aplikasi", "rekomendasi personal di beranda"),
    "rasio_konversi": ("Cepat Membeli", "rekomendasi produk pelengkap (cross-sell)"),
    "klik_iklan_30h": ("Responsif Iklan", "iklan retargeting"), "wishlist": ("Penimbun Wishlist", "notifikasi turun harga item wishlist"),
    "lama_member_bulan": ("Member Lama", "apresiasi loyalitas (poin ganda)"), "umur": ("Usia Matang", "kurasi produk keluarga/rumah tangga"),
    "perangkat_ios": ("Pengguna iOS", "kampanye produk premium di aplikasi iOS"),
    "perangkat_android": ("Pengguna Android", "promo aplikasi Android dan pembayaran e-wallet"),
    "perangkat_web": ("Pengguna Web", "ajakan memasang aplikasi dengan voucher pengguna baru aplikasi"),
    "kota_tier": ("Kota Tier Bawah", "gratis ongkir ke kota tier 2–3"), "jumlah_alamat": ("Banyak Alamat", "fitur kirim hadiah"),
}
for c in KAT:
    nm = c.replace("porsi_", "").replace("_", " ").title()
    FRASA[f"clr_{c}"] = (f"Penggemar {nm}", f"promo dan rekomendasi kategori {nm.lower()}")
NEG = {"umur": ("Usia Muda", "konten media sosial dan promo produk tren"), "recency_hari": ("Baru Saja Belanja", "rekomendasi produk pelengkap setelah pembelian"),
       "lama_member_bulan": ("Member Baru", "onboarding dan voucher pembelian kedua")}


def profile_table(raw: pd.DataFrame, labels) -> pd.DataFrame:
    d = raw.copy(); d["segmen"] = labels; d = d[d.segmen >= 0]
    num = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan", "persen_voucher", "persen_flash_sale",
           "persen_paylater", "persen_cod", "retur_90h", "sesi_per_minggu", "rasio_konversi", "umur", "lama_member_bulan", "kota_tier"]
    t = d.groupby("segmen")[num].median()
    t.insert(0, "ukuran_%", d.segmen.value_counts(normalize=True).sort_index() * 100)
    t["kategori_dominan"] = d.groupby("segmen")[KAT].mean().idxmax(axis=1).str.replace("porsi_", "")
    t["jam_dominan"] = d.groupby("segmen")[JAM].mean().idxmax(axis=1).str.replace("porsi_jam_", "")
    t["perangkat_dominan"] = d.groupby("segmen")["perangkat"].agg(lambda s: f"{s.value_counts().index[0]} ({s.value_counts(normalize=True).iloc[0]*100:.0f}%)")
    t["porsi_tidak_aktif_%"] = d.groupby("segmen")["frekuensi_90h"].apply(lambda s: (s == 0).mean() * 100)
    return t


def name_segments(X: pd.DataFrame, labels, raw: pd.DataFrame):
    Z = (X - X.mean()) / (X.std() + 1e-9)
    zc = Z.groupby(labels).mean()
    out = {}
    for c in zc.index:
        if c < 0:
            continue
        if (raw.loc[labels == c, "frekuensi_90h"] == 0).mean() > 0.5:
            out[int(c)] = {"nama": "Pelanggan Tidur", "aksi": "kampanye reaktivasi: voucher comeback + pengingat kategori favorit",
                           "fitur_pembeda": ["frekuensi_90h = 0"]}
            continue
        z = zc.loc[c].drop(labels=[k for k in ["tidak_aktif_90h"] if k in zc.columns])
        cand = []
        for f, v in z.sort_values(key=np.abs, ascending=False).items():
            if v > 0 and f in FRASA:
                cand.append((f, FRASA[f]))
            elif v < 0 and f in NEG:
                cand.append((f, NEG[f]))
            if len(cand) == 2:
                break
        nama = " & ".join(p[1][0] for p in cand) if cand else f"Segmen {c}"
        out[int(c)] = {"nama": nama, "aksi": cand[0][1][1] if cand else "-", "fitur_pembeda": [f"{f} (z={z[f]:+.2f})" for f, _ in cand]}
    return out, zc


def heatmap(zc: pd.DataFrame, names: dict, path_saver):
    cols = zc.abs().max().sort_values(ascending=False).head(18).index
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(zc) + 2))
    im = ax.imshow(zc[cols].values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("clr_porsi_", "") for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(zc))); ax.set_yticklabels([f"{i}: {names.get(int(i), {}).get('nama', '')}" for i in zc.index], fontsize=8)
    plt.colorbar(im, ax=ax, label="z-score rata-rata segmen"); ax.set_title("Profil segmen (z-score fitur terpilih)")
    path_saver(fig, "t3_profil_heatmap")

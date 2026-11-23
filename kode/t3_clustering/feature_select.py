"""Tugas 3 (b): seleksi & pembobotan fitur dengan bukti."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score, adjusted_rand_score as ari
from .preprocess import load_raw, build_features, GROUPS, NOISE, DEFAULT_WEIGHTS
from ..common.paths import KODE, SEED, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer


def stability(X, k, reps=4, frac=0.8, seed=SEED):
    Xv = X.values if hasattr(X, "values") else X
    rng = np.random.RandomState(seed); labs = []
    for r in range(reps):
        idx = rng.choice(len(Xv), int(frac * len(Xv)), replace=False)
        labs.append(KMeans(k, n_init=5, random_state=seed + r).fit(Xv[idx]).predict(Xv))  # n_init sama dengan model final
    v = [ari(labs[i], labs[j]) for i in range(reps) for j in range(i + 1, reps)]
    return float(np.mean(v)), float(np.std(v))


def run():
    df = load_raw()
    res, angka = {}, {}
    rng = np.random.RandomState(SEED); s = rng.choice(len(df), min(10000, len(df)), replace=False)
    K = 8
    with Timer("T3 (b) bukti kolom n01–n15"):
        Xn, _, _ = build_features(df, use_noise=True)
        n_cols = [c for c in NOISE if c in Xn.columns]; beh = [c for c in Xn.columns if c not in n_cols]
        corr = Xn.corr(method="spearman").loc[n_cols, beh].abs().max(axis=1)
        # MI dengan estimator kNN bias ke atas pada fitur diskret (one-hot, tier, indikator). Karena itu yang dilaporkan adalah
        # MI BERLEBIH = MI asli − MI ketika kolom n diacak (acuan "tanpa hubungan"). Nilai ≈ 0 berarti tidak ada hubungan.
        disc = np.array([Xn[c].nunique() <= 10 for c in beh])
        mi_ex = {}
        for c in n_cols[:5]:
            y = Xn.iloc[s][c].values
            a = mutual_info_regression(Xn.iloc[s][beh], y, discrete_features=disc, random_state=SEED)
            b = mutual_info_regression(Xn.iloc[s][beh], np.random.RandomState(SEED).permutation(y), discrete_features=disc, random_state=SEED)
            mi_ex[c] = float(np.max(a - b))
        lab_n = KMeans(4, n_init=3, random_state=SEED).fit_predict(Xn[n_cols])
        lab_b = KMeans(K, n_init=3, random_state=SEED).fit_predict(Xn[beh])
        lab_all = KMeans(K, n_init=3, random_state=SEED).fit_predict(Xn)
        sil_n = silhouette_score(Xn[n_cols].values[s], lab_n[s])
        ev = {"max_abs_spearman_n_vs_perilaku": float(corr.max()), "max_MI_berlebih_vs_acuan_acak": max(mi_ex.values()),
              "silhouette_cluster_n_saja": float(sil_n), "ARI(cluster_n, cluster_perilaku)": ari(lab_n, lab_b),
              "ARI(cluster_semua_kolom, cluster_n)": ari(lab_all, lab_n), "ARI(cluster_semua_kolom, cluster_perilaku)": ari(lab_all, lab_b),
              "jumlah_kolom_n": len(n_cols), "kamus_hanya_menyebut": "n01–n10"}
        for g in ["kota_tier", "perangkat"]:
            ev[f"ARI(cluster_n, {g})"] = ari(lab_n, df[g].astype(str))
        res["bukti_n"] = ev
        save_table(pd.DataFrame([{"bukti": k, "nilai": v} for k, v in ev.items() if not isinstance(v, str)]), "t3_bukti_n",
                   "Bukti bahwa n01–n15 membentuk struktur sendiri yang tidak terkait perilaku/profil pelanggan.")
        angka.update(TtigaNSil=round(sil_n, 2), TtigaNARIPerilaku=round(ev["ARI(cluster_n, cluster_perilaku)"], 3),
                     TtigaNMaxKorelasi=round(ev["max_abs_spearman_n_vs_perilaku"], 3), TtigaSemuaVsN=round(ev["ARI(cluster_semua_kolom, cluster_n)"], 2))

    with Timer("T3 (b) redundansi + bobot kelompok"):
        raw = df.copy()
        red = {"belanja_30h~belanja_90h": float(np.log1p(raw.belanja_30h).corr(np.log1p(raw.belanja_90h), method="spearman")),
               "frekuensi_30h~frekuensi_90h": float(raw.frekuensi_30h.corr(raw.frekuensi_90h, method="spearman"))}
        res["redundansi"] = red
        rows = []
        for nama, kw in [("semua kolom termasuk n01–n15", dict(use_noise=True)), ("tanpa n, tanpa bobot", dict()),
                         ("tanpa n, bobot per kelompok (dipakai)", dict(weights=DEFAULT_WEIGHTS))]:
            X, _, _ = build_features(df, **kw)
            lab = KMeans(K, n_init=3, random_state=SEED).fit_predict(X)
            st = stability(X, K, reps=3 if QUICK else 4)
            rows.append({"konfigurasi": nama, "n_fitur": X.shape[1], "silhouette": silhouette_score(X.values[s], lab[s]),
                         "stabilitas_ARI": st[0], "ARI_dengan_cluster_n": ari(lab, lab_n)})
        cdf = pd.DataFrame(rows)
        res["konfigurasi"] = cdf.to_dict("records")
        save_table(cdf, "t3_konfigurasi_fitur", f"Konfigurasi fitur pada k={K}: silhouette, stabilitas, dan seberapa jauh cluster ditarik oleh n01–n15.")

    with Timer("T3 (b) kepentingan kelompok fitur (permutasi)"):
        X, _, _ = build_features(df, weights=DEFAULT_WEIGHTS)
        km = KMeans(K, n_init=3, random_state=SEED).fit(X); base = km.labels_
        rows = []
        for g, cols in GROUPS.items():
            cc = [c for c in cols if c in X.columns]
            if not cc:
                continue
            Xp = X.copy(); Xp[cc] = Xp[cc].values[rng.permutation(len(Xp))]
            rows.append({"kelompok": g, "n_kolom": len(cc), "perubahan_label_%": float((km.predict(Xp) != base).mean() * 100),
                         "1-ARI": 1 - ari(base, km.predict(Xp))})
        gdf = pd.DataFrame(rows).sort_values("1-ARI", ascending=False)
        res["kepentingan_kelompok"] = gdf.to_dict("records")
        save_table(gdf, "t3_kepentingan_kelompok", "Kepentingan kelompok fitur: seberapa banyak label berubah jika kelompok diacak.")
        fig, ax = plt.subplots(figsize=(7, 3)); ax.barh(gdf.kelompok, gdf["1-ARI"]); ax.invert_yaxis(); ax.set_xlabel("1 − ARI setelah diacak")
        ax.set_title("Kepentingan kelompok fitur untuk struktur cluster"); save_fig(fig, "t3_kepentingan_kelompok")
        angka.update(TtigaKelompokTerpenting=str(gdf.iloc[0].kelompok))

    cfg = {"use_noise": False, "weights": "kelompok", "alasan": "n01–n15 independen dari perilaku & profil (lihat t3_bukti_n)"}
    p = KODE / "configs" / "t3_terpilih.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **cfg}, indent=2))
    save_result("t3_seleksi_fitur", res, angka)
    return res


if __name__ == "__main__":
    run()

"""Tugas 3 (c) perbandingan metode + (d) jumlah cluster.

Aturan pemilihan k (ditulis sebelum melihat hasil akhir): k terbesar yang masih STABIL
(rata-rata ARI antar run subsampling ≥ 0.95), dengan k ≥ 3. Alasan: silhouette pada data ini datar (±0.09)
sehingga tidak informatif; BIC terus turun seiring k; stabilitas menunjukkan partisi mana yang benar-benar ada di data
dan bukan potongan acak. k=2 hanya memisahkan pelanggan tidak aktif sehingga tidak actionable.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score as ari
from .preprocess import load_raw
from .pipeline import features, fit_method, load_config
from .feature_select import stability
from ..common.paths import KODE, SEED, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer

STABLE = 0.95


def run():
    df = load_raw(); cfg = load_config()
    X, _, meta = features(df, cfg); Xv = X.values
    rng = np.random.RandomState(SEED); s = rng.choice(len(Xv), min(10000, len(Xv)), replace=False)
    res, angka = {}, {}
    ks = list(range(2, 11 if QUICK else 13))
    with Timer("T3 (d) kriteria jumlah cluster"):
        rows = []
        for k in ks:
            km = KMeans(k, n_init=3, random_state=SEED).fit(Xv)
            g = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=SEED).fit(Xv[rng.choice(len(Xv), min(30000, len(Xv)), replace=False)].astype("float64"))
            st, sd = stability(X, k, reps=3 if QUICK else 5)
            rows.append({"k": k, "silhouette": silhouette_score(Xv[s], km.labels_[s]), "calinski_harabasz": calinski_harabasz_score(Xv[s], km.labels_[s]),
                         "davies_bouldin": davies_bouldin_score(Xv[s], km.labels_[s]), "bic_gmm": g.bic(Xv[s].astype("float64")), "stabilitas_ARI": st, "stabilitas_std": sd,
                         "ukuran_terkecil_%": float(np.bincount(km.labels_).min() / len(Xv) * 100)})
        kdf = pd.DataFrame(rows)
        cand = kdf[(kdf.k >= 3) & (kdf.stabilitas_ARI >= STABLE)]
        k_best = int(cand.k.max()) if len(cand) else int(kdf[kdf.k >= 3].sort_values("stabilitas_ARI", ascending=False).iloc[0].k)
        res["kriteria"] = kdf.to_dict("records"); res["k_terpilih"] = k_best; res["aturan"] = f"k terbesar dengan stabilitas ARI ≥ {STABLE} (k ≥ 3)"
        save_table(kdf, "t3_kriteria_k", "Kriteria jumlah cluster (K-Means). Stabilitas = rata-rata ARI antar 5 run subsampling 80%.")
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
        ax[0].plot(kdf.k, kdf.stabilitas_ARI, marker="o"); ax[0].axhline(STABLE, color="r", ls="--", lw=1); ax[0].set_title("Stabilitas (ARI antar run)")
        ax[1].plot(kdf.k, kdf.silhouette, marker="o"); ax[1].set_title("Silhouette")
        ax[2].plot(kdf.k, kdf.davies_bouldin, marker="o"); ax[2].set_title("Davies–Bouldin (↓)")
        ax[3].plot(kdf.k, kdf.bic_gmm, marker="o"); ax[3].set_title("BIC GMM (↓)")
        for a in ax:
            a.axvline(k_best, color="g", ls=":", lw=1.5); a.set_xlabel("k")
        save_fig(fig, "t3_kriteria_k")
        r = kdf.set_index("k").loc[k_best]
        angka.update(TtigaK=k_best, TtigaStabK=round(r.stabilitas_ARI, 3), TtigaSilK=round(r.silhouette, 3),
                     TtigaStabKplus=round(kdf.set_index("k").loc[min(k_best + 1, max(ks)), "stabilitas_ARI"], 3),
                     TtigaStabKmin=round(kdf.set_index("k").loc[k_best - 1, "stabilitas_ARI"], 3))

    with Timer("T3 (c) perbandingan metode"):
        rows, labs = [], {}
        for method in ["kmeans", "gmm", "agglomerative", "hdbscan"]:
            lab, m = fit_method(X, method, k_best)
            lab2, _ = fit_method(X, method, k_best, seed=SEED + 1)
            labs[method] = lab
            ok = lab >= 0
            rows.append({"metode": method, "n_cluster": int(len(set(lab[ok]))), "persen_noise": float((~ok).mean() * 100),
                         "silhouette": silhouette_score(Xv[s][ok[s]], lab[s][ok[s]]) if len(set(lab[s][ok[s]])) > 1 else np.nan,
                         "davies_bouldin": davies_bouldin_score(Xv[s][ok[s]], lab[s][ok[s]]) if len(set(lab[s][ok[s]])) > 1 else np.nan,
                         "ARI_antar_seed": ari(lab, lab2)})
        mdf = pd.DataFrame(rows)
        mdf["ARI_vs_kmeans"] = [ari(labs["kmeans"], labs[mm]) for mm in mdf.metode]
        res["metode"] = mdf.to_dict("records")
        save_table(mdf, "t3_metode", f"Perbandingan metode pada k={k_best} (HDBSCAN menentukan jumlah cluster sendiri).")
        best = mdf[mdf.metode != "hdbscan"].sort_values(["ARI_antar_seed", "silhouette"], ascending=False).iloc[0].metode
        res["metode_terpilih"] = best
        angka.update(TtigaMetode=str(best))

    p = KODE / "configs" / "t3_terpilih.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, "k": k_best, "method": best}, indent=2))
    save_result("t3_metode_k", res, angka)
    return res


if __name__ == "__main__":
    run()

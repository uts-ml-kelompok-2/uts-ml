"""Tugas 3 final: clustering akhir + anomali (-1) + profil & aksi → cluster.csv."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .preprocess import load_raw
from .pipeline import features, fit_method, load_config
from .anomalies import domain_rule, suspicious_clusters, compare_methods
from .profiles import profile_table, name_segments, heatmap
from ..common.paths import MOD, ROOT, SEED
from ..common.io_utils import save_result, save_fig, save_table, Timer


def assign(raw, X, labels, model):
    flag = domain_rule(raw)
    sus = suspicious_clusters(labels, flag)
    final = labels.copy()
    final[np.isin(labels, sus) | flag] = -1
    # nomori ulang cluster normal 0..K-1 (urut ukuran) supaya rapi
    keep = sorted([c for c in np.unique(final) if c >= 0], key=lambda c: -(final == c).sum())
    remap = {c: i for i, c in enumerate(keep)}; remap[-1] = -1
    return np.vectorize(remap.get)(final), sus


def train():
    raw = load_raw(); cfg = load_config()
    res, angka = {}, {}
    with Timer("T3 final: clustering"):
        X, _, meta = features(raw, cfg)
        labels, model = fit_method(X, cfg["method"], cfg["k"], seed=SEED)
    with Timer("T3 (g) anomali"):
        comp, sus = compare_methods(X, raw, labels, model)
        final, sus = assign(raw, X, labels, model)
        rows = []
        dom = comp["aturan domain"]
        for nm, mk in comp.items():
            rows.append({"metode": nm, "jumlah": int(mk.sum()), "persen": float(mk.mean() * 100),
                         "irisan_dgn_aturan_domain_%": float((mk & dom).sum() / max(mk.sum(), 1) * 100)})
        adf = pd.DataFrame(rows)
        res["anomali"] = {"perbandingan": adf.to_dict("records"), "cluster_bot": sus, "jumlah_minus1": int((final == -1).sum()),
                          "persen_minus1": float((final == -1).mean() * 100),
                          "rincian_aturan": {"sesi>100/minggu": int((raw.sesi_per_minggu > 100).sum()), "rasio_konversi>1": int((raw.rasio_konversi > 1).sum()),
                                             "belanja_30h>belanja_90h": int((raw.belanja_30h > raw.belanja_90h + 1).sum())}}
        save_table(adf, "t3_anomali", "Perbandingan detektor anomali. Label -1 = aturan domain ∪ cluster berprofil bot.")
        angka.update(TtigaAnomaliN=int((final == -1).sum()), TtigaAnomaliPersen=round(float((final == -1).mean() * 100), 2),
                     TtigaNSegmen=int(len(set(final[final >= 0]))))
    with Timer("T3 (f) profil & nama segmen"):
        prof = profile_table(raw, final)
        names, zc = name_segments(X[final >= 0], final[final >= 0], raw[final >= 0].reset_index(drop=True).set_index(X[final >= 0].index))
        prof.insert(0, "nama_segmen", [names[int(i)]["nama"] for i in prof.index])
        res["profil"] = prof.reset_index().to_dict("records"); res["nama_aksi"] = names
        t = prof[["ukuran_%", "frekuensi_90h", "belanja_90h", "recency_hari", "persen_paylater", "persen_flash_sale",
                  "kategori_dominan", "jam_dominan", "perangkat_dominan"]].copy()   # nama segmen ada di tabel t3_aksi
        t["belanja_90h"] = t["belanja_90h"] / 1e3
        save_table(t.rename(columns={"belanja_90h": "belanja_90h (ribu Rp)"}).reset_index(), "t3_profil", "Profil segmen (median fitur asli).", float_fmt="%.2f")
        save_table(pd.DataFrame([{"segmen": k, "nama": v["nama"], "aksi marketing": v["aksi"]} for k, v in names.items()]), "t3_aksi",
                   "Nama segmen dan usulan aksi marketing.")
        heatmap(zc, names, save_fig)
        p = PCA(2, random_state=SEED).fit_transform(X.values)
        s = np.random.RandomState(SEED).choice(len(p), min(15000, len(p)), replace=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in sorted(set(final)):
            mk = final[s] == c
            ax.scatter(p[s][mk, 0], p[s][mk, 1], s=3, alpha=.5, label=("-1 anomali" if c < 0 else f"{c}: {names[int(c)]['nama']}"), c="k" if c < 0 else None)
        ax.legend(markerscale=4, fontsize=7); ax.set_title("Segmen pada 2 komponen utama (PCA)"); save_fig(fig, "t3_pca")
    joblib.dump({"model": model, "cfg": cfg, "bot_clusters": sus}, MOD / "t3_model_final.joblib")
    out = pd.DataFrame({"id_pelanggan": raw["id_pelanggan"].values, "cluster": final.astype(int)})
    out.to_csv(ROOT / "cluster.csv", index=False)
    res["ukuran_cluster"] = out.cluster.value_counts().sort_index().to_dict()
    save_result("t3_final", res, angka)
    print(f"cluster.csv ditulis: {len(out)} baris, {res['ukuran_cluster']}")
    return res


def reproduce():
    raw = load_raw(); art = joblib.load(MOD / "t3_model_final.joblib")
    X, _, _ = features(raw, art["cfg"])
    labels = art["model"].predict(X.values)
    final, _ = assign(raw, X, labels, art["model"])
    old = pd.read_csv(ROOT / "cluster.csv")
    from ..common.metrics import ari
    print("REPRODUKSI T3: ARI(cluster.csv tersimpan, hasil ulang) =", round(ari(old.cluster.values, final), 4),
          "| ukuran:", pd.Series(final).value_counts().sort_index().to_dict())
    return {"ari_vs_tersimpan": ari(old.cluster.values, final)}


if __name__ == "__main__":
    train()

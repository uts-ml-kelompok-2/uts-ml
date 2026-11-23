"""Tugas 3 (e): stabilitas cluster dengan subsampling (20 run × 80%)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari
from .preprocess import load_raw
from .pipeline import features, fit_method, load_config
from ..common.paths import SEED, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer


def _runs(X, method, k, reps, frac=0.8):
    Xv = X.values
    rng = np.random.RandomState(SEED); labs = []
    for r in range(reps):
        idx = rng.choice(len(Xv), int(frac * len(Xv)), replace=False)
        lab_s, m = fit_method(Xv[idx], method, k, seed=SEED + r)
        labs.append(m.predict(Xv) if hasattr(m, "predict") else lab_s)
    return labs


def _jaccard_per_cluster(ref, other):
    out = []
    for c in np.unique(ref):
        a = ref == c
        best = max(((a & (other == d)).sum() / max((a | (other == d)).sum(), 1)) for d in np.unique(other))
        out.append(best)
    return np.array(out)


def run():
    df = load_raw(); cfg = load_config(); k = cfg["k"]; method = cfg["method"]
    X, _, _ = features(df, cfg)
    reps = 6 if QUICK else 20
    res, angka = {}, {}
    with Timer(f"T3 (e) stabilitas {reps} run"):
        rows = []
        for kk in [k - 1, k, k + 1]:
            labs = _runs(X, method, kk, reps if kk == k else max(5, reps // 3))
            v = [ari(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
            rows.append({"k": kk, "metode": method, "n_run": len(labs), "ARI_mean": float(np.mean(v)), "ARI_std": float(np.std(v)), "ARI_min": float(np.min(v))})
            if kk == k:
                ref = labs[0]
                jac = np.mean([_jaccard_per_cluster(ref, l) for l in labs[1:]], axis=0)
                mat = np.array([[ari(a, b) for b in labs] for a in labs])
        sdf = pd.DataFrame(rows)
        res["stabilitas"] = sdf.to_dict("records")
        jt = pd.DataFrame({"cluster_run_acuan": range(len(jac)), "ukuran": np.bincount(ref), "jaccard_rata2": jac})
        res["jaccard_per_cluster"] = jt.to_dict("records")
        save_table(sdf, "t3_stabilitas", f"Stabilitas: ARI antar run subsampling 80% ({reps} run untuk k={k}).")
        save_table(jt, "t3_jaccard", "Jaccard rata-rata tiap cluster terhadap cluster paling mirip di run lain.")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        im = ax[0].imshow(mat, vmin=0.5, vmax=1, cmap="viridis"); plt.colorbar(im, ax=ax[0]); ax[0].set_title(f"ARI antar {reps} run (k={k})")
        ax[1].bar(jt.cluster_run_acuan.astype(str), jt.jaccard_rata2); ax[1].set_ylim(0, 1); ax[1].set_title("Jaccard per cluster")
        save_fig(fig, "t3_stabilitas")
        r = sdf.set_index("k").loc[k]
        angka.update(TtigaStabMean=round(r.ARI_mean, 3), TtigaStabStd=round(r.ARI_std, 3), TtigaStabMin=round(r.ARI_min, 3),
                     TtigaJaccardMin=round(float(jac.min()), 3), TtigaNRun=reps)
    save_result("t3_stabilitas", res, angka)
    return res


if __name__ == "__main__":
    run()

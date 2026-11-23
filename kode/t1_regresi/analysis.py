"""Tugas 1 (f): breakdown error per layanan / pulau_tujuan / mitra_kurir + penjelasan model (SHAP, partial dependence).
Model analisis = konfigurasi final dilatih Jan–Jun, divalidasi Jul–Agu (fold F2), dengan kalibrasi CQR dari Jun."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import load_raw
from .common_t1 import prepare, load_config, make_xy_cfg
from .models import LGBQuantile, evaluate
from .conformal import CQRUpper
from ..common.splits import month_mask
from ..common.paths import MOD, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer


def make_cqr(d, cfg, last_month):
    """Kalibrator P90 sesuai varian terpilih, memakai skor out-of-time dari jendela yang selesai <= last_month."""
    from .conformal import pooled_scores
    if cfg["kalibrasi"] == "tanpa kalibrasi":
        return CQRUpper()  # tidak dipakai
    sc = pooled_scores(d, cfg, last_month=last_month)
    if cfg["kalibrasi"] == "CQR jendela terakhir":
        sc = sc[sc.akhir_latih == sc.akhir_latih.max()]
    group = "layanan" if "per layanan" in cfg["kalibrasi"] else None
    return CQRUpper(group_col=group).fit(sc.y.values, sc.p90.values, sc.layanan.values)


def fit_f2_model(d, cfg):
    """Model analisis fold F2: kalibrator dari jendela s.d. Jun, model dilatih Jan–Jun, dievaluasi Jul–Agu."""
    m_tr = month_mask(d.tgl_pickup, "2025-01", "2025-06"); m_va = month_mask(d.tgl_pickup, "2025-07", "2025-08")
    cqr = make_cqr(d, cfg, "2025-06")
    Xtr, Xva, enc = make_xy_cfg(d[m_tr], d[m_va], cfg)
    m = LGBQuantile(cfg["params"]).fit(Xtr, d.loc[m_tr, "y"])
    return m, enc, cqr, Xva, m_va


def predict_with(m, cqr, X, layanan, cfg):
    p50, p90 = m.predict(X)
    if cfg["kalibrasi"] != "tanpa kalibrasi":
        p90 = cqr.adjust(p90, layanan)
    return p50, np.maximum(p90, p50)


def run():
    tr, _ = load_raw()
    d = prepare(tr)
    cfg = load_config()
    res, angka = {}, {}
    with Timer("T1 (f) model analisis F2"):
        m, enc, cqr, Xva, m_va = fit_f2_model(d, cfg)
        va = d[m_va].reset_index(drop=True)
        p50, p90 = predict_with(m, cqr, Xva, va["layanan"].values, cfg)
        joblib.dump({"model": m, "enc": enc, "cqr": cqr, "cfg": cfg}, MOD / "t1_model_validasi_F2.joblib")
        tot = evaluate(va.y_eval, p50, p90)
        res["total_F2"] = tot
        angka.update(TsatuFinalValPinbal=round(tot["pinball50"], 2), TsatuFinalValPinbalNilan=round(tot["pinball90"], 2),
                     TsatuFinalValCov=round(tot["coverage90"], 3))

    va["p50"], va["p90"] = p50, p90
    for col in ["layanan", "pulau_tujuan", "mitra_kurir"]:
        rows = []
        for k, g in va.groupby(col):
            r = evaluate(g.y_eval, g.p50, g.p90); r[col] = k; r["median_durasi"] = float(g.y_eval.median()); rows.append(r)
        t = pd.DataFrame(rows)[[col, "n", "median_durasi", "pinball50", "pinball90", "coverage90", "mae"]].sort_values("pinball50", ascending=False)
        res[f"breakdown_{col}"] = t.to_dict("records")
        save_table(t, f"t1_breakdown_{col}", f"Error per {col} (fold F2, model final + kalibrasi).")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        ax[0].bar(t[col].astype(str), t["pinball50"]); ax[0].set_title(f"Pinball P50 per {col}"); ax[0].tick_params(axis="x", rotation=45)
        ax[1].bar(t[col].astype(str), t["coverage90"]); ax[1].axhline(0.9, color="k", ls="--", lw=1)
        ax[1].set_ylim(0.6, 1); ax[1].set_title(f"Coverage P90 per {col}"); ax[1].tick_params(axis="x", rotation=45)
        save_fig(fig, f"t1_breakdown_{col}")
        worst = t.iloc[0]
        key = {"layanan": "Layanan", "pulau_tujuan": "Pulau", "mitra_kurir": "Mitra"}[col]
        angka[f"Tsatu{key}Terburuk"] = str(worst[col]); angka[f"Tsatu{key}TerburukPinbal"] = round(worst["pinball50"], 2)

    with Timer("T1 (f) SHAP + partial dependence"):
        import shap
        n = 1000 if QUICK else 3000
        Xs = Xva.sample(n, random_state=0)
        expl = shap.TreeExplainer(m.m50_)
        sv = expl.shap_values(Xs)
        imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
        res["shap_importance"] = imp.head(15).to_dict()
        save_table(imp.head(10).rename("mean |SHAP|").reset_index().rename(columns={"index": "fitur"}), "t1_shap",
                   "Rata-rata |SHAP| model P50 (3.000 sampel validasi).")
        plt.figure(); shap.summary_plot(sv, Xs, show=False, max_display=12); save_fig(plt.gcf(), "t1_shap_summary")
        top = [c for c in imp.index if Xs[c].dtype.kind in "fi"][:3]
        fig, ax = plt.subplots(1, len(top), figsize=(4 * len(top), 3.2))
        for a, c in zip(np.atleast_1d(ax), top):
            a.scatter(Xs[c], sv[:, list(Xs.columns).index(c)], s=3, alpha=.4); a.set_xlabel(c); a.set_ylabel("SHAP")
        save_fig(fig, "t1_shap_dependence")
        res["shap_top3"] = top
        angka.update(TsatuShapSatu=str(imp.index[0]), TsatuShapDua=str(imp.index[1]), TsatuShapTiga=str(imp.index[2]))

        # partial dependence manual (rata-rata prediksi saat satu fitur diubah ke nilai grid)
        pd_res = {}
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        base = Xva.sample(2000, random_state=1)
        for a, c in zip(ax, ["jarak_km", "antrean_hub_asal"]):
            grid = np.nanquantile(base[c], np.linspace(0.02, 0.98, 20))
            v50, v90 = [], []
            for g in grid:
                b = base.copy(); b[c] = g
                q50, q90 = m.predict(b); v50.append(float(q50.mean())); v90.append(float(q90.mean()))
            a.plot(grid, v50, label="P50"); a.plot(grid, v90, label="P90 (sebelum kalibrasi)"); a.set_xlabel(c); a.set_ylabel("durasi (jam)"); a.legend()
            pd_res[c] = {"grid": grid.tolist(), "p50": v50, "p90": v90}
        save_fig(fig, "t1_partial_dependence")
        res["partial_dependence"] = pd_res
    save_result("t1_analisis", res, angka)
    return res


if __name__ == "__main__":
    run()

"""Tugas 1 (b)–(e): validasi waktu vs random CV, perbandingan model + tuning, kategori, ketidakpastian (CQR)."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from .data import load_raw
from .common_t1 import prepare, TIME_FOLDS, fold_masks, make_xy
from .models import LGBQuantile, GroupQuantileBaseline, LinearLogModel, evaluate, LGB_BASE
from .conformal import CQRUpper
from ..common.splits import month_mask
from ..common.paths import QUICK, SEED, KODE
from ..common.io_utils import save_result, save_fig, save_table, Timer

PARAM_GRID = [  # ruang tuning kecil (dievaluasi HANYA pada fold waktu)
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 100},
    {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 700, "min_child_samples": 50, "colsample_bytree": 0.6},
    {"num_leaves": 255, "learning_rate": 0.05, "n_estimators": 250, "min_child_samples": 200},
]


def _fit_eval(model_cls, d, mtr, mva, high_card="te", **kw):
    Xtr, Xva, _ = make_xy(d[mtr], d[mva], high_card=high_card)
    m = model_cls(**kw).fit(Xtr, d.loc[mtr, "y"])
    p50, p90 = m.predict(Xva)
    return m, Xva, p50, p90


def run():
    tr, _ = load_raw()
    d = prepare(tr)
    res, angka = {}, {}

    # ---------------- (b) validasi waktu vs random CV
    with Timer("T1 (b) validasi waktu vs random CV"):
        rows = []
        for f in TIME_FOLDS:
            mtr, mva = fold_masks(d, f)
            _, _, p50, p90 = _fit_eval(LGBQuantile, d, mtr, mva)
            r = evaluate(d.loc[mva, "y_eval"], p50, p90); r.update(skema="waktu", fold=f["nama"]); rows.append(r)
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        for i, (a, b) in enumerate(kf.split(d)):
            mtr = d.index.isin(a); mva = d.index.isin(b)
            _, _, p50, p90 = _fit_eval(LGBQuantile, d, mtr, mva)
            r = evaluate(d.loc[mva, "y_eval"], p50, p90); r.update(skema="random", fold=f"R{i+1}"); rows.append(r)
        vdf = pd.DataFrame(rows)
        summ = vdf.groupby("skema")[["pinball50", "pinball90", "coverage90", "mae"]].mean()
        res["validasi"] = {"per_fold": vdf.to_dict("records"), "rata2": summ.reset_index().to_dict("records")}
        save_table(vdf[["skema", "fold", "pinball50", "pinball90", "coverage90", "mae", "n"]], "t1_validasi",
                   "Validasi berbasis waktu vs random 3-fold CV (LightGBM quantile, fitur sama).")
        angka.update(TsatuValWaktuPinbal=round(summ.loc["waktu", "pinball50"], 2),
                     TsatuValRandomPinbal=round(summ.loc["random", "pinball50"], 2),
                     TsatuValWaktuCov=round(summ.loc["waktu", "coverage90"], 3),
                     TsatuValRandomCov=round(summ.loc["random", "coverage90"], 3))

    # ---------------- (c) model: baseline, linear log-target, LightGBM quantile + tuning
    with Timer("T1 (c) perbandingan model + tuning"):
        rows = []
        grid = PARAM_GRID[:2] if QUICK else PARAM_GRID
        tune = {i: [] for i in range(len(grid))}
        for f in TIME_FOLDS:
            mtr, mva = fold_masks(d, f)
            Xtr, Xva, _ = make_xy(d[mtr], d[mva])
            ytr, yva = d.loc[mtr, "y"], d.loc[mva, "y_eval"]
            for nama, m in [("Baseline kuantil (layanan×pulau)", GroupQuantileBaseline()), ("Linear log-target (Ridge)", LinearLogModel())]:
                m.fit(Xtr, ytr); r = evaluate(yva, *m.predict(Xva)); r.update(model=nama, fold=f["nama"]); rows.append(r)
            for i, g in enumerate(grid):
                m = LGBQuantile(g).fit(Xtr, ytr); r = evaluate(yva, *m.predict(Xva)); r.update(fold=f["nama"], cfg=i)
                tune[i].append(r)
        tdf = pd.DataFrame([{**r, **grid[r["cfg"]]} for v in tune.values() for r in v])
        tavg = tdf.groupby("cfg")[["pinball50", "pinball90", "coverage90"]].mean()
        tavg["skor"] = tavg["pinball50"] + tavg["pinball90"]
        best_i = int(tavg["skor"].idxmin())
        best_params = grid[best_i]
        res["tuning"] = {"grid": grid, "hasil": tdf.to_dict("records"), "terbaik": best_i, "params_terbaik": best_params}
        for r in tune[0]:
            rows.append({**r, "model": "LightGBM quantile (default)"})
        for r in tune[best_i]:
            rows.append({**r, "model": f"LightGBM quantile (tuned, cfg {best_i})"})
        mdf = pd.DataFrame(rows)
        msum = mdf.groupby("model")[["pinball50", "pinball90", "coverage90", "mae"]].mean().sort_values("pinball50")
        res["model"] = {"per_fold": mdf.drop(columns=["cfg"], errors="ignore").to_dict("records"), "rata2": msum.reset_index().to_dict("records")}
        save_table(msum.reset_index(), "t1_model", "Perbandingan model, rata-rata 2 fold waktu (pinball lebih kecil lebih baik).")
        save_table(tavg.reset_index(), "t1_tuning", "Tuning LightGBM pada fold waktu (skor = pinball50 + pinball90).")
        b = msum.loc["Baseline kuantil (layanan×pulau)"]; l = msum.loc["Linear log-target (Ridge)"]
        g = msum.loc[f"LightGBM quantile (tuned, cfg {best_i})"]
        angka.update(TsatuBasePinbal=round(b.pinball50, 2), TsatuLinPinbal=round(l.pinball50, 2), TsatuLgbPinbal=round(g.pinball50, 2),
                     TsatuBasePinbalNilan=round(b.pinball90, 2), TsatuLinPinbalNilan=round(l.pinball90, 2), TsatuLgbPinbalNilan=round(g.pinball90, 2),
                     TsatuLgbCov=round(g.coverage90, 3), TsatuCfgTerbaik=best_i)

    # ---------------- (d) kategori kardinalitas tinggi + kategori baru
    with Timer("T1 (d) encoding kategori"):
        f = TIME_FOLDS[1]; mtr, mva = fold_masks(d, f)
        rows = []
        for hc in ["te", "freq", "native"]:
            _, _, p50, p90 = _fit_eval(LGBQuantile, d, mtr, mva, high_card=hc, params=best_params)
            r = evaluate(d.loc[mva, "y_eval"], p50, p90); r["encoding kota"] = hc; rows.append(r)
        edf = pd.DataFrame(rows)[["encoding kota", "pinball50", "pinball90", "coverage90"]]
        res["encoding"] = edf.to_dict("records")
        save_table(edf, "t1_encoding", "Encoding kota_tujuan/kota_asal (fold F2).")
        best_hc = edf.sort_values("pinball50").iloc[0]["encoding kota"]
        res["encoding_terpilih"] = best_hc

        # simulasi mitra baru: keluarkan satu mitra dari data latih (meniru K13 yang hanya ada di test)
        share = d.loc[mva, "mitra_kurir"].value_counts(normalize=True)
        held = (share - 0.039).abs().idxmin()
        mtr2 = mtr & (d["mitra_kurir"] != held)
        is_new = (d.loc[mva, "mitra_kurir"] == held).values
        rows = []
        for nama, kw in [("__OTHER__ tanpa dropout", dict(unknown_frac=0.0)), ("__OTHER__ + kategori dropout 5%", dict(unknown_frac=0.05)),
                         ("tanpa fitur mitra_kurir", dict(unknown_frac=0.0, drop_extra=["mitra_kurir"]))]:
            Xtr, Xva, _ = make_xy(d[mtr2], d[mva], high_card=best_hc, **kw)
            m = LGBQuantile(best_params).fit(Xtr, d.loc[mtr2, "y"])
            p50, p90 = m.predict(Xva)
            yv = d.loc[mva, "y_eval"].values
            r1 = evaluate(yv[is_new], p50[is_new], p90[is_new]); r2 = evaluate(yv[~is_new], p50[~is_new], p90[~is_new])
            rows.append({"strategi": nama, "pinball50_mitra_baru": r1["pinball50"], "coverage90_mitra_baru": r1["coverage90"],
                         "pinball50_mitra_lama": r2["pinball50"]})
        udf = pd.DataFrame(rows)
        # pilih strategi dengan pinball terbobot porsi test (±3.9% baris test adalah mitra baru K13)
        udf["pinball50_terbobot"] = 0.039 * udf["pinball50_mitra_baru"] + 0.961 * udf["pinball50_mitra_lama"]
        strat = udf.sort_values("pinball50_terbobot").iloc[0]["strategi"]
        res["mitra_baru"] = {"mitra_disimulasikan": held, "hasil": udf.to_dict("records"), "strategi_terpilih": strat}
        save_table(udf, "t1_mitra_baru", f"Simulasi kategori baru: mitra {held} dikeluarkan dari data latih (fold F2).")
        angka.update(TsatuMitraSim=str(held))

    # ---------------- (e) ketidakpastian: tanpa kalibrasi vs CQR 1 bulan vs CQR gabungan lintas bulan (global / per layanan)
    with Timer("T1 (e) kalibrasi P90 (CQR)"):
        from .conformal import pooled_scores
        from .common_t1 import make_xy_cfg
        cfg_tmp = {"params": best_params, "high_card": best_hc, "unknown_frac": 0.05, "drop_extra": []}
        rows, per_layanan = [], []
        for nama, lt, val in [("F1", ("2025-01", "2025-04"), ("2025-05", "2025-06")), ("F2", ("2025-01", "2025-06"), ("2025-07", "2025-08"))]:
            mtr = month_mask(d.tgl_pickup, *lt); mva = month_mask(d.tgl_pickup, *val)
            Xtr, Xva, _ = make_xy_cfg(d[mtr], d[mva], cfg_tmp)
            m = LGBQuantile(best_params).fit(Xtr, d.loc[mtr, "y"])
            p50, p90 = m.predict(Xva)
            yv = d.loc[mva, "y_eval"].values; lay_v = d.loc[mva, "layanan"].values
            sc = pooled_scores(d, cfg_tmp, last_month=lt[1])           # skor dari jendela yang selesai sebelum validasi
            last = sc[sc.akhir_latih == sc.akhir_latih.max()]          # hanya jendela terakhir (setara kalibrasi 1 periode)
            v = {"tanpa kalibrasi": p90,
                 "CQR jendela terakhir": CQRUpper().fit(last.y.values, last.p90.values).adjust(p90),
                 "CQR gabungan lintas bulan": CQRUpper().fit(sc.y.values, sc.p90.values).adjust(p90),
                 "CQR gabungan per layanan": CQRUpper(group_col="layanan").fit(sc.y.values, sc.p90.values, sc.layanan.values).adjust(p90, lay_v)}
            for varian, up in v.items():
                up = np.maximum(up, p50)
                r = evaluate(yv, p50, up); r.update(fold=nama, varian=varian, lebar_median=float(np.median(up - p50)), n_jendela=int(sc.akhir_latih.nunique()))
                rows.append(r)
                for lay in np.unique(lay_v):
                    mm_ = lay_v == lay
                    per_layanan.append({"fold": nama, "varian": varian, "layanan": lay, "coverage90": float(np.mean(yv[mm_] <= up[mm_]))})
        cdf = pd.DataFrame(rows)
        order = list(v.keys())
        csum = cdf.groupby("varian")[["pinball90", "coverage90", "lebar_median"]].mean().reindex(order)
        pl = pd.DataFrame(per_layanan).groupby(["varian", "layanan"])["coverage90"].mean().unstack().reindex(order)
        res["kalibrasi"] = {"per_fold": cdf.to_dict("records"), "rata2": csum.reset_index().to_dict("records"),
                            "coverage_per_layanan": pl.reset_index().to_dict("records")}
        save_table(cdf[["fold", "varian", "pinball90", "coverage90", "lebar_median"]], "t1_kalibrasi",
                   "Kalibrasi P90 per fold: tanpa kalibrasi, CQR jendela terakhir, CQR gabungan lintas bulan (global / per layanan).")
        save_table(pl.reset_index(), "t1_kalibrasi_layanan", "Coverage P90 per layanan, rata-rata F1–F2.")
        # Aturan pemilihan: nilai hanya pada fold yang punya >= 2 jendela kalibrasi (pada F1 hanya ada 1 jendela, yang
        # kebetulan memuat lonjakan Maret, sehingga "gabungan" = "jendela terakhir" dan tidak mewakili metode gabungan).
        # Di antara varian yang pinball90-nya tidak lebih buruk dari tanpa kalibrasi (toleransi 1%), pilih coverage terdekat 0.90.
        ev = cdf[cdf["n_jendela"] >= 2] if (cdf["n_jendela"] >= 2).any() else cdf
        es = ev.groupby("varian")[["pinball90", "coverage90"]].mean()
        ok = es[es["pinball90"] <= es.loc["tanpa kalibrasi", "pinball90"] * 1.01]
        pilih = (ok["coverage90"] - 0.90).abs().sort_values().index[0]
        res["kalibrasi_terpilih"] = pilih
        res["kalibrasi_aturan"] = {"fold_dinilai": sorted(ev["fold"].unique().tolist()), "ringkas": es.reset_index().to_dict("records")}
        angka.update(TsatuCovRawFdua=round(es.loc["tanpa kalibrasi", "coverage90"], 3), TsatuCovPilihFdua=round(es.loc[pilih, "coverage90"], 3))
        angka.update(TsatuCovSebelum=round(csum.loc["tanpa kalibrasi", "coverage90"], 3),
                     TsatuCovSatuJendela=round(csum.loc["CQR jendela terakhir", "coverage90"], 3),
                     TsatuCovGabungan=round(csum.loc["CQR gabungan lintas bulan", "coverage90"], 3),
                     TsatuCovMondrian=round(csum.loc["CQR gabungan per layanan", "coverage90"], 3))
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        pl.T.plot(kind="bar", ax=ax); ax.axhline(0.9, color="k", ls="--", lw=1); ax.set_ylim(0.6, 1.0)
        ax.set_ylabel("coverage P90"); ax.set_title("Coverage P90 per layanan (rata-rata F1–F2)"); ax.legend(fontsize=7)
        save_fig(fig, "t1_kalibrasi_layanan")

    config = {"params": best_params, "high_card": best_hc, "kalibrasi": pilih,
              "unknown_frac": 0.05 if "dropout" in strat else 0.0,
              "drop_extra": ["mitra_kurir"] if strat.startswith("tanpa") else []}
    (KODE / "configs" / "t1_terpilih.json").write_text(json.dumps(config, indent=2))
    res["konfigurasi_final"] = config
    save_result("t1_experimen", res, angka)
    return res


if __name__ == "__main__":
    run()

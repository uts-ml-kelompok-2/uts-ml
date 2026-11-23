"""Tugas 2 (b)–(f): validasi waktu + label belum matang, fitur perilaku, imbalance, pergeseran pola, kalibrasi & keputusan."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from .common_t2 import get_features, FOLDS, masks, weekly_rate, estimate_mature_cutoff
from .models import (feature_cols, to_X, fit_lgb, fit_logreg, predict, evaluate, Calibrator, flag_cost_rule,
                     best_threshold, ece, LGB_PARAMS)
from ..common.metrics import cost_flag_all, cost_flag_none, total_cost
from ..common.splits import month_mask
from ..common.paths import KODE, SEED, QUICK
from ..common.io_utils import save_result, save_fig, save_table, Timer


def time_weights(df, mask, adaptasi):
    """Bobot sampel untuk adaptasi 'bobot waktu (paruh 60 hari)'; None untuk strategi lain."""
    if adaptasi != "bobot waktu (paruh 60 hari)":
        return None
    age = (df.loc[mask, "waktu"].max() - df.loc[mask, "waktu"]).dt.days.values
    return 0.5 ** (age / 60.0)


def _fit_variant(df, mtr, cols, variant, params=None, weight=None):
    """Latih LightGBM dengan satu strategi imbalance (opsional: bobot sampel untuk adaptasi waktu)."""
    X, y = to_X(df[mtr], cols), df.loc[mtr, "is_fraud"].astype(int).values
    if variant == "tanpa penanganan":
        return fit_lgb(X, y, weight=weight, params=params)
    if variant == "class weight (scale_pos_weight)":
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        return fit_lgb(X, y, weight=weight, params={**(params or {}), "scale_pos_weight": float(np.sqrt(spw))})
    if variant == "undersampling negatif 1:20":
        rng = np.random.RandomState(SEED)
        keep = (y == 1) | (rng.rand(len(y)) < min(1.0, 20 * (y == 1).sum() / max((y == 0).sum(), 1)))
        return fit_lgb(X[keep], y[keep], weight=None if weight is None else weight[keep], params=params)
    if variant == "SMOTE (rasio 1:10)":
        Xn = X.copy()
        for c in Xn.columns:
            if str(Xn[c].dtype) == "category":
                Xn[c] = Xn[c].cat.codes
        Xn = Xn.fillna(Xn.median(numeric_only=True)).fillna(0).astype(float)
        Xs, ys = SMOTE(sampling_strategy=0.1, random_state=SEED, k_neighbors=5).fit_resample(Xn, y)
        m = fit_lgb(Xs, ys, params=params); m._smote_cols = list(Xn.columns); m._smote_med = Xn.median()
        return m
    raise ValueError(variant)


def _pred(m, df, mask, cols):
    X = to_X(df[mask], cols)
    if hasattr(m, "_smote_cols"):
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].cat.codes
        X = X.fillna(m._smote_med).fillna(0).astype(float)
    return predict(m, X)


def _decide(p_kal, y_kal, nom_kal, p_val, nom_val):
    thr = best_threshold(p_kal, y_kal, nom_kal)
    return flag_cost_rule(p_val, nom_val), (p_val >= thr).astype(int), thr


def run():
    df = get_features()
    cols = feature_cols(df)
    res, angka = {}, {}
    cut, base = estimate_mature_cutoff(df)

    # ---------------- (b1) validasi waktu vs random stratified CV
    with Timer("T2 (b) validasi waktu vs random CV"):
        rows = []
        for f in FOLDS:
            mtr, mka, mva = masks(df, f)
            m = fit_lgb(to_X(df[mtr | mka], cols), df.loc[mtr | mka, "is_fraud"])
            r = evaluate(df.loc[mva, "is_fraud"], predict(m, to_X(df[mva], cols)), df.loc[mva, "nominal"]); r.update(skema="waktu", fold=f["nama"]); rows.append(r)
        lab = df[(df.set == "train") & (df.waktu < "2025-08-01")].reset_index(drop=True)
        for i, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(lab, lab.is_fraud)):
            m = fit_lgb(to_X(lab.iloc[a], cols), lab.is_fraud.iloc[a])
            r = evaluate(lab.is_fraud.iloc[b], predict(m, to_X(lab.iloc[b], cols)), lab.nominal.iloc[b]); r.update(skema="random", fold=f"R{i+1}"); rows.append(r)
        vdf = pd.DataFrame(rows)
        vs = vdf.groupby("skema")[["pr_auc", "roc_auc"]].mean()
        res["validasi"] = vdf.to_dict("records")
        save_table(vdf[["skema", "fold", "pr_auc", "roc_auc", "n_fraud"]], "t2_validasi", "Validasi berbasis waktu (label matang) vs random stratified 3-fold.")
        angka.update(TduaValWaktuPR=round(vs.loc["waktu", "pr_auc"], 3), TduaValRandomPR=round(vs.loc["random", "pr_auc"], 3))

    # ---------------- (b2) simulasi label belum matang di akhir data latih
    with Timer("T2 (b) simulasi label belum matang"):
        w = weekly_rate(df); vis = (w["mean"].tail(4) / base).clip(upper=1).values[1:]  # 3 minggu terakhir yang turun
        f = FOLDS[1]; mtr, mka, mva = masks(df, f); mlt = mtr | mka
        end = df.loc[mlt, "waktu"].max().normalize()
        y_sim = df["is_fraud"].copy()
        rng = np.random.RandomState(SEED)
        for k, v in enumerate(vis[::-1]):                    # minggu terakhir paling tidak matang
            wk = mlt & (df.waktu > end - pd.Timedelta(days=7 * (k + 1))) & (df.waktu <= end - pd.Timedelta(days=7 * k))
            hide = wk & (df.is_fraud == 1) & (rng.rand(len(df)) > v)
            y_sim[hide] = 0
        start_imm = end - pd.Timedelta(days=7 * len(vis))
        rows = []
        for nama, mask, yy in [("label lengkap (acuan, tidak realistis)", mlt, df.is_fraud),
                               ("label belum matang dipakai apa adanya", mlt, y_sim),
                               ("buang periode belum matang (dipakai)", mlt & (df.waktu <= start_imm), y_sim)]:
            m = fit_lgb(to_X(df[mask], cols), yy[mask].astype(int))
            p = predict(m, to_X(df[mva], cols))
            r = evaluate(df.loc[mva, "is_fraud"], p, df.loc[mva, "nominal"]); r["strategi"] = nama
            r["rata2_prob_val"] = float(p.mean()); r["fraud_tersembunyi"] = int(((df.is_fraud == 1) & (y_sim == 0) & mlt).sum()); rows.append(r)
        ldf = pd.DataFrame(rows)[["strategi", "pr_auc", "roc_auc", "rata2_prob_val", "fraud_tersembunyi"]]
        real = ldf.iloc[1:]
        lag_pilih = "buang" if real.iloc[1]["pr_auc"] >= real.iloc[0]["pr_auc"] - 0.005 else "apa adanya"
        res["label_lag"] = {"visibilitas_dipakai": vis.tolist(), "hasil": ldf.to_dict("records"), "terpilih": lag_pilih}
        save_table(ldf, "t2_label_lag", "Simulasi label belum matang pada 3 minggu terakhir data latih (fold F2).")

    # ---------------- (c) fitur perilaku
    with Timer("T2 (c) fitur perilaku"):
        rows = []
        for f in FOLDS:
            mtr, mka, mva = masks(df, f)
            for nama, cc in [("tanpa fitur perilaku buatan", feature_cols(df, behav=False)), ("+ fitur perilaku (dipakai)", cols)]:
                m = fit_lgb(to_X(df[mtr | mka], cc), df.loc[mtr | mka, "is_fraud"])
                r = evaluate(df.loc[mva, "is_fraud"], predict(m, to_X(df[mva], cc)), df.loc[mva, "nominal"]); r.update(fitur=nama, fold=f["nama"]); rows.append(r)
                if f["nama"] == "F2" and "dipakai" in nama:
                    imp = pd.Series(m.booster_.feature_importance("gain"), index=cc).sort_values(ascending=False)
        bdf = pd.DataFrame(rows)
        bs = bdf.groupby("fitur")[["pr_auc", "roc_auc"]].mean()
        res["fitur_perilaku"] = {"per_fold": bdf.to_dict("records"), "importance_gain_top": (imp.head(20) / imp.sum()).to_dict()}
        save_table(bs.reset_index(), "t2_fitur_perilaku", "Dampak fitur perilaku buatan (rata-rata F1–F2).")
        save_table((imp.head(10) / imp.sum()).rename("porsi gain").reset_index().rename(columns={"index": "fitur"}), "t2_importance", "10 fitur terpenting (gain) model F2.")
        angka.update(TduaPRTanpaPerilaku=round(bs.loc["tanpa fitur perilaku buatan", "pr_auc"], 3), TduaPRPerilaku=round(bs.loc["+ fitur perilaku (dipakai)", "pr_auc"], 3))

    # ---------------- (d) imbalance (+ kalibrasi isotonic per varian supaya aturan biaya adil)
    with Timer("T2 (d) class imbalance"):
        variants = ["tanpa penanganan", "class weight (scale_pos_weight)", "undersampling negatif 1:20", "SMOTE (rasio 1:10)"]
        if QUICK:
            variants = variants[:3]
        rows = []
        for f in FOLDS:
            mtr, mka, mva = masks(df, f)
            yk, nk = df.loc[mka, "is_fraud"].values.astype(int), df.loc[mka, "nominal"].values
            yv, nv = df.loc[mva, "is_fraud"].values.astype(int), df.loc[mva, "nominal"].values
            lr = fit_logreg(to_X(df[mtr], cols), df.loc[mtr, "is_fraud"].astype(int))
            for nama, getp in [("Logistic regression (baseline)", lambda msk: predict(lr, to_X(df[msk], cols)))] + \
                              [(v, (lambda vv: (lambda msk, _m=_fit_variant(df, mtr, cols, vv): _pred(_m, df, msk, cols)))(v)) for v in variants]:
                pk, pv = getp(mka), getp(mva)
                cal = Calibrator("isotonic").fit(pk, yk); pvc = cal.transform(pv); pkc = cal.transform(pk)
                fl_rule, fl_thr, thr = _decide(pkc, yk, nk, pvc, nv)
                r = evaluate(yv, pv, nv, fl_rule); r.update(fold=f["nama"], strategi=nama, biaya_threshold_tunggal=total_cost(yv, nv, fl_thr),
                                                          threshold_tunggal=thr, ece_mentah=ece(pv, yv), ece_kalibrasi=ece(pvc, yv))
                rows.append(r)
        idf = pd.DataFrame(rows)
        isum = idf.groupby("strategi")[["pr_auc", "roc_auc", "biaya", "biaya_threshold_tunggal", "recall", "presisi", "ece_mentah"]].mean().sort_values("biaya")
        res["imbalance"] = {"per_fold": idf.to_dict("records"), "rata2": isum.reset_index().to_dict("records")}
        t = isum.reset_index().copy(); t["biaya"] = t["biaya"] / 1e6; t["biaya_threshold_tunggal"] = t["biaya_threshold_tunggal"] / 1e6
        save_table(t.rename(columns={"biaya": "biaya aturan (juta Rp)", "biaya_threshold_tunggal": "biaya thr tunggal (juta Rp)"}), "t2_imbalance",
                   "Strategi class imbalance (rata-rata F1–F2). Probabilitas dikalibrasi isotonic di bulan kalibrasi sebelum aturan biaya.")
        best_imb = isum.drop(index="Logistic regression (baseline)", errors="ignore").sort_values(["biaya", "pr_auc"], ascending=[True, False]).index[0]
        res["imbalance_terpilih"] = best_imb
        angka.update(TduaImbTerpilih=best_imb, TduaPRLogreg=round(isum.loc["Logistic regression (baseline)", "pr_auc"], 3))

    # ---------------- (e) pergeseran pola
    with Timer("T2 (e) pergeseran pola"):
        tr = df[df.set == "train"]
        mk = tr.groupby(["bulan", "kanal"]).is_fraud.mean().unstack() * 100
        res["drift_kanal_bulan"] = mk.round(3).reset_index().to_dict("records")
        fig, ax = plt.subplots(figsize=(8, 3.4)); mk.plot(marker="o", ax=ax); ax.set_ylabel("% fraud"); ax.set_title("Tingkat fraud per kanal per bulan (Agu: label belum matang)")
        save_fig(fig, "t2_drift_kanal")
        # adversarial validation: bisakah model membedakan train (Jun–Agu) vs test? Bandingkan sebelum/sesudah
        # membuang fitur kumulatif (DRIFT_COLS) yang ternyata menjadi penanda waktu.
        from sklearn.metrics import roc_auc_score
        from .features import DRIFT_COLS
        adv = df[(df.waktu >= "2025-06-01")].copy(); adv["is_test"] = (adv.set == "test").astype(int)
        a_idx, b_idx = next(StratifiedKFold(3, shuffle=True, random_state=SEED).split(adv, adv.is_test))
        f2 = FOLDS[1]; m2tr, m2ka, m2va = masks(df, f2)
        adv_rows = {}
        for nama, cc in [("dengan fitur kumulatif", feature_cols(df, keep=DRIFT_COLS)), ("tanpa fitur kumulatif (dipakai)", cols)]:
            cca = [c for c in cc if c != "menit_sejak_trx_terakhir"]
            ma = fit_lgb(to_X(adv.iloc[a_idx], cca), adv.is_test.iloc[a_idx], params={"n_estimators": 150})
            auc = roc_auc_score(adv.is_test.iloc[b_idx], predict(ma, to_X(adv.iloc[b_idx], cca)))
            impa = pd.Series(ma.booster_.feature_importance("gain"), index=cca)
            mm = fit_lgb(to_X(df[m2tr | m2ka], cc), df.loc[m2tr | m2ka, "is_fraud"])
            prf = evaluate(df.loc[m2va, "is_fraud"], predict(mm, to_X(df[m2va], cc)), df.loc[m2va, "nominal"])["pr_auc"]
            adv_rows[nama] = {"adv_auc": float(auc), "pr_auc_F2": prf, "top": (impa.sort_values(ascending=False).head(6) / impa.sum()).round(3).to_dict()}
        res["adversarial"] = adv_rows
        save_table(pd.DataFrame([{"fitur": k, "AUC adversarial (train vs test)": v["adv_auc"], "PR-AUC F2": v["pr_auc_F2"]} for k, v in adv_rows.items()]),
                   "t2_adversarial", "Adversarial validation (Jun–Agu vs test Sep–Okt) sebelum/sesudah membuang fitur kumulatif.")
        angka.update(TduaAdvAUCSebelum=round(adv_rows["dengan fitur kumulatif"]["adv_auc"], 3),
                     TduaAdvAUC=round(adv_rows["tanpa fitur kumulatif (dipakai)"]["adv_auc"], 3))
        # adaptasi: semua data vs jendela terbaru vs bobot waktu (fold F2)
        f = FOLDS[1]; mtr, mka, mva = masks(df, f); mlt = mtr | mka
        age = (df.loc[mlt, "waktu"].max() - df.loc[mlt, "waktu"]).dt.days.values
        rows = []
        for nama, mask, wgt in [("semua data latih", mlt, None), ("3 bulan terakhir saja", mlt & (df.waktu >= "2025-03-01"), None),
                                ("bobot waktu (paruh 60 hari)", mlt, 0.5 ** (age / 60.0))]:
            ww = None if wgt is None else wgt
            m = fit_lgb(to_X(df[mask], cols), df.loc[mask, "is_fraud"], weight=ww)
            r = evaluate(df.loc[mva, "is_fraud"], predict(m, to_X(df[mva], cols)), df.loc[mva, "nominal"]); r["adaptasi"] = nama; rows.append(r)
        adf = pd.DataFrame(rows)[["adaptasi", "pr_auc", "roc_auc"]]
        res["adaptasi"] = adf.to_dict("records")
        save_table(adf, "t2_adaptasi", "Adaptasi terhadap pergeseran pola (fold F2).")
        best_adapt = adf.sort_values("pr_auc", ascending=False).iloc[0]["adaptasi"]
        res["adaptasi_terpilih"] = best_adapt

    # ---------------- (f) kalibrasi + keputusan
    with Timer("T2 (f) kalibrasi dan keputusan biaya"):
        rows, rel = [], {}
        for f in FOLDS:
            mtr, mka, mva = masks(df, f)
            m = _fit_variant(df, mtr, cols, best_imb)
            pk, pv = _pred(m, df, mka, cols), _pred(m, df, mva, cols)
            yk, nk = df.loc[mka, "is_fraud"].values.astype(int), df.loc[mka, "nominal"].values
            yv, nv = df.loc[mva, "is_fraud"].values.astype(int), df.loc[mva, "nominal"].values
            for meth in ["none", "sigmoid", "isotonic"]:
                cal = Calibrator(meth).fit(pk, yk) if meth != "none" else Calibrator("none")
                pvc = cal.transform(pv); pkc = cal.transform(pk)
                fl_rule, fl_thr, thr = _decide(pkc, yk, nk, pvc, nv)
                rows.append({"fold": f["nama"], "kalibrasi": meth, "brier": float(np.mean((pvc - yv) ** 2)), "ece": ece(pvc, yv),
                             "biaya_aturan_p_x_nominal": total_cost(yv, nv, fl_rule), "biaya_threshold_tunggal": total_cost(yv, nv, fl_thr),
                             "biaya_tandai_semua": cost_flag_all(yv, nv), "biaya_tidak_menandai": cost_flag_none(yv, nv),
                             "n_ditandai_aturan": int(fl_rule.sum()), "recall_aturan": float((fl_rule * yv).sum() / max(yv.sum(), 1))})
                if f["nama"] == "F2":
                    q = np.quantile(pvc, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(q, pvc, side="right") - 1, 0, 9)
                    rel[meth] = [(float(pvc[idx == b].mean()), float(yv[idx == b].mean())) for b in range(10) if (idx == b).any()]
        cdf = pd.DataFrame(rows)
        csum = cdf.groupby("kalibrasi").mean(numeric_only=True)
        res["kalibrasi"] = {"per_fold": cdf.to_dict("records"), "rata2": csum.reset_index().to_dict("records"), "reliability_F2": rel}
        t = csum[["brier", "ece", "biaya_aturan_p_x_nominal", "biaya_threshold_tunggal", "biaya_tandai_semua", "biaya_tidak_menandai"]].copy()
        for c in t.columns:
            if c.startswith("biaya"):
                t[c] = t[c] / 1e6
        save_table(t.reset_index(), "t2_kalibrasi_biaya", "Kalibrasi (Brier, ECE) dan biaya (juta Rp) di validasi, rata-rata F1–F2.", float_fmt="%.4f")
        best_cal = csum.sort_values(["biaya_aturan_p_x_nominal", "brier"]).index[0]
        res["kalibrasi_terpilih"] = best_cal
        b = csum.loc[best_cal]
        angka.update(TduaKalTerpilih=best_cal, TduaBiayaAturan=round(b.biaya_aturan_p_x_nominal / 1e6, 1), TduaBiayaThr=round(b.biaya_threshold_tunggal / 1e6, 1),
                     TduaBiayaSemua=round(b.biaya_tandai_semua / 1e6, 1), TduaBiayaTidak=round(b.biaya_tidak_menandai / 1e6, 1),
                     TduaRecallAturan=round(b.recall_aturan * 100, 1))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for meth, pts in rel.items():
            if pts:
                x, y = zip(*pts); ax.plot(x, y, marker="o", label=meth)
        mx = max(max(p[0] for p in v) for v in rel.values() if v)
        ax.plot([0, mx], [0, mx], "k--", lw=1); ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("probabilitas prediksi"); ax.set_ylabel("frekuensi fraud aktual"); ax.set_title("Reliability diagram (F2)"); ax.legend()
        save_fig(fig, "t2_reliability")

    config = {"imbalance": best_imb, "kalibrasi": best_cal, "adaptasi": best_adapt,
              "batas_label_matang": str(cut.date()) if cut is not None else None, "label_belum_matang": lag_pilih}
    (KODE / "configs" / "t2_terpilih.json").write_text(json.dumps(config, indent=2))
    res["konfigurasi_final"] = config
    save_result("t2_experimen", res, angka)
    return res


if __name__ == "__main__":
    run()

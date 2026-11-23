"""Tugas 2 (a): audit pembentukan label dan fitur yang tidak tersedia saat transaksi."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from .data import load_raw
from .common_t2 import get_features, FOLDS, masks, weekly_rate, estimate_mature_cutoff
from .models import feature_cols, to_X, fit_lgb, predict, evaluate
from ..common.io_utils import save_result, save_fig, save_table, Timer


def run():
    tr, te = load_raw()
    df = get_features()
    res, angka = {}, {}

    # --- label belum matang
    w = weekly_rate(df)
    cut, base = estimate_mature_cutoff(df)
    res["mingguan"] = w.reset_index().rename(columns={"waktu": "minggu"}).to_dict("records")
    res["batas_label_matang"] = str(cut.date()) if cut is not None else None
    res["tingkat_dasar_mingguan"] = float(base)
    tail = w.tail(4)
    vis = (tail["mean"] / base).clip(upper=1.0)
    res["perkiraan_label_terlihat_4_minggu_terakhir"] = vis.round(3).tolist()
    angka.update(TduaFraudRate=round(tr.is_fraud.mean() * 100, 2), TduaBatasMatang=str(cut.date()) if cut is not None else "-",
                 TduaRateDasarMingguan=round(base * 100, 2), TduaRateMingguTerakhir=round(float(tail["mean"].iloc[-1]) * 100, 2))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(w.index, w["mean"] * 100, marker=".")
    ax.axhline(base * 100, color="gray", ls="--", lw=1, label="median Jan–Jul")
    if cut is not None:
        ax.axvline(cut, color="r", ls="--", lw=1, label=f"label belum matang (≥ {cut.date()})")
    ax.set_ylabel("% fraud per minggu"); ax.set_title("Tingkat fraud mingguan: turun di akhir data karena label belum terkonfirmasi"); ax.legend()
    save_fig(fig, "t2_audit_label")

    # --- fitur bocor / penanda waktu
    f = tr[tr.is_fraud == 1]
    res["flag_review_manual"] = {
        "fraud_rate_jika_1": float(tr.loc[tr.flag_review_manual == 1, "is_fraud"].mean()),
        "fraud_rate_jika_0": float(tr.loc[tr.flag_review_manual == 0, "is_fraud"].mean()),
        "porsi_fraud_yang_direview": float(f.flag_review_manual.mean()),
        "rate_review_train": float(tr.flag_review_manual.mean()), "rate_review_test": float(te.flag_review_manual.mean()),
        "auc_tunggal": float(roc_auc_score(tr.is_fraud, tr.flag_review_manual))}
    fr = res["flag_review_manual"]
    angka.update(TduaReviewFraudJikaSatu=round(fr["fraud_rate_jika_1"] * 100, 1), TduaReviewPorsiFraud=round(fr["porsi_fraud_yang_direview"] * 100, 1),
                 TduaReviewAUC=round(fr["auc_tunggal"], 3), TduaReviewRateTrain=round(fr["rate_review_train"] * 100, 2),
                 TduaReviewRateTest=round(fr["rate_review_test"] * 100, 2))
    v15 = pd.concat([tr.groupby("bulan").v15.median(), te.groupby("bulan").v15.median()])
    res["v15_per_bulan"] = v15.to_dict()

    rows = []
    for c in [c for c in tr.select_dtypes("number").columns if c != "is_fraud"]:
        a, b = tr[c].dropna(), te[c].dropna()
        ks = ks_2samp(a.sample(min(60000, len(a)), random_state=0), b.sample(min(60000, len(b)), random_state=0)).statistic
        au = roc_auc_score(tr.is_fraud, tr[c].fillna(tr[c].median())); rows.append({"fitur": c, "KS train-test": ks, "AUC tunggal": max(au, 1 - au)})
    kt = pd.DataFrame(rows)
    res["ks_auc"] = kt.to_dict("records")
    save_table(kt.sort_values("AUC tunggal", ascending=False).head(8), "t2_audit_auc", "Fitur dengan AUC univariat tertinggi.")
    save_table(kt.sort_values("KS train-test", ascending=False).head(6), "t2_audit_ks", "Fitur dengan beda distribusi train–test terbesar.")
    first = tr.menit_sejak_trx_terakhir.isna()
    res["na_transaksi_pertama"] = {"persen_na": float(first.mean() * 100),
                                   "na_sama_dengan_rasio_na": bool((first == tr.rasio_nominal_rata2.isna()).all())}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].bar(["review=0", "review=1"], [fr["fraud_rate_jika_0"] * 100, fr["fraud_rate_jika_1"] * 100]); ax[0].set_ylabel("% fraud")
    ax[0].set_title("flag_review_manual vs fraud")
    ax[1].plot(v15.index, v15.values, marker="o"); ax[1].tick_params(axis="x", rotation=45); ax[1].set_title("v15: median per bulan (penanda waktu)")
    save_fig(fig, "t2_audit_bocor")

    # --- dampak (fold F2)
    with Timer("T2 audit: dampak fitur bermasalah"):
        mtr, mka, mva = masks(df, FOLDS[1]); mtr = mtr | mka
        rows = []
        for nama, keep in [("tanpa fitur bermasalah (dipakai)", []), ("+ flag_review_manual", ["flag_review_manual"]), ("+ v15", ["v15"])]:
            cols = feature_cols(df, keep=keep)
            m = fit_lgb(to_X(df[mtr], cols), df.loc[mtr, "is_fraud"])
            r = evaluate(df.loc[mva, "is_fraud"], predict(m, to_X(df[mva], cols)), df.loc[mva, "nominal"]); r["varian"] = nama; rows.append(r)
        idf = pd.DataFrame(rows)[["varian", "pr_auc", "roc_auc"]]
        res["dampak"] = idf.to_dict("records")
        save_table(idf, "t2_audit_dampak", "Dampak fitur bermasalah (fold F2: latih Jan–Mei, val Jun–Jul).")
        angka.update(TduaPRAUCBersih=round(idf.iloc[0].pr_auc, 3), TduaPRAUCReview=round(idf.iloc[1].pr_auc, 3))
    save_result("t2_audit", res, angka)
    return res


if __name__ == "__main__":
    run()

"""Tugas 1 (a): audit kualitas data dan fitur yang tidak tersedia saat pickup. Semua klaim disertai bukti + dampak."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, spearmanr
from .data import load_raw, UNIT_ERROR_THRESHOLD, LEAK_COLS, TIME_PROXY_COLS
from .common_t1 import prepare, TIME_FOLDS, fold_masks, make_xy
from .models import LGBQuantile, evaluate
from ..common.io_utils import save_result, save_fig, save_table, Timer


def run():
    tr, te = load_raw()
    out, angka = {}, {}
    y = tr["durasi_jam"].astype(float)

    # --- 1. kesalahan satuan durasi
    bad = y > UNIT_ERROR_THRESHOLD
    out["durasi_ekstrem"] = {"n": int(bad.sum()), "persen": float(bad.mean() * 100), "maks": float(y.max()),
                             "median_normal": float(y[~bad].median()), "median_ekstrem_dibagi60": float((y[bad] / 60).median()),
                             "q99_normal": float(y[~bad].quantile(0.99))}
    angka.update(TsatuDurasiEkstremN=int(bad.sum()), TsatuDurasiEkstremPersen=round(bad.mean() * 100, 2),
                 TsatuDurasiMaks=round(float(y.max())), TsatuMedianNormal=round(float(y[~bad].median()), 1),
                 TsatuMedianEkstremBagi=round(float((y[bad] / 60).median()), 1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(np.log10(y), bins=100); ax[0].axvline(np.log10(UNIT_ERROR_THRESHOLD), color="r", ls="--")
    ax[0].set_title("log10(durasi_jam) mentah"); ax[0].set_xlabel("log10 jam")
    ax[1].hist(y[~bad].clip(upper=300), bins=80, alpha=.6, density=True, label="normal")
    ax[1].hist((y[bad] / 60).clip(upper=300), bins=40, alpha=.6, density=True, label="ekstrem ÷ 60")
    ax[1].legend(); ax[1].set_title("Nilai ekstrem ÷ 60 jatuh di distribusi normal")
    save_fig(fig, "t1_audit_durasi")

    # --- 2. nilai tidak valid / missing
    q = {"jarak_9999": float((tr.jarak_km >= 9999).mean() * 100), "jarak_0": float((tr.jarak_km <= 0).mean() * 100),
         "jarak_9999_test": float((te.jarak_km >= 9999).mean() * 100), "berat_na": float(tr.berat_kg.isna().mean() * 100),
         "berat_na_test": float(te.berat_kg.isna().mean() * 100), "duplikat_id": int(tr.id_paket.duplicated().sum()),
         "volume_maks": float(tr.volume_liter.max()), "berat_maks": float(tr.berat_kg.max())}
    out["kualitas"] = q
    angka.update(TsatuJarakSentinelPersen=round(q["jarak_9999"], 2), TsatuJarakNolPersen=round(q["jarak_0"], 2),
                 TsatuBeratNAPersen=round(q["berat_na"], 1))

    # --- 3. ketersediaan fitur saat pickup: KS train vs test + Spearman dengan target
    yfix = y.where(~bad, y / 60)
    rows = []
    for c in [c for c in tr.select_dtypes("number").columns if c not in ("durasi_jam",)]:
        a, b = tr[c].dropna(), te[c].dropna()
        ks = ks_2samp(a.sample(min(50000, len(a)), random_state=0), b.sample(min(50000, len(b)), random_state=0)).statistic
        rows.append({"fitur": c, "ks_train_test": ks, "spearman_target": spearmanr(tr[c], yfix, nan_policy="omit").statistic,
                     "mean_train": a.mean(), "mean_test": b.mean()})
    kst = pd.DataFrame(rows).sort_values("ks_train_test", ascending=False)
    out["ks_tabel"] = kst.to_dict("records")
    save_table(kst.head(8).rename(columns={"ks_train_test": "KS", "spearman_target": "Spearman"}), "t1_audit_ks",
               "Fitur dengan beda distribusi train–test terbesar (KS) dan korelasi Spearman dengan durasi.")
    s = tr.jumlah_scan
    angka.update(TsatuScanMeanTrain=round(s.mean(), 1), TsatuScanMeanTest=round(te.jumlah_scan.mean(), 1),
                 TsatuScanKS=round(float(kst.set_index("fitur").loc["jumlah_scan", "ks_train_test"]), 2),
                 TsatuScanSpearman=round(float(kst.set_index("fitur").loc["jumlah_scan", "spearman_target"]), 2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(s.clip(upper=40), bins=40, alpha=.6, density=True, label="train"); ax[0].hist(te.jumlah_scan.clip(upper=40), bins=40, alpha=.6, density=True, label="test")
    ax[0].legend(); ax[0].set_title("jumlah_scan: train vs test")
    mon = pd.concat([tr.groupby("bulan")["f30"].median(), te.groupby("bulan")["f30"].median()])
    ax[1].plot(mon.index, mon.values, marker="o"); ax[1].set_title("f30: median per bulan (penanda waktu)"); ax[1].tick_params(axis="x", rotation=45)
    save_fig(fig, "t1_audit_bocor")
    out["f30_per_bulan"] = mon.to_dict()

    # --- 4. dampak: model dengan vs tanpa fitur bermasalah, dan perlakuan target (fold waktu F2)
    with Timer("T1 audit: dampak fitur bocor & perlakuan target"):
        d = prepare(tr)
        mtr, mva = fold_masks(d, TIME_FOLDS[1])
        imp = []
        for nama, keep in [("tanpa fitur bermasalah (dipakai)", []), ("+ jumlah_scan (bocor)", ["jumlah_scan"]),
                           ("+ f30 (penanda waktu)", ["f30"])]:
            Xtr, Xva, _ = make_xy(d[mtr], d[mva], keep=keep)
            m = LGBQuantile().fit(Xtr, d.loc[mtr, "y"])
            r = evaluate(d.loc[mva, "y_eval"], *m.predict(Xva)); r["varian"] = nama
            # simulasi kondisi test untuk jumlah_scan: di test scan baru 1–2 kali. Isi validasi dengan distribusi test.
            if "jumlah_scan" in keep:
                Xsim = Xva.copy()
                Xsim["jumlah_scan"] = np.random.RandomState(0).choice(te.jumlah_scan.values, size=len(Xsim))
                rs = evaluate(d.loc[mva, "y_eval"], *m.predict(Xsim)); rs["varian"] = nama + ", nilai seperti di test"
                imp.append(r); imp.append(rs); continue
            imp.append(r)
        impdf = pd.DataFrame(imp)[["varian", "pinball50", "pinball90", "coverage90", "mae"]]
        out["dampak_fitur"] = impdf.to_dict("records")
        save_table(impdf, "t1_audit_dampak", "Dampak fitur bermasalah pada validasi waktu F2 (latih Jan–Jun, val Jul–Agu).")
        base = impdf.iloc[0]
        angka.update(TsatuPinbalBersih=round(base.pinball50, 2),
                     TsatuPinbalScan=round(impdf.iloc[1].pinball50, 2), TsatuPinbalScanTest=round(impdf.iloc[2].pinball50, 2))

        tm = []
        for mode in ["bagi60", "buang", "biarkan"]:
            dd = prepare(tr, target_mode=mode)
            Xtr, Xva, _ = make_xy(dd[mtr], dd[mva])
            m = LGBQuantile().fit(Xtr, dd.loc[mtr, "y"])
            r = evaluate(dd.loc[mva, "y_eval"], *m.predict(Xva)); r["perlakuan"] = mode; tm.append(r)
        tmdf = pd.DataFrame(tm)[["perlakuan", "pinball50", "pinball90", "coverage90"]]
        out["perlakuan_target"] = tmdf.to_dict("records")
        save_table(tmdf, "t1_audit_target", "Perlakuan durasi ekstrem (> 500 jam), dievaluasi pada label yang sudah dikoreksi.")
        best = tmdf.sort_values("pinball90").iloc[0]["perlakuan"]
        out["perlakuan_terpilih"] = best

    out["keputusan"] = {"dibuang_bocor": LEAK_COLS, "dibuang_penanda_waktu": TIME_PROXY_COLS,
                        "durasi_ekstrem": best, "jarak_sentinel": "NaN + indikator jarak_tidak_valid",
                        "berat_na": "biarkan NaN (LightGBM) + indikator berat_hilang"}
    save_result("t1_audit", out, angka)
    return out


if __name__ == "__main__":
    run()

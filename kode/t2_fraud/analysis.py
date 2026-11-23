"""Tugas 2 (g): breakdown error per kanal & kategori merchant, dan pola fraud yang ditemukan (fold F2, model final)."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common_t2 import get_features, FOLDS, masks
from .models import feature_cols, to_X, Calibrator, flag_cost_rule
from .experiments import _fit_variant, _pred, time_weights
from ..common.metrics import pr_auc, total_cost
from ..common.paths import KODE, MOD
from ..common.io_utils import save_result, save_fig, save_table, Timer

POLA = {
    "pengambilalihan akun (ganti password 7h + perangkat baru)": lambda d: (d.ganti_password_7h > 0) & (d.perangkat_baru == 1),
    "akses dari IP luar negeri": lambda d: d.negara_ip_beda == 1,
    "card testing (≥3 trx/1 jam, nominal < 50rb)": lambda d: (d.trx_1j >= 3) & (d.nominal < 50_000),
    "kartu tidak hadir + jarak jauh (> 500 km)": lambda d: (d.kartu_tidak_hadir == 1) & (d.jarak_lokasi_km > 500),
    "merchant refund tinggi (> 10%)": lambda d: d.refund_merchant_30h > 10,
    "akun baru (< 30 hari)": lambda d: d.umur_akun_hari < 30,
}


def load_config():
    p = KODE / "configs" / "t2_terpilih.json"
    return json.loads(p.read_text()) if p.exists() else {"imbalance": "tanpa penanganan", "kalibrasi": "isotonic", "adaptasi": "semua data latih",
                                                          "batas_label_matang": "2025-08-11", "label_belum_matang": "buang"}


def run():
    df = get_features(); cols = feature_cols(df); cfg = load_config()
    res, angka = {}, {}
    with Timer("T2 (g) model analisis F2"):
        mtr, mka, mva = masks(df, FOLDS[1])
        m = _fit_variant(df, mtr, cols, cfg["imbalance"], weight=time_weights(df, mtr, cfg.get("adaptasi")))
        pk, pv = _pred(m, df, mka, cols), _pred(m, df, mva, cols)
        cal = Calibrator(cfg["kalibrasi"]).fit(pk, df.loc[mka, "is_fraud"].astype(int)) if cfg["kalibrasi"] != "none" else Calibrator("none")
        joblib.dump({"model": m, "cal": cal, "cols": cols, "cfg": cfg}, MOD / "t2_model_validasi_F2.joblib")
    v = df[mva].copy(); v["p"] = cal.transform(pv); v["tandai"] = flag_cost_rule(v.p, v.nominal); y = v.is_fraud.astype(int)
    v["tipe"] = np.select([(y == 1) & (v.tandai == 1), (y == 1) & (v.tandai == 0), (y == 0) & (v.tandai == 1)], ["TP", "FN", "FP"], "TN")
    tot = {"pr_auc": pr_auc(y, v.p), "biaya": total_cost(y, v.nominal, v.tandai), "n_FN": int((v.tipe == "FN").sum()),
           "nominal_FN": float(v.loc[v.tipe == "FN", "nominal"].sum()), "n_FP": int((v.tipe == "FP").sum())}
    res["total_F2"] = tot
    angka.update(TduaFinalValPR=round(tot["pr_auc"], 3), TduaFinalValBiaya=round(tot["biaya"] / 1e6, 1), TduaNominalFN=round(tot["nominal_FN"] / 1e6, 1))

    for col in ["kanal", "kategori_merchant"]:
        rows = []
        for k, g in v.groupby(col, observed=True):
            yy = g.is_fraud.astype(int)
            rows.append({col: k, "n": len(g), "n_fraud": int(yy.sum()), "rate_fraud_%": yy.mean() * 100,
                         "pr_auc": pr_auc(yy, g.p) if 0 < yy.sum() < len(g) else np.nan,
                         "recall": float(((g.tipe == "TP").sum()) / max(yy.sum(), 1)), "FN": int((g.tipe == "FN").sum()),
                         "FP": int((g.tipe == "FP").sum()), "nominal_FN_juta": g.loc[g.tipe == "FN", "nominal"].sum() / 1e6,
                         "biaya_juta": total_cost(yy, g.nominal, g.tandai) / 1e6})
        t = pd.DataFrame(rows).sort_values("biaya_juta", ascending=False)
        res[f"breakdown_{col}"] = t.to_dict("records")
        save_table(t.head(12), f"t2_breakdown_{col}", f"Error per {col} (fold F2, aturan biaya).", float_fmt="%.2f")
        fig, ax = plt.subplots(figsize=(8, 3.2)); tt = t.head(12)
        ax.bar(tt[col].astype(str), tt["nominal_FN_juta"], label="fraud lolos (juta Rp)")
        ax.bar(tt[col].astype(str), tt["FP"] * 0.025, bottom=tt["nominal_FN_juta"], label="biaya review FP (juta Rp)")
        ax.tick_params(axis="x", rotation=45); ax.legend(); ax.set_title(f"Sumber biaya per {col}")
        save_fig(fig, f"t2_breakdown_{col}")
    kt = pd.DataFrame(res["breakdown_kanal"]).sort_values("biaya_juta", ascending=False).iloc[0]
    angka.update(TduaKanalBiayaTerbesar=str(kt["kanal"]), TduaKanalRecall=round(kt["recall"] * 100, 1))

    rows = []
    for nama, fn in POLA.items():
        mk = fn(v).fillna(False).values
        yy = v.is_fraud.astype(int).values
        rows.append({"pola": nama, "n_trx": int(mk.sum()), "rate_fraud_%": float(yy[mk].mean() * 100) if mk.any() else np.nan,
                     "porsi_dari_semua_fraud_%": float(yy[mk].sum() / max(yy.sum(), 1) * 100),
                     "recall_model": float((v.tipe.values[mk] == "TP").sum() / max(yy[mk].sum(), 1))})
    pt = pd.DataFrame(rows).sort_values("porsi_dari_semua_fraud_%", ascending=False)
    res["pola_fraud"] = pt.to_dict("records")
    save_table(pt, "t2_pola_fraud", "Pola fraud: tingkat fraud dalam pola, porsi dari seluruh fraud, dan recall model (F2).", float_fmt="%.2f")
    fn = v[v.tipe == "FN"]
    res["profil_FN"] = {"nominal_median": float(fn.nominal.median()) if len(fn) else None,
                        "p_median": float(fn.p.median()) if len(fn) else None,
                        "kanal": fn.kanal.value_counts(normalize=True).round(3).to_dict()}
    save_result("t2_analisis", res, angka)
    return res


if __name__ == "__main__":
    run()

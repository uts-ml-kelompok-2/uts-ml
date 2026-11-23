"""Tugas 2 final.
1. Kalibrator: model dilatih s.d. Jun, dikalibrasi pada Jul s.d. batas label matang (out-of-time).
2. Model final: dilatih ulang pada semua data dengan label matang (s.d. batas), kalibrator diterapkan.
3. Prediksi test → prediksi_klasifikasi.csv (id_transaksi, prob, tandai) dengan aturan p × nominal > 25.000."""
import joblib
import numpy as np
import pandas as pd
from .common_t2 import get_features
from .models import feature_cols, to_X, Calibrator, flag_cost_rule
from .analysis import load_config
from .experiments import _fit_variant, _pred, time_weights
from ..common.paths import MOD, ROOT
from ..common.io_utils import save_result, Timer


def _train_mask(df, cfg, end):
    m = df.set.eq("train") & (df.waktu < pd.Timestamp(end))
    if cfg.get("adaptasi") == "3 bulan terakhir saja":
        m &= df.waktu >= pd.Timestamp(end) - pd.Timedelta(days=150)
    return m


def train():
    df = get_features(); cols = feature_cols(df); cfg = load_config()
    batas = cfg.get("batas_label_matang") if cfg.get("label_belum_matang", "buang") == "buang" else "2025-09-01"
    batas = batas or "2025-09-01"
    with Timer("T2 final: kalibrator out-of-time"):
        m_a = _train_mask(df, cfg, "2025-07-01")
        m_k = df.set.eq("train") & (df.waktu >= "2025-07-01") & (df.waktu < pd.Timestamp(batas))
        ma = _fit_variant(df, m_a, cols, cfg["imbalance"], weight=time_weights(df, m_a, cfg.get("adaptasi")))
        pk = _pred(ma, df, m_k, cols)
        cal = Calibrator(cfg["kalibrasi"]).fit(pk, df.loc[m_k, "is_fraud"].astype(int)) if cfg["kalibrasi"] != "none" else Calibrator("none")
    with Timer("T2 final: latih ulang s.d. batas label matang"):
        m_tr = _train_mask(df, cfg, batas)
        m = _fit_variant(df, m_tr, cols, cfg["imbalance"], weight=time_weights(df, m_tr, cfg.get("adaptasi")))
    joblib.dump({"model": m, "cal": cal, "cols": cols, "cfg": cfg, "batas": batas}, MOD / "t2_model_final.joblib")
    return write_predictions(df, m, cal, cols, cfg, batas)


def write_predictions(df, m, cal, cols, cfg, batas):
    te = df.set.eq("test")
    p = np.clip(cal.transform(_pred(m, df, te, cols)), 0, 1)
    tandai = flag_cost_rule(p, df.loc[te, "nominal"].values)
    out = pd.DataFrame({"id_transaksi": df.loc[te, "id_transaksi"].values, "prob": np.round(p, 6), "tandai": tandai})
    out.to_csv(ROOT / "prediksi_klasifikasi.csv", index=False)
    info = {"n": len(out), "n_ditandai": int(tandai.sum()), "persen_ditandai": float(tandai.mean() * 100),
            "prob_rata2": float(p.mean()), "batas_latih": batas, "konfigurasi": cfg,
            "expected_fraud_test": float(p.sum()), "biaya_review_test": float(tandai.sum() * 25_000)}
    save_result("t2_final", info, {"TduaTestDitandai": int(tandai.sum()), "TduaTestPersenDitandai": round(info["persen_ditandai"], 2),
                                   "TduaTestProbRata": round(info["prob_rata2"] * 100, 3), "TduaBatasLatih": batas})
    print(f"prediksi_klasifikasi.csv ditulis: {len(out)} baris, ditandai {int(tandai.sum())}")
    return info


def reproduce():
    from ..common.metrics import pr_auc, total_cost
    from .common_t2 import FOLDS, masks
    df = get_features()
    art = joblib.load(MOD / "t2_model_validasi_F2.joblib")
    _, _, mva = masks(df, FOLDS[1])
    p = art["cal"].transform(_pred(art["model"], df, mva, art["cols"]))
    y, nom = df.loc[mva, "is_fraud"].astype(int).values, df.loc[mva, "nominal"].values
    r = {"pr_auc": pr_auc(y, p), "biaya_juta": total_cost(y, nom, flag_cost_rule(p, nom)) / 1e6}
    print("REPRODUKSI T2 (fold F2, model + kalibrasi):", {k: round(v, 4) for k, v in r.items()})
    fin = joblib.load(MOD / "t2_model_final.joblib")
    write_predictions(df, fin["model"], fin["cal"], fin["cols"], fin["cfg"], fin["batas"])
    return r


if __name__ == "__main__":
    train()

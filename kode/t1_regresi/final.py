"""Tugas 1 final: latih pada Jan–Jul, kalibrasi CQR pada Agu, latih ulang Jan–Agu, prediksi test → prediksi_regresi.csv."""
import joblib
import numpy as np
import pandas as pd
from .data import load_raw, add_features
from .common_t1 import prepare, load_config, make_xy_cfg
from .models import LGBQuantile
from ..common.splits import month_mask
from ..common.paths import MOD, ROOT
from ..common.io_utils import save_result, Timer


def train():
    tr, te = load_raw()
    d = prepare(tr)
    cfg = load_config()
    with Timer("T1 final: kalibrasi CQR (skor out-of-time dari jendela s.d. Agu)"):
        from .analysis import make_cqr
        cqr = make_cqr(d, cfg, "2025-08")
    with Timer("T1 final: latih ulang Jan–Agu"):
        tst = add_features(te)
        Xtr, Xte, enc = make_xy_cfg(d, tst, cfg)
        m = LGBQuantile(cfg["params"]).fit(Xtr, d["y"])
    joblib.dump({"model": m, "enc": enc, "cqr": cqr, "cfg": cfg, "kolom": list(Xtr.columns)}, MOD / "t1_model_final.joblib")
    return write_predictions(m, cqr, cfg, Xte, tst)


def write_predictions(m, cqr, cfg, Xte, tst):
    p50, p90 = m.predict(Xte)
    n_koreksi_awal = int((m.m90_.predict(Xte) < p50).sum())  # P90 mentah < P50 → dikoreksi
    if cfg["kalibrasi"] != "tanpa kalibrasi":
        p90 = cqr.adjust(p90, tst["layanan"].values)
    p50 = np.clip(p50, 0, None); p90 = np.maximum(p90, p50)
    out = pd.DataFrame({"id_paket": tst["id_paket"].values, "p50": np.round(p50, 3), "p90": np.round(p90, 3)})
    out.to_csv(ROOT / "prediksi_regresi.csv", index=False)
    info = {"n": len(out), "p50_median": float(np.median(p50)), "p90_median": float(np.median(p90)),
            "koreksi_cqr": getattr(cqr, "global_", None), "n_p90_dikoreksi_ke_p50": n_koreksi_awal,
            "persen_p90_dikoreksi": n_koreksi_awal / len(out) * 100, "mitra_baru_test": int((tst["mitra_kurir"] == "K13").sum())}
    save_result("t1_final", info, {"TsatuTestPlimaMedian": round(info["p50_median"], 1), "TsatuTestPnilanMedian": round(info["p90_median"], 1),
                                   "TsatuKoreksiCQR": round(float(info["koreksi_cqr"] or 0), 2),
                                   "TsatuPersenKoreksiMonoton": round(info["persen_p90_dikoreksi"], 2)})
    print(f"prediksi_regresi.csv ditulis: {len(out)} baris")
    return info


def reproduce():
    """Untuk reproduksi live: muat model tersimpan, hitung ulang angka validasi F2 dan tulis ulang CSV (tanpa melatih ulang)."""
    from .models import evaluate
    from .analysis import predict_with
    tr, te = load_raw()
    d = prepare(tr)
    art = joblib.load(MOD / "t1_model_validasi_F2.joblib")
    m_tr = month_mask(d.tgl_pickup, "2025-01", "2025-06"); m_va = month_mask(d.tgl_pickup, "2025-07", "2025-08")
    from .data import feature_columns
    cols = feature_columns(d, drop_extra=["y", "y_eval"] + art["cfg"].get("drop_extra", []))
    Xva = art["enc"].transform(d.loc[m_va, cols])
    p50, p90 = predict_with(art["model"], art["cqr"], Xva, d.loc[m_va, "layanan"].values, art["cfg"])
    r = evaluate(d.loc[m_va, "y_eval"], p50, p90)
    print("REPRODUKSI T1 (fold F2, model final + kalibrasi):", {k: round(v, 4) for k, v in r.items()})
    fin = joblib.load(MOD / "t1_model_final.joblib")
    tst = add_features(te)
    cols_te = feature_columns(tst, drop_extra=fin["cfg"].get("drop_extra", []))
    Xte = fin["enc"].transform(tst[cols_te])[fin["kolom"]]
    write_predictions(fin["model"], fin["cqr"], fin["cfg"], Xte, tst)
    return r


if __name__ == "__main__":
    train()

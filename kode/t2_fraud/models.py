"""Tugas 2: penyiapan matriks fitur, model, kalibrasi, dan aturan keputusan berbasis biaya."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .data import CAT_COLS, LEAK_COLS, TIME_PROXY_COLS, REVIEW_COST
from .features import BEHAV_COLS, DRIFT_COLS
from ..common.metrics import pr_auc, roc_auc, total_cost
from ..common.paths import SEED

NON_FEATURES = {"id_transaksi", "id_pengguna", "id_merchant", "waktu", "bulan", "is_fraud", "set", "log_nom", "u_malam"}
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=-1, random_state=SEED)


def feature_cols(df, behav=True, keep=()):
    drop = (NON_FEATURES | set(LEAK_COLS) | set(TIME_PROXY_COLS) | set(DRIFT_COLS)) - set(keep)
    cols = [c for c in df.columns if c not in drop]
    if not behav:
        cols = [c for c in cols if c not in BEHAV_COLS + DRIFT_COLS]
    return cols


def to_X(df, cols):
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def fit_lgb(X, y, weight=None, params=None):
    return lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {})}).fit(X, y, sample_weight=weight)


def fit_logreg(X, y):
    num = X.select_dtypes("number")
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, C=0.5))
    m.fit(num, y); m.cols_ = list(num.columns)
    return m


def predict(m, X):
    if hasattr(m, "cols_"):
        return m.predict_proba(X[m.cols_])[:, 1]
    return m.predict_proba(X)[:, 1]


class Calibrator:
    """Kalibrasi probabilitas di set kalibrasi waktu: 'isotonic' atau 'sigmoid' (Platt pada logit skor)."""

    def __init__(self, method="isotonic"):
        self.method = method

    def fit(self, p, y):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        elif self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        if self.method == "none":
            return p
        if self.method == "isotonic":
            return self.m_.predict(p)
        return self.m_.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def flag_cost_rule(p, nominal):
    """Tandai jika expected loss jika lolos (p × nominal) > biaya review (25.000)."""
    return (np.asarray(p) * np.asarray(nominal) > REVIEW_COST).astype(int)


def best_threshold(p, y, nominal):
    """Threshold probabilitas tunggal yang meminimalkan biaya di set kalibrasi (pembanding aturan biaya)."""
    qs = np.unique(np.quantile(p, np.linspace(0.90, 0.9995, 200)))
    costs = [total_cost(y, nominal, (p >= t).astype(int)) for t in qs]
    return float(qs[int(np.argmin(costs))])


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    e = 0.0
    for b in range(len(q) - 1):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def evaluate(y, p, nominal, flag=None) -> dict:
    y = np.asarray(y).astype(int)
    r = {"pr_auc": pr_auc(y, p), "roc_auc": roc_auc(y, p), "n": int(len(y)), "n_fraud": int(y.sum())}
    if flag is not None:
        r.update(biaya=total_cost(y, nominal, flag), n_ditandai=int(np.sum(flag)),
                 recall=float(np.sum(flag * y) / max(y.sum(), 1)), presisi=float(np.sum(flag * y) / max(np.sum(flag), 1)))
    return r

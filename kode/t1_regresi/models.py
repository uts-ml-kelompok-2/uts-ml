"""Tugas 1: encoding kategori + model (baseline, LightGBM quantile, linear log-target)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import CAT_COLS
from ..common.paths import SEED

OTHER = "__OTHER__"
HIGH_CARD = ["kota_tujuan", "kota_asal"]
LGB_BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=SEED)


# ----------------------------------------------------------------------------- encoding
class CategoryEncoder:
    """Encoding kategori yang di-fit HANYA di data latih.

    high_card: 'native' (kategori LightGBM), 'te' (target encoding out-of-fold + smoothing), 'freq' (frekuensi).
    Kategori jarang (< min_count) dan kategori yang tidak dikenal saat prediksi dipetakan ke __OTHER__,
    sehingga kategori baru di test (mis. mitra_kurir K13) punya perlakuan yang sudah dipelajari.
    """

    def __init__(self, high_card="te", min_count=30, smoothing=50, n_folds=5, unknown_frac=0.0,
                 unknown_cols=("mitra_kurir",)):
        self.high_card, self.min_count, self.smoothing, self.n_folds = high_card, min_count, smoothing, n_folds
        # unknown_frac > 0: sebagian kecil baris latih diberi kategori __OTHER__ ("kategori dropout"),
        # supaya model belajar perilaku untuk kategori yang belum pernah dilihat (mis. mitra K13 di test).
        self.unknown_frac, self.unknown_cols = unknown_frac, tuple(unknown_cols)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.levels_ = {}
        self.cats_ = [c for c in CAT_COLS if c in X.columns]
        self.hc_ = [c for c in HIGH_CARD if c in X.columns]
        for c in self.cats_:
            vc = X[c].astype(str).value_counts()
            self.levels_[c] = sorted(vc[vc >= self.min_count].index.tolist()) + [OTHER]
        self.global_ = float(np.nanmean(y))
        if self.high_card in ("te", "freq"):
            self.maps_ = {}
            for c in self.hc_:
                s = self._map_rare(X[c], c)
                if self.high_card == "freq":
                    self.maps_[c] = s.value_counts(normalize=True).to_dict()
                else:
                    g = pd.DataFrame({"k": s.values, "y": np.asarray(y)}).groupby("k")["y"].agg(["sum", "count"])
                    self.maps_[c] = ((g["sum"] + self.smoothing * self.global_) / (g["count"] + self.smoothing)).to_dict()
        return self

    def _map_rare(self, s, c):
        s = s.astype(str)
        return s.where(s.isin(self.levels_[c]), OTHER)

    def transform(self, X: pd.DataFrame, y=None, oof=False) -> pd.DataFrame:
        X = X.copy()
        for c in self.cats_:
            s = self._map_rare(X[c], c)
            if oof and self.unknown_frac > 0 and c in self.unknown_cols:
                rng = np.random.RandomState(SEED + 7)
                s = s.where(rng.rand(len(s)) >= self.unknown_frac, OTHER)
            X[c] = pd.Categorical(s, categories=self.levels_[c])
        if self.high_card in ("te", "freq"):
            for c in self.hc_:
                key = X[c].astype(str)
                if self.high_card == "te" and oof and y is not None:
                    X[c + "_te"] = self._oof_te(key, np.asarray(y))
                else:
                    X[c + ("_te" if self.high_card == "te" else "_freq")] = key.map(self.maps_[c]).astype(float).fillna(
                        self.global_ if self.high_card == "te" else 0.0)
                X = X.drop(columns=c)
        return X

    def _oof_te(self, key, y):
        """Target encoding out-of-fold untuk data latih, supaya model tidak melihat targetnya sendiri."""
        out = np.full(len(key), np.nan)
        folds = np.arange(len(key)) % self.n_folds
        rng = np.random.RandomState(SEED); rng.shuffle(folds)
        for f in range(self.n_folds):
            tr = folds != f
            g = pd.DataFrame({"k": key.values[tr], "y": y[tr]}).groupby("k")["y"].agg(["sum", "count"])
            m = ((g["sum"] + self.smoothing * self.global_) / (g["count"] + self.smoothing))
            out_f = pd.Series(key.values[~tr]).map(m).astype(float).fillna(self.global_).values
            out[~tr] = out_f
        return out


# ----------------------------------------------------------------------------- models
class GroupQuantileBaseline:
    """Baseline: kuantil empiris durasi per (layanan × pulau_tujuan), fallback ke kuantil global."""

    def __init__(self, keys=("layanan", "pulau_tujuan")):
        self.keys = list(keys)

    def fit(self, X, y):
        d = X[self.keys].astype(str).copy(); d["y"] = np.asarray(y)
        self.tab_ = d.groupby(self.keys)["y"].quantile([0.5, 0.9]).unstack()
        self.glob_ = (float(np.nanquantile(y, 0.5)), float(np.nanquantile(y, 0.9)))
        return self

    def predict(self, X):
        k = X[self.keys].astype(str)
        idx = pd.MultiIndex.from_frame(k)
        t = self.tab_.reindex(idx)
        p50 = t[0.5].fillna(self.glob_[0]).values
        p90 = t[0.9].fillna(self.glob_[1]).values
        return p50, np.maximum(p90, p50)


class LGBQuantile:
    """Dua model LightGBM terpisah: objective quantile alpha=0.5 (P50) dan alpha=0.9 (P90)."""

    def __init__(self, params=None):
        self.params = {**LGB_BASE, **(params or {})}

    def fit(self, X, y):
        m = ~np.isnan(np.asarray(y, float))
        self.m50_ = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **self.params).fit(X[m], np.asarray(y)[m])
        self.m90_ = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **self.params).fit(X[m], np.asarray(y)[m])
        return self

    def predict(self, X):
        p50, p90 = self.m50_.predict(X), self.m90_.predict(X)
        return p50, np.maximum(p90, p50)


class LinearLogModel:
    """Model linear (Ridge) dengan target log(durasi). P90 = exp(prediksi + kuantil-0.9 residual log)."""

    SMALL_CATS = ["layanan", "mitra_kurir", "hub_asal", "pulau_tujuan"]

    def fit(self, X, y):
        m = ~np.isnan(np.asarray(y, float))
        X, ly = X[m], np.log(np.asarray(y, float)[m])
        num = [c for c in X.columns if c not in CAT_COLS and X[c].dtype.kind in "fiub" and not c.endswith("_te")]
        te = [c for c in X.columns if c.endswith("_te") or c.endswith("_freq")]
        self.num_ = num + te
        cats = [c for c in self.SMALL_CATS if c in X.columns]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), self.num_),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), cats)])
        self.pipe_ = make_pipeline(pre, Ridge(alpha=3.0))
        Xs = self._prep(X)
        self.pipe_.fit(Xs, ly)
        res = ly - self.pipe_.predict(Xs)
        self.q90_res_ = float(np.quantile(res, 0.9)); self.q50_res_ = float(np.quantile(res, 0.5))
        return self

    def _prep(self, X):
        X = X.copy()
        for c in self.num_:
            if c in X and X[c].min() >= 0 and X[c].max() > 50:
                X[c] = np.log1p(X[c])
        for c in self.SMALL_CATS:
            if c in X:
                X[c] = X[c].astype(str)
        return X

    def predict(self, X):
        z = self.pipe_.predict(self._prep(X))
        p50, p90 = np.exp(z + self.q50_res_), np.exp(z + self.q90_res_)
        return p50, np.maximum(p90, p50)


def evaluate(y, p50, p90) -> dict:
    from ..common.metrics import pinball_loss, coverage
    y = np.asarray(y, float); ok = ~np.isnan(y)
    y, p50, p90 = y[ok], np.asarray(p50)[ok], np.asarray(p90)[ok]
    return {"pinball50": pinball_loss(y, p50, 0.5), "pinball90": pinball_loss(y, p90, 0.9),
            "coverage90": coverage(y, p90), "mae": float(np.mean(np.abs(y - p50))), "n": int(len(y))}

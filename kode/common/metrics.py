"""Metrik sesuai soal UTS ML."""
import numpy as np
from sklearn.metrics import average_precision_score, adjusted_rand_score, roc_auc_score

REVIEW_COST = 25_000


def pinball_loss(y, q, alpha: float) -> float:
    """Pinball (quantile) loss rata-rata. alpha=0.5 → setengah MAE; alpha=0.9 → menghukum prediksi terlalu rendah 9x."""
    y, q = np.asarray(y, float), np.asarray(q, float)
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y, upper) -> float:
    """Proporsi nilai aktual yang <= batas atas P90. Target ideal 0.90."""
    return float(np.mean(np.asarray(y) <= np.asarray(upper)))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def total_cost(y, nominal, tandai) -> float:
    """cost = Σ nominal fraud yang lolos (tidak ditandai) + 25.000 × jumlah transaksi ditandai."""
    y, nominal, tandai = np.asarray(y), np.asarray(nominal, float), np.asarray(tandai)
    return float(np.sum(nominal * ((y == 1) & (tandai == 0))) + REVIEW_COST * np.sum(tandai == 1))


def cost_flag_all(y, nominal) -> float:
    return total_cost(y, nominal, np.ones(len(y), int))


def cost_flag_none(y, nominal) -> float:
    return total_cost(y, nominal, np.zeros(len(y), int))


def ari(a, b) -> float:
    return float(adjusted_rand_score(a, b))

"""Pembagian data berbasis waktu (bulan). Meniru pemakaian nyata: latih di masa lalu, prediksi bulan berikutnya."""
import pandas as pd


def month_mask(dates: pd.Series, start: str, end: str):
    """True untuk baris dengan tanggal di [start, end] (format 'YYYY-MM' atau 'YYYY-MM-DD', inklusif)."""
    d = pd.to_datetime(dates)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + (pd.offsets.MonthEnd(0) if len(end) == 7 else pd.Timedelta(0)) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return (d >= s) & (d <= e)

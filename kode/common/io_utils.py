"""Simpan hasil (JSON), gambar, dan tabel secara seragam. Setiap angka di laporan berasal dari file ini."""
import json, platform, time, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .paths import RES, FIG, TAB, CACHE, SEED, QUICK


def _versions():
    import sklearn, lightgbm
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime.date)):
        return str(o)
    return o


def save_result(name: str, data: dict, angka: dict | None = None) -> Path:
    """Simpan results/<name>.json. `angka` = angka kunci untuk macro LaTeX (nama → nilai)."""
    payload = {"name": name, "quick": QUICK, "seed": SEED, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "versions": _versions(), "data": _clean(data), "angka": _clean(angka or {})}
    p = RES / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def load_result(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text())


def save_fig(fig, name: str) -> Path:
    p = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def save_table(df: pd.DataFrame, name: str, caption: str = "", float_fmt: str = "%.3f", index: bool = False) -> Path:
    """Simpan tabel sebagai .csv dan .tex (booktabs) untuk di-\\input di laporan."""
    df.to_csv(TAB / f"{name}.csv", index=index)
    # kolom teks panjang → kolom paragraf (p{...}) supaya baris membungkus, bukan melebar keluar halaman
    is_txt = {c: not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns}
    long = {c: df[c].astype(str).str.len().max() for c in df.columns if is_txt[c]}
    long = {c: n for c, n in long.items() if n > 35} if df.shape[1] + int(index) <= 4 else {}
    col_format = None
    if long:
        tot = sum(long.values())
        col_format = ("l" if index else "") + "".join(
            (f"p{{{max(0.18, 0.75 * long[c] / tot):.2f}\\linewidth}}" if c in long else ("l" if is_txt[c] else "r")) for c in df.columns)
    tex = df.to_latex(index=index, float_format=lambda x: float_fmt % x, escape=True, column_format=col_format)
    if df.shape[1] + int(index) > 4 and not long:      # tabel lebar → dikecilkan agar muat selebar halaman
        # hanya MENGECILKAN tabel yang lebih lebar dari halaman; tabel yang sudah muat tidak diperbesar
        tex = "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n" + tex.strip() + "\n}\n"
    body = "\\begin{table}[ht]\n\\centering\\small\n" + tex + (f"\\caption{{{caption}}}\n" if caption else "") + \
           f"\\label{{tab:{name}}}\n\\end{{table}}\n"
    p = TAB / f"{name}.tex"
    p.write_text(body)
    return p


def cache_parquet(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(CACHE / f"{name}.parquet")


def read_cache(name: str):
    p = CACHE / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t = time.time(); print(f"[mulai] {self.label}", flush=True); return self
    def __exit__(self, *a):
        self.dt = time.time() - self.t; print(f"[selesai] {self.label} ({self.dt:.1f} dtk)", flush=True)

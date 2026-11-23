"""Lokasi folder proyek. Semua path di kode lewat modul ini (tidak ada path hardcode)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("UTS_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("UTS_DATA_DIR", ROOT / "data"))
KODE = ROOT / "kode"
OUT = KODE / "outputs"
RES, FIG, TAB, MOD, CACHE, LOG = (OUT / d for d in ["results", "figures", "tables", "models", "cache", "logs"])
for _d in (RES, FIG, TAB, MOD, CACHE, LOG):
    _d.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("UTS_QUICK", "0") == "1"   # mode uji cepat (subset data, konfigurasi lebih sedikit)
SEED = int(os.environ.get("UTS_SEED", "42"))


def data_file(name: str) -> Path:
    """Cari file data. Menerima 'regresi_train.csv.gz' maupun varian 'regresi_train_csv.gz'."""
    cands = [DATA / name, DATA / name.replace(".csv.gz", "_csv.gz"), DATA / name.replace(".csv.gz", ".csv")]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"File data '{name}' tidak ditemukan di {DATA}. Isi folder: {sorted(p.name for p in DATA.glob('*'))}")

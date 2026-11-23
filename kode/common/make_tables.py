"""Kumpulkan semua angka kunci dari results/*.json → tables/angka.tex (\\newcommand), dipakai laporan.
Tidak ada angka yang diketik manual di laporan."""
import json
import re
from .paths import RES, TAB, KODE


def _escape(v):
    s = str(v)
    s = s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    for a, b in UNICODE.items():
        s = s.replace(a, b)
    return s


UNICODE = {"≥": "$\\geq$", "≤": "$\\leq$", "∪": "$\\cup$", "×": "$\\times$", "→": "$\\to$", "≈": "$\\approx$",
           "±": "$\\pm$", "−": "-", "—": "---", "·": "$\\cdot$", "α": "$\\alpha$", "…": "...", "’": "'", "“": "``", "”": "''"}


def _fix_caption(m):
    body = m.group(1)
    body = re.sub(r"(?<!\\)_", r"\\_", body)
    body = re.sub(r"(?<!\\)%", r"\\%", body)
    body = re.sub(r"(?<!\\)&", r"\\&", body)
    body = re.sub(r"(?<!\\)#", r"\\#", body)
    return "\\caption{" + body + "}"


def sanitize_tables():
    """Pastikan semua tables/*.tex aman dikompilasi pdflatex: karakter Unicode → perintah LaTeX, caption di-escape.
    Idempoten (aman dijalankan berulang)."""
    n = 0
    for p in TAB.glob("*.tex"):
        if p.name == "angka.tex":
            continue
        t = p.read_text()
        for a, b in UNICODE.items():
            t = t.replace(a, b)
        t = re.sub(r"\\caption\{(.*)\}\n", lambda m: _fix_caption(m) + "\n", t)
        t = t.replace("\\resizebox{\\linewidth}{!}{%", "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%")
        spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", t)
        if spec and len(re.sub(r"[^lcrp]", "", spec.group(1))) > 4 and "\\resizebox" not in t:
            t = t.replace("\\begin{tabular}", "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}\n}", 1)
        p.write_text(t); n += 1
    return n


def run():
    sanitize_tables()
    lines = ["% Dibuat otomatis oleh kode/common/make_tables.py — JANGAN diedit manual.", ""]
    seen = {}
    for p in sorted(RES.glob("*.json")):
        d = json.loads(p.read_text())
        for k, v in (d.get("angka") or {}).items():
            if not re.fullmatch(r"[A-Za-z]+", k):
                raise ValueError(f"Nama macro tidak valid (hanya huruf): {k} di {p.name}")
            if isinstance(v, float):
                v = f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.1f}"
            seen[k] = (v, p.name)
    # konfigurasi terpilih (ditulis otomatis oleh experiments/select_k) → macro teks untuk laporan
    cfgmap = {"t1_terpilih.json": {"kalibrasi": "TsatuKonfigKalibrasi", "high_card": "TsatuKonfigEncoding"},
              "t2_terpilih.json": {"imbalance": "TduaKonfigImbalance", "kalibrasi": "TduaKonfigKalibrasi", "adaptasi": "TduaKonfigAdaptasi",
                                   "label_belum_matang": "TduaKonfigLabel", "batas_label_matang": "TduaKonfigBatas"},
              "t3_terpilih.json": {"k": "TtigaKonfigK", "method": "TtigaKonfigMetode"}}
    nama_enc = {"te": "target encoding out-of-fold", "freq": "frequency encoding", "native": "kategori native LightGBM"}
    for fn, mp in cfgmap.items():
        cp = KODE / "configs" / fn
        if cp.exists():
            c = json.loads(cp.read_text())
            for key, macro in mp.items():
                if key in c:
                    v = c[key]
                    if key == "high_card":
                        v = nama_enc.get(v, v)
                    seen[macro] = (v, fn)
            if fn == "t1_terpilih.json":
                seen["TsatuKonfigMitra"] = (r"tanpa fitur mitra\_kurir" if c.get("drop_extra") else
                                            (r"kategori \_\_OTHER\_\_ + dropout 5\%" if c.get("unknown_frac", 0) > 0 else r"kategori \_\_OTHER\_\_"), fn)
    for k, (v, src) in sorted(seen.items()):
        val = v if (isinstance(v, str) and "\\" in v) else _escape(v)
        lines.append(f"\\newcommand{{\\{k}}}{{{val}}}  % {src}")
    (TAB / "angka.tex").write_text("\n".join(lines) + "\n")
    print(f"angka.tex: {len(seen)} macro")
    return len(seen)


if __name__ == "__main__":
    run()

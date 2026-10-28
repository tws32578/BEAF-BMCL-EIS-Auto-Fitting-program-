"""직렬 저항 세 가지(추천 회로 R0 / 교차점 R_HF / L 보정 R0)를 샘플별 온도 경향으로 비교
  python tests/r0_reference.py <Gr||Gr 데이터 폴더>"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402


def arr_r2(T, v):
    x, y = 1000 / (np.array(T) + 273.15), np.log(1 / np.array(v, float))
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    c = np.polyfit(x[ok], y[ok], 1)
    return 1 - np.sum((y[ok] - np.polyval(c, x[ok])) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    by = defaultdict(list)
    for p in sorted(root.rglob("*.mpt")):
        e = load_file(p.read_bytes(), p.name)[0]
        r = analyze(e["cycles"][e["default_cycle"]], {"cell": "sym", "electrolyte": "liquid"}, {})
        ref = r["R0_ref"]
        if not ref:
            continue
        sample = p.relative_to(root).parts[0]
        by[sample].append((e["meta"]["T_C"], ref["R0_rec"], ref["R_HF"], ref["R0_L"] if ref["R0_L"] is not None else np.nan))
        print(f"{sample:7s} {e['meta']['T_C']:>4} R0 추천 {ref['R0_rec']:.3f} | R_HF {ref['R_HF']:.3f} | R0(L) "
              f"{ref['R0_L'] if ref['R0_L'] is not None else float('nan'):.3f} {ref['R0_L_note'] or ''}")
    print("\n샘플     Arrhenius R²: 추천 R0 / R_HF / R0(L)   | R_HF−R0(L) 비율")
    for s, rows in by.items():
        rows = sorted(set(rows))
        T, a, b, c = map(np.array, zip(*rows))
        print(f"{s:8s} {arr_r2(T, a):.2f} / {arr_r2(T, b):.2f} / {arr_r2(T, c):.2f}   | "
              f"{np.nanmedian(b / c - 1) * 100:+.0f}% (L 보정 결정 {int(np.isfinite(c).sum())}/{len(c)})")

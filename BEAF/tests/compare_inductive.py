"""인덕턴스 영역(−Z'' < 0) 점을 피팅에 넣을 때와 뺄 때 비교:  python tests/compare_inductive.py <파일 또는 폴더> [셀] [전해질]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402


def summary(r):
    c = r["candidates"][0]
    v = dict(zip(c["names"], c["values"]))
    R_sum = v.get("R0", 0) + sum(a["R"] for a in c["arcs"]) + sum(x["R"] for x in c.get("specials") or [])
    return {"circ": c["circuit"], "R0": v.get("R0"), "R_sum": R_sum, "L": v.get("L0"),
            "rms": c["rms_pct"], "status": r["status"], "arcsR": [a["R"] for a in c["arcs"]]}


if __name__ == "__main__":
    src = Path(sys.argv[1])
    meta = {"cell": sys.argv[2] if len(sys.argv) > 2 else "unknown",
            "electrolyte": sys.argv[3] if len(sys.argv) > 3 else "unknown"}
    files = [src] if src.is_file() else sorted(src.rglob("*.mpt"))
    rows = []
    for p in files:
        for e in load_file(p.read_bytes(), p.name):
            sp = e["cycles"][e["default_cycle"]]
            a = analyze(sp, meta, {})
            n_ind = a["range"]["start"]
            if n_ind == 0:
                continue
            b = analyze(sp, meta, {"with_L": True})
            sa, sb = summary(a), summary(b)
            rows.append((e["name"], n_ind, sa, sb))
            d = lambda k: (sb[k] / sa[k] - 1) * 100 if sa[k] else np.nan  # noqa: E731
            print(f"{e['name'][:38]:38s} 유도점 {n_ind:2d} | R0 {sa['R0']:.4g}→{sb['R0']:.4g} ({d('R0'):+.1f}%) "
                  f"R_Σ {d('R_sum'):+.1f}% | 잔차 {sa['rms']:.2f}→{sb['rms']:.2f}% | L {sa['L'] or 0:.2e}→{sb['L']:.2e} "
                  f"| {sa['status']}→{sb['status']} {'' if sa['circ'] == sb['circ'] else '회로 ' + sa['circ'] + ' → ' + sb['circ']}")
    if rows:
        dR0 = [abs(sb["R0"] / sa["R0"] - 1) * 100 for _, _, sa, sb in rows if sa["R0"]]
        dRs = [abs(sb["R_sum"] / sa["R_sum"] - 1) * 100 for _, _, sa, sb in rows]
        same = sum(sa["circ"].replace("L0-", "") == sb["circ"].replace("L0-", "") for _, _, sa, sb in rows)
        print(f"\n유도 영역 있는 측정 {len(rows)}개 | R0 변화 중앙값 {np.median(dR0):.1f}% (최대 {max(dR0):.1f}%) "
              f"| R_Σ 변화 중앙값 {np.median(dRs):.2f}% | L 빼고 같은 회로 {same}/{len(rows)}")

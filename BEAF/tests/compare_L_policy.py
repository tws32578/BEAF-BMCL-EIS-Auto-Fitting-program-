"""인덕턴스 영역이 있는 데이터에서 'L 없는 후보도 허용(예전)' vs 'L 필수(새 기본)' 비교
  python tests/compare_L_policy.py <파일 또는 폴더> [셀] [전해질]"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402


def r0_sum(r):
    c = r["candidates"][0]
    v = dict(zip(c["names"], c["values"]))
    return v.get("R0", np.nan), v.get("R0", 0) + sum(a["R"] for a in c["arcs"]) + sum(x["R"] for x in c.get("specials") or [])


if __name__ == "__main__":
    src = Path(sys.argv[1])
    meta = {"cell": sys.argv[2] if len(sys.argv) > 2 else "unknown",
            "electrolyte": sys.argv[3] if len(sys.argv) > 3 else "unknown"}
    files = [src] if src.is_file() else sorted(src.rglob("*.mpt"))
    st_old, st_new, d0, ds, n = Counter(), Counter(), [], [], 0
    for p in files:
        for e in load_file(p.read_bytes(), p.name):
            sp = e["cycles"][e["default_cycle"]]
            new = analyze(sp, meta, {"L_if_inductive": True})
            if not new["range"]["has_inductive"]:
                continue
            old = analyze(sp, meta, {"L_if_inductive": False})
            n += 1
            (a0, as_), (b0, bs) = r0_sum(old), r0_sum(new)
            st_old[old["status"]] += 1
            st_new[new["status"]] += 1
            d0.append(b0 / a0 - 1)
            ds.append(bs / as_ - 1)
            ix = new["range"]["start"]
            print(f"{e['name'][:34]:34s} 교차점 Re {sp.Z.real[ix]:.3f} | R0 {a0:.3f} → {b0:.3f} ({(b0 / a0 - 1) * 100:+.0f}%) "
                  f"| R_Σ {(bs / as_ - 1) * 100:+.1f}% | {old['status']} → {new['status']} | {old['recommended']} → {new['recommended']}")
    d0, ds = np.array(d0) * 100, np.array(ds) * 100
    print(f"\n인덕턴스 영역 있는 측정 {n}개 | 판정 {dict(st_old)} → {dict(st_new)}")
    print(f"R0 변화 중앙값 {np.median(d0):+.1f}% (범위 {d0.min():+.0f} ~ {d0.max():+.0f}%) | R0<0.05×예전 {int(np.sum(d0 < -95))}개 "
          f"| R_Σ 변화 중앙값 {np.median(np.abs(ds)):.2f}%")

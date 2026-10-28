"""
BEAF 자동 피팅 결과 vs 수동 EC-Lab Z-Fit 결과 (재료연구/fitting.xlsx) 비교

  python tests/validate_manual.py <데이터 폴더> <fitting.xlsx>

목표 (개발 방향 문서): 툴 결과 편차가 피팅자 간 편차 이내
  R_Σ ~1.3 %,  R_bulk ~3.9 %,  R_ct ~5.7 %,  R_SEI ~20 %  (중앙값)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import read_any  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

EXCLUDE = ["2A6PF_GrllGr_activation energy_30oC_C03"]       # 비정상 측정 (재측정 파일 사용)
# 세 번째 인자 --L-required: 인덕턴스 영역이 있으면 L 있는 후보만 (없으면 L 없는 후보도 허용 = 수동 피팅과 같은 조건)
EXTRA = {"L_if_inductive": "--L-required" in sys.argv}


def load_manual(xlsx):
    raw = pd.read_excel(xlsx, header=None)
    head = raw.index[raw[1].astype(str).str.strip() == "Rbulk"][-1]
    rows = []
    for c0 in range(0, raw.shape[1], 6):
        T = raw.iat[head, c0]
        if pd.isna(T):
            continue
        for r in range(head + 1, raw.shape[0]):
            s = raw.iat[r, c0]
            if pd.isna(s):
                break
            vals = [raw.iat[r, c0 + k] for k in range(1, 5)]
            rows.append({"sample": str(s).strip(), "T": float(T), "R_bulk": vals[0],
                         "R_SEI": vals[1], "R_ct": vals[2], "R_sum": vals[3]})
    return pd.DataFrame(rows)


def pick_files(root):
    files = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in (".mpt", ".mpr") or any(e in p.stem for e in EXCLUDE):
            continue
        key = p.with_suffix("")
        if key in files and files[key].suffix == ".mpt":
            continue                                        # .mpt 우선
        files[key] = p
    return list(files.values())


def main(root, xlsx):
    manual = load_manual(xlsx)
    rows = []
    t0 = time.time()
    for p in pick_files(root):
        sp = read_any(p.read_bytes(), p.name)
        sample = p.relative_to(root).parts[0]
        r = analyze(sp, {"cell": "sym", "electrolyte": "liquid"}, {"mode": "arcs", **EXTRA})
        # 수동 피팅과 같은 2-반원 회로 중 BIC 최소 (L 유무는 자동 선택)
        two = [c for c in r["candidates"] if c.get("ok") and c.get("n_arcs") == 2 and not c.get("shape")]
        best = two[0] if two else None                     # 이미 툴의 순위대로 정렬돼 있음
        rec = r["candidates"][0]
        row = {"sample": sample, "T": sp.meta["T_C"], "file": p.name, "status": r["status"],
               "rec": rec["circuit"], "n_drt": r["n_drt"], "kk_noise": r["kk"].get("noise_pct")}
        if best:
            d = dict(zip(best["display_names"], best["values"]))
            row.update({"R_bulk": d["R_bulk"], "R_SEI": d.get("R_SEI"), "R_ct": d.get("R_ct"),
                        "rms%": best["rms_pct"], "L": best.get("L"), "rep": f"{best['reproduce'][0]}/{best['reproduce'][1]}"})
            row["R_sum"] = row["R_SEI"] + row["R_ct"] + row["R_bulk"]
        rows.append(row)
        print(f"{sample:7s} {row['T']:>4} {r['status']:3s} n_drt={r['n_drt']} rec={rec['circuit']:<38} "
              f"R_SEI={row.get('R_SEI', np.nan):.3f} R_ct={row.get('R_ct', np.nan):.3f} L={row.get('L')} rms={row.get('rms%', np.nan):.2f}", flush=True)
    auto = pd.DataFrame(rows)
    print(f"\n{len(auto)} files, {time.time() - t0:.0f} s")

    m = auto.merge(manual, on=["sample", "T"], suffixes=("_auto", "_man"))
    m = m[m["R_sum_man"].notna()]
    for k in ("R_bulk", "R_SEI", "R_ct", "R_sum"):
        dev = (m[f"{k}_auto"] - m[f"{k}_man"]) / m[f"{k}_man"] * 100
        m[f"d_{k}%"] = dev
        print(f"{k:7s} |편차| 중앙값 {np.nanmedian(np.abs(dev)):5.1f} %   "
              f"±5% 이내 {int((np.abs(dev) <= 5).sum())}/{dev.notna().sum()}")
    out = Path(__file__).resolve().parent / "validate_manual_result.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print("저장:", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

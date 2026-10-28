"""인덕턴스 영역을 넣었을 때 어떤 유도 소자가 맞는지 탐색:  python tests/probe_inductor.py <폴더>
기본 분석의 추천 회로(L 제외 구조)에 L0 / La0 / p(L0,R8) 를 붙여, 유도 점 포함 구간에 피팅."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf import analysis  # noqa: E402
from beaf.eis_io import load_file  # noqa: E402
from beaf.fitting import fit_circuit  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

if __name__ == "__main__":
    out = {k: [] for k in ("L0", "La0", "p(L0,R8)")}
    for p in sorted(Path(sys.argv[1]).rglob("*.mpt")):
        e = load_file(p.read_bytes(), p.name)[0]
        sp = e["cycles"][e["default_cycle"]]
        a = analyze(sp, {"cell": "sym", "electrolyte": "liquid"}, {"shapes": False})
        n_ind = a["range"]["start"]
        if not n_ind:
            continue
        rec = a["candidates"][0]
        core = rec["circuit"].removeprefix("L0-")
        m0 = np.array(a["mask"])
        m = m0.copy()
        m[:n_ind] = True
        f, Z = sp.freq[m], sp.Z[m]
        R0_ref = dict(zip(rec["names"], rec["values"]))["R0"]
        line = f"{e['name'][:30]:30s} 기본 R0 {R0_ref:.3f} 잔차 {rec['rms_pct']:.2f}% |"
        for ind in out:
            fit = fit_circuit(f"{ind}-{core}", f, Z, drt_peaks=a["drt"].get("peaks"), n_starts=6)
            if not fit.get("ok"):
                line += f" {ind}: 실패 |"
                continue
            v = dict(zip(fit["names"], fit["values"]))
            # 반원 구간(기존 마스크) 만의 잔차 — 유도 점 때문에 반원 쪽이 망가지는지
            Zf = np.array(fit["Zfit"][0]) + 1j * np.array(fit["Zfit"][1]) if isinstance(fit["Zfit"], list) else fit["Zfit"]
            arc_part = m0[m]
            r_arc = np.sqrt(np.mean(np.abs((Zf[arc_part] - Z[arc_part]) / np.abs(Z[arc_part])) ** 2)) * 100
            out[ind].append((fit["rms_pct"], r_arc, v["R0"] / R0_ref - 1))
            extra = f" α={v['La0_1']:.2f}" if ind == "La0" else (f" R_L={v['R8']:.3g}" if "R8" in v else "")
            line += f" {ind}: 전체 {fit['rms_pct']:.2f}% 반원구간 {r_arc:.2f}% R0 {v['R0']:.3f}{extra} |"
        print(line)
    for ind, xs in out.items():
        xs = np.array(xs)
        print(f"{ind:9s} 전체 잔차 중앙값 {np.median(xs[:, 0]):.2f}% | 반원 구간 {np.median(xs[:, 1]):.2f}% "
              f"| R0 변화 중앙값 {np.median(np.abs(xs[:, 2])) * 100:.1f}%")

"""인덕턴스 영역에서 추정한 L 로 L 범위를 묶었을 때 R0 가 안정되는지:  python tests/probe_L_fixed.py <폴더>
같은 회로 구조(기본 추천에서 L 제외)를 L 자유 / L ±30% / L ±3% 로 피팅 (피팅 구간은 기본 그대로)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.fitting import diagnose, fit_circuit  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402


def L_from_inductive(freq, Z, start, n_top=3):
    im_w = Z.imag[:start] / (2 * np.pi * freq[:start])
    return float(np.median(im_w[:n_top])) if start >= 2 and np.all(im_w[:n_top] > 0) else None


if __name__ == "__main__":
    res = {k: [] for k in ("자유", "±30%", "±3%")}
    for p in sorted(Path(sys.argv[1]).rglob("*.mpt")):
        e = load_file(p.read_bytes(), p.name)[0]
        sp = e["cycles"][e["default_cycle"]]
        a = analyze(sp, {"cell": "sym", "electrolyte": "liquid"}, {"shapes": False})
        st = a["range"]["start"]
        Le = L_from_inductive(sp.freq, sp.Z, st)
        if not Le:
            continue
        m = np.array(a["mask"])
        f, Z = sp.freq[m], sp.Z[m]
        core = a["candidates"][0]["circuit"].removeprefix("L0-")
        line = f"{e['name'][:30]:30s} L추정 {Le:.2e} | Re(0.5~1MHz) {np.median(sp.Z.real[(sp.freq > 4e5) & (sp.freq < 1.1e6)]):.3f} |"
        for k, b in (("자유", None), ("±30%", (0.7 * Le, 1.3 * Le)), ("±3%", (0.97 * Le, 1.03 * Le))):
            fit = fit_circuit("L0-" + core, f, Z, drt_peaks=a["drt"].get("peaks"), n_starts=8,
                              bounds={"L0": b} if b else None)
            v = dict(zip(fit["names"], fit["values"]))
            sev = [x["msg"] for x in diagnose(fit, f) if x["severe"]]
            res[k].append((v["R0"], fit["rms_pct"], not sev))
            line += f" {k}: R0 {v['R0']:.3f} L {v['L0']:.2e} 잔차 {fit['rms_pct']:.2f}% {'유효' if not sev else '탈락'} |"
        print(line)
    for k, xs in res.items():
        xs = np.array(xs, float)
        print(f"{k:5s} R0≈0(<0.05) {int(np.sum(xs[:, 0] < 0.05))}개 | 유효 {int(xs[:, 2].sum())}/{len(xs)} | 잔차 중앙값 {np.median(xs[:, 1]):.2f}%")

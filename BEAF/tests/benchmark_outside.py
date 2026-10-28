"""
BEAF 후보 범위 밖의 회로로 만든 합성 데이터에서 어떻게 행동하는지 확인
(잡아내서 '검토'로 표시하는지, 아니면 그럴듯하게 틀린 답을 내는지)

  python tests/benchmark_outside.py [노이즈 %]
회로는 pyimpspec 의 CDC 표기:  나란히 = 직렬, ( ) = 병렬, [ ] = 병렬 안의 직렬
"""
import sys
from pathlib import Path

import numpy as np
import pyimpspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import Spectrum  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

# (이름, CDC, 구간 모드, 참값 설명)
CASES = [
    ("Randles (확산이 병렬 안)", "R{R=1.5}(Q{Y=1e-4,n=0.85}[R{R=4}Wo{Y=0.3,B=2}])", "full", "R0=1.5, Rct=4"),
    ("SEI + Randles", "R{R=1.5}(R{R=1}Q{Y=2e-5,n=0.9})(Q{Y=5e-4,n=0.85}[R{R=4}Wo{Y=0.3,B=2}])", "full", "R0=1.5, R_SEI=1, Rct=4"),
    ("사다리형 2반원 (SEI 통과 후 ct)", "R{R=1.5}(Q{Y=2e-5,n=0.9}[R{R=1}(R{R=4}Q{Y=5e-4,n=0.85})])", "arcs", "R0=1.5, R1=1, R2=4 (합 5)"),
    ("Gerischer 단독", "R{R=2}G{Y=0.5,k=20,n=0.5}", "arcs", "R0=2, Gerischer"),
    ("SEI 반원 + Gerischer", "R{R=2}(R{R=1}Q{Y=2e-5,n=0.9})G{Y=0.3,k=5,n=0.5}", "arcs", "R0=2, R1=1, Gerischer"),
    ("다공성 전극 TLM (비블로킹)", "R{R=1}Tlmno{R_i=3,R_ct=4,Y=0.005,n=0.85,L=1}", "arcs", "R0=1, TLM (45° 직선 + 반원)"),
    ("다공성 전극 TLM (블로킹, de Levie)", "R{R=1}Ls{R_i=5,R_r=1e6,Y=0.01,n=0.9,d=0.2}", "full", "R0=1, 블로킹 다공성"),
    ("비대칭 반원 2개 (Havriliak-Negami)", "R{R=1}Ha{R=3,tau=1e-4,a=0.8,b=0.6}Ha{R=5,tau=1e-2,a=0.85,b=0.7}", "arcs", "R0=1, 3+5"),
    # 반원 5개: 꼭대기 주파수 1e5 · 3e3 · 100 · 3 · 0.1 Hz (모두 측정 범위 1 MHz ~ 10 mHz 안)
    ("반원 5개", "R{R=1}" + "".join(f"(R{{R={r}}}Q{{Y={(1 / (2 * np.pi * fp)) ** 0.9 / r:.4g},n=0.9}})" for r, fp in
                                     [(1, 1e5), (2, 3e3), (3, 100), (2, 3), (1, 0.1)]), "arcs", "R0=1, 1+2+3+2+1"),
]

if __name__ == "__main__":
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    rng = np.random.default_rng(3)
    f = np.logspace(6, -2, 65)
    for name, cdc, mode, truth in CASES:
        Z0 = pyimpspec.parse_cdc(cdc).get_impedances(f)
        Z = Z0 * (1 + noise / 100 * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)))
        r = analyze(Spectrum(name, f, Z), {}, {"mode": mode})
        c = r["candidates"][0]
        Rs = ", ".join(f"{n}={v:.3g}" for n, v, u in zip(c["display_names"], c["values"], c["units"]) if u == "Ω")
        why = "; ".join(x for x in r["reasons"])[:150]
        print(f"[{r['status']}] {name}\n    참값 {truth}\n    BEAF {c['circuit']} | {Rs} | 잔차 {c['rms_pct']:.2f}% "
              f"(KK 노이즈 {r['kk'].get('noise_pct', 0):.2f}%) | DRT 반원 {r['n_drt']}개\n    이유: {why}", flush=True)

"""
정답을 아는 합성 스펙트럼으로 BEAF 의 회로 선정·피팅 정확도를 확인합니다.
(45개 실측 데이터에 맞춘 규칙이 아니라 '모르는 데이터'에서도 맞는지 보기 위함)

  python tests/benchmark_synthetic.py [노이즈 %] [반복 수]

각 경우: 정답 회로로 스펙트럼 생성 → 비례 가우시안 노이즈 → BEAF analyze → 추천 회로의
반원 수 · 확산 요소 · L 유무가 정답과 같은지, 반원 R 오차가 얼마인지 기록
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.circuits import Circuit  # noqa: E402
from beaf.eis_io import Spectrum  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402


def rq(R, f_peak, a):
    """반원 꼭대기 주파수로 Q 계산: τ = 1/(2πf), Q = τ^α / R"""
    tau = 1 / (2 * np.pi * f_peak)
    return [R, tau ** a / R, a]


# (이름, 회로, 파라미터, 구간 모드, 정답: 반원 수, 확산 요소, L, 반원 R 목록)
CASES = [
    ("1 arc (포메이션 전)", "R0-p(R1,CPE1)", [2.0] + rq(10, 300, 0.85), "arcs", 1, None, False),
    ("2 arcs 잘 분리 (100배)", "R0-p(R1,CPE1)-p(R2,CPE2)", [1.5] + rq(1.0, 2e4, 0.9) + rq(5.0, 200, 0.8), "arcs", 2, None, False),
    ("2 arcs 겹침 (15배)", "R0-p(R1,CPE1)-p(R2,CPE2)", [1.5] + rq(2.0, 3e3, 0.85) + rq(4.0, 200, 0.8), "arcs", 2, None, False),
    ("2 arcs 작은 SEI (R 10%)", "R0-p(R1,CPE1)-p(R2,CPE2)", [1.5] + rq(0.4, 1.5e4, 0.9) + rq(4.0, 250, 0.75), "arcs", 2, None, False),
    ("3 arcs (하프셀)", "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
     [3.0] + rq(5.0, 5e4, 0.9) + rq(10.0, 1e3, 0.85) + rq(20.0, 10, 0.8), "arcs", 3, None, False),
    ("L + 2 arcs", "L0-R0-p(R1,CPE1)-p(R2,CPE2)", [3e-7, 1.5] + rq(1.0, 2e4, 0.9) + rq(5.0, 200, 0.8), "arcs", 2, None, True),
    ("2 arcs + Wo (전체 구간)", "R0-p(R1,CPE1)-p(R2,CPE2)-Wo1",
     [1.5] + rq(1.0, 2e4, 0.9) + rq(5.0, 200, 0.8) + [8.0, 20.0], "full", 2, "Wo", False),
    ("1 arc + W (전체 구간)", "R0-p(R1,CPE1)-W1", [5.0] + rq(20.0, 500, 0.85) + [3.0], "full", 1, "W", False),
    ("2 arcs + 블로킹 Q (전체 구간)", "R0-p(R1,CPE1)-p(R2,CPE2)-CPE3",
     [1.5] + rq(1.0, 2e4, 0.9) + rq(5.0, 200, 0.8) + [0.05, 0.8], "full", 2, "CPE", False),
    ("2 arcs + Wo, 반원 구간만", "R0-p(R1,CPE1)-p(R2,CPE2)-Wo1",
     [1.5] + rq(1.0, 2e4, 0.9) + rq(5.0, 200, 0.8) + [8.0, 20.0], "arcs", 2, None, False),
    ("고체전해질 3 arcs (입계)", "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
     [20.0] + rq(30.0, 2e5, 0.95) + rq(15.0, 2e3, 0.85) + rq(40.0, 5, 0.8), "arcs", 3, None, False),
    ("큰 저항 (kΩ 스케일)", "R0-p(R1,CPE1)-p(R2,CPE2)", [50.0] + rq(300, 1e4, 0.9) + rq(2000, 20, 0.8), "arcs", 2, None, False),
]


def true_arc_R(circ_str, p):
    c = Circuit(circ_str)
    arcs = c.rq_arcs()
    taus = [(p[a["iR"]] * p[a["iQ"]]) ** (1 / p[a["iA"]]) for a in arcs]
    return [p[a["iR"]] for _, a in sorted(zip(taus, arcs), key=lambda t: t[0])]


def run(noise_pct=0.3, reps=2, seed=1):
    rng = np.random.default_rng(seed)
    f = np.logspace(np.log10(1e6), np.log10(0.01), 8 * 8 + 1)
    ok_struct, rows = 0, []
    t0 = time.time()
    for name, cs, p, mode, n_true, tail_true, L_true in CASES:
        Z0 = Circuit(cs).Z(p, f)
        Rt = true_arc_R(cs, p)
        for k in range(reps):
            Z = Z0 * (1 + noise_pct / 100 * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)))
            sp = Spectrum(name=name, freq=f, Z=Z)
            r = analyze(sp, {"cell": "half"}, {"mode": mode})
            rec = r["candidates"][0]
            n_ok = rec.get("n_arcs") == n_true
            t_ok = rec.get("tail") == tail_true
            l_ok = bool(rec.get("L")) == L_true
            good = n_ok and t_ok and l_ok
            ok_struct += good
            err = ""
            if n_ok and rec.get("ok"):
                Rf = [a["R"] for a in rec["arcs"]]
                err = " ".join(f"{(a - b) / b * 100:+.1f}%" for a, b in zip(Rf, Rt))
            rows.append((name, good))
            print(f"{'O' if good else 'X'} {name:28s} rec={rec['circuit']:<44} n_drt={r['n_drt']} "
                  f"status={r['status']}  R오차 {err}", flush=True)
    n = len(CASES) * reps
    print(f"\n구조 정답률 {ok_struct}/{n}  (노이즈 {noise_pct}%, {time.time() - t0:.0f} s)")
    return ok_struct, n


if __name__ == "__main__":
    nz = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run(nz, reps)

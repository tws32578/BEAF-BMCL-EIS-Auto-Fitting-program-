"""분석 단계별 소요 시간 확인:  python tests/profile_one.py"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf import pipeline  # noqa: E402
from beaf.circuits import Circuit  # noqa: E402
from beaf.eis_io import Spectrum  # noqa: E402
from tests.benchmark_synthetic import CASES  # noqa: E402

if __name__ == "__main__":
    f = np.logspace(6, -2, 65)
    name, cs, p, mode, *_ = CASES[6]
    Z = Circuit(cs).Z(p, f) * (1 + 0.003 * np.random.default_rng(0).normal(size=65))
    for label in ("cold", "warm"):
        t = [time.time()]

        def prog(m):
            t.append(time.time())
            print(f"  {t[-1] - t[-2]:5.2f}s  -> {m}", flush=True)

        r = pipeline.analyze(Spectrum("x", f, Z), {}, {"mode": mode}, progress=prog)
        print(f"{label}: total {time.time() - t[0]:.2f}s, {len(r['candidates'])} candidates, rec {r['recommended']}")

"""측정 하나의 단계별 시간 (KK · DRT · 후보 피팅):  python tests/profile_stages.py <파일> [측정 번호들]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf import analysis  # noqa: E402
from beaf.eis_io import load_file  # noqa: E402
from beaf.fitting import fit_circuit  # noqa: E402
from beaf.pipeline import make_candidates  # noqa: E402

if __name__ == "__main__":
    p = Path(sys.argv[1])
    es = load_file(p.read_bytes(), p.name)
    for k in map(int, sys.argv[2:] or ["0", "30", "59"]):
        sp = es[k]["cycles"][es[k]["default_cycle"]]
        f, Z = sp.freq, sp.Z
        rg = analysis.auto_range(f, Z)
        t = time.time(); kk = analysis.kk_test(f[rg["start"]:], Z[rg["start"]:]); t_kk = time.time() - t
        m = analysis.fit_mask(f, Z, "arcs")
        t = time.time(); drt = analysis.drt_analysis(f[m], Z[m]); t_drt = time.time() - t
        cands = make_candidates({}, drt.get("n_robust", 0), "arcs", rg["has_tail"], (False, True))
        times = []
        for c in cands:
            t = time.time(); fit_circuit(c["circuit"], f[m], Z[m], drt_peaks=drt.get("peaks"), n_starts=3); times.append((time.time() - t, c["circuit"]))
        times.sort(reverse=True)
        print(f"#{k}: KK {t_kk:.2f}s (RC {kk.get('num_RC')}) | DRT {t_drt:.2f}s | 피팅 {len(cands)}개 합 {sum(x for x, _ in times):.2f}s, "
              f"가장 느린 것 {times[0][0]:.2f}s {times[0][1]}", flush=True)

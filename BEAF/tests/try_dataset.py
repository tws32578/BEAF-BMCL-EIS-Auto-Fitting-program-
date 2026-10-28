"""외부 데이터셋 몇 개를 분석해 결과 확인:  python tests/try_dataset.py <파일> [셀구성] [구간]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

if __name__ == "__main__":
    p = Path(sys.argv[1])
    cell = sys.argv[2] if len(sys.argv) > 2 else "full"
    mode = sys.argv[3] if len(sys.argv) > 3 else "arcs"
    es = load_file(p.read_bytes(), p.name)
    pick = [es[2], es[len(es) // 2], es[-3]]
    for e in pick:
        sp = e["cycles"][e["default_cycle"]]
        t = time.time()
        r = analyze(sp, {"cell": cell, "electrode": "porous"}, {"mode": mode})
        c = r["candidates"][0]
        vals = ", ".join(f"{n}={v * 1000:.3g} mΩ" for n, v in zip(c["display_names"], c["values"]) if c["units"][c["display_names"].index(n)] == "Ω")
        print(f"{e['name'][13:70]:<58} {time.time() - t:4.1f}s {r['status']} n_drt={r['n_drt']} range={r['range']['start']}..{r['range']['valley']}/{len(sp.freq)}")
        print(f"   rec {c['circuit']}  rms {c['rms_pct']:.2f}%  KK {r['kk'].get('noise_pct', 0):.2f}%  | {vals}")
        print(f"   reasons {r['reasons']}")

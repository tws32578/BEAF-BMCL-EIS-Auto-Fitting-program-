"""실측 Gr||Gr 에서 2반원 회로와 TLM 회로 비교:  python tests/compare_tlm.py <데이터 폴더> <샘플> ..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for sample in sys.argv[2:]:
        print(f"== {sample}")
        for p in sorted((root / sample).rglob("*.mpt")):
            e = load_file(p.read_bytes(), p.name)[0]
            r = analyze(e["cycles"][e["default_cycle"]], {"cell": "sym", "electrolyte": "liquid"}, {})
            ok = [c for c in r["candidates"] if c.get("ok")]
            plain = next((c for c in ok if not c.get("shape") and c.get("n_arcs") == 2 and c["valid"]), None)
            tlm = min((c for c in ok if c.get("shape") == "TLMn" and c["valid"]), key=lambda c: c["bic"], default=None)
            rec = r["candidates"][0]

            def vals(c):
                return ", ".join(f"{n}={v:.3g}" for n, v, u in zip(c["display_names"], c["values"], c["units"]) if u == "Ω") if c else "-"
            print(f"  {e['meta'].get('T_C')}°C 추천 {rec['circuit']}")
            if plain:
                print(f"     2반원: 잔차 {plain['rms_pct']:.2f}% BIC {plain['bic']:.0f} | {vals(plain)}")
            if tlm:
                sp = tlm["specials"][0]
                print(f"     TLM  : 잔차 {tlm['rms_pct']:.2f}% BIC {tlm['bic']:.0f} | {vals(tlm)} | TLM 직류 {sp['R']:.3g} Ω, f {sp['f_peak']:.3g} Hz, DRT일치 {tlm.get('drt_supported')}")

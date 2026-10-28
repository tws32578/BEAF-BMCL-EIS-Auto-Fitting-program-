"""시리즈 분석 동작 확인: python tests/series_demo.py <데이터 폴더> <샘플 폴더 이름> ..."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf import series  # noqa: E402
from beaf.eis_io import read_any  # noqa: E402
from beaf.pipeline import DEFAULT_META, analyze  # noqa: E402
from tests.validate_manual import pick_files  # noqa: E402

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for sample in sys.argv[2:]:
        t = time.time()
        items = []
        for k, p in enumerate(pick_files(root / sample)):
            sp = read_any(p.read_bytes(), p.name)
            r = analyze(sp, {}, {})
            items.append({"id": str(k), "name": p.name, "x": sp.meta["T_C"], "result": r})
        g = series.analyze_group(sample, items, "T_C", DEFAULT_META)
        print(f"== {sample}: circuit {g['circuit']}  ({time.time() - t:.0f}s)")
        print("   ranking:", [(x["circuit"], f"{x['n_supported']}/{x['n']}") for x in g["ranking"][:3]])
        for r in g["rows"]:
            vals = " ".join(f"{n}={v:.3f}" for n, v in zip(g["param_names"], r["values"]) if n.startswith("R"))
            print(f"   {r["x"]:>5} 제외사유={r.get("excluded")} flags={r.get("flags")} {r["seed"]:<6} rms={r['rms_pct']:.2f}% {vals}")
        for tr in g["trends"]:
            a = tr.get("arrhenius")
            out = [g["rows"][k]["x"] for k, o in enumerate(tr["outlier"]) if o]
            print(f"   {tr["param"]:<10} 제외={a.get("excluded_x")} Ea={a['Ea']:.1f}±{a['Ea_err']:.1f} kJ/mol R²={a['r2']:.3f}  이탈={out}")

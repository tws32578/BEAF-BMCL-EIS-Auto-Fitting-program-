"""AutoEIS 베이지안: 출발점·우도 방식에 따른 결과 비교"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf import autoeis_bridge as ab  # noqa: E402
from beaf.eis_io import read_any  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

if __name__ == "__main__":
    p = Path(sys.argv[1])
    sp = read_any(p.read_bytes(), p.name)
    r = analyze(sp, {}, {})
    c = r["candidates"][0]
    import numpy as np
    m = np.array(r["mask"])
    f, Z = sp.freq[m], sp.Z[m]
    s, p0, pmap = ab.from_beaf(c["circuit"], c["names"], c["values"])
    print("CNLS", {n: round(v, 4) for n, v in zip(c["names"], c["values"])})
    for method, refine in (("bode", True), ("bode", False), ("nyquist", False), ("chi-squared", False)):
        res = ab.bayes(s, f, Z, p0, 1000, 500, method, refine)
        if not res["ok"]:
            print(method, refine, res["error"]); continue
        post = ab.summarize_posterior(c["circuit"], {pmap[k]: v for k, v in res["samples"].items() if k in pmap})
        print(f"{method:<12} refine={refine!s:<5} {res['seconds']:.0f}s",
              {n: f"{post[n]['median']:.4g} [{post[n]['lo95']:.3g},{post[n]['hi95']:.3g}]" for n in c["names"] if n.startswith("R") or n.endswith("_1")})

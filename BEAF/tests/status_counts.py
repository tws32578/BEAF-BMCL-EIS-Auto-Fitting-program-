"""데이터 묶음 전체의 판정 분포와 시간, '모양 주의'·모양 후보 추천 수:  python tests/status_counts.py <파일 또는 폴더> [셀] [전해질]"""
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import SUPPORTED_ALL, load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

if __name__ == "__main__":
    src = Path(sys.argv[1])
    meta = {"cell": sys.argv[2] if len(sys.argv) > 2 else "unknown",
            "electrolyte": sys.argv[3] if len(sys.argv) > 3 else "unknown"}
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in (".mpt",) )
    specs = []
    for p in files:
        for e in load_file(p.read_bytes(), p.name):
            specs.append((e["name"], e["cycles"][e["default_cycle"]]))
    t = time.time()
    with ThreadPoolExecutor(3) as ex:
        rs = list(ex.map(lambda x: analyze(x[1], meta, {}), specs))
    dt = time.time() - t
    st = Counter(r["status"] for r in rs)
    shape_warn = [n for (n, _), r in zip(specs, rs) if any(x.startswith("모양 주의") for x in r["reasons"])]
    shape_rec = [(n, r["recommended"]) for (n, _), r in zip(specs, rs) if r["candidates"] and r["candidates"][0].get("shape")]
    print(f"{len(rs)}개, {dt:.0f} s ({dt / len(rs):.1f} s/개) | 판정 {dict(st)}")
    print(f"모양 주의 {len(shape_warn)}개: {[n[:40] for n in shape_warn[:8]]}")
    print(f"모양 후보가 추천된 것 {len(shape_rec)}개: {[(n[:30], c) for n, c in shape_rec[:8]]}")

"""
설정별 분석 시간 비교:  python tests/timing.py <파일> [측정 개수]
화면과 같게 3개씩 동시에 분석
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402

CONFIGS = [
    ("셀 모름 · 반원 구간", {}, {"mode": "arcs"}),
    ("풀셀 · 액체 · 반원 구간", {"cell": "full", "electrolyte": "liquid"}, {"mode": "arcs"}),
    ("셀 모름 · 전체 구간(확산 포함)", {}, {"mode": "full"}),
    ("풀셀 · 액체 · 전체 구간", {"cell": "full", "electrolyte": "liquid"}, {"mode": "full"}),
]

if __name__ == "__main__":
    p = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    es = load_file(p.read_bytes(), p.name)
    specs = [e["cycles"][e["default_cycle"]] for e in es[::max(1, len(es) // n)][:n]]
    analyze(specs[0], {}, {})                              # 프로세스 풀 데우기
    for name, meta, st in CONFIGS:
        t = time.time()
        with ThreadPoolExecutor(3) as ex:
            rs = list(ex.map(lambda sp: analyze(sp, meta, st), specs))
        dt = time.time() - t
        nc = sum(len(r["candidates"]) for r in rs) / len(rs)
        print(f"{name:<28} {dt / len(specs):5.2f} s/측정  후보 {nc:4.1f}개  → 60개면 약 {dt / len(specs) * 60 / 60:4.1f} 분", flush=True)

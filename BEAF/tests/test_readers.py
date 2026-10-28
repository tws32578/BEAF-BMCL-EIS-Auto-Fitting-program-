"""여러 파일 형식 읽기 확인:  python tests/test_readers.py <파일> ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import load_file  # noqa: E402

if __name__ == "__main__":
    for p in map(Path, sys.argv[1:]):
        es = load_file(p.read_bytes(), p.name)
        print(f"== {p.name}: 측정 {len(es)}개")
        for e in es[:3] + es[-1:]:
            sp = e["cycles"][e["default_cycle"]]
            meta = {k: v for k, v in e["meta"].items() if v is not None and k not in ("_labels", "x")}
            print(f"   {e['name'][:80]} | cycles {list(e['cycles'])} | n={len(sp.freq)} "
                  f"f {sp.freq[0]:.4g}→{sp.freq[-1]:.4g} | Z(hi)={sp.Z[0]:.4g} Z(lo)={sp.Z[-1]:.4g} | {meta}")

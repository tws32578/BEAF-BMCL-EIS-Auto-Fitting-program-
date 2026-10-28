"""그룹 정보로 회로 순위가 어떻게 바뀌는지 확인: python tests/group_demo.py <데이터 폴더> <샘플> ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaf.eis_io import read_any  # noqa: E402
from beaf.group import apply_group_prior  # noqa: E402
from beaf.pipeline import analyze  # noqa: E402
from tests.validate_manual import pick_files  # noqa: E402

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for sample in sys.argv[2:]:
        rs = [analyze(read_any(p.read_bytes(), p.name), {}, {}) for p in pick_files(root / sample)]
        own = [r["recommended"] for r in rs]
        sm = apply_group_prior(sample, rs)
        print(f"== {sample}: 공통 회로 {sm['consensus']}  ({sm['ranking'][0]['n_supported']}/{sm['n_files']} 지지)")
        for r, o in zip(rs, own):
            print(f"   {r['file_meta']['T_C']:>5}  단독 {o:<40} → 추천 {r['recommended']:<40} | {r['group']['note']}")

"""
그룹 정보로 회로 우선순위 정하기

파일은 각각 독립적으로 피팅합니다 (파라미터 공유 없음).
다만 같이 넣은 파일들(온도별·전해질별 등)은 같은 물리 과정을 가졌을 가능성이 높으므로,
회로를 추천할 때 그룹 전체의 근거를 참고합니다.

  1) 그룹 공통 회로 = 그룹 안에서 '유효 + DRT 일치'인 파일이 가장 많은 회로 (동점이면 BIC 합이 작은 쪽)
  2) 각 파일에서 공통 회로가 유효하고 DRT 와 맞으며, 그 파일 단독 최적 회로보다 BIC 가 Δ(기본 10) 이하로만 나쁘면
     공통 회로를 1순위로 올림.  Δ = 10 은 '강한 증거' 기준(Kass & Raftery) — 이보다 크게 나쁘면 데이터가
     다른 회로를 분명히 원한다는 뜻이므로 그 파일 단독 최적을 유지하고 이유를 표시
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .pipeline import order_candidates

DELTA_BIC = 10.0


def own_rank_key(c):
    return (not c.get("ok"), not c.get("valid"), not c.get("drt_supported", True), c.get("bic", np.inf))


def consensus(results):
    """그룹 전체에서 회로별 점수 → 순위 목록"""
    score = defaultdict(lambda: [0, 0, 0.0, 0])
    for r in results:
        for c in r.get("candidates", []):
            if not c.get("ok"):
                continue
            s = score[c["circuit"]]
            s[0] += int(c["valid"] and c.get("drt_supported", True))
            s[1] += int(c["valid"])
            s[2] += c["bic"]
            s[3] += 1
    n = len(results)
    full = {k: v for k, v in score.items() if v[3] == n}       # 모든 파일에서 피팅된 회로만 비교
    pool = full or score
    ranked = sorted(pool.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[1][2]))
    return [{"circuit": k, "n_supported": v[0], "n_valid": v[1], "sum_bic": v[2], "n": v[3]} for k, v in ranked]


def _rerank(r, first=None):
    ok = order_candidates([c for c in r["candidates"] if c.get("ok")])
    if first is not None:
        ok = [first] + [c for c in ok if c is not first]
    bad = [c for c in r["candidates"] if not c.get("ok")]
    r["candidates"] = ok + bad
    if ok:
        b0 = ok[0]["bic"]
        for c in ok:
            c["dBIC"] = c["bic"] - b0
        r["recommended"] = ok[0]["circuit"]


def apply_group_prior(name, results, delta_bic=DELTA_BIC):
    """
    results: 같은 그룹의 단독 분석 결과들 (제자리에서 수정)
    반환: 그룹 요약 dict
    """
    results = [r for r in results if r.get("candidates")]
    if len(results) < 2:
        for r in results:
            _clear(r)
            _rerank(r)
        return None
    ranking = consensus(results)
    if not ranking:
        return None
    cons = ranking[0]["circuit"]
    summary = {"name": name, "consensus": cons, "ranking": ranking[:6], "n_files": len(results),
               "adopted": 0, "kept_own": []}
    for r in results:
        _clear(r)
        _rerank(r)                                        # 단독 기준 순위로 되돌린 뒤
        own = r["candidates"][0]
        r["recommended_own"] = own["circuit"]
        c = next((c for c in r["candidates"] if c.get("ok") and c["circuit"] == cons), None)
        note = None
        if own["circuit"] == cons:
            note = f"그룹 공통 회로와 같음 ({ranking[0]['n_supported']}/{len(results)} 파일 지지)"
            summary["adopted"] += 1
        elif c is None:
            note = f"그룹 공통 회로 {cons} 를 이 파일에 피팅하지 못함 → 단독 최적 유지"
        elif not c["valid"]:
            note = f"그룹 공통 회로 {cons} 는 이 파일에서 탈락 → 단독 최적 {own['circuit']} 유지"
        elif not c.get("drt_supported", True):
            note = f"그룹 공통 회로 {cons} 는 이 파일의 DRT 피크와 안 맞음 → 단독 최적 {own['circuit']} 유지"
        else:
            d = c["bic"] - own["bic"]
            if d <= delta_bic:
                _rerank(r, first=c)
                note = (f"그룹 공통 회로 채택 (단독 최적 {own['circuit']} 보다 ΔBIC {d:+.1f}, "
                        f"기준 {delta_bic:g} 이내)")
                summary["adopted"] += 1
            else:
                note = (f"그룹 공통 회로 {cons} 는 ΔBIC {d:+.1f} 로 크게 나쁨 → 이 파일은 다른 회로가 필요 "
                        f"({own['circuit']})")
        if r["recommended"] != cons:
            summary["kept_own"].append(r["name"])
        r["group"] = {"name": name, "consensus": cons, "note": note,
                      "support": f"{ranking[0]['n_supported']}/{len(results)}"}
        r["reasons"] = [x for x in r["reasons"] if not x.startswith("[그룹]")] + [f"[그룹] {note}"]
    return summary


def _clear(r):
    r.pop("group", None)
    r["reasons"] = [x for x in r.get("reasons", []) if not x.startswith("[그룹]")]

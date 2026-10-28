"""
BEAF 분석 파이프라인
  KK 검증 → 피팅 구간 → DRT → 블록 조립식 회로 후보 → 다중 시작점 피팅 → AIC/BIC + 물리 판정 → 추천

회로는 완성본을 저장하지 않고 블록을 조립합니다.
  [L0] - R0 - (R||Q) × n - [확산 요소]
  n 은 (DRT 확실한 피크 수 ± 1) 과 (셀 메타데이터로 예상한 반원 수) 에서 고름
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import analysis
from .circuits import Circuit
from .fitting import diagnose, fit_circuit

# ---------------------------------------------------------------- 병렬 피팅
_POOL = None


def _fit_job(job):
    circuit, f, Z, peaks, n_starts, noise, p0 = job
    try:
        if p0 is not None:
            # 외부(AutoEIS 등)에서 받은 초기값과 BEAF 자체 초기값 둘 다 출발점으로 시도
            a = fit_circuit(circuit, f, Z, p0=p0, drt_peaks=peaks, n_starts=n_starts, kk_noise_pct=noise)
            b = fit_circuit(circuit, f, Z, drt_peaks=peaks, n_starts=n_starts, kk_noise_pct=noise)
            ok = [x for x in (a, b) if x.get("ok")]
            return min(ok, key=lambda x: x["chi2"]) if ok else a
        return fit_circuit(circuit, f, Z, drt_peaks=peaks, n_starts=n_starts, kk_noise_pct=noise)
    except Exception as e:                                # noqa: BLE001
        return {"ok": False, "circuit": circuit, "error": str(e)}


def _pool():
    global _POOL
    if _POOL is None and not os.environ.get("BEAF_SERIAL"):
        _POOL = ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 2) - 1)))
    return _POOL


def submit(fn, *args):
    """다른 프로세스에서 실행할 작업 (풀이 없으면 None → 호출한 쪽에서 직접 실행)"""
    try:
        pool = _pool()
        return pool.submit(fn, *args) if pool else None
    except Exception:                                     # noqa: BLE001
        return None


def run_fits(jobs):
    """후보 피팅을 여러 프로세스로 나눠 실행 (실패하면 순차 실행)"""
    global _POOL
    if len(jobs) <= 1 or os.environ.get("BEAF_SERIAL"):
        return [_fit_job(j) for j in jobs]
    try:
        return list(_pool().map(_fit_job, jobs))
    except Exception:                                     # noqa: BLE001
        _POOL = None
        return [_fit_job(j) for j in jobs]


# ---------------------------------------------------------------- 메타데이터 선택지
# 셀 정보: 피팅 자체는 정보 없이도 됨 (DRT 가 반원 수를 정함).
#   셀 구성 · 전해질 → 예상 반원 수를 후보에 하나 더 넣고, 반원 이름(R_SEI, R_ct …)을 붙이는 데 씀
#   '모름' 이면 이름은 R1, R2 … (고주파 → 저주파)
CELL_TYPES = {"unknown": "모름", "sym": "대칭셀", "half": "하프셀 (Li 상대전극)", "full": "풀셀 (상용 셀 포함)",
              "3e": "3전극 (측정 전극 1개)"}
ELECTROLYTES = {"unknown": "모름", "liquid": "액체", "solid": "고체"}
ELECTRODES = {"unknown": "모름", "porous": "다공성 복합전극 (Gr, Si, NCM…)", "li": "Li 금속", "blocking": "블로킹 (SS 등)"}
STATES = {"unknown": "모름", "formed": "포메이션 후 (SEI 있음)", "fresh": "포메이션 전"}

DEFAULT_META = {"cell": "unknown", "electrode": "unknown", "electrolyte": "unknown", "state": "unknown"}


def expected_arcs(meta):
    """셀 정보로 예상하는 반원 수와 반원 이름(고주파 → 저주파)"""
    cell, el, ely, st = (meta.get(k, DEFAULT_META[k]) for k in ("cell", "electrode", "electrolyte", "state"))
    if cell == "unknown":
        return []                                    # 정보 없음 → 이름은 R1, R2 …
    if el == "blocking":
        names = []
    elif el == "li":
        names = ["R_SEI+ct"] if st == "fresh" else ["R_SEI", "R_ct"]
    else:
        names = ["R_ct"] if st == "fresh" else ["R_SEI", "R_ct"]
    if cell in ("half", "full") and el != "li":
        names = names + ["R_LF"]                     # 상대전극(Li / 양극) 반원이 하나 더 보일 수 있음
    if ely == "solid":
        names = ["R_gb"] + names                     # 고체 전해질 입계 반원
    return names


def arc_labels(meta, n):
    """반원 n 개에 τ 오름차순으로 붙일 이름"""
    base = expected_arcs(meta)
    if not base:
        return [f"R{k + 1}" for k in range(n)]
    if n == len(base):
        return base
    if n < len(base):
        # 앞쪽(고주파) 반원들이 하나로 합쳐진 것으로 표시
        merged = base[:len(base) - n + 1]
        return ["R_" + "+".join(x[2:] for x in merged)] + base[len(base) - n + 1:]
    extra = n - len(base)
    pre = ["R_c"] if extra >= 1 else []              # 고주파 접촉/집전체 반원
    rest = [f"R_x{k + 1}" for k in range(extra - 1)]
    return pre + base + rest


def series_label(meta):
    return "R_bulk" if meta.get("electrolyte") == "liquid" else "R_s"


# ---------------------------------------------------------------- 회로 후보
def build_circuit(n_arcs, tail=None, with_L=False, nested=False):
    parts = []
    if with_L:
        parts.append("L0")
    parts.append("R0")
    if nested and n_arcs == 2:
        parts.append("p(R1-p(R2,CPE2),CPE1)")
    else:
        parts += [f"p(R{k},CPE{k})" for k in range(1, n_arcs + 1)]
    if tail == "CPE":
        parts.append("CPE9")
    elif tail in ("W", "Wo", "Ws"):
        parts.append(f"{tail}9")
    return "-".join(parts)


TAIL_DESC = {None: "", "CPE": "+ Q (확산/블로킹)", "W": "+ W (반무한 확산)",
             "Wo": "+ Wo (유한 확산, 반사형)", "Ws": "+ Ws (유한 확산, 투과형)"}


MAX_ARCS = 6

SHAPE_DESC = {"G": "Gerischer (비대칭 과정: 반응과 확산이 함께 일어나는 경우 등)",
              "TLMn": "다공성 전극 TLM (전극 속 이온 저항 + 전하전달)",
              "TLMQ": "블로킹 다공성 전극 TLM (전극 속 이온 저항 + 이중층)"}


def build_shape_circuit(n_arcs, kind, with_L=False, tail=None):
    """반원 n 개 + Gerischer 또는 전송선 요소 하나 (이름 9 번: 다른 소자와 겹치지 않게)"""
    parts = (["L0"] if with_L else []) + ["R0"] + [f"p(R{k},CPE{k})" for k in range(1, n_arcs + 1)] + [f"{kind}9"]
    if tail == "CPE":
        parts.append("CPE8")
    elif tail in ("W", "Wo", "Ws"):
        parts.append(f"{tail}8")
    return "-".join(parts)


def shape_candidates(n_drt, mode, has_tail, L_options):
    """
    R||CPE 반원으로는 흉내만 낼 수 있는 모양을 직접 표현하는 후보.
    같은 기준(BIC · DRT 일치)으로 반원 후보와 경쟁시켜, 비슷하거나 더 잘 맞으면 경고하거나 추천
    """
    k = max(n_drt, 1)
    combos = [("G", k - 1), ("G", k), ("TLMn", k - 1)]
    if k >= 2:
        combos.append(("TLMn", k - 2))
    if mode == "full" and has_tail:
        combos.append(("TLMQ", k - 1))
    out = []
    for wl in L_options:
        Ld = "L + " if wl else ""
        for kind, n in combos:
            if n < 0:
                continue
            arcs = f" + (RQ)×{n}" if n else ""
            out.append({"circuit": build_shape_circuit(n, kind, wl), "n_arcs": n, "tail": None, "L": wl,
                        "shape": kind, "desc": f"{Ld}R{arcs} + {SHAPE_DESC[kind]}"})
    return out


def make_candidates(meta, n_drt, mode, has_tail, L_options=(False,), nested=False, ns_extra=()):
    """
    L_options: (False,) | (True,) | (False, True)
      데이터에 인덕턴스 영역이 있으면 같은 구간에서 L 있는/없는 후보를 모두 피팅해 BIC 로 비교
    ns_extra: 추가로 시도할 반원 수 (추천이 시도한 최대 반원 수에 닿으면 하나 더 늘려 다시 부름)
    """
    n_exp = len(expected_arcs(meta))
    ns = set(ns_extra)
    if n_drt > 0:
        ns |= {n_drt - 1, n_drt, n_drt + 1}
    ns.add(n_exp)
    ns = sorted(n for n in ns if 1 <= n <= MAX_ARCS)
    if not ns:
        ns = [1, 2]
    tails = [None]
    if mode == "full" and has_tail:
        tails = ["CPE", "Wo", "W"]
    cands = []
    for wl in L_options:
        Ld = "L + " if wl else ""
        for n in ns:
            for t in tails:
                cands.append({"circuit": build_circuit(n, t, wl), "n_arcs": n, "tail": t, "L": wl,
                              "desc": f"{Ld}R + (RQ)×{n} {TAIL_DESC[t]}".strip()})
            if n == 2 and nested:
                t = tails[0]
                cands.append({"circuit": build_circuit(2, t, wl, nested=True), "n_arcs": 2, "tail": t, "L": wl,
                              "desc": f"{Ld}R + 중첩 (R(RQ))Q {TAIL_DESC[t]} — SEI 통과 후 전하전달".strip()})
    return cands


# ---------------------------------------------------------------- 표시용 이름
def display_names(fit, meta):
    """impedance.py 파라미터 이름 → 연구실 표기 (R_bulk, R_SEI, Q_SEI, α_SEI …)"""
    names = fit["names"]
    arcs = fit["arcs"]
    n_sp = len(fit.get("specials") or [])
    labels = arc_labels(meta, len(arcs) + n_sp)[:len(arcs)]    # Gerischer · TLM 은 저주파 과정 자리를 차지
    m = {}
    for a, lab in zip(arcs, labels):
        tag = lab[2:]
        m[a["R_name"]] = lab
        m[a["Q_name"] + "_0"] = f"Q_{tag}"
        m[a["Q_name"] + "_1"] = f"α_{tag}"
        m[a["Q_name"]] = f"C_{tag}"
    m["R0"] = series_label(meta)
    m["L0"] = "L"
    m["CPE9_0"], m["CPE9_1"] = "Q_d", "α_d"
    m["CPE8_0"], m["CPE8_1"] = "Q_d", "α_d"
    m.update({"G9_0": "R_G", "G9_1": "t_G", "TLMn9_0": "R_ion", "TLMn9_1": "R_ct(TLM)", "TLMn9_2": "Q_dl",
              "TLMn9_3": "α_dl", "TLMQ9_0": "R_ion", "TLMQ9_1": "Q_dl", "TLMQ9_2": "γ_dl",
              "Wo8_0": "R_W", "Wo8_1": "τ_W", "Ws8_0": "R_W", "Ws8_1": "τ_W", "W8": "A_W"})
    for t in ("W9", "Wo9_0", "Wo9_1", "Ws9_0", "Ws9_1"):
        m[t] = {"W9": "A_W", "Wo9_0": "R_W", "Wo9_1": "τ_W", "Ws9_0": "R_W", "Ws9_1": "τ_W"}[t]
    return [m.get(n, n) for n in names], labels


# ---------------------------------------------------------------- 전체 분석
def analyze(spec, meta=None, settings=None, progress=None):
    """
    spec: eis_io.Spectrum
    settings: mode('arcs'|'full'), fmin, fmax, exclude_idx, with_L, n_starts, custom(회로 문자열 목록)
    """
    meta = {**DEFAULT_META, **(meta or {})}
    st = {"mode": "arcs", "fmin": None, "fmax": None, "exclude_idx": [], "with_L": False,
          "n_starts": 8, "top_refit": 4, "custom": [], "nested": False, "auto_L": True,
          "quality_limit": 3.0, "target_rms_pct": 2.0, **(settings or {})}
    freq_all, Z_all = spec.freq, spec.Z

    def step(msg):
        if progress:
            progress(msg)

    # 1) KK (전체 데이터, 인덕턴스 영역 제외)
    step("KK 검증")
    rng = analysis.auto_range(freq_all, Z_all)
    kk_m = np.zeros(len(freq_all), bool)
    kk_m[rng["start"]:] = True
    # KK 는 오래 걸려서(1~4 s) 다른 프로세스에서 피팅과 동시에 돌림
    kk_future = submit(analysis.kk_test, freq_all[kk_m], Z_all[kk_m])

    # 2) 피팅 구간
    mask = analysis.fit_mask(freq_all, Z_all, st["mode"], st["fmin"], st["fmax"], st["exclude_idx"])
    if st["with_L"]:
        mask[:rng["start"]] = True if st["fmax"] is None else (freq_all[:rng["start"]] <= st["fmax"])
    f, Z = freq_all[mask], Z_all[mask]

    # 3) DRT (반원 구간에서)
    step("DRT")
    arc_m = analysis.fit_mask(freq_all, Z_all, "arcs", st["fmin"], st["fmax"], st["exclude_idx"])
    # DRT 는 λ 3개를 다른 프로세스에서 동시에 (KK 와도 동시에) 계산 — 가장 오래 걸리는 단계
    fa, Za = freq_all[arc_m], Z_all[arc_m]
    futs = [submit(analysis.drt_once_safe, fa, Za, lam) for lam in analysis.LAMBDAS]
    raw_runs = []
    for fu, lam in zip(futs, analysis.LAMBDAS):
        try:
            raw_runs.append(fu.result() if fu is not None else analysis.drt_once_safe(fa, Za, lam))
        except Exception:                             # noqa: BLE001
            raw_runs.append(analysis.drt_once_safe(fa, Za, lam))
    drt = analysis.drt_analysis(fa, Za, raw_runs)
    peaks = drt["peaks"] if drt.get("ok") else []
    peaks_weak = [q for q in drt.get("peaks_all", []) if not q["edge"]] if drt.get("ok") else []
    n_drt = len(peaks)

    # 4) 후보
    if st["with_L"]:
        L_opts = (True,)                      # 인덕턴스 영역 점까지 쓰면 L 은 필수
    elif st["auto_L"] and rng["has_inductive"] and st.get("L_if_inductive", False):   # 기본값 결정 대기 (tests/compare_L_policy.py)
        # −Z'' < 0 점이 있으면 L 이 있다는 직접 증거 → L 없는 회로는 R0 를 과대평가 (교차점이 반원 쪽으로 밀림)
        L_opts = (True,)
    elif st["auto_L"]:
        L_opts = (False, True)                # L 도 회로 성분 중 하나: 항상 있는/없는 후보 모두
    else:
        L_opts = (False,)
    cands = make_candidates(meta, n_drt, st["mode"], rng["has_tail"], L_opts, st["nested"])
    if st.get("shapes", True):
        cands += shape_candidates(n_drt, st["mode"], rng["has_tail"], L_opts)
    for c in st["custom"]:
        # 문자열(저장한 회로) 또는 {"circuit", "p0", "desc", "source"} (AutoEIS 탐색 결과 등)
        c = {"circuit": c} if isinstance(c, str) else dict(c)
        if not c.get("circuit") or any(c["circuit"] == x["circuit"] for x in cands):
            # 같은 회로가 이미 있으면 출처만 덧붙임
            for x in cands:
                if x["circuit"] == c.get("circuit") and c.get("source"):
                    x["source"] = (x.get("source", "BEAF") + " + " + c["source"])
            continue
        try:
            n_arcs = len(Circuit(c["circuit"]).rq_arcs())
        except Exception:                             # noqa: BLE001
            n_arcs = None
        cands.append({"circuit": c["circuit"], "n_arcs": n_arcs, "tail": None, "custom": True,
                      "desc": c.get("desc", "사용자 회로"), "source": c.get("source", "사용자"),
                      "p0": c.get("p0"), "L": bool(re.search(r"(?<![A-Za-z])L\d", c["circuit"]))})

    # 5) 피팅: 1단계 모든 후보를 시작점 3개로 빠르게 → 2단계 상위 후보만 전체 시작점
    noise = None                                      # 품질 비율은 KK 결과가 나온 뒤 계산

    def evaluate(c, fit):
        fit.update({k: v for k, v in c.items() if k not in ("circuit", "p0")})
        fit["flags"] = diagnose(fit, f)
        if fit.get("ok"):
            more, fit["drt_supported"] = drt_crosscheck(fit, peaks, peaks_weak)
            fit["flags"] += more
        fit["valid"] = fit.get("ok", False) and not any(x["severe"] for x in fit["flags"])
        return fit

    rank_key = lambda r: (not r.get("ok"), not r.get("valid"), not r.get("drt_supported", True),  # noqa: E731
                          r.get("bic", np.inf))
    step(f"피팅 1단계: 후보 {len(cands)}개")
    n_quick = min(3, st["n_starts"])
    jobs = [(c["circuit"], f, Z, peaks, n_quick, noise, c.get("p0")) for c in cands]
    results = [evaluate(c, fit) for c, fit in zip(cands, run_fits(jobs))]
    if st["n_starts"] > n_quick:
        top = sorted(range(len(results)), key=lambda i: rank_key(results[i]))[:st["top_refit"]]
        step(f"피팅 2단계: 상위 {len(top)}개 × 시작점 {st['n_starts']}개")
        jobs = [(cands[i]["circuit"], f, Z, peaks, st["n_starts"], noise, cands[i].get("p0")) for i in top]
        for i, fit in zip(top, run_fits(jobs)):
            results[i] = evaluate(cands[i], fit)
            results[i]["full_starts"] = True

    # 추천 후보가 시도한 가장 많은 반원 수를 쓰면 → 반원이 더 있을 수 있으므로 하나 더 늘려 시도 (최대 MAX_ARCS)
    for _ in range(MAX_ARCS):
        okr = [r for r in results if r.get("ok") and not r.get("shape")]
        if not okr:
            break
        best = min(okr, key=rank_key)
        n_max = max(r.get("n_arcs") or 0 for r in okr)
        if best.get("n_arcs") != n_max or n_max >= MAX_ARCS:
            break
        extra = [c for c in make_candidates(meta, 0, st["mode"], rng["has_tail"], L_opts, False, (n_max + 1,))
                 if c["n_arcs"] == n_max + 1 and all(c["circuit"] != x["circuit"] for x in cands)]
        if not extra:
            break
        step(f"반원 {n_max + 1}개 후보 추가")
        jobs = [(c["circuit"], f, Z, peaks, st["n_starts"], noise, None) for c in extra]
        for c, fit in zip(extra, run_fits(jobs)):
            cands.append(c)
            results.append(evaluate(c, fit))

    for fit in results:
        if fit.get("ok"):
            fit["display_names"], fit["arc_labels"] = display_names(fit, meta)
            # 성분별 반원 (고주파부터 R 을 누적해 옆으로 이어 그림)
            fit["arc_curves"] = arc_curves(fit, f)
            fl = np.logspace(np.log10(f.max()), np.log10(f.min()), 200)
            Zl = Circuit(fit["circuit"]).Z(fit["values"], fl)
            fit["line"] = {"freq": fl.tolist(), "re": Zl.real.tolist(), "im": Zl.imag.tolist()}

    try:
        kk = kk_future.result() if kk_future is not None else None
    except Exception:                                 # noqa: BLE001
        kk = None
    if kk is None:
        kk = analysis.kk_test(freq_all[kk_m], Z_all[kk_m])
    if kk.get("ok"):
        off = rng["start"]
        kk["suspect_idx"] = [i + off for i in kk["suspect_idx"]]
        noise = kk["noise_pct"]
        for fit in results:
            if fit.get("ok"):
                fit["quality_ratio"] = fit["rms_pct"] / noise if noise else None

    # 6) 순위 (유효 후보 BIC 순, 그 다음 무효 후보)
    ok = [r for r in results if r.get("ok")]
    # 순위: 유효 > DRT 피크와 일치 (합치거나 근거 없는 반원 없음) > BIC
    ok = order_candidates(ok)
    if ok:
        best_bic = ok[0]["bic"]                       # 추천 회로 기준 (음수 = 추천보다 BIC 가 낮음)
        for r in ok:
            r["dBIC"] = r["bic"] - best_bic
    failed = [r for r in results if not r.get("ok")]
    ranked = ok + failed
    rec = ranked[0] if ok else None

    status, reasons = "실패", []
    if rec:
        status = "통과"
        if not rec["valid"]:
            status = "검토"
            reasons.append("모든 후보에 물리적 문제가 있음 → 회로 직접 수정 또는 AutoEIS 탐색 권장")
        qr = rec.get("quality_ratio")
        # 품질: 잔차 RMS 가 (KK 노이즈 × 배수) 와 (목표 잔차) 중 큰 값 이하이면 통과
        limit = max(st["quality_limit"] * (noise or 0), st["target_rms_pct"])
        rec["rms_limit_pct"] = limit
        if rec["rms_pct"] > limit:
            status = "검토"
            kk_txt = f" (KK 노이즈의 {qr:.0f}배)" if qr else ""
            reasons.append(f"잔차 {rec['rms_pct']:.2f}% > 기준 {limit:.2f}%{kk_txt} → 회로가 데이터를 다 설명하지 못함")
        for x in rec["flags"]:
            if x.get("sum_only"):
                # 상태는 바꾸지 않고 경고만 (겹친 반원은 사람이 해도 20% 편차가 나는 구간)
                reasons.append(f"⚠ {x['msg']}")
        if rec["reproduce"][0] < max(2, rec["reproduce"][1] // 2):
            status = "검토"
            reasons.append("시작점에 따라 해가 달라짐")
        if len(ok) > 1 and ok[1]["valid"] and abs(ok[1]["dBIC"]) < 2:
            reasons.append(f"2순위 회로({ok[1]['circuit']})와 ΔBIC {ok[1]['dBIC']:.1f} (근소한 차이)")
        lower = [r for r in ok[1:] if r["valid"] and r["dBIC"] < -2]
        if lower:
            reasons.append(f"BIC 가 더 낮은 유효 회로 {len(lower)}개는 DRT 피크와 맞지 않아 뒤로 보냄 "
                           f"(예: {lower[0]['circuit']}, ΔBIC {lower[0]['dBIC']:.0f})")
        if kk.get("ok") and kk["verdict"] == "검토 필요":
            status = "데이터 이상"
            reasons.append(f"KK 잔차 {kk['noise_pct']:.2f}% → 측정 데이터 자체 확인 필요")
        # 모양 경고: Gerischer · TLM 후보가 추천 회로와 비슷하거나 더 잘 맞으면 반원별 해석에 주의
        if not rec.get("shape"):
            alt = [r for r in ok if r.get("shape") and r["valid"] and r["bic"] - rec["bic"] <= 2]
            if alt:
                a = min(alt, key=lambda r: r["bic"])
                reasons.append(f"모양 주의: '{SHAPE_DESC[a['shape']]}' 회로로도 비슷하게 설명됨 (ΔBIC {a['bic'] - rec['bic']:+.1f}) "
                               f"→ 반원을 나눈 값의 물리적 의미가 불확실. 전체 저항(R_Σ)을 우선 쓰세요")
                if status == "통과":
                    status = "검토"

    r0_ref = r0_reference(freq_all, Z_all, rng, ok, rec)

    return {
        "name": spec.name, "meta": meta, "settings": st, "R0_ref": r0_ref,
        "freq": freq_all.tolist(), "Zre": Z_all.real.tolist(), "Zim": Z_all.imag.tolist(),
        "mask": mask.tolist(), "range": rng, "kk": kk, "drt": drt, "n_drt": n_drt,
        "candidates": ranked, "recommended": rec["circuit"] if rec else None,
        "status": status, "reasons": reasons,
        "cycles": spec.cycles, "cycle": spec.cycle, "file_meta": spec.meta,
    }


SHAPE_MARGIN = 6.0


def r0_reference(freq, Z, rng, ok, rec):
    """
    인덕턴스 영역(−Z'' < 0)이 있을 때 직렬 저항을 두 가지로 함께 보고
      R_HF : −Z'' = 0 교차점의 실수부 (모델 없이 읽은 값, 수동 피팅 시작점과 같음)
      R0_L : 추천 회로와 같은 구조에 L 을 넣은 회로의 R0
    직렬 L 은 −Z'' 를 ωL 만큼 내려 교차점을 반원 쪽으로 밀어내므로, L 없는 회로의 R0 와 R_HF 는 크게 나오는 쪽.
    반대로 L 을 넣으면 고주파 반원과 R0 를 구분할 정보가 줄어 R0 가 정해지지 않을 수 있음 (그때 R0_L = None).
    """
    if not rng.get("has_inductive") or rec is None:
        return None
    i = rng["start"]
    y0, y1 = -Z.imag[i - 1], -Z.imag[i]               # y0 ≤ 0 < y1
    t = y0 / (y0 - y1) if y1 != y0 else 0.0
    R_HF = float(Z.real[i - 1] + t * (Z.real[i] - Z.real[i - 1]))
    f_x = float(10 ** (np.log10(freq[i - 1]) + t * (np.log10(freq[i]) - np.log10(freq[i - 1]))))
    has_L = any(n.startswith("L") for n in rec["names"])
    twin = rec if has_L else next((c for c in ok if c["circuit"] == "L0-" + rec["circuit"]), None)
    out = {"R_HF": R_HF, "f_cross": f_x, "rec_has_L": has_L,
           "R0_rec": dict(zip(rec["names"], rec["values"])).get("R0"),
           "R0_L": None, "L": None, "R0_L_circuit": twin["circuit"] if twin else None, "R0_L_note": None}
    if twin is None:
        out["R0_L_note"] = "L 을 넣은 같은 회로를 피팅하지 않음"
        return out
    v, e = dict(zip(twin["names"], twin["values"])), dict(zip(twin["names"], twin["stderr"]))
    R0, err = v.get("R0"), e.get("R0")
    out["L"] = v.get("L0")
    if R0 is None:
        out["R0_L_note"] = "직렬 저항이 없는 회로"
    elif R0 < 0.02 * R_HF or (err is not None and np.isfinite(err) and err > R0) or "R0" in twin.get("bound_hits", {}):
        out["R0_L_note"] = "L 을 넣으면 R0 가 첫 반원과 구분되지 않음 (고주파 쪽이 반원이 아닌 45° 모양일 때 흔함)"
    else:
        out["R0_L"] = float(R0)
        if not twin.get("valid"):
            out["R0_L_note"] = "L 을 넣은 회로는 다른 성분에 물리적 문제가 있음 — 참고값"
    return out


def order_candidates(ok):
    """
    피팅에 성공한 후보 순위: 유효 → DRT 일치 → BIC.
    모양 후보(Gerischer · TLM)는 파라미터가 적어 BIC 에서 근소하게 이기기 쉬우므로, 같은 단계의 일반 후보보다
    BIC 가 SHAPE_MARGIN(6) 이상 낮을 때만 1순위 (아니면 일반 후보가 1순위 + '모양 주의' 경고)
    """
    ok = sorted(ok, key=lambda r: (not r["valid"], not r.get("drt_supported", True), r["bic"]))
    if ok and ok[0].get("shape"):
        top = ok[0]
        plain = [r for r in ok if not r.get("shape") and r["valid"] == top["valid"]
                 and r.get("drt_supported", True) == top.get("drt_supported", True)]
        if plain and plain[0]["bic"] - top["bic"] < SHAPE_MARGIN:
            ok.remove(plain[0])
            ok.insert(0, plain[0])
    return ok


def drt_crosscheck(fit, peaks, peaks_weak=(), tol_dec=0.5):
    """
    피팅된 반원의 꼭대기 주파수가 DRT 피크와 맞는지 확인 (Notion 6단계)
      peaks      : 모든 λ 에서 보인 확실한 피크
      peaks_weak : 기준 λ 에서 보인 피크 (가장자리 제외)
    반환: (flags, supported)
      - 반원 수 = 확실한 피크 수인데 0.5 decade(약 3배) 이상 어긋남 → 탈락
        (α 가 작은 '뭉개진 CPE' 로 잔차만 줄인 해를 걸러냄)
      - 확실한 피크 2개를 반원 1개로 합침 → 참고 + 순위 뒤로
      - 어느 DRT 피크로도 설명 안 되는 반원 → 참고 + 순위 뒤로
      supported = 모든 확실한 피크에 반원이 하나씩 있고, 모든 반원이 어떤 DRT 피크와 맞음
    """
    arcs = list(fit["arcs"])
    specials = fit.get("specials") or []
    if not peaks:
        return [], True
    if not arcs and not specials:
        return [{"msg": f"DRT 피크 {len(peaks)}개가 있는데 R||CPE 반원이 없는 회로", "severe": False}], False
    flags = []
    pf = np.array([q["f"] for q in peaks])
    wf = np.array([q["f"] for q in peaks_weak] or pf)
    near = lambda x, arr, tol=tol_dec: len(arr) > 0 and np.min(np.abs(np.log10(x / np.asarray(arr)))) <= tol  # noqa: E731
    if specials:
        # Gerischer · TLM 은 DRT 에서 특성 주파수보다 높은 쪽으로 넓게(두 갈래로도) 퍼짐
        #   → [f/10, f×100] 안의 피크는 그 요소 몫으로 봄
        sf = [s_["f_peak"] for s_ in specials]
        af = [a["f_peak"] for a in arcs]
        in_sp = lambda y: any(fs / 10 <= y <= fs * 100 for fs in sf)  # noqa: E731
        covered = all(near(y, af) or in_sp(y) for y in pf)
        lone = [a["R_name"] for a in arcs if not near(a["f_peak"], np.concatenate([pf, wf]))]
        if not covered:
            flags.append({"msg": "DRT 확실한 피크 일부를 회로가 덮지 못함", "severe": False})
        if lone:
            flags.append({"msg": f"DRT 로 확인 안 되는 반원: {', '.join(lone)}", "severe": False})
        return flags, covered and not lone
    af = np.array([a["f_peak"] for a in arcs])

    if len(arcs) == len(peaks):
        bad = [(x, y) for x, y in zip(np.sort(af)[::-1], np.sort(pf)[::-1])
               if abs(np.log10(x / y)) > tol_dec]
        if bad:
            txt = ", ".join(f"{x:.3g} Hz ↔ {y:.3g} Hz" for x, y in bad)
            flags.append({"msg": f"DRT 피크와 불일치 (반원 ↔ 피크: {txt})", "severe": True})
            return flags, False
    covered = all(near(y, af) for y in pf)
    if not covered or len(arcs) < len(peaks):
        flags.append({"msg": f"DRT 확실한 피크 {len(peaks)}개 중 일부를 반원이 덮지 못함 (합쳐짐)", "severe": False})
    lone = [a["R_name"] for a in arcs if not near(a["f_peak"], np.concatenate([pf, wf]))]
    if lone:
        flags.append({"msg": f"DRT 로 확인 안 되는 반원: {', '.join(lone)}", "severe": False})
    return flags, covered and len(arcs) >= len(peaks) and not lone


def arc_curves(fit, f):
    """반원별 단독 곡선 (오프셋 = R0 + 앞선 반원 R 합)"""
    names, vals = fit["names"], np.array(fit["values"])
    fl = np.logspace(np.log10(max(f.max(), 1e-3)) + 0.5, np.log10(max(f.min(), 1e-6)) - 0.5, 200)
    w = 2 * np.pi * fl
    off = vals[names.index("R0")] if "R0" in names else 0.0
    out = []
    for a in fit["arcs"]:
        Zarc = 1 / (1 / a["R"] + a["Q"] * (1j * w) ** a["alpha"])
        out.append({"name": a["R_name"], "re": (off + Zarc.real).tolist(), "im": Zarc.imag.tolist()})
        off += a["R"]
    return out

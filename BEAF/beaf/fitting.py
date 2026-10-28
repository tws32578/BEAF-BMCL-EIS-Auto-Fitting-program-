"""
등가회로 피팅 (CNLS, 모듈러스 가중) + 다중 시작점 + 진단

개발 방향 문서의 '어려움 → 기능' 대응
  χ² 판단 기준 없음      → KK 노이즈 바닥 대비 품질 비율 (quality_ratio)
  Minimize 반복 무효      → 다중 시작점 자동 실행, 같은 해로 모인 비율 '재현 k/n'
  상하한 도달            → bound_hits ('α 상한 도달' 등)
  성분 순서 뒤바뀜        → τ 크기로 반원 정렬 (작은 τ = 고주파 = SEI 쪽)
  R_SEI/R_ct 분리 불안정  → 파라미터 상관·표준오차, '합만 신뢰' 경고 + 합의 오차
모델 비교: AIC = N ln(χ²/N) + 2k,  BIC = N ln(χ²/N) + k ln N   (N = 주파수 수 × 2)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .circuits import Circuit, arc_capacitance, arc_tau, special_processes

ALPHA_MIN, ALPHA_MAX = 0.4, 1.0


# ---------------------------------------------------------------- 초기값
def guess_params(circ: Circuit, freq, Z, drt_peaks=None, arc_R_hint=None):
    """
    회로 구조 + 데이터 + DRT 피크로 초기값을 만듭니다.
      R0(직렬)         ≈ 고주파 실수축 절편
      반원 k 의 R, τ  ≈ DRT 피크 (τ 작은 순으로 배정), 없으면 전체 저항을 나눠 τ 를 고르게 배치
      Q               = τ^α / R
    """
    p = np.zeros(circ.n_params)
    re = Z.real
    R_s = max(float(re[0]), 1e-4)
    R_pol = max(float(re.max() - re[0]), 1e-3 * max(R_s, 1e-3))
    f_hi, f_lo = float(freq.max()), float(freq.min())

    arcs = circ.rq_arcs()
    n_arc0 = len(arcs)
    # Gerischer · 전송선(TLM) 요소도 DRT 피크 하나씩을 맡음 (보통 반원보다 저주파 쪽)
    specials = [(name, kind, s_) for name, kind, s_ in circ.elements if kind in ("G", "TLMn", "TLMQ")]
    n_arc = n_arc0 + len(specials)
    peaks = sorted(drt_peaks or [], key=lambda q: q["tau"])
    # 피크 수가 반원 수와 다르면 τ 를 로그 간격으로 채움
    if len(peaks) >= n_arc and n_arc > 0:
        # 가장 큰 피크 n_arc 개를 고른 뒤 τ 순 정렬
        sel = sorted(sorted(peaks, key=lambda q: -q["R"])[:n_arc], key=lambda q: q["tau"])
        taus = [q["tau"] for q in sel]
        Rs = [max(q["R"], 1e-3 * R_pol) for q in sel]
        scale = R_pol / max(sum(Rs), 1e-12)
        Rs = [r * scale for r in Rs]
    else:
        tau_lo, tau_hi = 1 / (2 * np.pi * f_hi), 1 / (2 * np.pi * f_lo)
        taus = list(np.logspace(np.log10(tau_lo) + 0.7, np.log10(tau_hi) - 0.5, max(n_arc, 1)))
        for q in peaks:                                   # 있는 피크는 가장 가까운 자리에 반영
            k = int(np.argmin([abs(np.log(t / q["tau"])) for t in taus]))
            taus[k] = q["tau"]
        Rs = [R_pol / max(n_arc, 1)] * max(n_arc, 1)
    if arc_R_hint is not None:
        Rs = list(arc_R_hint)

    arc_R_idx = {a["iR"] for a in arcs}
    arc_Q_idx = {a["iQ"]: k for k, a in enumerate(arcs)}
    arc_R_pos = {a["iR"]: k for k, a in enumerate(arcs)}

    for name, kind, s in circ.elements:
        if kind == "R":
            p[s] = Rs[arc_R_pos[s]] if s in arc_R_idx else R_s
        elif kind == "CPE":
            a = 0.85
            if s in arc_Q_idx:
                k = arc_Q_idx[s]
                p[s] = taus[k] ** a / Rs[k]
            else:                                         # 확산/블로킹 CPE
                a = 0.7
                p[s] = 1 / (max(abs(Z[-1]), 1e-6) * (2 * np.pi * f_lo) ** a)
            p[s + 1] = a
        elif kind == "C":
            if s in arc_Q_idx:
                k = arc_Q_idx[s]
                p[s] = taus[k] / Rs[k]
            else:
                p[s] = 1 / (max(abs(Z[-1].imag), 1e-6) * 2 * np.pi * f_lo)
        elif kind in ("G", "TLMn", "TLMQ"):
            k = n_arc0 + [x[2] for x in specials].index(s)       # 반원 다음 자리 (τ 큰 쪽)
            Rk, tk = Rs[k], taus[k]
            if kind == "G":
                p[s], p[s + 1] = Rk, tk
            elif kind == "TLMn":
                a = 0.85
                p[s], p[s + 1], p[s + 3] = 0.5 * Rk, 0.7 * Rk, a
                p[s + 2] = tk ** a / (0.7 * Rk)
            else:
                g = 0.85
                p[s], p[s + 2] = max(R_pol, 1e-3), g
                p[s + 1] = 1 / (max(abs(Z[-1]), 1e-6) * (2 * np.pi * f_lo) ** g)
        elif kind == "L":
            p[s] = 1e-7
        elif kind == "La":
            p[s], p[s + 1] = 1e-7, 0.9
        elif kind == "W":
            p[s] = max(abs(Z[-1].imag), 1e-6) * np.sqrt(2 * np.pi * f_lo)
        elif kind in ("Wo", "Ws"):
            p[s] = max(0.5 * R_pol, 1e-3)
            p[s + 1] = 1 / (2 * np.pi * f_lo)
    return p


# ---------------------------------------------------------------- 변환 (로그 공간)
class _Transform:
    def __init__(self, circ: Circuit, p0, alpha_bounds=(ALPHA_MIN, ALPHA_MAX), bounds=None):
        self.kinds = circ.param_kinds
        # 파라미터별 소자 종류 (L 은 물리적으로 가능한 범위로 제한: 평평한 방향으로 끝없이 내려가는 것 방지)
        elem_of = [None] * circ.n_params
        for _, kind, s in circ.elements:
            elem_of[s] = kind
        lo, hi = [], []
        for i, (k, v) in enumerate(zip(self.kinds, p0)):
            if k == "alpha":
                lo.append(alpha_bounds[0])
                hi.append(alpha_bounds[1])
            elif circ.param_names[i] in (bounds or {}):  # 데이터에서 정한 범위 (예: 인덕턴스 영역에서 추정한 L)
                b_lo, b_hi = bounds[circ.param_names[i]]
                lo.append(np.log10(b_lo))
                hi.append(np.log10(b_hi))
            elif elem_of[i] in ("L", "La"):
                lo.append(-11.0)                          # 10 pH
                hi.append(-3.0)                           # 1 mH
            else:
                c = np.log10(max(v, 1e-30))
                lo.append(max(c - 8, -30))
                hi.append(c + 8)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.islog = np.array([k != "alpha" for k in self.kinds])

    def to_x(self, p):
        x = np.where(self.islog, np.log10(np.maximum(p, 1e-30)), p)
        return np.clip(x, self.lo + 1e-9, self.hi - 1e-9)

    def to_p(self, x):
        return np.where(self.islog, 10.0 ** x, x)

    def dp_dx(self, p):
        return np.where(self.islog, p * np.log(10), 1.0)


# ---------------------------------------------------------------- 1회 피팅
def _fit_once(circ, freq, Z, p0, tr):
    absZ = np.abs(Z)

    def resid(x):
        Zm = circ.Z(tr.to_p(x), freq)
        d = (Zm - Z) / absZ
        return np.concatenate([d.real, d.imag])

    x0 = tr.to_x(p0)
    res = least_squares(resid, x0, bounds=(tr.lo, tr.hi), method="trf",
                        x_scale="jac", max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)  # 수렴하는 시작점은 20~60회면 끝남
    return res


def _sort_arcs_by_tau(circ, p, cov):
    """직렬로 연결된 (중첩 아닌) R||CPE 반원들을 τ 오름차순으로 재배치"""
    arcs = [a for a in circ.rq_arcs() if not a["nested"]]
    if len(arcs) < 2:
        return p, cov
    # 같은 종류(CPE/C)끼리만 교환 가능
    if len({a["iA"] is None for a in arcs}) > 1:
        return p, cov

    def tau_of(a):
        al = p[a["iA"]] if a["iA"] is not None else 1.0
        return arc_tau(p[a["iR"]], p[a["iQ"]], al)

    order = sorted(range(len(arcs)), key=lambda k: tau_of(arcs[k]))
    if order == list(range(len(arcs))):
        return p, cov
    perm = np.arange(len(p))
    for slot, src in enumerate(order):
        a_dst, a_src = arcs[slot], arcs[src]
        perm[a_dst["iR"]] = a_src["iR"]
        perm[a_dst["iQ"]] = a_src["iQ"]
        if a_dst["iA"] is not None:
            perm[a_dst["iA"]] = a_src["iA"]
    p2 = p[perm]
    cov2 = cov[np.ix_(perm, perm)] if cov is not None else None
    return p2, cov2


# ---------------------------------------------------------------- 다중 시작점 피팅
def fit_circuit(circuit_str, freq, Z, p0=None, drt_peaks=None, n_starts=8,
                kk_noise_pct=None, seed=0, alpha_bounds=(ALPHA_MIN, ALPHA_MAX), bounds=None):
    """bounds: {파라미터 이름: (하한, 상한)} — 데이터에서 따로 정한 범위 (그 파라미터의 초기값은 범위의 기하 평균)"""
    circ = Circuit(circuit_str)
    if p0 is None:
        p0 = guess_params(circ, freq, Z, drt_peaks)
    p0 = np.array(p0, float)
    for nm, (b_lo, b_hi) in (bounds or {}).items():
        if nm in circ.param_names:
            p0[circ.param_names.index(nm)] = np.sqrt(b_lo * b_hi)
    bounds = {nm: b for nm, b in (bounds or {}).items() if nm in circ.param_names}
    tr = _Transform(circ, p0, alpha_bounds, bounds)
    rng = np.random.default_rng(seed)

    starts = [np.array(p0, float)]
    for _ in range(max(n_starts - 1, 0)):
        x = tr.to_x(p0)
        x = np.where(tr.islog, x + rng.normal(0, 0.5, len(x)),
                     np.clip(x + rng.uniform(-0.2, 0.1, len(x)), alpha_bounds[0] + 0.01, alpha_bounds[1] - 0.01))
        starts.append(tr.to_p(x))

    runs = []
    for s in starts:
        try:
            r = _fit_once(circ, freq, Z, s, tr)
        except Exception:                                 # noqa: BLE001
            continue
        if np.all(np.isfinite(r.fun)):
            runs.append(r)
    if not runs:
        return {"ok": False, "circuit": circuit_str, "error": "모든 시작점에서 수렴 실패"}

    best = min(runs, key=lambda r: float(np.sum(r.fun ** 2)))
    chi2 = float(np.sum(best.fun ** 2))
    # 고정 파라미터 (범위가 0.1% 이내로 묶인 것, 예: R0 = x절편) 는 자유도에서 뺌
    fixed = [nm for nm, (b_lo, b_hi) in bounds.items() if b_hi <= b_lo * 1.001]
    free = np.array([nm not in fixed for nm in circ.param_names])
    N, k = 2 * len(freq), int(free.sum())
    p = tr.to_p(best.x)

    # 공분산 (변환 공간 → 원래 공간)
    cov = None
    try:
        J = best.jac[:, free]
        s2 = chi2 / max(N - k, 1)
        cov_x = np.zeros((circ.n_params, circ.n_params))
        cov_x[np.ix_(free, free)] = np.linalg.pinv(J.T @ J) * s2
        D = tr.dp_dx(p)
        cov = cov_x * np.outer(D, D)
    except Exception:                                     # noqa: BLE001
        pass

    # 상하한 도달 (τ 정렬 전 index 기준)
    span = tr.hi - tr.lo
    hit_lo = (best.x - tr.lo) < 1e-3 * span + 1e-6
    hit_hi = (tr.hi - best.x) < 1e-3 * span + 1e-6
    hits = {}
    for i, nm in enumerate(circ.param_names):
        if (hit_lo[i] or hit_hi[i]) and nm not in bounds:
            hits[i] = "상한" if hit_hi[i] else "하한"

    # τ 정렬
    p_sorted, cov_sorted = _sort_arcs_by_tau(circ, p, cov)
    if not np.allclose(p_sorted, p):
        # 상하한 표시도 값으로 다시 찾음
        new_hits = {}
        for i, h in hits.items():
            j = int(np.argmin(np.abs(p_sorted - p[i])))
            new_hits[j] = h
        hits = new_hits
    p, cov = p_sorted, cov_sorted
    err = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov is not None else np.full(k, np.nan)

    # 재현성: 최적 χ² 의 2% 이내 + 반원 R 들이 5% 이내로 같은 해
    arcs = circ.rq_arcs()
    idx_R = [a["iR"] for a in arcs] or [i for i, kd in enumerate(circ.param_kinds) if kd == "pos"][:1]
    same = 0
    for r in runs:
        c2 = float(np.sum(r.fun ** 2))
        pr, _ = _sort_arcs_by_tau(circ, tr.to_p(r.x), None)
        if c2 <= chi2 * 1.02 + 1e-15 and np.all(np.abs(pr[idx_R] - p[idx_R]) <= 0.05 * np.abs(p[idx_R]) + 1e-9):
            same += 1

    Zfit = circ.Z(p, freq)
    rms_pct = float(np.sqrt(chi2 / N) * 100)
    aic = float(N * np.log(chi2 / N) + 2 * k)
    bic = float(N * np.log(chi2 / N) + k * np.log(N))

    out = {
        "ok": True, "circuit": circuit_str, "k": k, "N": N, "fixed": fixed,
        "names": circ.param_names, "units": circ.param_units, "kinds": circ.param_kinds,
        "values": p.tolist(), "stderr": err.tolist(),
        "chi2": chi2, "rms_pct": rms_pct, "aic": aic, "bic": bic,
        "quality_ratio": (rms_pct / kk_noise_pct) if kk_noise_pct else None,
        "reproduce": [same, len(starts)],
        "bound_hits": {circ.param_names[i]: h for i, h in hits.items()},
        "Zfit": [Zfit.real.tolist(), Zfit.imag.tolist()],
        "cov": cov.tolist() if cov is not None else None,
    }
    out["arcs"] = describe_arcs(circ, p, cov)
    out["specials"] = special_processes(circ, p)          # Gerischer · TLM (반원이 아닌 과정)
    return out


def describe_arcs(circ, p, cov):
    """반원별 R, τ, 꼭대기 주파수, C_eff 와 반원끼리의 R 상관 / 합의 오차"""
    arcs = circ.rq_arcs()
    out = []
    for a in arcs:
        R, Q = p[a["iR"]], p[a["iQ"]]
        al = p[a["iA"]] if a["iA"] is not None else 1.0
        tau = arc_tau(R, Q, al)
        d = {"R_name": a["R"], "Q_name": a["Q"], "R": float(R), "Q": float(Q), "alpha": float(al),
             "tau": float(tau), "f_peak": float(1 / (2 * np.pi * tau)),
             "C_eff": float(arc_capacitance(R, Q, al)), "nested": a["nested"]}
        if cov is not None:
            d["R_err"] = float(np.sqrt(max(cov[a["iR"], a["iR"]], 0)))
        out.append(d)
    if cov is not None and len(arcs) >= 2:
        for i in range(len(arcs) - 1):
            a, b = arcs[i], arcs[i + 1]
            va, vb, cab = cov[a["iR"], a["iR"]], cov[b["iR"], b["iR"]], cov[a["iR"], b["iR"]]
            corr = cab / np.sqrt(va * vb) if va > 0 and vb > 0 else np.nan
            out[i]["corr_next"] = float(corr)
            out[i]["sum_with_next"] = float(p[a["iR"]] + p[b["iR"]])
            out[i]["sum_err"] = float(np.sqrt(max(va + vb + 2 * cab, 0)))
    return out


# ---------------------------------------------------------------- 판정
def diagnose(fit, freq):
    """
    물리적 타당성 플래그 (severe=True 면 추천 후보에서 제외)
    """
    flags = []
    if not fit.get("ok"):
        return [{"msg": fit.get("error", "실패"), "severe": True}]
    vals, errs, names, kinds = fit["values"], fit["stderr"], fit["names"], fit["kinds"]
    for nm, h in fit["bound_hits"].items():
        i = names.index(nm)
        if kinds[i] == "alpha":
            flags.append({"msg": f"{nm} (α) {h} 도달", "severe": h == "하한"})
        else:
            flags.append({"msg": f"{nm} 탐색 {h} 도달", "severe": True})
    for i, (v, e) in enumerate(zip(vals, errs)):
        if kinds[i] == "pos" and np.isfinite(e) and v > 0 and e / v > 1.0:
            # 저항이 결정 안 되면 탈락. Q 는 α 가 상한(1)에 붙으면 오차가 부풀려지므로 참고만
            is_R = names[i].startswith("R")
            flags.append({"msg": f"{names[i]} 결정 안 됨 (오차 {e / v * 100:.0f}%)", "severe": is_R})

    arcs = fit["arcs"]
    R_tot = sum(a["R"] for a in arcs) or 1
    fmin, fmax = float(np.min(freq)), float(np.max(freq))
    for a in arcs:
        if a["R"] < 0.01 * R_tot:
            flags.append({"msg": f"{a['R_name']} 이 전체의 1% 미만 (불필요한 요소)", "severe": True})
        if not (fmin / 3 <= a["f_peak"] <= fmax * 3):
            flags.append({"msg": f"{a['R_name']} 반원 꼭대기({a['f_peak']:.2g} Hz)가 측정 범위 밖", "severe": True})
    for i in range(len(arcs) - 1):
        a, b = arcs[i], arcs[i + 1]
        ratio = b["tau"] / a["tau"] if a["tau"] > 0 else np.inf
        corr = a.get("corr_next", np.nan)
        if ratio < 3:
            flags.append({"msg": f"{a['R_name']}·{b['R_name']} 시정수 차이 {ratio:.1f}배 (반원 겹침)", "severe": False})
        if np.isfinite(corr) and corr < -0.9:
            flags.append({"msg": f"{a['R_name']}·{b['R_name']} 강한 음의 상관 ({corr:.2f}) → 합만 신뢰",
                          "severe": False, "sum_only": [a["R_name"], b["R_name"]]})
    rep = fit["reproduce"]
    if rep[0] < max(2, rep[1] // 2):
        flags.append({"msg": f"재현 {rep[0]}/{rep[1]} (시작점 의존)", "severe": False})
    return flags

"""
전처리 · Kramers–Kronig 검증 · DRT

  1) 피팅 구간 자동 결정   auto_range()
       - 고주파 인덕턴스 영역(-Im < 0) 제외
       - 저주파 확산 꼬리 시작점(-Im 골짜기) 검출 → '반원 구간' 끝
  2) KK 검증               kk_test()      (pyimpspec, lin-KK 자동 RC 개수)
       - 잔차 RMS = 측정 노이즈 바닥 → 피팅 품질 판정 기준으로 사용
       - 저주파 끝에서 잔차가 큰 점(드리프트) 표시
  3) DRT                   drt_analysis() (pyimpspec TR-RBF)
       - λ 를 3가지로 바꿔 계산해 모든 λ 에서 남는 피크만 '확실한 피크'로 채택
       - 측정 주파수 창 가장자리 피크(가짜 피크 가능성)는 제외
       - 피크별 저항(면적)과 τ → 피팅 초기값
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- 1) 구간
def auto_range(freq, Z):
    """
    반환: dict(start, valley, end_full, has_tail, has_inductive)
      start   : 처음으로 -Im > 0 이 되는 index (freq 는 높은→낮은 순)
      valley  : 반원과 확산 꼬리 사이 -Im 골짜기 index (꼬리가 없으면 마지막 점)
    """
    mim = -Z.imag
    n = len(freq)
    start = int(np.argmax(mim > 0)) if np.any(mim > 0) else 0
    end = n - 1
    while end - 1 > start and mim[end - 1] <= mim[end]:
        end -= 1
    # 꼬리 판정: 골짜기 이후 점이 3개 이상이고 -Im 이 의미 있게 다시 커질 때
    has_tail = (n - 1 - end) >= 3 and (mim[-1] - mim[end]) > 0.05 * (np.ptp(mim[start:]) + 1e-12)
    if not has_tail:
        end = n - 1
    # 반원 끝 = 골짜기(최저점) 바로 앞 점. 최저점은 반원과 확산(45°) 직선이 섞인 점이라 반원 피팅에서 뺌
    #   (연구실 수동 피팅 기준: 'Warburg 로 꺾이기 직전 점')
    arc_end = end - 1 if has_tail and end - 1 > start else end
    return {"start": start, "valley": end, "arc_end": arc_end, "end_full": n - 1,
            "first_inductive": start - 1 if start > 0 else None,
            "has_tail": bool(has_tail), "has_inductive": bool(start > 0)}


def x_intercept(freq, Z, start):
    """−Z'' = 0 교차점의 실수부 R_HF 와 교차 주파수 (start = 처음 −Z'' > 0 인 index, start ≥ 1 일 때만)"""
    if start is None or start < 1:
        return None, None
    i = start
    y0, y1 = -Z.imag[i - 1], -Z.imag[i]               # y0 ≤ 0 < y1
    t = y0 / (y0 - y1) if y1 != y0 else 0.0
    R_HF = float(Z.real[i - 1] + t * (Z.real[i] - Z.real[i - 1]))
    f_x = float(10 ** (np.log10(freq[i - 1]) + t * (np.log10(freq[i]) - np.log10(freq[i - 1]))))
    return R_HF, f_x


def fit_mask(freq, Z, mode="arcs", fmin=None, fmax=None, exclude_idx=()):
    """피팅에 쓸 점 마스크. mode = 'arcs'(반원 구간: 반원 끝까지) | 'full'(확산 꼬리 포함)"""
    r = auto_range(freq, Z)
    m = np.zeros(len(freq), bool)
    last = r["arc_end"] if mode == "arcs" else r["end_full"]
    m[r["start"]:last + 1] = True
    if fmin is not None:
        m &= freq >= fmin
    if fmax is not None:
        m &= freq <= fmax
    for i in exclude_idx:
        if 0 <= i < len(m):
            m[i] = False
    return m


# ---------------------------------------------------------------- 2) KK
def kk_test(freq, Z):
    """
    pyimpspec lin-KK (복소 모드, RC 개수 자동). 인덕턴스 영역도 포함해 검사합니다.
    반환: dict(ok, noise_pct, res_re, res_im (%, 점별), num_RC, suspect_idx)
    """
    import pyimpspec

    try:
        ds = pyimpspec.DataSet(freq, Z)
        kk = pyimpspec.perform_kramers_kronig_test(ds, test="complex", num_procs=1)
        f, rr, ri = kk.get_residuals_data()          # 단위: %
    except Exception as e:                            # noqa: BLE001
        return {"ok": False, "error": str(e)}

    # pyimpspec 은 주파수 순서를 유지하지만, 혹시 몰라 원래 순서에 맞춤
    order = np.argsort(-f)
    f, rr, ri = f[order], rr[order], ri[order]
    absres = np.maximum(np.abs(rr), np.abs(ri))
    noise = float(np.sqrt(np.mean(np.concatenate([rr, ri]) ** 2)))
    med = float(np.median(absres))

    # 저주파 끝에서 연속으로 잔차가 큰 점 (드리프트·비정상 상태 의심)
    suspect = []
    thr = max(5 * med, 0.5)                           # 0.5 % 미만은 무시
    for i in range(len(f) - 1, -1, -1):
        if absres[i] > thr:
            suspect.append(i)
        else:
            break
    # 중간의 튀는 점
    for i in range(len(f)):
        if absres[i] > max(8 * med, 1.0) and i not in suspect:
            suspect.append(i)

    if noise < 0.5:
        verdict = "좋음"
    elif noise < 2:
        verdict = "보통"
    else:
        verdict = "검토 필요"
    return {"ok": True, "noise_pct": noise, "median_pct": med, "verdict": verdict,
            "freq": f.tolist(), "res_re": rr.tolist(), "res_im": ri.tolist(),
            "num_RC": int(kk.num_RC), "suspect_idx": sorted(suspect)}


# ---------------------------------------------------------------- 3) DRT
LAMBDAS = (1e-4, 1e-3, 1e-2)


def _drt_once(freq, Z, lam):
    import pyimpspec

    ds = pyimpspec.DataSet(freq, Z)
    d = pyimpspec.calculate_drt(ds, method="tr-rbf", lambda_value=lam, num_procs=1)
    tau, gamma = d.get_drt_data()
    return np.asarray(tau), np.asarray(gamma), float(d.lambda_value)


def _peaks(tau, gamma, tau_lo, tau_hi, rel_prom=0.05):
    """피크 index 와 면적(저항). 면적 = 이웃한 골짜기 사이 ∫γ d(ln τ)"""
    if gamma.max() <= 0:
        return []
    pk, _ = find_peaks(gamma, prominence=rel_prom * gamma.max())
    lnt = np.log(tau)
    out = []
    minima = [0] + list(find_peaks(-gamma)[0]) + [len(gamma) - 1]
    for p in pk:
        left = max([m for m in minima if m < p], default=0)
        right = min([m for m in minima if m > p], default=len(gamma) - 1)
        area = float(abs(np.trapezoid(gamma[left:right + 1], lnt[left:right + 1])))
        edge = not (tau_lo * 1.5 <= tau[p] <= tau_hi / 1.5)
        out.append({"tau": float(tau[p]), "f": float(1 / (2 * np.pi * tau[p])),
                    "gamma": float(gamma[p]), "R": area, "edge": bool(edge)})
    return out


def drt_once_safe(freq, Z, lam):
    """λ 하나의 DRT (다른 프로세스에서 병렬로 돌릴 수 있게 따로 둠). 실패하면 오류 문자열"""
    try:
        return _drt_once(freq, Z, lam)
    except Exception as e:                            # noqa: BLE001
        return str(e)


WARBURG_DEC = 0.5      # 반원 구간 저주파 끝에서 이 decade 안의 피크 = Warburg 꼬리가 남긴 피크로 봄


def drt_analysis(freq, Z, raw_runs=None, tail=False):
    """
    freq, Z : 피팅 구간 데이터.
    raw_runs: 미리 계산한 [drt_once_safe 결과] (λ 순서 = LAMBDAS). 없으면 여기서 차례로 계산
    tail    : 데이터에 저주파 확산 꼬리(Warburg)가 있음 → 구간 저주파 끝 0.5 decade 안의 피크는 반원으로 세지 않음
              (반원 구간이 Warburg 로 꺾이기 직전에서 끝나므로, 그 근처 피크는 확산 응답이 남긴 것)
    반환: dict(tau, gamma (기본 λ), lambda, peaks(확실), peaks_all, peaks_warburg, n_robust)
    """
    tau_lo, tau_hi = 1 / (2 * np.pi * freq.max()), 1 / (2 * np.pi * freq.min())
    if raw_runs is None:
        raw_runs = [drt_once_safe(freq, Z, lam) for lam in LAMBDAS]
    runs = []
    for res in raw_runs:
        if isinstance(res, str):
            return {"ok": False, "error": res}
        tau, gamma, lam_used = res
        runs.append((lam_used, tau, gamma, _peaks(tau, gamma, tau_lo, tau_hi)))

    base = runs[1]                                    # λ = 1e-3 를 표시용 기준으로
    robust, warb = [], []
    R_total = sum(p["R"] for p in base[3]) or 1.0
    f_lo = float(freq.min())
    for p in base[3]:
        if p["edge"]:
            continue
        if tail and p["f"] < f_lo * 10 ** WARBURG_DEC:
            p["warburg"] = True
            warb.append(p)
            continue
        if p["R"] < 0.02 * R_total:                   # 전체 분극 저항의 2% 미만 = 노이즈 피크로 봄
            continue
        # 다른 λ 에서도 τ 가 반 decade 이내에 피크가 있으면 '확실'
        seen = sum(any(abs(np.log10(q["tau"] / p["tau"])) < 0.5 for q in r[3]) for r in runs)
        p = dict(p, support=f"{seen}/{len(runs)}")
        if seen == len(runs):
            robust.append(p)
    return {"ok": True, "tau": base[1].tolist(), "gamma": base[2].tolist(),
            "lambda": base[0], "peaks": robust, "peaks_all": base[3], "peaks_warburg": warb,
            "n_robust": len(robust), "tau_window": [tau_lo, tau_hi],
            "lambda_runs": [{"lambda": r[0], "tau": r[1].tolist(), "gamma": r[2].tolist()}
                            for r in runs]}

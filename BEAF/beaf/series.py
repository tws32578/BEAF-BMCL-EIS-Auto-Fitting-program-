"""
시리즈 분석: 같은 그룹(샘플)의 여러 파일을 하나의 변수(온도, SOC, 사이클, 압력, 시간 … 무엇이든)로 정렬해
  1) 공통 회로 선택       — 그룹 전체에서 가장 잘 지지되는 회로 (또는 사용자가 지정)
  2) 순차 재피팅          — 기준 파일에서 시작해 이웃 파일의 결과를 초기값으로 넘김 (성분 뒤바뀜·엉뚱한 해 방지)
  3) 추세 이탈 감지       — 한 점씩 빼고 나머지로 추세선을 만들어 크게 벗어나는 점 표시
  4) Arrhenius            — 변수가 온도일 때 ln(1/R) vs 1000/T 기울기로 활성화에너지
"""

from __future__ import annotations

import numpy as np

from .fitting import fit_circuit
from .group import consensus
from .pipeline import arc_curves, diagnose, display_names, drt_crosscheck

R_GAS = 8.314462618e-3    # kJ/(mol·K)

VARIABLES = {
    "T_C": ("온도", "°C"), "SOC": ("SOC", "%"), "cycle_no": ("사이클", "회"),
    "P": ("압력", "MPa"), "time_h": ("시간", "h"), "E_V": ("전압", "V"), "x": ("사용자 정의", ""),
}


# ---------------------------------------------------------------- 1) 공통 회로
def choose_common_circuit(results):
    """그룹 전체에서 가장 잘 지지되는 회로 순위 (group.consensus 와 같은 기준)"""
    return consensus(results)


# ---------------------------------------------------------------- 2) 순차 재피팅
def _fit_item(item, circuit, p0, meta, n_starts):
    r = item["result"]
    f = np.array(r["freq"])[np.array(r["mask"])]
    Z = (np.array(r["Zre"]) + 1j * np.array(r["Zim"]))[np.array(r["mask"])]
    peaks = r["drt"].get("peaks", []) if r.get("drt") else []
    noise = r["kk"].get("noise_pct") if r.get("kk") else None
    fit = fit_circuit(circuit, f, Z, p0=p0, drt_peaks=peaks, n_starts=n_starts, kk_noise_pct=noise)
    if not fit.get("ok"):
        return fit
    fit["flags"] = diagnose(fit, f)
    weak = [q for q in r["drt"].get("peaks_all", []) if not q["edge"]] if r.get("drt") else []
    more, fit["drt_supported"] = drt_crosscheck(fit, peaks, weak)
    fit["flags"] += more
    fit["valid"] = not any(x["severe"] for x in fit["flags"])
    fit["display_names"], fit["arc_labels"] = display_names(fit, meta)
    fit["arc_curves"] = arc_curves(fit, f)
    return fit


def refit_series(items, circuit, meta, n_starts=8):
    """
    items: [{id, x, result}] (x 로 정렬됨). 반환: 같은 순서의 fit 목록
    기준 파일 = 이 회로의 단독 피팅이 가장 좋은 파일 (유효 · 잔차 최소)
    """
    own = []
    for it in items:
        c = next((c for c in it["result"]["candidates"] if c.get("ok") and c["circuit"] == circuit), None)
        own.append(c)
    order_key = [(0 if (c and c["valid"]) else 1, c["rms_pct"] if c else np.inf) for c in own]
    start = int(np.argmin([k[0] * 1e9 + k[1] for k in order_key]))

    fits = [None] * len(items)
    fits[start] = _fit_item(items[start], circuit, own[start]["values"] if own[start] else None, meta, n_starts)
    for direction in (1, -1):
        prev = start
        i = start + direction
        while 0 <= i < len(items):
            p_prev = fits[prev]["values"] if fits[prev] and fits[prev].get("ok") else None
            # 두 출발점(이웃 파일의 결과 / 이 파일의 단독 피팅)에서 모두 피팅해
            # 유효 → DRT 일치 → χ² 순으로 고름. χ² 차이가 5% 안이면 이웃 쪽(추세 연속성)을 우선
            tries = []
            seeded = _fit_item(items[i], circuit, p_prev, meta, n_starts)
            if seeded.get("ok"):
                seeded["seed"] = "이웃 파일"
                tries.append(seeded)
            if own[i] is not None:
                alt = _fit_item(items[i], circuit, own[i]["values"], meta, n_starts)
                if alt.get("ok"):
                    alt["seed"] = "단독 피팅"
                    tries.append(alt)
            if tries:
                best_chi2 = min(t["chi2"] for t in tries)
                fits[i] = min(tries, key=lambda t: (not t["valid"], not t.get("drt_supported", True),
                                                    t["chi2"] > 1.05 * best_chi2, t["seed"] != "이웃 파일"))
            else:
                fits[i] = seeded
            prev = i
            i += direction
    if fits[start] and fits[start].get("ok"):
        fits[start]["seed"] = "기준 파일"
    return fits, start


# ---------------------------------------------------------------- 3) 추세 이탈
def _transform(var, x, y):
    """온도면 Arrhenius 좌표 (1000/T, ln(1/R)), 그 밖에는 (x, ln R)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if var == "T_C":
        return 1000.0 / (x + 273.15), np.log(1.0 / y)
    return x, np.log(y)


def trend_outliers(var, x, y, use=None, min_dev=np.log(1.15)):
    """
    추세 이탈 점 찾기 (한 번에 하나씩, 가장 크게 벗어난 점부터)
      남은 점들로 한 점씩 빼고 직선을 맞춰 그 점을 예측 → 편차가 (다른 점 편차 중앙값 × 3)과 15% 중 큰 값보다 크면
      이탈로 표시하고 빼낸 뒤 다시 반복. 크게 틀린 점 하나가 다른 점까지 이탈로 보이게 만드는 것을 막음.
    use: 처음부터 추세 계산에서 뺄 점 (피팅 탈락 등) = False
    점이 4개 미만이면 판단하지 않음
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    keep = np.isfinite(y) & (y > 0) & np.isfinite(x)
    if use is not None:
        keep &= np.asarray(use, bool)
    flags, dev = [False] * n, [None] * n
    while keep.sum() >= 4:
        idx = np.where(keep)[0]
        X, Y = _transform(var, x[idx], y[idx])
        # 온도(Arrhenius)는 직선, 그 밖의 변수(사이클·SOC …)는 점이 6개 이상이면 휘는 추세(2차)도 허용
        deg = 1 if (var == "T_C" or len(X) < 6) else 2
        loo = []
        for j in range(len(X)):
            m = np.arange(len(X)) != j
            coef = np.polyfit(X[m], Y[m], deg)
            loo.append(Y[j] - np.polyval(coef, X[j]))
        loo = np.abs(np.array(loo))
        for j, i in enumerate(idx):
            dev[i] = float(np.exp(loo[j]) - 1)               # 예측 대비 상대 편차
        j = int(np.argmax(loo))
        thr = max(3 * np.median(np.delete(loo, j)), min_dev)
        if loo[j] <= thr:
            break
        flags[idx[j]] = True
        keep[idx[j]] = False
    return flags, dev


# ---------------------------------------------------------------- 4) Arrhenius
def arrhenius(T_C, R, R_err=None):
    """ln(1/R) = ln A − Ea/(R_gas T).  반환: Ea ± 오차 (kJ/mol), R², 점 수"""
    T_C, R = np.asarray(T_C, float), np.asarray(R, float)
    ok = np.isfinite(T_C) & np.isfinite(R) & (R > 0)
    if ok.sum() < 2:
        return None
    X, Y = 1000.0 / (T_C[ok] + 273.15), np.log(1.0 / R[ok])
    if ok.sum() == 2:
        a = (Y[1] - Y[0]) / (X[1] - X[0])
        return {"Ea": float(-a * R_GAS * 1000), "Ea_err": None, "r2": 1.0, "n": 2,
                "slope": float(a), "intercept": float(Y[0] - a * X[0])}
    (a, b), cov = np.polyfit(X, Y, 1, cov=True)
    pred = a * X + b
    r2 = 1 - np.sum((Y - pred) ** 2) / max(np.sum((Y - Y.mean()) ** 2), 1e-30)
    return {"Ea": float(-a * R_GAS * 1000), "Ea_err": float(np.sqrt(cov[0, 0]) * R_GAS * 1000),
            "r2": float(r2), "n": int(ok.sum()), "slope": float(a), "intercept": float(b)}


# ---------------------------------------------------------------- 회로와 무관한 값 (측정마다 자기 최적 회로)
OWN_PARAMS = ("R_total", "R_p (분극 합)", "R_s (직렬)")


def own_params(items):
    """
    조성·회로가 달라도 비교할 수 있는 값: 각 측정의 추천(최적) 회로에서
      R_s = 직렬 저항(R0), R_p = 반원 R 합 + Gerischer · TLM 직류 저항, R_total = R_s + R_p
    공통 회로가 어떤 측정에서 안 맞아도 빈칸이 생기지 않음
    """
    vals = {k: [] for k in OWN_PARAMS}
    ok = []
    for it in items:
        r = it["result"]
        c = next((c for c in r.get("candidates", []) if c.get("ok") and c["circuit"] == r.get("recommended")),
                 next((c for c in r.get("candidates", []) if c.get("ok")), None))
        if c is None:
            for k in OWN_PARAMS:
                vals[k].append(np.nan)
            ok.append(False)
            continue
        v = dict(zip(c["names"], c["values"]))
        Rs = v.get("R0", 0.0)
        Rp = sum(a["R"] for a in c["arcs"]) + sum(x["R"] for x in (c.get("specials") or []))
        vals["R_total"].append(Rs + Rp)
        vals["R_p (분극 합)"].append(Rp if Rp > 0 else np.nan)
        vals["R_s (직렬)"].append(Rs if Rs > 0 else np.nan)
        ok.append(bool(c.get("valid")) and r.get("status") != "데이터 이상")
    return [{"param": k, "y": vals[k], "err": [np.nan] * len(items), "own": True, "use": ok} for k in OWN_PARAMS]


# ---------------------------------------------------------------- 전체
def analyze_group(name, items, var, meta, circuit=None, n_starts=8):
    """
    items: [{id, name, x, result}] — 이미 단독 분석된 파일들
    """
    items = sorted([it for it in items if it.get("x") is not None and it["result"].get("candidates")],
                   key=lambda it: it["x"])
    if not items:
        return {"name": name, "ok": False, "error": f"'{VARIABLES.get(var, (var,))[0]}' 값이 있는 분석된 파일이 없습니다"}
    ranking = choose_common_circuit([it["result"] for it in items])
    if not ranking:
        return {"name": name, "ok": False, "error": "공통으로 피팅된 회로가 없습니다"}
    chosen = circuit or ranking[0]["circuit"]
    fits, start = refit_series(items, chosen, meta, n_starts)

    ok_fits = [f for f in fits if f and f.get("ok")]
    names = ok_fits[0]["display_names"] if ok_fits else []
    rows = []
    for it, f in zip(items, fits):
        row = {"id": it["id"], "name": it["name"], "x": it["x"], "ok": bool(f and f.get("ok"))}
        if row["ok"]:
            row.update({"values": f["values"], "stderr": f["stderr"], "rms_pct": f["rms_pct"],
                        "valid": f["valid"], "drt_supported": f.get("drt_supported"),
                        "flags": [x["msg"] for x in f["flags"]], "seed": f.get("seed"),
                        "reproduce": f["reproduce"], "arcs": f["arcs"], "arc_labels": f["arc_labels"],
                        "Zfit": f["Zfit"], "chi2": f["chi2"], "bic": f["bic"]})
        else:
            row["error"] = f.get("error") if f else "실패"
        rows.append(row)

    # 저항 성분별 추세 + Arrhenius (반원 R 합도 함께: '합만 신뢰' 대비)
    params = []
    if ok_fits:
        units = ok_fits[0]["units"]
        R_idx = [i for i, u in enumerate(units) if u == "Ω" and ok_fits[0]["kinds"][i] == "pos"]
        for i in R_idx:
            params.append((names[i], [r["values"][i] if r["ok"] else np.nan for r in rows],
                           [r["stderr"][i] if r["ok"] else np.nan for r in rows]))
        raw = ok_fits[0]["names"]
        arc_R = [raw.index(a["R_name"]) for a in ok_fits[0]["arcs"]]
        if len(arc_R) >= 2:
            # 겹친 반원은 나눈 값보다 합이 훨씬 정확하므로 합의 추세도 같이 봄
            params.append(("반원 R 합", [sum(r["values"][i] for i in arc_R) if r["ok"] else np.nan for r in rows],
                           [np.nan] * len(rows)))
        if "R0" in raw and arc_R:
            i0 = raw.index("R0")
            params.append(("R_Σ", [r["values"][i0] + sum(r["values"][i] for i in arc_R) if r["ok"] else np.nan
                                   for r in rows], [np.nan] * len(rows)))
    rms = np.array([r["rms_pct"] if r["ok"] else np.inf for r in rows])
    rms_lim = max(3 * np.median(rms[np.isfinite(rms)]) if np.isfinite(rms).any() else 5.0, 5.0)
    for r, q in zip(rows, rms):
        why = None
        if not r["ok"]:
            why = "피팅 실패"
        elif not r["valid"]:
            why = "피팅 탈락"
        elif q > rms_lim:
            why = f"잔차 {q:.1f}% (그룹 기준 {rms_lim:.1f}%)"
        r["excluded"] = why                               # 툴이 제안하는 제외 사유
        r["include"] = why is None                        # 사용자가 바꿀 수 있음
    g = {"name": name, "ok": True, "variable": var, "circuit": chosen, "ranking": ranking[:8],
         "start_index": start, "param_names": names, "rows": rows, "refit": True,
         "series_params": own_params(items) + [{"param": p, "y": ys, "err": es} for p, ys, es in params
                                               if p not in ("R_Σ",)]}
    compute_trends(g)
    return g


def compute_trends(g, include=None):
    """
    추세·이탈·Arrhenius 계산 (다시 피팅하지 않음).
    include: {파일 id: True/False} — 사용자가 '추세에 포함'을 바꾼 값
    """
    rows, var = g["rows"], g["variable"]
    if include:
        for r in rows:
            if r["id"] in include:
                r["include"] = bool(include[r["id"]]) and r["ok"]
    use0 = [r["include"] for r in rows]
    off = {rid for rid, v in (include or {}).items() if not v}       # 사용자가 직접 뺀 측정
    xs = [r["x"] for r in rows]
    trends = []
    for sp in g["series_params"]:
        ys, es = sp["y"], sp["err"]
        # 회로와 무관한 값은 각 측정의 최적 회로 기준으로 포함 여부를 정함 (공통 회로 탈락과 무관)
        use = [u and r["id"] not in off for u, r in zip(sp["use"], rows)] if sp.get("own") else use0
        fl, dev = trend_outliers(var, xs, ys, use)
        t = {"param": sp["param"], "y": ys, "err": es, "outlier": fl, "dev": dev, "used": use, "own": bool(sp.get("own"))}
        if var == "T_C":
            sel = [u and not o for u, o in zip(use, fl)]
            t["arrhenius"] = arrhenius([x for x, k in zip(xs, sel) if k], [y for y, k in zip(ys, sel) if k])
            if t["arrhenius"]:
                t["arrhenius"]["excluded_x"] = [x for x, k in zip(xs, sel) if not k]
        trends.append(t)
    g["trends"] = trends
    return g


# ---------------------------------------------------------------- 독립 피팅 결과로 추세 (다시 피팅하지 않음)
def trends_from_fits(name, items, var, circuit=None, include=None):
    """
    items: [{id, name, x, result}] — 파일마다 따로 분석한 결과
    각 파일은 이미 모든 후보 회로를 독립적으로 피팅해 두었으므로, 그룹 공통 회로(또는 지정 회로)의
    결과를 꺼내 같은 파라미터끼리 비교합니다.  그 회로가 어떤 파일에서 탈락이면 추세에서 기본 제외(이유 표시).
    """
    items = sorted([it for it in items if it.get("x") is not None and it["result"].get("candidates")],
                   key=lambda it: it["x"])
    if len(items) < 2:
        return {"name": name, "ok": False, "error": "값이 있는 분석된 파일이 2개 이상 필요합니다"}
    ranking = consensus([it["result"] for it in items])
    if not ranking:
        return {"name": name, "ok": False, "error": "공통으로 피팅된 회로가 없습니다"}
    chosen = circuit or ranking[0]["circuit"]
    rows, names, units, raw_names, arcs0 = [], None, None, None, None
    for it in items:
        c = next((c for c in it["result"]["candidates"] if c.get("ok") and c["circuit"] == chosen), None)
        ref = it["result"].get("R0_ref") or {}
        row = {"id": it["id"], "name": it["name"], "x": it["x"], "ok": c is not None,
               "recommended": it["result"].get("recommended"), "R_HF": ref.get("R_HF"), "R0_L": ref.get("R0_L")}
        if c is not None:
            names = names or c["display_names"]
            units, raw_names, arcs0 = units or c["units"], raw_names or c["names"], arcs0 or c["arcs"]
            row.update({"values": c["values"], "stderr": c["stderr"], "rms_pct": c["rms_pct"], "valid": c["valid"],
                        "sp_R": sum(x["R"] for x in (c.get("specials") or [])),
                        "drt_supported": c.get("drt_supported"), "flags": [f["msg"] for f in c["flags"]],
                        "Zfit": c["Zfit"], "is_recommended": c["circuit"] == it["result"].get("recommended")})
        else:
            row["error"] = "이 파일에서는 이 회로를 피팅하지 않음"
        rows.append(row)
    if names is None:
        return {"name": name, "ok": False, "error": "공통 회로 결과가 없습니다"}

    for r in rows:
        why = None
        if not r["ok"]:
            why = r["error"]
        elif not r["valid"]:
            why = "이 파일에서 회로 탈락"
        r["excluded"] = why
        r["include"] = why is None
    params = []
    R_idx = [i for i, u in enumerate(units) if u == "Ω"]
    for i in R_idx:
        params.append((names[i], [r["values"][i] if r["ok"] else np.nan for r in rows],
                       [r["stderr"][i] if r["ok"] else np.nan for r in rows]))
    # 분극 저항 = 반원 R 합 + Gerischer · TLM 의 직류 저항
    arc_R = [raw_names.index(a["R_name"]) for a in arcs0]
    pol = [sum(r["values"][i] for i in arc_R) + r["sp_R"] if r["ok"] else np.nan for r in rows]
    n_proc = len(arc_R) + (1 if any(r.get("sp_R") for r in rows if r["ok"]) else 0)
    if n_proc >= 2:
        params.append(("분극 R 합", pol, [np.nan] * len(rows)))
    if "R0" in raw_names and n_proc:
        i0 = raw_names.index("R0")
        params.append(("R_Σ", [r["values"][i0] + q if r["ok"] else np.nan for r, q in zip(rows, pol)],
                       [np.nan] * len(rows)))
    # 회로와 무관한 직렬 저항 기준값 (인덕턴스 영역이 있는 측정만)
    for key, label in (("R_HF", "R_HF (교차점)"), ("R0_L", "R0 (L 보정)")):
        ys = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        if np.sum(np.isfinite(ys)) >= 2:
            params.append((label, ys, [np.nan] * len(rows)))
    g = {"name": name, "ok": True, "variable": var, "circuit": chosen, "ranking": ranking[:8],
         "param_names": names, "rows": rows,
         "series_params": own_params(items) + [{"param": p, "y": ys, "err": es} for p, ys, es in params
                                               if p not in ("분극 R 합", "R_Σ")]}
    return compute_trends(g, include)

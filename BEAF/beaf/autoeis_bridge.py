"""
AutoEIS 연동

AutoEIS(EquivalentCircuits.jl, Julia)의 진화 알고리즘으로 회로 후보를 '탐색'하고,
나온 회로를 impedance.py 표기로 바꿔 BEAF 의 같은 기준(다중 시작점 CNLS · AIC/BIC · DRT 교차 확인 · 물리 판정)으로
다른 후보들과 함께 평가합니다.  (AutoEIS 회로를 그대로 믿지 않고, 비교 가능한 같은 저울에 올림)

AutoEIS 표기:  R1-[P2,R3]-L4     '-' 직렬, [a,b] 병렬, P = CPE (파라미터 P2w = Q, P2n = α)
BEAF 표기:     R0-p(CPE2,R3)-L0  (맨 바깥 직렬의 첫 R → R0, 첫 L → L0 으로 이름만 바꿈)

Julia 는 처음 실행할 때 몇 초~수십 초 걸리므로 전용 프로세스 하나에서 계속 살려 둡니다.
"""

from __future__ import annotations

import re
import time

import numpy as np

from .circuits import Circuit


def is_available():
    try:
        from autoeis import julia_helpers
        return bool(julia_helpers.is_julia_installed()) and bool(julia_helpers.is_backend_installed())
    except Exception:                                     # noqa: BLE001
        return False


# ---------------------------------------------------------------- 표기 변환
def to_beaf(circuitstring: str, params: dict | None = None):
    """AutoEIS 회로 문자열(+파라미터) → (BEAF 회로 문자열, 초기값 list 또는 None)"""
    s = circuitstring.replace(" ", "")
    tokens = re.findall(r"\[|\]|,|-|[A-Za-z]+\d+", s)
    depth, out, rename = 0, [], {}
    used = set(re.findall(r"[A-Za-z]+\d+", s))
    for t in tokens:
        if t == "[":
            depth += 1
            out.append("p(")
        elif t == "]":
            depth -= 1
            out.append(")")
        elif t in (",", "-"):
            out.append(t)
        else:
            kind, num = re.match(r"([A-Za-z]+)(\d+)", t).groups()
            new = ("CPE" + num) if kind == "P" else t
            # 맨 바깥 직렬의 첫 R / L 은 R0 / L0 (화면의 R_bulk, L 이름과 시리즈 R_Σ 계산에 쓰임)
            if depth == 0 and kind in ("R", "L") and (kind + "0") not in rename.values() and (kind + "0") not in used:
                new = kind + "0"
            rename[t] = new
            out.append(new)
    beaf = "".join(out)
    circ = Circuit(beaf)                                  # 해석 안 되면 여기서 예외

    p0 = None
    if params:
        p0 = []
        inv = {v: k for k, v in rename.items()}
        for name in circ.param_names:
            base, _, suf = name.partition("_")
            src = inv.get(base, base)
            if src.startswith("P"):
                key = src + ("w" if suf == "0" else "n")
            else:
                key = src
            v = params.get(key)
            if v is None or not np.isfinite(v):
                p0 = None
                break
            p0.append(float(v))
    return beaf, p0


def from_beaf(circuit: str, names=None, values=None):
    """
    BEAF 회로 → AutoEIS 회로 문자열 (소자 번호를 1부터 다시 매김) + 초기값 dict + 이름 대응표
    반환: (autoeis_str, p0 dict 또는 None, {AutoEIS 파라미터 이름: BEAF 파라미터 이름})
    """
    c = Circuit(circuit)
    k = 0
    elem_map = {}
    for name, kind, _ in c.elements:
        k += 1
        if kind == "CPE":
            elem_map[name] = f"P{k}"
        elif kind in ("R", "L", "C"):
            elem_map[name] = f"{kind}{k}"
        else:
            raise ValueError(f"AutoEIS 는 {kind} 소자를 지원하지 않습니다 (R, L, C, CPE 만 가능)")
    s = re.sub(r"[A-Za-z]+\d*\w*", lambda m: elem_map.get(m.group(0), m.group(0)) if m.group(0) != "p" else "p",
               circuit.replace(" ", ""))
    s = s.replace("p(", "[").replace(")", "]")
    pmap = {}
    for name, kind, start in c.elements:
        a = elem_map[name]
        if kind == "CPE":
            pmap[a + "w"] = c.param_names[start]
            pmap[a + "n"] = c.param_names[start + 1]
        else:
            pmap[a] = c.param_names[start]
    p0 = None
    if names is not None and values is not None:
        d = dict(zip(names, values))
        p0 = {a: float(d[b]) for a, b in pmap.items()}
    return s, p0, pmap


def summarize_posterior(circuit, samples_beaf):
    """
    BEAF 이름으로 바꾼 사후 샘플 → 표본마다 반원을 τ 순서로 정렬(라벨 뒤바뀜 방지) → 중앙값·95% 구간
    """
    c = Circuit(circuit)
    names = c.param_names
    if not all(n in samples_beaf for n in names):
        return None
    X = np.column_stack([np.asarray(samples_beaf[n], float) for n in names])     # (표본, 파라미터)
    arcs = [a for a in c.rq_arcs() if not a["nested"] and a["iA"] is not None]
    if len(arcs) >= 2:
        R = np.column_stack([X[:, a["iR"]] for a in arcs])
        Q = np.column_stack([X[:, a["iQ"]] for a in arcs])
        A = np.column_stack([X[:, a["iA"]] for a in arcs])
        tau = (R * Q) ** (1 / A)
        order = np.argsort(tau, axis=1)
        Y = X.copy()
        for slot, a in enumerate(arcs):
            src = order[:, slot]
            for key in ("iR", "iQ", "iA"):
                cols = np.array([arcs[k][key] for k in range(len(arcs))])
                Y[:, a[key]] = X[np.arange(len(X)), cols[src]]
        X = Y
    def stat(v):
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5))}

    out = {n: stat(X[:, i]) for i, n in enumerate(names)}
    # 합성 값: 겹친 반원은 나눈 값보다 합이 잘 정해지므로 표본별 합의 분포도 계산
    all_arcs = c.rq_arcs()
    if len(all_arcs) >= 2:
        out["반원 R 합"] = stat(sum(X[:, a["iR"]] for a in all_arcs))
    if "R0" in names and all_arcs:
        out["R_Σ"] = stat(X[:, names.index("R0")] + sum(X[:, a["iR"]] for a in all_arcs))
    # 사후 중앙값 기준 반원 꼭대기 주파수 (DRT 와 교차 확인용)
    med = np.median(X, axis=0)
    out["_arcs"] = []
    for a in all_arcs:
        al = med[a["iA"]] if a["iA"] is not None else 1.0
        tau = (med[a["iR"]] * med[a["iQ"]]) ** (1 / al)
        out["_arcs"].append({"R_name": a["R"], "R": float(med[a["iR"]]), "Q": float(med[a["iQ"]]), "alpha": float(al),
                             "tau": float(tau), "f_peak": float(1 / (2 * np.pi * tau))})
    return out


# ---------------------------------------------------------------- 탐색 (별도 프로세스에서 실행)
def search(freq, Z, iters=24, complexity=12, generations=30, population_size=100, seed=1, filter_plausible=True):
    """
    반환: {"ok", "circuits": [{"circuit", "p0", "autoeis"}], "n_raw", "n_plausible", "seconds"}
    """
    t0 = time.time()
    try:
        import autoeis as ae
    except Exception as e:                                # noqa: BLE001
        return {"ok": False, "error": f"AutoEIS 를 불러오지 못했습니다: {e}"}
    try:
        df = ae.core.generate_equivalent_circuits(np.asarray(freq, float), np.asarray(Z, complex),
                                                  iters=iters, complexity=complexity, generations=generations,
                                                  population_size=population_size, terminals="RLP",
                                                  parallel=True, seed=seed)
    except Exception as e:                                # noqa: BLE001
        return {"ok": False, "error": f"AutoEIS 회로 생성 실패: {type(e).__name__}: {e}"}
    n_raw = len(df)
    try:
        dff = ae.core.filter_implausible_circuits(df) if filter_plausible else df
    except Exception:                                     # noqa: BLE001
        dff = df
    out, seen = [], set()
    for _, row in dff.iterrows():
        try:
            beaf, p0 = to_beaf(row["circuitstring"], dict(row["Parameters"]))
        except Exception:                                 # noqa: BLE001
            continue
        if beaf in seen:
            continue
        seen.add(beaf)
        out.append({"circuit": beaf, "p0": p0, "autoeis": row["circuitstring"]})
    # 물리적으로 그럴듯한 회로가 하나도 없으면 걸러진 것도 (참고용으로) 돌려줌
    raw_list = []
    if not out:
        for _, row in df.iterrows():
            try:
                beaf, p0 = to_beaf(row["circuitstring"], dict(row["Parameters"]))
            except Exception:                             # noqa: BLE001
                continue
            if beaf not in seen:
                seen.add(beaf)
                raw_list.append({"circuit": beaf, "p0": p0, "autoeis": row["circuitstring"], "implausible": True})
    return {"ok": True, "circuits": out + raw_list, "n_raw": n_raw, "n_plausible": len(dff),
            "seconds": time.time() - t0, "iters": iters}


def bayes(circuit_autoeis, freq, Z, p0=None, num_warmup=1000, num_samples=500, method="nyquist", refine_p0=False):
    """선택한 회로 하나에 AutoEIS 베이지안 추론 (사후분포 평균·95% 구간). 느림 (수 분)"""
    t0 = time.time()
    try:
        import autoeis as ae
        res = ae.core.perform_bayesian_inference(circuit_autoeis, np.asarray(freq, float), np.asarray(Z, complex),
                                                 p0=p0, num_warmup=num_warmup, num_samples=num_samples,
                                                 method=method, refine_p0=refine_p0 or p0 is None, seed=0,
                                                 progress_bar=False, parallel=False)
        r = res[0]
        samples = r.samples if hasattr(r, "samples") else r.mcmc.get_samples()
        return {"ok": True, "samples": {k: np.asarray(v, float).ravel().tolist() for k, v in samples.items()},
                "seconds": time.time() - t0, "converged": bool(getattr(r, "converged", True))}
    except Exception as e:                                # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}

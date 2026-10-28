"""
엑셀 내보내기

  요약          측정당 한 줄, 논문 표에 바로 쓰는 값 (저항 '값 ± 오차', Ω·cm², 조건, 회로)
  전체 파라미터  모든 파라미터 · 오차 · Q · α · C_eff · τ · f_peak · 품질 지표 · 경고 (숫자)
  추세·Ea       묶음별 조건-파라미터 표 + Arrhenius 활성화에너지
  방법          BEAF 버전, 피팅 방법 · 판정 기준, 측정별 피팅 구간 방식 · 재분석 기록 (재현성 · 논문 Methods 용)
  그림_<측정>   원자료 · 매끈한 피팅 곡선(200점) · 성분별 반원 · DRT 를 열로 나란히 → Origin 에 바로 붙여넣기
"""

from __future__ import annotations

import io
import math
import re
import time

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import __version__, series

HEAD_FILL = PatternFill("solid", fgColor="EEF1F5")
TITLE_FONT = Font(bold=True, size=12)
BOLD = Font(bold=True)
GREY = Font(color="5B6474", size=9)

EC_K = {"R": "R", "C": "C", "L": "L", "CPE": "Q", "W": "W", "Wo": "M", "Ws": "Wd", "La": "La", "G": "G",
        "TLMn": "TLM", "TLMQ": "TLMQ"}


# ---------------------------------------------------------------- 표기
def pm(v, e):
    """값 ± 오차 (오차 유효숫자 2자리, 값은 같은 자리까지)"""
    if v is None or not np.isfinite(v):
        return ""
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.4g}"
    d = max(0, 1 - int(math.floor(math.log10(e))))
    if d > 6 or abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.3e} ± {e:.1e}"
    return f"{v:.{d}f} ± {e:.{d}f}"


def eclab(circuit):
    """impedance.py 표기 → EC-Lab ZFit 표기 (R0-p(R1,CPE1) → R1+Q2/R2)"""
    toks = re.findall(r"p\(|\(|\)|,|-|[A-Za-z]+\d*_?\w*", circuit.replace(" ", ""))
    pos = [0]

    def series_():
        it = [item()]
        while pos[0] < len(toks) and toks[pos[0]] == "-":
            pos[0] += 1
            it.append(item())
        return ("s", it) if len(it) > 1 else it[0]

    def item():
        t = toks[pos[0]]
        pos[0] += 1
        if t == "p(":
            b = [series_()]
            while toks[pos[0]] == ",":
                pos[0] += 1
                b.append(series_())
            pos[0] += 1
            return ("p", b)
        m = re.fullmatch(r"(TLMQ|TLMn|CPE|Wo|Ws|La|R|C|L|W|G)(\w*)", t)
        return ("e", m.group(1))

    try:
        tree = series_()
    except Exception:                                     # noqa: BLE001
        return circuit
    cnt = [0]

    def nxt():
        cnt[0] += 1
        return cnt[0]

    def cap(n):
        return 1 if n[0] == "e" and n[1] in ("CPE", "C") else 0

    def ec(n, k=None, top=False):
        if n[0] == "e":
            return EC_K[n[1]] + str(k if k is not None else nxt())
        if n[0] == "s":
            x = "+".join(ec(c, nxt()) if c[0] == "p" else ec(c, None if top else k) for c in n[1])
            return x if top else f"({x})"
        return "/".join(ec(c, k) for c in sorted(n[1], key=lambda c: -cap(c)))

    if tree[0] == "s":
        return ec(tree, top=True)
    return ec(tree, nxt()) if tree[0] == "p" else ec(tree)


# ---------------------------------------------------------------- 시트 도우미
def _table(ws, rows, start_row=1, title=None, note=None):
    """dict 목록 → 표 (머리글 굵게·배경, 열 너비 자동). 반환: 다음 빈 행"""
    r = start_row
    if title:
        ws.cell(r, 1, title).font = TITLE_FONT
        r += 1
    if note:
        ws.cell(r, 1, note).font = GREY
        r += 1
    if not rows:
        return r + 1
    cols = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    for j, k in enumerate(cols, 1):
        c = ws.cell(r, j, k)
        c.font, c.fill = BOLD, HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, row in enumerate(rows, 1):
        for j, k in enumerate(cols, 1):
            v = row.get(k)
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                v = None
            ws.cell(r + i, j, v)
    return r + len(rows) + 2


def _autowidth(ws, lo=8, hi=42):
    for col in ws.columns:
        n = max((len(str(c.value)) if c.value is not None else 0) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(lo, min(hi, n * 1.15 + 2))


def _sheet_name(name, used):
    sn = "".join(ch for ch in name if ch not in r"[]:*?/\\")[:28] or "sheet"
    base, k = sn, 1
    while sn in used:
        sn = base[:25] + f"_{k}"
        k += 1
    used.add(sn)
    return sn


# ---------------------------------------------------------------- 행 만들기
def _conditions(r, meta_edit):
    fm = {**r.get("file_meta", {}), **(meta_edit or {})}
    labels = fm.get("_labels") or {}
    out = {}
    for k, v in fm.items():
        if k in ("sample", "group", "_labels", "x") or v is None:
            continue
        out[series.VARIABLES[k][0] if k in series.VARIABLES else labels.get(k, k)] = v
    return out


def _r_items(c):
    """저항 성분 (이름, 값, 오차, 고정?): 직렬 R, 반원 R, 그 밖의 Ω 단위 (R_W, R_G, R_ion …)"""
    fixed = set(c.get("fixed") or [])
    return [(n, v, e, nm in fixed) for nm, n, v, e, u in zip(c["names"], c["display_names"], c["values"], c["stderr"], c["units"])
            if u == "Ω"]


def summary_row(r, c, meta_edit, area):
    row = {"측정": r["name"], **_conditions(r, meta_edit), "판정": r["status"]}
    if not c:
        row["비고"] = " / ".join(r.get("reasons", []))
        return row
    row["등가회로 (EC-Lab)"] = eclab(c["circuit"])
    items = _r_items(c)
    for n, v, e, fx in items:
        row[f"{n} (Ω)"] = f"{v:.4g} (x절편 고정)" if fx else pm(v, e)
    # 전체 저항: 직렬 + 반원 + Gerischer · TLM 직류 저항 (오차는 공분산으로)
    R_sum = (dict(zip(c["names"], c["values"])).get("R0", 0) + sum(a["R"] for a in c["arcs"])
             + sum(x["R"] for x in c.get("specials") or []))
    row["R_total (Ω)"] = f"{R_sum:.4g}"
    if area:
        for n, v, e, fx in items:
            row[f"{n} (Ω·cm²)"] = f"{v * area:.4g}" if fx else pm(v * area, e * area if e is not None else None)
        row["R_total (Ω·cm²)"] = f"{R_sum * area:.4g}"
    for a, lab in zip(c["arcs"], c["arc_labels"]):
        row[f"f_peak {lab[2:] if lab.startswith('R_') else lab[1:]} (Hz)"] = float(f"{a['f_peak']:.3g}")
    row["잔차 (%)"] = round(c["rms_pct"], 3)
    warn = [x["msg"] for x in c.get("flags", [])] + [x for x in r.get("reasons", []) if x.startswith("⚠")]
    row["주의"] = " / ".join(dict.fromkeys(warn))
    return row


def full_row(r, c, meta_edit, area):
    row = {"측정": r["name"], "파일": (r.get("file_meta") or {}).get("file"), **_conditions(r, meta_edit),
           "판정": r["status"], "판정 이유": " / ".join(r.get("reasons", []))}
    if not c:
        return row
    m = np.array(r["mask"], bool)
    fr = np.array(r["freq"])[m]
    pol = r.get("fit_policy") or {}
    chosen = next((v for v in pol.get("variants", []) if v.get("chosen")), {})
    row.update({"등가회로": c["circuit"], "EC-Lab 표기": eclab(c["circuit"]), "피팅 기록": c.get("label", "자동 분석"),
                "피팅 구간 (Hz)": f"{fr.max():.4g} → {fr.min():.4g}", "피팅 점 수": int(m.sum()),
                "구간 방식": chosen.get("label"),
                "χ² (모듈러스 가중)": c["chi2"], "잔차 RMS (%)": c["rms_pct"],
                "KK 노이즈 (%)": (r.get("kk") or {}).get("noise_pct"), "잔차/노이즈": c.get("quality_ratio"),
                "AIC": c["aic"], "BIC": c["bic"], "재현 (같은 해/시작점)": f"{c['reproduce'][0]}/{c['reproduce'][1]}",
                "DRT 반원 수": r.get("n_drt"), "DRT 일치": c.get("drt_supported")})
    fixed = set(c.get("fixed") or [])
    for nm, n, v, e, u in zip(c["names"], c["display_names"], c["values"], c["stderr"], c["units"]):
        key = f"{n} ({u})" if u else n
        row[key] = v
        row[f"{n} 오차"] = None if nm in fixed else e
        if nm in fixed:
            row[f"{n} 오차"] = "고정 (x절편)"
        if area and u == "Ω":
            row[f"{n} (Ω·cm²)"] = v * area
    for a, lab in zip(c["arcs"], c["arc_labels"]):
        tag = lab[2:] if lab.startswith("R_") else lab[1:]
        row[f"τ {tag} (s)"] = a["tau"]
        row[f"f_peak {tag} (Hz)"] = a["f_peak"]
        row[f"C_eff {tag} (F)"] = a["C_eff"]
        if area:
            row[f"C_eff {tag} (F/cm²)"] = a["C_eff"] / area
    for x in c.get("specials") or []:
        row[f"{x['name']} 직류 저항 (Ω)"] = x["R"]
        row[f"{x['name']} 특성 주파수 (Hz)"] = x["f_peak"]
    ref = r.get("R0_ref")
    if ref:
        row["R_HF 교차점 실수부 (Ω)"] = ref["R_HF"]
        row["R0 L 보정 (Ω)"] = ref["R0_L"]
    row["경고"] = " / ".join(x["msg"] for x in c.get("flags", []))
    return row


# ---------------------------------------------------------------- 시트
def sheet_trends(ws, trends):
    r = 1
    ea_rows = []
    for gname, g in trends.items():
        if not g.get("ok"):
            continue
        var = series.VARIABLES.get(g["variable"], (g["variable"], ""))
        xlab = f"{var[0]} ({var[1]})" if var[1] else var[0]
        rows = []
        for k, rr in enumerate(g["rows"]):
            d = {xlab: rr["x"], "측정": rr["name"], "추세 포함": "예" if rr.get("include") else "아니오"}
            if rr["ok"]:
                for n, v, e in zip(g["param_names"], rr["values"], rr["stderr"]):
                    d[n], d[f"{n} 오차"] = v, e
            for t in g["trends"]:
                if t["outlier"][k]:
                    d[f"{t['param']} 추세 이탈"] = f"{t['dev'][k] * 100:+.0f}%"
            d["제외 이유"] = rr.get("excluded") or ""
            rows.append(d)
        r = _table(ws, rows, r, f"묶음: {gname}", f"공통 회로 {g['circuit']} ({eclab(g['circuit'])})")
        for t in g["trends"]:
            a = t.get("arrhenius")
            if a:
                ea_rows.append({"묶음": gname, "성분": t["param"], "Ea (kJ/mol)": a["Ea"], "Ea 오차": a["Ea_err"],
                                "R²": a["r2"], "점 수": a["n"],
                                "식": "ln(1/R) = −Ea/(R_gas·T) + c,  T = 측정 온도 (K)"})
    if ea_rows:
        r = _table(ws, ea_rows, r, "Arrhenius 활성화에너지")
    if r == 1:
        ws.cell(1, 1, "추세 없음: 온도 · 사이클 · SOC 처럼 값이 다른 측정이 2개 이상인 묶음이 없습니다").font = GREY


def sheet_method(ws, items):
    gen = [
        ("BEAF 버전", __version__),
        ("만든 시각", time.strftime("%Y-%m-%d %H:%M")),
        ("피팅 방법", "복소 비선형 최소제곱(CNLS), 모듈러스 가중 (잔차를 |Z| 로 나눔), 파라미터는 로그 공간, 다중 시작점"),
        ("χ²", "Σ[(Z'fit − Z')² + (Z''fit − Z'')²] / |Z|²  (EC-Lab 의 χ²/|Z| 가중과 같은 방식)"),
        ("잔차 RMS (%)", "√(χ² / 2N) × 100  — 피팅이 측정점에서 평균적으로 |Z| 의 몇 % 벗어나는지"),
        ("KK 노이즈 (%)", "Kramers–Kronig 검정(lin-KK, pyimpspec) 잔차 RMS = 데이터 자체의 잡음 수준"),
        ("통과 기준", "물리적 문제 없음 + 잔차 ≤ max(3 × KK 노이즈, 2%) + 시작점 절반 이상이 같은 해"),
        ("반원 수", "DRT (Tikhonov, λ = 1e-4 · 1e-3 · 1e-2) 에서 세 λ 모두 나타나는 피크 수 ± 1 후보 비교. "
                    "반원 구간 저주파 끝 0.5 decade 안의 피크는 Warburg 몫으로 보고 세지 않음"),
        ("회로 고르기", "유효 → DRT 피크 위치 일치 → BIC 순. 같이 분석한 측정은 공통 회로를 우선 (ΔBIC 10 이내)"),
        ("피팅 구간 (연구실 방식)", "끝점 = Warburg 로 꺾이는 최저점 바로 앞 점. 시작점 = 인덕티브 첫 점 포함(+L) 과 "
                                  "x절편 고정(R0 = R_HF) 두 가지로 피팅해 공통 구간 BIC 가 낮은 쪽"),
        ("반원 순서", "τ = (R·Q)^(1/α) 작은 것(고주파)부터 R1, R2 … (셀 정보가 있으면 R_SEI, R_ct …)"),
        ("C_eff", "Brug 식  C_eff = (Q · R^(1−α))^(1/α)"),
    ]
    r = _table(ws, [{"항목": a, "내용": b} for a, b in gen], 1, "분석 방법",
               "논문 Methods · 재현용. 아래는 측정별 피팅 구간 방식과 재분석 기록")
    rows = []
    for res, c in items:
        pol = res.get("fit_policy") or {}
        for v in pol.get("variants", []):
            rows.append({"측정": res["name"], "구간 방식": v.get("label"), "채택": "✔" if v.get("chosen") else "",
                         "점 수": v.get("n_points"), "회로": v.get("circuit"),
                         "공통 구간 잔차 (%)": v.get("rms_pct_common"), "공통 구간 BIC": v.get("bic_common")})
    r = _table(ws, rows, r, "측정별 피팅 구간")
    hrows = []
    for res, c in items:
        for circ, hist in (res.get("history") or {}).items():
            use = (res.get("history_using") or {}).get(circ)
            for h in hist:
                hrows.append({"측정": res["name"], "회로": circ, "기록": h.get("label"), "사용": "✔" if h.get("round") == use else "",
                              "χ²": h.get("chi2"), "잔차 (%)": h.get("rms_pct"),
                              **{f"{n}": v for n, v, u in zip(h["display_names"], h["values"], h["units"]) if u == "Ω"}})
    if hrows:
        _table(ws, hrows, r, "재분석 (Minimize) 기록")


def sheet_figure(ws, r, c):
    """원자료 · 피팅 곡선 · 성분별 반원 · DRT 를 블록으로 나란히 (블록 사이 빈 열 하나)"""
    blocks = []
    f, re_, im = np.array(r["freq"]), np.array(r["Zre"]), np.array(r["Zim"])
    m = np.array(r["mask"], bool)
    blocks.append(("측정 데이터", ["freq (Hz)", "Z' (Ω)", "-Z'' (Ω)", "|Z| (Ω)", "-phase (°)", "피팅 사용 (1/0)"],
                   [f, re_, -im, np.hypot(re_, im), -np.degrees(np.arctan2(im, re_)), m.astype(int)]))
    if c:
        ln = c["line"]
        lf, lr, li = np.array(ln["freq"]), np.array(ln["re"]), np.array(ln["im"])
        blocks.append((f"피팅 곡선 ({c['circuit']})", ["freq (Hz)", "Z' fit (Ω)", "-Z'' fit (Ω)", "|Z| fit (Ω)", "-phase fit (°)"],
                       [lf, lr, -li, np.hypot(lr, li), -np.degrees(np.arctan2(li, lr))]))
        for a, lab in zip(c.get("arc_curves") or [], c["arc_labels"]):
            blocks.append((f"반원 {lab} 단독 (앞 성분만큼 옮김)", ["Z' (Ω)", "-Z'' (Ω)"],
                           [np.array(a["re"]), -np.array(a["im"])]))
    d = r.get("drt") or {}
    if d.get("ok"):
        tau = np.array(d["tau"])
        blocks.append((f"DRT (λ = {d['lambda']:g})", ["freq (Hz)", "τ (s)", "γ (Ω)"],
                       [1 / (2 * np.pi * tau), tau, np.array(d["gamma"])]))
    col = 1
    for title, heads, cols in blocks:
        ws.cell(1, col, title).font = BOLD
        for j, h in enumerate(heads):
            cell = ws.cell(2, col + j, h)
            cell.font, cell.fill = BOLD, HEAD_FILL
            for i, v in enumerate(cols[j]):
                v = float(v) if isinstance(v, (np.floating, float)) else int(v)
                ws.cell(3 + i, col + j, v if (not isinstance(v, float) or np.isfinite(v)) else None)
            ws.column_dimensions[get_column_letter(col + j)].width = 13
        col += len(heads) + 1


# ---------------------------------------------------------------- 전체
def build(items, results, trends, area=None):
    """items: [{id, circuit, meta_edit}] → xlsx bytes"""
    picked = []
    for it in items:
        r = results.get(it["id"])
        if not r:
            continue
        c = next((x for x in r["candidates"] if x.get("ok") and x["circuit"] == it.get("circuit")),
                 next((x for x in r["candidates"] if x.get("ok")), None))
        picked.append((r, c, it.get("meta_edit")))
    if not picked:
        raise ValueError("내보낼 분석 결과가 없습니다")
    area = area if area and area > 0 else None

    wb = Workbook()
    ws = wb.active
    ws.title = "요약"
    note = ("저항 = 값 ± 표준오차 (Ω" + (f", 면적 {area:g} cm² 로 Ω·cm² 도 함께" if area else "") +
            "). 숫자 원본 · Q · α · C_eff · τ 는 '전체 파라미터', 그림 데이터는 '그림_' 시트")
    _table(ws, [summary_row(r, c, m, area) for r, c, m in picked], 1, "EIS 피팅 결과 요약", note)
    _autowidth(ws)
    ws.freeze_panes = "B4"

    ws = wb.create_sheet("전체 파라미터")
    _table(ws, [full_row(r, c, m, area) for r, c, m in picked], 1)
    _autowidth(ws, hi=30)
    ws.freeze_panes = "B2"

    ws = wb.create_sheet("추세·Ea")
    sheet_trends(ws, trends)
    _autowidth(ws, hi=30)

    ws = wb.create_sheet("방법")
    sheet_method(ws, [(r, c) for r, c, _ in picked])
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 60

    used = {"요약", "전체 파라미터", "추세·Ea", "방법"}
    # 시트 이름은 31자 제한 → 측정 이름들의 공통 앞부분을 떼어 구별되는 부분(예: 10oC_C03)만 씀
    stems = [re.sub(r"\.\w{2,4}$", "", r["name"]) for r, _, _ in picked]
    pre = 0
    if len(stems) > 1:
        while all(len(x) > pre and x[pre] == stems[0][pre] for x in stems):
            pre += 1
        pre = max(0, stems[0].rfind("_", 0, pre + 1) + 1) if pre else 0
    for (r, c, _), stem in zip(picked, stems):
        short = stem[pre:] or stem
        sheet_figure(wb.create_sheet(_sheet_name("그림_" + short[-28:], used)), r, c)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

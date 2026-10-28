"""
BEAF 로컬 웹 서버 (FastAPI)

  python -m beaf.server        → http://127.0.0.1:8765 에서 화면이 열립니다
"""

from __future__ import annotations

import math
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import group, pipeline, series
from .eis_io import SUPPORTED_ALL, load_file

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PORT = 8765

app = FastAPI(title="BEAF")
app.mount("/static", StaticFiles(directory=WEB), name="static")

FILES: dict[str, dict] = {}       # 측정 id → {name, file, path, meta, cycles: {번호: Spectrum}, default_cycle}
RESULTS: dict[str, dict] = {}     # 측정 id → 마지막 분석 결과
TRENDS: dict[str, dict] = {}      # 그룹 이름 → 마지막 추세 결과
LOCK = threading.Lock()


def clean(o):
    """JSON 으로 보낼 수 없는 nan/inf/numpy 형 정리"""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def get_spec(fid, cycle=None):
    """측정 하나의 스펙트럼 (cycle 지정 없으면 기본 = 마지막 cycle)"""
    f = FILES[fid]
    try:
        cycle = float(cycle)
    except (TypeError, ValueError):
        cycle = f["default_cycle"]
    return f["cycles"].get(cycle) or f["cycles"][f["default_cycle"]]


def points_note(e, sp):
    """'51점' 같은 데이터 점 수가 어디서 왔는지 설명"""
    f = sp.freq
    dec = np.log10(f.max() / f.min()) if f.min() > 0 else 0
    per_dec = (len(f) - 1) / dec if dec > 0 else 0
    lines = [f"주파수 {len(f)}개 = 측정 점 {len(f)}개",
             f"{f.max():.4g} Hz → {f.min():.4g} Hz ({dec:.1f} decade, decade 당 약 {per_dec:.0f}점, 로그 간격)"]
    if e.get("source"):
        lines.append(e["source"])
    if len(e["cycles"]) > 1:
        lines.append(f"분석에는 마지막 반복(cycle {e['default_cycle']:g})을 사용 — 옆 'cycle' 칸에서 바꿀 수 있음")
    return "\n".join(lines)


def _add(name, raw, path=""):
    """파일 하나 → 측정 여러 개일 수 있음. 반환: 측정 요약 목록 (읽기 실패면 오류 1개)"""
    try:
        entries = load_file(raw, name)
    except Exception as e:                                # noqa: BLE001
        return [{"id": None, "name": name, "file": name, "path": path, "ok": False,
                 "error": f"{type(e).__name__}: {e}"}]
    out = []
    for e in entries:
        fid = uuid.uuid4().hex[:10]
        FILES[fid] = {"name": e["name"], "file": name, "path": path, "meta": e["meta"],
                      "cycles": e["cycles"], "default_cycle": e["default_cycle"]}
        sp = e["cycles"][e["default_cycle"]]
        out.append({"id": fid, "name": e["name"], "file": name, "path": path, "ok": True,
                    "points_note": points_note(e, sp),
                    "meta": e["meta"], "cycles": list(e["cycles"]), "cycle": e["default_cycle"], "n": len(sp.freq),
                    "fmax": float(sp.freq.max()), "fmin": float(sp.freq.min()), "format": sp.source_format})
    return out


# ---------------------------------------------------------------- 페이지
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/guide")
def guide():
    return FileResponse(ROOT / "BEAF_설명서.html")


@app.get("/api/options")
def options():
    return {"cell": pipeline.CELL_TYPES, "electrode": pipeline.ELECTRODES,
            "electrolyte": pipeline.ELECTROLYTES, "state": pipeline.STATES, "default": pipeline.DEFAULT_META,
            "variables": {k: {"label": v[0], "unit": v[1]} for k, v in series.VARIABLES.items()}}


# ---------------------------------------------------------------- 파일
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), paths: list[str] = Form(default=[])):
    """여러 파일 한 번에. paths = 화면에서 넣은 폴더 안 상대 경로 (표시 · 정렬용, 없으면 파일명)"""
    out = []
    for k, uf in enumerate(files):
        rel = paths[k] if k < len(paths) else uf.filename
        if Path(uf.filename).suffix.lower() not in SUPPORTED_ALL:
            out.append({"id": None, "name": uf.filename, "path": rel, "ok": False,
                        "error": "지원하지 않는 형식 (" + ", ".join(SUPPORTED_ALL) + ")"})
            continue
        out += _add(Path(uf.filename).name, await uf.read(), rel)
    return clean(out)


class FolderReq(BaseModel):
    path: str
    recursive: bool = True


@app.post("/api/load_folder")
def load_folder(req: FolderReq):
    root = Path(req.path.strip().strip('"'))
    if not root.is_dir():
        raise HTTPException(400, f"폴더가 없습니다: {root}")
    paths = sorted(root.rglob("*") if req.recursive else root.glob("*"))
    # 같은 이름의 .mpt 와 .mpr 이 있으면 .mpt 만 사용
    chosen = {}
    for p in paths:
        if p.suffix.lower() not in SUPPORTED_ALL or not p.is_file() or p.name.startswith("~$"):
            continue
        key = p.with_suffix("")
        if key in chosen and chosen[key].suffix.lower() == ".mpt":
            continue
        chosen[key] = p
    out = []
    for p in chosen.values():
        out += _add(p.name, p.read_bytes(), str(p.relative_to(root)))
    return clean(out)


@app.get("/api/raw/{fid}")
def raw(fid: str):
    """원자료: 파일 안의 모든 cycle (분석 여부·성공 여부와 상관없이 항상 볼 수 있게)"""
    if fid not in FILES:
        raise HTTPException(404, "파일이 없습니다")
    from . import analysis
    f = FILES[fid]

    def auto(sp):
        """연구실 방식 자동 구간 (직접 설정 창의 기본값)"""
        r = analysis.auto_range(sp.freq, sp.Z)
        r["R_HF"], r["f_cross"] = analysis.x_intercept(sp.freq, sp.Z, r["start"])
        return r

    cycles = [{"cycle": c, "freq": sp.freq.tolist(), "Zre": sp.Z.real.tolist(), "Zim": sp.Z.imag.tolist(), "auto": auto(sp)}
              for c, sp in f["cycles"].items()]
    return clean({"name": f["name"], "cycles": cycles, "selected": f["default_cycle"]})


@app.delete("/api/file/{fid}")
def remove(fid: str):
    FILES.pop(fid, None)
    RESULTS.pop(fid, None)
    return {"ok": True}


# ---------------------------------------------------------------- 분석
class AnalyzeReq(BaseModel):
    id: str
    meta: dict = {}
    settings: dict = {}
    cycle: str | float | None = "last"


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    if req.id not in FILES:
        raise HTTPException(404, "파일이 없습니다 (서버를 다시 켰다면 파일을 다시 넣어 주세요)")
    sp = get_spec(req.id, req.cycle)
    res = pipeline.analyze(sp, req.meta, dict(req.settings))
    res["id"] = req.id
    res = clean(res)
    with LOCK:
        RESULTS[req.id] = res
    return res


# ---------------------------------------------------------------- 재분석 (EC-Lab Minimize 처럼, 단 이전 결과를 모두 남김)
class RefineReq(BaseModel):
    id: str
    circuit: str
    n_starts: int = 8


def _hist_summary(r, circuit):
    h = r.get("history", {}).get(circuit, [])
    return {"list": h, "using": r.get("history_using", {}).get(circuit, len(h) - 1)}


@app.post("/api/refine")
def refine(req: RefineReq):
    """지금 쓰고 있는 결과의 값에서 다시 출발해 피팅 (같은 회로 · 같은 구간). 새 결과를 기록에 쌓고 그것을 사용"""
    from .fitting import fit_circuit
    r = RESULTS.get(req.id)
    if not r:
        raise HTTPException(404, "먼저 분석하세요")
    idx = next((i for i, c in enumerate(r["candidates"]) if c.get("ok") and c["circuit"] == req.circuit), None)
    if idx is None:
        raise HTTPException(400, "그 회로의 피팅 결과가 없습니다")
    cur = r["candidates"][idx]
    hist = r.setdefault("history", {}).setdefault(req.circuit, [])
    if not hist:
        hist.append({**cur, "round": 0, "label": "자동 분석", "t": time.time()})
    m = np.array(r["mask"], bool)
    f = np.array(r["freq"])[m]
    Z = (np.array(r["Zre"]) + 1j * np.array(r["Zim"]))[m]
    vals = dict(zip(cur["names"], cur["values"]))
    bounds = {nm: (vals[nm] * (1 - 1e-6), vals[nm] * (1 + 1e-6)) for nm in cur.get("fixed") or [] if nm in vals}
    drt = r.get("drt") or {}
    peaks = drt.get("peaks", []) if drt.get("ok") else []
    weak = [q for q in drt.get("peaks_all", []) if not q["edge"] and not q.get("warburg")] if drt.get("ok") else []
    fit = fit_circuit(req.circuit, f, Z, p0=cur["values"], drt_peaks=peaks, n_starts=max(1, min(req.n_starts, 40)),
                      seed=len(hist), bounds=bounds or None)
    if not fit.get("ok"):
        raise HTTPException(400, fit.get("error", "피팅 실패"))
    fit = pipeline.finish_fit(fit, cur, f, r["meta"], peaks, weak, float(np.max(r["freq"])), Z)
    noise = (r.get("kk") or {}).get("noise_pct")
    fit["quality_ratio"] = fit["rms_pct"] / noise if noise else None
    fit["dBIC"] = (cur.get("dBIC") or 0) + fit["bic"] - cur["bic"]
    fit.update({"round": len(hist), "label": f"재분석 {len(hist)}", "t": time.time(),
                "from_round": r.get("history_using", {}).get(req.circuit, 0), "prev_chi2": cur["chi2"]})
    fit = clean(fit)
    with LOCK:
        hist.append(fit)
        r.setdefault("history_using", {})[req.circuit] = fit["round"]
        r["candidates"][idx] = fit
    return clean(_hist_summary(r, req.circuit))


class UseRoundReq(BaseModel):
    id: str
    circuit: str
    round: int


@app.post("/api/use_round")
def use_round(req: UseRoundReq):
    """기록 중 하나를 이 측정의 결과로 사용 (표 · 추세 · 엑셀에 그 값이 들어감)"""
    r = RESULTS.get(req.id)
    hist = (r or {}).get("history", {}).get(req.circuit)
    if not hist or not 0 <= req.round < len(hist):
        raise HTTPException(404, "기록이 없습니다")
    idx = next(i for i, c in enumerate(r["candidates"]) if c.get("ok") and c["circuit"] == req.circuit)
    with LOCK:
        r["candidates"][idx] = hist[req.round]
        r["history_using"][req.circuit] = req.round
    return clean(_hist_summary(r, req.circuit))


# ---------------------------------------------------------------- 회로 추천 (분석 전, 셀 정보 + DRT 만으로)
class SuggestReq(BaseModel):
    ids: list[str]
    meta: dict = {}
    cycles: dict = {}
    mode: str = "arcs"


@app.post("/api/suggest")
def suggest(req: SuggestReq):
    ids = [i for i in req.ids if i in FILES][:8]          # 최대 8개 측정으로 추천 (시간 절약)
    if not ids:
        raise HTTPException(404, "파일이 없습니다")
    res = pipeline.suggest([get_spec(i, req.cycles.get(i)) for i in ids], req.meta, req.mode)
    res["n_used"], res["n_total"] = len(ids), len(req.ids)
    return clean(res)


# ---------------------------------------------------------------- 추세 (독립 피팅 결과로, 다시 피팅하지 않음)
class TrendsReq(BaseModel):
    variable: str
    groups: list[dict]          # [{name, items:[{id, x}], circuit|None, include{id: bool}|None}]
    unify: bool = False         # True: 모든 묶음을 한 회로로 다시 피팅 (조성끼리 R_SEI · R_ct 비교)
    circuit: str | None = None  # unify 일 때 쓸 회로 (없으면 전체에서 가장 잘 지지되는 회로)


@app.post("/api/trends")
def trends(req: TrendsReq):
    out = []
    TRENDS.clear()
    groups = [(g, [{"id": it["id"], "name": RESULTS[it["id"]]["name"], "x": float(it["x"]), "result": RESULTS[it["id"]]}
                   for it in g["items"] if it["id"] in RESULTS and it.get("x") is not None]) for g in req.groups]
    if req.unify:
        allr = [it["result"] for _, items in groups for it in items]
        # 전체를 한 회로로 비교할 때는 '가장 많은 측정에서 물리적으로 유효한' 회로를 먼저
        #   (DRT 일치를 먼저 보면 2반원 회로가 뽑혀 대부분 조성에서 탈락: 46개 중 유효 7 vs 3반원 33)
        ranking = sorted(group.consensus(allr), key=lambda x: (-x["n_valid"], -x["n_supported"], x["sum_bic"]))
        chosen = req.circuit or (ranking[0]["circuit"] if ranking else None)
        # 조성(묶음)마다 다른 프로세스에서 동시에 다시 피팅 (풀이 없으면 차례로)
        todo = [(g, items) for g, items in groups if chosen and items]
        futs = [pipeline.submit(series.analyze_group, g["name"], items, req.variable,
                                items[0]["result"].get("meta") or {}, chosen) for g, items in todo]
        for (g, items), fu in zip(todo, futs):
            try:
                res = fu.result() if fu is not None else None
            except Exception:                             # noqa: BLE001
                res = None
            if res is None:
                res = series.analyze_group(g["name"], items, req.variable, items[0]["result"].get("meta") or {}, chosen)
            if res.get("ok") and g.get("include"):
                series.compute_trends(res, g.get("include"))
            res["global_ranking"] = ranking[:10]
            res = clean(res)
            TRENDS[g["name"]] = res
            out.append(res)
        return out
    for g, items in groups:
        res = clean(series.trends_from_fits(g["name"], items, req.variable, g.get("circuit"), g.get("include")))
        TRENDS[g["name"]] = res
        out.append(res)
    return out


class GroupReq(BaseModel):
    groups: dict               # {그룹 이름: [파일 id, ...]}
    delta_bic: float = group.DELTA_BIC


@app.post("/api/group_prior")
def group_prior(req: GroupReq):
    """그룹 정보로 각 파일의 추천 회로 순위를 다시 정함 (피팅은 그대로, 순위만)"""
    summaries, touched = [], {}
    for name, ids in req.groups.items():
        rs = [RESULTS[i] for i in ids if i in RESULTS]
        sm = group.apply_group_prior(name, rs, req.delta_bic)
        if sm:
            summaries.append(sm)
        for i in ids:
            if i in RESULTS:
                touched[i] = {"recommended": RESULTS[i]["recommended"], "candidates": RESULTS[i]["candidates"],
                              "reasons": RESULTS[i]["reasons"], "group": RESULTS[i].get("group"),
                              "recommended_own": RESULTS[i].get("recommended_own")}
    return clean({"summaries": summaries, "results": touched})


class UCReq(BaseModel):
    circuit: str
    note: str = ""


@app.post("/api/check_circuit")
def check_circuit(req: UCReq):
    from .circuits import Circuit
    try:
        c = Circuit(req.circuit)
        return {"ok": True, "params": c.param_names}
    except Exception as e:                                # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------- 내보내기
class ExportReq(BaseModel):
    items: list[dict]          # [{id, circuit(채택), meta_edit{T_C, SOC, cycle_no, sample}}]
    area: float | None = None  # cm² (Ω·cm² 환산)


@app.post("/api/export")
def export(req: ExportReq):
    from . import export as xl
    try:
        buf = xl.build(req.items, RESULTS, TRENDS, req.area)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=BEAF_results.xlsx"})


def main():
    import os

    import uvicorn
    if not os.environ.get("BEAF_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

"""
BEAF 로컬 웹 서버 (FastAPI)

  python -m beaf.server        → http://127.0.0.1:8765 에서 화면이 열립니다
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import group, pipeline, series
from .eis_io import SUPPORTED_ALL, load_file

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
USER_CIRCUITS = ROOT / "user_circuits.json"
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
            "variables": {k: {"label": v[0], "unit": v[1]} for k, v in series.VARIABLES.items()},
            "autoeis": _autoeis_ok()}


_AE_AVAILABLE = {"value": None}


def _check_autoeis():
    from . import autoeis_bridge
    _AE_AVAILABLE["value"] = autoeis_bridge.is_available()


def _autoeis_ok():
    if _AE_AVAILABLE["value"] is None:
        return True                                       # 확인 중이면 버튼은 보여 주고, 실행할 때 다시 확인
    return _AE_AVAILABLE["value"]


# ---------------------------------------------------------------- 파일
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    out = []
    for uf in files:
        if Path(uf.filename).suffix.lower() not in SUPPORTED_ALL:
            out.append({"id": None, "name": uf.filename, "ok": False,
                        "error": "지원하지 않는 형식 (" + ", ".join(SUPPORTED_ALL) + ")"})
            continue
        out += _add(Path(uf.filename).name, await uf.read())
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
    f = FILES[fid]
    cycles = [{"cycle": c, "freq": sp.freq.tolist(), "Zre": sp.Z.real.tolist(), "Zim": sp.Z.imag.tolist()}
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
    f = FILES[req.id]
    sp = get_spec(req.id, req.cycle)
    settings = dict(req.settings)
    if f.get("autoeis_custom"):
        settings["custom"] = list(settings.get("custom", [])) + f["autoeis_custom"]
    res = pipeline.analyze(sp, req.meta, settings)
    res["id"] = req.id
    if f.get("autoeis_info"):
        res["autoeis"] = f["autoeis_info"]
    res = clean(res)
    with LOCK:
        RESULTS[req.id] = res
    return res


# ---------------------------------------------------------------- 추세 (독립 피팅 결과로, 다시 피팅하지 않음)
class TrendsReq(BaseModel):
    variable: str
    groups: list[dict]          # [{name, items:[{id, x}], circuit|None, include{id: bool}|None}]


@app.post("/api/trends")
def trends(req: TrendsReq):
    out = []
    TRENDS.clear()
    for g in req.groups:
        items = [{"id": it["id"], "name": RESULTS[it["id"]]["name"], "x": float(it["x"]), "result": RESULTS[it["id"]]}
                 for it in g["items"] if it["id"] in RESULTS and it.get("x") is not None]
        res = clean(series.trends_from_fits(g["name"], items, req.variable, g.get("circuit"), g.get("include")))
        TRENDS[g["name"]] = res
        out.append(res)
    return out


# ---------------------------------------------------------------- AutoEIS (오래 걸리는 작업 → 백그라운드)
JOBS: dict[str, dict] = {}
_AE_POOL = None


def _ae_pool():
    """Julia 를 계속 살려 두는 전용 프로세스 1개"""
    global _AE_POOL
    if _AE_POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        _AE_POOL = ProcessPoolExecutor(max_workers=1)
    return _AE_POOL


def _fit_data(fid, settings, cycle):
    """분석과 같은 피팅 구간의 데이터"""
    from . import analysis
    sp = get_spec(fid, cycle)
    st = {"mode": "arcs", "fmin": None, "fmax": None, "exclude_idx": [], **(settings or {})}
    m = analysis.fit_mask(sp.freq, sp.Z, st["mode"], st["fmin"], st["fmax"], st["exclude_idx"])
    return sp, sp.freq[m], sp.Z[m]


class AutoEISReq(BaseModel):
    id: str
    iters: int = 24
    meta: dict = {}
    settings: dict = {}
    cycle: str | float | None = "last"


@app.post("/api/autoeis")
def autoeis_start(req: AutoEISReq):
    from . import autoeis_bridge
    if req.id not in FILES:
        raise HTTPException(404, "파일이 없습니다")
    if not autoeis_bridge.is_available():
        raise HTTPException(400, "AutoEIS(Julia)가 설치되어 있지 않습니다. 설치.bat 을 다시 실행하세요.")
    sp, f, Z = _fit_data(req.id, req.settings, req.cycle)
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"kind": "search", "id": req.id, "status": "running", "started": time.time(), "iters": req.iters,
                 "msg": f"AutoEIS 회로 탐색 중 (반복 {req.iters}회)"}
    fut = _ae_pool().submit(autoeis_bridge.search, f, Z, req.iters)

    def done(fu):
        job = JOBS[jid]
        try:
            res = fu.result()
        except Exception as e:                            # noqa: BLE001
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if not res.get("ok"):
            job.update(status="error", msg=res.get("error"))
            return
        job["msg"] = f"AutoEIS 후보 {len(res['circuits'])}개를 BEAF 기준으로 평가 중"
        ae_custom = []
        for c in res["circuits"]:
            ae_custom.append({"circuit": c["circuit"], "p0": c["p0"], "source": "AutoEIS",
                           "desc": f"AutoEIS 탐색: {c['autoeis']}" + (" (AutoEIS 필터 탈락)" if c.get("implausible") else "")})
        # 이 파일에 AutoEIS 후보를 기억 → 이후 다시 분석해도 계속 후보에 포함
        FILES[req.id]["autoeis_custom"] = ae_custom
        FILES[req.id]["autoeis_info"] = {"seconds": res["seconds"], "iters": res["iters"], "n_raw": res["n_raw"],
                                         "n_plausible": res["n_plausible"],
                                         "circuits": [c["circuit"] for c in res["circuits"]]}
        try:
            custom = list(req.settings.get("custom", [])) + ae_custom
            r = pipeline.analyze(sp, req.meta, {**req.settings, "custom": custom})
            r["id"] = req.id
            r["autoeis"] = FILES[req.id]["autoeis_info"]
            r = clean(r)
            with LOCK:
                RESULTS[req.id] = r
            job.update(status="done", msg=f"완료: AutoEIS 회로 {len(res['circuits'])}개 추가 ({res['seconds']:.0f} s)")
        except Exception as e:                            # noqa: BLE001
            job.update(status="error", msg=f"평가 실패: {type(e).__name__}: {e}")

    fut.add_done_callback(done)
    return {"job": jid}


class BayesReq(BaseModel):
    id: str
    circuit: str
    settings: dict = {}
    cycle: str | float | None = "last"
    num_warmup: int = 1000
    num_samples: int = 500


@app.post("/api/autoeis_bayes")
def autoeis_bayes(req: BayesReq):
    from . import autoeis_bridge
    r = RESULTS.get(req.id)
    if not r:
        raise HTTPException(404, "먼저 분석하세요")
    cand = next((c for c in r["candidates"] if c.get("ok") and c["circuit"] == req.circuit), None)
    if not cand:
        raise HTTPException(400, "그 회로의 피팅 결과가 없습니다")
    try:
        ae_str, p0, pmap = autoeis_bridge.from_beaf(req.circuit, cand["names"], cand["values"])
    except Exception as e:                                # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    _, f, Z = _fit_data(req.id, r.get("settings"), req.cycle)
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"kind": "bayes", "id": req.id, "circuit": req.circuit, "status": "running", "started": time.time(),
                 "msg": f"AutoEIS 베이지안 추론 중 ({ae_str})"}
    fut = _ae_pool().submit(autoeis_bridge.bayes, ae_str, f, Z, p0, req.num_warmup, req.num_samples)

    def done(fu):
        job = JOBS[jid]
        try:
            res = fu.result()
        except Exception as e:                            # noqa: BLE001
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if not res.get("ok"):
            job.update(status="error", msg=res.get("error"))
            return
        # AutoEIS 이름 → BEAF 이름, 표본마다 τ 순서로 반원 정렬 후 요약
        samples = {pmap[k]: v for k, v in res["samples"].items() if k in pmap}
        post = autoeis_bridge.summarize_posterior(req.circuit, samples)
        if post is None:
            job.update(status="error", msg="사후 표본에 빠진 파라미터가 있습니다")
            return
        # CNLS 해와 비교: CNLS 값이 사후 95% 구간 밖이면 다른 해(모드)로 간 것
        notes = []
        moved = [n for n, v in zip(cand["names"], cand["values"])
                 if n in post and n.startswith("R") and not (post[n]["lo95"] <= v <= post[n]["hi95"])]
        if moved:
            notes.append(f"사후분포가 CNLS 와 다른 해로 이동: {', '.join(moved)} 의 CNLS 값이 95% 구간 밖 "
                         f"→ 이 저항들의 분리는 데이터로 결정되지 않음")
        arcs_post = post.pop("_arcs", [])
        peaks = (r.get("drt") or {}).get("peaks", [])
        weak = [q for q in (r.get("drt") or {}).get("peaks_all", []) if not q["edge"]]
        if arcs_post and peaks:
            fl, sup = pipeline.drt_crosscheck({"arcs": arcs_post}, peaks, weak)
            notes += [f"사후 해: {x['msg']}" for x in fl]
            if sup:
                notes.append("사후 해의 반원도 DRT 피크와 일치")
        for key in ("반원 R 합", "R_Σ"):
            if key in post:
                q = post[key]
                notes.append(f"{key} = {q['median']:.4g} [{q['lo95']:.4g}, {q['hi95']:.4g}] Ω (95%)")
        with LOCK:
            cand["bayes"] = clean({"params": post, "seconds": res["seconds"], "converged": res["converged"],
                                   "autoeis_circuit": ae_str, "notes": notes, "arcs": arcs_post})
        job.update(status="done", msg=f"완료 ({res['seconds']:.0f} s)")

    fut.add_done_callback(done)
    return {"job": jid}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "작업이 없습니다")
    out = {k: v for k, v in j.items()}
    out["elapsed"] = time.time() - j["started"]
    if j["status"] == "done":
        out["result"] = RESULTS.get(j["id"])
    return clean(out)


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


# ---------------------------------------------------------------- 사용자 회로
def _load_uc():
    if USER_CIRCUITS.exists():
        return json.loads(USER_CIRCUITS.read_text(encoding="utf-8"))
    return []


class UCReq(BaseModel):
    circuit: str
    note: str = ""


@app.get("/api/user_circuits")
def get_uc():
    return _load_uc()


@app.post("/api/user_circuits")
def add_uc(req: UCReq):
    from .circuits import Circuit
    try:
        Circuit(req.circuit)
    except Exception as e:                                # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    uc = [x for x in _load_uc() if x["circuit"] != req.circuit]
    uc.append({"circuit": req.circuit, "note": req.note})
    USER_CIRCUITS.write_text(json.dumps(uc, ensure_ascii=False, indent=2), encoding="utf-8")
    return uc


@app.delete("/api/user_circuits")
def del_uc(circuit: str):
    uc = [x for x in _load_uc() if x["circuit"] != circuit]
    USER_CIRCUITS.write_text(json.dumps(uc, ensure_ascii=False, indent=2), encoding="utf-8")
    return uc


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


@app.post("/api/export")
def export(req: ExportReq):
    rows, sheets = [], {}
    for it in req.items:
        r = RESULTS.get(it["id"])
        if not r:
            continue
        cand = next((c for c in r["candidates"] if c.get("ok") and c["circuit"] == it.get("circuit")),
                    next((c for c in r["candidates"] if c.get("ok")), None))
        fm = {**r.get("file_meta", {}), **(it.get("meta_edit") or {})}
        row = {"측정": r["name"], "group": fm.get("group")}
        labels = fm.get("_labels") or {}
        for k, v in fm.items():                           # 파일·표에서 읽은 조건 (온도, SOC, 사이클 …)
            if k not in ("sample", "group", "_labels", "x") and v is not None:
                row[series.VARIABLES[k][0] if k in series.VARIABLES else labels.get(k, k)] = v
        row.update({"status": r["status"], "notes": " / ".join(r["reasons"])})
        if cand:
            row.update({"circuit": cand["circuit"], "chi2": cand["chi2"], "rms (%)": cand["rms_pct"],
                        "KK noise (%)": r["kk"].get("noise_pct"), "AIC": cand["aic"], "BIC": cand["bic"],
                        "재현": f"{cand['reproduce'][0]}/{cand['reproduce'][1]}",
                        "fit range (Hz)": f"{min(np.array(r['freq'])[r['mask']]):.3g} ~ "
                                          f"{max(np.array(r['freq'])[r['mask']]):.3g}"})
            for n, v, e in zip(cand["display_names"], cand["values"], cand["stderr"]):
                row[n] = v
                row[f"{n} err"] = e
            for a, lab in zip(cand["arcs"], cand["arc_labels"]):
                row[f"f_peak {lab[2:]} (Hz)"] = a["f_peak"]
                row[f"C_eff {lab[2:]} (F)"] = a["C_eff"]
            for x in cand.get("specials") or []:                      # Gerischer · TLM
                row[f"{x['name']} 직류 저항 (Ω)"] = x["R"]
                row[f"{x['name']} 특성 주파수 (Hz)"] = x["f_peak"]
            ref = r.get("R0_ref")
            if ref:                                                   # 직렬 저항 기준값 두 가지
                row["R_HF 교차점 실수부 (Ω)"] = ref["R_HF"]
                row["R0 L 보정 (Ω)"] = ref["R0_L"]
                row["L 보정 회로 L (H)"] = ref["L"]
                row["R0 L 보정 비고"] = ref["R0_L_note"]
            flags = [x["msg"] for x in cand["flags"]]
            row["flags"] = " / ".join(flags)
            fr = np.array(r["freq"])[r["mask"]]
            zre = np.array(r["Zre"])[r["mask"]]
            zim = np.array(r["Zim"])[r["mask"]]
            sheets[r["name"]] = pd.DataFrame({"freq (Hz)": fr, "Re(Z) data": zre, "-Im(Z) data": -zim,
                                              "Re(Z) fit": cand["Zfit"][0],
                                              "-Im(Z) fit": [-x for x in cand["Zfit"][1]]})
        rows.append(row)
    if not rows and not TRENDS:
        raise HTTPException(400, "내보낼 분석 결과가 없습니다")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        if rows:
            pd.DataFrame(rows).to_excel(xw, sheet_name="summary", index=False)
        used = {"summary"}
        ea_rows = []
        for gname, g in TRENDS.items():
            if not g.get("ok"):
                continue
            var = series.VARIABLES.get(g["variable"], (g["variable"], ""))
            srows = []
            for k, r in enumerate(g["rows"]):
                d = {(f"{var[0]} ({var[1]})" if var[1] else var[0]): r["x"], "측정": r["name"],
                     "공통 회로": g["circuit"], "파일 추천 회로": r.get("recommended"), "rms (%)": r.get("rms_pct"),
                     "추세 포함": r.get("include"),
                     "툴 제외 사유": r.get("excluded"), "flags": " / ".join(r.get("flags", []))}
                if r["ok"]:
                    for n, v, e in zip(g["param_names"], r["values"], r["stderr"]):
                        d[n], d[f"{n} err"] = v, e
                for t in g["trends"]:
                    if t["outlier"][k]:
                        d[f"{t['param']} 추세 이탈"] = f"{t['dev'][k] * 100:+.0f}%"
                srows.append(d)
            sn = ("S_" + "".join(ch for ch in gname if ch not in "[]:*?/\\"))[:28]
            used.add(sn)
            pd.DataFrame(srows).to_excel(xw, sheet_name=sn, index=False)
            for t in g["trends"]:
                a = t.get("arrhenius")
                if a:
                    ea_rows.append({"group": gname, "parameter": t["param"], "Ea (kJ/mol)": a["Ea"],
                                    "Ea err": a["Ea_err"], "R²": a["r2"], "points": a["n"]})
        if ea_rows:
            pd.DataFrame(ea_rows).to_excel(xw, sheet_name="Arrhenius", index=False)
            used.add("Arrhenius")
        for name, df in sheets.items():
            sn = "".join(ch for ch in Path(name).stem if ch not in r'[]:*?/\\')[:28]
            k = 1
            while sn in used:
                sn = sn[:25] + f"_{k}"
                k += 1
            used.add(sn)
            df.to_excel(xw, sheet_name=sn, index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=BEAF_results.xlsx"})


def main():
    import os

    import uvicorn
    # AutoEIS 를 불러오는 데 몇 초 걸리므로 서버 시작 때 백그라운드에서 한 번만 확인
    threading.Thread(target=_check_autoeis, daemon=True).start()
    if not os.environ.get("BEAF_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

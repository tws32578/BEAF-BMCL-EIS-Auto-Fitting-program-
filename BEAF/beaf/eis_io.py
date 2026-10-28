"""
EIS 파일 읽기 + 파일명 메타데이터 추출

지원 형식
  - EC-Lab ASCII (.mpt)      : 헤더 줄 수를 읽어 자동 파싱
  - EC-Lab binary (.mpr)     : galvani 라이브러리 사용
  - 텍스트 (.txt / .csv)     : freq, Re(Z), -Im(Z) 열을 이름 또는 순서로 찾음

반환 형식 (Spectrum)
  freq : 주파수 (Hz), 높은 → 낮은 순
  Z    : 복소 임피던스 (Re - j·(-Im)),  impedance.py 규약 (용량성이면 Im < 0)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Spectrum:
    name: str
    freq: np.ndarray
    Z: np.ndarray
    cycles: list = field(default_factory=list)   # 파일 안의 cycle 번호 목록
    cycle: object = None                         # 선택된 cycle
    meta: dict = field(default_factory=dict)     # 파일명에서 읽은 온도·SOC 등
    source_format: str = ""


# ---------------------------------------------------------------- 공통 정리
def _to_spectrum(name, df, fmt, cycle="last"):
    """freq / re / mim (/ cycle) 열을 가진 DataFrame → Spectrum"""
    cycles = []
    if "cycle" in df.columns and df["cycle"].notna().any():
        cycles = sorted(pd.unique(df["cycle"].dropna()).tolist())
        if len(cycles) > 1 and cycle is not None:
            pick = cycles[-1] if cycle == "last" else float(cycle)
            df = df[df["cycle"] == pick]
        else:
            pick = cycles[0] if cycles else None
    else:
        pick = None

    df = df[(df["freq"] > 0) & np.isfinite(df["re"]) & np.isfinite(df["mim"])]
    df = df.sort_values("freq", ascending=False)
    # 같은 주파수가 중복되면 첫 번째만 사용
    df = df.drop_duplicates(subset="freq", keep="first")
    freq = df["freq"].to_numpy(float)
    Z = df["re"].to_numpy(float) - 1j * df["mim"].to_numpy(float)
    return Spectrum(name=name, freq=freq, Z=Z, cycles=[float(c) for c in cycles],
                    cycle=None if pick is None else float(pick),
                    meta=parse_filename_meta(name), source_format=fmt)


# ---------------------------------------------------------------- .mpt
def read_mpt(raw: bytes, name: str, cycle="last") -> Spectrum:
    text = raw.decode("latin-1")
    lines = text.splitlines()
    n_header = 1
    if len(lines) > 1 and "Nb header lines" in lines[1]:
        n_header = int(re.findall(r"\d+", lines[1])[0])
    body = "\n".join(lines[n_header - 1:])
    df = pd.read_csv(io.StringIO(body), sep="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "freq": pd.to_numeric(df["freq/Hz"], errors="coerce"),
        "re": pd.to_numeric(df["Re(Z)/Ohm"], errors="coerce"),
        "mim": pd.to_numeric(df["-Im(Z)/Ohm"], errors="coerce"),
    })
    if "cycle number" in df.columns:
        out["cycle"] = pd.to_numeric(df["cycle number"], errors="coerce")
    return _to_spectrum(name, out, "mpt", cycle)


# ---------------------------------------------------------------- .mpr
def read_mpr(raw: bytes, name: str, cycle="last") -> Spectrum:
    from galvani import BioLogic

    mpr = BioLogic.MPRfile(io.BytesIO(raw))
    df = pd.DataFrame(mpr.data)
    cols = {c.lower(): c for c in df.columns}

    def col(*cands):
        for c in cands:
            if c.lower() in cols:
                return df[cols[c.lower()]]
        raise KeyError(f".mpr 파일에서 {cands[0]} 열을 찾지 못했습니다 (EIS 파일이 맞는지 확인)")

    out = pd.DataFrame({
        "freq": col("freq/Hz").astype(float),
        "re": col("Re(Z)/Ohm").astype(float),
        "mim": col("-Im(Z)/Ohm").astype(float),
    })
    try:
        out["cycle"] = col("cycle number").astype(float)
    except KeyError:
        pass
    return _to_spectrum(name, out, "mpr", cycle)


# ---------------------------------------------------------------- .txt / .csv
def read_table(raw: bytes, name: str, cycle="last") -> Spectrum:
    text = raw.decode("utf-8", errors="replace")
    if text.startswith("EC-Lab ASCII FILE"):
        return read_mpt(raw, name, cycle)
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    low = {c.lower(): c for c in df.columns}

    def find(keys):
        for k, c in low.items():
            if all(s in k for s in keys):
                return c
        return None

    fcol = find(["freq"])
    rcol = find(["re"]) or find(["z'"])
    icol = find(["-im"]) or find(["im"]) or find(["z''"])
    if fcol is None or rcol is None or icol is None:
        # 열 이름을 못 찾으면 앞의 세 열 = freq, Re, -Im 로 가정
        fcol, rcol, icol = df.columns[:3]
    mim = pd.to_numeric(df[icol], errors="coerce")
    if not str(icol).strip().startswith("-"):
        # Im(Z) 그대로 저장된 경우: 대부분 음수(용량성)이면 -Im 로 뒤집음
        if np.nanmedian(mim) < 0:
            mim = -mim
    out = pd.DataFrame({"freq": pd.to_numeric(df[fcol], errors="coerce"),
                        "re": pd.to_numeric(df[rcol], errors="coerce"), "mim": mim})
    ccol = find(["cycle"])
    if ccol is not None:
        out["cycle"] = pd.to_numeric(df[ccol], errors="coerce")
    return _to_spectrum(name, out, Path(name).suffix.lower().lstrip("."), cycle)


def read_any(raw: bytes, name: str, cycle="last") -> Spectrum:
    ext = Path(name).suffix.lower()
    if ext == ".mpt":
        return read_mpt(raw, name, cycle)
    if ext == ".mpr":
        return read_mpr(raw, name, cycle)
    return read_table(raw, name, cycle)


SUPPORTED = (".mpt", ".mpr", ".txt", ".csv")


# ---------------------------------------------------------------- 파일명 → 메타데이터
def parse_filename_meta(name: str) -> dict:
    """
    파일명에서 온도·SOC·사이클·샘플명을 추정합니다. 틀린 값은 화면에서 고칩니다.
      '..._25oC_...' / '25 oC' / '25C' / '25degC' → 온도
      'after pre_25_C01' 처럼 숫자만 있는 경우도 10~80 범위면 온도로 추정
      'SOC50' / '50%SOC' / 'SOC_50'           → SOC
      '100cyc' / 'cycle100' / '100th'         → 사이클
    """
    stem = Path(name).stem
    meta = {"sample": re.split(r"[_ ]", stem)[0] if stem else "",
            "T_C": None, "SOC": None, "cycle_no": None}

    m = (re.search(r"(-?\d+(?:\.\d+)?)\s*(?:o|°|deg)\s*C(?![a-z])", stem, re.I)
         or re.search(r"[_\s](-?\d+(?:\.\d+)?)C(?:[_\s]|$)", stem))
    if m:
        meta["T_C"] = float(m.group(1))
    else:
        m = re.search(r"_(\d{1,2})_C\d+$", stem)        # after pre_25_C01
        if m and 0 <= int(m.group(1)) <= 80:
            meta["T_C"] = float(m.group(1))

    m = (re.search(r"SOC\s*[_-]?\s*(\d+(?:\.\d+)?)", stem, re.I)
         or re.search(r"(\d+(?:\.\d+)?)\s*%\s*SOC", stem, re.I))
    if m:
        meta["SOC"] = float(m.group(1))

    m = (re.search(r"(\d+)\s*(?:cyc|cycle|cy)(?![a-z])", stem, re.I)
         or re.search(r"cycle\s*[_-]?\s*(\d+)", stem, re.I)
         or re.search(r"(\d+)(?:st|nd|rd|th)(?![a-z])", stem, re.I))
    if m:
        meta["cycle_no"] = int(m.group(1))

    # 그 밖의 측정 조건 (단위가 붙은 숫자). 파일명은 틀릴 수 있으니 화면에서 확인·수정
    m = re.search(r"(\d+(?:\.\d+)?)\s*(MPa|kPa|bar|psi)(?![a-z])", stem, re.I)
    if m:
        v, u = float(m.group(1)), m.group(2).lower()
        meta["P"] = v * {"mpa": 1, "kpa": 1e-3, "bar": 0.1, "psi": 0.006894757}[u]
    m = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|day|days|d)(?![a-z])", stem, re.I)
    if m and not re.fullmatch(r"\d+(st|nd|rd|th)", m.group(0), re.I):
        v, u = float(m.group(1)), m.group(2).lower()
        meta["time_h"] = v * (1 / 60 if u == "min" else 24 if u in ("day", "days", "d") else 1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mV|V)(?![a-z])", stem)
    if m:
        meta["E_V"] = float(m.group(1)) * (1e-3 if m.group(2) == "mV" else 1)
    meta["x"] = None
    return meta


# ================================================================ 표 형식 일반 읽기 (.xlsx / .csv / .txt)
# 한 파일 안에 스펙트럼이 여러 개일 수 있음 (예: Mendeley 데이터셋 — 사이클·SOC 별 60개가 한 시트에)
#   - 주파수 · 임피던스 열을 이름으로 찾음 (Re/Im 또는 |Z|/위상)
#   - 스펙트럼 안에서 값이 일정한 열(사이클, SOC, 온도 …)이 바뀌는 곳에서 나눔 → 그 값은 메타데이터
#   - 그런 열 없이 주파수만 다시 시작하면 같은 측정의 반복(cycle)으로 봄
#   - 시트마다 따로 처리

TABLE_EXT = (".xlsx", ".xlsm", ".csv", ".txt", ".tsv", ".dat")

# (열 이름에 들어 있는 말, 메타 키) — 앞에서부터 먼저 맞는 것
_KNOWN_META = [
    (("soc",), "SOC"), (("soh",), "SOH"),
    (("temp", "°c", "degc", "t(c)", "t (c)"), "T_C"),
    (("aging cycle", "cycle no", "cycle#", "cycles", "cycle"), "cycle_no"),
    (("ocv", "voltage", "ewe", "e(v)", "potential"), "E_V"),
    (("pressure", "mpa", "kpa"), "P"), (("time", "hour"), "time_h"),
]


def _norm(c):
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def _is_re(b):
    return any(k in b for k in ("re(z)", "zreal", "zre", "z'", "real", "z_re")) and "''" not in b and "im" not in b


def _find_cols(cols):
    """주파수 / Re / Im 또는 |Z| / 위상 열 찾기"""
    nz = {c: _norm(c).replace(" ", "") for c in cols}

    def pick(test):
        return next((c for c in cols if test(nz[c])), None)

    minus_keys = ("-im", "-z''", '-z"', "-zimg", "-zim", "-zimag")
    im_keys = ("im(z)", "zimg", "zimag", "zim", "z''", 'z"', "imag", "z_im", "img")
    out = {"freq": pick(lambda b: "freq" in b or b in ("f", "f(hz)", "f/hz", "hz"))}
    out["im"] = pick(lambda b: any(k in b for k in minus_keys)) or pick(lambda b: any(k in b for k in im_keys))
    out["im_is_minus"] = bool(out["im"]) and nz[out["im"]].lstrip("(").startswith("-")
    out["re"] = pick(lambda b: _is_re(b) and (out["im"] is None or b != nz[out["im"]]))
    out["mod"] = pick(lambda b: any(k in b for k in ("|z|", "zmod", "z_mod", "modulus", "magnitude", "abs(z)")))
    out["phase"] = pick(lambda b: any(k in b for k in ("phase", "zphz", "phz", "angle", "theta")))
    return out


def _meta_key(col):
    a = _norm(col)
    for keys, k in _KNOWN_META:
        if any(x in a for x in keys):
            return k
    return re.sub(r"[^\w%]+", "_", str(col)).strip("_") or "x"


def _read_tables(raw: bytes, name: str):
    """파일 → [(시트 이름, DataFrame)]  (헤더 줄 자동 찾기)"""
    ext = Path(name).suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
    else:
        text = raw.decode("utf-8", errors="replace").lstrip("﻿")
        sheets = {"": pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None)}
    out = []
    for sn, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        hdr = None                                        # 주파수 같은 이름이 있는 첫 줄 (위 30줄 안)
        for i in range(min(30, len(df))):
            row = [_norm(x).replace(" ", "") for x in df.iloc[i].tolist()]
            if any("freq" in x or x in ("f", "f(hz)", "f/hz") for x in row):
                hdr = i
                break
        if hdr is None:                                   # 헤더가 없으면 앞의 세 열 = freq, Re, -Im
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            df.columns = ["freq", "Re(Z)", "-Im(Z)"] + [f"col{k}" for k in range(3, df.shape[1])]
        else:
            cols = [str(x) if pd.notna(x) else f"col{k}" for k, x in enumerate(df.iloc[hdr].tolist())]
            df = df.iloc[hdr + 1:].copy()
            df.columns = cols
        out.append((str(sn), df.reset_index(drop=True)))
    return out


def _segments(df, fcol, id_cols):
    """스펙트럼 경계: id 열 값이 바뀌거나 주파수가 반대 방향으로 반 decade 이상 튀는 곳 (새 스윕)"""
    f = pd.to_numeric(df[fcol], errors="coerce").to_numpy(float)
    n = len(f)
    if n == 0:
        return []
    with np.errstate(all="ignore"):
        d = np.diff(np.log10(np.clip(f, 1e-12, None)))
    down = (np.nanmedian(d) < 0) if len(d) else True
    starts = [0]
    for i in range(1, n):
        new = any(not (pd.isna(df[c].iloc[i - 1]) and pd.isna(df[c].iloc[i])) and df[c].iloc[i - 1] != df[c].iloc[i]
                  for c in id_cols)
        if not new and f[i] > 0 and f[i - 1] > 0:
            jump = np.log10(f[i] / f[i - 1])
            new = (down and jump > 0.5) or ((not down) and jump < -0.5)
        if new:
            starts.append(i)
    starts.append(n)
    return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1) if starts[k + 1] - starts[k] >= 5]


def _fmt_v(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def load_table(raw: bytes, name: str):
    """표 형식 파일 → 측정 목록 [{name, meta, cycles: {번호: Spectrum}, default_cycle}]"""
    entries = []
    stem = Path(name).stem
    fname_meta = parse_filename_meta(name)
    sheets = _read_tables(raw, name)
    for sn, df in sheets:
        cols = list(df.columns)
        fc = _find_cols(cols)
        if fc["freq"] is None or not ((fc["re"] and fc["im"]) or (fc["mod"] and fc["phase"])):
            raise ValueError(f"[{sn or name}] 주파수와 임피던스 열을 찾지 못했습니다. 열 이름: {cols}")
        imp_cols = {fc[k] for k in ("freq", "re", "im", "mod", "phase") if fc.get(k)}
        # EC-Lab 'cycle number' = 같은 측정의 반복.  그 밖에 스펙트럼 안에서 일정한 열 = 다른 측정(메타데이터)
        rep_col = next((c for c in cols if _norm(c) in ("cycle number", "cycle_number", "cycle no.")), None)
        id_cols = []
        for c in cols:
            if c in imp_cols or c == rep_col or df[c].isna().all():
                continue
            changes = int((df[c] != df[c].shift()).sum()) - 1
            if 0 <= changes <= max(len(df) / 8, 1):      # 값이 드물게 바뀌는 열
                id_cols.append(c)
        segs = _segments(df, fc["freq"], id_cols + ([rep_col] if rep_col else []))

        groups = {}                                       # 메타 값이 같은 구간끼리 = 같은 측정의 반복 cycle
        for a, b in segs:
            part = df.iloc[a:b]
            f = pd.to_numeric(part[fc["freq"]], errors="coerce").to_numpy(float)
            if fc["re"] and fc["im"]:
                re_ = pd.to_numeric(part[fc["re"]], errors="coerce").to_numpy(float)
                im_ = pd.to_numeric(part[fc["im"]], errors="coerce").to_numpy(float)
                mim = im_ if fc["im_is_minus"] else -im_
            else:
                mod = pd.to_numeric(part[fc["mod"]], errors="coerce").to_numpy(float)
                ph = np.deg2rad(pd.to_numeric(part[fc["phase"]], errors="coerce").to_numpy(float))
                re_, mim = mod * np.cos(ph), -mod * np.sin(ph)
            # 부호 확인: 저주파 쪽 절반에서 -Im 이 대부분 음수면 부호가 반대로 저장된 것
            low = mim[np.argsort(-f)][len(f) // 2:]
            if np.isfinite(low).any() and np.nanmedian(low) < 0:
                mim = -mim
            meta, labels = {}, {}
            for c in id_cols:
                k = _meta_key(c)
                v = part[c].iloc[0]
                try:
                    meta[k] = float(v)
                except (TypeError, ValueError):
                    meta[k] = str(v)
                labels[k] = str(c)
            key = (sn,) + tuple(sorted((k, str(v)) for k, v in meta.items()))
            g = groups.setdefault(key, {"meta": meta, "labels": labels, "parts": []})
            g["parts"].append((f, re_, mim))

        for g in groups.values():
            label = ", ".join(f"{g['labels'][k]}={_fmt_v(v)}" for k, v in g["meta"].items())
            nm = stem + (f" · {sn}" if len(sheets) > 1 else "") + (f" [{label}]" if label else "")
            meta = {**fname_meta, **g["meta"], "_labels": g["labels"]}
            cycles = {}
            for k, (f, re_, mim) in enumerate(g["parts"]):
                sp = _to_spectrum(nm, pd.DataFrame({"freq": f, "re": re_, "mim": mim}),
                                  Path(name).suffix.lower().lstrip("."), None)
                sp.meta = meta
                cycles[float(k + 1)] = sp
            n_rows = sum(len(f) for f, _, _ in g["parts"])
            src = (f"{'시트 ' + repr(sn) + ' ' if sn and len(sheets) > 1 else ''}전체 {len(df)}행 중 "
                   + (f"조건({label})이 같은 {n_rows}행" if label else f"{n_rows}행"))
            if len(g["parts"]) > 1:
                src += f" = 같은 조건 반복 측정 {len(g['parts'])}번 (주파수가 다시 시작하는 곳에서 나눔)"
            entries.append({"name": nm, "meta": meta, "cycles": cycles, "default_cycle": max(cycles), "source": src})
    if not entries:
        raise ValueError("읽을 수 있는 스펙트럼이 없습니다")
    return entries


def load_file(raw: bytes, name: str):
    """
    어떤 파일이든 → 측정 목록 [{name, meta, cycles: {번호: Spectrum}, default_cycle}]
      .mpt / .mpr : 측정 1개 (EC-Lab cycle number 별 반복)
      그 밖의 표   : load_table (측정 여러 개일 수 있음)
    """
    ext = Path(name).suffix.lower()
    if ext in (".mpt", ".mpr") or raw[:17] == b"EC-Lab ASCII FILE":
        first = read_any(raw, name, "last")
        cycles = {}
        for c in (first.cycles if len(first.cycles) > 1 else [first.cycle]):
            sp = first if len(first.cycles) <= 1 else read_any(raw, name, c)
            cycles[c if c is not None else 1.0] = sp
        n_tot = sum(len(sp.freq) for sp in cycles.values())
        src = (f"EC-Lab 파일: 같은 측정을 {len(cycles)}번 반복 (cycle number), 합계 {n_tot}행"
               if len(cycles) > 1 else f"EC-Lab 파일: {n_tot}행")
        return [{"name": Path(name).name, "meta": first.meta, "cycles": cycles,
                 "default_cycle": first.cycle if first.cycle is not None else 1.0, "source": src}]
    return load_table(raw, name)


SUPPORTED_ALL = (".mpt", ".mpr") + TABLE_EXT

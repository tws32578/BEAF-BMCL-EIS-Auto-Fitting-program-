"""
등가회로 문자열 파서 + 빠른 계산기

회로 표기는 impedance.py 와 같습니다.
    'R0-p(R1,CPE1)-p(R2,CPE2)'     직렬 '-',  병렬 p(a,b,...)
지원 소자 (파라미터 순서·정의도 impedance.py 와 동일)
    R   [R]                  Z = R
    C   [C]                  Z = 1/(jωC)
    L   [L]                  Z = jωL
    CPE [Q, α]               Z = 1/(Q (jω)^α)          (EC-Lab 의 Q 소자)
    W   [Aw]                 Z = Aw (1-j)/√ω           (반무한 Warburg)
    Wo  [Z0, τ]              Z = Z0 coth(√(jωτ))/√(jωτ) (유한, 반사형 / blocking)
    Ws  [Z0, τ]              Z = Z0 tanh(√(jωτ))/√(jωτ) (유한, 투과형)
    La  [L, α]               Z = L (jω)^α
    G   [R_G, t_G]           Z = R_G / √(1 + jω t_G)    (Gerischer, impedance.py 와 같음)
    TLMQ [R_ion, Q, γ]       Z = √(R_ion Z_s) coth √(R_ion/Z_s),  Z_s = 1/(Q (jω)^γ)
                             (블로킹 다공성 전극 전송선 모델, impedance.py 와 같음)
    TLMn [R_ion, R_ct, Q, α] 위와 같고 Z_s = R_ct ‖ CPE(Q, α)
                             (비블로킹 다공성 전극 = 전하전달이 있는 de Levie 모델. BEAF 확장 —
                              pyimpspec 의 Tlmno 와 같은지 tests/test_circuits.py 에서 확인)

impedance.py 의 eval 기반 계산보다 수십 배 빨라 다중 시작점 피팅에 사용하고,
결과는 tests/test_circuits.py 에서 impedance.py 와 일치하는지 확인합니다.
"""

from __future__ import annotations

import re

import numpy as np

# 소자별: (파라미터 이름 접미사, 단위, 종류)
#   종류 'pos' = 양수(로그 공간에서 피팅), 'alpha' = 0~1 지수
ELEMENTS = {
    "R":   [("", "Ω", "pos")],
    "C":   [("", "F", "pos")],
    "L":   [("", "H", "pos")],
    "CPE": [("_0", "F·s^(α-1)", "pos"), ("_1", "", "alpha")],
    "W":   [("", "Ω·s^-½", "pos")],
    "Wo":  [("_0", "Ω", "pos"), ("_1", "s", "pos")],
    "Ws":  [("_0", "Ω", "pos"), ("_1", "s", "pos")],
    "La":  [("_0", "H", "pos"), ("_1", "", "alpha")],
    "G":   [("_0", "Ω", "pos"), ("_1", "s", "pos")],
    "TLMQ": [("_0", "Ω", "pos"), ("_1", "F·s^(γ-1)", "pos"), ("_2", "", "alpha")],
    "TLMn": [("_0", "Ω", "pos"), ("_1", "Ω", "pos"), ("_2", "F·s^(α-1)", "pos"), ("_3", "", "alpha")],
}


def _tanh(x):
    """복소 tanh. 실수부가 크면 1 로 (np.tanh overflow 방지)"""
    big = x.real > 20
    out = np.ones_like(x)
    out[~big] = np.tanh(x[~big])
    return out


def _elem_impedance(kind, p, w):
    jw = 1j * w
    if kind == "R":
        return np.full_like(jw, p[0])
    if kind == "C":
        return 1 / (jw * p[0])
    if kind == "L":
        return jw * p[0]
    if kind == "CPE":
        return 1 / (p[0] * jw ** p[1])
    if kind == "W":
        return p[0] * (1 - 1j) / np.sqrt(w)
    if kind == "Wo":
        x = np.sqrt(jw * p[1])
        return p[0] / (_tanh(x) * x)
    if kind == "Ws":
        x = np.sqrt(jw * p[1])
        return p[0] * _tanh(x) / x
    if kind == "La":
        return p[0] * jw ** p[1]
    if kind == "G":
        return p[0] / np.sqrt(1 + jw * p[1])
    if kind == "TLMQ":
        Zs = 1 / (p[1] * jw ** p[2])
        return np.sqrt(p[0] * Zs) / _tanh(np.sqrt(p[0] / Zs))
    if kind == "TLMn":
        Zs = 1 / (1 / p[1] + p[2] * jw ** p[3])
        return np.sqrt(p[0] * Zs) / _tanh(np.sqrt(p[0] / Zs))
    raise ValueError(kind)


_TOKEN = re.compile(r"\s*(p\(|\(|\)|,|-|[A-Za-z]+\d*_?\w*)")


class Circuit:
    """
    회로 문자열을 트리로 컴파일합니다.
      c = Circuit('R0-p(R1,CPE1)')
      c.param_names  → ['R0', 'R1', 'CPE1_0', 'CPE1_1']
      c.Z(params, freq)
    """

    def __init__(self, string: str):
        self.string = string.replace(" ", "")
        self.param_names, self.param_units, self.param_kinds = [], [], []
        self.elements = []                      # [(이름, 종류, 파라미터 시작 index)]
        self._pos = 0
        self._tokens = self._tokenize(self.string)
        self.tree = self._parse_series()
        if self._pos != len(self._tokens):
            raise ValueError(f"회로 문자열 해석 실패: '{string}'")

    # -------------------------------------------------- 파서
    @staticmethod
    def _tokenize(s):
        tokens, i = [], 0
        while i < len(s):
            m = _TOKEN.match(s, i)
            if not m:
                raise ValueError(f"회로 문자열에 알 수 없는 문자: '{s[i:]}'")
            tokens.append(m.group(1))
            i = m.end()
        return tokens

    def _peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _take(self):
        t = self._tokens[self._pos]
        self._pos += 1
        return t

    def _parse_series(self):
        items = [self._parse_item()]
        while self._peek() == "-":
            self._take()
            items.append(self._parse_item())
        return ("s", items) if len(items) > 1 else items[0]

    def _parse_item(self):
        t = self._take()
        if t == "p(":
            branches = [self._parse_series()]
            while self._peek() == ",":
                self._take()
                branches.append(self._parse_series())
            if self._take() != ")":
                raise ValueError("괄호 짝이 맞지 않습니다")
            return ("p", branches)
        m = re.fullmatch(r"(TLMQ|TLMn|CPE|Wo|Ws|La|R|C|L|W|G)(\w*)", t)
        if not m:
            raise ValueError(f"지원하지 않는 소자: '{t}' (R, C, L, CPE, W, Wo, Ws, La, G, TLMQ, TLMn)")
        kind = m.group(1)
        start = len(self.param_names)
        for suffix, unit, pk in ELEMENTS[kind]:
            self.param_names.append(t + suffix)
            self.param_units.append(unit)
            self.param_kinds.append(pk)
        self.elements.append((t, kind, start))
        return ("e", kind, start, len(ELEMENTS[kind]), t)

    # -------------------------------------------------- 계산
    def _eval(self, node, p, w):
        if node[0] == "e":
            _, kind, s, n, _ = node
            return _elem_impedance(kind, p[s:s + n], w)
        if node[0] == "s":
            return sum(self._eval(c, p, w) for c in node[1])
        return 1 / sum(1 / self._eval(c, p, w) for c in node[1])

    def Z(self, params, freq):
        w = 2 * np.pi * np.asarray(freq, float)
        return self._eval(self.tree, np.asarray(params, float), w)

    def Z_element(self, name, params, freq):
        """소자 하나만의 임피던스"""
        for nm, kind, s in self.elements:
            if nm == name:
                n = len(ELEMENTS[kind])
                w = 2 * np.pi * np.asarray(freq, float)
                return _elem_impedance(kind, np.asarray(params, float)[s:s + n], w)
        raise KeyError(name)

    @property
    def n_params(self):
        return len(self.param_names)

    # -------------------------------------------------- 구조 정보
    def rq_arcs(self):
        """
        병렬 블록 p(R?, CPE?|C?) 을 '반원'으로 보고 목록을 돌려줍니다.
        반환: [{'R': 'R1', 'Q': 'CPE1', 'iR':.., 'iQ':.., 'iA':.. or None, 'nested': bool}]
        (중첩 회로 p(R1-p(R2,CPE2),CPE1) 의 바깥 블록도 R 이 들어 있으면 반원으로 취급)
        """
        arcs = []

        def first_elem(node, kinds):
            if node[0] == "e" and node[1] in kinds:
                return node
            if node[0] == "s":
                for c in node[1]:
                    if c[0] == "e" and c[1] in kinds:
                        return c
            return None

        def walk(node, depth):
            if node[0] == "p" and len(node[1]) == 2:
                a, b = node[1]
                for x, y in ((a, b), (b, a)):
                    r = first_elem(x, ("R",))
                    q = y if (y[0] == "e" and y[1] in ("CPE", "C")) else None
                    if r is not None and q is not None:
                        arcs.append({"R": r[4], "Q": q[4], "iR": r[2], "iQ": q[2],
                                     "iA": q[2] + 1 if q[1] == "CPE" else None,
                                     "nested": x[0] == "s"})
                        break
            if node[0] in ("s", "p"):
                for c in node[1]:
                    walk(c, depth + 1)

        walk(self.tree, 0)
        return arcs


def special_processes(circ, p):
    """
    R||CPE 반원이 아닌 과정 요소 (Gerischer · 전송선 모델) 의 직류 저항과 특성 주파수
    반환: [{"name", "kind", "R", "f_peak", "tau"}]
      G    : R = R_G,  τ = t_G
      TLMn : R = √(R_ion R_ct) coth √(R_ion/R_ct),  τ = (R_ct Q)^(1/α)  (계면 반응)
      TLMQ : 블로킹 → 직류 저항 없음. 이온 저항 몫 R_ion/3 을 R 로, τ = (R_ion Q)^(1/γ)
    """
    out = []
    p = np.asarray(p, float)
    for name, kind, s in circ.elements:
        if kind == "G":
            R, tau = p[s], p[s + 1]
        elif kind == "TLMn":
            Ri, Rct, Q, a = p[s:s + 4]
            x = np.sqrt(Ri / Rct)
            R, tau = np.sqrt(Ri * Rct) / np.tanh(x), (Rct * Q) ** (1 / a)
        elif kind == "TLMQ":
            Ri, Q, g = p[s:s + 3]
            R, tau = Ri / 3, (Ri * Q) ** (1 / g)
        else:
            continue
        out.append({"name": name, "kind": kind, "R": float(R), "tau": float(tau),
                    "f_peak": float(1 / (2 * np.pi * tau))})
    return out


def arc_tau(R, Q, alpha=1.0):
    """R||CPE 시정수 τ = (R·Q)^(1/α), 반원 꼭대기 주파수 f = 1/(2πτ)"""
    return (R * Q) ** (1.0 / alpha)


def arc_capacitance(R, Q, alpha=1.0):
    """유효 정전용량 C_eff = (Q·R^(1-α))^(1/α)  (Brug 식, R||CPE)"""
    return (Q * R ** (1 - alpha)) ** (1.0 / alpha)

"""beaf.circuits 계산 결과가 impedance.py 와 같은지 확인"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impedance.models.circuits import CustomCircuit  # noqa: E402

from beaf.circuits import Circuit  # noqa: E402

CASES = [
    ("R0-p(R1,CPE1)-p(R2,CPE2)", [1.5, 0.8, 1e-5, 0.85, 2.7, 3e-4, 0.8]),
    ("L0-R0-p(R1,CPE1)-p(R2,CPE2)-Wo1", [2e-7, 1.5, 0.8, 1e-5, 0.85, 2.7, 3e-4, 0.8, 5.0, 3.0]),
    ("R0-p(R1-p(R2,CPE2),CPE1)-CPE3", [1.0, 0.5, 2.0, 1e-3, 0.9, 1e-5, 0.8, 0.1, 0.7]),
    ("R0-p(R1,C1)-W1-Ws1", [1.0, 2.0, 1e-5, 0.3, 4.0, 2.0]),
    ("R0-p(R1,CPE1)-G1", [2.0, 1.0, 1e-5, 0.9, 0.45, 0.05]),
    ("L0-R0-TLMQ1", [1e-7, 1.0, 3.0, 0.01, 0.85]),
]


def test_matches_impedance_py():
    f = np.logspace(6, -2, 60)
    for s, p in CASES:
        ref = CustomCircuit(s, initial_guess=p).predict(f, use_initial=True)
        mine = Circuit(s).Z(p, f)
        assert np.allclose(mine, ref, rtol=1e-9), s


def test_tlmn_matches_pyimpspec():
    """TLMn (BEAF 확장) = pyimpspec Tlmno (비블로킹 다공성 전극, L=1)"""
    import pyimpspec
    f = np.logspace(6, -2, 60)
    Ri, Rct, Q, a = 3.0, 4.0, 0.005, 0.85
    ref = pyimpspec.parse_cdc(f"Tlmno{{R_i={Ri},R_ct={Rct},Y={Q},n={a},L=1}}").get_impedances(f)
    mine = Circuit("TLMn1").Z([Ri, Rct, Q, a], f)
    assert np.allclose(mine, ref, rtol=1e-9), np.max(np.abs(mine / ref - 1))


if __name__ == "__main__":
    test_matches_impedance_py()
    print("OK: beaf.circuits == impedance.py")
    test_tlmn_matches_pyimpspec()
    print("OK: TLMn == pyimpspec Tlmno")

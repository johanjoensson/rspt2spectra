import numpy as np

from rspt2spectra.scripts.build_h0 import run


def _write_rspt_dir(path, w, hyb, h_dft):
    header = "#   Energy      orbitals\n# indexmap\n# 2\n"
    for part, fname in ((hyb.real, "real-hyb-cl.dat"), (hyb.imag, "imag-hyb-cl.dat")):
        body = "\n".join(f"{wi: .10e}  {p: .10e}" for wi, p in zip(w, part))
        (path / fname).write_text(header + body + "\n")
    (path / "out").write_text(f"Cluster cl Local hamiltonian\n real part\n {h_dft:.6f}\n imag part\n  0.000000\n")


def _read_h00(path):
    for line in (path / "cl_h0_op.dict").read_text().splitlines():
        i, j, re, im = line.split()
        if i == "0" and j == "0":
            return float(re) + 1j * float(im)
    raise AssertionError("impurity element not found in operator file")


def test_fitted_constant_offset_shifts_impurity_level(tmp_path, monkeypatch):
    # A constant offset in the hybridization function is static content that
    # is already part of the local Hamiltonian; build_h0 must subtract the
    # fitted offset from the impurity block like a double-counting term.
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    c = 0.35
    e_pole, v = -2.0, 0.7
    h_dft = -1.0
    hyb = v**2 / (w + 1j * eim - e_pole) + c
    _write_rspt_dir(tmp_path, w, hyb, h_dft)

    monkeypatch.chdir(tmp_path)  # the operator file is written to the cwd
    run(
        "cl",
        "star",
        eim,
        2,
        0.01,
        False,  # fit_unocc
        False,  # fit_imag
        str(tmp_path),
        False,  # verbose
        False,  # plot
        "l2",
        "unit",
        2.0,
        0.0,
        False,  # natural_orbitals
        "linear",
    )

    h00 = _read_h00(tmp_path)
    assert abs(h00.imag) < 1e-10
    assert abs(h00.real - (h_dft - c)) < 0.02

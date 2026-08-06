import numpy as np

from rspt2spectra.scripts.build_h0 import run


def _write_rspt_dir(path, w, hyb, h_dft, e_fermi=0.0):
    header = "#   Energy      orbitals\n# indexmap\n# 2\n"
    for part, fname in ((hyb.real, "real-hyb-cl.dat"), (hyb.imag, "imag-hyb-cl.dat")):
        body = "\n".join(f"{wi: .10e}  {p: .10e}" for wi, p in zip(w, part))
        (path / fname).write_text(header + body + "\n")
    # RSPt prints the local Hamiltonian on an absolute scale and the Fermi energy separately;
    # the hybridization mesh above already has E_F = 0.
    (path / "out").write_text(
        f" fermi energy = {e_fermi: .13E}\n"
        f"Cluster cl Local hamiltonian\n real part\n {h_dft:.6f}\n imag part\n  0.000000\n"
    )
    (path / "green.inp").write_text("cluster\n 1 Idcl\n 1 2 1 1 0\n")


def _run_build_h0(tmp_path, eim):
    """Invoke the build_h0 pipeline with the default knobs of these tests."""
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


def _read_h00(path):
    for line in (path / "cl_h0_op.dict").read_text().splitlines():
        i, j, re, im = line.split()
        if i == "0" and j == "0":
            return float(re) + 1j * float(im)
    raise AssertionError("impurity element not found in operator file")


def test_fitted_constant_offset_shifts_impurity_level(tmp_path, monkeypatch):
    # A constant offset in the hybridization function is hybridization content
    # the discrete poles cannot carry; build_h0 must add it to the impurity
    # level (Delta_fit = Delta_pole + C and g0^-1 = z - H_imp - Delta imply
    # E_imp = H_imp + C).
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
    assert abs(h00.real - (h_dft + c)) < 0.02


def test_fermi_energy_is_subtracted_from_the_local_hamiltonian(tmp_path, monkeypatch):
    # RSPt prints the local Hamiltonian on an absolute energy scale, while the hybridization
    # function it is fitted against has E_F = 0. Without this shift the impurity block lands
    # a few eV above the whole bath -- for NiO, +8.2 eV against an O 2p bath at -6.7..-1.9 eV.
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    e_pole, v = -2.0, 0.7
    e_fermi = 0.6
    h_dft = -1.0 + e_fermi  # the same physical level as the test above, on RSPt's scale
    hyb = v**2 / (w + 1j * eim - e_pole)
    _write_rspt_dir(tmp_path, w, hyb, h_dft, e_fermi=e_fermi)

    monkeypatch.chdir(tmp_path)
    _run_build_h0(tmp_path, eim)

    h00 = _read_h00(tmp_path)
    assert abs(h00.real - (-1.0)) < 0.02


def test_impurity_level_lands_inside_the_bath_energy_range(tmp_path, monkeypatch):
    # The invariant that catches a lost energy zero regardless of its cause: a partially
    # filled impurity level must sit within the span of the bath it hybridizes with, not
    # above all of it.
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    e_fermi = 0.6
    hyb = 0.7**2 / (w + 1j * eim + 2.0) + 0.5**2 / (w + 1j * eim + 0.5)
    _write_rspt_dir(tmp_path, w, hyb, -1.0 + e_fermi, e_fermi=e_fermi)

    monkeypatch.chdir(tmp_path)
    _run_build_h0(tmp_path, eim)

    diag = {}
    for line in (tmp_path / "cl_h0_op.dict").read_text().splitlines():
        i, j, re, _im = line.split()
        if i == j:
            diag[int(i)] = float(re)
    impurity, bath = diag[0], [v for k, v in diag.items() if k > 0]
    assert min(bath) <= impurity <= max(bath), f"impurity {impurity} outside bath {min(bath)}..{max(bath)}"

import warnings

import numpy as np
import pytest

from rspt2spectra.scripts.build_h0 import (
    _is_cubic_crystal_field,
    _spherical_signature,
    _verify_spherical_basis,
    run,
)


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


def _read_terms(path, name="cl_h0.h0"):
    """The (i, j) -> amplitude map of a written .h0 file, skipping magic line and header."""
    terms = {}
    for line in (path / name).read_text().splitlines():
        if line.startswith("#") or line.startswith("{") or line.strip() == "--":
            continue
        i, j, re, im = line.split()
        terms[(int(i), int(j))] = float(re) + 1j * float(im)
    return terms


def _read_h00(path):
    terms = _read_terms(path)
    if (0, 0) not in terms:
        raise AssertionError("impurity element not found in operator file")
    return terms[(0, 0)]


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


def test_impurity_level_lands_inside_a_straddling_bath(tmp_path, monkeypatch):
    # With bath states on both sides of E_F the impurity level must sit among them rather
    # than above all of them, which is what an unsubtracted Fermi energy looks like.
    #
    # Only valid when the bath straddles E_F. build_h0 defaults to fit_unocc=False, which
    # fits the occupied part only, and then a partially filled level is legitimately above
    # every bath state -- on the real NiO run, -1.3 eV against a bath at -5.3..-4.2 eV.
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    e_fermi = 0.6
    hyb = 0.7**2 / (w + 1j * eim + 2.0) + 0.5**2 / (w + 1j * eim + 0.5)
    _write_rspt_dir(tmp_path, w, hyb, -1.0 + e_fermi, e_fermi=e_fermi)

    monkeypatch.chdir(tmp_path)
    _run_build_h0(tmp_path, eim)

    diag = {i: v.real for (i, j), v in _read_terms(tmp_path).items() if i == j}
    impurity, bath = diag[0], [v for k, v in diag.items() if k > 0]
    assert min(bath) <= impurity <= max(bath), f"impurity {impurity} outside bath {min(bath)}..{max(bath)}"


# Diagonals measured on real RSPt output (see doc/plans/... for the full table): NiO/base's
# Ni_win_2 cluster after rotation (genuinely spherical) and FCC_Ni's raw, unrotated local
# Hamiltonian (genuinely cubic: eg, eg, t2g, t2g, t2g -- not palindromic).
_SPHERICAL_DIAG = [0.58158, 0.57953, 0.58363, 0.57953, 0.58158]
_CUBIC_DIAG = [0.71961, 0.71961, 0.71192, 0.71192, 0.71192]


def test_spherical_signature_accepts_a_genuinely_spherical_block():
    H = np.diag(_SPHERICAL_DIAG).astype(complex)
    H[0, 4] = H[4, 0] = 0.01  # the corner element a genuinely spherical, split block has
    assert _spherical_signature(H, l_val=2) is True


def test_spherical_signature_rejects_a_cubic_block():
    H = np.diag(_CUBIC_DIAG).astype(complex)  # H[-2, +2] == 0 exactly in the cubic basis
    assert _spherical_signature(H, l_val=2) is False


def test_spherical_signature_undetermined_for_unexpected_size():
    # Neither 5 (one spin sector) nor 10 (two) -- e.g. a composite multi-shell cluster.
    assert _spherical_signature(np.eye(3, dtype=complex), l_val=2) is None


def test_is_cubic_crystal_field():
    assert _is_cubic_crystal_field(2, 3) is True
    assert _is_cubic_crystal_field(2, 0) is False
    assert _is_cubic_crystal_field(4, 1) is False  # g shell: not a tag generate_rspt_T_matrix knows


def test_verify_spherical_basis_raises_on_non_unitary_t():
    T = 2 * np.eye(5, dtype=complex)
    H = np.eye(5, dtype=complex)
    with pytest.raises(RuntimeError, match="not unitary"):
        _verify_spherical_basis(H, T, "cl", l_val=2, basis_tag=3)


def test_verify_spherical_basis_raises_when_cubic_tag_but_signature_fails():
    T = np.eye(5, dtype=complex)
    H = np.diag(_CUBIC_DIAG).astype(complex)
    with pytest.raises(RuntimeError, match="spherical-under-cubic"):
        _verify_spherical_basis(H, T, "cl", l_val=2, basis_tag=3)


def test_verify_spherical_basis_warns_when_tag_is_not_cubic():
    # basis_tag 0 does not prove the crystal field is cubic, so a failed fingerprint is
    # inconclusive -- a warning, not a raise.
    T = np.eye(5, dtype=complex)
    H = np.diag(_CUBIC_DIAG).astype(complex)
    with pytest.warns(UserWarning, match="could not conclusively verify"):
        _verify_spherical_basis(H, T, "cl", l_val=2, basis_tag=0)


def test_verify_spherical_basis_is_silent_for_a_good_signature():
    T = np.eye(5, dtype=complex)
    H = np.diag(_SPHERICAL_DIAG).astype(complex)
    H[0, 4] = H[4, 0] = 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _verify_spherical_basis(H, T, "cl", l_val=2, basis_tag=3)  # must not raise or warn


def test_run_raises_when_cluster_is_not_in_green_inp(tmp_path, monkeypatch):
    # A cluster label that does not resolve in green.inp (e.g. the hybridization filenames'
    # "-obs" suffix defeating the match) must not be silently treated as "already spherical".
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    _write_rspt_dir(tmp_path, w, np.zeros_like(w, dtype=complex), -1.0)
    (tmp_path / "green.inp").write_text("cluster\n 1 Idother\n 1 2 1 1 0\n")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match=r"cl.*green\.inp"):
        _run_build_h0(tmp_path, eim)


def test_run_always_writes_basis_spherical_and_identity_rotation(tmp_path, monkeypatch):
    # Even when no rotation was needed (basis_tag 0, no Cf flag) the file must still say
    # basis: spherical and carry rot_to_spherical (the identity), rather than omitting it --
    # the file always says how to get to spherical, not just when a rotation happened to run.
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    hyb = 0.7**2 / (w + 1j * eim + 2.0)
    _write_rspt_dir(tmp_path, w, hyb, -1.0)

    monkeypatch.chdir(tmp_path)
    _run_build_h0(tmp_path, eim)

    header = (tmp_path / "cl_h0.h0").read_text().splitlines()[1]
    assert '"basis": "spherical"' in header
    assert '"rotation_applied": false' in header
    assert '"rot_to_spherical": [[[1.0, 0.0]]]' in header


def test_written_h0_is_read_back_by_impurity_model(tmp_path, monkeypatch):
    """The seam this whole format exists for: build_h0's output must load in impurityModel.

    Guarded by importorskip -- the two packages are independent by design and CI for either
    one does not have the other.
    """
    h0_format = pytest.importorskip("impurityModel.ed.h0_format")

    w = np.linspace(-6, 3, 800)
    eim = 0.05
    e_fermi = 0.6
    hyb = 0.7**2 / (w + 1j * eim + 2.0) + 0.5**2 / (w + 1j * eim + 0.5)
    _write_rspt_dir(tmp_path, w, hyb, -1.0 + e_fermi, e_fermi=e_fermi)

    monkeypatch.chdir(tmp_path)
    _run_build_h0(tmp_path, eim)

    parsed = h0_format.read_h0_file(tmp_path / "cl_h0.h0")

    # The file declares Rydberg; the reader converts, so the two differ by exactly that.
    ry_to_ev = h0_format.RY_TO_EV
    written = _read_terms(tmp_path)
    assert parsed.h0[((0, "c"), (0, "a"))] == pytest.approx(written[(0, 0)] * ry_to_ev)

    assert parsed.energy_reference == "fermi"
    assert parsed.header["fermi_energy"] == pytest.approx(e_fermi)
    assert parsed.impurity_orbitals == {0: (0,)}
    # And the invariant that catches a lost energy zero, now in the consumer's units.
    h = parsed.to_matrix()
    diag = h.diagonal().real
    assert diag[1:].min() <= diag[0] <= diag[1:].max()

import json
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from rspt2spectra.block_structure import BlockStructure, build_block_structure
from rspt2spectra.edchain import build_full_bath
from rspt2spectra.h0 import BLOCK_EQUIVALENCE_TOL, assemble_h0, prepare_hyb_fit
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.scripts.build_h0 import (
    _check_kramers_degeneracy,
    _detect_soc,
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
    """Invoke the build_h0 pipeline with the default knobs of these tests.

    ``allow_broken_time_reversal=True``: these tests fit a scalar, non-m-resolved synthetic
    hybridization to isolate one narrow property each (an offset, the Fermi shift, the
    straddling-bath check) and were never constructed to satisfy Kramers degeneracy -- the
    Kramers check itself is exercised directly, below (test__check_kramers_degeneracy_*).
    """
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
        allow_broken_time_reversal=True,
    )


def _read_terms(path, name="cl.h0"):
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
        # This test's scalar, non-m-resolved synthetic hybridization was never constructed to
        # satisfy Kramers degeneracy; see _run_build_h0's docstring for the same reasoning.
        allow_broken_time_reversal=True,
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

    header = (tmp_path / "cl.h0").read_text().splitlines()[1]
    assert '"basis": "spherical"' in header
    assert '"rotation_applied": false' in header
    assert '"rot_to_spherical": [[[1.0, 0.0]]]' in header
    assert '"spin_ordering": "down_first"' in header
    assert "spin_ordering" in json.loads(header)["required_features"]


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

    parsed = h0_format.read_h0_file(tmp_path / "cl.h0")

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


def test_check_kramers_degeneracy_silent_on_a_tr_symmetric_matrix():
    h = np.diag([-2.0, -2.0, -1.0, -1.0, 0.5, 0.5]).astype(complex)
    assert _check_kramers_degeneracy(h) == []


def test_check_kramers_degeneracy_flags_a_displaced_pair():
    h = np.diag([-2.0, -2.0, -1.0, -0.5, 0.5, 0.5]).astype(complex)
    violations = _check_kramers_degeneracy(h)
    energies = sorted(e for e, _ in violations)
    assert energies == [pytest.approx(-1.0), pytest.approx(-0.5)]
    assert all(m == 1 for _, m in violations)


def test_detect_soc_false_on_a_spin_diagonal_block():
    # down-down and up-up sectors populated, down-up sector exactly zero -- pure crystal field.
    h = np.zeros((10, 10), dtype=complex)
    h[:5, :5] = np.diag([1.0, 2.0, 3.0, 2.0, 1.0])
    h[5:, 5:] = np.diag([1.0, 2.0, 3.0, 2.0, 1.0])
    assert _detect_soc(h, l_val=2) is False


def test_detect_soc_true_when_the_spin_block_is_populated():
    h = np.zeros((10, 10), dtype=complex)
    h[:5, :5] = np.diag([1.0, 2.0, 3.0, 2.0, 1.0])
    h[5:, 5:] = np.diag([1.0, 2.0, 3.0, 2.0, 1.0])
    h[0, 5] = h[5, 0] = 0.1  # down m=-2 <-> up m=-2: SOC-style spin mixing
    assert _detect_soc(h, l_val=2) is True


def test_detect_soc_undetermined_for_unknown_l():
    assert _detect_soc(np.eye(10, dtype=complex), l_val=-1) is None


def test_run_refuses_a_kramers_violating_fit(tmp_path, monkeypatch):
    """The end-to-end guard: a hybridization with no down-up symmetry between its two m=+/-2
    poles is exactly the signature a bath fit landing time-reversal-partner blocks in
    different local minima would produce; run() must refuse to write the file rather than
    silently ship it.
    """
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    # Two well-separated poles at very different weights -- a genuinely asymmetric target, so
    # a fit converging to it is not "close enough" the way roundoff noise would be.
    hyb = 0.9**2 / (w + 1j * eim + 2.0) + 0.1**2 / (w + 1j * eim + 0.3)
    _write_rspt_dir(tmp_path, w, hyb, -1.0)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="time-reversal"):
        run(
            "cl",
            "star",
            eim,
            1,
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
    assert not (tmp_path / "cl.h0").exists()


def test_run_allow_broken_time_reversal_writes_anyway(tmp_path, monkeypatch):
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    hyb = 0.9**2 / (w + 1j * eim + 2.0) + 0.1**2 / (w + 1j * eim + 0.3)
    _write_rspt_dir(tmp_path, w, hyb, -1.0)

    monkeypatch.chdir(tmp_path)
    run(
        "cl",
        "star",
        eim,
        1,
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
        allow_broken_time_reversal=True,
    )
    assert (tmp_path / "cl.h0").exists()


def test_run_skips_the_kramers_check_when_soc_is_detected(tmp_path, monkeypatch):
    """When SOC is genuinely present, losing exact Kramers degeneracy in the independently-fit
    blocks is expected (see the comment at the check site in build_h0.py), so run() must write
    the file without the flag -- the same Kramers-violating fixture as
    test_run_refuses_a_kramers_violating_fit, but with SOC detected.

    _detect_soc's own correctness (measuring SOC from a real spin-resolved block) is covered
    directly by test_detect_soc_true_when_the_spin_block_is_populated; this test is only about
    the integration point -- the conditional gate in run() -- so patching is appropriate rather
    than building a full spin-resolved RSPt fixture (_write_rspt_dir's Local hamiltonian block
    is a single scalar, not spin-resolved; see test_detect_soc_undetermined_for_unknown_l for
    why that shape alone leaves contains_soc undetermined, not True).
    """
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    hyb = 0.9**2 / (w + 1j * eim + 2.0) + 0.1**2 / (w + 1j * eim + 0.3)
    _write_rspt_dir(tmp_path, w, hyb, -1.0)

    monkeypatch.chdir(tmp_path)
    with patch("rspt2spectra.scripts.build_h0._detect_soc", return_value=True):
        run(
            "cl",
            "star",
            eim,
            1,
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
    assert (tmp_path / "cl.h0").exists()


def test_run_still_checks_when_soc_is_undetermined(tmp_path, monkeypatch):
    """contains_soc is not True (rather than a falsiness test) treats an undetermined shell
    (_detect_soc returns None, e.g. l_val == -1) the same as "no SOC": still checked. There is
    no positive evidence to explain a Kramers violation away, so the conservative default is to
    keep refusing.
    """
    w = np.linspace(-6, 3, 800)
    eim = 0.05
    hyb = 0.9**2 / (w + 1j * eim + 2.0) + 0.1**2 / (w + 1j * eim + 0.3)
    _write_rspt_dir(tmp_path, w, hyb, -1.0)

    monkeypatch.chdir(tmp_path)
    with patch("rspt2spectra.scripts.build_h0._detect_soc", return_value=None):
        with pytest.raises(RuntimeError, match="time-reversal"):
            run(
                "cl",
                "star",
                eim,
                1,
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
    assert not (tmp_path / "cl.h0").exists()


def test_block_equivalence_tol_merges_the_measured_rspt_noise_floor():
    """Regression test for the ``tol=1e-15`` bug this fixes.

    On real RSPt-derived data (the NiO XAS tutorial's ``Ni_d`` cluster), orbitals that are
    exactly equivalent by symmetry -- spin degeneracy, or a cubic crystal field's t2g/eg
    manifolds -- differ only by ``max|Delta| ~= 1.4e-10`` between each other: floating-point
    and text-file-I/O noise from the RSPt -> .dat -> fit pipeline, not physics (see
    ``BLOCK_EQUIVALENCE_TOL``'s docstring in ``h0.py``). At the old ``tol=1e-15`` default this
    fractures a single equivalence class into several, each then fit by an independently-seeded
    stochastic optimizer -- producing measurably different, non-symmetric bath parameters between
    orbitals that must be exactly equivalent. ``BLOCK_EQUIVALENCE_TOL`` must sit safely above
    this noise floor.
    """
    w = np.linspace(-2, 1, 5)
    base = 1.0 / (w + 1j * 0.05 + 0.5)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, size=4) * 1.4e-10
    hyb = np.zeros((len(w), 4, 4), dtype=complex)
    for i in range(4):
        hyb[:, i, i] = base + noise[i]

    bs_shipped = build_block_structure(hyb, tol=BLOCK_EQUIVALENCE_TOL)
    assert len(bs_shipped.inequivalent_blocks) == 1

    # The old default: tight enough that the measured noise floor above defeats it, splitting
    # a single physical equivalence class into several independently-fit pieces.
    bs_machine_eps = build_block_structure(hyb, tol=1e-15)
    assert len(bs_machine_eps.inequivalent_blocks) > 1


def test_build_full_bath_raises_on_an_unreachable_block():
    """A block not covered by any inequivalent representative's identical/transposed/
    particle-hole relations would silently keep a ``None`` ``H_bath``/``v`` entry below --
    exactly the failure mode a ``BlockStructure`` inconsistency would produce, one step further
    downstream than where it is easiest to see (this is the new self-consistency guard rail,
    not a reproduction of the tol bug itself -- that bug never actually left a block
    unreachable, it just made the equivalence classes finer than they should be).
    """
    blocks = [[0], [1], [2]]
    bs = BlockStructure(
        blocks=blocks,
        identical_blocks=[[0], [1], []],  # block 2 is never listed anywhere
        transposed_blocks=[[], [], []],
        particle_hole_blocks=[[], [], []],
        particle_hole_transposed_blocks=[[], [], []],
        inequivalent_blocks=[0, 1],
    )
    H_bath_inequiv = [np.array([[-0.5]]), np.array([[-0.3]])]
    v_inequiv = [np.array([[0.1]]), np.array([[0.2]])]
    with pytest.raises(RuntimeError, match="not reached"):
        build_full_bath(H_bath_inequiv, v_inequiv, bs)


def test_assemble_h0_shares_one_fit_across_the_full_equivalence_class():
    """End-to-end (``prepare_hyb_fit`` -> ``fit_hyb`` -> ``assemble_h0``): four orbitals with a
    hybridization function that is identical up to the measured ~1.4e-10 RSPt noise floor must
    end up with bit-identical assembled bath parameters -- the concrete guarantee the tol fix
    (and the ``mat=H_local_Q`` anchoring) exists to provide. This mirrors what a spin-degenerate,
    cubic-crystal-field cluster like NiO's ``Ni_d`` requires: spin-up must equal spin-down, and a
    degenerate orbital manifold must be fit once and shared, not independently re-fit per member.
    """
    w = np.linspace(-2, 1, 300)
    delta = 0.05
    base = 0.1**2 / (w + 1j * delta + 0.5)
    rng = np.random.default_rng(1)
    noise = rng.uniform(-1, 1, size=4) * 1.4e-10
    n_orb = 4
    hyb = np.zeros((len(w), n_orb, n_orb), dtype=complex)
    H_local = np.zeros((n_orb, n_orb), dtype=complex)
    for i in range(n_orb):
        hyb[:, i, i] = base + noise[i]
        H_local[i, i] = -0.2 + noise[i]

    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(
        hyb, H_local, tol=BLOCK_EQUIVALENCE_TOL, verbose=False
    )
    assert len(block_structure.inequivalent_blocks) == 1

    ebs_star, vs_star, cs_star = fit_hyb(
        w,
        delta,
        phase_hyb,
        1,
        block_structure,
        0.1,
        (w[0], 0),
        False,
        None,
        weight_fun=np.ones_like,
        regularization="l2",
    )
    H, *_ = assemble_h0(
        ebs_star,
        vs_star,
        cs_star,
        H_local,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry="star",
        w=w,
        eim=delta,
        verbose=False,
    )

    n_bath_per_orbital = (H.shape[0] - n_orb) // n_orb
    assert n_bath_per_orbital > 0
    bath_couplings = np.abs(H[n_orb:, :n_orb])
    # Every orbital's own bath-coupling column, sorted, must match every other's exactly --
    # each orbital received the one shared fit, not an independent re-fit.
    reference = np.sort(bath_couplings[:, 0][bath_couplings[:, 0] > 0])
    for i in range(1, n_orb):
        column = np.sort(bath_couplings[:, i][bath_couplings[:, i] > 0])
        np.testing.assert_allclose(column, reference)

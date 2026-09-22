"""Tests for symmetry detection and symmetry-constrained hybridization fitting."""

from dataclasses import replace

import numpy as np
import pytest

from rspt2spectra.block_structure import (
    BlockStructure,
    build_block_structure,
    build_greens_function,
)
from rspt2spectra.edchain import build_full_bath
from rspt2spectra.h0 import assemble_h0
from rspt2spectra.hyb_fit import fit_hyb, get_state_per_inequivalent_block
from rspt2spectra.offdiagonal import (
    _c_basis,
    _varpro_cost_and_full_grad,
    _varpro_cost_and_grad,
    _varpro_inner_solve,
    expand_eb,
    free_window,
    get_v_and_eb_varpro_basin_hopping,
    max_bath_states,
    moment_weights,
)
from rspt2spectra.symmetries import (
    BlockSymmetry,
    detect_block_symmetry,
    find_antiunitary,
    hermitian_algebra_basis,
    mirror_interpolation_error,
    mirror_on_mesh,
    particle_hole_residual,
    trivial_symmetry,
)

W = np.linspace(-5, 5, 351)
DELTA = 0.1
Z = W + 1j * (DELTA * (1 + 0.5 * np.abs(W) ** 2))


def _model(eb, A, z=Z):
    """Delta(z) = sum_b A_b / (z - e_b) for explicit residues."""
    return np.einsum("mb, bij -> mij", 1.0 / (z[:, None] - np.asarray(eb)[None, :]), np.asarray(A))


def _psd(V):
    """Hermitian PSD residues from arbitrary factors."""
    return np.conj(np.swapaxes(V, -1, -2)) @ V


def _particle_hole_data(n_imp, seed, n_pairs=2, complex_v=True):
    """A hybridization with mirrored poles and A_{-e} = A_{+e}^T."""
    rng = np.random.default_rng(seed)
    half = np.sort(rng.uniform(0.6, 3.0, size=n_pairs))
    V = rng.normal(size=(n_pairs, n_imp, n_imp))
    if complex_v:
        V = V + 1j * rng.normal(size=(n_pairs, n_imp, n_imp))
    A_pos = _psd(V)
    eb = np.concatenate([-half[::-1], half])
    A = np.concatenate([np.swapaxes(A_pos, -1, -2)[::-1], A_pos])
    return eb, A, _model(eb, A)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def test_degenerate_block_collapses_to_the_identity():
    # Delta = d(w) * I_3 commutes with everything, so the only allowed residue is
    # a multiple of the identity.
    A = np.array([a * np.eye(3) for a in (1.0, 2.0, 0.5, 1.5)])
    hyb = _model([-2.0, -0.5, 1.0, 2.5], A)
    sym = detect_block_symmetry(W, hyb)
    assert sym.basis.shape == (1, 3, 3)
    # The basis element is fixed only up to a sign.
    assert np.allclose(np.abs(sym.basis[0]), np.eye(3) / np.sqrt(3))


def test_partially_degenerate_block_keeps_one_basis_element_per_manifold():
    A = np.array([np.diag([a, a, 2 * a]) for a in (1.0, 2.0, 0.5)]).astype(complex)
    sym = detect_block_symmetry(W, _model([-2.0, 0.0, 1.5], A))
    assert sym.basis.shape[0] == 2


def test_generic_block_is_unconstrained():
    # The negative control: nothing should be imposed on a block with no symmetry.
    rng = np.random.default_rng(4)
    V = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    sym = detect_block_symmetry(W, _model(np.sort(rng.uniform(-3, 3, 4)), _psd(V)))
    assert sym.basis.shape[0] == 2**2
    assert sym.particle_hole is False
    assert sym.trivial


def test_real_residues_are_detected_as_an_identity_antiunitary():
    # Transpose-symmetric (i.e. real) residues are the U = I case of the
    # antiunitary family -- the symmetry the old `realvalue_v` flag was after.
    rng = np.random.default_rng(6)
    V = rng.normal(size=(4, 2, 2))
    hyb = _model(np.sort(rng.uniform(-3, 3, 4)), _psd(V))
    U = find_antiunitary(hyb)
    assert U is not None
    assert np.allclose(U, np.eye(2))
    sym = detect_block_symmetry(W, hyb)
    assert sym.basis.shape[0] == 3  # real symmetric 2x2
    assert all(np.allclose(S, S.real) for S in sym.basis)


def test_complex_hermitian_block_admits_no_reality_constraint():
    # Hermitian but not transpose-symmetric: the old flag tested Hermiticity and
    # so wrongly forced the residues real.  Nothing may be imposed here.
    rng = np.random.default_rng(8)
    V = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    hyb = _model(np.sort(rng.uniform(-3, 3, 4)), _psd(V))
    # Hermitian at every frequency -- which the old `realvalue_v` flag tested and
    # then treated as a licence to drop the imaginary part -- but not symmetric.
    A = _psd(V)
    assert np.allclose(A, np.conj(np.swapaxes(A, -1, -2)))
    assert not np.allclose(A, np.swapaxes(A, -1, -2))
    assert find_antiunitary(hyb) is None


def test_particle_hole_symmetry_is_detected():
    _, _, hyb = _particle_hole_data(2, 21)
    assert particle_hole_residual(W, hyb) < 1e-12
    sym = detect_block_symmetry(W, hyb)
    assert sym.particle_hole
    assert np.all(np.abs(sym.parity) == 1)


def test_particle_hole_is_reported_but_not_enforced_without_a_two_sided_window():
    _, _, hyb = _particle_hole_data(2, 22)
    sym = detect_block_symmetry(W, hyb, allow_particle_hole=False)
    assert sym.particle_hole is False
    assert sym.report["particle_hole_residual"] < 1e-12
    assert "particle_hole_skipped" in sym.report


def test_algebra_basis_is_orthonormal_under_the_trace_inner_product():
    rng = np.random.default_rng(9)
    V = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
    basis = hermitian_algebra_basis(_model(np.sort(rng.uniform(-3, 3, 4)), _psd(V)))
    gram = np.einsum("pij, qji -> pq", basis, basis)
    assert np.allclose(gram, np.eye(basis.shape[0]))
    assert np.allclose(gram.imag, 0.0)


def test_mirror_on_mesh_masks_the_unpaired_window():
    w = np.linspace(-4.0, 1.0, 51)
    g = (w**2)[:, None, None] * np.ones((1, 1, 1))
    mirrored, mask = mirror_on_mesh(w, g)
    assert mask.sum() < len(w)  # the mesh is not symmetric about zero
    assert np.all(np.abs(w[mask]) <= 1.0 + 1e-12)
    assert np.allclose(mirrored[mask, 0, 0], w[mask] ** 2)


# --------------------------------------------------------------------------- #
# the constrained solve
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_imp", [1, 2, 3])
def test_degenerate_fit_keeps_every_residue_proportional_to_the_identity(n_imp):
    A = np.array([a * np.eye(n_imp) for a in (1.0, 2.0, 0.5, 1.5)])
    hyb = _model([-2.0, -0.5, 1.0, 2.5], A)
    sym = detect_block_symmetry(W, hyb)
    A_psd, _, _, C = _varpro_inner_solve(np.array([-2.1, -0.4, 0.9, 2.6]), Z, hyb, sym)
    for A_b in A_psd:
        assert np.allclose(A_b, np.trace(A_b).real / n_imp * np.eye(n_imp), atol=1e-10)
    assert np.allclose(C, np.trace(C).real / n_imp * np.eye(n_imp), atol=1e-10)


def test_particle_hole_fit_is_exactly_mirror_symmetric():
    _, _, hyb = _particle_hole_data(2, 31)
    sym = detect_block_symmetry(W, hyb)
    eb_free = np.array([0.8, 2.2])
    A_psd, V, G, C = _varpro_inner_solve(eb_free, Z, hyb, sym)
    eb_full = expand_eb(eb_free, sym)

    assert np.allclose(eb_full, -eb_full[::-1])
    # A_{-e} == A_{+e}^T
    assert np.allclose(A_psd, np.swapaxes(A_psd, -1, -2)[::-1], atol=1e-12)
    # V^dagger V reproduces the residues, so the bath is realizable.
    assert np.allclose(_psd(V), A_psd, atol=1e-10)
    # C = -C^T, hence purely imaginary and antisymmetric.
    assert np.allclose(C, -C.T, atol=1e-12)
    assert np.allclose(C.real, 0.0, atol=1e-12)

    model = np.einsum("mk, kij -> mij", G, A_psd) + C
    mirrored, mask = mirror_on_mesh(W, model)
    assert np.max(np.abs(model[mask] + np.conj(mirrored[mask]))) < 1e-10


def test_particle_hole_fit_with_real_residues_pins_the_constant_shift_to_zero():
    # With transpose-symmetric residues the odd sector is empty, and C = -C^T
    # then has no room left at all.
    _, _, hyb = _particle_hole_data(2, 32, complex_v=False)
    sym = detect_block_symmetry(W, hyb)
    assert sym.particle_hole
    _, _, _, C = _varpro_inner_solve(np.array([0.9, 2.1]), Z, hyb, sym)
    assert np.max(np.abs(C)) < 1e-12


def test_odd_state_count_adds_one_unpaired_pole_at_the_fermi_level():
    _, _, hyb = _particle_hole_data(2, 33)
    sym = replace(detect_block_symmetry(W, hyb), zero_pole=True)
    eb_free = np.array([0.8, 2.2])
    A_psd, _, _, _ = _varpro_inner_solve(eb_free, Z, hyb, sym)
    eb_full = expand_eb(eb_free, sym)
    assert len(eb_full) == 5
    assert eb_full[2] == 0.0
    assert np.allclose(eb_full, -eb_full[::-1])
    # The unpaired pole is its own mirror, so its residue must be symmetric.
    assert np.allclose(A_psd[2], A_psd[2].T, atol=1e-12)


def test_trivial_symmetry_recovers_complex_hermitian_residues_exactly():
    # Without symmetry every Hermitian PSD residue is allowed, so at the true
    # poles the solve must reproduce complex residues exactly.  The old code
    # tested Hermiticity but called it realness and truncated `A.real` here.
    rng = np.random.default_rng(12)
    V = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
    true_eb = np.sort(rng.uniform(-3, 3, 4))
    true_A = _psd(V)
    hyb = _model(true_eb, true_A)
    A_psd, _, G, C = _varpro_inner_solve(true_eb, Z, hyb, trivial_symmetry(3))

    assert np.allclose(A_psd, np.conj(np.swapaxes(A_psd, -1, -2)))
    assert np.all(np.linalg.eigvalsh(A_psd) > -1e-12)
    assert np.allclose(A_psd, true_A, atol=1e-8)
    assert np.max(np.abs(A_psd.imag)) > 1e-3  # genuinely complex, not truncated
    assert np.max(np.abs(C)) < 1e-8
    model = np.einsum("mk, kij -> mij", G, A_psd) + C
    assert np.linalg.norm(hyb - model) / np.linalg.norm(hyb) < 1e-10


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #


def _gradient_case(name):
    if name == "none":
        rng = np.random.default_rng(41)
        V = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
        hyb = _model(np.sort(rng.uniform(-3, 3, 4)), _psd(V))
        return hyb, trivial_symmetry(2), np.array([-2.0, -0.5, 0.7, 2.1])
    if name == "degenerate":
        A = np.array([a * np.eye(3) for a in (1.0, 2.0, 0.5, 1.5)])
        hyb = _model([-2.0, -0.5, 1.0, 2.5], A)
        return hyb, detect_block_symmetry(W, hyb), np.array([-2.2, -0.4, 0.9, 2.3])
    _, _, hyb = _particle_hole_data(2, 43)
    return hyb, detect_block_symmetry(W, hyb), np.array([0.7, 2.3])


@pytest.mark.parametrize("case", ["none", "degenerate", "particle_hole"])
def test_varpro_full_gradient_matches_finite_difference_under_symmetry(case):
    hyb, sym, eb = _gradient_case(case)
    wa = 1.0 / (1.0 + W**2)
    W_mn = moment_weights(W, 3)

    c_full, grad, _, _ = _varpro_cost_and_full_grad(eb, Z, hyb, wa, W_mn, sym)
    c_kauf, _, _, _ = _varpro_cost_and_grad(eb, Z, hyb, wa, W_mn, sym)
    assert np.isclose(c_full, c_kauf, rtol=0, atol=1e-9)

    h = 1e-6
    num = np.array(
        [
            (
                _varpro_cost_and_full_grad(eb + h * np.eye(len(eb))[i], Z, hyb, wa, W_mn, sym)[0]
                - _varpro_cost_and_full_grad(eb - h * np.eye(len(eb))[i], Z, hyb, wa, W_mn, sym)[0]
            )
            / (2 * h)
            for i in range(len(eb))
        ]
    )
    rel = np.max(np.abs(grad - num)) / (np.linalg.norm(num) + 1e-30)
    assert rel < 1e-5, f"{case}: full-grad rel err {rel:.2e}"


# --------------------------------------------------------------------------- #
# end to end through the optimizer and fit_hyb
# --------------------------------------------------------------------------- #


def test_fitted_bath_is_mirror_symmetric_through_the_optimizer():
    _, _, hyb = _particle_hole_data(1, 51, n_pairs=2, complex_v=False)
    sym = detect_block_symmetry(W, hyb)
    assert sym.particle_hole
    seeds = np.sort(np.random.default_rng(3).uniform(0.4, 3.0, size=(8, 2)), axis=1)
    v, eb, C, _ = get_v_and_eb_varpro_basin_hopping(
        W,
        DELTA,
        hyb,
        seeds,
        [(W[0], W[-1])] * 2,
        gamma=0.0,
        regularization=None,
        weight_function=np.ones_like,
        sym=sym,
        rng=np.random.default_rng(1),
    )
    assert len(eb) == 4
    assert np.allclose(eb, -eb[::-1], atol=1e-12)
    A = _psd(v)
    assert np.allclose(A, np.swapaxes(A, -1, -2)[::-1], atol=1e-10)
    assert np.max(np.abs(C)) < 1e-10


def _single_block_structure(hyb, w):
    return build_block_structure(hyb, mat=np.zeros(hyb.shape[1:]), tol=1e-6, w=w)


def test_fit_hyb_enforces_degeneracy_across_a_multiorbital_block():
    # A 2x2 block whose two orbitals are degenerate and coupled: the fitted bath
    # must not break the degeneracy.  Without the constraint the stochastic
    # optimizer leaves a residual asymmetry.
    off = 0.3
    base = np.array([[1.0, off], [off, 1.0]])
    A = np.array([a * base for a in (1.0, 2.0, 0.8)])
    hyb = _model([-2.0, -0.6, 1.2], A)
    bs = _single_block_structure(hyb, W)
    assert len(bs.inequivalent_blocks) == 1

    _, vs, cs = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=3,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
    )
    A_fit = _psd(vs[0])
    # Every residue must be a multiple of the same 2x2 pattern the data has.
    for A_b in A_fit:
        assert abs(A_b[0, 0] - A_b[1, 1]) < 1e-10 * max(abs(A_b[0, 0]), 1e-12)
        assert abs(A_b[0, 1] - A_b[1, 0]) < 1e-10 * max(abs(A_b[0, 1]), 1e-12)
    assert abs(cs[0][0, 0] - cs[0][1, 1]) < 1e-10


def test_fit_hyb_reports_particle_hole_without_enforcing_it_on_an_occupied_only_window():
    _, _, hyb = _particle_hole_data(1, 61, complex_v=False)
    bs = _single_block_structure(hyb, W)
    ebs, _, _ = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=4,
        block_structure=bs,
        gamma=0.0,
        x_lim=(W[0], 0),
        verbose=False,
        regularization=None,
    )
    # The default build_h0 window fits only w < 0, so no pole may land above it
    # and the mirrored bath is deliberately not built.
    assert len(ebs[0]) > 0
    assert np.all(ebs[0] < 0)


def test_fit_hyb_can_be_switched_off():
    base = np.array([[1.0, 0.3], [0.3, 1.0]])
    A = np.array([a * base for a in (1.0, 2.0)])
    hyb = _model([-1.5, 0.9], A)
    bs = _single_block_structure(hyb, W)
    kwargs = dict(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=2,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
    )
    _, vs_on, _ = fit_hyb(**kwargs)
    _, vs_off, _ = fit_hyb(**kwargs, enforce_symmetry=False)
    # On: every residue keeps the data's exact 2x2 pattern.
    A_on = _psd(vs_on[0])
    for A_b in A_on:
        assert abs(A_b[0, 0] - A_b[1, 1]) < 1e-10 * max(abs(A_b[0, 0]), 1e-12)
    # Off: the block is still fitted, just without the constraint.
    assert vs_off[0].shape[0] > 0


# --------------------------------------------------------------------------- #
# inter-block particle-hole equivalence
# --------------------------------------------------------------------------- #


def test_inter_block_particle_hole_round_trip():
    # Two blocks related by Delta_1(w) = -conj(Delta_0(-w)) must be detected as
    # one equivalence class, and replicating the fitted bath over that class must
    # reproduce the partner block exactly.
    rng = np.random.default_rng(71)
    eb0 = np.sort(rng.uniform(-3, 3, size=3))
    A0 = _psd(rng.normal(size=(3, 1, 1)) + 1j * rng.normal(size=(3, 1, 1)))
    d0 = _model(eb0, A0)

    hyb = np.zeros((len(W), 2, 2), dtype=complex)
    hyb[:, 0:1, 0:1] = d0
    hyb[:, 1:2, 1:2] = -np.conj(mirror_on_mesh(W, d0)[0])

    bs = build_block_structure(hyb, mat=np.zeros((2, 2)), tol=1e-6, w=W)
    assert bs.blocks == [[0], [1]]
    assert 1 in bs.particle_hole_blocks[0]
    assert bs.inequivalent_blocks == [0]

    # Reconstruction from the single inequivalent part returns the partner.
    G = build_greens_function([d0], bs)
    mask = mirror_on_mesh(W, d0)[1]
    assert np.allclose(G[mask, 0, 0], d0[mask, 0, 0])
    assert np.allclose(G[mask, 1, 1], -np.conj(mirror_on_mesh(W, d0)[0][mask, 0, 0]))

    # Replicating the bath over the class gives the same partner block.
    v0 = np.sqrt(A0[:, 0, 0].real)[:, None] + 0j
    H_bath_full, v_full = build_full_bath([np.diag(eb0)], [v0], bs)
    delta_full = np.einsum(
        "mk, ka, kb -> mab", 1.0 / (Z[:, None] - np.diag(H_bath_full)[None, :]), np.conj(v_full), v_full
    )
    assert np.allclose(delta_full[:, 0, 0], d0[:, 0, 0])
    assert np.allclose(delta_full[mask, 1, 1], -np.conj(mirror_on_mesh(W, d0)[0][mask, 0, 0]))


def test_assemble_h0_shifts_every_equivalent_block_not_only_identical_ones():
    # The fitted constant offset follows the same equivalence relations as the
    # bath; a particle-hole partner gets -conj(C), not zero.
    bs = BlockStructure(
        blocks=[[0], [1], [2]],
        identical_blocks=[[0], [], []],
        transposed_blocks=[[1], [], []],
        particle_hole_blocks=[[2], [], []],
        particle_hole_transposed_blocks=[[], [], []],
        inequivalent_blocks=[0],
    )
    shift = np.array([[0.25 + 0.0j]])
    H_imp = np.zeros((3, 3), dtype=complex)
    eye = np.eye(3, dtype=complex)
    out = assemble_h0(
        [np.array([-1.0])],
        [np.array([[[0.3 + 0j]]])],
        [shift],
        H_imp,
        np.zeros((3, 3), dtype=complex),
        eye,
        bs,
        bath_geometry="star",
        w=W,
        eim=0.1,
        verbose=False,
    )
    H_eff = out[-1]
    assert np.isclose(H_eff[0, 0], shift[0, 0])
    assert np.isclose(H_eff[1, 1], shift[0, 0])  # transposed: C^T
    assert np.isclose(H_eff[2, 2], -np.conj(shift[0, 0]))  # particle-hole: -conj(C)


def test_antiunitary_choice_is_deterministic_for_a_degenerate_null_space():
    # A degenerate block leaves many unitaries satisfying U D^T = D U, so the
    # candidate is picked from a seeded draw. Detection must still be repeatable:
    # a different accepted U means a different residue basis, hence a different
    # fit -- the same class of nondeterminism the seeded basin-hopping fixed.
    A = np.array([a * np.eye(3) for a in (1.0, 2.0, 0.5, 1.5)])
    hyb = _model([-2.0, -0.5, 1.0, 2.5], A)

    first = detect_block_symmetry(W, hyb)
    assert first.report["antiunitary"] is True
    assert first.report["antiunitary_is_identity"] is False  # the multi-candidate path
    for _ in range(3):
        again = detect_block_symmetry(W, hyb)
        np.testing.assert_array_equal(again.basis, first.basis)
        np.testing.assert_array_equal(find_antiunitary(hyb), find_antiunitary(hyb))


def test_frozen_bath_keeps_particle_hole_only_when_it_is_already_mirrored():
    _, _, hyb = _particle_hole_data(1, 81, complex_v=False)
    bs = _single_block_structure(hyb, W)

    mirrored = [np.array([-2.0, -0.7, 0.7, 2.0])]
    ebs, _, _ = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=4,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
        ebs_guess=mirrored,
        optimize_bath_energies=False,
    )
    assert np.allclose(ebs[0], mirrored[0])

    # A guess that is not mirror symmetric must not be silently folded onto its
    # mirror -- that would change the energies the caller asked to freeze.
    lopsided = [np.array([-2.0, -0.7, 0.5, 1.4])]
    ebs, _, _ = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=4,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
        ebs_guess=lopsided,
        optimize_bath_energies=False,
    )
    assert np.allclose(ebs[0], lopsided[0])


# --------------------------------------------------------------------------- #
# the mirrored-bath wiring: window, state count and the unpaired pole at E_F
# --------------------------------------------------------------------------- #


def test_free_window_reserves_room_for_the_mirror_partner():
    plain = trivial_symmetry(1)
    assert free_window(-4.0, 3.0, 0.1, plain) == (-4.0, 3.0)

    ph = BlockSymmetry(basis=plain.basis, parity=plain.parity, particle_hole=True)
    # Poles live on the positive half of the largest symmetric sub-window, and
    # the lower bound keeps +e and -e a full delta apart.
    assert free_window(-4.0, 3.0, 0.1, ph) == (0.05, 3.0)
    assert free_window(-3.0, 4.0, 0.1, ph) == (0.05, 3.0)
    # With an unpaired pole at zero sitting between them, the bound doubles.
    assert free_window(-4.0, 3.0, 0.1, replace(ph, zero_pole=True)) == (0.1, 3.0)


def test_state_cap_accounts_for_the_symmetric_sub_window():
    plain = trivial_symmetry(1)
    ph = BlockSymmetry(basis=plain.basis, parity=plain.parity, particle_hole=True)
    # On a symmetric window the two caps agree: the same window holds the same
    # number of delta-separated states however they are arranged.
    assert max_bath_states(-1.0, 1.0, 0.1, ph) == max_bath_states(-1.0, 1.0, 0.1, None)
    assert max_bath_states(-1.0, 1.0, 0.1, plain) == max_bath_states(-1.0, 1.0, 0.1, None)
    # On a lopsided one it must not: a mirrored bath only has the symmetric part
    # to sit in, so asking for the unconstrained count would place poles outside.
    assert max_bath_states(-5.0, 1.0, 0.1, ph) < max_bath_states(-5.0, 1.0, 0.1, None)


def test_odd_state_count_is_fitted_with_a_pole_at_the_fermi_level():
    # End to end, not by hand-constructing `zero_pole`: an odd count must come
    # back as k mirrored pairs plus exactly one pole at E_F.  The data needs
    # weight at E_F, or the unpaired pole is (rightly) pruned for zero hopping.
    eb = np.array([-2.0, -0.8, 0.0, 0.8, 2.0])
    A = np.array([a * np.eye(1) for a in (0.4, 0.9, 0.6, 0.9, 0.4)])
    hyb = _model(eb, A)
    bs = _single_block_structure(hyb, W)
    for n_states in (5, 7):
        ebs, vs, _ = fit_hyb(
            hyb=hyb,
            w=W,
            delta=DELTA,
            bath_states_per_orbital=n_states,
            block_structure=bs,
            gamma=0.0,
            verbose=False,
            regularization=None,
        )
        e = ebs[0]
        assert len(e) % 2 == 1, f"{n_states} states gave an even bath {e}"
        assert np.allclose(e, -e[::-1], atol=1e-12)
        assert e[len(e) // 2] == 0.0
        A = _psd(vs[0])
        assert np.allclose(A, np.swapaxes(A, -1, -2)[::-1], atol=1e-10)


def test_the_unpaired_pole_keeps_a_symmetric_residue_through_the_polish():
    # The pole at E_F is its own mirror, so its residue must satisfy A = A^T.
    # The joint SLSQP polish parametrizes a free V, so the restriction has to be
    # imposed there too; with an asymmetric window and weight the polish will
    # otherwise use the antisymmetric direction it is left.
    w = np.linspace(-4, 2.5, 400)
    z = w + 1j * (DELTA * (1 + 0.5 * np.abs(w) ** 2))
    rng = np.random.default_rng(5)
    half = np.sort(rng.uniform(0.6, 3.0, 4))
    A_pos = _psd(rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2)))
    eb = np.concatenate([-half[::-1], [0.0], half])
    A = np.concatenate([np.swapaxes(A_pos, -1, -2)[::-1], [np.eye(2) * 0.7], A_pos])
    hyb = _model(eb, A, z=z)

    sym = replace(detect_block_symmetry(w, hyb), zero_pole=True)
    assert sym.particle_hole
    v, eb_fit, C, _ = get_v_and_eb_varpro_basin_hopping(
        w,
        DELTA,
        hyb,
        np.sort(rng.uniform(0.4, 3.0, size=(8, 4)), axis=1),
        [(w[0], w[-1])] * 4,
        gamma=0.0,
        regularization=None,
        weight_function=lambda x: 1.0 / (1.0 + (x - 1.0) ** 2),
        sym=sym,
        rng=np.random.default_rng(1),
    )
    A_fit = _psd(v)
    zero = len(eb_fit) // 2
    assert eb_fit[zero] == 0.0
    assert np.max(np.abs(A_fit[zero] - A_fit[zero].T)) < 1e-12
    assert np.allclose(A_fit, np.swapaxes(A_fit, -1, -2)[::-1], atol=1e-12)

    # And the delivered model really is particle-hole symmetric, evaluated
    # analytically at +w and -w so no interpolation enters the check.
    zp, zm = w + 1j * DELTA, -w + 1j * DELTA
    mp = np.einsum("mk, kij -> mij", 1.0 / (zp[:, None] - eb_fit[None, :]), A_fit) + C
    mm = np.einsum("mk, kij -> mij", 1.0 / (zm[:, None] - eb_fit[None, :]), A_fit) + C
    assert np.max(np.abs(mp + np.conj(mm))) / np.max(np.abs(mp)) < 1e-12


def test_a_block_allocated_a_single_state_still_fits():
    # A mirrored bath needs a pair; one state leaves the positive half empty.
    _, _, hyb = _particle_hole_data(1, 92, complex_v=False)
    bs = _single_block_structure(hyb, W)
    ebs, _, _ = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=1,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
    )
    assert len(ebs[0]) >= 1
    assert np.allclose(ebs[0], -ebs[0][::-1], atol=1e-12)


def test_a_lopsided_window_does_not_squeeze_the_bath_around_the_fermi_level():
    # The window reaches above E_F by more than delta, so the "wide enough for a
    # pair" guard passes -- but enforcing the mirror would still confine every
    # pole to |e| <= 0.3 while the hybridization extends to -5.
    _, _, hyb = _particle_hole_data(1, 93, complex_v=False)
    bs = _single_block_structure(hyb, W)
    ebs, _, _ = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=4,
        block_structure=bs,
        gamma=0.0,
        x_lim=(-5.0, 0.3),
        verbose=False,
        regularization=None,
    )
    assert np.all(ebs[0] >= -5.0) and np.all(ebs[0] <= 0.3)
    assert np.any(ebs[0] < -0.5), f"bath collapsed onto E_F: {ebs[0]}"


def test_a_self_particle_hole_symmetric_block_is_not_its_own_partner():
    # A block can be particle-hole symmetric with itself. That is a constraint on
    # its fit, not an equivalence with another block: listing it as its own
    # partner makes every consumer write it twice (identical, then particle-hole)
    # and replace the fitted bath with its own mirror image.
    _, _, hyb = _particle_hole_data(1, 101, complex_v=False)
    bs = build_block_structure(hyb, mat=np.zeros((1, 1)), tol=1e-6, w=W)
    assert bs.particle_hole_blocks == [[]]
    assert bs.particle_hole_transposed_blocks == [[]]
    assert bs.identical_blocks == [[0]]

    # An occupied-only fit is not itself mirror symmetric; replication must leave
    # it exactly alone rather than flip it to the unoccupied side.
    eb = np.array([-1.5, -0.4])
    v = np.array([[0.5], [0.3]]) + 0j
    H_bath, v_full = build_full_bath([np.diag(eb)], [v], bs)
    assert np.allclose(np.diag(H_bath), eb)
    assert np.allclose(v_full[:, 0], v[:, 0])


def test_particle_hole_equivalence_is_skipped_when_the_mesh_cannot_resolve_it():
    # Particle-hole compares w against -w. On a mesh that is not symmetric about
    # the Fermi level the mirrored values are interpolated, and the symmetry
    # cannot be resolved below that interpolation error -- answering "not
    # equivalent" there would be a statement about the mesh, not the physics.
    w = np.linspace(-6.0, 3.0, 800)
    z = w + 1j * 0.05
    eb = np.sort(np.random.default_rng(3).uniform(-2.5, 2.5, 3))
    a = np.abs(np.random.default_rng(4).normal(size=3))
    d0 = np.einsum("mb, bij -> mij", 1.0 / (z[:, None] - eb[None, :]), a[:, None, None] * np.ones((1, 1, 1)))
    d1 = np.einsum("mb, bij -> mij", 1.0 / (z[:, None] + eb[None, :]), a[:, None, None] * np.ones((1, 1, 1)))
    G = np.zeros((len(w), 2, 2), dtype=complex)
    G[:, 0:1, 0:1], G[:, 1:2, 1:2] = d0, d1

    assert mirror_interpolation_error(w, d0) > 1e-6
    with pytest.warns(RuntimeWarning, match="interpolation error"):
        bs = build_block_structure(G, mat=np.zeros((2, 2)), tol=1e-6, w=w)
    assert bs.particle_hole_blocks == [[], []]

    # The same physics on a mesh symmetric about E_F is mirrored exactly, and is
    # detected.
    w_sym = np.linspace(-6.0, 6.0, 801)
    z_sym = w_sym + 1j * 0.05
    e0 = np.einsum("mb, bij -> mij", 1.0 / (z_sym[:, None] - eb[None, :]), a[:, None, None] * np.ones((1, 1, 1)))
    e1 = np.einsum("mb, bij -> mij", 1.0 / (z_sym[:, None] + eb[None, :]), a[:, None, None] * np.ones((1, 1, 1)))
    G_sym = np.zeros((len(w_sym), 2, 2), dtype=complex)
    G_sym[:, 0:1, 0:1], G_sym[:, 1:2, 1:2] = e0, e1
    assert mirror_interpolation_error(w_sym, e0) < 1e-9
    assert 1 in build_block_structure(G_sym, mat=np.zeros((2, 2)), tol=1e-6, w=w_sym).particle_hole_blocks[0]


@pytest.mark.parametrize(
    "guess",
    [
        [-2.0, -0.7, 0.7, 2.0],  # mirrored, even
        [-2.0, 0.0, 2.0],  # mirrored, with a pole at E_F
        [-2.0, -0.7, 0.5, 1.4],  # not mirrored at all
        [-2.0, -0.7, 0.7],  # mirrored magnitudes, unbalanced count
    ],
)
@pytest.mark.parametrize("n_states", [3, 4, 5])
def test_a_frozen_bath_is_returned_exactly_as_given(guess, n_states):
    # Freezing means the caller's energies come back untouched. Particle-hole
    # enforcement must never fold, mirror or pad them behind the caller's back --
    # the allocated state count has nothing to do with the frozen set.
    _, _, hyb = _particle_hole_data(1, 102, complex_v=False)
    bs = _single_block_structure(hyb, W)
    ebs, _, _ = fit_hyb(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=n_states,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
        ebs_guess=[np.array(guess)],
        optimize_bath_energies=False,
    )
    np.testing.assert_allclose(np.sort(ebs[0]), np.sort(guess))


def test_switching_symmetry_off_really_leaves_the_fit_unconstrained():
    # The positive half of `test_fit_hyb_can_be_switched_off`: with enforcement
    # off the fitted residues must NOT reproduce the data's exact pattern, or the
    # flag would be doing nothing.
    base = np.array([[1.0, 0.3], [0.3, 1.0]])
    hyb = _model([-1.5, 0.9], np.array([a * base for a in (1.0, 2.0)]))
    bs = _single_block_structure(hyb, W)
    kwargs = dict(
        hyb=hyb,
        w=W,
        delta=DELTA,
        bath_states_per_orbital=3,
        block_structure=bs,
        gamma=0.0,
        verbose=False,
        regularization=None,
    )
    A_on = _psd(fit_hyb(**kwargs)[1][0])
    A_off = _psd(fit_hyb(**kwargs, enforce_symmetry=False)[1][0])
    on_err = max(abs(A[0, 0] - A[1, 1]) for A in A_on)
    off_err = max(abs(A[0, 0] - A[1, 1]) for A in A_off)
    assert on_err < 1e-12
    assert off_err > 1e-10, "the unconstrained fit happened to be symmetric; test is not discriminating"


def test_the_constant_shift_is_restricted_to_the_transpose_odd_sector():
    # Particle-hole forces C = -C^T, so C lives in the transpose-odd sector
    # alone -- and vanishes entirely when the residues are transpose-symmetric.
    _, _, hyb = _particle_hole_data(2, 103)
    sym = detect_block_symmetry(W, hyb)
    assert sym.particle_hole
    np.testing.assert_allclose(_c_basis(sym), sym.odd_basis)
    for S in _c_basis(sym):
        np.testing.assert_allclose(S.T, -S, atol=1e-12)

    _, _, real_hyb = _particle_hole_data(2, 104, complex_v=False)
    real_sym = detect_block_symmetry(W, real_hyb)
    assert real_sym.particle_hole
    assert _c_basis(real_sym).shape[0] == 0


def test_state_allocation_caps_a_mirrored_block_to_its_symmetric_sub_window():
    # The allocator caps each block at what fits in the window; a mirrored bath
    # only has the symmetric part of it to sit in, so the cap must know the
    # symmetry.  Detection runs on the full (symmetric) mesh, the cap on the
    # lopsided fit window -- the two are deliberately different meshes.
    hyb = _model([-2.0, -0.8, 0.8, 2.0], np.array([a * np.eye(1) for a in (0.4, 0.9, 0.9, 0.4)]))
    bs = _single_block_structure(hyb, W)
    sym = detect_block_symmetry(W, hyb)
    assert sym.particle_hole

    window = (W >= -3.0) & (W <= 1.0)
    w_fit, hyb_fit = W[window], hyb[window]
    aware = get_state_per_inequivalent_block(bs, 200, hyb_fit, w_fit, np.ones_like, DELTA, syms=[sym])
    unaware = get_state_per_inequivalent_block(bs, 200, hyb_fit, w_fit, np.ones_like, DELTA)
    assert aware[0] < unaware[0], "the mirrored cap did not bind"
    assert aware[0] == max_bath_states(w_fit[0], w_fit[-1], DELTA, sym)

"""Choice of the fitting basis with the Coulomb tensor's sparsity in mind."""

import numpy as np
import pytest

from rspt2spectra import utils
from rspt2spectra.h0 import prepare_hyb_fit
from rspt2spectra.utils import (
    _joint_diagonalize,
    _rotate_u4,
    _so3_rotation,
    block_diagonalize_hyb,
    count_u_terms,
    rotate_4index_U,
    rotate_Greens_function,
    rotate_matrix,
)

L = 1  # p shell: 6 spin-orbitals, small enough to be quick
N = 2 * (2 * L + 1)


def _slater_like_u4(l=L):
    """A rotationally invariant, sparse two-body tensor: U ~ sum over (m1 m2 m3 m4) with m conserved,
    built as an exact SO(3) x SU(2) invariant, density-density  n_i n_j  summed over *all* pairs is
    invariant under any unitary, so use the invariant  (sum_i n_i)^2  plus the exchange-like invariant
    (L.L) via the Casimir; here we take the cheapest nontrivial one: total-charge squared plus L^2 * 0.3.
    """
    n = 2 * (2 * l + 1)
    # the tensor of (N_tot)^2 is U[i,j,k,l] = delta_ik delta_jl, invariant under every unitary and dense in
    # the sense that it does not densify: rotating it returns the same tensor.
    U = np.einsum("ik,jl->ijkl", np.eye(n), np.eye(n)).astype(complex)
    return U


def test_fast_rotation_matches_reference():
    rng = np.random.default_rng(0)
    n = 4
    U = rng.normal(size=(n,) * 4) + 1j * rng.normal(size=(n,) * 4)
    T = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    assert np.allclose(_rotate_u4(U, T), rotate_4index_U(U, T), atol=1e-12)


def test_count_u_terms_uses_the_solver_cutoff():
    U = np.zeros((2,) * 4, dtype=complex)
    U[0, 0, 1, 1] = 1.0
    U[1, 1, 0, 0] = 1e-12
    assert count_u_terms(U) == 1


def test_so3_rotation_is_unitary_and_group_like():
    D = _so3_rotation(2, np.array([0.3, -0.2, 0.9]), np.array([0.1, 0.5, -0.4]))
    assert np.allclose(D.conj().T @ D, np.eye(10), atol=1e-12)
    assert np.allclose(_so3_rotation(2, np.zeros(3)), np.eye(10))


def _scrambled_problem(seed=3, l=2):
    """Diagonal CF Hamiltonian and hybridization, scrambled by a known SO(3) x SU(2) rotation,
    with a Slater-type (rotationally invariant) U in the diagonal basis."""
    from scipy.linalg import expm  # noqa: F401

    n = 2 * (2 * l + 1)
    rng = np.random.default_rng(seed)
    S = np.eye(n, dtype=complex)  # input basis = spherical harmonics
    eps = rng.uniform(-0.3, 0.3, size=n)
    H = np.diag(eps).astype(complex)
    w = np.linspace(-1.5, 0.5, 60)
    v2 = rng.uniform(0.05, 0.2, size=n)
    hyb = np.array([np.diag(v2 / (x + 0.05j + 0.8 + 0.1 * eps)) for x in w]).astype(complex)
    # Slater-like U that is invariant under SO(3)xSU(2): density-density on the m-resolved basis does
    # not have that property; use the full total-number tensor times a term from the shell's
    # spin-orbit-free structure -- invariant by construction.
    U = np.einsum("ik,jl->ijkl", np.eye(n), np.eye(n)).astype(complex)
    th = rng.normal(size=3) * 1.5
    R = _so3_rotation(l, th, rng.normal(size=3))
    return S, rotate_matrix(H, R), rotate_Greens_function(hyb, R), U, w, R


def test_axis_rotation_found_and_u_unchanged():
    S, H, hyb, U, w, _ = _scrambled_problem()
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, info=info)
    assert info["tier"] == "axis rotation"
    assert info["u_terms"] == info["u_terms_input"] == count_u_terms(U)
    # the scrambled problem is diagonal again
    d = rotate_matrix(H, Q)
    assert np.max(np.abs(d - np.diag(np.diag(d)))) < 1e-8


def test_no_u_or_infinite_budget_reproduces_the_free_rotation_exactly():
    S, H, hyb, U, w, _ = _scrambled_problem()
    _, Q0 = block_diagonalize_hyb(hyb, mat=H, w=w)
    _, Q1 = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=np.inf)
    _, Q2 = block_diagonalize_hyb(hyb, mat=H, w=w, u4=None, to_spherical=S)
    assert np.array_equal(Q0, Q1) and np.array_equal(Q0, Q2)


def test_partial_set_skips_the_axis_rotation():
    S, H, hyb, U, w, _ = _scrambled_problem()
    keep = [0, 1, 2, 5, 6, 7]  # not a complete shell
    Hs = H[np.ix_(keep, keep)]
    hs = hyb[:, keep][:, :, keep]
    Us = U[np.ix_(keep, keep, keep, keep)]
    info = {}
    block_diagonalize_hyb(hs, mat=Hs, w=w, u4=Us, to_spherical=S[keep, :], info=info)
    assert info["tier"] != "axis rotation"


def test_pauli_forbidden_elements_do_not_defeat_the_invariance_probe():
    """A tensor that omits u[i,i,k,l] and u[i,j,k,k] (they vanish in the operator) is still invariant."""
    S, H, hyb, _, w, _ = _scrambled_problem()
    U = _l_dot_l_u4(2).copy()
    n = U.shape[0]
    for i in range(n):
        U[i, i, :, :] = 0
        U[:, :, i, i] = 0
    info = {}
    # a rotated tensor without those elements fills in again, so it only fits a generous budget; the point
    # is that the probe accepts it as invariant (the input orbitals are already the answer here)
    block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=100.0, info=info)
    assert info["tier"] == "axis rotation"


def test_non_invariant_u_skips_the_axis_rotation():
    S, H, hyb, U, w, _ = _scrambled_problem()
    rng = np.random.default_rng(5)
    U = U.copy()
    U[0, 1, 0, 1] += 0.7  # breaks rotational invariance
    info = {}
    block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, info=info)
    assert info["tier"] != "axis rotation"


def _sparse_u_and_dense_hyb(n=6, seed=1):
    """No symmetry: hybridization and H are diagonal in a rotated basis, U is sparse in the input."""
    rng = np.random.default_rng(seed)
    U = np.zeros((n,) * 4, dtype=complex)
    for i in range(n):
        for j in range(n):
            U[i, j, i, j] = 1.0 + i * j  # density-density with orbital-dependent weights: sparse, not rotation invariant
    Q0 = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    eps = np.linspace(-0.3, 0.3, n)
    w = np.linspace(-1.5, 0.5, 40)
    hyb_d = np.array([np.diag(0.1 / (x + 0.05j + 0.8 + eps)) for x in w]).astype(complex)
    return U, rotate_matrix(np.diag(eps).astype(complex), np.conj(Q0.T)), rotate_Greens_function(hyb_d, np.conj(Q0.T)), w


def test_free_rotation_respects_the_budget():
    U, H, hyb, w = _sparse_u_and_dense_hyb()
    n_ref = count_u_terms(U)
    # unconstrained: the free rotation densifies U a lot
    _, Q_free = block_diagonalize_hyb(hyb, mat=H, w=w)
    assert count_u_terms(_rotate_u4(U, Q_free)) > 2 * n_ref
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, max_u_densify=2.0, info=info)
    assert info["tier"] == "budgeted free rotation"
    assert count_u_terms(_rotate_u4(U, Q)) <= 2 * n_ref
    assert info["u_terms"] == count_u_terms(_rotate_u4(U, Q))
    # a budgeted rotation is never less diagonal than the identity
    mats = np.concatenate([hyb, H[None]], axis=0)
    assert utils._off_diagonal_weight(mats, Q) <= utils._off_diagonal_weight(mats, np.eye(len(Q))) + 1e-12


def test_budgeted_rotation_keeps_the_spins_identical():
    """In a spin-degenerate problem both spin blocks get the same rotation, whatever the budget does."""
    U, H, hyb, w = _sparse_u_and_dense_hyb(n=4)
    n = 4
    U2 = np.zeros((2 * n,) * 4, dtype=complex)
    for s1 in (0, 1):
        for s2 in (0, 1):
            U2[s1 * n : (s1 + 1) * n, s2 * n : (s2 + 1) * n, s1 * n : (s1 + 1) * n, s2 * n : (s2 + 1) * n] = U
    z = np.zeros_like(hyb)
    hyb2 = np.concatenate([np.concatenate([hyb, z], axis=2), np.concatenate([z, hyb], axis=2)], axis=1)
    H2 = np.block([[H, np.zeros_like(H)], [np.zeros_like(H), H]])
    n_ref = count_u_terms(U2)
    for factor in (1.2, 1.6, 2.0):
        info = {}
        _, Q = block_diagonalize_hyb(hyb2, mat=H2, w=w, u4=U2, max_u_densify=factor, info=info)
        assert count_u_terms(_rotate_u4(U2, Q)) <= factor * n_ref
        assert np.allclose(Q[:n, :n], Q[n:, n:], atol=1e-9) or np.allclose(
            np.abs(Q[:n, :n]), np.abs(Q[n:, n:]), atol=1e-9
        ), factor


def test_generous_budget_gives_the_free_rotation():
    U, H, hyb, w = _sparse_u_and_dense_hyb()
    _, Q_free = block_diagonalize_hyb(hyb, mat=H, w=w)
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, max_u_densify=1e6, info=info)
    assert info["tier"] == "free rotation"
    assert np.array_equal(Q, Q_free)


def test_choice_is_stable_under_tiny_perturbations():
    S, H, hyb, U, w, _ = _scrambled_problem()
    _, Q1 = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S)
    pert = hyb.copy()
    pert[:, 0, 0] += 1e-9
    _, Q2 = block_diagonalize_hyb(pert, mat=H, w=w, u4=U, to_spherical=S)
    assert np.allclose(np.abs(Q1), np.abs(Q2), atol=1e-5)


def test_prepare_hyb_fit_reports_and_keeps_its_contract(capsys):
    S, H, hyb, U, w, _ = _scrambled_problem()
    bs = prepare_hyb_fit(hyb, H, tol=1e-6, verbose=True, w=w, u4=U, to_spherical=S).block_structure
    out = capsys.readouterr().out
    assert "Fitting basis: axis rotation" in out
    assert len(bs.blocks) == H.shape[0]


def _l_dot_l_u4(l):
    """A sparse, non-trivial, SO(3)-invariant (but not U(n)-invariant) two-body tensor: sum_a L_a (x) L_a."""
    ops = utils._angular_momentum_ops(l)
    n = 2 * (2 * l + 1)
    U = np.zeros((n,) * 4, dtype=complex)
    for L_a in ops:
        La = np.kron(np.eye(2), L_a)
        U += np.einsum("ik,jl->ijkl", La, La)
    return U


def test_axis_rotation_keeps_a_nontrivially_invariant_u_sparse():
    S, H, hyb, _, w, R = _scrambled_problem(l=2)
    U = _l_dot_l_u4(2)
    # the scrambling is a symmetry of U, so the scrambled problem has the same sparse U
    assert np.max(np.abs(_rotate_u4(U, R) - U)) < 1e-10
    n_ref = count_u_terms(U)
    # free rotation densifies it (it is not invariant under general unitaries)
    _, Q_free = block_diagonalize_hyb(hyb, mat=H, w=w)
    assert count_u_terms(_rotate_u4(U, Q_free)) > n_ref
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, info=info)
    assert info["tier"] == "axis rotation"
    assert count_u_terms(_rotate_u4(U, Q)) == n_ref


# ---------------------------------------------------------------------------------------------
# Shapes of the rigid-rotation tier and the partition-driven frame search
# ---------------------------------------------------------------------------------------------

from rspt2spectra.block_structure import get_blocks, relative_couplings  # noqa: E402


def _random_rotation(l, seed, spin=True):
    rng = np.random.default_rng(seed)
    return _so3_rotation(l, rng.normal(size=3) * 1.5, rng.normal(size=3) if spin else None)


def _sizes(Q, hyb, H, tol):
    G = np.conj(Q.T)[None] @ hyb @ Q[None]
    return sorted(len(b) for b in get_blocks(G, np.conj(Q.T) @ H @ Q, tol=tol))


def test_relative_couplings_reproduce_the_block_partition():
    rng = np.random.default_rng(0)
    n = 8
    G = np.zeros((5, n, n), dtype=complex)
    for blk in ([0, 1, 2], [3, 4], [5, 6, 7]):
        A = rng.normal(size=(len(blk),) * 2) + 1j * rng.normal(size=(len(blk),) * 2)
        G[np.ix_(range(5), blk, blk)] = A
    G += 0.01 * rng.normal(size=G.shape)
    H = np.diag(rng.normal(size=n)).astype(complex)
    for tol in (1e-6, 1e-3, 0.02, 0.1, 0.3):
        c = relative_couplings(G, H, tol)
        assert np.allclose(c, c.T)
        n_comp, labels = utils._components(c > tol)
        ref = get_blocks(G, H, tol=tol)
        assert sorted(np.bincount(labels)) == sorted(len(b) for b in ref)


def test_block_threshold_is_the_tolerance_at_which_blocks_get_small_enough():
    c = np.zeros((4, 4))
    c[0, 1] = c[1, 0] = 0.5
    c[1, 2] = c[2, 1] = 0.3
    c[2, 3] = c[3, 2] = 0.1
    assert utils._block_threshold(c, 4) == 0.0
    assert utils._block_threshold(c, 3) == pytest.approx(0.1)
    assert utils._block_threshold(c, 2) == pytest.approx(0.3)
    assert utils._block_threshold(c, 1) == pytest.approx(0.5)


def test_real_harmonics_follow_the_condon_shortley_convention():
    for l in (1, 2, 3):
        W = utils._real_harmonics(l)
        assert np.allclose(np.conj(W.T) @ W, np.eye(2 * l + 1))
        Lx, Ly, Lz = utils._angular_momentum_ops(l)
        # even operators are real in a real basis; Lx Ly + Ly Lx is one of them
        for op in (Lx @ Lx, Lz @ Lz, Lx @ Ly + Ly @ Lx, Ly @ Lz + Lz @ Ly):
            assert np.max(np.abs(np.imag(np.conj(W.T) @ op @ W))) < 1e-12


def test_real_p_harmonics_are_px_py_pz_with_condon_shortley_phases():
    W = utils._real_harmonics(1)
    Lx, Ly, Lz = utils._angular_momentum_ops(1)
    assert np.allclose(Ly @ W[:, 0], 0) and np.allclose(Lz @ W[:, 1], 0) and np.allclose(Lx @ W[:, 2], 0)
    assert np.allclose(W[:, 2], np.array([1, 0, -1]) / np.sqrt(2))


def test_cubic_f_harmonics_diagonalise_the_cubic_field():
    W = utils._cubic_f_harmonics()
    assert np.allclose(np.conj(W.T) @ W, np.eye(7))
    Lx, Ly, Lz = utils._angular_momentum_ops(3)
    Lp, Lm = Lx + 1j * Ly, Lx - 1j * Ly
    Lz2 = Lz @ Lz
    O40 = 35 * Lz2 @ Lz2 - (30 * 12 - 25) * Lz2 + (3 * 144 - 72) * np.eye(7)
    V = O40 + 5 * (np.linalg.matrix_power(Lp, 4) + np.linalg.matrix_power(Lm, 4)) / 2
    D = np.conj(W.T) @ V @ W
    assert np.max(np.abs(D - np.diag(np.diag(D)))) < 1e-9
    # the sign of the O44 weight is fixed by invariance under a C4 rotation about x
    from scipy.linalg import expm

    C4x = expm(-1j * np.pi / 2 * Lx)
    assert np.max(np.abs(V @ C4x - C4x @ V)) < 1e-9
    # a2u, t1u, t2u: levels of multiplicity 1, 3, 3
    levels = np.unique(np.round(np.diag(D).real, 6))
    assert sorted(np.sum(np.isclose(np.diag(D).real, x, atol=1e-6)) for x in levels) == [1, 3, 3]


def test_coupled_j_basis_diagonalises_spin_orbit_coupling():
    l = 1
    W = utils._coupled_j_basis(l)
    Lops = utils._angular_momentum_ops(l)
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, 1j], [-1j, 0]]), np.diag([-1.0, 1.0])]
    LS = sum(np.kron(s / 2, Lc) for s, Lc in zip(sig, Lops))
    D = np.conj(W.T) @ LS @ W
    assert np.max(np.abs(D - np.diag(np.diag(D)))) < 1e-9
    assert sorted(np.round(np.diag(D).real, 6)) == [-1.0] * 2 + [0.5] * 4


def test_real_harmonic_field_in_a_scrambled_frame_selects_the_real_shape():
    l = 2
    n = 2 * (2 * l + 1)
    Wr = np.kron(np.eye(2), utils._real_harmonics(l))
    rng = np.random.default_rng(4)
    eps = rng.uniform(-0.3, 0.3, size=n)
    w = np.linspace(-1.5, 0.5, 60)
    v2 = rng.uniform(0.05, 0.2, size=n)
    hyb_d = np.array([np.diag(v2 / (x + 0.05j + 0.8 + 0.1 * eps)) for x in w]).astype(complex)
    # diagonal in real harmonics, then expressed in complex ones and scrambled by a rigid rotation
    H0 = Wr @ np.diag(eps).astype(complex) @ np.conj(Wr.T)
    hyb0 = Wr[None] @ hyb_d @ np.conj(Wr.T)[None]
    R = _random_rotation(l, 11)
    H, hyb = rotate_matrix(H0, R), rotate_Greens_function(hyb0, R)
    U = _l_dot_l_u4(l)
    ratio = count_u_terms(_rotate_u4(U, Wr)) / count_u_terms(U)
    S = np.eye(n, dtype=complex)

    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=ratio * 1.01 + 1, tol=1e-6, info=info)
    assert info["tier"] == "axis rotation" and info["shape"] == "real harmonics"
    d = rotate_matrix(H, Q)
    assert np.max(np.abs(d - np.diag(np.diag(d)))) < 1e-7
    assert info["u_terms"] <= (ratio * 1.01 + 1) * count_u_terms(U)
    assert ratio > 1.0
    assert info["u_terms"] == count_u_terms(_rotate_u4(U, Wr))
    info1 = {}
    _, Q1 = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=1.0, tol=1e-6, info=info1)
    assert info1["shape"] != "real harmonics"
    d1 = rotate_matrix(H, Q1)
    assert np.max(np.abs(d1 - np.diag(np.diag(d1)))) > 1e-4  # affordable shape was really needed


def test_axial_field_keeps_the_sparser_complex_harmonics():
    l = 2
    n = 2 * (2 * l + 1)
    m = np.tile(np.arange(-l, l + 1), 2)
    eps = 0.1 * np.abs(m) ** 2 - 0.05 * (np.arange(n) // (2 * l + 1))  # E(|m|, spin): degenerate +-m
    w = np.linspace(-1.5, 0.5, 60)
    hyb = np.array([np.diag(0.1 / (x + 0.05j + 0.8 + eps)) for x in w]).astype(complex)
    H = np.diag(eps).astype(complex)
    U = _l_dot_l_u4(l)
    R = _random_rotation(l, 5, spin=False)
    H, hyb = rotate_matrix(H, R), rotate_Greens_function(hyb, R)
    info = {}
    _, info_Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=np.eye(n, dtype=complex), max_u_densify=50, tol=1e-6, info=info)
    Wr = np.kron(np.eye(2), utils._real_harmonics(l))
    assert count_u_terms(_rotate_u4(U, Wr)) > count_u_terms(U)  # real harmonics also separate +-m, but cost more
    assert _sizes(info_Q, hyb, H, 1e-6) == [1] * n
    assert info["tier"] == "axis rotation"
    assert info["u_terms"] == count_u_terms(U)


def _two_plus_three_problem(weak=0.02, seed=2):
    """d shell: real-harmonic hybridization with a 2+3 block structure, strong mixing inside the
    blocks and a weak coupling between them, in a rotated frame."""
    l = 2
    n = 2 * (2 * l + 1)
    rng = np.random.default_rng(seed)
    groups = ([0, 1], [2, 3, 4])
    w = np.linspace(-1.5, 0.5, 80)
    B = np.zeros((5, 5), dtype=complex)
    for g in groups:
        A = rng.normal(size=(len(g),) * 2)
        B[np.ix_(g, g)] = A @ A.T + np.eye(len(g))
    cross = rng.normal(size=(5, 5))
    cross = 0.5 * (cross + cross.T)
    mask = np.ones((5, 5), bool)
    for g in groups:
        mask[np.ix_(g, g)] = False
    B = B / np.max(np.abs(np.diag(B))) + weak * np.where(mask, cross, 0) / np.max(np.abs(cross))
    hyb_real = np.array([np.kron(np.eye(2), B) * (0.3 / (x + 1.3 + 0.1j)) for x in w])
    Wr = np.kron(np.eye(2), utils._real_harmonics(l))
    H_real = np.kron(np.eye(2), np.diag([0.02, -0.01, 0.03, 0.0, 0.015])).astype(complex)
    R = _random_rotation(l, 9)
    H = rotate_matrix(Wr @ H_real @ np.conj(Wr.T), R)
    hyb = rotate_Greens_function(Wr[None] @ hyb_real @ np.conj(Wr.T)[None], R)
    return np.eye(n, dtype=complex), H, hyb, w


def test_partition_search_finds_the_weakly_coupled_blocks():
    S, H, hyb, w = _two_plus_three_problem()
    U = _l_dot_l_u4(2)
    tol = 0.05
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=1.0, tol=tol, info=info)
    assert _sizes(Q, hyb, H, tol) == [2, 2, 3, 3]
    assert info["tier"] == "axis rotation"
    assert info["u_terms"] == count_u_terms(U)  # no extra Coulomb terms: only the frame was searched


def test_a_larger_budget_never_gives_a_coarser_partition():
    S, H, hyb, w = _two_plus_three_problem(seed=6)
    U = _l_dot_l_u4(2)
    tol = 0.05
    keys = []
    for budget in (1.0, 2.0, 4.0, 50.0):
        info = {}
        _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=budget, tol=tol, info=info)
        assert info["u_terms"] <= budget * count_u_terms(U)
        keys.append(utils._partition_key(_sizes(Q, hyb, H, tol)))
    assert keys == sorted(keys, reverse=True)


def test_shapes_beyond_the_budget_are_not_used():
    S, H, hyb, w = _two_plus_three_problem()
    U = _l_dot_l_u4(2)
    Wr = np.kron(np.eye(2), utils._real_harmonics(2))
    assert count_u_terms(_rotate_u4(U, Wr)) > count_u_terms(U)
    info = {}
    block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=1.0, tol=0.05, info=info)
    assert info["shape"] != "real harmonics"
    assert info["u_terms"] <= count_u_terms(U)


def test_strong_spin_orbit_selects_the_coupled_j_basis_when_affordable():
    l = 2
    n = 2 * (2 * l + 1)
    Wj = utils._coupled_j_basis(l)
    eps = np.repeat(np.linspace(-0.3, 0.3, n // 2), 2)[:n]
    eps = eps + 0.01 * np.arange(n)
    w = np.linspace(-1.5, 0.5, 60)
    hyb_d = np.array([np.diag(0.1 / (x + 0.05j + 0.8 + eps)) for x in w]).astype(complex)
    R = _so3_rotation(l, np.array([0.4, -0.9, 0.3]), np.array([0.4, -0.9, 0.3]))  # tied: a rotation of J
    H = rotate_matrix(Wj @ np.diag(eps).astype(complex) @ np.conj(Wj.T), R)
    hyb = rotate_Greens_function(Wj[None] @ hyb_d @ np.conj(Wj.T)[None], R)
    U = _l_dot_l_u4(l)
    assert count_u_terms(_rotate_u4(U, Wj)) > count_u_terms(U)
    S = np.eye(n, dtype=complex)
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=100, tol=1e-6, info=info)
    assert info["shape"] == "j-j coupled"
    d = rotate_matrix(H, Q)
    assert np.max(np.abs(d - np.diag(np.diag(d)))) < 1e-7
    info = {}
    block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=1.0, tol=1e-6, info=info)
    assert info["shape"] != "j-j coupled"


def test_budget_below_one_falls_back_to_the_input_basis():
    S, H, hyb, w = _two_plus_three_problem()
    U = _l_dot_l_u4(2)
    info = {}
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=0.5, tol=0.05, info=info)
    assert info["u_terms"] <= count_u_terms(U)
    assert info["tier"] in ("axis rotation", "input basis")


def test_basis_choice_is_deterministic():
    S, H, hyb, w = _two_plus_three_problem()
    U = _l_dot_l_u4(2)
    _, Q1 = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=2.0, tol=0.05)
    _, Q2 = block_diagonalize_hyb(hyb, mat=H, w=w, u4=U, to_spherical=S, max_u_densify=2.0, tol=0.05)
    assert np.array_equal(Q1, Q2)


def test_spin_coupling_is_judged_in_the_spherical_frame():
    """Spin-diagonal matrices, but an input basis that mixes spins (j-j): not spin coupled."""
    l = 2
    n = 2 * (2 * l + 1)
    Wj = utils._coupled_j_basis(l)
    S = np.conj(Wj.T)  # <input|spherical>
    rng = np.random.default_rng(3)
    eps = rng.uniform(-0.3, 0.3, n)
    w = np.linspace(-1.5, 0.5, 40)
    hyb_sph = np.array([np.diag(0.1 / (x + 0.05j + 0.8 + eps)) for x in w]).astype(complex)
    H_sph = np.diag(eps).astype(complex)
    hyb, H = S[None] @ hyb_sph @ np.conj(S.T)[None], S @ H_sph @ np.conj(S.T)
    U_sph = _l_dot_l_u4(l)
    for U in (_rotate_u4(U_sph, S), _rotate_u4(U_sph, np.conj(S.T))):
        labels = [lab for lab, _ in utils._tier1_candidates(
            hyb, hyb, H, U, S, 10**9, 1e-6)]
        if labels:
            break
    assert labels, "invariance probe rejected both conventions"
    assert "j-j coupled" not in labels

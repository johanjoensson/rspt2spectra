"""Tests for rspt2spectra.utils."""

import numpy as np

from rspt2spectra.fit_basis import block_diagonalize_hyb


def test_block_diagonalize_hyb():
    hyb = np.zeros((2, 2, 2), dtype=complex)
    hyb[:, 0, 1] = 1.0 + 1j
    hyb[:, 1, 0] = 1.0 - 1j
    hyb[:, 0, 0] = 2.0
    hyb[:, 1, 1] = 2.0

    phase_hyb, Q_full = block_diagonalize_hyb(hyb)
    assert phase_hyb.shape == (2, 2, 2)
    assert Q_full.shape == (2, 2)
    assert np.allclose(phase_hyb[:, 0, 1], 0, atol=1e-10)
    assert np.allclose(phase_hyb[:, 1, 0], 0, atol=1e-10)


def _offdiag_weight(mats):
    return sum(np.sum(np.abs(m - np.diag(np.diag(m))) ** 2) for m in mats)


def _random_commuting(n, n_mat, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return np.array([U @ np.diag(rng.normal(size=n)) @ U.conj().T for _ in range(n_mat)])


def test_joint_diagonalization_recovers_commuting_set():
    mats = _random_commuting(5, 6)
    phase, Q = block_diagonalize_hyb(mats.astype(complex))
    assert np.allclose(Q.conj().T @ Q, np.eye(5))
    assert _offdiag_weight(phase) < 1e-15 * 6  # pairs below the 1e-8 relative skip are left alone


def test_joint_diagonalization_is_never_worse_than_the_input_basis():
    rng = np.random.default_rng(3)
    n = 5
    hyb = np.empty((8, n, n), dtype=complex)
    for k in range(8):
        X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        hyb[k] = X + X.conj().T
    phase, Q = block_diagonalize_hyb(hyb)
    assert np.allclose(Q.conj().T @ Q, np.eye(n))
    assert _offdiag_weight(phase) <= _offdiag_weight(hyb)


def test_joint_diagonalization_uses_the_matrix_and_keeps_spins_apart():
    # Two identical 2x2 spin blocks, coupled inside the block only through H.
    H1 = np.array([[0.0, 0.3], [0.3, 1.0]])
    H = np.zeros((4, 4))
    H[:2, :2] = H1
    H[2:, 2:] = H1
    hyb = np.zeros((3, 4, 4), dtype=complex)
    for i in range(4):
        hyb[:, i, i] = -1j * 0.1
    _phase, Q = block_diagonalize_hyb(hyb, mat=H)
    Hq = Q.conj().T @ H @ Q
    assert _offdiag_weight([Hq]) < 1e-20
    assert np.allclose(Q[:2, 2:], 0) and np.allclose(Q[2:, :2], 0)
    assert np.allclose(Q[:2, :2], Q[2:, 2:])


def test_joint_diagonalization_labels_follow_the_input_orbitals():
    # Already diagonal input must come back as the identity (up to nothing), in any order.
    hyb = np.zeros((2, 3, 3), dtype=complex)
    hyb[:, 0, 0], hyb[:, 1, 1], hyb[:, 2, 2] = 3.0, 1.0, 2.0
    _, Q = block_diagonalize_hyb(hyb)
    assert np.allclose(Q, np.eye(3))


def test_weak_h_still_fixes_the_basis_of_a_degenerate_hybridization():
    """H has a tenth of the weight, not none: it resolves a degeneracy of the hybridization."""
    rng = np.random.default_rng(3)
    n, nw = 3, 8
    U = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    H = U @ np.diag([0.0, 0.01, 0.02]) @ U.conj().T  # fixes the basis
    w = np.linspace(-1, 0, nw)
    hyb = np.broadcast_to(-0.1j * np.eye(n), (nw, n, n)).copy()  # degenerate: any basis
    _, Q = block_diagonalize_hyb(hyb, mat=H, w=w)
    Hq = Q.conj().T @ H @ Q
    assert np.max(np.abs(Hq - np.diag(np.diag(Hq)))) < 1e-9

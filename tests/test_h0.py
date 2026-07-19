"""Tests for the high-level h0 driver (prepare_hyb_fit / assemble_h0).

Uses an analytic star model with known poles, bypassing the stochastic
hybridization fit, so the assembled Hamiltonians can be checked exactly
against the analytic impurity Green's function
G0(z) = [z - H_loc - Delta(z)]^-1.
"""

import numpy as np
import pytest

from rspt2spectra.h0 import assemble_h0, flatten_star_levels, prepare_hyb_fit

# Two impurity orbitals with diagonal hybridization but coupled through the
# local Hamiltonian: the combined partition must merge them into one block.
H_LOC = np.array([[-0.5, 0.2], [0.2, 0.3]], dtype=complex)
# (energy, coupling row) of the analytic bath poles.
POLES = [
    (-2.0, np.array([0.5, 0.0], dtype=complex)),
    (-1.5, np.array([0.0, 0.4], dtype=complex)),
    (0.8, np.array([0.0, 0.2], dtype=complex)),
    (1.0, np.array([0.3, 0.0], dtype=complex)),
]
W = np.linspace(-4, 4, 401)
EIM = 0.1


def analytic_hyb(z):
    """Delta(z) with the analytic poles; z is a 1D complex array."""
    delta = np.zeros((len(z), 2, 2), dtype=complex)
    for e, v in POLES:
        delta += np.conj(v[None, :, None]) * v[None, None, :] / (z - e)[:, None, None]
    return delta


def analytic_g0(z):
    return np.linalg.inv(z[:, None, None] * np.eye(2)[None] - H_LOC[None] - analytic_hyb(z))


def prepared():
    hyb = analytic_hyb(W + 1j * EIM)
    return prepare_hyb_fit(hyb, H_LOC, tol=1e-6, verbose=False)


def exact_fit(Q, block_structure):
    """Express the analytic poles as a (flattened) star fit in the Q basis."""
    assert len(block_structure.inequivalent_blocks) == 1
    ebs = np.array([e for e, _ in POLES])
    vs = np.array([v @ Q for _, v in POLES]).reshape((len(POLES), 1, 2))
    order = np.argsort(ebs, kind="stable")
    ebs_flat, vs_flat = flatten_star_levels(ebs[order], vs[order])
    shifts = [np.zeros((2, 2), dtype=complex)]
    return [ebs_flat], [vs_flat], shifts


def test_combined_partition_merges_local_coupling():
    _, _, H_local_Q, block_structure = prepared()
    # Hybridization alone is diagonal, but H_loc couples the orbitals: the
    # union connectivity must put both orbitals in a single block.
    assert sorted(map(sorted, block_structure.blocks)) == [[0, 1]]
    # And the rotated local Hamiltonian must be block diagonal (trivially so
    # for a single block, but the guard in prepare_hyb_fit must not raise).
    assert H_local_Q.shape == (2, 2)


def test_star_geometry_reproduces_analytic_g0():
    Q, _, H_local_Q, block_structure = prepared()
    ebs, vs, shifts = exact_fit(Q, block_structure)
    H, H_star, imp, val, cond, _v_solver, _H_bath, H_imp = assemble_h0(
        ebs,
        vs,
        shifts,
        H_LOC,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry="star",
        w=W,
        eim=EIM,
        verbose=False,
    )
    assert H is H_star
    assert imp == [0, 1]
    # The returned effective impurity block is exactly the impurity sub-block of H.
    assert np.allclose(H_imp, H[:2, :2])
    # 3 poles below zero energy... 2 valence poles, 2 conduction poles
    assert len(val) == 2 and len(cond) == 2
    z = W[::10] + 1j * EIM
    G0 = np.linalg.inv(z[:, None, None] * np.eye(H.shape[0])[None] - H[None])[:, :2, :2]
    assert np.allclose(G0, analytic_g0(z), atol=1e-12)
    # Hermiticity of the assembled Hamiltonian.
    assert np.allclose(H, np.conj(H.T))


@pytest.mark.parametrize("geometry", ["chain", "haver"])
def test_chain_geometries_preserve_impurity_g0(geometry):
    Q, _, H_local_Q, block_structure = prepared()
    ebs, vs, shifts = exact_fit(Q, block_structure)
    H, H_star, _imp, _val, _cond, _v_solver, _H_bath, _H_imp = assemble_h0(
        ebs,
        vs,
        shifts,
        H_LOC,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry=geometry,
        w=W,
        eim=EIM,
        verbose=False,
    )
    assert H.shape == H_star.shape
    z = W[::10] + 1j * EIM
    G0 = np.linalg.inv(z[:, None, None] * np.eye(H.shape[0])[None] - H[None])[:, :2, :2]
    G0_star = np.linalg.inv(z[:, None, None] * np.eye(H_star.shape[0])[None] - H_star[None])[:, :2, :2]
    assert np.allclose(G0, G0_star, atol=1e-10)
    assert np.allclose(G0, analytic_g0(z), atol=1e-10)


def test_shift_moves_impurity_block():
    Q, _, H_local_Q, block_structure = prepared()
    ebs, vs, _ = exact_fit(Q, block_structure)
    shift = [0.1 * np.eye(2, dtype=complex)]
    H, *_ = assemble_h0(
        ebs,
        vs,
        shift,
        H_LOC,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry="star",
        w=W,
        eim=EIM,
        verbose=False,
    )
    H0, *_ = assemble_h0(
        ebs,
        vs,
        [np.zeros((2, 2), dtype=complex)],
        H_LOC,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry="star",
        w=W,
        eim=EIM,
        verbose=False,
    )
    # A constant shift C adds to the impurity block (and to the chain anchor):
    # Delta_fit = Delta_pole + C, so matching g0^-1 = z - H_imp - Delta needs
    # E_imp = H_imp + C. For the star geometry only the impurity block changes.
    assert np.allclose(H[:2, :2] - H0[:2, :2], 0.1 * np.eye(2))
    assert np.allclose(H[2:, 2:], H0[2:, 2:])

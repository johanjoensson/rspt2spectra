"""Neglecting weak couplings: prepare_hyb_fit and assemble_h0 hand over exact blocks."""

import numpy as np

from rspt2spectra.block_structure import get_blocks
from rspt2spectra.h0 import prepare_hyb_fit
from rspt2spectra.utils import block_diagonalize_hyb


def _system(coupling):
    w = np.linspace(-3, 3, 121)
    n = 4
    hyb = np.zeros((len(w), n, n), dtype=complex)
    for i, width in enumerate([0.30, 0.35, 0.10, 0.12]):
        hyb[:, i, i] = 0.1 / (w - 0.5 * i + 1j * width)
    # weak hybridization coupling between the two pairs
    hyb[:, 1, 2] = hyb[:, 2, 1] = coupling * hyb[:, 1, 1]
    H = np.diag([0.0, 0.2, 1.0, 1.4]).astype(complex)
    H[0, 1] = H[1, 0] = 0.15
    H[2, 3] = H[3, 2] = 0.15
    H[1, 2] = H[2, 1] = coupling
    return w, hyb, H


def test_weak_coupling_joins_blocks_by_default_and_is_neglected_above_tol():
    w, hyb, H = _system(1e-3)
    bs_strict = prepare_hyb_fit(hyb, H, tol=1e-6, verbose=False, w=w).block_structure
    assert len(bs_strict.blocks) == 1

    fb = prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w)
    Q, phase_hyb, H_Q, bs = fb.Q, fb.phase_hyb, fb.H_local_Q, fb.block_structure
    assert len(bs.blocks) == 2
    for blk_a, blk_b in [(bs.blocks[0], bs.blocks[1]), (bs.blocks[1], bs.blocks[0])]:
        assert np.all(H_Q[np.ix_(blk_a, blk_b)] == 0)
        assert np.all(phase_hyb[:, blk_a][:, :, blk_b] == 0)
    assert np.allclose(Q.conj().T @ Q, np.eye(4))
    # the two inequivalent pairs must not be merged just because tol is loose
    assert len(bs.inequivalent_blocks) == 2


# --- regressions from the adversarial review -------------------------------------------------


def _two_orbital_hyb(w, weight_occ, weight_unocc, coupling, eim=0.05):
    hyb = np.zeros((len(w), 2, 2), dtype=complex)
    hyb[:, 0, 0] = weight_occ / (w + 1j * eim + 0.5) + weight_unocc / (w + 1j * eim - 0.9)
    hyb[:, 1, 1] = 0.5 * weight_occ / (w + 1j * eim + 0.3)
    hyb[:, 0, 1] = hyb[:, 1, 0] = coupling * weight_occ / (w + 1j * eim + 0.4)
    return hyb


def test_strong_peak_outside_the_window_does_not_split_a_block_after_the_fact():
    # The partition is decided on the fit window; a large conduction peak outside it raises the
    # scale of the whole mesh and must not make the equivalence pass disagree with the window.
    w = np.linspace(-2, 2, 400)
    hyb = _two_orbital_hyb(w, 0.01, 10.0, 5.0)
    bs = prepare_hyb_fit(hyb, np.diag([0.0, 0.1]), tol=0.05, w=w, x_lim=(-1, 0), verbose=False).block_structure
    assert bs.blocks == [[0, 1]]


def test_degenerate_crystal_field_noise_does_not_merge_blocks():
    rng = np.random.default_rng(1)
    noise = 1e-12 * rng.standard_normal((4, 4))
    H = -0.2 * np.eye(4) + (noise + noise.T) / 2
    G = np.zeros((1, 4, 4), dtype=complex)
    G[0] = np.diag([1.0, 2.0, 3.0, 4.0])
    assert get_blocks(G, H, tol=1e-6) == [[0], [1], [2], [3]]


def test_default_tol_keeps_the_old_absolute_threshold():
    H = np.diag([0.0, 0.02, 0.04])
    H[0, 1] = H[1, 0] = 5e-7  # below the old absolute 1e-6
    assert get_blocks(None, H, tol=1e-6) == [[0], [1], [2]]
    H[0, 1] = H[1, 0] = 2e-6
    assert get_blocks(None, H, tol=1e-6) == [[0, 1], [2]]


def test_joint_diagonalization_labels_survive_roundoff_ties():
    # Two degenerate, coupled orbitals mix at 45 degrees; a 1e-9 perturbation must not swap the
    # labels of the rotated orbitals between two calls (the warm start follows the labels).
    w = np.linspace(-1, 1, 20)
    base = np.zeros((20, 2, 2), dtype=complex)
    base[:, 0, 0] = base[:, 1, 1] = 0.3 / (w + 0.05j + 0.2)
    base[:, 0, 1] = base[:, 1, 0] = 0.1 / (w + 0.05j + 0.2)
    pert = base.copy()
    pert[:, 0, 0] += 1e-9
    _, Q1 = block_diagonalize_hyb(base)
    _, Q2 = block_diagonalize_hyb(pert)
    assert np.allclose(np.abs(Q1), np.abs(Q2), atol=1e-6)
    assert np.allclose(Q1, Q2, atol=1e-6)


def test_split_margin_brackets_the_coupling():
    """A single H coupling of relative size c splits for tol > c and merges below it."""
    from rspt2spectra.h0 import split_margin

    n = 4
    H = np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex)
    H[0, 2] = H[2, 0] = 0.02
    hyb = np.zeros((3, n, n), dtype=complex)
    for k in range(n):
        hyb[:, k, k] = -0.1j
    lo, hi = split_margin(hyb, H, tol=0.5)
    assert lo is not None and 0.0 < lo < 0.5
    assert hi is None or hi > 0.5
    lo2, hi2 = split_margin(hyb, H, tol=1e-3)
    assert lo2 is None or lo2 < 1e-3
    assert hi2 is not None and hi2 < 0.5
    # the two windows are adjacent but different partitions
    assert abs(hi2 - lo) / lo < 1e-6

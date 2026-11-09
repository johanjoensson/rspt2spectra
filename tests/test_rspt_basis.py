import itertools

import numpy as np
import pytest

from rspt2spectra.rspt_basis import cluster_rotation, projection_from_files, shell_rotation


def _is_unitary(T):
    return T.shape[0] == T.shape[1] and np.allclose(T @ T.conj().T, np.eye(T.shape[0]), atol=1e-12)


@pytest.mark.parametrize("shell", [(0, 1), (1, 1), (2, 3), (2, 4), (2, 7), (3, 7), (3, 0), (2, 0)])
def test_full_orbital_sets_are_unitary(shell):
    assert _is_unitary(cluster_rotation([shell]))


@pytest.mark.parametrize("tag", [3, 4, 5, 6, 7])
def test_d_shell_tags_decode_sequentially_like_rspt(tag):
    # RSPt subtracts 2 then tests >= 1 for l = 2, so every tag >= 3 means Eg + T2g.
    assert np.allclose(shell_rotation(2, tag), shell_rotation(2, 3))


def test_d_shell_cubic_rotation_gives_the_spherical_corner_element():
    eg, t2g = 0.6, 0.4
    h_cubic = np.diag([eg, eg, t2g, t2g, t2g]).astype(complex)
    T = cluster_rotation([(2, 3)], spin_resolved=False)
    h = T @ h_cubic @ T.conj().T
    assert np.allclose(np.diag(h).real, [(eg + t2g) / 2, t2g, eg, t2g, (eg + t2g) / 2])
    assert h[0, 4] == pytest.approx((eg - t2g) / 2)


def test_multi_shell_layout_is_block_diagonal_with_shell_sizes():
    T = cluster_rotation([(3, 7), (2, 3), (1, 0), (0, 0)])
    assert _is_unitary(T)
    assert T.shape == (32, 32)
    edges = np.cumsum([0, 14, 10, 6, 2])
    mask = np.zeros(T.shape, dtype=bool)
    for a, b in itertools.pairwise(edges):
        mask[a:b, a:b] = True
    assert np.all(T[~mask] == 0)


def test_spin_blocks_are_down_first_and_identical():
    P = shell_rotation(2, 3)
    assert np.allclose(P[:5, :5], P[5:, 5:])
    assert np.all(P[:5, 5:] == 0)


@pytest.mark.parametrize("shell", [(2, 1), (2, 2), (3, 1)])
def test_partial_orbital_sets_are_refused(shell):
    with pytest.raises(ValueError, match="partial"):
        cluster_rotation([shell])


@pytest.mark.parametrize("tag", [8, 10, 11, -1, 12])
def test_unsupported_tags_raise(tag):
    with pytest.raises(NotImplementedError):
        shell_rotation(2, tag)


def _ls_operator(l):
    """L.S in the spherical, down-first basis (rows m = -l..l, then the same for spin up)."""
    n = 2 * l + 1
    m = np.arange(-l, l + 1)
    lz = np.diag(m).astype(complex)
    lp = np.zeros((n, n), dtype=complex)
    for k in range(n - 1):  # raises m = m_k to m_k + 1
        lp[k + 1, k] = np.sqrt(l * (l + 1) - m[k] * (m[k] + 1))
    lm = lp.conj().T
    ls = np.zeros((2 * n, 2 * n), dtype=complex)
    ls[:n, :n] = -0.5 * lz
    ls[n:, n:] = 0.5 * lz
    ls[n:, :n] = 0.5 * lp  # S+ L-  acting on down; (L+S- + L-S+)/2 pieces below
    ls[:n, n:] = 0.5 * lm
    ls[n:, :n] = 0.5 * lm
    ls[:n, n:] = 0.5 * lp
    return ls


@pytest.mark.parametrize("l", [1, 2, 3])
def test_jj_basis_diagonalizes_spin_orbit_coupling(l):
    T = cluster_rotation([(l, 9)])
    assert _is_unitary(T)
    ls = T.conj().T @ _ls_operator(l) @ T
    assert np.allclose(ls, np.diag(np.diag(ls)), atol=1e-12)
    expected = [l / 2] * (2 * l + 2) + [-(l + 1) / 2] * (2 * l)
    assert sorted(np.diag(ls).real) == pytest.approx(sorted(expected))


def _entries(T, rows, cols, col_offset=0):
    return [(r + 1, c + 1 - col_offset, complex(T[r, c])) for r in rows for c in cols if T[r, c] != 0]


def test_projection_from_files_spin_polarized_round_trip():
    T = cluster_rotation([(2, 3)])
    entries = _entries(T, range(10), range(10))
    assert np.allclose(projection_from_files([(2, 0)], [([10], entries)]), T)


def test_projection_from_files_spin_unpolarized_eg_t2g_files():
    full = cluster_rotation([(2, 3)])
    # Spin-down spatial columns only: eg = 0, 1; t2g = 2, 3, 4. Rows 1..5 (down block).
    spatial = full[:5, :5]
    eg = _entries(spatial, range(5), range(2))
    t2g = _entries(spatial, range(5), range(2, 5), col_offset=2)
    T = projection_from_files([(2, 0)], [([4], eg), ([6], t2g)])
    # Column layout: eg_dn, eg_up, t2g_dn, t2g_up.
    expected = np.hstack([full[:, [0, 1]], full[:, [5, 6]], full[:, [2, 3, 4]], full[:, [7, 8, 9]]])
    assert T.shape == (10, 10)
    assert np.allclose(T, expected)
    assert _is_unitary(T)


def test_projection_from_files_spin_unpolarized_two_shells_land_in_their_own_up_blocks():
    # d then p, identity projection; pre-basis rows 1..5 (d) and 6..8 (p), all spin-down columns.
    entries = [(k + 1, k + 1, 1 + 0j) for k in range(8)]
    T = projection_from_files([(2, 0), (1, 0)], [([16], entries)])
    assert T.shape == (16, 16)
    d_dn, d_up, p_dn, p_up = (0, 5), (5, 10), (10, 13), (13, 16)
    # spin-down columns 0..7 map to d_dn then p_dn rows, up columns 8..15 to d_up then p_up rows
    expected = np.zeros((16, 16), dtype=complex)
    for k in range(5):
        expected[d_dn[0] + k, k] = 1
        expected[d_up[0] + k, 8 + k] = 1
    for k in range(3):
        expected[p_dn[0] + k, 5 + k] = 1
        expected[p_up[0] + k, 13 + k] = 1
    assert np.allclose(T, expected)


def test_projection_from_files_rejects_out_of_range_rows():
    with pytest.raises(ValueError, match="row index"):
        projection_from_files([(2, 0)], [([10], [(11, 1, 1 + 0j)])])

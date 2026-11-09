"""RSPt's correlated-orbital basis, expressed as a rotation from spherical harmonics.

The basis index (5th integer of each cluster site line in ``green.inp``) only selects a
set of projection columns, ``proj(m, k)``, mapping the correlated orbitals RSPt prints its
local Hamiltonian and hybridization in onto complex spherical harmonics. It says nothing
about the crystal's point group. This module mirrors ``lda_mlmsatomicqn`` in RSPt's
``green_trunk_interface.F90`` so that ``H_sph = T @ H_corr @ T^dagger`` can be formed for any
supported index.

Custom projectors (``Irr`` tokens plus ``proj-<cluster>-<token>.inp`` files) replace the basis
index altogether; ``projection_from_files`` builds the rotation from such files.

Two structural facts are relied upon (both follow from ``lda_mlmsatomicqn`` and were checked
against a 32x32 ``Local hamiltonian`` of ``f7 d3 p0 s0`` clusters):

* Shells appear in ``green.inp`` order (shell-major), and inside every spin-resolved shell the
  spin-down block comes first, then spin-up.
* Within a spin block the spherical rows run over ``m = -l..l``.
"""

import numpy as np
from scipy.linalg import block_diag

_S2 = 1 / np.sqrt(2)
_C5 = np.sqrt(5) / 4
_C3 = np.sqrt(3) / 4


def _d_columns(tag):
    cols = []
    if tag >= 2:  # Eg
        cols.append([0, 0, 1, 0, 0])  # dz2
        cols.append([_S2, 0, 0, 0, _S2])  # dx2-y2
        tag -= 2
    if tag >= 1:  # T2g
        cols.append([0, 1j * _S2, 0, 1j * _S2, 0])  # dyz
        cols.append([0, _S2, 0, -_S2, 0])  # dxz
        cols.append([1j * _S2, 0, 0, 0, -1j * _S2])  # dxy
    return cols


def _f_columns(tag):
    cols = []
    if tag >= 4:  # A2u
        cols.append([0, 1j * _S2, 0, 0, 0, -1j * _S2, 0])
        tag -= 4
    if tag >= 2:  # T1u
        cols.append([_C5, 0, -_C3, 0, _C3, 0, -_C5])
        cols.append([-1j * _C5, 0, -1j * _C3, 0, -1j * _C3, 0, -1j * _C5])
        cols.append([0, 0, 0, 1, 0, 0, 0])
        tag -= 2
    if tag >= 1:  # T2u
        cols.append([-_C3, 0, -_C5, 0, _C5, 0, _C3])
        cols.append([-1j * _C3, 0, 1j * _C5, 0, 1j * _C5, 0, -1j * _C3])
        cols.append([0, _S2, 0, 0, 0, _S2, 0])
    return cols


def _real_harmonic_columns(l, tag):
    """Spatial projection columns for tags 1..7, decoded sequentially as RSPt does."""
    if l == 0:
        return [[1]]
    if l == 1:
        return [[1j * _S2, 0, 1j * _S2], [_S2, 0, -_S2], [0, 1, 0]]
    if l == 2:
        return _d_columns(tag)
    if l == 3:
        return _f_columns(tag)
    raise NotImplementedError(f"RSPt basis index {tag} for l={l} is not supported (RSPt defines it for l <= 3).")


def _jj_rotation(l):
    """Spin-resolved JJ projection (tag 9): columns ordered j=l-1/2 then j=l+1/2, mj ascending."""
    n = 2 * l + 1
    if l == 0:
        return np.eye(2, dtype=complex)
    cols = []

    def column(j_minus, mj):
        col = np.zeros(2 * n, dtype=complex)
        m_down = mj + 0.5
        m_up = mj - 0.5
        if j_minus:
            down = -np.sqrt((l + mj + 0.5) / n)
            up = np.sqrt((l - mj + 0.5) / n)
        else:
            down = np.sqrt((l - mj + 0.5) / n)
            up = np.sqrt((l + mj + 0.5) / n)
        if abs(m_down) <= l:
            col[round(m_down) + l] = down
        if abs(m_up) <= l:
            col[n + round(m_up) + l] = up
        return col

    for k in range(2 * l):
        cols.append(column(True, -(l - 0.5) + k))
    for k in range(2 * l + 2):
        cols.append(column(False, -(l + 0.5) + k))
    return np.array(cols).T


def shell_rotation(l, basis_tag, spin_resolved=True):
    """Projection from one shell's correlated orbitals to complex spherical harmonics.

    Parameters
    ----------
    l : int
        Angular momentum of the shell.
    basis_tag : int
        RSPt basis index of the shell. 0 is the spherical basis itself, 1..7 are the real
        (cubic) harmonic sets, 9 is the JJ basis.
    spin_resolved : bool, default True
        Whether the shell carries a spin-down block followed by a spin-up block.

    Returns
    -------
    np.ndarray
        Matrix ``P`` of shape ``(n_spherical, n_correlated)`` so that
        ``H_sph = P @ H_corr @ P^dagger``. It is not square when the index selects only a
        subset of the shell's orbitals.

    Raises
    ------
    NotImplementedError
        For indices RSPt defines but this module does not: the rotated JJ bases (10, 11),
        and unknown values (including negative ones, which RSPt itself rejects).
    """
    n = 2 * l + 1
    n_sph = 2 * n if spin_resolved else n
    if basis_tag == 0 or l < 0:
        return np.eye(n_sph, dtype=complex)
    if 1 <= basis_tag <= 7:
        cols = _real_harmonic_columns(l, basis_tag)
        spatial = np.array(cols, dtype=complex).T
        return block_diag(spatial, spatial) if spin_resolved else spatial
    if basis_tag == 9:
        if not spin_resolved:
            raise NotImplementedError("The JJ basis (index 9) needs a spin-resolved shell.")
        return _jj_rotation(l)
    if basis_tag in (10, 11):
        raise NotImplementedError(
            f"RSPt basis index {basis_tag} (JJ basis rotated onto the "
            f"{'x' if basis_tag == 10 else 'y'} axis) is not supported."
        )
    if basis_tag < 0:
        raise NotImplementedError(
            f"RSPt basis index {basis_tag} is invalid: RSPt rejects negative indices. Custom "
            "projectors are declared with Irr tokens and proj-*.inp files instead."
        )
    raise NotImplementedError(f"RSPt basis index {basis_tag} is not supported.")


def cluster_rotation(shells, spin_resolved=True):
    """Block-diagonal projection for a whole cluster.

    Parameters
    ----------
    shells : list[tuple[int, int]]
        ``(l, basis_tag)`` per shell, in ``green.inp`` order.
    spin_resolved : bool, default True
        Whether every shell carries spin-down and spin-up blocks.

    Returns
    -------
    np.ndarray
        Square unitary ``T`` with ``H_sph = T @ H_corr @ T^dagger``.

    Raises
    ------
    ValueError
        If a basis index selects only part of a shell, which has no spherical representation
        of the full shell.
    """
    blocks = []
    for l, tag in shells:
        p = shell_rotation(l, tag, spin_resolved)
        if p.shape[0] != p.shape[1]:
            raise ValueError(
                f"Shell l={l} with basis index {tag} selects {p.shape[1]} of {p.shape[0]} "
                "orbitals; a partial orbital set cannot be rotated to spherical harmonics."
            )
        blocks.append(p)
    if not blocks:
        return np.zeros((0, 0), dtype=complex)
    return block_diag(*blocks)


def projection_from_files(shells, files):
    """Rotation from RSPt custom projector files (``Irr`` tokens) to spherical harmonics.

    Parameters
    ----------
    shells : list[tuple[int, int]]
        ``(l, basis_tag)`` per shell, in ``green.inp`` order. Only ``l`` is used.
    files : list[tuple[list[int], list[tuple[int, int, complex]]]]
        ``(set_sizes, entries)`` per file, as returned by ``read_projection_file``, in
        ``Irr`` token order.

    Returns
    -------
    np.ndarray
        Matrix ``T`` of shape ``(2 * sum(2l+1), msize)`` with ``H_sph = T @ H_corr @ T^dagger``,
        in the spherical layout (shell-major, spin down then spin up).

    Raises
    ------
    ValueError
        If a row index lies outside the cluster's pre-basis.
    """
    sizes_l = [2 * l + 1 for l, _ in shells]
    pre1 = sum(sizes_l)
    n_sph = 2 * pre1
    msize = sum(sum(sizes) for sizes, _ in files)
    nspmat = 2 if any(i > pre1 for _, entries in files for i, _, _ in entries) else 1
    premsize = nspmat * pre1

    P = np.zeros((premsize, msize), dtype=complex)
    sets = []  # (column offset, size)
    offset = 0
    for set_sizes, entries in files:
        for i, j, value in entries:
            if not 1 <= i <= premsize:
                raise ValueError(f"projection row index {i} outside the cluster's {premsize} pre-basis orbitals")
            P[i - 1, offset + j - 1] = value
        for size in set_sizes:
            sets.append((offset, size))
            offset += size
    if nspmat == 2:
        return P

    down_rows, up_rows = [], []
    cum = 0
    for n in sizes_l:
        down_rows += [2 * cum + m for m in range(n)]
        up_rows += [2 * cum + n + m for m in range(n)]
        cum += n
    T = np.zeros((n_sph, msize), dtype=complex)
    for c, size in sets:
        half = size // 2
        T[down_rows, c : c + half] = P[:, c : c + half]
        T[up_rows, c + half : c + size] = P[:, c : c + half]
    return T

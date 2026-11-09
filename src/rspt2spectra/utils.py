"""Matrix utilities: rotations, pretty-printing, and block diagonalization."""

from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment


def rotate_matrix(M, T):
    r"""Rotate the matrix M as :math:`M' = T^{\dagger} M T`.

    Parameters
    ----------
    M : NDArray - Matrix to rotate
    T : NDArray or dict - Rotation matrix to use, or dict of rotation matrices for blocks.

    Returns
    -------
    M' : NDArray - The rotated matrix
    """
    if isinstance(T, dict):
        sorted_keys = sorted(T.keys())
        T_matrix = block_diag(*(T[k] for k in sorted_keys))
        return np.conj(T_matrix.T) @ M @ T_matrix
    return np.conj(T.T) @ M @ T


def _float_field_width(values: np.ndarray, n_prec: int, force_sign: bool = False) -> int:
    """Field width for printing ``values`` with ``n_prec`` decimals, columns aligned.

    The width reserves room for the integer digits of the largest-magnitude entry, the
    decimal point, ``n_prec`` decimals, and a leading sign column when any value is
    negative (or always, if ``force_sign`` — e.g. the ``+`` flag used for imaginary parts).
    """
    flat = np.real(np.asarray(values)).ravel()
    max_abs = round(float(np.max(np.abs(flat))), n_prec) if flat.size else 0.0
    int_digits = max(1, int(np.floor(np.log10(max_abs))) + 1) if max_abs >= 1 else 1
    sign = 1 if force_sign or (flat.size and np.any(flat < 0)) else 0
    return sign + int_digits + 1 + n_prec


def vector_to_string(
    v: np.ndarray,
    realvalue: Optional[bool] = None,
    n_prec: int = 15,
    real_width: Optional[int] = None,
    imag_width: Optional[int] = None,
) -> str:
    """Pretty string representation of a (row) vector.

    Parameters
    ----------
    v : np.ndarray
        The vector to print.
    realvalue : bool, optional
        If True, only print the real parts. If None, it is automatically
        determined based on whether imaginary parts are close to zero.
    n_prec : int, default 15
        Number of decimal places to print.
    real_width, imag_width : int, optional
        Field widths for the real/imaginary parts. When None they are derived from ``v``;
        :func:`matrix_to_string` passes matrix-wide widths so every row lines up.

    Returns
    -------
    str
        Formatted string representation of the vector.
    """
    assert v.ndim == 1, f"{v.shape=}"
    if realvalue is None:
        realvalue = not np.any(np.abs(v.imag) > float(f"1e-{n_prec}"))
    if real_width is None:
        real_width = _float_field_width(v.real, n_prec)
    if realvalue:
        return " ".join(f"{np.real(el):>{real_width}.{n_prec}f}" for el in v)
    if imag_width is None:
        imag_width = _float_field_width(v.imag, n_prec, force_sign=True)
    return " ".join(f"{np.real(el):>{real_width}.{n_prec}f} {np.imag(el):>+{imag_width}.{n_prec}f}j" for el in v)


def matrix_to_string(m: np.ndarray, n_prec: int = 15, offset: int = 0) -> str:
    """Pretty string representation of a matrix.

    Columns are right-aligned to a common width derived from the whole matrix, so the
    entries form an aligned grid regardless of sign or magnitude.

    Parameters
    ----------
    m : np.ndarray
        The matrix to print.
    n_prec : int, default 15
        Number of decimal places to print.
    offset : int, default 0
        Indentation offset (number of spaces) for each line.

    Returns
    -------
    str
        Formatted string representation of the matrix.
    """
    realvalue = not np.any(np.abs(m.imag) > float(f"1e-{n_prec}"))
    real_width = _float_field_width(m.real, n_prec)
    imag_width = None if realvalue else _float_field_width(m.imag, n_prec, force_sign=True)
    pad = " " * offset
    return "\n".join(
        pad + vector_to_string(row, realvalue, n_prec, real_width=real_width, imag_width=imag_width) for row in m
    )


def matrix_print(m: np.ndarray, label: Optional[str] = None, n_prec: int = 15, **kwargs) -> None:
    """Pretty print the matrix m.

    Parameters
    ----------
    m : np.ndarray
        Matrix to print.
    label : str, optional
        Text to print above the matrix.
    n_prec : int, default 15
        Number of decimal places to print.
    **kwargs : dict
        Additional keyword arguments passed to the print function.
    """
    if label is not None:
        print(label)
    if len(m.shape) == 1:
        print(vector_to_string(m, n_prec=n_prec), **kwargs)
        return
    print(
        matrix_to_string(
            m,
            n_prec,
            4 + (len(label) - len(label.lstrip())) if label is not None else 0,
        ),
        **kwargs,
    )


def matrix_connectivity_print(m: np.ndarray, block_size: int = 1, label: Optional[str] = None) -> None:
    """Print the connections in a matrix.

    "O" signifies a (block-) diagonal term, "X" represents a (block-) offdiagonal term.

    Parameters
    ----------
    m : np.ndarray
        Matrix to print.
    block_size : int, default 1
        Size of blocks.
    label : str, optional
        Label to print above the matrix.
    """

    def get_char(el: float | complex, i: int, j: int) -> str:
        """Get the character representation for a matrix element.

        Parameters
        ----------
        el : float or complex
            The matrix element value to represent.
        i : int
            The block row index of the element.
        j : int
            The block column index of the element.

        Returns
        -------
        str
            "O" if diagonal, "X" if off-diagonal, or " " if zero.
        """
        if np.abs(el) <= np.finfo(float).eps:
            return " "
        if i == j:
            return "O"
        return "X"

    offset = 4 + (len(label) - len(label.lstrip())) if label is not None else 0
    if label is not None:
        print(label)
        print(" " * offset, end="")

    print(
        ("\n" + " " * offset).join(
            [
                " ".join([get_char(el, i // block_size, j // block_size) for j, el in enumerate(row)])
                for i, row in enumerate(m)
            ]
        )
    )


def partition(l: Iterable[Any], predicate: Callable[[Any], bool] = bool) -> Tuple[List[Any], List[Any]]:
    """Partition elements of an iterable into two lists based on a predicate.

    Parameters
    ----------
    l : Iterable
        The collection of elements to partition.
    predicate : callable, optional
        A function that takes an element and returns a boolean value.
        Defaults to `bool(a)`.

    Returns
    -------
    passed : list
        Elements for which the predicate returned True.
    failed : list
        Elements for which the predicate returned False.
    """
    passed = []
    failed = []
    for item in l:
        if predicate(item):
            passed.append(item)
        else:
            failed.append(item)
    return passed, failed


def rotate_Greens_function(G, T):
    r"""Rotate the Greens function G as :math:`G'(\omega) = T^{\dagger} G(\omega) T`.

    Parameters
    ----------
    G : NDArray - Greens function to rotate
    T : NDArray - Rotation matrix to use

    Returns
    -------
    G' : NDArray - The rotated Greens function
    """
    return np.conj(T.T)[np.newaxis, :, :] @ G @ T[np.newaxis, :, :]


def rotate_4index_U(U4, T):
    r"""Rotate the four-index tensor U4 as :math:`U4' = T^{\dagger}T^{\dagger} U4 TT`.

    Parameters
    ----------
    U4 : NDArray - Tensor function to rotate
    T : NDArray - Rotation matrix to use

    Returns
    -------
    U4' : NDArray - The rotated tensor function
    """
    return np.einsum("ij,kl, jlmo, mn, op", np.conj(T.T), np.conj(T.T), U4, T, T)


_JD_MAX_FREQ = 50
_JD_MAX_SWEEPS = 100
_JD_SKIP = 1e-8  # relative to the O(1) scaled matrices; couplings below are numerical noise
_JD_REL_DECREASE = 1e-10  # stop when a sweep removes less than this fraction of the off-diagonal weight
_JD_MAT_WEIGHT = 0.1  # weight of H relative to the whole hyb set: H breaks hyb degeneracies, hyb sets the basis
_JD_MAT_FLOOR = 1e-3  # Ry; crystal-field spread below this is noise when weighting H against hyb


def _joint_diagonalize(mats, max_sweeps=_JD_MAX_SWEEPS, skip=_JD_SKIP):
    """Approximately diagonalize a set of Hermitian matrices with one unitary.

    Complex Jacobi iteration after Cardoso and Souloumiac: every sweep rotates each
    orbital pair by the angle that minimises the summed squared off-diagonal weight of
    all matrices.  The start is the identity and every rotation is non-increasing, so
    the result is never less diagonal than the input.  A pair whose couplings vanish in
    every matrix is skipped, which keeps decoupled orbitals (spin up/down, separate
    blocks) from being mixed.

    Parameters
    ----------
    mats : ndarray of shape (n_mat, n, n)
        Hermitian matrices, already scaled to be of order one.
    max_sweeps : int
        Maximum number of Jacobi sweeps.
    skip : float
        Pairs with total coupling weight below this are left alone.

    Returns
    -------
    Q : ndarray of shape (n, n)
        Unitary with ``Q^dagger mats[m] Q`` as diagonal as possible.
    """
    A = np.array(mats, dtype=complex)
    n = A.shape[-1]
    Q = np.eye(n, dtype=complex)
    def off_weight():
        return float(np.sum(np.abs(A) ** 2) - np.sum(np.abs(np.einsum("mii->mi", A)) ** 2))

    previous = off_weight()
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                a_pq = A[:, p, q]
                if np.sum(np.abs(a_pq) ** 2) < skip**2:
                    continue
                h = np.stack(
                    [
                        (A[:, p, p] - A[:, q, q]).real,
                        2 * a_pq.real,
                        2 * a_pq.imag,
                    ]
                )
                # h[2] = i(a_qp - a_pq) = 2 Im(a_pq) for Hermitian a_qp = conj(a_pq)
                _, eigvecs = np.linalg.eigh(h @ h.T)
                x, y, z = eigvecs[:, -1]
                if x < 0:
                    x, y, z = -x, -y, -z
                r = np.sqrt(x * x + y * y + z * z)
                c = np.sqrt((x + r) / (2 * r))
                sn = (y - 1j * z) / np.sqrt(2 * r * (x + r)) if (x + r) > 0 else 0.0
                if abs(sn) < 1e-15:
                    continue
                rotated = True
                # A <- J^H A J and Q <- Q J, with J = [[c, -conj(sn)], [sn, c]] on (p, q)
                for M in (A, Q[None]):
                    cp = M[:, :, p].copy()
                    cq = M[:, :, q].copy()
                    M[:, :, p] = c * cp + sn * cq
                    M[:, :, q] = -np.conj(sn) * cp + c * cq
                rp = A[:, p, :].copy()
                rq = A[:, q, :].copy()
                A[:, p, :] = c * rp + np.conj(sn) * rq
                A[:, q, :] = -sn * rp + c * rq
        if not rotated:
            break
        current = off_weight()
        if previous - current <= _JD_REL_DECREASE * previous:
            break
        previous = current
    return Q


def block_diagonalize_hyb(hyb, mat=None, w=None, x_lim=None):
    """Rotate the hybridization function (and optionally a matrix) to a near-diagonal basis.

    The unitary ``Q`` jointly and approximately diagonalizes the Hermitian and
    anti-Hermitian parts of the hybridization function at (up to 50) frequencies in the
    fit window together with ``mat`` (typically the local Hamiltonian).  Off-diagonal
    couplings that no common rotation can remove stay in the result; they are small
    when the system is close to symmetric and are what the block threshold then
    decides about.

    Parameters
    ----------
    hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The hybridization matrix.
    mat : ndarray of shape (n_orb, n_orb), optional
        Matrix to diagonalize together with ``hyb``, e.g. the local Hamiltonian.
    w : ndarray of shape (n_freq,), optional
        Frequency mesh of ``hyb``; needed for ``x_lim``.
    x_lim : tuple of float, optional
        ``(lo, hi)`` window; only frequencies inside it are used.

    Returns
    -------
    phase_hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The hybridization function in the rotated basis.
    Q_full : ndarray of shape (n_orb, n_orb)
        The transformation matrix.  Column ``j`` is the orbital that overlaps most with
        the input orbital ``j``, so orbital labels are stable between calls.
    """
    n_freq, n_orb, _ = hyb.shape
    idx = np.arange(n_freq)
    if x_lim is not None:
        assert w is not None, "x_lim needs the frequency mesh w"
        in_window = np.flatnonzero((w >= x_lim[0]) & (w <= x_lim[1]))
        if len(in_window) > 0:
            idx = in_window
    if len(idx) > _JD_MAX_FREQ:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, _JD_MAX_FREQ).astype(int))]
    h = hyb[idx]
    herm = 0.5 * (h + np.conj(np.transpose(h, (0, 2, 1))))
    anti = (h - np.conj(np.transpose(h, (0, 2, 1)))) / 2j
    scale = max(float(np.max(np.sqrt(np.mean(np.abs(np.einsum("wii->wi", h)) ** 2, axis=0)))), 1e-10)
    mats = [herm / scale, anti / scale]
    mats = np.concatenate(mats, axis=0)
    if mat is not None:
        mat = 0.5 * (mat + np.conj(mat.T))
        ev = np.linalg.eigvalsh(mat)
        mat_scale = max(float(ev[-1] - ev[0]), _JD_MAT_FLOOR)
        # H gets a tenth of the weight of the whole set of frequencies: it breaks degeneracies of the
        # hybridization and tie-breaks the basis, but must not drag it away from the hybridization's.
        # Weight 0 would leave the rotation arbitrary wherever the hybridization is degenerate.
        weight = _JD_MAT_WEIGHT * np.sqrt(len(mats))
        mats = np.concatenate([mats, (weight * mat / mat_scale)[None]], axis=0)
    Q = _joint_diagonalize(mats)

    # Label stability: column j is the rotated orbital closest to input orbital j.
    # Overlaps are rounded so that round-off cannot decide a tie (a 45 degree mixing of two
    # degenerate orbitals has equal overlaps); ties then resolve by orbital index, always.
    overlap = np.round(np.abs(Q) ** 2, 6)
    rows, cols = linear_sum_assignment(-overlap)
    Q = Q[:, cols[np.argsort(rows)]]
    Q_full = np.zeros_like(Q)
    for column in range(n_orb):
        j = np.argmax(np.round(np.abs(Q[:, column]), 6))
        Q_full[:, column] = Q[:, column] * abs(Q[j, column]) / Q[j, column]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full

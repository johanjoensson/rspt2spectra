"""Matrix utilities: rotations, pretty-printing, and block diagonalization."""

from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np


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
        from scipy.linalg import (
            block_diag,
        )  # noqa: PLC0415 - avoid scipy cost for the common path

        sorted_keys = sorted(T.keys())
        T_matrix = block_diag(*(T[k] for k in sorted_keys))
        return np.conj(T_matrix.T) @ M @ T_matrix
    return np.conj(T.T) @ M @ T


def _float_field_width(
    values: np.ndarray, n_prec: int, force_sign: bool = False
) -> int:
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
    return " ".join(
        f"{np.real(el):>{real_width}.{n_prec}f} {np.imag(el):>+{imag_width}.{n_prec}f}j"
        for el in v
    )


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
    imag_width = (
        None if realvalue else _float_field_width(m.imag, n_prec, force_sign=True)
    )
    pad = " " * offset
    return "\n".join(
        pad
        + vector_to_string(
            row, realvalue, n_prec, real_width=real_width, imag_width=imag_width
        )
        for row in m
    )


def matrix_print(
    m: np.ndarray, label: Optional[str] = None, n_prec: int = 15, **kwargs
) -> None:
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


def matrix_connectivity_print(
    m: np.ndarray, block_size: int = 1, label: Optional[str] = None
) -> None:
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
                " ".join(
                    [
                        get_char(el, i // block_size, j // block_size)
                        for j, el in enumerate(row)
                    ]
                )
                for i, row in enumerate(m)
            ]
        )
    )


def partition(
    l: Iterable[Any], predicate: Callable[[Any], bool] = bool
) -> Tuple[List[Any], List[Any]]:
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


def block_diagonalize_hyb(hyb, tol=1e-6):
    """
    Block diagonalize the hybridization function matrix.

    Parameters
    ----------
    hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The hybridization matrix.
    tol : float, default 1e-6
        Tolerance threshold for the block partitioning.

    Returns
    -------
    phase_hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The block-diagonalized hybridization function.
    Q_full : ndarray of shape (n_orb, n_orb)
        The transformation matrix.
    """
    from rspt2spectra.block_structure import (
        get_blocks,
    )  # noqa: PLC0415 - avoid an import cycle

    hyb_herm = 1 / 2 * (hyb + np.conj(np.transpose(hyb, (0, 2, 1))))
    blocks = get_blocks(hyb_herm, tol=tol)
    Q_full = np.zeros((hyb.shape[1], hyb.shape[2]), dtype=complex)
    treated_orbitals = 0
    for block in blocks:
        block_idx = np.ix_(range(hyb.shape[0]), block, block)
        if len(block) == 1:
            Q_full[block_idx[1:], treated_orbitals] = 1
            treated_orbitals += 1
            continue
        block_hyb = hyb_herm[block_idx]
        upper_triangular_hyb = np.triu(hyb_herm, k=1)
        ind_max_offdiag = np.unravel_index(
            np.argmax(np.abs(upper_triangular_hyb)), upper_triangular_hyb.shape
        )
        eigvals, Q = np.linalg.eigh(block_hyb[ind_max_offdiag[0], :, :])
        sorted_indices = np.argsort(eigvals)
        Q = Q[:, sorted_indices]
        for column in range(Q.shape[1]):
            j = np.argmax(np.abs(Q[:, column]))
            Q_full[block, treated_orbitals + column] = (
                Q[:, column] * abs(Q[j, column]) / Q[j, column]
            )
        treated_orbitals += Q.shape[1]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full

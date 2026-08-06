"""
Write the non-interacting impurity Hamiltonian to text files read by impurityModel.

:func:`write_h0_file` produces the versioned, self-describing ``.h0`` format
(a JSON header plus ``i j re im`` terms) specified in impurityModel's
``doc/h0_file_format.md``. :func:`write_operator_to_file` writes the legacy
bare-integer form, which records no units, energy reference, basis or orbital
layout and is kept only so stored workloads stay readable.
"""

import json

import numpy as np

#: Version of the .h0 format this module writes.
H0_SPEC_VERSION = 1

#: Default relative threshold below which an element *pair* is dropped.
DEFAULT_DROP_TOLERANCE = 1e-12

#: Tolerance for the writer's own hermiticity check, relative to max|H|.
HERMITICITY_RTOL = 1e-10


def key_to_string(key):
    """Format an operator key ``((i, "c"), (j, "a"))`` as ``"  i   j"``."""
    (state1, _), (state2, _) = key
    return f"{state1:3d} {state2:3d}"


def value_to_string(value):
    """Format a complex amplitude as ``"re im"`` with 15 decimals."""
    return f"{value.real: .15f} {value.imag: .15f}"


def key_value_to_string(key, value):
    """Format one operator element as a full output line."""
    return key_to_string(key) + " " + value_to_string(value) + "\n"


def write_operator_to_file(operator, filename):
    """Write a sequence of operators to ``filename``, one block per operator.

    Parameters
    ----------
    operator : iterable of dict
        Operators in the ``{((i, "c"), (j, "a")): amplitude}`` format.
    filename : str
        Output file; overwritten if it exists. Operator blocks are separated
        by blank lines.
    """
    strings = []
    for op in operator:
        s = ""
        for key, value in op.items():
            s += key_value_to_string(key, value)
        strings.append(s)
    with open(filename, "w") as f:
        f.write("\n".join(strings))


def _hermitian_part(h_matrix):
    """Return ``(H + H^dagger)/2`` with a real diagonal, after checking it is a small change.

    ``assemble_h0`` builds the impurity block through a similarity transform, so the result
    is Hermitian only up to rounding and the two triangles differ in their last bits. Writing
    them independently makes hermiticity an accident: the consumer's check is exact, and
    filtering the triangles separately can even drop one half of a pair. Symmetrizing here
    makes the file Hermitian by construction -- complex addition commutes and ``*0.5`` is
    exact, so the result is bitwise symmetric.

    Parameters
    ----------
    h_matrix : numpy.ndarray
        Dense ``(n, n)`` Hamiltonian.

    Returns
    -------
    numpy.ndarray
        The Hermitian part.

    Raises
    ------
    ValueError
        If the input is not Hermitian to ``HERMITICITY_RTOL``, which would mean a real error
        upstream rather than accumulated rounding.
    """
    h = np.asarray(h_matrix, dtype=complex)
    scale = np.abs(h).max() if h.size else 0.0
    deviation = np.abs(h - h.conj().T).max() if h.size else 0.0
    if deviation > HERMITICITY_RTOL * max(scale, 1.0):
        raise ValueError(
            f"h0 is not Hermitian: max|H - H^dagger| = {deviation:.3e} exceeds "
            f"{HERMITICITY_RTOL:.1e} * max|H| = {HERMITICITY_RTOL * max(scale, 1.0):.3e}. "
            "That is too large to be rounding; check the assembly."
        )
    h = 0.5 * (h + h.conj().T)
    np.fill_diagonal(h, h.diagonal().real)
    return h


def write_h0_file(
    filename,
    h_matrix,
    *,
    impurity_orbitals,
    unit,
    energy_reference="fermi",
    drop_tolerance=DEFAULT_DROP_TOLERANCE,
    rot_to_spherical=None,
    **header_extra,
):
    """Write ``h_matrix`` in impurityModel's ``.h0`` format.

    The format is specified in impurityModel's ``doc/h0_file_format.md``; this writer and
    that reader are pinned against each other by a shared, hand-authored fixture rather than
    by a shared dependency.

    Amplitudes are written with ``repr``, which round-trips exactly. The fixed-point
    ``%.15f`` of :func:`write_operator_to_file` is an absolute 1e-15 grid and so is lossless
    only for ``|x| >= 8``; on real f-shell data it wrote genuine 1e-12 elements with four
    significant digits and flushed sub-1e-16 elements to zero outright.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output file.
    h_matrix : numpy.ndarray
        Dense ``(n, n)`` one-particle Hamiltonian, impurity block first.
    impurity_orbitals : dict
        ``{group_label: sequence_of_indices}``.
    unit : str
        Energy unit of ``h_matrix``: ``"Ry"`` for a default RSPt run, ``"eV"`` if already
        converted. Recorded, not converted -- this package works in its input's unit, and
        the consumer converts.
    energy_reference : str, default "fermi"
        ``"fermi"`` when the Fermi level has been subtracted, else ``"absolute"``.
    drop_tolerance : float, default 1e-12
        Elements whose ``(i, j)``/``(j, i)`` pair is below this times ``max|H|`` are omitted.
        Applied pairwise so the stored key sets stay symmetric.
    rot_to_spherical : numpy.ndarray, optional
        ``n_imp x n_imp`` rotation from the impurity basis to spherical harmonics.
    **header_extra
        Further header keys (``basis``, ``bath_geometry``, ``valence_bath``, ``producer``, ...).

    Raises
    ------
    ValueError
        If ``h_matrix`` holds a non-finite value, or is not Hermitian to tolerance.
    """
    h = np.asarray(h_matrix, dtype=complex)
    if not np.isfinite(h).all():
        raise ValueError("h0 holds non-finite values, which the .h0 format forbids")
    h = _hermitian_part(h)

    n_orb = h.shape[0]
    cutoff = drop_tolerance * (np.abs(h).max() if h.size else 0.0)

    header = {
        "version": H0_SPEC_VERSION,
        "required_features": ["unit", "energy_reference", "index_convention", "storage"],
        "unit": unit,
        "energy_reference": energy_reference,
        "n_orb": int(n_orb),
        "index_convention": "impurity-block-first",
        "storage": "full",
        "impurity_orbitals": {str(k): [int(o) for o in orbs] for k, orbs in impurity_orbitals.items()},
        "drop_tolerance": float(drop_tolerance),
    }
    if rot_to_spherical is not None:
        rot = np.asarray(rot_to_spherical, dtype=complex)
        header["rot_to_spherical"] = [[[float(v.real), float(v.imag)] for v in row] for row in rot]
    header.update({k: v for k, v in header_extra.items() if v is not None})

    lines = [f"# impurityModel-h0 v{H0_SPEC_VERSION}", json.dumps(header, allow_nan=False), "--"]
    for i in range(n_orb):
        for j in range(n_orb):
            if max(abs(h[i, j]), abs(h[j, i])) <= cutoff:
                continue
            # complex() first: repr of a numpy scalar is "np.float64(...)", not a number.
            value = complex(h[i, j])
            lines.append(f"{i} {j} {value.real!r} {value.imag!r}")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

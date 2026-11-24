"""
Write the non-interacting impurity Hamiltonian to text files read by impurityModel.

:func:`write_h0_file` produces the versioned, self-describing ``.h0`` format
(a JSON header plus ``i j re im`` terms) specified in impurityModel's
``doc/h0_file_format.md``.
"""

import json

import numpy as np

#: Version of the .h0 format this module writes.
H0_SPEC_VERSION = 1

#: Default relative threshold below which an element *pair* is dropped.
DEFAULT_DROP_TOLERANCE = 1e-12

#: Tolerance for the writer's own hermiticity check, relative to max|H|.
HERMITICITY_RTOL = 1e-10


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


_OPERATOR_MATRICES = ("sz", "s_plus", "lz", "l_plus", "s_axis", "dipole")


def _pairs(m):
    m = np.asarray(m, dtype=complex)
    return np.stack([m.real, m.imag], axis=-1).tolist()


def _operators_block(operators, n_orb_imp):
    """The ``operators`` header block: matrices as ``[re, im]`` nests, validated against the impurity size."""
    unknown = set(operators) - set(_OPERATOR_MATRICES) - {"symmetry_ops", "source", "frame"}
    if unknown:
        raise ValueError(f"unknown operator entries {sorted(unknown)}")
    block = {}
    for name in _OPERATOR_MATRICES:
        m = operators.get(name)
        if m is None:
            continue
        m = np.asarray(m, dtype=complex)
        want = (3, n_orb_imp, n_orb_imp) if name == "dipole" else (n_orb_imp, n_orb_imp)
        if m.shape != want:
            raise ValueError(f"operator {name} has shape {m.shape}, expected {want}")
        if not np.isfinite(m).all():
            raise ValueError(f"operator {name} holds non-finite values")
        block[name] = _pairs(m)
    syms = operators.get("symmetry_ops")
    if syms:
        block["symmetry_ops"] = [{"matrix": _pairs(u), "antiunitary": bool(a)} for u, a in syms]
    block["source"] = operators.get("source", "file")
    block["frame"] = operators.get("frame", "local")
    return block


def write_h0_file(
    filename,
    h_matrix,
    *,
    impurity_orbitals,
    unit,
    energy_reference="fermi",
    drop_tolerance=DEFAULT_DROP_TOLERANCE,
    rot_to_spherical=None,
    spin_ordering=None,
    operators=None,
    **header_extra,
):
    """Write ``h_matrix`` in impurityModel's ``.h0`` format.

    The format is specified in impurityModel's ``doc/h0_file_format.md``; this writer and
    that reader are pinned against each other by a shared, hand-authored fixture rather than
    by a shared dependency.

    Amplitudes are written with ``repr``, which round-trips exactly. The fixed-point
    ``%.15f`` of the retired bare-integer writer was an absolute 1e-15 grid and so was lossless
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
    spin_ordering : str, optional
        ``"down_first"`` or ``"up_first"``. Added to ``required_features`` when given, so a
        reader that does not understand it must refuse rather than silently guess.
    operators : dict, optional
        One-body operators in the impurity basis, written as the header's ``operators`` block (see
        impurityModel's ``doc/h0_file_format.md``): any of ``sz``, ``s_plus``, ``lz``, ``l_plus``,
        ``s_axis`` (``n_imp x n_imp``) and ``dipole`` (``3 x n_imp x n_imp``) as arrays, ``symmetry_ops`` as a
        list of ``(matrix, antiunitary)``, plus ``source`` (``"rspt"``, ``"file"``, ``"spherical"`` or
        ``"index"``) and ``frame`` (``"global"`` or ``"local"``). Plain arrays on purpose: this package does
        not import impurityModel.
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
        "required_features": [
            "unit",
            "energy_reference",
            "index_convention",
            "storage",
        ],
        "unit": unit,
        "energy_reference": energy_reference,
        "n_orb": int(n_orb),
        "index_convention": "impurity-block-first",
        "storage": "full",
        "impurity_orbitals": {
            str(k): [int(o) for o in orbs] for k, orbs in impurity_orbitals.items()
        },
        "drop_tolerance": float(drop_tolerance),
    }
    if rot_to_spherical is not None:
        rot = np.asarray(rot_to_spherical, dtype=complex)
        header["rot_to_spherical"] = [
            [[float(v.real), float(v.imag)] for v in row] for row in rot
        ]
    if operators:
        header["operators"] = _operators_block(operators, n_orb_imp=len(
            [o for orbs in impurity_orbitals.values() for o in orbs]))
    if spin_ordering is not None:
        header["required_features"].append("spin_ordering")
        header["spin_ordering"] = spin_ordering
    header.update({k: v for k, v in header_extra.items() if v is not None})

    lines = [
        f"# impurityModel-h0 v{H0_SPEC_VERSION}",
        json.dumps(header, allow_nan=False),
        "--",
    ]
    for i in range(n_orb):
        for j in range(n_orb):
            if max(abs(h[i, j]), abs(h[j, i])) <= cutoff:
                continue
            # complex() first: repr of a numpy scalar is "np.float64(...)", not a number.
            value = complex(h[i, j])
            lines.append(f"{i} {j} {value.real!r} {value.imag!r}")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

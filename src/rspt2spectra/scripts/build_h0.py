"""Console script ``build_h0``; see ``build_h0 --help``.

Reads RSPt output in the current directory, fits the hybridization function,
and writes the non-interacting impurity Hamiltonian (h0) as an operator file.
"""

import warnings
from argparse import ArgumentParser
from collections.abc import Iterable
from typing import Callable

import numpy as np

try:
    from mpi4py import MPI
except (
    ImportError,
    RuntimeError,
):  # pragma: no cover - MPI is optional; serial runs work without it
    MPI = None

from rspt2spectra.block_structure import (
    BlockStructure,
    build_block_structure,
    build_matrix,
)
from rspt2spectra.dat import extract_dat
from rspt2spectra.h0 import BLOCK_EQUIVALENCE_TOL, assemble_h0
from rspt2spectra.h2imp import matrixToIOp, write_to_file
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.natural_orbitals import fit_hyb_natural_orbitals
from rspt2spectra.op_printer import write_h0_file
from rspt2spectra.readfile import (
    list_cluster_labels,
    parse_cluster_basis,
    parse_fermi_energy,
    parse_matrices,
)
from rspt2spectra.utils import block_diagonalize_hyb, matrix_print
from rspt2spectra.weight_functions import weight_functions


def partition_index(l: Iterable, pred: Callable = bool) -> tuple[list[int], list[int]]:
    """Partition the indices of an iterable based on a predicate function.

    Parameters
    ----------
    l : Iterable
        The iterable collection of items.
    pred : callable, default bool
        The predicate function to test each item.

    Returns
    -------
    yes : list of int
        Indices for which pred(item) is True.
    no : list of int
        Indices for which pred(item) is False.
    """
    yes, no = [], []

    for idx, item in enumerate(l):
        if pred(item):
            yes.append(idx)
        else:
            no.append(idx)
    return yes, no


def generate_rspt_T_matrix(l, basis_tag, spinpol):
    """
    Generate the exact transformation matrix T mapping the RSPt local basis to spherical harmonics.

    This exactly reproduces the projection vectors in RSPt's lda_mlmsatomicqn.
    T has shape (N_sph, N_corr), such that v_sph = T @ v_corr.
    """
    cisqrt2 = 1j / np.sqrt(2)
    csqrt2 = 1 / np.sqrt(2)

    if l == 3:
        mmsize = 7
        cols = []
        csqrt5o4 = np.sqrt(5) / 4
        csqrt3o4 = np.sqrt(3) / 4
        cisqrt5o4 = 1j * np.sqrt(5) / 4
        cisqrt3o4 = 1j * np.sqrt(3) / 4

        if basis_tag & 4:  # A2u
            cols.append([0, cisqrt2, 0, 0, 0, -cisqrt2, 0])
        if basis_tag & 2:  # T1u
            cols.append([csqrt5o4, 0, -csqrt3o4, 0, csqrt3o4, 0, -csqrt5o4])
            cols.append([-cisqrt5o4, 0, -cisqrt3o4, 0, -cisqrt3o4, 0, -cisqrt5o4])
            cols.append([0, 0, 0, 1, 0, 0, 0])
        if basis_tag & 1:  # T2u
            cols.append([-csqrt3o4, 0, -csqrt5o4, 0, csqrt5o4, 0, csqrt3o4])
            cols.append([-cisqrt3o4, 0, cisqrt5o4, 0, cisqrt5o4, 0, -cisqrt3o4])
            cols.append([0, csqrt2, 0, 0, 0, csqrt2, 0])

        T_spatial = np.array(cols).T  # Shape: (7, N_spatial)

    elif l == 2:
        mmsize = 5
        # Columns in order: Eg, T2g
        cols = []
        if basis_tag & 2:  # Eg
            # dz2: [0, 0, 1, 0, 0]
            cols.append([0, 0, 1, 0, 0])
            # dx2-y2: [csqrt2, 0, 0, 0, csqrt2]
            cols.append([csqrt2, 0, 0, 0, csqrt2])
        if basis_tag & 1:  # T2g
            # dyz: [0, cisqrt2, 0, cisqrt2, 0]
            cols.append([0, cisqrt2, 0, cisqrt2, 0])
            # dxz: [0, csqrt2, 0, -csqrt2, 0]
            cols.append([0, csqrt2, 0, -csqrt2, 0])
            # dxy: [cisqrt2, 0, 0, 0, -cisqrt2]
            cols.append([cisqrt2, 0, 0, 0, -cisqrt2])

        T_spatial = np.array(cols).T  # Shape: (5, N_spatial)

    elif l == 1:
        mmsize = 3
        cols = []
        if basis_tag & 1:
            # py: [cisqrt2, 0, cisqrt2]
            cols.append([cisqrt2, 0, cisqrt2])
            # px: [csqrt2, 0, -csqrt2]
            cols.append([csqrt2, 0, -csqrt2])
            # pz: [0, 1, 0]
            cols.append([0, 1, 0])

        T_spatial = np.array(cols).T
    else:
        raise NotImplementedError(f"Basis generation for l={l} is not yet implemented.")

    if not spinpol:
        return T_spatial

    # If spin-polarized, RSPt packs spin-down then spin-up: green_trunk_interface.F90's
    # lda_mlmsatomicqn assigns qn(2, :offset) = -1 (2*ms, spin down) to the first block and
    # qn(2, offset+1:) = 1 (spin up) to the second -- i.e. this is a *structural* fact about
    # how RSPt lays out the array, not a per-run physical property to detect. It happens to
    # match impurityModel's own c2i convention (s=0 first, s=0 = down; atomic_physics.py),
    # so build_h0 needs no permutation to satisfy the .h0 format's spin_ordering: down_first.
    N_sph = mmsize * 2
    N_corr = T_spatial.shape[1] * 2
    T = np.zeros((N_sph, N_corr), dtype=complex)
    T[:mmsize, : T_spatial.shape[1]] = T_spatial
    T[mmsize:, T_spatial.shape[1] :] = T_spatial
    return T


# The basis tags for which generate_rspt_T_matrix builds cubic (Oh)-irrep columns, keyed by l.
# A cluster whose green.inp basis tag is one of these is *guaranteed* cubic, independent of
# whether T came from the dynamic generator or from a matrix RSPt itself printed -- either way
# it represents the same rotation, so the spherical-under-cubic-symmetry fingerprint below must
# hold. basis_tag == 0 (or a tag outside this set) proves nothing either way.
_CUBIC_BASIS_TAGS = {
    1: {1},
    2: {1, 2, 3},
    3: {1, 2, 3, 4, 5, 6, 7},
}


def _is_cubic_crystal_field(l_val, basis_tag):
    """Whether ``basis_tag`` names one of RSPt's cubic (Oh) irrep decompositions for shell l_val."""
    return basis_tag in _CUBIC_BASIS_TAGS.get(l_val, set())


def _spherical_signature(H, l_val, tol=1e-8):
    """Check ``H`` for the spherical-under-cubic-symmetry fingerprint.

    Under Oh symmetry, a Hamiltonian written in the spherical harmonics basis (m ordered
    -l..l, per the column order ``generate_rspt_T_matrix`` builds) for a crystal-field-split
    shell has a diagonal palindromic in m (``H[m,m] == H[-m,-m]``) and a nonzero ``H[-l,+l]``
    off-diagonal element linking the extremal m states.

    That statement is about the *orbital* (crystal-field) part of ``H``, which is what
    Oh symmetry constrains. Spin-orbit coupling adds a ``xi * m * s`` diagonal term per spin
    sector, which is antisymmetric in ``m`` on its own -- so a genuinely spherical, SOC-dressed
    ``H`` fails a *per-spin-sector* palindrome check (verified on a real NiO workload: the
    down-spin sector alone ran -0.857 .. -1.057 eV, not palindromic) while still being exactly
    spherical. Averaging the two spin sectors cancels the ``xi * m * s`` term (opposite sign
    for each spin) and recovers the pure crystal-field diagonal the fingerprint is about,
    without assuming ``H`` has no SOC in it.

    Parameters
    ----------
    H : np.ndarray
        The impurity Hamiltonian: one or two (spin) ``mmsize x mmsize`` sectors on its
        diagonal, ``mmsize = 2 * l_val + 1``.
    l_val : int
        The shell's angular momentum.
    tol : float, default 1e-8
        Absolute tolerance for both checks.

    Returns
    -------
    bool or None
        True/False if the check could be evaluated. None if ``H``'s size does not match one or
        two spherical-harmonics sectors for ``l_val`` (e.g. a composite multi-shell cluster) --
        the layout could not be determined, so the fingerprint is not checkable.
    """
    mmsize = 2 * l_val + 1
    n = H.shape[0]
    if n == mmsize:
        block = H[:mmsize, :mmsize]
    elif n == 2 * mmsize:
        # Spin-average first: SOC's xi*m*s term has opposite sign in the two sectors and
        # cancels here, leaving the spin-independent crystal-field part the fingerprint tests.
        block = 0.5 * (H[:mmsize, :mmsize] + H[mmsize:, mmsize:])
    else:
        return None

    if mmsize < 3:
        return True  # no eg/t2g-like split to fingerprint (s or unsplit p shell)

    diag = np.diag(block).real
    if not np.allclose(diag, diag[::-1], atol=tol):
        return False
    if abs(block[0, mmsize - 1]) <= tol:
        return False
    return True


def _report_fit_fidelity(
    w, eim, phase_hyb, block_structure, ebs_star, vs_star, verbose
):
    """Print, per inequivalent block, how much of the occupied hybridization weight the fit
    captured.

    A 12% t2g capture (measured on a real NiO star-bath fit) was previously only discoverable
    by reverse-engineering the written ``.h0`` file after the fact. This reports it at fit
    time instead: ``integral(|Im D_fit|) / integral(|Im D_RSPt|)`` below the Fermi level
    (``w < 0``), diagonal per orbital in the block.

    Reconstructs ``D_fit`` from ``(ebs_star, vs_star)`` directly, in the same block-diagonal
    basis ``phase_hyb`` is already in -- each pole ``i`` of a block contributes
    ``V_i^dagger @ V_i / (z - e_i)`` (block-native fitting couples every orbital in the block to
    one shared degenerate bath energy per pole; see ``hyb_fit.fit_hyb``). Failure to reconstruct
    a given block (an unexpected pole shape) is reported and skipped rather than raised --
    this is a diagnostic, not a correctness gate, and must not block writing the file.
    """
    if not verbose:
        return
    occupied = w < 0
    if not np.any(occupied):
        return
    z = w[occupied, None, None] + 1j * eim
    print()
    print(
        "Hybridization fit fidelity (integral |Im D| below E_F, per orbital in each block):"
    )
    for idx, b in enumerate(block_structure.inequivalent_blocks):
        orbs = block_structure.blocks[b]
        try:
            hyb_true = phase_hyb[occupied][:, orbs, :][:, :, orbs]
            hyb_fit = np.zeros_like(hyb_true)
            for e_i, v_i in zip(ebs_star[idx], vs_star[idx]):
                hyb_fit += (np.conjugate(v_i.T) @ v_i)[None] / (z[:, 0, 0] - e_i)[
                    :, None, None
                ]
        except (IndexError, ValueError) as exc:
            print(
                f"  block {list(orbs)}: could not reconstruct the fit for reporting ({exc}); skipped."
            )
            continue
        true_int = np.trapezoid(np.abs(hyb_true.imag), w[occupied], axis=0)
        fit_int = np.trapezoid(np.abs(hyb_fit.imag), w[occupied], axis=0)
        for k, orb in enumerate(orbs):
            ratio = (
                fit_int[k, k] / true_int[k, k] if true_int[k, k] > 0 else float("nan")
            )
            print(f"  orbital {orb} (block {list(orbs)}): {100 * ratio:5.1f}% captured")


def _check_kramers_degeneracy(H, rtol=1e-8):
    """Odd-multiplicity eigenvalue clusters of ``H`` -- a basis-independent witness that ``H``
    breaks time-reversal (Kramers) symmetry.

    Every eigenvalue of a time-reversal-symmetric spin-orbital one-body Hamiltonian is paired
    with its Kramers partner at the same energy, so every cluster has even size. This is a
    local copy of the same clustering idea impurityModel's ``symmetries.check_kramers_degeneracy``
    uses to warn on read -- rspt2spectra has no dependency on impurityModel, so it is
    reimplemented rather than imported across the repo boundary.

    Parameters
    ----------
    H : np.ndarray
        Dense, Hermitian one-body matrix (impurity + bath).
    rtol : float, default 1e-8
        Eigenvalues within ``rtol * max(abs(eigenvalues))`` of each other are one cluster.

    Returns
    -------
    list of tuple
        ``(energy, multiplicity)`` for each odd cluster, sorted by energy. Empty if ``H`` is
        Kramers-degenerate to within ``rtol``.
    """
    w = np.linalg.eigvalsh(H)
    if w.size == 0:
        return []
    scale = np.abs(w).max()
    atol = rtol * scale if scale > 0 else rtol
    clusters = [[w[0]]]
    for x in w[1:]:
        if x - clusters[-1][-1] <= atol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [(float(np.mean(c)), len(c)) for c in clusters if len(c) % 2]


def _detect_soc(H, l_val, tol_rtol=1e-8):
    """Whether ``H``'s impurity block already contains spin-orbit coupling.

    A pure crystal-field impurity block conserves spin: in the ``down_first`` spherical
    layout, the down-down and up-up sectors are the only ones populated, and the down-up
    coupling block is exactly zero. Spin-orbit coupling is precisely what breaks this -- it
    mixes ``down m`` into ``up (m +/- 1)`` -- so a nonzero down-up block is direct evidence of
    SOC already present in the amplitudes, independent of which specific ladder pattern it
    follows. This is measured, not assumed, so it survives the impurity block being rotated,
    fit-adjusted, or otherwise not bit-identical to a bare ``getSOCop`` output.

    Parameters
    ----------
    H : np.ndarray
        The impurity Hamiltonian, spin-resolved (``2*mmsize x 2*mmsize``), spherical,
        down_first.
    l_val : int
        The shell's angular momentum. ``-1`` (undetermined) cannot be checked.
    tol_rtol : float, default 1e-8
        Relative tolerance against ``max(abs(H))``.

    Returns
    -------
    bool or None
        Whether SOC is present. ``None`` if ``H``'s size does not match a spin-resolved
        single-shell impurity block for ``l_val``, or if ``l_val`` is undetermined -- the
        measurement is not meaningful and callers should omit ``contains_soc`` rather than
        guess.
    """
    if l_val == -1:
        return None
    mmsize = 2 * l_val + 1
    if H.shape[0] != 2 * mmsize:
        return None
    scale = np.abs(H).max()
    if scale == 0:
        return False
    offdiag_spin_block = H[:mmsize, mmsize:]
    return bool(np.abs(offdiag_spin_block).max() > tol_rtol * scale)


def _verify_spherical_basis(H, T, cluster, l_val, basis_tag):
    """Verify a rotated impurity Hamiltonian is genuinely in the spherical harmonics basis.

    Raises when the rotation is provably wrong: T is not unitary, or the cluster's basis tag
    names a cubic crystal field and the fingerprint check fails. Only warns when the check
    could not be conclusive (non-cubic or composite cluster), since a failure there is not
    proof of anything -- never a silent pass either way.
    """
    if not np.allclose(T @ np.conjugate(T.T), np.eye(T.shape[0]), atol=1e-8):
        raise RuntimeError(
            f"Cluster {cluster}: the rotation matrix T is not unitary; refusing to trust it."
        )

    if l_val == -1:
        warnings.warn(
            f"Cluster {cluster}: angular momentum l could not be determined, so the "
            "spherical-basis signature cannot be checked. Writing basis: spherical on the "
            "strength of the rotation alone.",
            stacklevel=2,
        )
        return

    sig = _spherical_signature(H, l_val)
    is_cubic = _is_cubic_crystal_field(l_val, basis_tag)
    if is_cubic and sig is False:
        raise RuntimeError(
            f"Cluster {cluster}: basis tag {basis_tag} names a cubic (Oh) crystal field for "
            f"l={l_val}, but the rotated Hamiltonian does not show the expected "
            "spherical-under-cubic-symmetry signature (diagonal palindromic in m, "
            "H[-l,+l] != 0). Refusing to write a file claiming basis: spherical."
        )
    if sig is False or (sig is None and is_cubic):
        warnings.warn(
            f"Cluster {cluster}: could not conclusively verify the spherical-basis signature "
            f"(basis tag {basis_tag}, l={l_val} -- not a recognized cubic decomposition, or a "
            "composite/multi-shell cluster). Writing basis: spherical on the strength of the "
            "rotation alone; verify by hand before trusting it.",
            stacklevel=2,
        )


def filter_and_shift(
    ebs: list[np.ndarray],
    vs: list[np.ndarray],
    w_min: float,
    w_max: float,
    block_structure: BlockStructure,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Filter bath energies and shift hopping parameters based on an energy window.

    Parameters
    ----------
    ebs : list of np.ndarray
        List of bath energies.
    vs : list of np.ndarray
        List of hopping parameters.
    w_min : float
        Lower energy bound.
    w_max : float
        Upper energy bound.
    block_structure : BlockStructure
        The block structure mappings.

    Returns
    -------
    shift_matrix : np.ndarray
        The energy shift matrix.
    filtered_ebs : list of np.ndarray
        The filtered bath energies.
    filtered_vs : list of np.ndarray
        The filtered hopping parameters.
    """
    filtered_ebs_star, filtered_vs_star = ([], [])
    shifts = [
        np.zeros(
            (
                len(block_structure.blocks[i_block]),
                len(block_structure.blocks[i_block]),
            ),
            dtype=complex,
        )
        for i_block in block_structure.inequivalent_blocks
    ]
    for eb_block, v_block, shift in zip(ebs, vs, shifts):
        f = np.logical_or(eb_block < w_min, eb_block > w_max)
        shift += np.sum(  # noqa: PLW2901 - in-place update of the arrays in shifts
            np.conj(np.transpose(v_block[f], (0, 2, 1)))
            @ v_block[f]
            / eb_block[f, None, None],
            axis=0,
        )
        filtered_ebs_star.append(eb_block[np.logical_not(f)].copy())
        filtered_vs_star.append(v_block[np.logical_not(f)].copy())
    return build_matrix(shifts, block_structure), filtered_ebs_star, filtered_vs_star


def run(
    cluster: str,
    bath_geometry: str,
    eim: float,
    bath_states_per_orbital: int,
    gamma: float,
    fit_unocc: bool,
    fit_imag: bool,
    prefix: str,
    verbose: bool,
    plot: bool,
    regularization: str,
    weight_function: str,
    weight_factor: float,
    fit_center: float,
    natural_orbitals: bool,
    grid_type: str,
    peel_weight: float = 0.05,
    legacy_dict: bool = False,
    allow_broken_time_reversal: bool = False,
    *kwargs,
) -> None:
    """Execute the full non-interacting Hamiltonian (h0) building workflow.

    Extracts hybridization and Hamiltonian data, runs fitters to obtain bath states,
    transforms bath geometry (e.g. star to chain), builds the full bath/hopping matrices,
    and writes the resulting Hamiltonian parameters (h0) to a pickle file.
    """
    comm = MPI.COMM_WORLD if MPI is not None else None
    rank = comm.rank if comm is not None else 0

    verbose = verbose and rank == 0

    hyb_dat = extract_dat("hyb", cluster, prefix)
    hs = parse_matrices(
        out_file="out", search_phrase="Local hamiltonian", prefix=prefix
    )
    qs = parse_matrices(
        out_file="out",
        search_phrase="Transformation to the local cf basis:",
        prefix=prefix,
    )
    sharm_qs = parse_matrices(
        out_file="out",
        search_phrase="sharm2corr of the cluster",
        prefix=prefix,
    )
    if cluster not in hs:
        raise RuntimeError(
            f"Could not extract local hamiltonian for cluster {cluster} from file {prefix}/out."
        )
    H_dft = hs[cluster]
    hyb = hyb_dat.orbitals
    w = hyb_dat.w

    # RSPt prints the local Hamiltonian on an *absolute* energy scale, while the
    # hybridization function is written on a mesh whose zero is the Fermi level. Combining
    # them without this shift puts the impurity block several eV above the entire bath -- for
    # NiO, +8.2 eV against an O 2p bath at -6.7..-1.9 eV, instead of the -0.9 eV a Ni 3d
    # level should sit at. Subtract at the source, before any rotation; a scalar shift
    # commutes with the unitaries below, but doing it here keeps the two energy zeros from
    # ever being separate in the first place.
    e_fermi = parse_fermi_energy(out_file="out", prefix=prefix)
    H_dft = H_dft - e_fermi * np.eye(H_dft.shape[0])
    if verbose:
        print(
            f"Fermi energy {e_fermi: .8f} subtracted from the local Hamiltonian (bath mesh has E_F = 0)."
        )

    cluster_basis = parse_cluster_basis(cluster, inp_file="green.inp", prefix=prefix)
    if cluster_basis is None:
        known = list_cluster_labels(inp_file="green.inp", prefix=prefix)
        raise RuntimeError(
            f"Cluster {cluster!r} was not found in {prefix}/green.inp (clusters defined there: "
            f"{known!r}). Cannot determine whether its data needs rotating to the spherical "
            "harmonics basis without this match -- refusing to silently assume it is already "
            "spherical."
        )
    has_cf_flag, basis_tag, l_val = cluster_basis
    needs_rotation = has_cf_flag or (basis_tag != 0)

    # If transformations to the CF basis were found, use them
    T = None
    if cluster in qs:
        T = qs[cluster]
    elif cluster in sharm_qs:
        # "sharm2corr of the cluster" plausibly names the opposite direction from "Transformation
        # to the local cf basis:" ("sph -> corr" vs "corr -> cf"), but every rotation applied
        # below assumes the same direction as `qs`. No stored workload prints a sharm2corr
        # matrix to check this against, so using it here would be a guess that could silently
        # rotate away from the spherical basis rather than into it.
        raise RuntimeError(
            f"Cluster {cluster} only has a 'sharm2corr of the cluster' rotation matrix in "
            f"{prefix}/out (no 'Transformation to the local cf basis:' matrix). Its direction "
            "convention relative to the latter is unverified against any known workload, so "
            "using it here would risk silently rotating away from the spherical basis. Verify "
            "the direction empirically before enabling this path."
        )

    if needs_rotation and T is None:
        if has_cf_flag:
            raise RuntimeError(
                f"Cluster {cluster} has Cf flag set in green.inp, but the rotation matrix "
                f"was not found in {prefix}/out. Please run RSPt with verbose=True to print it."
            )
        elif basis_tag > 0:
            # Generate dynamically based on RSPt exact projection matrices
            try:
                N = H_dft.shape[0]
                if l_val == -1:
                    raise RuntimeError(
                        f"Could not determine l quantum number for cluster {cluster} from green.inp"
                    )

                # We determine spinpol by checking if N matches 2 * subset size or 1 * subset size
                # But it's simpler: if N is even, it's very likely spin polarized for ED models.
                # Let's count subset size
                subset_size = 0
                if l_val == 3:
                    if basis_tag & 4:
                        subset_size += 1
                    if basis_tag & 2:
                        subset_size += 3
                    if basis_tag & 1:
                        subset_size += 3
                elif l_val == 2:
                    if basis_tag & 2:
                        subset_size += 2
                    if basis_tag & 1:
                        subset_size += 3
                elif l_val == 1:
                    if basis_tag & 1:
                        subset_size += 3
                else:
                    raise NotImplementedError(
                        f"Dynamic rotation for l={l_val} not supported."
                    )

                if subset_size * 2 == N:
                    spinpol = True
                elif subset_size == N:
                    spinpol = False
                else:
                    raise RuntimeError(
                        f"Hamiltonian size {N} does not match expected size for basis tag {basis_tag} "
                        f"(expected {subset_size} or {subset_size * 2})"
                    )

                T = generate_rspt_T_matrix(l_val, basis_tag, spinpol)
            except Exception as e:
                raise RuntimeError(
                    f"Cluster {cluster} uses basis tag {basis_tag}, but dynamic generation failed: {e}. "
                    f"Please run RSPt with verbose=True to print the rotation matrix in the output."
                ) from e
        else:
            raise RuntimeError(
                f"Cluster {cluster} requires rotation but basis tag is not handled dynamically. "
                f"Please run RSPt with verbose=True to print the rotation matrix in the output."
            )

    if T is None:
        T = np.eye(H_dft.shape[0], dtype=complex)

    if needs_rotation:
        if verbose:
            print(f"Cluster {cluster} uses a non-spherical basis (or Cf flag).")
            if T is not None and cluster not in qs and cluster not in sharm_qs:
                print(
                    f"Dynamically generated RSPt rotation matrix for l={l_val}, basis_tag={basis_tag}."
                )
            print(
                "Applying transformation T to rotate to the Spherical Harmonics basis (T @ H @ T.T.conj())."
            )
            matrix_print(T, "Transformation matrix T:")
            print()
        # Data is in CF basis, rotate to Spherical Harmonics
        hyb_sph = T[None] @ hyb @ np.conjugate(T.T)[None]
        H_dft_sph = T @ H_dft @ np.conjugate(T.T)
    else:
        if verbose:
            print(f"Cluster {cluster} is in the Spherical Harmonics basis.")
            print("No initial basis rotation required.")
            print()
        # Data is already in Spherical Harmonics
        hyb_sph = hyb
        H_dft_sph = H_dft

    _verify_spherical_basis(H_dft_sph, T, cluster, l_val, basis_tag)
    contains_soc = _detect_soc(H_dft_sph, l_val)
    if verbose:
        print(f"Impurity block already contains spin-orbit coupling: {contains_soc}.")

    phase_hyb, Q = block_diagonalize_hyb(hyb_sph)

    if verbose:
        print("Block diagonalizing the hybridization function.")
        matrix_print(
            Q,
            "Unitary transformation Q (from Spherical to block-diagonal fitting basis):",
        )
        print()

    H_imp = H_dft_sph
    H_local_Q = np.conjugate(Q.T) @ H_imp @ Q

    # Anchor equivalence detection to the local Hamiltonian as well as the hybridization
    # function (matching `h0.prepare_hyb_fit`'s use of both), and use the shared, measured
    # tolerance rather than machine epsilon -- see BLOCK_EQUIVALENCE_TOL's docstring. At
    # tol=1e-15 real RSPt data's ~1.4e-10 noise floor between symmetry-equivalent orbitals
    # (e.g. a cubic shell's t2g members, or spin-up/spin-down with no SOC) fractures a single
    # equivalence class into several, each then fit by an independently-seeded stochastic
    # optimizer -- producing measurably different bath parameters between orbitals that must
    # be exactly equivalent by symmetry.
    block_structure = build_block_structure(phase_hyb, mat=H_local_Q, tol=BLOCK_EQUIVALENCE_TOL)

    if natural_orbitals:
        H_imp_blocks = [
            H_local_Q[np.ix_(block_structure.blocks[b], block_structure.blocks[b])]
            for b in block_structure.inequivalent_blocks
        ]
        ebs_star, vs_star = fit_hyb_natural_orbitals(
            w,
            phase_hyb,
            H_imp_blocks,
            bath_states_per_orbital,
            block_structure,
            n_bins=1000,
            grid_type=grid_type,
        )
        # The natural-orbitals discretization fits no constant offset.
        cs_star = [
            np.zeros(
                (len(block_structure.blocks[b]), len(block_structure.blocks[b])),
                dtype=complex,
            )
            for b in block_structure.inequivalent_blocks
        ]
    else:
        ebs_star, vs_star, cs_star = fit_hyb(
            w,
            eim,
            phase_hyb,
            bath_states_per_orbital,
            block_structure,
            gamma,
            (w[0], 0) if not fit_unocc else None,
            verbose,
            comm,
            regularization=regularization,
            weight_fun=weight_functions[weight_function](fit_center, weight_factor),
        )
    for ebss, vss in zip(ebs_star, vs_star):
        if len(ebss) == 0:
            continue
        sorted_indices = np.argsort(ebss, kind="stable")
        ebss[:] = ebss[sorted_indices]
        vss[:] = vss[sorted_indices]
    if verbose:
        print("Star bath energies and hopping parameters:")
        for eb, vb in zip(ebs_star, vs_star):
            for eb_i, vb_i in zip(eb, vb):
                matrix_print(vb_i, f"Energy {eb_i: 9.6f} :")
            print()
        print("=" * 80)
    _report_fit_fidelity(w, eim, phase_hyb, block_structure, ebs_star, vs_star, verbose)
    (
        H,
        _H_star,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        _v_solver,
        _H_bath,
        _H_imp,
    ) = assemble_h0(
        ebs_star,
        vs_star,
        cs_star,
        H_imp,
        H_local_Q,
        Q,
        block_structure,
        bath_geometry=bath_geometry,
        w=w,
        eim=eim,
        label=cluster,
        verbose=verbose,
        peel_weight=peel_weight,
    )

    # SOC is time-reversal *even* (L.S is invariant under T, since both L and S flip sign), so
    # the true continuous Hamiltonian stays Kramers degenerate with SOC present. But the star
    # bath discretizes each *inequivalent* block independently (fit_hyb / block_structure), and
    # time-reversal-conjugate blocks are related by an antiunitary map that identical_blocks /
    # transposed_blocks / particle_hole_blocks do not track -- without SOC the up/down blocks
    # are literally identical and share one fit, so degeneracy survives by construction; with
    # SOC they are not identical, and nothing currently enforces their pairing. Losing exact
    # Kramers degeneracy in the discretized fit is therefore an expected consequence of SOC
    # being present, not a fitting defect -- refusing on it would be refusing legitimate output
    # for the majority of real transition-metal workloads (NiO, FCC Ni, BCC Fe... all have real
    # 3d SOC). `contains_soc is not True` (rather than testing falsiness) means an undetermined
    # shell (l_val == -1, contains_soc is None) is treated like "no SOC" -- conservative, since
    # there is no positive evidence to explain a violation away.
    if not allow_broken_time_reversal and contains_soc is not True:
        violations = _check_kramers_degeneracy(H)
        if violations:
            detail = ", ".join(f"E={e:.4g} (x{m})" for e, m in violations)
            raise RuntimeError(
                f"Cluster {cluster}: the assembled Hamiltonian has {len(violations)} "
                f"odd-multiplicity eigenvalue cluster(s) -- it breaks time-reversal (Kramers) "
                f"symmetry: {detail}. No spin-orbit coupling was detected in the impurity "
                "block, so this is not the expected SOC-driven loss of exact degeneracy -- it "
                "usually means the cluster is genuinely spin-polarised or field-dressed. If so, "
                "pass --allow-broken-time-reversal. Otherwise this may indicate a fitting "
                "problem; try a different --gamma or bath_states_per_orbital."
            )

    # The valence/conduction split is read off the *star* bath diagonal. For a chain geometry
    # each bath site is a Lanczos combination of star modes, so the same index means something
    # different in the matrix actually written -- the labels would be misleading rather than
    # approximate. Record them only when the two coincide.
    star_geometry = bath_geometry.lower() == "star"

    write_h0_file(
        f"{cluster}.h0",
        H,
        impurity_orbitals={0: impurity_indices},
        # RSPt writes Rydberg unless green.inp asks otherwise; this package works in its
        # input's unit and records it, and the consumer converts.
        unit="Ry",
        energy_reference="fermi",
        fermi_energy=e_fermi,
        valence_bath=list(valence_bath_indices) if star_geometry else None,
        conduction_bath=list(conduction_bath_indices) if star_geometry else None,
        # By construction: _verify_spherical_basis has already raised if this does not hold, so
        # the file always says how to get to spherical (the identity, since it is already
        # there) rather than omitting the key when no rotation happened to be needed.
        rot_to_spherical=np.eye(len(impurity_indices), dtype=complex),
        basis="spherical",
        # Structural, not measured: see the comment in generate_rspt_T_matrix. RSPt always
        # packs the first spin block as down (2*ms = -1) and the second as up (2*ms = +1).
        spin_ordering="down_first",
        basis_tag=int(basis_tag),
        rotation_applied=bool(needs_rotation),
        bath_geometry=bath_geometry,
        impurity_l=int(l_val) if l_val != -1 else None,
        # Measured from H_dft_sph's down-up spin block (see _detect_soc), not assumed --
        # None (l_val undetermined) is dropped by write_h0_file, so the key is simply omitted
        # rather than written with a guessed value.
        contains_soc=contains_soc,
        producer={"name": "rspt2spectra", "cluster": cluster},
    )

    if legacy_dict:
        write_to_file(matrixToIOp(H), f"{cluster}_h0_op", save_as_dict=True)

    if plot and rank == 0:
        try:
            # matplotlib is an optional dependency; import it only when plotting.
            import matplotlib.pyplot as plt  # noqa: PLC0415

            from rspt2spectra.plot import plot_hyb_fit  # noqa: PLC0415

            plot_hyb_fit(
                w,
                eim,
                phase_hyb,
                ebs_star,
                vs_star,
                cs_star,
                H_local_Q,
                block_structure,
                bath_geometry,
                peel_weight=peel_weight,
            )
            plt.show()
        except ImportError:
            print("Warning: matplotlib is not installed. Plotting skipped.")


def main() -> None:
    """Parse command line arguments and execute the non-interacting Hamiltonian builder.

    Fits the hybridization function and constructs the non-interacting Hamiltonian.
    """
    parser = ArgumentParser(
        prog="build_h0",
        description="Create local hamiltonians by reading RSPt out files and fitting hybridization functions.",
    )
    parser.add_argument("cluster", type=str)
    parser.add_argument("bath_states_per_orbital", type=int)
    parser.add_argument("-bg", "--bath-geometry", type=str, default="Star")
    parser.add_argument(
        "--peel-weight",
        type=float,
        default=0.05,
        help=(
            "For --bath-geometry peeled_linked_chain: keep star modes carrying at "
            "least this fraction of the block's total hybridization weight as "
            "direct impurity couplings; only the remainder is chained."
        ),
    )
    parser.add_argument("--eim", type=float, default=0.010)
    parser.add_argument("--gamma", type=float, default=0.100)
    parser.add_argument("--fit-unocc", action="store_true", dest="fit_unocc")
    parser.add_argument("-i", "--imag-only", action="store_true", dest="fit_imag")
    parser.add_argument("-d", "--directory", type=str, default=".", dest="prefix")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument(
        "--legacy-dict",
        dest="legacy_dict",
        action="store_true",
        help=(
            "Also write the legacy <cluster>_h0_op.dict, which records no unit, energy "
            "reference, basis or orbital layout. Only for tooling not yet updated to .h0."
        ),
    )
    parser.add_argument(
        "--allow-broken-time-reversal",
        dest="allow_broken_time_reversal",
        action="store_true",
        help=(
            "Skip the Kramers-degeneracy check on the assembled Hamiltonian when spin-orbit "
            "coupling was NOT detected in the impurity block. (When SOC is detected, the check "
            "is skipped automatically -- exact Kramers degeneracy in the discretized bath fit "
            "is not expected once SOC is present, since the fit does not enforce it; see the "
            "comment at the check site in build_h0.py.) Without SOC, every eigenvalue of a "
            "time-reversal-symmetric one-body H is at least doubly degenerate, so an odd "
            "cluster there usually means the cluster is genuinely spin-polarised or "
            "field-dressed -- pass this flag for that case."
        ),
    )
    parser.add_argument("--regularization", type=str, default="l2")
    parser.add_argument("--weight-function", type=str, default="unit")
    parser.add_argument("--weight-factor", type=float, default=2.0)
    parser.add_argument("--fit-center", type=float, default=0)
    parser.add_argument(
        "--natural-orbitals",
        action="store_true",
        help="Use Natural Orbitals approach instead of non-linear fitting",
    )
    parser.add_argument(
        "--grid-type",
        type=str,
        default="linear",
        choices=["linear", "logarithmic"],
        help="Grid type for Natural Orbitals (linear or logarithmic)",
    )
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()

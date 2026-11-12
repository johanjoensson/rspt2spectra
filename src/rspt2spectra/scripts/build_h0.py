"""Console script ``build_h0``; see ``build_h0 --help``.

Reads RSPt output in the current directory, fits the hybridization function,
and writes the non-interacting impurity Hamiltonian (h0) as an operator file.
"""

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
    build_matrix,
    print_block_structure,
)
from rspt2spectra.dat import extract_dat
from rspt2spectra.h0 import assemble_h0, flatten_star_levels, prepare_hyb_fit
from rspt2spectra.h2imp import matrixToIOp, write_to_file
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.natural_orbitals import fit_hyb_natural_orbitals
from rspt2spectra.op_printer import write_h0_file
from rspt2spectra.readfile import (
    list_cluster_labels,
    parse_cluster_basis,
    parse_fermi_energy,
    parse_matrices,
    read_projection_file,
)
from rspt2spectra.rspt_basis import cluster_rotation, projection_from_files
from rspt2spectra.symmetries import DEFAULT_SYMMETRY_TOL
from rspt2spectra.utils import matrix_print
from rspt2spectra.weight_functions import weight_functions

DEFAULT_SYMM_PREC = 1e-6


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


def _report_fit_fidelity(w, eim, phase_hyb, block_structure, ebs_star, vs_star, verbose):
    """Print, per inequivalent block, how much of the occupied hybridization weight the fit captured.

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
    print("Hybridization fit fidelity (integral |Im D| below E_F, per orbital in each block):")
    for idx, b in enumerate(block_structure.inequivalent_blocks):
        orbs = block_structure.blocks[b]
        try:
            hyb_true = phase_hyb[occupied][:, orbs, :][:, :, orbs]
            hyb_fit = np.zeros_like(hyb_true)
            for e_i, v_i in zip(ebs_star[idx], vs_star[idx]):
                hyb_fit += (np.conjugate(v_i.T) @ v_i)[None] / (z[:, 0, 0] - e_i)[:, None, None]
        except (IndexError, ValueError) as exc:
            print(f"  block {list(orbs)}: could not reconstruct the fit for reporting ({exc}); skipped.")
            continue
        true_int = np.trapezoid(np.abs(hyb_true.imag), w[occupied], axis=0)
        fit_int = np.trapezoid(np.abs(hyb_fit.imag), w[occupied], axis=0)
        for k, orb in enumerate(orbs):
            ratio = fit_int[k, k] / true_int[k, k] if true_int[k, k] > 0 else float("nan")
            print(f"  orbital {orb} (block {list(orbs)}): {100 * ratio:5.1f}% captured")


def _check_kramers_degeneracy(H, rtol=1e-8):
    """Odd-multiplicity eigenvalue clusters of ``H``: a witness that ``H`` breaks Kramers symmetry.

    The clusters are basis-independent, so they witness broken time-reversal symmetry in any basis.

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


def _detect_soc(H, shells, tol_rtol=1e-8):
    """Whether ``H``'s impurity block already contains spin-orbit coupling.

    A pure crystal-field impurity block conserves spin: inside every shell of the
    ``down_first`` spherical layout the down-up coupling block is exactly zero. Spin-orbit
    coupling mixes ``down m`` into ``up (m +/- 1)``, so a nonzero down-up block in any shell is
    direct evidence of SOC already present in the amplitudes. This is measured, not assumed.

    Parameters
    ----------
    H : np.ndarray
        The spin-resolved impurity Hamiltonian in the spherical, down_first basis, with shells
        laid out one after another.
    shells : list[tuple[int, int]]
        ``(l, basis_tag)`` per shell, in ``green.inp`` order.
    tol_rtol : float, default 1e-8
        Relative tolerance against ``max(abs(H))``.

    Returns
    -------
    bool or None
        Whether SOC is present. ``None`` if the shell sizes do not add up to the size of ``H``,
        in which case the measurement is not meaningful and callers should omit
        ``contains_soc`` rather than guess.
    """
    sizes = [2 * (2 * l + 1) for l, _ in shells]
    if not sizes or any(l < 0 for l, _ in shells) or sum(sizes) != H.shape[0]:
        return None
    scale = np.abs(H).max()
    if scale == 0:
        return False
    offset = 0
    for size in sizes:
        half = size // 2
        block = H[offset : offset + half, offset + half : offset + size]
        if np.abs(block).max() > tol_rtol * scale:
            return True
        offset += size
    return False


def _verify_spherical_basis(T, cluster):
    """Refuse a rotation to the spherical basis that is not unitary.

    The spherical basis is defined by the rotation alone, not by any symmetry the data may or
    may not have, so unitarity of ``T`` is the whole check.
    """
    if not np.allclose(T @ np.conjugate(T.T), np.eye(T.shape[0]), atol=1e-8):
        raise RuntimeError(f"Cluster {cluster}: the rotation matrix T is not unitary; refusing to trust it.")


def _shell_orbital_indices(shells, spin_resolved=True):
    """Impurity orbital indices of every shell, as ``{shell: [indices]}`` in cluster layout."""
    groups = {}
    offset = 0
    for k, (l, _) in enumerate(shells):
        size = (2 if spin_resolved else 1) * (2 * l + 1)
        groups[k] = list(range(offset, offset + size))
        offset += size
    return groups


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
            np.conj(np.transpose(v_block[f], (0, 2, 1))) @ v_block[f] / eb_block[f, None, None],
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
    enforce_symmetry: bool = True,
    symmetry_tol: float = DEFAULT_SYMMETRY_TOL,
    symm_prec: float = DEFAULT_SYMM_PREC,
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
    hs = parse_matrices(out_file="out", search_phrase="Local hamiltonian", prefix=prefix)
    qs = parse_matrices(
        out_file="out",
        search_phrase="Transformation to the local cf basis:",
        prefix=prefix,
    )
    if cluster not in hs:
        raise RuntimeError(f"Could not extract local hamiltonian for cluster {cluster} from file {prefix}/out.")
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
        print(f"Fermi energy {e_fermi: .8f} subtracted from the local Hamiltonian (bath mesh has E_F = 0).")

    cluster_basis = parse_cluster_basis(cluster, inp_file="green.inp", prefix=prefix)
    if cluster_basis is None:
        known = list_cluster_labels(inp_file="green.inp", prefix=prefix)
        raise RuntimeError(
            f"Cluster {cluster!r} was not found in {prefix}/green.inp (clusters defined there: "
            f"{known!r}). Cannot determine whether its data needs rotating to the spherical "
            "harmonics basis without this match -- refusing to silently assume it is already "
            "spherical."
        )
    has_cf_flag, shells, irr_labels = cluster_basis
    n_orb = H_dft.shape[0]

    # RSPt's basis index is a fixed rotation of each shell's orbitals (see rspt_basis); it is
    # not tied to the crystal's point group. A non-spin-resolved Hamiltonian carries no
    # down/up split.
    if n_orb == sum(2 * l + 1 for l, _ in shells) and n_orb != sum(2 * (2 * l + 1) for l, _ in shells):
        spin_resolved = False
    else:
        spin_resolved = True
    proj_files = []
    if irr_labels:
        # Custom projectors: RSPt ignores the shells' basis indices and reads the projection
        # from proj-<cluster>-<token>.inp, one file per Irr token, side by side in token order.
        proj_files = [f"proj-{cluster}-{label}.inp" for label in irr_labels]
        parsed = []
        for name in proj_files:
            try:
                parsed.append(read_projection_file(f"{prefix}/{name}"))
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Cluster {cluster} declares the custom projector {name[len('proj-' + cluster + '-') : -4]!r} "
                    f"in green.inp, but {prefix}/{name} does not exist."
                ) from exc
            except ValueError as exc:
                raise RuntimeError(f"Cluster {cluster}: cannot read projector file: {exc}") from exc
        try:
            T = projection_from_files(shells, parsed)
        except ValueError as exc:
            raise RuntimeError(f"Cluster {cluster}: invalid projector files {proj_files}: {exc}") from exc
    else:
        try:
            T = cluster_rotation(shells, spin_resolved=spin_resolved)
        except (ValueError, NotImplementedError) as exc:
            raise RuntimeError(
                f"Cluster {cluster}: cannot rotate basis indices {shells} to spherical harmonics: {exc}"
            ) from exc
    if T.shape[1] != n_orb:
        source = (
            f"the projector files {proj_files} define"
            if irr_labels
            else f"green.inp shells (l, basis index) {shells} describe"
        )
        raise RuntimeError(
            f"Cluster {cluster}: {source} {T.shape[1]} orbitals, but the local Hamiltonian in "
            f"{prefix}/out is {n_orb}x{n_orb}."
            + (
                " A projection covering only part of the cluster's shells cannot be rotated to spherical harmonics."
                if irr_labels
                else ""
            )
        )

    if has_cf_flag:
        # With the Cf flag RSPt additionally applies corr2cf ("Transformation to the local cf
        # basis:") on the right of the projection, so the data are in basis proj @ alocal.
        if cluster not in qs:
            raise RuntimeError(
                f"Cluster {cluster} has Cf flag set in green.inp, but the rotation matrix "
                f"was not found in {prefix}/out. Please set the 'Projection' flag in the RSPt "
                "input and rerun RSPt to print it."
            )
        alocal = qs[cluster]
        if alocal.shape != (n_orb, n_orb) or not np.allclose(alocal @ np.conjugate(alocal.T), np.eye(n_orb), atol=1e-8):
            raise RuntimeError(
                f"Cluster {cluster}: the printed Cf transformation is not a unitary "
                f"{n_orb}x{n_orb} matrix; refusing to trust it."
            )
        T = T @ alocal

    _verify_spherical_basis(T, cluster)
    rotation_applied = not np.allclose(T, np.eye(n_orb))

    if rotation_applied:
        if verbose:
            print(f"Cluster {cluster} shells (l, basis index): {shells}; Cf flag: {has_cf_flag}.")
            if proj_files:
                print(f"Custom projectors read from: {proj_files}.")
            print("Applying transformation T to rotate to the Spherical Harmonics basis (T @ H @ T.T.conj()).")
            matrix_print(T, "Transformation matrix T:")
            print()
        hyb_sph = T[None] @ hyb @ np.conjugate(T.T)[None]
        H_dft_sph = T @ H_dft @ np.conjugate(T.T)
    else:
        if verbose:
            print(f"Cluster {cluster} is in the Spherical Harmonics basis.")
            print("No initial basis rotation required.")
            print()
        hyb_sph = hyb
        H_dft_sph = H_dft

    contains_soc = _detect_soc(H_dft_sph, shells)
    if verbose:
        print(f"Impurity block already contains spin-orbit coupling: {contains_soc}.")

    # The fit window: occupied states only unless --fit-unocc.  Rotation and block
    # connectivity look at the hybridization inside it, as the fit itself does.
    x_lim = (w[0], 0) if not fit_unocc else None
    # `symm_prec` is the block threshold, relative to the strongest diagonal
    # hybridization and the crystal-field spread of H_dft: couplings below it are
    # neglected, giving exactly block-diagonal hybridization and local Hamiltonian.
    # Block *equivalence* (identical/transposed/particle-hole) keeps the shared, measured
    # BLOCK_EQUIVALENCE_TOL -- real RSPt data has a ~1.4e-10 noise floor between
    # symmetry-equivalent orbitals that machine epsilon would fracture into several
    # independently-seeded fits, while a loose `symm_prec` would merge inequivalent blocks.
    fit_basis = prepare_hyb_fit(hyb_sph, H_dft_sph, tol=symm_prec, verbose=False, w=w, x_lim=x_lim, report=verbose)
    Q, phase_hyb, H_local_Q, block_structure = (
        fit_basis.Q,
        fit_basis.phase_hyb,
        fit_basis.H_local_Q,
        fit_basis.block_structure,
    )

    if verbose:
        print("Rotating to the joint near-diagonal basis of the hybridization function and H_dft.")
        matrix_print(
            Q,
            "Unitary transformation Q (from Spherical to block-diagonal fitting basis):",
        )
        print()
        print_block_structure(block_structure)

    H_imp = H_dft_sph

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
            x_lim,
            verbose,
            comm,
            regularization=regularization,
            weight_fun=weight_functions[weight_function](fit_center, weight_factor),
            enforce_symmetry=enforce_symmetry,
            symmetry_tol=symmetry_tol,
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
    # assemble_h0 takes the flattened form: one bath orbital per coupled component of each
    # level.  Without this a rank-deficient level leaves the chain geometries with fewer bath
    # states than the star.
    if not natural_orbitals:
        for i in range(len(ebs_star)):
            if len(ebs_star[i]) == 0:
                continue
            ebs_star[i], vs_star[i] = flatten_star_levels(ebs_star[i], vs_star[i], verbose=False)
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
    # shell (contains_soc is None) is treated like "no SOC" -- conservative, since
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
    impurity_orbitals = (
        _shell_orbital_indices(shells, spin_resolved) if len(impurity_indices) == n_orb else {0: impurity_indices}
    )

    write_h0_file(
        f"{cluster}.h0",
        H,
        impurity_orbitals=impurity_orbitals,
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
        # Structural, not measured: RSPt packs the first spin block of each shell as down
        # (2*ms = -1) and the second as up (2*ms = +1); see rspt_basis.
        spin_ordering="down_first",
        rspt_shells=[[int(l), int(tag)] for l, tag in shells],
        rspt_projection_files=proj_files if proj_files else None,
        shell_layout="single" if len(shells) == 1 else "multi",
        rotation_applied=bool(rotation_applied),
        bath_geometry=bath_geometry,
        impurity_l=int(shells[0][0]) if len(shells) == 1 else None,
        # Measured from H_dft_sph's down-up spin block (see _detect_soc), not assumed --
        # None (undetermined) is dropped by write_h0_file, so the key is simply omitted
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
    parser.add_argument(
        "--no-enforce-symmetry",
        dest="enforce_symmetry",
        action="store_false",
        help=(
            "Fit every block unconstrained. By default the symmetries each block's "
            "hybridization actually has -- orbital degeneracy, time reversal, and (when "
            "the fit window reaches above the Fermi level) particle-hole -- are detected "
            "and imposed exactly on the fitted bath."
        ),
    )
    parser.add_argument(
        "--symmetry-tol",
        type=float,
        default=DEFAULT_SYMMETRY_TOL,
        help=(
            "Relative tolerance for accepting a symmetry. Too loose imposes a symmetry "
            "the data does not have and distorts the fit; too tight simply falls back to "
            "an unconstrained fit."
        ),
    )
    parser.add_argument(
        "--symm-prec",
        type=float,
        default=DEFAULT_SYMM_PREC,
        dest="symm_prec",
        help=(
            "Block threshold. After rotating the hybridization function and H_dft to a "
            "joint near-diagonal basis, couplings between orbitals smaller than this "
            "fraction of the strongest diagonal hybridization (and of the crystal-field "
            "spread of H_dft) are neglected, so a dense block can split, e.g. 5x5 into "
            "2x2 + 3x3. Never below 1e-6 Ry absolute."
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

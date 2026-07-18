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
    build_block_structure,
    build_matrix,
)
from rspt2spectra.dat import extract_dat
from rspt2spectra.h0 import assemble_h0
from rspt2spectra.h2imp import matrixToIOp, write_to_file
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.natural_orbitals import fit_hyb_natural_orbitals
from rspt2spectra.readfile import parse_cluster_basis, parse_matrices
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

    # If spin-polarized, RSPt packs spin-up then spin-down
    N_sph = mmsize * 2
    N_corr = T_spatial.shape[1] * 2
    T = np.zeros((N_sph, N_corr), dtype=complex)
    T[:mmsize, : T_spatial.shape[1]] = T_spatial
    T[mmsize:, T_spatial.shape[1] :] = T_spatial
    return T


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

    has_cf_flag, basis_tag, l_val = parse_cluster_basis(
        cluster, inp_file="green.inp", prefix=prefix
    )
    needs_rotation = has_cf_flag or (basis_tag != 0)

    # If transformations to the CF basis were found, use them
    T = None
    if cluster in qs:
        T = qs[cluster]
    elif cluster in sharm_qs:
        T = sharm_qs[cluster]

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
                        f"Hamiltonian size {N} does not match expected size for basis tag {basis_tag} (expected {subset_size} or {subset_size*2})"
                    )

                T = generate_rspt_T_matrix(l_val, basis_tag, spinpol)
            except Exception as e:
                raise RuntimeError(
                    f"Cluster {cluster} uses basis tag {basis_tag}, but dynamic generation failed: {e}. "
                    f"Please run RSPt with verbose=True to print the rotation matrix in the output."
                )
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

    phase_hyb, Q = block_diagonalize_hyb(hyb_sph)

    if verbose:
        print("Block diagonalizing the hybridization function.")
        matrix_print(
            Q,
            "Unitary transformation Q (from Spherical to block-diagonal fitting basis):",
        )
        print()

    block_structure = build_block_structure(phase_hyb, tol=1e-15)

    H_imp = H_dft_sph
    H_local_Q = np.conjugate(Q.T) @ H_imp @ Q

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
    H, *_ = assemble_h0(
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

    h_op = matrixToIOp(H)
    write_to_file(h_op, f"{cluster}_h0_op", save_as_dict=True)

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

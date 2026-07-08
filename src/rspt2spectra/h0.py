"""High-level driver: real-frequency hybridization function -> impurity h0.

The pipeline is split in three steps so callers can cache or persist the
(expensive) bath fit between them:

1. :func:`prepare_hyb_fit` -- block-diagonalize the hybridization function,
   rotate the local Hamiltonian into the fitting basis and build the block
   partition from the *union* of both connectivities.
2. :func:`rspt2spectra.hyb_fit.fit_hyb` -- fit star bath energies and
   hoppings per inequivalent block (callers may reuse a stored fit instead).
3. :func:`assemble_h0` -- turn the (flattened) star fit into the requested
   bath geometry and assemble the full one-particle Hamiltonian.

Everything operates on plain numpy arrays; the energy unit follows the
inputs (e.g. Ry when the data comes from RSPt).
"""

import logging

import numpy as np

from rspt2spectra.block_structure import build_block_structure, print_block_structure
from rspt2spectra.edchain import build_full_bath, build_H_bath_v, build_imp_bath_blocks
from rspt2spectra.utils import block_diagonalize_hyb, matrix_print, rotate_matrix

logger = logging.getLogger(__name__)


def prepare_hyb_fit(hyb, H_local, tol=1e-6, verbose=True):
    """Prepare the fitting basis and block partition for a hybridization fit.

    The hybridization function is rotated into a basis where each block is
    (hopefully) close to diagonal, ``phase_hyb = Q^dagger hyb Q``. The local
    Hamiltonian is rotated into the same basis. The block partition is built
    from the union of the connectivity of both: the bath geometries (in
    particular the linked double chain) are anchored on the local-Hamiltonian
    block of each hybridization block, so every orbital pair coupled by
    either must end up in the same block. Block equivalence
    (identical/transposed/particle-hole) is likewise tested against both.

    Parameters
    ----------
    hyb : np.ndarray of shape (n_w, n_orb, n_orb)
        Real-frequency hybridization function.
    H_local : np.ndarray of shape (n_orb, n_orb)
        Local (impurity) Hamiltonian used to anchor the bath geometry,
        including any double-counting contribution the caller wants the
        geometry construction to see.
    tol : float, default 1e-6
        Connectivity/equivalence tolerance.
    verbose : bool, default True
        Print the resulting block structure.

    Returns
    -------
    Q : np.ndarray of shape (n_orb, n_orb)
        Rotation to the fitting basis.
    phase_hyb : np.ndarray of shape (n_w, n_orb, n_orb)
        The hybridization function in the fitting basis.
    H_local_Q : np.ndarray of shape (n_orb, n_orb)
        The local Hamiltonian in the fitting basis.
    block_structure : BlockStructure
        Partition and equivalence information on the combined connectivity.
    """
    phase_hyb, Q = block_diagonalize_hyb(hyb, tol=tol)
    H_local_Q = rotate_matrix(H_local, Q)
    block_structure = build_block_structure(phase_hyb, mat=H_local_Q, tol=tol)
    # Guaranteed by the union connectivity above; guard against regressions.
    off_block = np.abs(H_local_Q.copy())
    for orbs in block_structure.blocks:
        off_block[np.ix_(orbs, orbs)] = 0
    if np.max(off_block) > tol:
        raise RuntimeError(
            "The local hamiltonian is not block diagonal on the combined "
            f"block partition. Max off-block element: {np.max(off_block):.3e}"
        )
    if verbose:
        print_block_structure(block_structure)
    return Q, phase_hyb, H_local_Q, block_structure


def flatten_star_levels(ebs, vs, coupling_tol=1e-6, verbose=False):
    r"""
    Split each fitted bath level into its coupled orbital components.

    A fitted bath level at energy e carries an (n_orb x n_orb) hopping matrix
    v; the level expands into n_orb degenerate bath orbitals with hopping rows
    v[b, :]. If v is rank deficient (common when the block structure merges
    orbitals whose hybridization is block diagonal, e.g. orbitals only coupled
    through the local hamiltonian), some unitary combinations of the level's
    bath orbitals decouple from the impurity. The chain constructions
    (Lanczos) silently drop such decoupled orbitals, which would leave the
    star and chain geometries with different numbers of bath states and break
    the positional valence/conduction classification.

    Rotate the degenerate orbitals of each level with the SVD
    :math:`v = U S W^\dagger` (the level energy block :math:`e\,\mathbb{1}` is
    invariant, the hopping becomes :math:`U^\dagger v = S W^\dagger`) and keep
    only rows with singular value above coupling_tol. The result is packed in
    the flat (N, 1, n_orb) form accepted by all bath geometry builders.

    Returns
    -------
    ebs_flat : np.ndarray of shape (N,)
        Bath energies, one per kept bath orbital.
    vs_flat : np.ndarray of shape (N, 1, n_orb)
        Hopping rows.
    """
    n_imp = vs.shape[2]
    flat_e = []
    flat_v = []
    n_dropped = 0
    for e, v in zip(ebs, vs):
        if v.shape[0] == 1:
            # Already flat; keep the row as is (an SVD would only change the
            # gauge), just drop it if it is uncoupled.
            if np.linalg.norm(v[0]) > coupling_tol:
                flat_e.append(e)
                flat_v.append(v[0])
            else:
                n_dropped += 1
            continue
        _, s, wh = np.linalg.svd(v)
        for s_k, row in zip(s, wh[: len(s)]):
            if s_k > coupling_tol:
                flat_e.append(e)
                flat_v.append(s_k * row)
            else:
                n_dropped += 1
    if n_dropped > 0 and verbose:
        print(
            f"Dropped {n_dropped} bath orbitals that do not couple to the impurity (rank deficient bath level hopping)."
        )
    return (
        np.array(flat_e, dtype=float),
        np.array(flat_v, dtype=complex).reshape((len(flat_e), 1, n_imp)),
    )


def assemble_h0(
    ebs_star,
    vs_star,
    shifts,
    H_imp,
    H_local_Q,
    Q,
    block_structure,
    bath_geometry="star",
    w=None,
    eim=0.0,
    label=None,
    verbose=True,
    extra_verbose=False,
    comm=None,
):
    """Assemble the full one-particle Hamiltonian from a star bath fit.

    In block form the result is
    ``[[H_imp + shift, V^dagger], [V, H_bath]]``,
    where the impurity block is expressed in the caller's input basis and the
    bath block in the requested geometry (built in the fitting basis and
    coupled back through Q).

    The fit models the hybridization as ``Delta_fit(z) = Delta_pole(z) + C``;
    the discrete poles carry no constant part, so matching RSPt's Weiss field
    ``g0^-1 = z - H_imp - Delta(z)`` with the model ``G0^-1 = z - E_imp -
    Delta_pole(z)`` requires the constant to move into the impurity level:
    ``E_imp = H_imp + C``.

    Parameters
    ----------
    ebs_star : list of np.ndarray
        Fitted bath energies per inequivalent block (flattened form, see
        :func:`flatten_star_levels`).
    vs_star : list of np.ndarray
        Fitted bath hoppings per inequivalent block, shape (N, 1, n_block).
    shifts : list of np.ndarray
        Constant Hermitian shift fitted per inequivalent block (the third
        return value of :func:`rspt2spectra.hyb_fit.fit_hyb`).
    H_imp : np.ndarray of shape (n_orb, n_orb)
        Impurity block of the Hamiltonian, in the caller's input basis.
    H_local_Q : np.ndarray of shape (n_orb, n_orb)
        Local Hamiltonian in the fitting basis (from
        :func:`prepare_hyb_fit`); anchors the chain constructions.
    Q : np.ndarray of shape (n_orb, n_orb)
        Rotation to the fitting basis (from :func:`prepare_hyb_fit`).
    block_structure : BlockStructure
        Partition returned by :func:`prepare_hyb_fit`.
    bath_geometry : str, default "star"
        One of the geometries supported by
        :func:`rspt2spectra.edchain.build_H_bath_v` (e.g. "star", "chain",
        "haver").
    w : np.ndarray, optional
        Real-frequency mesh; used (subsampled) for the star-vs-chain impurity
        Green's function consistency check. Required for non-star geometries.
    eim : float, default 0.0
        Imaginary offset used in the consistency check.
    label : str, optional
        Cluster label used when extra_verbose dumps the Hamiltonian to file.
    verbose, extra_verbose : bool
        Diagnostic printing.
    comm : MPI communicator, optional
        If given, the bath Hamiltonian and couplings are broadcast from rank 0.

    Returns
    -------
    H : np.ndarray
        The full one-particle Hamiltonian in the requested geometry.
    H_star : np.ndarray
        The same Hamiltonian in the star geometry (equal to ``H`` when
        ``bath_geometry == "star"``); its bath diagonal classifies the bath
        states.
    impurity_indices, valence_bath_indices, conduction_bath_indices : list of int
        Orbital index classification based on the star geometry.
    v_solver : np.ndarray
        Impurity-bath coupling in the caller's input basis.
    H_bath : np.ndarray
        The bath block of ``H``.
    """
    n_orb = H_imp.shape[0]
    H_shift = np.zeros_like(H_imp)
    for inequiv_block_i, shift in zip(block_structure.inequivalent_blocks, shifts):
        for block_i in block_structure.identical_blocks[inequiv_block_i]:
            orbs = block_structure.blocks[block_i]
            H_shift[np.ix_(orbs, orbs)] = shift
    if verbose:
        matrix_print(H_shift, r"Shift of $\Delta(\omega=0)$")
    # Any double counting must be present in H_local_Q: the linked double
    # chain geometry links the valence and conduction chains through the
    # local hamiltonian block, so leaving it out would produce two separate
    # chains that only link to the impurity, not to each other.
    H_baths, vs = build_H_bath_v(
        H_local_Q + H_shift,
        ebs_star,
        vs_star,
        bath_geometry,
        block_structure,
        verbose,
        extra_verbose,
    )
    H_bath, v = build_full_bath(H_baths, vs, block_structure)
    if comm is not None:
        comm.Bcast(H_bath)
        comm.Bcast(v)

    H = np.zeros((n_orb + H_bath.shape[0], n_orb + H_bath.shape[0]), dtype=complex)
    H[:n_orb, :n_orb] = H_imp + rotate_matrix(H_shift, np.conj(Q.T))
    H[n_orb:, n_orb:] = H_bath
    H[n_orb:, :n_orb] = v @ np.conj(Q.T)
    H[:n_orb, n_orb:] = np.conj(H[n_orb:, :n_orb].T)

    if verbose:
        print(f"Total number of spin orbitals: {H.shape[0]}")
        print(f"----> Impurity orbitals: {n_orb}")
        print(f"----> Bath orbitals: {H_bath.shape[0]}")

    # The star geometry hamiltonian is needed for classifying bath states as
    # valence/conduction (the star diagonal holds the bath energies) and for
    # the consistency check of the chain construction.
    if bath_geometry == "star":
        # build_H_bath_v with "star" would rebuild exactly H.
        H_star = H
    else:
        H_baths_star, vs_star_geom = build_H_bath_v(
            H_local_Q + H_shift,
            ebs_star,
            vs_star,
            "star",
            block_structure,
            verbose,
            extra_verbose,
        )
        H_bath_star, v_star = build_full_bath(H_baths_star, vs_star_geom, block_structure)
        H_star = np.zeros((n_orb + H_bath_star.shape[0], n_orb + H_bath_star.shape[0]), dtype=complex)
        H_star[:n_orb, :n_orb] = H_imp + rotate_matrix(H_shift, np.conj(Q.T))
        H_star[n_orb:, n_orb:] = H_bath_star
        H_star[n_orb:, :n_orb] = v_star @ np.conj(Q.T)
        H_star[:n_orb, n_orb:] = np.conj(H_star[n_orb:, :n_orb].T)
        if H.shape != H_star.shape:
            # The chain constructions drop bath states that decouple from the
            # impurity. The uncoupled states are pruned right after the fit
            # (flatten_star_levels), so a size mismatch here means the
            # valence/conduction classification below would be invalid.
            raise RuntimeError(
                f"The {bath_geometry} bath construction changed the number of "
                f"bath states ({H_star.shape[0] - n_orb} -> {H.shape[0] - n_orb}). "
                "Cannot classify bath states as valence/conduction."
            )
        # The star and chain geometries must describe the same impurity
        # physics; compare the impurity-projected Green's functions.
        if w is not None:
            z_check = w[::10] + 1j * eim
            G0 = np.linalg.inv(z_check[:, None, None] * np.identity(H.shape[0])[None] - H[None])[:, :n_orb, :n_orb]
            G0_star = np.linalg.inv(z_check[:, None, None] * np.identity(H_star.shape[0])[None] - H_star[None])[
                :, :n_orb, :n_orb
            ]
            if not np.allclose(G0, G0_star, atol=1e-8):
                warning = (
                    "WARNING: The bath geometry transformation changed the impurity "
                    "Green's function!\n"
                    f"Max abs deviation: {np.max(np.abs(G0 - G0_star)):.3e}"
                )
                # stdout may be redirected to a file; the logger keeps the
                # warning visible on the terminal (stderr) as well.
                print(warning, flush=True)
                logger.warning(warning)

    if extra_verbose:
        print("Local hamiltonian, with baths, in solver basis")
        matrix_print(H)
        print("=" * 80)

        print()
        print("Local hamiltonian, with star geometry baths, in solver basis")
        matrix_print(H_star)
        print("=" * 80, flush=True)
        with open(f"Ham-{label}.inp", "w") as f:
            for i in range(H_star.shape[0]):
                for j in range(H_star.shape[1]):
                    f.write(f" 0 0 0 {i + 1} {j + 1} {np.real(H_star[i, j])} {np.imag(H_star[i, j])}\n")
    impurity_indices, valence_bath_indices, conduction_bath_indices = build_imp_bath_blocks(H_star, n_orb)

    return (
        H,
        H_star,
        impurity_indices,
        valence_bath_indices,
        conduction_bath_indices,
        v @ np.conj(Q.T),
        H_bath,
    )

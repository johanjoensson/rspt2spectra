"""High-level driver: real-frequency hybridization function -> impurity h0.

The pipeline is split in three steps so callers can cache or persist the
(expensive) bath fit between them:

1. :func:`prepare_hyb_fit` -- block-diagonalize the hybridization function,
   rotate the local Hamiltonian into the fitting basis and build the block
   partition from the *union* of both connectivities.  Returns a :class:`FitBasis`,
   which also carries a fingerprint of the inputs for deciding whether a stored
   fit can be reused.
2. :func:`fit_bath` -- fit star bath energies and hoppings per inequivalent block
   (callers may hand in a stored fit instead, or the previous fit's energies to
   start from).  Storing fits is the caller's business.
3. :func:`assemble_h0` -- turn the (flattened) star fit into the requested
   bath geometry and assemble the full one-particle Hamiltonian.

Everything operates on plain numpy arrays; the energy unit follows the
inputs (e.g. Ry when the data comes from RSPt).
"""

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

from rspt2spectra.block_structure import (
    build_block_structure,
    get_blocks,
    mat_scale_with_hyb,
    print_block_structure,
    project_to_blocks,
)
from rspt2spectra.edchain import build_full_bath, build_H_bath_v, build_imp_bath_blocks
from rspt2spectra.fit_basis import block_diagonalize_hyb
from rspt2spectra.fit_options import DEFAULT_SYMM_PREC
from rspt2spectra.hyb_fit import fit_hyb
from rspt2spectra.utils import matrix_print, rotate_matrix
from rspt2spectra.weight_functions import weight_functions

logger = logging.getLogger(__name__)

# Numeric equality tolerance for recognizing symmetry-equivalent orbital blocks
# (spin-degenerate partners, or point-group-equivalent manifolds like a cubic shell's eg/t2g)
# in `build_block_structure`. Measured on real RSPt-derived NiO data: orbitals that are
# equivalent by an *exact* symmetry (spin degeneracy with no SOC, or a cubic crystal field)
# differ only by ~1.4e-10 between each other -- floating-point/text-I/O noise from the
# RSPt -> .dat -> fit pipeline, not physics. This constant sits ~4 orders of magnitude above
# that measured noise floor (so real noise never blocks a merge that should happen) and ~4
# orders of magnitude below the energy scales fit here (so it does not risk merging two
# genuinely distinct orbitals). A far tighter value (e.g. 1e-15, machine epsilon) fails to
# merge real symmetry-equivalent blocks and silently fits them as independent, differently-
# seeded stochastic optimizations -- producing a Hamiltonian that looks nearly, but not
# exactly, symmetric.
BLOCK_EQUIVALENCE_TOL = 1e-6


#: Identifies how the fitting basis and the block partition are decided.  Anything stored that
#: depends on them (a reusable bath fit) must be keyed on it: a changed rotation changes the basis
#: even where the partition does not change.
BLOCKING_VERSION = "joint-jacobi-v3;h-weight=0.1;h-scale=max(spread,min-broadening);u-aware-shapes-partition-v2"


def _partition(hyb, H, tol):
    return frozenset(frozenset(int(i) for i in b) for b in get_blocks(hyb, H, tol=tol))


def split_margin(hyb, H, tol, limit=1e3, bisections=24):
    """The range ``(lo, hi)`` of ``symm_prec`` over which the partition at ``tol`` does not change.

    ``lo`` is ``None`` when lowering the threshold never changes it (below the absolute coupling
    floor the threshold stops falling); ``hi`` is ``None`` when raising it up to ``limit`` never
    does.  The partition is monotone in the threshold (a higher one only neglects more), so each
    end is found by doubling and then bisecting in the logarithm.
    """
    ref = _partition(hyb, H, tol)

    def edge(direction):
        inside, step = tol, 2.0**direction
        outside = None
        for _ in range(60):
            trial = inside * step
            if trial > limit or trial < 1e-14:
                return None
            if _partition(hyb, H, trial) != ref:
                outside = trial
                break
            inside = trial
        if outside is None:
            return None
        for _ in range(bisections):
            mid = np.sqrt(inside * outside)
            if _partition(hyb, H, mid) == ref:
                inside = mid
            else:
                outside = mid
        return inside

    return edge(-1), edge(+1)


def _print_split_margin(hyb, H, tol):
    lo, hi = split_margin(hyb, H, tol)
    lo_txt = f"{lo:.3g}" if lo is not None else "0 (absolute coupling floor)"
    hi_txt = f"{hi:.3g}" if hi is not None else "no upper limit"
    print(
        f"Partition holds for symm_prec in [{lo_txt}, {hi_txt}] (set {tol:g}); "
        "pick a value well inside this window so DMFT iterations do not flip it."
    )


def _print_neglected(hyb, H, blocks, tol):
    """Print the size of the couplings between blocks that are about to be neglected."""
    n = H.shape[0]
    mask = np.ones((n, n), dtype=bool)
    for block in blocks:
        mask[np.ix_(block, block)] = False
    if not mask.any():
        return
    g_scale = max(float(np.max(np.abs(np.einsum("wii->wi", hyb)))), 1e-300)
    hyb_off = float(np.max(np.abs(hyb[:, mask])))
    l2_off = float(np.sqrt(np.sum(np.abs(hyb[:, mask]) ** 2)))
    l2_all = float(np.sqrt(np.sum(np.abs(hyb) ** 2)))
    h_off = float(np.max(np.abs(H[mask])))
    h_scale = mat_scale_with_hyb(H, hyb)
    print(
        f"Neglected couplings between blocks (symm_prec {tol:g}):\n"
        f"  hybridization: max {hyb_off / g_scale:.3g} of the strongest diagonal "
        f"({hyb_off:.3g} Ry), L2 fraction {l2_off / max(l2_all, 1e-300):.3g}\n"
        f"  local H      : max {h_off:.3g} Ry = {h_off / h_scale:.3g} of its scale "
        f"({h_scale:.3g} Ry; crystal-field spread or smallest hybridization width)"
    )


def fit_window(w, hyb, valence_bath_only):
    """Return the ``(lo, hi)`` frequency window the bath fit uses.

    From the first to the last frequency at which the trace of the hybridization function exceeds
    1e-6, cut at the Fermi level when only the valence bath is fitted.  The trace is invariant under
    the unitary rotation to the fitting basis and under dropping off-block couplings, so the window
    is the same for ``hyb`` in any basis.
    """
    trace = np.abs(np.trace(hyb, axis1=1, axis2=2))
    w_min = w[np.argmax(trace > 1e-6)]
    w_max = w[-(np.argmax(trace[::-1] > 1e-6) + 1)]
    return (w_min, min(0, w_max) if valence_bath_only else w_max)


def fit_fingerprint(hyb, H_local, u4, to_spherical, symm_prec, u_densify, x_lim):
    """Return a hex digest of what the fit is made from.

    A caller that stores fits compares it to decide whether a stored fit can be reused (identical
    input) or has to be redone (a new DMFT iteration).  Only inputs are hashed, never the output of
    the rotation, so the decision does not depend on BLAS details.
    """
    u_aware = u4 is not None and np.isfinite(u_densify)
    return hashlib.sha256(
        np.ascontiguousarray(hyb).tobytes()
        + np.ascontiguousarray(H_local).tobytes()
        # the Coulomb tensor and shell rotation steer the basis choice, but only when the choice looks at them
        + (np.ascontiguousarray(u4).tobytes() + repr(float(u_densify)).encode() if u_aware else b"")
        + (np.ascontiguousarray(to_spherical).tobytes() if u_aware and to_spherical is not None else b"")
        + repr(
            (
                float(symm_prec),
                None if x_lim is None else tuple(float(x) for x in x_lim),
                # how the basis is chosen: a changed rotation must not reuse a fit made in the old one
                BLOCKING_VERSION,
            )
        ).encode()
    ).hexdigest()


def bath_fit_fingerprint(basis, w, eim, n_baths, options):
    """Return a hex digest of everything a stored star fit depends on.

    ``basis.fingerprint`` covers the hybridization and the basis choice; this adds what
    :func:`fit_bath` itself is told: the number of bath states, the real-axis mesh and its
    broadening, the regularisation, the weight function and the fit window.  A caller that reuses a
    stored fit compares this one, so a changed ``nBaths`` or ``gamma`` is never answered with the
    old fit.
    """
    return hashlib.sha256(
        basis.fingerprint.encode()
        + np.ascontiguousarray(w, dtype=float).tobytes()
        + repr(
            (
                int(n_baths),
                float(eim),
                float(options.gamma),
                options.weight_function,
                float(options.weight),
                float(options.weight_w0),
                bool(options.fit_unocc),
            )
        ).encode()
    ).hexdigest()


def bath_hybridization(w, eim, H_bath, v):
    r"""Return the hybridization function of a bath, :math:`V^\dagger (z - H_{bath})^{-1} V`.

    ``z = w + i*eim``; ``H_bath`` is (n_bath, n_bath), ``v`` is (n_bath, n_imp); the result has
    shape (n_w, n_imp, n_imp).
    """
    z = (w + 1j * eim)[:, None, None]
    identity = np.identity(H_bath.shape[0], dtype=complex)[None, :, :]
    return np.conj(v).T @ np.linalg.solve(z * identity - H_bath, v)


@dataclass
class FitBasis:
    """The fitting basis and block partition :func:`prepare_hyb_fit` decided on.

    Everything is in the fitting basis ``Q`` (columns: fitting orbitals expressed in the input
    orbitals), where ``phase_hyb`` and ``H_local_Q`` are exactly block diagonal.
    """

    Q: np.ndarray
    phase_hyb: np.ndarray
    H_local_Q: np.ndarray
    block_structure: object
    #: The ``(lo, hi)`` window the decision looked at (``None``: the whole mesh).
    x_lim: tuple
    #: Whether ``tol`` neglected a coupling that the default threshold would have kept.
    neglected_couplings: bool
    #: See :func:`fit_fingerprint`.
    fingerprint: str


def _decide_on_root(comm, decide):
    """Run ``decide()`` on rank 0 and give every rank its result.

    The rotation and the partition come out of iterative floating-point work whose last bits can
    differ between ranks on mixed hardware; a different partition would give ranks different numbers
    of collective calls in the fit.  A failure on rank 0 is broadcast and re-raised everywhere, so no
    rank is left waiting in the broadcast.
    """
    if comm is None:
        return decide()
    outcome = None
    if comm.rank == 0:
        try:
            outcome = (None, decide())
        except Exception as exc:  # re-raised on every rank below
            outcome = (f"{type(exc).__name__}: {exc}", None)
    outcome = comm.bcast(outcome, root=0)
    if outcome[0] is not None:
        raise RuntimeError(f"Fit-basis search failed on rank 0: {outcome[0]}")
    return outcome[1]


def prepare_hyb_fit(
    hyb,
    H_local,
    tol=BLOCK_EQUIVALENCE_TOL,
    verbose=True,
    w=None,
    x_lim=None,
    equiv_tol=None,
    report=None,
    u4=None,
    to_spherical=None,
    max_u_densify=2.0,
    comm=None,
):
    """Prepare the fitting basis and block partition for a hybridization fit.

    The hybridization function and the local Hamiltonian are jointly rotated to a basis
    that is as diagonal as possible, ``phase_hyb = Q^dagger hyb Q`` (see
    :func:`rspt2spectra.fit_basis.block_diagonalize_hyb`).  The block partition is the
    connectivity of both in that basis, keeping only couplings above ``tol`` relative
    to the strongest diagonal hybridization and to the crystal-field spread of
    ``H_local``.  Weaker couplings are *neglected*: ``phase_hyb`` and ``H_local_Q``
    are returned exactly block diagonal, so everything built from them (bath, impurity
    block) carries no coupling between blocks.  The bath geometries are anchored on
    the local-Hamiltonian block of each hybridization block, so an orbital pair coupled
    by either must end up in the same block.

    Parameters
    ----------
    hyb : np.ndarray of shape (n_w, n_orb, n_orb)
        Real-frequency hybridization function.
    H_local : np.ndarray of shape (n_orb, n_orb)
        Local (impurity) Hamiltonian used to anchor the bath geometry,
        including any double-counting contribution the caller wants the
        geometry construction to see.
    tol : float, default 1e-6
        Relative connectivity threshold; raise it to neglect weak couplings.
    verbose : bool, default True
        Print the resulting block structure.
    w : np.ndarray of shape (n_w,), optional
        Frequency mesh of ``hyb``. Needed to test particle-hole equivalence
        between blocks, which relates ``omega`` to ``-omega``; without it those
        relations are skipped, and to apply ``x_lim``.
    x_lim : tuple of float, optional
        ``(lo, hi)`` fit window.  Rotation and connectivity only look at ``hyb``
        inside it.
    equiv_tol : float, optional
        Absolute tolerance for identical/transposed/particle-hole equivalence of
        blocks.  Defaults to ``BLOCK_EQUIVALENCE_TOL`` and is deliberately
        independent of ``tol``: raising ``tol`` must not merge inequivalent blocks.
    u4 : np.ndarray of shape (n_orb,) * 4, optional
        Coulomb tensor in the input basis.  If given, the fitting basis is chosen so that it does
        not densify ``u4`` by more than ``max_u_densify``: a rigid (SO(3)) rotation of the shell
        when that separates the orbitals as well as a free rotation, else a free rotation within
        the budget (see :func:`rspt2spectra.fit_basis.block_diagonalize_hyb`).
    to_spherical : np.ndarray of shape (n_orb, n_sph), optional
        Input orbitals -> complex spherical harmonics; enables the SO(3) choice for a full shell.
    max_u_densify : float, default 2.0
        Allowed growth of the number of Coulomb-tensor terms; ``inf`` disables the constraint.
    comm : mpi4py communicator, optional
        Rank 0 decides the basis and the partition and the other ranks receive them (see
        :func:`_decide_on_root`).

    Returns
    -------
    FitBasis
        The rotation ``Q`` to the fitting basis, the hybridization function ``phase_hyb`` and the
        local Hamiltonian ``H_local_Q`` in it (both exactly block diagonal), the partition and
        equivalence information ``block_structure`` on the combined connectivity, and whether
        ``tol`` neglected anything.
    """
    args = (hyb, H_local, tol, verbose, w, x_lim, equiv_tol, report, u4, to_spherical, max_u_densify)
    return _decide_on_root(comm, lambda: _prepare_hyb_fit(*args))


def _prepare_hyb_fit(hyb, H_local, tol, verbose, w, x_lim, equiv_tol, report, u4, to_spherical, max_u_densify):
    if equiv_tol is None:
        equiv_tol = BLOCK_EQUIVALENCE_TOL
    basis_info = {}
    phase_hyb, Q = block_diagonalize_hyb(
        hyb,
        mat=H_local,
        w=w,
        x_lim=x_lim,
        u4=u4,
        to_spherical=to_spherical,
        max_u_densify=max_u_densify,
        tol=tol,
        info=basis_info,
    )
    if verbose and basis_info:
        n_in, n_out = basis_info["u_terms_input"], basis_info["u_terms"]
        print(
            f"Fitting basis: {basis_info['tier']}{', ' + basis_info['shape'] if basis_info.get('shape') else ''}; Coulomb tensor terms {n_in} -> {n_out} "
            f"(x{n_out / max(n_in, 1):.2f}, allowed x{max_u_densify:g})."
        )
    H_local_rot = rotate_matrix(H_local, Q)
    window = slice(None)
    if x_lim is not None:
        assert w is not None, "x_lim needs the frequency mesh w"
        in_window = (w >= x_lim[0]) & (w <= x_lim[1])
        if np.any(in_window):
            window = in_window
    blocks = get_blocks(phase_hyb[window], H_local_rot, tol=tol)
    # Compared in the fitting basis, the one the partition lives in: the same data in the input
    # basis is denser and would give fewer blocks whatever was neglected.
    neglected = len(blocks) > len(get_blocks(phase_hyb[window], H_local_rot, tol=DEFAULT_SYMM_PREC))
    if verbose if report is None else report:
        _print_neglected(phase_hyb[window], H_local_rot, blocks, tol)
        _print_split_margin(phase_hyb[window], H_local_rot, tol)
    phase_hyb = project_to_blocks(phase_hyb, blocks)
    H_local_Q = project_to_blocks(H_local_rot, blocks)
    # The partition was decided on the fit window; the equivalences between the blocks are
    # tested on the whole mesh (particle-hole relates omega to -omega) without deciding it again.
    block_structure = build_block_structure(phase_hyb, mat=H_local_Q, tol=tol, w=w, equiv_tol=equiv_tol, blocks=blocks)
    if verbose:
        print_block_structure(block_structure)
    return FitBasis(
        Q=Q,
        phase_hyb=phase_hyb,
        H_local_Q=H_local_Q,
        block_structure=block_structure,
        x_lim=x_lim,
        neglected_couplings=neglected,
        fingerprint=fit_fingerprint(hyb, H_local, u4, to_spherical, tol, max_u_densify, x_lim),
    )


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


@dataclass
class BathFit:
    """A star bath fit: one entry per inequivalent block, levels sorted and flattened."""

    #: Bath energies, one array of shape (N,) per block.
    ebs_star: list
    #: Hopping rows, one array of shape (N, 1, n_block) per block.
    vs_star: list
    #: Constant Hermitian shift, one per block.
    shifts: list
    #: One entry per block, ``{"status": ..., "cost warm": ..., "cost global": ...}``, describing
    #: how a warm start was decided; ``None`` when a stored fit was used and nothing was fitted.
    warm_start_info: list = None


def fit_bath(
    basis,
    w,
    eim,
    n_baths,
    options,
    *,
    stored_fit=None,
    previous_energies=None,
    verbose=True,
    comm=None,
):
    """Fit star bath states to the hybridization function of a :class:`FitBasis`.

    Parameters
    ----------
    basis : FitBasis
        From :func:`prepare_hyb_fit`.
    w : np.ndarray of shape (n_w,)
        Real-frequency mesh of ``basis.phase_hyb``.
    eim : float
        Imaginary offset of the evaluation.
    n_baths : int
        Bath states to fit per impurity orbital.
    options : FitOptions
        ``gamma``, the weight function, ``fit_unocc``, ``freeze_bath_energies`` and ``warm_tol``
        are used here.
    stored_fit : tuple of (ebs_star, vs_star, shifts), optional
        A fit stored earlier from the same ``basis.fingerprint`` and block structure.  If given,
        nothing is fitted; the levels are only sorted and flattened.
    previous_energies : list of np.ndarray, optional
        The previous fit's bath energies per inequivalent block.  With
        ``options.freeze_bath_energies`` they are held fixed and only the hoppings are re-solved;
        otherwise the fit starts from them and keeps that warm fit unless a global search beats
        its cost by more than ``options.warm_tol``.
    verbose : bool
        Print the fitted levels.
    comm : mpi4py communicator, optional
        The fit is collective.

    Returns
    -------
    BathFit
    """
    block_structure = basis.block_structure
    warm_start_info = None
    if stored_fit is not None:
        ebs_star, vs_star, shifts = (list(part) for part in stored_fit)
    else:
        warm_start_info = []
        x_lim = basis.x_lim
        if x_lim is None:
            x_lim = fit_window(w, basis.phase_hyb, not options.fit_unocc)
        ebs_star, vs_star, shifts = fit_hyb(
            w,
            eim,
            basis.phase_hyb,
            n_baths,
            block_structure,
            gamma=options.gamma,
            x_lim=x_lim,
            verbose=verbose,
            comm=comm,
            weight_fun=weight_functions[options.weight_function](options.weight_w0, options.weight),
            ebs_guess=previous_energies,
            optimize_bath_energies=not (options.freeze_bath_energies and previous_energies is not None),
            warm_accept_tol=options.warm_tol,
            warm_start_info=warm_start_info,
        )
    for i in range(len(ebs_star)):
        if len(ebs_star[i]) == 0:
            continue
        order = np.argsort(ebs_star[i], kind="stable")
        ebs_star[i], vs_star[i] = flatten_star_levels(ebs_star[i][order], vs_star[i][order], verbose=verbose)
    assert len(vs_star) == len(block_structure.inequivalent_blocks), "Number of inequivalent blocks is inconsistent"
    if verbose:
        _print_star_fit(block_structure, ebs_star, vs_star)
    return BathFit(ebs_star, vs_star, shifts, warm_start_info)


def _print_star_fit(block_structure, ebs_star, vs_star):
    print("Star bath energies and hopping parameters:")
    for bi, (eb, vb) in enumerate(zip(ebs_star, vs_star)):
        print(
            f"Energy   :  Hopping  (impurity orbitals "
            f"{block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
        )
        for eb_i, vb_i in zip(eb, vb):
            print(f"{eb_i: 9.6f}:  ", "  ".join(f"{val: 9.6f}" for vb_row in vb_i for val in vb_row))
        print("")
    print("=" * 80)


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
    peel_weight=0.05,
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
        Impurity block of the Hamiltonian, in the caller's input basis.  Couplings
        between blocks are neglected (zeroed) before use.
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
        "linked_chain", "peeled_linked_chain").
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
    peel_weight : float, default 0.05
        Weight-fraction threshold for ``bath_geometry == "peeled_linked_chain"``
        (see :func:`rspt2spectra.edchain.peel_resonant_modes`); ignored by the
        other geometries.

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
    H_imp : np.ndarray
        The (block-projected) effective impurity block ``H[:n_orb, :n_orb]`` -- the DFT block with the fitted
        constant hybridization offset folded in (``E_imp = H_imp + C``). Returned explicitly so
        callers can hand the ``(H_imp, V, H_bath)`` blocks straight to a solver without
        re-slicing ``H``.
    """
    n_orb = H_imp.shape[0]
    # Couplings between blocks were neglected in the fit (see prepare_hyb_fit), so the
    # impurity block must not carry them either.  Project in the fitting basis and rotate
    # back; this is a plain projection of H_imp when Q is block diagonal.
    H_imp = rotate_matrix(project_to_blocks(rotate_matrix(H_imp, Q), block_structure.blocks), np.conj(Q.T))
    H_shift = np.zeros_like(H_imp)
    # The constant offset is part of the fitted hybridization, so it follows the
    # same equivalence relations as the bath: a transposed block gets C^T and a
    # particle-hole equivalent one gets -conj(C).  Broadcasting it over identical
    # blocks only would leave every other equivalent block with no impurity-level
    # shift at all.
    for inequiv_block_i, shift in zip(block_structure.inequivalent_blocks, shifts):
        related = (
            (block_structure.identical_blocks[inequiv_block_i], shift),
            (block_structure.transposed_blocks[inequiv_block_i], shift.T),
            (block_structure.particle_hole_blocks[inequiv_block_i], -np.conj(shift)),
            (
                block_structure.particle_hole_transposed_blocks[inequiv_block_i],
                -np.conj(shift).T,
            ),
        )
        for block_indices, block_shift in related:
            for block_i in block_indices:
                orbs = block_structure.blocks[block_i]
                H_shift[np.ix_(orbs, orbs)] = block_shift
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
        peel_weight=peel_weight,
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
        H[:n_orb, :n_orb],
    )

"""
Fit a discrete bath to a real-frequency hybridization function.

`fit_hyb` distributes a pool of bath states over the inequivalent blocks of
the hybridization function and fits each block with `fit_block`: bath-energy
guesses are seeded from the peaks of the hybridization spectrum, then
optimized with the VARPRO basin-hopping search in
:mod:`rspt2spectra.offdiagonal`. When an MPI communicator is passed, each
rank fits from its own seeds and the lowest-cost fit is kept.
"""

import functools
from dataclasses import replace

try:
    from mpi4py import MPI
except (
    ImportError,
    RuntimeError,
):  # pragma: no cover - MPI is optional; serial fits work without it
    MPI = None
import numpy as np
import scipy as sp
from scipy.signal import find_peaks, peak_widths

from .offdiagonal import (
    _max_bath_states,
    free_window,
    get_v_and_eb_varpro_basin_hopping,
    max_bath_states,
)
from .symmetries import (
    DEFAULT_SYMMETRY_TOL,
    describe,
    detect_block_symmetry,
    trivial_symmetry,
)

_LINE_WIDTH = 72

PARTICLE_HOLE_WINDOW_FRACTION = 1.0 / 3.0
"""Smallest share of the fit window that must be symmetric about the Fermi level.

Enforcing particle-hole symmetry confines the bath to that symmetric part, so a
strongly lopsided window would trade a much worse fit for the symmetry.  One
third admits the ordinary ``--fit-unocc`` case (an RSPt mesh such as
``[-6, 3]``) while rejecting windows that barely cross the Fermi level.
"""


def _rule(title="", char="="):
    """Return a horizontal rule, optionally with a centered title."""
    if not title:
        return char * _LINE_WIDTH
    label = f" {title} "
    pad = max(_LINE_WIDTH - len(label), 2)
    left = pad // 2
    return char * left + label + char * (pad - left)


def _fmt_floats(values, fmt="{: .3f}"):
    """Format a sequence of numbers as a compact, comma-separated string."""
    values = np.real(np.atleast_1d(np.asarray(values, dtype=complex)))
    if values.size == 0:
        return "(none)"
    return ", ".join(fmt.format(v) for v in values)


def _print_block_structure(block_structure):
    """Print a compact table of the block partition and equivalences."""
    rows = [
        ("Blocks", block_structure.blocks),
        ("Inequivalent", block_structure.inequivalent_blocks),
        ("Identical", block_structure.identical_blocks),
        ("Transposed", block_structure.transposed_blocks),
        ("Particle-hole", block_structure.particle_hole_blocks),
        ("Particle-hole + transposed", block_structure.particle_hole_transposed_blocks),
    ]
    width = max(len(label) for label, _ in rows)
    print("Block structure")
    for label, value in rows:
        print(f"  {label:<{width}} : {value}")


def _print_symmetries(block_structure, syms):
    """Print the symmetry detected for each inequivalent block."""
    print("Detected symmetries")
    width = max(len(str(block_structure.blocks[ib])) for ib in block_structure.inequivalent_blocks)
    for sym, block_i in zip(syms, block_structure.inequivalent_blocks, strict=True):
        print(f"  {block_structure.blocks[block_i]!s:<{width}} : {describe(sym)}")


def _print_peaks(positions, left, right, scores):
    """Print the detected hybridization peaks and their seeding scores."""
    if len(positions) == 0:
        print("Peaks: none found (falling back to uniform guesses)")
        return
    print(f"Peaks ({len(positions)})")
    print(f"  {'position':>9}  {'interval':>18}  {'score':>6}")
    for p, l, r, s in zip(positions, left, right, scores):
        print(f"  {p:>9.3f}  [{l:>7.3f}, {r:>7.3f}]  {s:>6.3f}")


def v_opt(a, b, _):
    """MPI reduction: pick the lower-cost of two (eb, v, C, cost) fits."""
    return a if abs(a[-1]) <= abs(b[-1]) else b


@functools.cache
def _get_v_opt_op():
    """Create the lowest-cost-fit MPI reduction op once and reuse it."""
    return MPI.Op.Create(v_opt, commute=True)


def fit_hyb(
    w,
    delta,
    hyb,
    bath_states_per_orbital,
    block_structure,
    gamma,
    x_lim=None,
    verbose=True,
    comm=None,
    weight_fun=np.ones_like,
    ebs_guess=None,
    regularization=None,
    optimize_bath_energies=True,
    enforce_symmetry=True,
    symmetry_tol=DEFAULT_SYMMETRY_TOL,
):
    """Fit bath energies and hoppings to the hybridization function.

    Parameters
    ----------
    w : (M,) np.ndarray
        Real frequency mesh.
    delta : float
        All quantities are evaluated ``i*delta`` above the real axis (the
        broadening grows with ``|w|`` inside the optimizers to focus the fit
        near the Fermi energy).
    hyb : (M, n_orb, n_orb) np.ndarray
        Hybridization function, in the (block-diagonalized) fitting basis.
    bath_states_per_orbital : int
        Average number of bath states per block; distributed over the blocks
        by `get_state_per_inequivalent_block`.
    block_structure : BlockStructure
        Block partition of the hybridization function.
    gamma : float
        Regularization strength for the hopping parameters.
    x_lim : tuple of float, optional
        ``(w_min, w_max)``; fit only frequencies inside this window.
    verbose : bool, default True
        Print fit progress and results.
    comm : MPI communicator, optional
        When given, each rank fits from different seeds and the lowest-cost
        fit is kept on all ranks.
    weight_fun : callable, default ``np.ones_like``
        Energy-dependent fit weight, ``weight_fun(w) -> (M,) array``.
    ebs_guess : list of (n_b,) np.ndarray, optional
        Bath-energy seed per inequivalent block (e.g. from a previous fit). Only the
        energies are seeded; the hoppings are always solved for.
    regularization : {"L1", "L2", "none", None}
        Regularization type for the hopping parameters.
    optimize_bath_energies : bool, default True
        ``False`` freezes the bath energies at ``ebs_guess`` and solves only the
        hoppings (least squares). ``ebs_guess`` is then required.
    enforce_symmetry : bool, default True
        Detect the symmetries of each block and constrain the fitted model to
        satisfy them exactly (see :mod:`rspt2spectra.symmetries`).  ``False``
        fits every block unconstrained.
    symmetry_tol : float
        Relative tolerance for accepting a symmetry; see
        :data:`rspt2spectra.symmetries.DEFAULT_SYMMETRY_TOL`.

    Returns
    -------
    ebs_star : list of (n_b,) np.ndarray
        Fitted bath energies per inequivalent block.
    vs_star : list of (n_b, n_block, n_block) np.ndarray
        Fitted hopping matrices per inequivalent block.
    Cs_star : list of (n_block, n_block) np.ndarray
        Fitted constant (Hermitian) hybridization offset per block.
    """
    n_blocks = len(block_structure.inequivalent_blocks)
    if bath_states_per_orbital == 0:
        return (
            [np.array([], dtype=float) for _ in range(n_blocks)],
            [
                np.empty(
                    (
                        0,
                        len(block_structure.blocks[ib]),
                        len(block_structure.blocks[ib]),
                    ),
                    dtype=complex,
                )
                for ib in block_structure.inequivalent_blocks
            ],
            [
                np.zeros(
                    (len(block_structure.blocks[ib]), len(block_structure.blocks[ib])),
                    dtype=complex,
                )
                for ib in block_structure.inequivalent_blocks
            ],
        )
    mask = np.logical_and(x_lim[0] <= w, w < x_lim[1]) if x_lim is not None else np.ones(len(w), bool)

    if verbose:
        print(_rule("Hybridization fit"))
        _print_block_structure(block_structure)

    ebs_star = [np.empty((0,), dtype=float) for _ in block_structure.inequivalent_blocks]
    vs_star = [
        np.empty(
            (0, len(block_structure.blocks[ib]), len(block_structure.blocks[ib])),
            dtype=complex,
        )
        for ib in block_structure.inequivalent_blocks
    ]
    Cs_star = [
        np.zeros(
            (len(block_structure.blocks[ib]), len(block_structure.blocks[ib])),
            dtype=complex,
        )
        for ib in block_structure.inequivalent_blocks
    ]
    # Detect symmetries first: the state allocation caps each block at what fits
    # in the window, and a particle-hole symmetric bath is mirrored, so the cap
    # depends on the symmetry.
    syms = _detect_symmetries(
        w,
        hyb,
        mask,
        block_structure,
        delta,
        enforce_symmetry=enforce_symmetry,
        symmetry_tol=symmetry_tol,
        optimize_bath_energies=optimize_bath_energies,
        ebs_guess=ebs_guess,
    )
    states_per_inequivalent_block = get_state_per_inequivalent_block(
        block_structure,
        bath_states_per_orbital,
        hyb[mask, :, :],
        w[mask],
        weight_fun,
        delta,
        syms=syms,
    )
    # An odd state count on a mirrored bath needs one unpaired pole at E_F.
    # `zero_pole` is deliberately not set here: `fit_block` is the only place that
    # knows the final state count, after its own window cap, and deriving it in
    # two places would let them disagree. A frozen bath is the exception -- its
    # pole set comes from the caller, so `_detect_symmetries` settles it there.
    if verbose:
        _print_symmetries(block_structure, syms)

    # Do the fit
    for inequivalent_block_i, block_i in enumerate(block_structure.inequivalent_blocks):
        if states_per_inequivalent_block[inequivalent_block_i] == 0:
            continue
        block = block_structure.blocks[block_i]
        if verbose:
            n_states = states_per_inequivalent_block[inequivalent_block_i]
            print()
            print(_rule(f"Orbitals {block}  ·  {n_states} bath states", "-"))
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        sym = syms[inequivalent_block_i]

        bath_guess = None
        if ebs_guess is not None:
            bath_guess = np.unique(np.asarray(ebs_guess[inequivalent_block_i], dtype=float))
            if sym.particle_hole:
                # Only the positive half of a mirrored bath is parametrized; an
                # unpaired pole at E_F is carried by `sym.zero_pole`, not here.
                bath_guess = _positive_half(bath_guess, delta)

        if not optimize_bath_energies and bath_guess is None:
            raise ValueError(
                "fit_hyb(optimize_bath_energies=False) needs ebs_guess for every hybridizing "
                f"block; block {block_i} has none"
            )

        block_eb_star, block_vs_star, block_C_star = fit_block(
            block_hyb[mask, :, :],
            w[mask],
            delta,
            states_per_inequivalent_block[inequivalent_block_i],
            gamma=gamma,
            sym=sym,
            comm=comm,
            verbose=verbose,
            weight_fun=weight_fun,
            bath_guess=bath_guess,
            regularization=regularization,
            use_bounds=True,
            optimize_bath_energies=optimize_bath_energies,
        )
        # Remove states with negligible hopping.  On a mirrored bath the two
        # members of a pair are dropped together, so pruning cannot leave behind
        # an asymmetric bath.
        bath_mask = np.linalg.norm(block_vs_star, axis=(1, 2)) > 1e-10
        if sym.particle_hole:
            bath_mask = bath_mask & bath_mask[::-1]
        block_vs_star = block_vs_star[bath_mask]
        block_eb_star = block_eb_star[bath_mask]

        vs_star[inequivalent_block_i] = block_vs_star
        ebs_star[inequivalent_block_i] = block_eb_star
        Cs_star[inequivalent_block_i] = block_C_star
    if verbose:
        print(_rule(), flush=True)

    return ebs_star, vs_star, Cs_star


def _detect_symmetries(
    w,
    hyb,
    mask,
    block_structure,
    delta,
    enforce_symmetry,
    symmetry_tol,
    optimize_bath_energies,
    ebs_guess,
):
    """Detect the symmetry of every inequivalent block.

    Orbital symmetries are measured on the fitted window -- a symmetry only has
    to hold where the model is asked to reproduce the data -- while particle-hole
    symmetry is measured on the full mesh, since it cannot be seen at all on a
    one-sided window.

    Particle-hole symmetry is *enforced* only when it is also usable:

    * the fit window must reach above the Fermi level, otherwise the mirrored
      poles would sit outside the region being fitted;
    * a frozen bath must already be mirror symmetric, since folding the frozen
      energies onto their mirror partners would change the energies the caller
      asked to keep fixed.

    In both cases the symmetry is still reported, so the verbose log says it was
    detected and why it was not used.
    """
    syms = []
    for block_i in block_structure.inequivalent_blocks:
        block = block_structure.blocks[block_i]
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        if not enforce_symmetry:
            syms.append(trivial_symmetry(len(block)))
            continue
        allow_ph, ph_reason = _particle_hole_window(w[mask], delta)
        sym = detect_block_symmetry(
            w[mask],
            block_hyb[mask],
            tol=symmetry_tol,
            allow_particle_hole=allow_ph,
            ph_data=(w, block_hyb),
        )
        if sym.particle_hole and not optimize_bath_energies:
            guess = np.asarray(ebs_guess[len(syms)], dtype=float) if ebs_guess is not None else None
            mirrored, zero_pole = _frozen_bath_is_mirrored(guess, delta)
            if not mirrored:
                sym = _skip_particle_hole(sym, "frozen bath is not mirror symmetric")
            else:
                sym = replace(sym, zero_pole=zero_pole)
        if not allow_ph and "particle_hole_skipped" in sym.report:
            sym = _skip_particle_hole(sym, f"detected, but {ph_reason}")
        syms.append(sym)
    return syms


def _skip_particle_hole(sym, reason):
    """Return ``sym`` with particle-hole enforcement dropped and the reason recorded."""
    return replace(
        sym,
        particle_hole=False,
        zero_pole=False,
        report={**sym.report, "particle_hole_skipped": reason},
    )


def _positive_half(energies, delta):
    """Return the positive half of a mirrored bath: ``|e|``, without a pole at E_F.

    A pole at the Fermi level is its own mirror partner and is carried by
    ``sym.zero_pole``; folding it in with ``abs`` would make `expand_eb` emit a
    spurious ``-0, 0`` pair.
    """
    energies = np.asarray(energies, dtype=float).ravel()
    return np.unique(np.abs(energies[np.abs(energies) >= 0.5 * delta]))


def _frozen_bath_is_mirrored(energies, delta):
    """Report whether a frozen bath can be mirrored without changing it.

    Freezing means the caller's energies come back untouched, so particle-hole
    symmetry may only be enforced on a frozen bath when folding it onto the
    positive half and mirroring it back reproduces the original set exactly --
    not merely when it looks symmetric.  A guess with an odd count, a duplicated
    magnitude, or an unaccounted pole at E_F would otherwise be silently altered.

    Parameters
    ----------
    energies : array_like or None
        The bath energies to freeze.
    delta : float
        Broadening; sets the scale on which two energies count as equal and on
        which a pole counts as sitting at the Fermi level.

    Returns
    -------
    mirrored : bool
        Whether the set survives the round trip unchanged.
    zero_pole : bool
        Whether it contains an unpaired pole at the Fermi level.
    """
    if energies is None:
        return False, False
    energies = np.sort(np.asarray(energies, dtype=float).ravel())
    if energies.size == 0:
        return True, False
    zero_pole = bool(np.any(np.abs(energies) < 0.5 * delta))
    half = _positive_half(energies, delta)
    if half.size == 0:
        # Nothing but a pole at E_F: there is no pair to mirror, and the gap
        # parametrization has no free energy to work with.
        return False, False
    rebuilt = np.concatenate([-half[::-1], np.zeros(1) if zero_pole else np.zeros(0), half])
    if rebuilt.shape != energies.shape:
        return False, False
    return bool(np.allclose(rebuilt, energies, atol=0.1 * delta)), zero_pole


def _particle_hole_window(w_fit, delta):
    """Whether a fit window can host a mirrored bath, and why not when it cannot.

    Mirroring puts a partner at ``-e`` for every pole at ``+e``, so the bath is
    confined to the largest sub-window symmetric about the Fermi level.  It is not
    enough for that sub-window to be non-empty: a window like ``[-5, 0.05]``
    reaches above E_F yet would squeeze every bath state into ``|e| <= 0.05``
    while the hybridization extends to ``-5``, which fits the data far worse than
    an unconstrained bath would.  The symmetric part must therefore be both wide
    enough to hold a separated pair and a meaningful share of the window.
    """
    if w_fit.size == 0:
        return False, "the fit window is empty"
    w_min, w_max = float(w_fit[0]), float(w_fit[-1])
    half = min(w_max, -w_min)
    if half <= 0:
        return False, "the fit window does not straddle the Fermi level"
    if half < delta:
        return (
            False,
            f"the symmetric part of the fit window ({half:.3g}) is narrower than delta",
        )
    if half < PARTICLE_HOLE_WINDOW_FRACTION * max(w_max, -w_min):
        return (
            False,
            f"the fit window is too lopsided about the Fermi level (symmetric part {half:.3g})",
        )
    return True, ""


def get_state_per_inequivalent_block(
    block_structure,
    bath_states_per_orbital,
    hyb,
    w,
    weight_fun,
    delta,
    syms=None,
):
    """Distribute a pool of bath states across the inequivalent blocks.

    The user parameter ``bath_states_per_orbital`` (B) is treated as an average
    "bath states per block": the total pool is ``B * n_blocks``, shared between
    the blocks in proportion to their hybridization strength so blocks with
    strong hybridization get more states.  Two guards make the split sensible:

    * **Coverage.** Every block that hybridizes at all gets at least one bath
      state per orbital, so a weak block is never silently dropped by the
      weighting/rounding and an n-orbital block can span its n x n hybridization.
    * **Window cap.** No block is asked to fit more states than can reasonably
      sit inside the frequency window separated by ``delta`` (see
      `rspt2spectra.offdiagonal.max_bath_states`); an over-large share is capped,
      not fitted out of the window.  A particle-hole symmetric block only varies
      the positive half of its bath, so its cap is set by the half-window.

    ``syms`` are the per-block symmetries from `_detect_symmetries`; passing
    ``None`` caps every block as if it were unconstrained.

    The result is a rough guide -- coverage and cap mean the counts need not sum
    exactly to ``B * n_blocks``.  Blocks with no hybridization weight get zero.
    """
    blocks = block_structure.blocks
    identical_blocks = block_structure.identical_blocks
    transposed_blocks = block_structure.transposed_blocks
    particle_hole_blocks = block_structure.particle_hole_blocks
    particle_hole_and_transpose_blocks = block_structure.particle_hole_transposed_blocks
    inequivalent_blocks = block_structure.inequivalent_blocks

    n_blocks = len(inequivalent_blocks)
    weight_per_inequivalent_block = np.zeros((n_blocks,), dtype=float)
    orbitals_per_block = np.zeros((n_blocks,), dtype=int)
    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        block = blocks[block_i]
        orbitals_per_block[inequivalent_block_i] = len(block)
        block_multiplicity = (
            len(identical_blocks[block_i])
            + len(transposed_blocks[block_i])
            + len(particle_hole_blocks[block_i])
            + len(particle_hole_and_transpose_blocks[block_i])
        )
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        weight_per_inequivalent_block[inequivalent_block_i] = (
            sp.integrate.simpson(
                -np.imag(np.sum(np.diagonal(block_hyb, axis1=1, axis2=2), axis=1)) * weight_fun(w),
                w,
            )
            * block_multiplicity
        )

    # Negative integrated weight (numerical noise on an essentially empty block)
    # is not real hybridization; clamp it so it neither steals nor gets states.
    weight_per_inequivalent_block = np.clip(weight_per_inequivalent_block, 0.0, None)
    total_weight = np.sum(weight_per_inequivalent_block)

    # Pool of B states per block, shared out by hybridization strength.
    pool = bath_states_per_orbital * n_blocks
    if total_weight > 0:
        states = np.round(pool * weight_per_inequivalent_block / total_weight).astype(int)
    else:
        states = np.zeros(n_blocks, dtype=int)

    # Coverage: every hybridizing block gets at least one bath state per orbital,
    # so an n-orbital block can represent its full n x n hybridization.
    hybridizing = weight_per_inequivalent_block > 0
    states[hybridizing] = np.maximum(states[hybridizing], orbitals_per_block[hybridizing])

    # Window cap: never request more states than reasonably fit in the window.
    if syms is None:
        np.clip(states, 0, _max_bath_states(w[0], w[-1], delta), out=states)
    else:
        caps = np.array([max_bath_states(w[0], w[-1], delta, sym) for sym in syms])
        states = np.minimum(states, caps)
    return states


def fit_block(
    hyb,
    w,
    delta,
    bath_states_per_orbital,
    gamma,
    sym,
    comm,
    verbose,
    weight_fun,
    bath_guess=None,
    regularization=None,
    use_bounds=True,
    optimize_bath_energies=True,
):
    """Fit one hybridization block with VARPRO basin-hopping.

    Bath-energy seeds are drawn around the peaks of the block's spectral
    trace (weighted by ``weight_fun``); each MPI rank uses its own RNG seed
    and the lowest-cost fit across ranks is returned everywhere.

    ``optimize_bath_energies`` (default ``True``): set ``False`` to freeze the bath
    energies at ``bath_guess`` and solve only the hoppings (least squares); the
    basin-hopping search over energies is skipped. ``bath_guess`` is then required.

    ``sym`` is the block's :class:`rspt2spectra.symmetries.BlockSymmetry`.  When
    it carries particle-hole symmetry only the positive half of the bath is
    seeded and searched; the returned energies are the full mirrored set.

    Returns
    -------
    bath_energies : (n_b,) np.ndarray
    v : (n_b, n_orb, n_orb) np.ndarray
    C : (n_orb, n_orb) np.ndarray
        The fitted constant hybridization offset.
    """
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1
    # Set up a sequence of RNG seeds, so that each MPI rank gets its own unique seed, and therefore also initial guess.
    base_seed = 12  # Just because
    seed_sequence = np.random.SeedSequence(base_seed)
    child_seeds = seed_sequence.spawn(size)
    rng = np.random.default_rng(seed=child_seeds[rank])

    if not optimize_bath_energies:
        # Freeze the energies at the supplied guess and solve only the hoppings.
        if bath_guess is None:
            raise ValueError(
                "fit_block(optimize_bath_energies=False) requires bath_guess -- the bath energies to freeze"
            )
        eb_guess = np.sort(np.asarray(bath_guess, dtype=float))[None, :]
        if sym.particle_hole:
            eb_guess = _positive_half(eb_guess, delta)[None, :]
        eb_bounds = [(w[0], w[-1])] * eb_guess.shape[1]
        v, bath_energies, C, min_cost = get_v_and_eb_varpro_basin_hopping(
            w,
            delta,
            hyb,
            eb_guess,
            eb_bounds,
            gamma=gamma,
            regularization=regularization,
            weight_function=weight_fun,
            sym=sym,
            rng=rng,
            optimize_bath_energies=False,
        )
        if comm is not None:
            bath_energies, v, C, _ = comm.allreduce((bath_energies, v, C, min_cost), op=_get_v_opt_op())
        if verbose:
            print(f"Final cost:    {abs(min_cost):.3e}  (bath energies frozen)")
            print(f"Bath energies: {_fmt_floats(bath_energies)}")
        return bath_energies, v, C

    # Cap the requested count at what reasonably fits in the window (min
    # separation delta), so seeds are built at a feasible size from the start.
    bath_states_per_orbital = min(bath_states_per_orbital, max_bath_states(w[0], w[-1], delta, sym))
    # A mirrored bath only parametrizes its positive half (plus, for an odd
    # count, one unpaired pole at E_F that carries no free energy).  The cap
    # above can change the parity of the count, so re-derive the unpaired pole
    # from the capped value rather than trusting the caller's.
    if sym.particle_hole and bath_states_per_orbital < 2:
        # A mirrored bath needs at least one pair: with a single state the
        # positive half is empty and there is nothing to optimize.  Take a pair
        # if the window has room, otherwise drop the symmetry rather than fit a
        # bath that cannot express it.
        if max_bath_states(w[0], w[-1], delta, sym) >= 2:
            bath_states_per_orbital = 2
        else:
            sym = replace(sym, particle_hole=False, zero_pole=False)
    if sym.particle_hole:
        sym = replace(sym, zero_pole=bool(bath_states_per_orbital % 2))
        n_free = bath_states_per_orbital // 2
    else:
        n_free = bath_states_per_orbital
    lo, hi = free_window(w[0], w[-1], delta, sym)

    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=1, axis2=2), axis=1))
    hyb_trace[hyb_trace < 0] = 0
    peaks, _ = find_peaks(
        hyb_trace,
    )
    _, _, l_lims, r_lims = peak_widths(hyb_trace, peaks, rel_height=0.9)

    scores = weight_fun(w[peaks]) * hyb_trace[peaks]
    score_sum = np.sum(scores)
    normalised_scores = scores / score_sum if score_sum > 0 else np.ones_like(scores) / len(scores)

    if verbose:
        _print_peaks(
            w[peaks],
            np.interp(l_lims, range(len(w)), w),
            np.interp(r_lims, range(len(w)), w),
            normalised_scores,
        )
    population_size = 200

    if len(peaks) > 0:
        peak_index = rng.choice(
            np.arange(len(peaks)),
            size=(population_size, n_free),
            p=normalised_scores,
            replace=True,
        )
        eb_guess = rng.uniform(
            low=np.interp(l_lims[peak_index], range(len(w)), w),
            high=np.interp(r_lims[peak_index], range(len(w)), w),
        )
    else:
        eb_guess = rng.uniform(low=w[0], high=w[-1], size=(population_size, n_free))
    if sym.particle_hole:
        # Fold the seeds onto the positive half and keep them inside the
        # symmetric sub-window the mirrored poles have to fit in.
        eb_guess = np.clip(np.abs(eb_guess), lo, hi)
    if bath_guess is not None:
        n = min(bath_guess.shape[0], n_free)
        eb_guess[0, :n] = bath_guess[:n]
    eb_guess = np.sort(eb_guess, axis=1)

    eb_bounds = [(w[0], w[-1])] * n_free
    v, bath_energies, C, min_cost = get_v_and_eb_varpro_basin_hopping(
        w,
        delta,
        hyb,
        eb_guess,
        eb_bounds,
        gamma=gamma,
        regularization=regularization,
        weight_function=weight_fun,
        sym=sym,
        rng=rng,
    )
    if comm is not None:
        bath_energies, v, C, _ = comm.allreduce((bath_energies, v, C, min_cost), op=_get_v_opt_op())

    if verbose:
        print(f"Final cost:    {abs(min_cost):.3e}")
        print(f"Bath energies: {_fmt_floats(bath_energies)}")
    return bath_energies, v, C

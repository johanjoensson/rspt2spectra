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

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):  # pragma: no cover - MPI is optional; serial fits work without it
    MPI = None
import numpy as np
import scipy as sp
from scipy.signal import find_peaks, peak_widths

from .offdiagonal import (
    _max_bath_states,
    get_v_and_eb_varpro_basin_hopping,
)

_LINE_WIDTH = 72


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
    vs_guess=None,
    regularization=None,
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
    ebs_guess, vs_guess : list of np.ndarray, optional
        Initial guesses per inequivalent block (e.g. from a previous fit).
    regularization : {"L1", "L2", "none", None}
        Regularization type for the hopping parameters.

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
    states_per_inequivalent_block = get_state_per_inequivalent_block(
        block_structure,
        bath_states_per_orbital,
        hyb[mask, :, :],
        w[mask],
        weight_fun,
        delta,
    )

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
        realvalue_v = np.all(np.abs(block_hyb - np.conj(np.transpose(block_hyb, (0, 2, 1)))) < 1e-6)

        bath_guess = None
        v_guess = None
        if vs_guess is not None:
            v_guess = vs_guess[inequivalent_block_i]
        if ebs_guess is not None:
            bath_guess = ebs_guess[inequivalent_block_i]

        # Block structure has changed!
        # Remove all hopping guesses, but keep the bath energies
        if v_guess is not None and bath_guess is not None and v_guess.shape[1] != block_hyb.shape[1]:
            v_guess = None

        block_eb_star, block_vs_star, block_C_star = fit_block(
            block_hyb[mask, :, :],
            w[mask],
            delta,
            states_per_inequivalent_block[inequivalent_block_i],
            gamma=gamma,
            realvalue_v=realvalue_v,
            comm=comm,
            verbose=verbose,
            weight_fun=weight_fun,
            bath_guess=bath_guess,
            hopping_guess=v_guess,
            regularization=regularization,
            use_bounds=True,
        )
        # Remove states with negligible hopping
        bath_mask = np.linalg.norm(block_vs_star, axis=(1, 2)) > 1e-10
        block_vs_star = block_vs_star[bath_mask]
        block_eb_star = block_eb_star[bath_mask]

        vs_star[inequivalent_block_i] = block_vs_star
        ebs_star[inequivalent_block_i] = block_eb_star
        Cs_star[inequivalent_block_i] = block_C_star
    if verbose:
        print(_rule(), flush=True)

    return ebs_star, vs_star, Cs_star


def get_state_per_inequivalent_block(
    block_structure,
    bath_states_per_orbital,
    hyb,
    w,
    weight_fun,
    delta,
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
      `_max_bath_states`); an over-large share is capped, not fitted out of the
      window.

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
    n_max = _max_bath_states(w[0], w[-1], delta)
    np.clip(states, 0, n_max, out=states)
    return states


def fit_block(
    hyb,
    w,
    delta,
    bath_states_per_orbital,
    gamma,
    realvalue_v,
    comm,
    verbose,
    weight_fun,
    bath_guess=None,
    hopping_guess=None,
    regularization=None,
    use_bounds=True,
):
    """Fit one hybridization block with VARPRO basin-hopping.

    Bath-energy seeds are drawn around the peaks of the block's spectral
    trace (weighted by ``weight_fun``); each MPI rank uses its own RNG seed
    and the lowest-cost fit across ranks is returned everywhere.

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

    # Cap the requested count at what reasonably fits in the window (min
    # separation delta), so seeds are built at a feasible size from the start.
    bath_states_per_orbital = min(bath_states_per_orbital, _max_bath_states(w[0], w[-1], delta))

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
            size=(population_size, bath_states_per_orbital),
            p=normalised_scores,
            replace=True,
        )
        eb_guess = rng.uniform(
            low=np.interp(l_lims[peak_index], range(len(w)), w),
            high=np.interp(r_lims[peak_index], range(len(w)), w),
        )
    else:
        eb_guess = rng.uniform(low=w[0], high=w[-1], size=(population_size, bath_states_per_orbital))
    if bath_guess is not None:
        n = min(bath_guess.shape[0], bath_states_per_orbital)
        eb_guess[0, :n] = bath_guess[:n]
    eb_guess = np.sort(eb_guess, axis=1)

    eb_bounds = [(w[0], w[-1])] * bath_states_per_orbital
    v, bath_energies, C, min_cost = get_v_and_eb_varpro_basin_hopping(
        w,
        delta,
        hyb,
        eb_guess,
        eb_bounds,
        gamma=gamma,
        regularization=regularization,
        weight_function=weight_fun,
        realvalue_v=realvalue_v,
    )
    if comm is not None:
        bath_energies, v, C, _ = comm.allreduce((bath_energies, v, C, min_cost), op=_get_v_opt_op())

    if verbose:
        print(f"Final cost:    {abs(min_cost):.3e}")
        print(f"Bath energies: {_fmt_floats(bath_energies)}")
    return bath_energies, v, C

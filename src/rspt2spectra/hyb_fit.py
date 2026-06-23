import itertools
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import scipy as sp
from scipy.signal import find_peaks, peak_widths
from .offdiagonal import (
    get_hyb,
    get_hyb_2,
    get_v_and_eb,
    get_v_and_eb_multiple_optimizations,
    get_v_and_eb_basin_hopping,
    get_v_and_eb_differential_evolution,
    generate_hopping_guess,
)
import warnings


def v_opt(a, b, _):
    return a if abs(a[-1]) <= abs(b[-1]) else b


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
    weight_fun=lambda w: np.ones_like(w),
    ebs_guess=None,
    vs_guess=None,
    regularization=None,
):
    """
    Calculate the bath energies and hopping parameters for fitting the
    hybridization function.

    Parameters:
    w           -- Real frequency mesh
    delta       -- All quantities will be evaluated i*delta above the real
                   frequency line.
    hyb         -- Hybridization function
    bath_states_per_orbital --Number of bath states to fit for each orbital
    w_lim       -- (w_min, w_max) Only fit for frequencies w_min <= w <= w_max.
                   If not set, fit for all w.
    Returns:
    eb          -- Bath energies
    v           -- Hopping parameters
    """
    if bath_states_per_orbital == 0:
        return [
            np.array([], dtype=float) for ib in block_structure.inequivalent_blocks
        ], [
            np.empty(
                (0, len(block_structure.blocks[ib]), len(block_structure.blocks[ib])),
                dtype=complex,
            )
            for ib in block_structure.inequivalent_blocks
        ]
    if x_lim is not None:
        mask = np.logical_and(x_lim[0] <= w, w < x_lim[1])
    else:
        mask = np.ones(len(w), bool)

    if verbose:

        print(f"Blocks: {block_structure.blocks}")
        print(f"Inequivalent blocks: {block_structure.inequivalent_blocks}")
        print(f"Identical blocks: {block_structure.identical_blocks}")
        print(f"Transposed blocks: {block_structure.transposed_blocks}")
        print(f"Particle hole blocks: {block_structure.particle_hole_blocks}")
        print(
            f"Particle hole transposed blocks: {block_structure.particle_hole_transposed_blocks}"
        )
        print("=" * 80)

    ebs_star = [
        np.empty((0,), dtype=float) for ib in block_structure.inequivalent_blocks
    ]
    vs_star = [
        np.empty(
            (0, len(block_structure.blocks[ib]), len(block_structure.blocks[ib])),
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
    )

    # Do the fit
    for inequivalent_block_i, block_i in enumerate(block_structure.inequivalent_blocks):
        if states_per_inequivalent_block[inequivalent_block_i] == 0:
            continue
        block = block_structure.blocks[block_i]
        if verbose:
            print(f"Fitting hybridization function for impurity orbitals {block}")
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        realvalue_v = np.all(
            np.abs(block_hyb - np.conj(np.transpose(block_hyb, (0, 2, 1)))) < 1e-6
        )

        bath_guess = None
        v_guess = None
        if vs_guess is not None:
            v_guess = vs_guess[inequivalent_block_i]
        if ebs_guess is not None:
            bath_guess = ebs_guess[inequivalent_block_i]

        # Block structure has changed!
        # Remove all hopping guesses, but keep the bath energies
        if (
            v_guess is not None
            and bath_guess is not None
            and v_guess.shape[1] != block_hyb.shape[1]
        ):
            n_orb_old = v_guess.shape[1]
            n_orb = block_hyb.shape[1]

            v_guess = None
            # bath_guess = np.array([eb for eb in bath_guess])

        block_eb_star, block_vs_star = fit_block(
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
            # use_bounds=x_lim is not None,
        )
        if verbose:
            print()
        # Remove states with negligleble hopping
        bath_mask = np.linalg.norm(block_vs_star, axis=(1, 2)) > 1e-10
        block_vs_star = block_vs_star[bath_mask]
        block_eb_star = block_eb_star[bath_mask]

        vs_star[inequivalent_block_i] = block_vs_star
        ebs_star[inequivalent_block_i] = block_eb_star
    if verbose:
        print("=" * 80, flush=True)

    return ebs_star, vs_star


def get_state_per_inequivalent_block(
    block_structure,
    bath_states_per_orbital,
    hyb,
    w,
    weight_fun,
):
    blocks = block_structure.blocks
    identical_blocks = block_structure.identical_blocks
    transposed_blocks = block_structure.transposed_blocks
    particle_hole_blocks = block_structure.particle_hole_blocks
    particle_hole_and_transpose_blocks = block_structure.particle_hole_transposed_blocks
    inequivalent_blocks = block_structure.inequivalent_blocks

    orbitals_per_inequivalent_block = [0] * len(inequivalent_blocks)
    weight_per_inequivalent_block = np.zeros((len(inequivalent_blocks)), dtype=float)
    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        block = blocks[block_i]
        block_multiplicity = (
            len(identical_blocks[block_i])
            + len(transposed_blocks[block_i])
            + len(particle_hole_blocks[block_i])
            + len(particle_hole_and_transpose_blocks[block_i])
        )
        orbitals_per_inequivalent_block[inequivalent_block_i] = (
            len(block) * block_multiplicity
        )
        idx = np.ix_(range(hyb.shape[0]), block, block)
        block_hyb = hyb[idx]
        weight_per_inequivalent_block[inequivalent_block_i] = (
            # np.trapezoid(
            sp.integrate.simpson(
                -np.imag(np.sum(np.diagonal(block_hyb, axis1=1, axis2=2), axis=1))
                * weight_fun(w),
                w,
            )
            * block_multiplicity
        )
    states_per_inequivalent_block = np.round(
        weight_per_inequivalent_block
        / np.sum(weight_per_inequivalent_block)
        * np.sum(orbitals_per_inequivalent_block)
        * bath_states_per_orbital
        / orbitals_per_inequivalent_block
    ).astype(int)
    states_per_inequivalent_block[states_per_inequivalent_block < 0] = 0
    return states_per_inequivalent_block


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
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1
    # Set up a sequence of RNG seeds, so that each MPI rank gets its own unique seed, and therefore also initial guess.
    base_seed = 12  # Just because
    seed_sequence = np.random.SeedSequence(base_seed)
    child_seeds = seed_sequence.spawn(size)
    rng = np.random.default_rng(seed=child_seeds[rank])

    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=1, axis2=2), axis=1))
    hyb_trace[hyb_trace < 0] = 0
    n_orb = hyb.shape[1]
    peaks, info = find_peaks(
        hyb_trace,
    )
    _, w_h, l_lims, r_lims = peak_widths(hyb_trace, peaks, rel_height=0.9)
    if False:
        plt.plot(w, hyb_trace, color="tab:blue")
        plt.plot(w, weight_fun(w), "--", color="tab:gray")
        plt.plot(w, hyb_trace * weight_fun(w), "--", color="tab:blue")
        plt.hlines(
            w_h,
            np.interp(l_lims, range(len(w)), w),
            np.interp(r_lims, range(len(w)), w),
            colors="tab:orange",
        )
        plt.show()

    scores = weight_fun(w[peaks]) * hyb_trace[peaks]
    score_sum = np.sum(scores)
    normalised_scores = (
        scores / score_sum if score_sum > 0 else np.ones_like(scores) / len(scores)
    )

    if verbose:
        print("Peak positions:    ", ", ".join(f"{el: ^16.3f}" for el in w[peaks]))
        print(
            "Peak intervals:    ",
            ", ".join(
                (
                    f"[{el: >6.3f}, {er: >6.3f}]"
                    for el, er in zip(
                        np.interp(l_lims, range(len(w)), w),
                        np.interp(r_lims, range(len(w)), w),
                    )
                ),
            ),
        )
        print(
            "Peak scores:       ",
            ", ".join(f"{el: ^16.3f}" for el in normalised_scores),
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
        eb_guess = rng.uniform(
            low=w[0], high=w[-1], size=(population_size, bath_states_per_orbital)
        )
    if bath_guess is not None:
        n = min(bath_guess.shape[0], bath_states_per_orbital)
        eb_guess[0, :n] = bath_guess[:n]
    eb_guess = np.sort(eb_guess, axis=1)
    if use_bounds:
        eb_guess = np.append(
            eb_guess,
            # np.array([[w[-1] + 5]] * population_size), axis=1
            rng.uniform(low=w[-1] + 1, high=w[-1] + 10, size=(population_size, 1)),
            axis=1,
        )

    v_guess = generate_hopping_guess(
        w + 1j * delta, hyb, eb_guess, gamma, realvalue_v, rng
    )
    # v_guess = (
    #     rng.normal(
    #         loc=0.0, scale=2.0, size=(population_size, eb_guess.shape[1], n_orb, n_orb)
    #     )
    #     + 0j
    # )
    # if not realvalue_v:
    #     v_guess += 1j * rng.normal(
    #         loc=0.0, scale=2.0, size=(population_size, eb_guess.shape[1], n_orb, n_orb)
    #     )
    # v_guess = np.linalg.cholesky(
    #     np.conj(np.transpose(v_guess, axes=(0, 1, 3, 2))) @ v_guess, upper=True
    # )
    if hopping_guess is not None:
        n = min(hopping_guess.shape[0], bath_states_per_orbital)
        v_guess[0, :n] = hopping_guess[:n]

    if False:
        hyb_model = get_hyb_2(w + 1j * delta, eb_guess, v_guess)
        for pop_i in range(population_size):
            fig, ax = plt.subplots(nrows=n_orb, ncols=n_orb, squeeze=False)
            fig.suptitle(r"Re{$\Delta - \tilde{\Delta}$}")
            for i, j in itertools.product(range(n_orb), repeat=2):
                ax[i, j].fill_between(
                    w,
                    hyb[:, i, j].real,
                    0,
                    alpha=0.2,
                    color="tab:blue",
                )
                ax[i, j].plot(w, hyb_model[pop_i, :, i, j].real, color="tab:orange")
            fig, ax = plt.subplots(nrows=n_orb, ncols=n_orb, squeeze=False)
            fig.suptitle(r"Im{$\Delta - \tilde{\Delta}$}")
            for i, j in itertools.product(range(n_orb), repeat=2):
                ax[i, j].fill_between(
                    w,
                    hyb[:, i, j].imag,
                    0,
                    alpha=0.2,
                    color="tab:blue",
                )
                ax[i, j].plot(w, hyb_model[pop_i, :, i, j].imag, color="tab:orange")
            plt.show()
    eb_bounds = [(w[0], w[-1])] * bath_states_per_orbital
    if use_bounds:
        eb_bounds += [(w[-1] + 1, w[-1] + 10)]
    if n_orb == 1 or False:
        v, bath_energies, min_cost = get_v_and_eb_differential_evolution(
            w,
            delta,
            hyb,
            eb_guess,
            eb_bounds,
            v_guess,
            gamma=gamma,
            regularization=regularization,
            weight_function=weight_fun,
        )
    else:
        # v, bath_energies, min_cost = get_v_and_eb_multiple_optimizations(
        v, bath_energies, min_cost = get_v_and_eb_basin_hopping(
            w,
            delta,
            hyb,
            eb_guess,
            eb_bounds,
            v_guess,
            gamma=gamma,
            regularization=regularization,
            weight_function=weight_fun,
        )
    if comm is not None:
        bath_energies, v, _ = comm.allreduce(
            (bath_energies, v, min_cost), op=MPI.Op.Create(v_opt, commute=True)
        )

    # V = np.linalg.cholesky(A, upper=True)

    return bath_energies, v

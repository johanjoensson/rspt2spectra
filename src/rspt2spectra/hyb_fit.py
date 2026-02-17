import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from scipy.signal import find_peaks, peak_widths
from .offdiagonal import (
    get_hyb,
    get_v_and_eb,
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
    imag_only,
    x_lim=None,
    verbose=True,
    comm=None,
    weight_fun=lambda w: np.ones_like(w),
    ebs_guess=None,
    vs_guess=None,
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
        mask = np.array([True] * len(w))

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
            np.abs(block_hyb - np.transpose(block_hyb, (0, 2, 1))) < 1e-6
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
            imag_only=imag_only,
            realvalue_v=realvalue_v,
            comm=comm,
            verbose=verbose,
            weight_fun=weight_fun,
            bath_guess=bath_guess,
            hopping_guess=v_guess,
        )
        if verbose:
            print()
        # Remove states with negligleble hopping
        bath_mask = []
        for group_i in range(block_vs_star.shape[0]):
            bath_mask.append(np.linalg.norm(block_vs_star[group_i]) > 1e-10)
            # if np.any(
            #     np.all(
            #         np.abs(block_vs_star[group_i]) ** 2 < 1e-10,
            #         axis=1,
            #     )
            # ):
            # bath_mask.extend([False] * len(block))
            # else:
            # bath_mask.extend([True] * len(block))
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
    (
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transpose_blocks,
        inequivalent_blocks,
    ) = block_structure

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
            np.trapezoid(
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
    imag_only,
    realvalue_v,
    comm,
    verbose,
    weight_fun,
    bath_guess=None,
    hopping_guess=None,
):
    n_proc = 1 if comm is None else comm.size
    rng = np.random.default_rng()

    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=1, axis2=2), axis=1))
    hyb_trace[hyb_trace < 0] = 0
    n_orb = hyb.shape[1]
    peaks, info = find_peaks(
        hyb_trace,
    )
    scores = weight_fun(w[peaks]) * hyb_trace[peaks]
    normalised_scores = scores / np.sum(scores)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, left_ips, right_ips = peak_widths(hyb_trace, peaks, rel_height=0.8)

    if verbose:
        print("Peak positions:    ", ", ".join(f"{el: ^16.3f}" for el in w[peaks]))
        print(
            "Peak intervals:    ",
            ", ".join(
                f"[{w[int(max(0, el))]: >6.3f}, {w[int(min(len(w) - 1, er))]: >6.3f}]"
                for el, er in zip(left_ips, right_ips)
            ),
        )
        print(
            "Peak scores:       ",
            ", ".join(f"{el: ^16.3f}" for el in normalised_scores),
        )
    population_size = 100
    eb_guess = np.empty((0, bath_states_per_orbital), dtype=float)
    v_guess = np.empty((0, bath_states_per_orbital, n_orb, n_orb), dtype=complex)
    for i in range(population_size):
        if bath_guess is not None and i == 0:
            bath_energies = np.array(list(bath_guess)).reshape((len(bath_guess)))
        elif len(peaks) > 0:
            bath_index = rng.choice(
                np.arange(len(peaks)),
                size=min(len(peaks), bath_states_per_orbital),
                p=normalised_scores,
                replace=False,
            )
            bath_energies = w[peaks[bath_index]]
        else:
            bath_energies = []
        if hopping_guess is not None and i == 0:
            v = v_guess
        else:
            v = np.empty((0, n_orb, n_orb), dtype=complex)

        bath_energies = np.append(
            bath_energies,
            rng.uniform(
                low=w[0],
                high=w[-1],
                size=max(bath_states_per_orbital - len(bath_energies), 0),
            ),
        )

        remainder = max(bath_states_per_orbital - v.shape[0], 0)
        v = np.append(
            v,
            generate_hopping_guess(
                w + 1j * delta,
                hyb,
                bath_energies[-remainder:],
                gamma,
                imag_only,
                realvalue_v,
            ),
            axis=0,
        )
        v_guess = np.append(v_guess, [v], axis=0)
        eb_guess = np.append(
            eb_guess, [bath_energies[:bath_states_per_orbital]], axis=0
        )

    v, bath_energies, min_cost = get_v_and_eb_differential_evolution(
        # v, bath_energies, min_cost = get_v_and_eb_basin_hopping(
        w,
        delta,
        hyb,
        eb_guess,
        v_guess,
        eb_bounds=[(w[0], w[-1])] * eb_guess.shape[1],
        gamma=gamma,
        imag_only=imag_only,
        realvalue_v=realvalue_v,
        scale_function=weight_fun,
    )
    if comm is not None:
        bath_energies, v, _ = comm.allreduce(
            (bath_energies, v, min_cost), op=MPI.Op.Create(v_opt, commute=True)
        )

    return bath_energies, v

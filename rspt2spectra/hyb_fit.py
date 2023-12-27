from .offdiagonal import get_v, get_v_new, get_hyb, get_v_and_eb
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
import matplotlib.pyplot as plt
from mpi4py import MPI
from math import ceil


def block_diagonalize_hyb(hyb):
    hyb_herm = 1 / 2 * (hyb + np.conj(np.transpose(hyb, (1, 0, 2))))
    blocks = get_block_structure(hyb_herm)
    Q_full = np.zeros((hyb.shape[0], hyb.shape[1]), dtype=complex)
    treated_orbitals = 0
    for block in blocks:
        block_idx = np.ix_(block, block)
        if len(block) == 1:
            Q_full[block_idx, treated_orbitals] = 1
            treated_orbitals += 1
            continue
        block_hyb = hyb_herm[block_idx]
        upper_triangular_hyb = np.triu(hyb_herm, k=1)
        ind_max_offdiag = np.unravel_index(
            np.argmax(np.abs(upper_triangular_hyb)), upper_triangular_hyb.shape
        )
        eigvals, Q = np.linalg.eigh(block_hyb[:, :, ind_max_offdiag[2]])
        sorted_indices = np.argsort(eigvals)
        Q = Q[:, sorted_indices]
        for column in range(Q.shape[1]):
            j = np.argmax(np.abs(Q[:, column]))
            Q_full[block, treated_orbitals + column] = (
                Q[:, column] * abs(Q[j, column]) / Q[j, column]
            )
        treated_orbitals += Q.shape[1]
    phase_hyb = np.moveaxis(
        np.conj(Q_full.T)[np.newaxis, :, :]
        @ np.moveaxis(hyb, -1, 0)
        @ Q_full[np.newaxis, :, :],
        0,
        -1,
    )

    return phase_hyb, Q_full


def get_block_structure(hyb, hamiltonian=None, tol=1e-6):
    # Extract matrix elements with nonzero hybridization function
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    max_val = np.abs(np.max(hyb))
    mask = np.logical_or(np.any(np.abs(hyb) > tol, axis=2),
                         np.abs(hamiltonian) > tol)

    # Use the extracted mask to extract blocks
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_blocks, block_idxs = connected_components(
        csgraph=csr_matrix(mask), directed=False, return_labels=True
    )

    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)

    return blocks


def get_identical_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    identical_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in identical_blocks]):
            continue
        identical = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in identical_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(np.abs(hyb[idx_i] - hyb[idx_j]) < tol) and np.all(
                np.abs(hamiltonian[idx_i] - hamiltonian[idx_j]) < tol
            ):
                identical.append(j)
        identical_blocks.append(identical)
    return identical_blocks


def get_transposed_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    transposed_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if len(block_i) == 1 or np.any([i in b for b in transposed_blocks]):
            continue
        transposed = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i + 1:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp + 1
            if any(j in b for b in transposed_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(
                np.abs(hyb[idx_i] - np.transpose(hyb[idx_j], (1, 0, 2))) < tol
            ) and np.all(np.abs(hamiltonian[idx_i] - hamiltonian[idx_j].T) < tol):
                transposed.append(j)
        transposed_blocks[i] = transposed
    return transposed_blocks


def get_particle_hole_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    particle_hole_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in particle_hole_blocks]):
            continue
        particle_hole = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in particle_hole_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if (
                np.all(np.abs(np.real(hyb[idx_i] + hyb[idx_j])) < tol)
                and np.all(np.abs(np.imag(hyb[idx_i] - hyb[idx_j])) < tol)
                and np.all(
                    np.abs(
                        np.real(hamiltonian[idx_i] - hamiltonian[idx_j])) < tol
                )
                and np.all(
                    np.abs(
                        np.imag(hamiltonian[idx_i] - hamiltonian[idx_j])) < tol
                )
            ):
                particle_hole.append(j)
        particle_hole_blocks.append(particle_hole)
    return particle_hole_blocks


def get_particle_hole_and_transpose_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    patricle_hole_and_transpose_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in patricle_hole_and_transpose_blocks]):
            continue
        patricle_hole_and_transpose = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in patricle_hole_and_transpose_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(
                np.abs(hyb[idx_i] + np.transpose(hyb[idx_j], (1, 0, 2))) < tol
            ) and np.all(np.abs(hamiltonian[idx_i] + hamiltonian[idx_j].T) < tol):
                patricle_hole_and_transpose.append(j)
        patricle_hole_and_transpose_blocks.append(patricle_hole_and_transpose)
    return patricle_hole_and_transpose_blocks


def fit_hyb(
    w,
    delta,
    hyb,
    bath_states_per_orbital,
    gamma,
    imag_only,
    x_lim=None,
    tol=1e-6,
    verbose=True,
    comm=None,
    new_v=False,
    exp_weight=2,
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
    if x_lim is not None:
        mask = np.logical_and(w >= x_lim[0], w <= x_lim[1])
    else:
        mask = np.array([True] * len(w))

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ cf_hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    phase_blocks = get_block_structure(phase_hyb, tol=tol)
    phase_identical_blocks = get_identical_blocks(
        phase_blocks, phase_hyb, tol=tol)
    phase_transposed_blocks = get_transposed_blocks(
        phase_blocks, phase_hyb, tol=tol)
    phase_particle_hole_blocks = get_particle_hole_blocks(
        phase_blocks, phase_hyb, tol=tol
    )
    phase_particle_hole_and_transpose_blocks = get_particle_hole_and_transpose_blocks(
        phase_blocks, phase_hyb, tol=tol
    )

    if verbose:
        print(f"block structure : {phase_blocks}")
        print(f"identical blocks : {phase_identical_blocks}")
        print(f"transposed blocks : {phase_transposed_blocks}")
        print(f"particle hole blocks : {phase_particle_hole_blocks}")
        print(
            f"particle hole and transpose blocks : { phase_particle_hole_and_transpose_blocks}"
        )

    n_orb = sum(len(block) for block in phase_blocks)

    eb = np.empty((0,))
    v = np.empty((0, n_orb), dtype=complex)
    inequivalent_blocks = []
    for blocks in phase_identical_blocks:
        unique = True
        for transpose in phase_transposed_blocks:
            if blocks[0] in transpose[1:]:
                unique = False
                break
        for particle_hole in phase_particle_hole_blocks:
            if blocks[0] in particle_hole[1:]:
                unique = False
                break
        for particle_hole_and_transpose in phase_particle_hole_and_transpose_blocks:
            if blocks[0] in particle_hole_and_transpose[1:]:
                unique = False
                break
        if unique:
            inequivalent_blocks.append(blocks[0])
    if verbose:
        print(f"inequivalent blocks = {inequivalent_blocks}")
    for equivalent_block_i, inequivalent_block_i in enumerate(inequivalent_blocks):
        block = phase_blocks[inequivalent_block_i]
        idx = np.ix_(block, block)
        block_hyb = phase_hyb[idx]
        realvalue_v = np.all(
            np.abs(block_hyb - np.transpose(block_hyb, (1, 0, 2))) < 1e-6
        )
        block_eb, block_v = fit_block(
            block_hyb[:, :, mask],
            w[mask],
            delta,
            bath_states_per_orbital,
            gamma=gamma,
            imag_only=imag_only,
            realvalue_v=realvalue_v,
            w0=0,
            comm=comm,
            new_v=new_v,
            exp_weight=exp_weight,
        )
        if verbose:
            print(f"--> eb {block_eb}")
            print(f"--> v  {block_v}")

        for b in phase_identical_blocks[equivalent_block_i]:
            eb = np.append(eb, block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype=complex)
            v_tmp[:, phase_blocks[b]] = block_v
            v = np.append(v, v_tmp, axis=0)
        for b in phase_transposed_blocks[equivalent_block_i]:
            eb = np.append(eb, block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype=complex)
            v_tmp[:, phase_blocks[b]] = np.conj(block_v)
            v = np.append(v, v_tmp, axis=0)
        for b in phase_particle_hole_blocks[equivalent_block_i]:
            eb = np.append(eb, -block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype=complex)
            v_tmp[:, phase_blocks[b]] = block_v
            v = np.append(v, v_tmp, axis=0)
        for b in phase_particle_hole_and_transpose_blocks[equivalent_block_i]:
            eb = np.append(eb, -block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype=complex)
            v_tmp[:, phase_blocks[b]] = np.conj(block_v)
            v = np.append(v, v_tmp, axis=0)
    # Transform hopping parameters back from the (close to) diagonal
    # basis to the spherical harmonics basis
    v = v @ np.conj(Q.T)

    mask = np.any(np.abs(v) ** 2 / delta > 1e-8, axis=1)
    v = v[mask, :]
    eb = eb[mask]
    return eb, v


def v_opt(a, b, datatype):
    return a if a[-1] <= b[-1] else b


v_opt_op = MPI.Op.Create(v_opt, commute=True)


def fit_block(
    hyb,
    w,
    delta,
    bath_states_per_orbital,
    gamma,
    imag_only,
    realvalue_v,
    w0=0,
    comm=None,
    new_v=False,
    exp_weight=2,
):
    def weight(peak):
        return np.exp(-exp_weight * np.abs(w[peak] - w0))

    world_size = 1
    if comm is not None:
        world_size = comm.size
    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=0, axis2=1), axis=1))
    n_orb = hyb.shape[0]
    n_bath_states = bath_states_per_orbital * n_orb
    de = w[1] - w[0]
    peak_min = np.inf

    if n_bath_states == 0:
        return np.empty((0, n_orb)), np.empty((0, n_orb), dtype=complex)

    bath_energies = np.empty((n_bath_states), dtype=float)
    found_bath_states = 0
    fit_hyb = np.zeros_like(hyb)
    while found_bath_states < n_bath_states:
        fit_trace = - \
            np.imag(np.sum(np.diagonal(fit_hyb, axis1=0, axis2=1), axis=1))
        peaks, _ = find_peaks(
            hyb_trace - fit_trace,
            distance=int(delta / de),
            width=int(delta / de),
        )
        scores = weight(peaks) * (hyb_trace - fit_trace)[peaks]
        sorted_indices = np.argsort(scores)
        sorted_peaks = peaks[sorted_indices]
        sorted_energies = w[sorted_peaks]
        n_b = min(
            len(sorted_energies) - 1, (n_bath_states - found_bath_states) // n_orb
        )
        bath_energies[found_bath_states: found_bath_states + n_b * n_orb] = np.repeat(
            sorted_energies[-n_b:], n_orb
        )
        peak_min = min(np.min(hyb_trace[sorted_peaks[-n_b:]]), peak_min)
        found_bath_states += n_b * n_orb
        try_again = True
        while try_again:
            v, cost = get_v_new(
                w + delta * 1j,
                hyb,
                bath_energies[:found_bath_states],
                gamma=gamma,
                imag_only=imag_only,
                realvalue_v=realvalue_v,
            )
            try_again = np.any(
                [
                    np.all(n_orb*np.abs(v[i: i + n_orb, :]) ** 2 / delta < 1e-4*peak_min)
                    for i in range(0, v.shape[0], n_orb)
                ]
            )
        fit_hyb = get_hyb(w + delta * 1j, bath_energies[:found_bath_states], v)
    v_best = v
    min_cost = np.abs(cost)

    for _ in range(min(10 // world_size, 2)):
        try_again = True
        while try_again:
            v, cost = get_v_new(
                w + delta * 1j,
                hyb,
                bath_energies,
                gamma=gamma,
                imag_only=imag_only,
                realvalue_v=realvalue_v,
            )
            try_again = np.any(
                [
                    np.all(n_orb*np.abs(v[i: i + n_orb, :]) ** 2 < 1e-4*peak_min)
                    for i in range(0, v.shape[0], n_orb)
                ]
            )
        if abs(cost) < min_cost:
            v_best = v
            min_cost = abs(cost)
    if comm is not None:
        bath_energies, v, _ = comm.allreduce(
            (bath_energies, v_best, min_cost), op=v_opt_op)

    # if comm is None or comm.rank == 0:
    #     fit_hyb = get_hyb(w + delta * 1j, bath_energies[:found_bath_states], v)
    #     fig, ax = plt.subplots(
    #         nrows=n_orb, ncols=n_orb, squeeze=False, sharex='all', sharey='all')
    #     for i in range(n_orb):
    #         for j in range(n_orb):
    #             ax[i, j].plot(w, -np.imag(hyb[i, j, :]), color='tab:blue')
    #             ax[i, j].plot(w, -np.imag(fit_hyb[i, j, :]),
    #                           color='tab:orange')
    #             ax[i, j].vlines(bath_energies[:found_bath_states], 0, -np.min(
    #                 np.imag(fit_hyb[i, j, :])), linestyles='dashed', colors='tab:gray')
    #             # ax[i, j].hlines(peak_min, w[0], w[-1],
    #             ax[i, j].hlines(hyb_trace[sorted_peaks[-n_b:]], w[0], w[-1],
    #                             linestyles='dashed', colors='tab:gray')
    #     plt.ylim(bottom=-np.max(-np.imag(np.diagonal(hyb, axis1=0, axis2=1))),
    #              top=np.max(-np.imag(np.diagonal(hyb, axis1=0, axis2=1))))
    #     plt.show()

    return bath_energies, v


def fit_block_new(
    hyb,
    w,
    delta,
    bath_states_per_orbital,
    gamma,
    imag_only,
    realvalue_v,
    w0=0,
    comm=None,
    new_v=False,
    exp_weight=2,
):
    def weight(peak):
        return np.exp(-exp_weight * np.abs(w[peak] - w0))

    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=0, axis2=1), axis=1))
    n_orb = hyb.shape[0]
    de = w[1] - w[0]
    peaks, _ = find_peaks(
        hyb_trace,
        # distance=int(delta / de),
        # width=int(delta / de),
    )
    n_b = min(len(peaks), bath_states_per_orbital)
    sorted_energies = (w[-1] - w[0])*np.random.rand(bath_states_per_orbital) + w[0]
    scores = weight(peaks) * hyb_trace[peaks]
    sorted_indices = np.argsort(scores)
    sorted_peaks = peaks[sorted_indices]
    print(f"{len(peaks)=}")
    print(f"{len(sorted_peaks)=}", flush=True)
    sorted_energies[:n_b] = w[sorted_peaks][-n_b:]
    bath_energies = np.repeat(
            sorted_energies, n_orb
        )

    print(f"bath_guess={bath_energies}")
    min_cost = np.inf
    for _ in range(10):
        v, eb, cost = get_v_and_eb(
            w + delta * 1j,
            hyb,
            bath_energies,
            gamma=gamma,
            imag_only=imag_only,
            realvalue_v=realvalue_v,
        )
        if abs(cost) < min_cost:
            eb_best = eb
            v_best = v
            min_cost = abs(cost)

    if comm is not None:
        bath_energies, v, _ = comm.allreduce(
            (eb_best, v_best, min_cost), op=v_opt_op)
    mask = np.logical_and(bath_energies >= w[0], bath_energies <= w[-1])
    bath_energies = bath_energies[mask]
    v = v[mask]
    print(f"{(w[0], w[-1])=}")
    print(f"{mask=}")
    print(f"{bath_energies=}")
    print(f"{v=}", flush=True)

    # if comm is None or comm.rank == 0:
    #     fit_hyb = get_hyb(w + delta * 1j, bath_energies, v)
    #     fig, ax = plt.subplots(
    #         nrows=n_orb, ncols=n_orb, squeeze=False, sharex='all', sharey='all')
    #     for i in range(n_orb):
    #         for j in range(n_orb):
    #             ax[i, j].plot(w, -np.imag(hyb[i, j, :]), color='tab:blue')
    #             ax[i, j].plot(w, -np.imag(fit_hyb[i, j, :]),
    #                           color='tab:orange')
    #             ax[i, j].vlines(bath_energies, 0, -np.min(
    #                 np.imag(fit_hyb[i, j, :])), linestyles='dashed', colors='tab:gray')
    #             ax[i, j].hlines(hyb_trace[sorted_peaks[-n_b:]], w[0], w[-1],
    #                             linestyles='dashed', colors='tab:gray')
    #     plt.ylim(bottom=-np.max(-np.imag(np.diagonal(hyb, axis1=0, axis2=1))),
    #              top=np.max(-np.imag(np.diagonal(hyb, axis1=0, axis2=1))))
    #     plt.show()
    return bath_energies, v

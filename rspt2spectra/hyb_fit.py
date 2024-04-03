import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from scipy.signal import find_peaks, peak_widths
from .offdiagonal import get_hyb, get_v_and_eb
import warnings


def block_diagonalize_hyb(hyb):
    hyb_herm = 1 / 2 * (hyb + np.conj(np.transpose(hyb, (0, 2, 1))))
    blocks = get_block_structure(hyb_herm)
    Q_full = np.zeros((hyb.shape[1], hyb.shape[2]), dtype=complex)
    treated_orbitals = 0
    for block in blocks:
        block_idx = np.ix_(range(hyb.shape[0]), block, block)
        if len(block) == 1:
            Q_full[block_idx[1:], treated_orbitals] = 1
            treated_orbitals += 1
            continue
        block_hyb = hyb_herm[block_idx]
        upper_triangular_hyb = np.triu(hyb_herm, k=1)
        ind_max_offdiag = np.unravel_index(
            np.argmax(np.abs(upper_triangular_hyb)), upper_triangular_hyb.shape
        )
        eigvals, Q = np.linalg.eigh(block_hyb[ind_max_offdiag[0], :, :])
        sorted_indices = np.argsort(eigvals)
        Q = Q[:, sorted_indices]
        for column in range(Q.shape[1]):
            j = np.argmax(np.abs(Q[:, column]))
            Q_full[block, treated_orbitals + column] = (
                Q[:, column] * abs(Q[j, column]) / Q[j, column]
            )
        treated_orbitals += Q.shape[1]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full


def get_block_structure(hyb: np.ndarray, hamiltonian=None, tol=1e-6):
    # Extract matrix elements with nonzero hybridization function
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    mask = np.logical_or(np.any(np.abs(hyb) > tol, axis=0), np.abs(hamiltonian) > tol)

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
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    identical_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in identical_blocks]):
            continue
        identical = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in identical_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(np.abs(hyb[idx_i] - hyb[idx_j]) < tol) and np.all(
                np.abs(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]]) < tol
            ):
                identical.append(j)
        identical_blocks.append(identical)
    return identical_blocks


def get_transposed_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    transposed_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if len(block_i) == 1 or np.any([i in b for b in transposed_blocks]):
            continue
        transposed = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i + 1 :]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp + 1
            if any(j in b for b in transposed_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(
                np.abs(hyb[idx_i] - np.transpose(hyb[idx_j], (0, 2, 1))) < tol
            ) and np.all(
                np.abs(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]].T) < tol
            ):
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
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in particle_hole_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if (
                np.all(np.abs(np.real(hyb[idx_i] + hyb[idx_j])) < tol)
                and np.all(np.abs(np.imag(hyb[idx_i] - hyb[idx_j])) < tol)
                and np.all(
                    np.abs(np.real(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]]))
                    < tol
                )
                and np.all(
                    np.abs(np.imag(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]]))
                    < tol
                )
            ):
                particle_hole.append(j)
        particle_hole_blocks.append(particle_hole)
    return particle_hole_blocks


def get_particle_hole_and_transpose_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    patricle_hole_and_transpose_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in patricle_hole_and_transpose_blocks]):
            continue
        patricle_hole_and_transpose = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in patricle_hole_and_transpose_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(
                np.abs(hyb[idx_i] + np.transpose(hyb[idx_j], (0, 2, 1))) < tol
            ) and np.all(
                np.abs(hamiltonian[idx_i[1:]] + hamiltonian[idx_j[1:]].T) < tol
            ):
                patricle_hole_and_transpose.append(j)
        patricle_hole_and_transpose_blocks.append(patricle_hole_and_transpose)
    return patricle_hole_and_transpose_blocks


def exp_weight(w, w0, e):
    return np.exp(-e * np.abs(w - w0))


def gauss_weight(w, w0, e):
    return np.exp(-e / 2 * np.abs(w - w0) ** 2)


def get_weight_function(weight_function_name, w0, e):
    if weight_function_name == "Gaussian":
        return lambda w: gauss_weight(w, w0, e)
    elif weight_function_name == "Exponential":
        return lambda w: exp_weight(w, w0, e)
    elif weight_function_name == "RSPt":
        return lambda w: np.abs(w - w0) / (1 + e * np.abs(w - w0)) ** 3
    else:
        raise RuntimeError(f"Unknown weight function {weight_function_name}")
    return None


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
    weight_param=2,
    w0=0,
    weight_function_name="RSPt",
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
    weight_fun = get_weight_function(weight_function_name, w0, weight_param)
    if bath_states_per_orbital == 0:
        return np.empty((0,), dtype=float), np.empty((0, hyb.shape[1]), dtype=complex)
    if x_lim is not None:
        mask = np.logical_and(w >= x_lim[0], w <= x_lim[1])
    else:
        mask = np.array([True] * len(w))

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ cf_hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    phase_blocks = get_block_structure(phase_hyb, tol=tol)
    phase_identical_blocks = get_identical_blocks(phase_blocks, phase_hyb, tol=tol)
    phase_transposed_blocks = get_transposed_blocks(phase_blocks, phase_hyb, tol=tol)
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

    ebs = [np.empty((0,))] * n_orb
    vs = [np.empty((0, n_orb), dtype=complex)] * n_orb
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
    orbitals_per_inequivalent_block = [0] * len(inequivalent_blocks)
    weight_per_inequivalent_block = np.zeros((len(inequivalent_blocks)), dtype=float)
    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        block = phase_blocks[block_i]
        block_multiplicity = (
            len(phase_identical_blocks[inequivalent_block_i])
            + len(phase_transposed_blocks[inequivalent_block_i])
            + len(phase_particle_hole_blocks[inequivalent_block_i])
            + len(phase_particle_hole_and_transpose_blocks[inequivalent_block_i])
        )
        orbitals_per_inequivalent_block[inequivalent_block_i] = (
            len(block) * block_multiplicity
        )
        idx = np.ix_(range(phase_hyb.shape[0]), block, block)
        block_hyb = phase_hyb[idx]
        weight_per_inequivalent_block[inequivalent_block_i] = (
            np.trapz(
                -np.imag(
                    np.sum(np.diagonal(block_hyb[mask, :, :], axis1=1, axis2=2), axis=1)
                )
                * weight_fun(w[mask]),
                w[mask],
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

    for inequivalent_block_i, block_i in enumerate(inequivalent_blocks):
        if states_per_inequivalent_block[inequivalent_block_i] == 0:
            continue
        block = phase_blocks[block_i]
        idx = np.ix_(range(phase_hyb.shape[0]), block, block)
        block_hyb = phase_hyb[idx]
        realvalue_v = np.all(
            np.abs(block_hyb - np.transpose(block_hyb, (0, 2, 1))) < 1e-6
        )
        block_eb, block_v = fit_block(
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
        )
        n_block_orb = len(block)

        if verbose:
            print(f"--> eb {block_eb}")
            print(f"--> v  {block_v}")

        for b in phase_identical_blocks[inequivalent_block_i]:
            for i_orb, orb in enumerate(phase_blocks[b]):
                ebs[orb] = np.append(ebs[orb], [block_eb[i_orb::n_block_orb]])
                v_tmp = np.zeros((len(block_eb) // n_block_orb, n_orb), dtype=complex)
                v_tmp[:, phase_blocks[b]] = block_v[i_orb::n_block_orb, :]
                vs[orb] = np.append(vs[orb], v_tmp, axis=0)
        for b in phase_transposed_blocks[inequivalent_block_i]:
            for i_orb, orb in enumerate(phase_blocks[b]):
                ebs[orb] = np.append(ebs[orb], [block_eb[i_orb::n_block_orb]])
                v_tmp = np.zeros((len(block_eb) // n_block_orb, n_orb), dtype=complex)
                v_tmp[:, phase_blocks[b]] = np.conj(block_v[i_orb::n_block_orb, :])
                vs[orb] = np.append(vs[orb], v_tmp, axis=0)
        for b in phase_particle_hole_blocks[inequivalent_block_i]:
            for i_orb, orb in enumerate(phase_blocks[b]):
                ebs[orb] = np.append(ebs[orb], [-block_eb[i_orb::n_block_orb]])
                v_tmp = np.zeros((len(block_eb) // n_block_orb, n_orb), dtype=complex)
                v_tmp[:, phase_blocks[b]] = block_v[i_orb::n_block_orb, :]
                vs[orb] = np.append(vs[orb], v_tmp, axis=0)
        for b in phase_particle_hole_and_transpose_blocks[inequivalent_block_i]:
            for i_orb, orb in enumerate(phase_blocks[b]):
                ebs[orb] = np.append(ebs[orb], [-block_eb[i_orb::n_block_orb]])
                v_tmp = np.zeros((len(block_eb) // n_block_orb, n_orb), dtype=complex)
                v_tmp[:, phase_blocks[b]] = np.conj(block_v[i_orb::n_block_orb, :])
                vs[orb] = np.append(vs[orb], v_tmp, axis=0)
        eb = np.concatenate(ebs, axis=0)
        v = np.vstack(vs)

    # Transform hopping parameters back from the (close to) diagonal
    # basis to the spherical harmonics basis
    v = v @ np.conj(Q.T)
    # Sort bath states, it is important for impurityModel that all unoccupied states come after the occupied states
    # sort_indices = np.argsort(eb, kind="stable")
    # eb = eb[sort_indices]
    # v = v[sort_indices]

    return eb, v


def v_opt(a, b, _):
    return a if abs(a[-1]) <= abs(b[-1]) else b


v_opt_op = MPI.Op.Create(v_opt, commute=True)


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
):
    rng = np.random.default_rng()

    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=1, axis2=2), axis=1))
    n_orb = hyb.shape[1]
    peaks, info = find_peaks(
        hyb_trace,
    )
    scores = weight_fun(w[peaks]) * hyb_trace[peaks]
    normalised_scores = scores / np.sum(scores)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, left_ips, right_ips = peak_widths(hyb_trace, peaks, rel_height=0.5)

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
    min_cost = np.inf
    for _ in range(max(1000 // comm.size, 2) if comm is not None else 100):
        if len(peaks) > 0:
            bath_index = rng.choice(
                np.arange(len(peaks)), size=bath_states_per_orbital, p=normalised_scores
            )
            bath_energies = w[peaks[bath_index]]
            bounds = [
                (
                    w[int(max(0, left_ips[i]))],
                    w[int(min(len(w) - 1, right_ips[i]))],
                )
                for i in bath_index
            ]
        else:
            bath_energies = rng.uniform(
                low=w[0], high=w[-1], size=bath_states_per_orbital
            )
            bounds = [(w[0], w[-1])] * bath_states_per_orbital
        v, eb, cost = get_v_and_eb(
            w + delta * 1j,
            hyb,
            bath_energies,
            eb_bounds=bounds,
            gamma=gamma,
            imag_only=imag_only,
            realvalue_v=realvalue_v,
            scale_function=weight_fun,
        )
        if abs(cost) < min_cost:
            eb_best = eb
            v_best = v
            min_cost = abs(cost)
    if comm is not None:
        bath_energies, v, _ = comm.allreduce((eb_best, v_best, min_cost), op=v_opt_op)
    return np.repeat(bath_energies, n_orb), v

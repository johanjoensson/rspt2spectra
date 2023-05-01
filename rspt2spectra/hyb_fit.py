from .offdiagonal import get_v, get_hyb
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
import matplotlib.pyplot as plt

def block_diagonalize_hyb(hyb):
    hyb_herm = 1/2*(hyb + np.conj(np.transpose(hyb, ( 1, 0, 2))))
    blocks = get_block_structure(hyb_herm)
    Q_full = np.zeros((hyb.shape[0], hyb.shape[1]), dtype = np.complex)
    treated_orbitals = 0
    for block in blocks:
        block_idx = np.ix_(block, block)
        if len(block) == 1:
            Q_full[block_idx, treated_orbitals] = 1
            treated_orbitals += 1
            continue
        block_hyb = hyb_herm[block_idx]
        upper_triangular_hyb = np.triu(hyb_herm, k = 1)
        ind_max_offdiag = np.unravel_index(np.argmax(np.abs(upper_triangular_hyb)), upper_triangular_hyb.shape)
        # _, Q_full[block_idx, treated_orbitals:treated_orbitals + len(block)] = np.linalg.eigh(block_hyb[:,:, ind_max_offdiag[2]])
        _, Q = np.linalg.eigh(block_hyb[:,:, ind_max_offdiag[2]])
        for column in range(Q.shape[1]):
            j = np.argmax(np.abs(Q[:, column]))
            Q_full[block, treated_orbitals + column] = Q[:, column]*abs(Q[j, column])/Q[j, column]
        treated_orbitals += len(block)
    phase_hyb = np.moveaxis(
              np.conj(Q_full.T)[np.newaxis, :, :]
            @ np.moveaxis(hyb, -1, 0) 
            @ Q_full[np.newaxis, :, :]
            , 0, -1)

    return phase_hyb, Q_full

def get_block_structure(hyb, hamiltonian = None, tol = 1e-6):
    # Extract matrix elements with nonzero hybridization function
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    max_val = np.abs(np.max(hyb))
    mask = np.logical_or(np.any(np.abs(hyb) > tol, axis = 2), np.abs(hamiltonian) > tol)                    

    # Use the extracted mask to extract blocks
    from scipy.sparse import csr_matrix                                              
    from scipy.sparse.csgraph import connected_components                            
                                                                                             
    n_blocks, block_idxs = connected_components(                                     
    csgraph=csr_matrix(mask), directed=False, return_labels=True)

    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)

    return blocks

def get_identical_blocks(blocks, hyb, hamiltonian = None, tol = 1e-6):
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
            if (    np.all(np.abs(hyb[idx_i] - hyb[idx_j]) < tol) 
                and np.all(np.abs(hamiltonian[idx_i] - hamiltonian[idx_j]) < tol)
                ):
                identical.append(j)
        identical_blocks.append(identical)
    return identical_blocks

def get_transposed_blocks(blocks, hyb, hamiltonian = None, tol = 1e-6):
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
            if (    np.all(np.abs(hyb[idx_i] - np.transpose(hyb[idx_j], (1, 0, 2))) < tol) 
                and np.all(np.abs(hamiltonian[idx_i] - hamiltonian[idx_j].T) < tol)
                ):
                transposed.append(j)
        transposed_blocks[i] = transposed
    return transposed_blocks

def fit_hyb(w, delta, hyb, rot_spherical, bath_states_per_orbital, gamma, imag_only, x_lim = None, tol = 1e-6):
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
        mask = np.logical_and(w>=x_lim[0], w<=x_lim[1])
    else:
        mask = np.array([True]*len(w))

    # We do the fitting by first transforming the hyridization function into a basis
    # where each block is (hopefully) close to diagonal
    # np.conj(Q.T) @ hyb @ Q is the transformation performed
    phase_hyb, Q = block_diagonalize_hyb(hyb)

    phase_blocks = get_block_structure(phase_hyb, tol = tol)
    phase_identical_blocks = get_identical_blocks(phase_blocks, phase_hyb, tol = tol)
    phase_transposed_blocks = get_transposed_blocks(phase_blocks, phase_hyb, tol = tol)

    print (f"phase equalized block structure : {phase_blocks}")
    print (f"phase equalized identical blocks : {phase_identical_blocks}")
    print (f"phase equalized transposed blocks : {phase_transposed_blocks}")

    n_orb = sum(len(block) for block in phase_blocks)

    eb = np.empty((0,))
    v = np.empty((0, n_orb), dtype = complex)
    inequivalent_blocks = []
    for blocks in phase_identical_blocks:
        unique = True
        for transpose in phase_transposed_blocks:
            if blocks[0] in transpose[1:]:
                unique = False
                break
        if unique:
            inequivalent_blocks.append(blocks[0])
    print (f'inequivalent blocks = {inequivalent_blocks}')
    for equivalent_block_i, inequivalent_block_i in enumerate(inequivalent_blocks):
        block = phase_blocks[inequivalent_block_i]
        idx = np.ix_(block, block)
        block_hyb = phase_hyb[idx]
        realvalue_v = np.all(np.abs(block_hyb - np.transpose(block_hyb, (1, 0 , 2))) < 1e-6)
        print (f"realvalued v ? {realvalue_v}")
        block_eb, block_v = fit_block_new(block_hyb[:, :, mask], w[mask], delta, bath_states_per_orbital, gamma = gamma, imag_only = imag_only, realvalue_v = realvalue_v, w0 = 0)
        print (f"--> eb {block_eb}")
        print (f"--> v  {block_v}")

        for b in phase_identical_blocks[equivalent_block_i]:
            eb = np.append(eb, block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype = complex)
            v_tmp[:, phase_blocks[b]] = block_v
            v = np.append(v, v_tmp, axis = 0)
        for b in phase_transposed_blocks[equivalent_block_i]:
            eb = np.append(eb, block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype = complex)
            v_tmp[:, phase_blocks[b]] = np.conj(block_v)
            v = np.append(v, v_tmp, axis = 0)
    # Transform hopping parameters back from the (close to) diagonal 
    # basis to the spherical harmonics basis
    v = v @ np.conj(Q.T) @ rot_spherical

    # sorted_indices = np.argsort(eb)
    sorted_indices = range(len(eb))
    return eb[sorted_indices], v[sorted_indices]
    

def fit_block(hyb, w, delta, bath_states_per_orbital, gamma, imag_only, realvalue_v, w0 = 0):
    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1 = 0, axis2 = 1), axis = 1))
    n_bath_states = bath_states_per_orbital * hyb.shape[0]
    de = w[1] - w[0]
    # widths is the peak widths we are interested in, in units of de = w[1] - w[0]
    max_width = bath_states_per_orbital*int(delta/de)
    peaks, properties = find_peaks(hyb_trace, width = max_width)
        # peaks = find_peaks_cwt(hyb_trace, widths=np.arange(1, max_width))
    peaks, _ = find_peaks(hyb_trace, width = int(delta/de), distance = int(delta/de))
    if len(peaks) < n_bath_states:
        print (f"Cannot place {n_bath_states} energy windows, only found {len(peaks)} peaks.")
        print (f"Placing {len(peaks)} energy windows instead.")
        n_bath_states = len(peaks)
    def weight(peak):
        return np.exp(-2*np.abs(w[peak] - w0))
    weights = weight(peaks)/np.sum(weight(peaks))
    sorted_indices = np.argsort(weights*(hyb_trace[peaks]))
    sorted_peaks = peaks[sorted_indices][::-1]

    bath_energies = w[sorted_peaks[:n_bath_states]]
    # plt.plot(w, hyb_trace)
    # plt.scatter(w[sorted_peaks], hyb_trace[sorted_peaks])
    # plt.show()
    
    min_cost = np.inf
    v = None
    for _ in range(1):
        v_try, costs = get_v(w + delta*1j, hyb, bath_energies, gamma = gamma, imag_only = imag_only, realvalue_v = realvalue_v)
        if np.max(np.abs(costs)) < min_cost:
            min_cost = np.max(np.abs(costs))
            v = v_try
    # model_hyb = get_hyb(w + delta*1j, bath_energies, v)
    # if hyb.shape[0] > 1:
    #     fig, ax = plt.subplots(nrows = hyb.shape[0], ncols = hyb.shape[1], sharex = 'all', sharey = 'all')
    #     for row in range(hyb.shape[0]):
    #         for col in range(hyb.shape[1]):
    #             ax[row, col].plot(w, np.imag(hyb[row, col]), color = 'tab:blue')
    #             ax[row, col].plot(w, np.imag(model_hyb[row, col]), color = 'tab:orange')
    # else:
    #     plt.plot(w, np.imag(hyb[0, 0]), color = 'tab:blue')
    #     plt.plot(w, np.imag(model_hyb[0, 0]), color = 'tab:orange')
    # plt.show()
    return bath_energies, v

def fit_block_new(hyb, w, delta, bath_states_per_orbital, gamma, imag_only, realvalue_v, w0 = 0):
    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1 = 0, axis2 = 1), axis = 1))
    n_orb = hyb.shape[0]
    n_bath_states = bath_states_per_orbital * n_orb
    de = w[1] - w[0]

    bath_energies = np.zeros((n_bath_states,), dtype = float)
    # hopping_parameters = np.zeros((n_bath_states, n_orb), dtype = complex)
    fit_hyb = np.zeros_like(hyb)
    # for bath_i in range(n_bath_states):
    for bath_i in range(bath_states_per_orbital):
        # widths is the peak widths we are interested in, in units of de = w[1] - w[0]
        fit_trace = -np.imag(np.sum(np.diagonal(fit_hyb, axis1 = 0, axis2 = 1), axis = 1))
        candidate_energies = []
        for orb in range(n_orb):
            peaks, _ = find_peaks(-np.imag(hyb[orb, orb] - fit_hyb[orb, orb]), distance = 1)

            def weight(peak):
                return np.exp(-1*np.abs(w[peak] - w0))
            
            weights = weight(peaks)/np.sum(weight(peaks))
            scores = weights*(-np.imag(hyb[orb, orb][peaks]))
            [candidate_energies.append((w[p], score)) for p, score in zip(peaks, scores)]
        sorted_candidates = sorted(candidate_energies, key = lambda candidate: candidate[1])
        bath_energies[bath_i*n_orb : (bath_i + 1)*n_orb] = [energy for energy, _ in sorted_candidates[-n_orb:]]

        
        min_cost = np.inf
        v = None
        for _ in range(10):
            v_try, costs = get_v(w + delta*1j, hyb, bath_energies[ : (bath_i + 1)*n_orb], gamma = gamma, imag_only = imag_only, realvalue_v = realvalue_v)
            if np.max(np.abs(costs)) < min_cost:
                min_cost = np.max(np.abs(costs))
                v = v_try
        fit_hyb = get_hyb(w + delta*1j, bath_energies[: (bath_i + 1)*n_orb], v)

    # fig, ax = plt.subplots(nrows = n_orb, ncols = n_orb, squeeze = False)
    # for i in range(n_orb):
    #     for j in range(n_orb):
    #         ax[i, j].plot(w, np.imag(fit_hyb[i, j]), color = 'tab:red')
    #         ax[i, j].plot(w, np.imag(hyb[i, j]), color = 'tab:blue')
    # plt.show()
    return bath_energies, v

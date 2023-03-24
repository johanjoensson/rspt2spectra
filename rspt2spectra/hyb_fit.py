from .offdiagonal import get_v, get_hyb
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
import matplotlib.pyplot as plt

def diagonalize_imaginary_part_of_hyb(hyb):
    hyb = np.moveaxis(hyb, -1, 0)
    upper_triangular_hyb = np.triu(np.imag(hyb), k = 1)
    ind_max_offdiag = np.unravel_index(np.argmax(np.max(np.abs(upper_triangular_hyb))), upper_triangular_hyb.shape)
    _, Q = np.linalg.eigh(np.imag(hyb[ind_max_offdiag[0], :, :]))
    diag_hyb = np.array(Q.T[np.newaxis, :, :] @ hyb @ Q[np.newaxis, :, :])
    return np.moveaxis(diag_hyb, 0, -1), Q

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

def get_equivalent_blocks(blocks, hyb, hamiltonian = None, tol = 1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    equivalent_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in equivalent_blocks]):
            continue
        equiv = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if j in equivalent_blocks:
                continue
            idx_j = np.ix_(block_j, block_j)
            if (    np.all(np.abs(hyb[idx_i] - hyb[idx_j]) < tol) 
                and np.all(np.abs(hamiltonian[idx_i] - hamiltonian[idx_j]) < tol)
                ):
                equiv.append(j)
        equivalent_blocks.append(equiv)
    return equivalent_blocks

def fit_hyb(w, delta, hyb, rot_spherical, bath_states_per_orbital, gamma, imag_only, x_lim = None):
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
    # where the imaginary part is (close to) diagonal
    # Q.T @ hyb @ Q is the transformation performed
    diagonalish_hyb, Q = diagonalize_imaginary_part_of_hyb(hyb)
    # diagonalish_hyb = hyb
    diagonalish_blocks = get_block_structure(diagonalish_hyb)
    equivalent_blocks = get_equivalent_blocks(diagonalish_blocks, diagonalish_hyb)

    # spherical_hyb = np.moveaxis(
    #         np.conj(rot_spherical.T)[np.newaxis, :, :] 
    #         @ np.moveaxis(hyb, -1, 0) 
    #         @ rot_spherical[np.newaxis, :, :], 0, -1
    #         )

    n_orb = sum(len(block) for block in diagonalish_blocks)

    eb = np.empty((0,))
    v = np.empty((0, n_orb), dtype = complex)
    # for block_i, block in enumerate(blocks):
    for equiv_blocks in equivalent_blocks:
        block = diagonalish_blocks[equiv_blocks[0]]
        idx = np.ix_(block, block)
        block_hyb = diagonalish_hyb[idx]
        block_eb, block_v = fit_block(block_hyb[:, :, mask], w[mask], delta, bath_states_per_orbital, gamma = gamma, imag_only = imag_only)
        print (f"--> eb {block_eb}")
        print (f"--> v  {block_v}")
        for b in equiv_blocks:
            eb = np.append(eb, block_eb)
            # eb.append(block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype = complex)
            v_tmp[:, diagonalish_blocks[b]] = block_v
            v = np.append(v, v_tmp, axis = 0)
        # v.append(v_tmp)
    # Transform hopping parameters back from the (close to) diagonal imaginary
    # part basis to the spherical harmonics basis
    v = v @ Q.T @ rot_spherical

    sorted_indices = np.argsort(eb)
    return eb[sorted_indices], v[sorted_indices]
    

def fit_block(hyb, w, delta, bath_states_per_orbital, gamma, imag_only):
    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1 = 0, axis2 = 1), axis = 1))
    n_bath_states = bath_states_per_orbital * hyb.shape[0]
    # widths is the peak widths we are interested in, in units of de = w[1] - w[0]
    max_width = 2*int(delta/(w[1] - w[0]))
    peaks, properties = find_peaks(hyb_trace, width = max_width)
    while peaks.shape[0] < n_bath_states and max_width > 1:
        max_width -= 1
        # peaks, properties = find_peaks(hyb_trace, width = max_width, wlen = min_width)
        # peaks, properties = find_peaks(hyb_trace, width = max_width)
        peaks = find_peaks_cwt(hyb_trace, widths=np.arange(1, max_width))
    # _, peak_heights, peak_borders_left, peak_borders_right = peak_widths(-np.imag(hyb[i, i]), peaks, rel_height = 0.8)
    if len(peaks) < n_bath_states:
        print (f"Cannot place {n_bath_states} energy windows, only found {len(peaks)} peaks.")
        print (f"Placing {len(peaks)} energy windows instead.")
        n_bath_states = len(peaks)
    def weight(peak):
        return np.exp(-1*np.abs(w[peak]))
    weights = weight(peaks) # /np.sum(weight(peaks))
    sorted_indices = np.argsort(weights*(hyb_trace[peaks]))
    sorted_peaks = peaks[sorted_indices][-n_bath_states:][::-1]

    bath_energies = w[sorted_peaks]
    # plt.plot(w, hyb_trace)
    # plt.scatter(w[sorted_peaks], hyb_trace[sorted_peaks])
    # plt.show()
    
    min_cost = np.inf
    v = None
    for _ in range(20):
        v_try, costs = get_v(w + delta*1j, hyb, bath_energies, gamma = gamma, imag_only = imag_only)
        if np.max(np.abs(costs)) < min_cost:
            min_cost = np.max(np.abs(costs))
            v = v_try
    model_hyb = get_hyb(w + delta*1j, bath_energies, v)
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
    v_max = np.max(np.abs(v))
    v[np.abs(v) < 0.01*v_max] = 0
    return bath_energies, v

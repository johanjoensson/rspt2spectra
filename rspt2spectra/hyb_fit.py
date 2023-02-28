from .offdiagonal import get_v, get_hyb
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
import matplotlib.pyplot as plt

def diagonalize_imaginary_part_of_hyb(hyb):
    upper_triangular_hyb = np.array([np.triu(np.imag(hyb[:, :, w_i]), k = 1) for w_i in range(hyb.shape[2])])
    w_max_offdiag = np.argmax([np.max(np.abs(upper_triangular_hyb[w_i])) for w_i in range(hyb.shape[2])])
    _, Q = np.linalg.eigh(np.imag(hyb[:, :, w_max_offdiag]))
    diag_hyb = np.array([Q.T @ hyb[:, :, w_i] @ Q for w_i in range(hyb.shape[2])])
    return np.moveaxis(diag_hyb, 0, -1), Q

def get_block_structure(hyb):
    tol = 1e-6

    # Extract matrix elements with nonzero hybridization function
    # mask = np.any(np.abs(hyb) > tol, axis = 2)                    
    mask = np.abs(hyb[:, :, 0]) > tol                    
    print(f'mask =\n{mask}')                                                         
                                                                                             

    # Use the extracted mask to extract blocks
    from scipy.sparse import csr_matrix                                              
    from scipy.sparse.csgraph import connected_components                            
                                                                                             
    n_blocks, block_idxs = connected_components(                                     
    csgraph=csr_matrix(mask), directed=False, return_labels=True)

    print (f"n_blocks = {n_blocks}")
    print (f"block_idxs = {block_idxs}")

    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)

    print (f"blocks = {blocks}")
    return blocks

def get_equivalent_blocks(blocks, hyb):
    equivalent_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in equivalent_blocks]):
            continue
        equiv = []
        idx_i = np.ix_(block_i, block_i)
        hyb_i = hyb[idx_i]
        max_val = np.max(np.abs(hyb_i))
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            idx_j = np.ix_(block_j, block_j)
            hyb_j = hyb[idx_j]
            if np.all(np.abs(hyb_i - hyb_j) < 1e-1*max_val):
                equiv.append(j)
        equivalent_blocks.append(equiv)
    return equivalent_blocks

def fit_hyb(w, delta, hyb, bath_states_per_orbital, gamma, imag_only, x_lim = None):
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
    diagonal_hyb, Q = diagonalize_imaginary_part_of_hyb(hyb)
    blocks = get_block_structure(diagonal_hyb)

    equivalent_blocks = get_equivalent_blocks(blocks, diagonal_hyb)
    print (f"equivalent blocks  {equivalent_blocks}") 

    n_orb = sum(len(block) for block in blocks)

    eb = np.empty((0,))
    v = np.empty((0, n_orb), dtype = complex)
    # for block_i, block in enumerate(blocks):
    for equiv_blocks in equivalent_blocks:
        block = blocks[equiv_blocks[0]]
        print (f"treating equivalent blocks {[blocks[b] for b in equiv_blocks]} fit only {block}")
        idx = np.ix_(block, block)
        block_hyb = diagonal_hyb[idx]
        block_eb, block_v = fit_block(block_hyb[:, :, mask], w[mask], delta, bath_states_per_orbital, gamma = gamma, imag_only = imag_only)
        print (f"--> eb {block_eb}")
        print (f"--> v  {block_v}")
        for b in equiv_blocks:
            eb = np.append(eb, block_eb)
            # eb.append(block_eb)
            v_tmp = np.zeros((len(block_eb), n_orb), dtype = complex)
            v_tmp[:, blocks[b]] = block_v
            v = np.append(v, v_tmp, axis = 0)
        # v.append(v_tmp)
    # Transform hopping parameters back from the (close to) diagonal imaginary
    # part basis to the RSPt basis
    v = v @ Q.T

    sorted_indices = np.argsort(eb)
    # sorted_indices = range(len(eb))
    return eb[sorted_indices], v[sorted_indices], blocks
    # return eb, v

def fit_block(hyb, w, delta, bath_states_per_orbital, gamma, imag_only):
    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1 = 0, axis2 = 1), axis = 1))
    n_bath_states = bath_states_per_orbital * hyb.shape[0]
    # widths is the peak widths we are interested in, in units of de = w[1] - w[0]
    max_width = int(delta/(w[1] - w[0]))
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
    def weight(p):
        return np.exp(-2*np.abs(w[p]))
    weights = weight(peaks) # /np.sum(weight(peaks))
    sorted_indices = np.argsort(weights*(hyb_trace[peaks]))
    sorted_peaks = peaks[sorted_indices][-n_bath_states:][::-1]

    bath_energies = w[sorted_peaks]
    plt.plot(w, hyb_trace)
    plt.scatter(w[sorted_peaks], hyb_trace[sorted_peaks])
    plt.show()
    
    min_cost = np.inf
    v_best = None
    for _ in range(20):
        v, costs = get_v(w + delta*1j, hyb, bath_energies, gamma = gamma, imag_only = imag_only)
        if np.max(np.abs(costs)) < min_cost:
            min_cost = np.max(np.abs(costs))
            v_best = v
    model_hyb = get_hyb(w + delta*1j, bath_energies, v_best)
    if hyb.shape[0] > 1:
        fig, ax = plt.subplots(nrows = hyb.shape[0], ncols = hyb.shape[1], sharex = 'all', sharey = 'all')
        for row in range(hyb.shape[0]):
            for col in range(hyb.shape[1]):
                ax[row, col].plot(w, np.imag(hyb[row, col]), color = 'tab:blue')
                ax[row, col].plot(w, np.imag(model_hyb[row, col]), color = 'tab:orange')
    else:
        plt.plot(w, np.imag(hyb[0, 0]), color = 'tab:blue')
        plt.plot(w, np.imag(model_hyb[0, 0]), color = 'tab:orange')
    plt.show()
    # v_max = np.max(np.abs(v_best))
    # v_best[np.abs(v_best) < 0.01*v_max] = 0
    return bath_energies, v_best

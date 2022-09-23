import numpy as np

from rspt2spectra import orbitals
from rspt2spectra import offdiagonal
from rspt2spectra import energies
from rspt2spectra import h2imp


def run(hyb, hdft, e_wins, blocks, bath_states, rot_spherical, w, eim, wsparse = 1, gamma = 0.01):
    """
    Calculate h0. In block form h0 can be written
    [ hlda  V^+ ]
    [  V     Eb ]
    This function will fit the bath energies (Eb) and the hopping (V, V^+ ) 
    to the real energy hybridization function (hyb).
    Parameters:
    hyb           -- The real frequency hybridiaztion function. Used to fit the bath states.
    hdft          -- The DFT hamiltonian, projected onto the impurity orbitals.
    e_wins        -- Energy windows, used to fit bath states.
    blocks        -- Orbital blocks in the hybdridization function/DFT hamiltonian.
    n_bath_states   -- For every block, for every energy window, how many bath states to fit in the energy window. (Recommended 1).
    rot_spherical -- Transformation matrix to transform to spherical harmonics basis.
    w             -- Real frequency mesh.
    eim           -- All real frequency quantities are evaluated i*eim above the real frequency axis.
    w_sparse      -- Use every w_sparse frequency in w.
    gamma         -- Regularization parameter.

    Returns:
    h0   -- The non-interacting impurity hamiltonian in operator form.
    """
    n_orb = sum(len(block) for block in blocks)

    # Calculate bath and hopping parameters.
    eb, v = offdiagonal.get_eb_v(w, eim, hyb, blocks, w_sparse,
                                 e_wins,
                                 n_bath_states,
                                 (w[0], w[-1]), False, gamma)
    print('\n \n')
    print('Bath state energies')
    print(np.array_str(eb, max_line_width=1000, precision=3, suppress_small=True))
    print('Hopping parameters')
    print(np.array_str(v, max_line_width=1000, precision=3, suppress_small=True))
    print('Shape of bath state energies:', np.shape(eb))
    print('Shape of hopping parameters:', np.shape(v))

    eig_dft = np.linalg.eigvalsh(hdft)
    print ('Eigenvalues of the DFT hamiltonian')
    print (eig_dft)

    h = np.zeros((n_imp+len(eb),n_imp+len(eb)), dtype=np.complex)
    h[:n_orb, :n_orb] = hdft
    h[:n_orb, n_orb:] = np.conj(v.T)
    h[n_orb:, :n_orb] = v
    np.fill_diagonal(h[n_orb:, n_orb:], eb)
    assert np.sum(np.abs(h - np.conj(h.T))) < 1e-10

    u = np.identity_like(h)
    u[:n_orb, :n_orb] = rot_spherical
    h_sph = np.dot(np.transpose(np.conj(u)), np.dot(h, u))
    assert np.sum(np.abs(h_sph - np.conj(h_sph.T))) < 1e-10

    h_op = h2imp.get_H_operator_from_dense_rspt_H_matrix(h_sph,
                                                          ang=(n_orb//2-1)//2)
    return h_op


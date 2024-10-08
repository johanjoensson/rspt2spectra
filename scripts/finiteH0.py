#!/usr/bin/env python3

# finiteH0
# Script for analyzing hybridization functions generated by RSPt.
# Generate a finite size non-interacting Hamiltonian, expressed in
# a spherical harmonics basis.


import matplotlib.pylab as plt
import numpy as np

from rspt2spectra.constants import eV
from rspt2spectra import readfile
from rspt2spectra import orbitals
from rspt2spectra import offdiagonal
from rspt2spectra import energies
from rspt2spectra import h2imp
from rspt2spectra import hyb_fit

# Read input parameters from local file
import rspt2spectra_parameters as r2s
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size
# We need to specify how many bath sets should be used for each block.
assert np.shape(r2s.n_bath_sets_foreach_block_and_window)[0] == len(r2s.blocks)
# We need to specify how many bath sets should be used for each window.
assert np.shape(r2s.n_bath_sets_foreach_block_and_window)[1] == len(r2s.wborders)

# Help variables
# Files for hybridization function
file_re_hyb = "real-hyb-" + r2s.basis_tag + ".dat"
file_im_hyb = "imag-hyb-" + r2s.basis_tag + ".dat"
# Name of RSPt's output file.
outfile = "out"
# Number of considered impurity orbitals
n_imp = sum(len(block) for block in r2s.blocks)

# Read RSPt's diagonal hybridization functions
w, hyb_diagonal = readfile.hyb(file_re_hyb, file_im_hyb, only_diagonal_part=True)
assert n_imp == np.shape(hyb_diagonal)[0]
# All hybridization functions
w, hyb = readfile.hyb(file_re_hyb, file_im_hyb)

# Get unitary transformation matrix.
rot_spherical = orbitals.get_u_transformation(
    np.shape(hyb)[0],
    r2s.basis_tag,
    (n_imp // 2 - 1) // 2,
    irr_flag=r2s.irr_flag,
    verbose_text=r2s.verbose_text,
)
corr_to_cf = np.identity(rot_spherical.shape[0])

if r2s.verbose_fig and rank == 0:
    # Plot diagonal and off diagonal hybridization functions separately.
    offdiagonal.plot_diagonal_and_offdiagonal(w, hyb_diagonal, hyb, r2s.xlim)
    # Plot all orbitals, both real and imaginary parts.
    offdiagonal.plot_all_orbitals(w, hyb, xlim=r2s.xlim)

# Calculate bath and hopping parameters.
# eb, v = offdiagonal.get_eb_v(w, r2s.eim, hyb, r2s.blocks, r2s.wsparse,
#                              r2s.wborders,
#                              r2s.n_bath_sets_foreach_block_and_window,
#                              r2s.xlim, r2s.verbose_fig, r2s.gamma)
v, H_bath = hyb_fit.fit_hyb(
    w,
    r2s.eim,
    hyb,
    corr_to_cf,
    rot_spherical,
    r2s.bath_states_per_orbital,
    gamma=r2s.gamma,
    imag_only=r2s.imag_only,
    x_lim=(w[0], 0 if r2s.valence_bath_only else w[-1]),
    verbose=rank == 0,
    comm=comm,
    new_v=True,
    exp_weight=2 / eV,
)

if rank == 0:
    print("\n \n")
    print("Bath state energies")
    print(np.array_str(eb, max_line_width=1000, precision=3, suppress_small=True))
    print("Hopping parameters")
    print(np.array_str(v, max_line_width=1000, precision=3, suppress_small=True))
    print("Shape of bath state energies:", np.shape(eb))
    print("Shape of hopping parameters:", np.shape(v))

if r2s.verbose_fig and rank == 0:
    # Relative distribution of hopping parameters
    plt.figure()
    plt.hist(np.abs(v).flatten() / np.max(np.abs(v)), bins=100)
    plt.xlabel("|v|/max(|v|)")
    # plt.show()
    plt.savefig("hopping_distribution_rel.png")
    # Relative values of the hopping parameters
    plt.figure()
    plt.plot(sorted(np.abs(v).flatten()) / np.max(np.abs(v)), "-o")
    plt.ylabel("|v|/max(|v|)")
    # plt.show()
    plt.savefig("hopping_values_rel.png")

    # Distribution of hopping parameters
    plt.figure()
    plt.hist(np.abs(v).flatten(), bins=100)
    plt.xlabel("|v|")
    # plt.show()
    plt.savefig("hopping_distribution.png")
    # Absolute values of the hopping parameters
    plt.figure()
    plt.plot(sorted(np.abs(v).flatten()), "-o")
    plt.ylabel("|v|")
    # plt.show()
    plt.savefig("hopping_values.png")

if rank == 0:
    print("{:d} elements in v.".format(v.size))
    v_mean = np.mean(np.abs(v))
    v_median = np.median(np.abs(v))
    print("<v> = ", v_mean)
    print("v_median = ", v_median)
    r_cutoff = 0.01
    mask = np.abs(v) < r_cutoff * np.max(np.abs(v))
    print(
        "{:d} elements in v are smaller than {:.3f}*v_max.".format(
            v[mask].size, r_cutoff
        )
    )

    # Check small non-zero values.
    mask = np.logical_and(0 < np.abs(v), np.abs(v) < r_cutoff * np.max(np.abs(v)))
    print("{:d} elements in v are close to zero (of {:d})".format(v[mask].size, v.size))
    # One might want to put these hopping parameters to zero.
    # v[mask] = 0

    # print('Absolut values of these elements:')
    # print(sorted(np.abs(v[mask])))


# Extract the impurity energies from the local Hamiltonian
# and the chemical potential.
hs, labels = energies.parse_matrices(outfile)
mu = energies.get_mu()
for h, label in zip(hs, labels):
    # Select Hamiltonian from correct cluster
    if label == r2s.basis_tag:
        if rank == 0:
            print("Extract local H0 from cluster:", label)
            print()
        e_rspt = eV * (h - mu * np.eye(n_imp))
eig, _ = np.linalg.eigh(e_rspt)
if rank == 0:
    print("RSPt's local hamiltonian")
    print(np.array_str(e_rspt, max_line_width=1000, precision=3, suppress_small=True))
    print()
    print("Eigenvalues of RSPt's local Hamiltonian:")
    print(eig)
    print()

# Construct the non-interacting Hamiltonian
h = np.zeros((n_imp + len(eb), n_imp + len(eb)), dtype=complex)
# Onsite energies of impurity orbitals
h[:n_imp, :n_imp] = e_rspt
# Bath state energies
np.fill_diagonal(h[n_imp:, n_imp:], eb)
# Hopping parameters
h[n_imp:, :n_imp] = v
h[:n_imp, n_imp:] = np.conj(v).T

u = orbitals.get_u_transformation(
    np.shape(h)[0],
    r2s.basis_tag,
    (n_imp // 2 - 1) // 2,
    irr_flag=r2s.irr_flag,
    verbose_text=r2s.verbose_text,
)
# Make sure Hamiltonian is hermitian
assert np.sum(np.abs(h - np.conj(h.T))) < 1e-10
# Rotate (back) to spherical harmonics basis
h_sph = np.dot(np.transpose(np.conj(u)), np.dot(h, u))
# Make sure Hamiltonian is hermitian
assert np.sum(np.abs(h_sph - np.conj(h_sph.T))) < 1e-10

if r2s.verbose_text and rank == 0:
    print("Dimensions of Hamiltonian:", np.shape(h_sph))
    print("Hamiltonian in spherical harmonics basis:")
    print("Correlated block:")
    print("Real part:")
    print(
        np.array_str(np.real(h_sph[:n_imp, :n_imp]), precision=3, suppress_small=True)
    )
    print("Imag part:")
    print(
        np.array_str(np.imag(h_sph[:n_imp, :n_imp]), precision=3, suppress_small=True)
    )
    print("Number of non-zero elements in H:", len(np.flatnonzero(h_sph)))

hOperator = h2imp.get_H_operator_from_dense_rspt_H_matrix(
    h_sph, ang=(n_imp // 2 - 1) // 2
)
if r2s.verbose_text and rank == 0:
    print("Hamiltonian operator:")
    print(hOperator)
    # repr(hOperator)
    print()
    print("len(hOperator) = {:d}".format(len(hOperator)))
    print(
        "{:.3f} bath states per impurity spin-orbital.".format(
            (np.shape(h_sph)[0] - n_imp) / n_imp
        )
    )
    print("{:d} bath states in total.".format(np.shape(h_sph)[0] - n_imp))
if rank == 0:
    h2imp.write_to_file(hOperator, r2s.output_filename)

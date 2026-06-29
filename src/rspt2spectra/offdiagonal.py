#!/usr/bin/env python3

"""

offdiagonal
===========

This module contains functions to treat
off-diagonal hybridization functions.

"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from os import environ
from scipy.optimize import (
    minimize,
    Bounds,
    basinhopping,
    differential_evolution,
)

from time import perf_counter

from rspt2spectra.energies import cog
from rspt2spectra.weight_functions import weight_functions
import matplotlib.pyplot as plt


def plot_diagonal_and_offdiagonal(w, hyb_diagonal, hyb, xlim):
    """
    Plot diagonal and offdiagonal hybridization functions separately.
    """
    # Number of considered impurity orbitals
    n_imp = np.shape(hyb_diagonal)[0]

    # Plot mask
    mask = np.logical_and(xlim[0] < w, w < xlim[1])
    # Diagonal functions
    # Real part
    plt.figure()
    for i in range(n_imp):
        plt.plot(w[mask], hyb_diagonal[i, mask].real, label=str(i))
    plt.legend()
    plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.title("Diagonal functions, real part")
    # plt.show()
    plt.savefig("diag_real.png")
    # -Imag part
    plt.figure()
    for i in range(n_imp):
        plt.plot(w[mask], -hyb_diagonal[i, mask].imag, label=str(i))
    plt.legend()
    plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.title("Diagonal functions, -imag part")
    # plt.show()
    plt.savefig("diag_imag.png")

    # Off diagonal functions
    # Real part
    plt.figure()
    for i in range(n_imp):
        for j in list(range(i)) + list(range(i + 1, n_imp)):
            plt.plot(w[mask], hyb[i, j, mask].real)
    plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.title("Off diagonal functions, real part")
    # plt.show()
    plt.savefig("offdiag_real.png")
    # -Imag part
    plt.figure()
    for i in range(n_imp):
        for j in list(range(i)) + list(range(i + 1, n_imp)):
            plt.plot(w[mask], -hyb[i, j, mask].imag)
    plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.title("Off diagonal functions, -imag part")
    # plt.show()
    plt.savefig("offdiag_imag.png")


label = -1


def plot_all_orbitals(w, hyb_orig, hyb_model=None, xlim=None):
    """
    Plot functions for all orbitals, for both hyb and hyb_model.
    """

    global label

    assert np.shape(hyb_orig)[0] == np.shape(hyb_orig)[1]
    if hyb_model is not None:
        assert np.shape(hyb_orig) == np.shape(hyb_model)
    if xlim is None:
        mask = np.ones_like(w, dtype=np.bool)
    else:
        # Mask for plotting
        mask = np.logical_and(xlim[0] < w, w < xlim[1])
    # Number of rows and columns in the figure
    n = np.shape(hyb_orig)[0]
    if n > 1:
        # All real functions
        fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
        for i in range(n):
            for j in range(n):
                axes[i, j].plot(w[mask], hyb_orig[i, j, mask].real, label="original")
                if hyb_model is not None:
                    axes[i, j].plot(w[mask], hyb_model[i, j, mask].real, label="model")
                # axes[i,j].grid()
        if xlim is not None:
            plt.xlim(xlim)
        # plt.ylim(ylim)
        axes[0, n // 2].set_title("Real part")
        # plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.98, wspace=0.03, hspace=0.03)
        # plt.show()
        if label >= 0:
            plt.savefig("real_orbital_" + repr(label) + ".png")
        else:
            plt.savefig("real_orbitals.png")

        # All -imag functions
        fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
        for i in range(n):
            for j in range(n):
                axes[i, j].plot(w[mask], -hyb_orig[i, j, mask].imag, label="original")
                if hyb_model is not None:
                    axes[i, j].plot(w[mask], -hyb_model[i, j, mask].imag, label="model")
                # axes[i,j].grid()
        if xlim is not None:
            plt.xlim(xlim)
        # plt.ylim(ylim)
        axes[0, -1].legend()
        axes[0, n // 2].set_title("- Imag part")
        # plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.98, wspace=0.03, hspace=0.03)
        # plt.show()
        if label >= 0:
            plt.savefig("imag_orbital_" + repr(label) + ".png")
        else:
            plt.savefig("imag_orbitals.png")
    elif n == 1:
        # All real functions
        fig = plt.figure()
        for i in range(n):
            for j in range(n):
                plt.plot(w[mask], hyb_orig[i, j, mask].real, label="original")
                if hyb_model is not None:
                    plt.plot(w[mask], hyb_model[i, j, mask].real, label="model")
                # plt.grid()
        if xlim is not None:
            plt.xlim(xlim)
        # plt.ylim(ylim)
        plt.title("Real part")
        plt.tight_layout()
        # plt.subplots_adjust(top=0.95,right=0.98, wspace=0.03, hspace=0.03)
        # plt.show()
        if label >= 0:
            plt.savefig("real_orbital_" + repr(label) + ".png")
        else:
            plt.savefig("real_orbitals.png")

        # All -imag functions
        fig = plt.figure()
        for i in range(n):
            for j in range(n):
                plt.plot(w[mask], -hyb_orig[i, j, mask].imag, label="original")
                if hyb_model is not None:
                    plt.plot(w[mask], -hyb_model[i, j, mask].imag, label="model")
                # plt.grid()
        if xlim is not None:
            plt.xlim(xlim)
        # plt.ylim(ylim)
        plt.legend()
        plt.title("- Imag part")
        plt.tight_layout()
        # plt.subplots_adjust(top=0.95,right=0.98, wspace=0.03, hspace=0.03)
        # plt.show()
        if label >= 0:
            plt.savefig("imag_orbital_" + repr(label) + ".png")
        else:
            plt.savefig("imag_orbitals.png")
    else:
        sys.exit("Positive number of impurity orbitals required.")

    label += 1


def get_eb_v_for_one_block(
    w,
    eim,
    hyb,
    block,
    wsparse,
    wborders,
    n_bath_sets_foreach_window,
    xlim=None,
    verbose_fig=False,
    gamma=0.0,
    imag_only=True,
    v_cutoff=None,
):
    """
    Return bath energies and hybridization hopping parameters
    for a specific block.

    """
    if v_cutoff is None:
        v_cutoff = 0
    # Select energies in real axis mesh.
    w_select = w[::wsparse]
    assert w_select[1] - w_select[0] < 2 * eim
    # For each block, fit a discretized hybridization function to
    # the original hybridization function.
    # Number of bath orbitals, for each energy window.
    n_bath_foreach_window = np.array(n_bath_sets_foreach_window) * len(block)
    # Select a subset of all impurity orbitals
    hyb_block = hyb[np.ix_(block, block, w == w_select)]
    ebs, w_index = get_ebs(w_select, hyb_block, wborders, n_bath_foreach_window)
    mask_tmp = np.logical_and(np.min(wborders) < w_select, w_select < np.max(wborders))
    n_data_points = len(block) ** 2 * len(w_select[mask_tmp])
    print("Fit to approx {:d} data points.".format(n_data_points))
    n_param = np.sum(n_bath_foreach_window) * len(block) * 2
    print("Use {:d} real-valued parameters in the fit.".format(n_param))
    vs, costs = get_vs(
        w_select + 1j * eim, hyb_block, wborders, ebs, gamma=gamma, imag_only=imag_only
    )
    print("Cost function values (without regularization):")
    print(costs)
    eb = merge_ebs(ebs)
    print(eb)
    v = merge_vs(vs)
    v_max = np.max(np.abs(v))
    mask = np.any(np.abs(v) > v_cutoff * v_max, axis=1)
    if verbose_fig:
        hyb_model = get_hyb(w_select + 1j * eim, eb, v)
        print("Plot model and original hybridization functions..")
        plot_all_orbitals(w_select, hyb_block, hyb_model, xlim)
        # Distribution of hopping parameters
        plt.figure()
        plt.hist(np.abs(v).flatten() / np.max(np.abs(v)), bins=30)
        plt.xlabel("|v|/max(|v|)")
        plt.show()
        # Absolute values of the hopping parameters
        plt.figure()
        plt.plot(sorted(np.abs(v).flatten()) / np.max(np.abs(v)), "o")
        plt.plot([v_cutoff] * len(v.flatten()), "--", color="tab:red")
        plt.ylabel("|v|/max(|v|)")
        plt.show()
        # plt.savefig('hopping_distribution.png')
        plt.close()
        print("{:d} elements in v.".format(v.size))
        v_mean = np.mean(np.abs(v))
        v_median = np.median(np.abs(v))
    # v[np.abs(v) < r_cutoff*v_max] = 0
    # for i, (v_i, eb_i) in enumerate(zip(v, eb)):
    #     if np.any(np.abs(v_i) > r_cutoff*v_max):
    #         ebv.append(eb_i)
    #         vp.append(v_i)
    return eb[mask], v[mask], w_index


def get_eb_v(
    w,
    eim,
    hyb,
    blocks,
    wsparse,
    wborders,
    n_bath_sets_foreach_block_and_window,
    xlim=None,
    verbose_fig=False,
    gamma=0.0,
    imag_only=True,
    v_cutoff=None,
):
    """
    Return bath and hopping parameters by discretizing hybridization functions.
    """
    # Number of considered impurity orbitals
    n_imp = sum(len(block) for block in blocks)
    # Bath energies
    eb = []
    # Energy window associated to each bath state
    window_indices = []
    # Hopping parameters
    v = []
    # Loop over blocks
    for block_i, (block, n_bath_sets_foreach_window) in enumerate(
        zip(blocks, n_bath_sets_foreach_block_and_window)
    ):
        # Calculate bath energies and hopping parameters for each block.
        eb_block, v_block, window_index_block = get_eb_v_for_one_block(
            w,
            eim,
            hyb,
            block,
            wsparse,
            wborders,
            n_bath_sets_foreach_window,
            xlim,
            verbose_fig,
            gamma=gamma,
            imag_only=imag_only,
            v_cutoff=v_cutoff,
        )
        eb.append(eb_block)
        v_sparse = np.zeros((len(eb_block), n_imp), dtype=complex)
        v_sparse[:, block] = v_block
        v.append(v_sparse)
        window_indices += [i for i in window_index_block]
    eb = np.hstack(eb)
    v = np.vstack(v)
    # Sort bath states according to the energy windows.
    # For example, the bath states with energies in the range of
    # the first energy window is placed first in the sorted list.
    # This is important if have both occupied and unoccupied bath states,
    # since we then want the unoccupied bath states to be sorted after
    # the occupied bath states.
    eb, v = reshuffle(eb, v, wborders, np.array(window_indices))
    return eb, v


def reshuffle(eb, v, wborders, w_indices):
    """
    Sort bath states according to the energy windows.

    For example, the bath states with energies in the range of
    the first energy window is placed first in the sorted list.
    This is important if have both occupied and unoccupied bath states,
    since we then want the unoccupied bath states to be sorted after
    the occupied bath states.
    """
    eb_new = []
    v_new = []
    for i, wborder in enumerate(wborders):
        # mask = np.logical_and(np.logical_and(wborder[0] <= eb, eb < wborder[1]), w_indices == i)
        mask = np.logical_and(wborder[0] <= eb, eb < wborder[1])
        eb_new.append(eb[mask.flatten()])
        v_new.append(v[mask.flatten(), :])
    eb_new = np.hstack(eb_new)
    v_new = np.vstack(v_new)
    return eb_new, v_new


def get_eb(w, hyb, n_b):
    """
    Return bath energies.

    Parameters
    ----------
    w : array(M)
        Real part of energy mesh.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    n_b : int
        Number of bath orbitals per window.

    Returns
    -------
    eb : array(n_b)
        Bath energies.

    """
    n_w = len(w)
    n_imp = np.shape(hyb)[0]
    eb = np.zeros(n_b, dtype=float)
    # Selection of bath energies depends on how many
    # bath orbitals have compared to the number of
    # impurity orbitals.
    if n_b < 0:
        sys.exit("Positive number of bath energies expected.")
    elif n_b == 0:
        # No bath orbitals
        print("Skipping orbital with no bath states")
    elif n_b == 1:
        # Bath energy at the center of gravity of the imaginary part
        # of the hybridization function trace.
        eb[:] = cog(w, -np.trace(hyb.imag))
    elif n_b % n_imp == 0:
        # Remove the False variable below to activate a special treatment
        # of the case n_b == n_imp.
        if n_b == n_imp and False:
            # Bath energies at the center of gravities of each
            # diagonal hybridization function (its imaginary part).
            for i in range(n_b):
                eb[i] = cog(w, -hyb[i, i, :].imag)
        else:
            # Uniformly distribute n_b // n_imp energies on the mesh.
            dw = (w[-1] - w[0]) / (n_b / n_imp + 1)
            es = np.linspace(w[0] + dw, w[-1] - dw, n_b // n_imp)
            # Give each energy degeneracy n_imp.
            for i, e in enumerate(es):
                eb[n_imp * i : n_imp * (1 + i)] = e
    else:
        # Uniformly distribute the bath energies on the mesh.
        dw = (w[-1] - w[0]) / (n_b + 1)
        eb = np.linspace(w[0] + dw, w[-1] - dw, n_b)
    return eb


def get_ebs(w, hyb, wborders, n_b):
    """
    Return bath energies, for each energy window.

    Parameters
    ----------
    w : array(M)
        Real part of energy mesh.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    wborders : array(K, 2)
        Window borders.
    n_b : array(K)
        Number of bath orbitals for each window.

    Returns
    -------
    ebs : tuple(K)
        Bath energies, for each energy window.
        Each element contains the bath energies for that energy window,
        as an array. len(ebs[i]) == n_b[i]


    """
    n_w = len(w)
    n_imp = np.shape(hyb)[0]
    n_windows = np.shape(wborders)[0]
    # ebs = np.zeros((n_windows, n_b), dtype=float)
    ebs = []
    w_index = []
    # Treat each energy window as seperate.
    for a, wborder in enumerate(wborders):
        mask = np.logical_and(wborder[0] <= w, w <= wborder[1])
        # ebs[a,:] = get_eb(w[mask], hyb[:,:,mask], n_b)
        ebs.append(get_eb(w[mask], hyb[:, :, mask], n_b[a]))
        # if(n_b[a] > 0):
        w_index += [a] * n_b[a]
    return ebs, w_index


def get_hyb(z, eb, v):
    """
    Return the hybridization functions, as a rank 3 tensor.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh.
    eb : array(B)
        Bath energies.
    v : array(B, N)
        Hopping parameters.

    Returns
    -------
    hyb : array(M, N,N,)
        Hybridization functions.

    """
    n_w = len(z)
    n_b, n_imp = v.shape
    hyb = np.zeros((n_w, n_imp, n_imp), dtype=complex)

    # Loop over all bath energies
    for b, e in enumerate(eb):
        # Add contributions from each bath
        hyb[:] += (
            np.outer(v[b].conj(), v[b])[np.newaxis, ...]
            * (1 / (z - e))[:, np.newaxis, np.newaxis]
        )

    return hyb


def get_hyb_2(z, eb, v, C=None):
    """
    Return the hybridization functions, as a rank 3 tensor.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh.
    eb : array(S, N_B)
        Bath energies.
    v : array(S, N_B, N, N)
        Hopping parameters.
    C : array(N, N), optional
        Constant Hermitian shift added to every frequency point.

    Returns
    -------
    hyb : array(S, M, N,N)
        Hybridization functions.

    """
    A = np.conj(np.transpose(v, (0, 1, 3, 2))) @ v  # (S, N_B, N, N)
    G = 1.0 / (z[np.newaxis, :, np.newaxis] - eb[:, np.newaxis, :])  # (S, M, N_B)
    result = np.einsum("smb,sbij->smij", G, A)
    if C is not None:
        if C.ndim == 2:  # single (n_imp, n_imp): broadcast over S and M
            result = result + C[np.newaxis, np.newaxis]
        else:  # batched (S, n_imp, n_imp): broadcast over M only
            result = result + C[:, np.newaxis]
    return result


def get_vs(z, hyb, wborders, ebs, gamma=0.0, imag_only=True):
    """
    Return optimized hopping parameters.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh, just above the real axis.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    wborders : array(K, 2)
        Window borders.
    ebs : tuple(K)
        Bath energies, for each energy window.
        Each element contains the bath energies for that energy window,
        as an array(B), where B is different for each element.
    gamma : float
        Regularization parameter

    Returns
    -------
    vs : tuple(K)
        Hopping parameters, for each energy window.
        Each element in an array(B, N), where B is different for each element.
    costs : array(K)
        Cost function values, without regularization, for each energy window.

    """
    n_w = len(z)
    n_imp = np.shape(hyb)[0]
    # n_windows, n_b = np.shape(ebs)
    n_windows = len(ebs)
    # Hopping parameters
    # vs = np.zeros((n_windows, n_b, n_imp), dtype=complex)
    vs = []
    # Cost function values
    costs = np.zeros(n_windows, dtype=float)
    # Treat each energy window as seperate.
    for a, wborder in enumerate(wborders):
        if len(ebs[a]) > 0:
            mask = np.logical_and(wborder[0] <= z.real, z.real <= wborder[1])
            # vs[a,:,:], costs[a] = get_v(z[mask], hyb[:,:,mask], ebs[a,:], gamma)
            v, costs[a] = get_v(
                z[mask], hyb[:, :, mask], ebs[a], gamma, imag_only=imag_only
            )
        else:
            v = np.zeros((0, n_imp), dtype=float)
            costs[a] = 0.0
        vs.append(v)
    return vs, costs


def generate_hopping_guess(z, hyb, eb, gamma, realvalue_v, rng):
    """
    Build hopping-parameter guesses for each population member via a
    vectorised linear least-squares solve.

    For fixed bath energies eb[p], the hybridization model is linear in
    A[b] = V[b]^H V[b]:
        hyb_model[z, i, j] = sum_b  A[b, i, j] / (z - eb[p, b])
    We solve G @ A_flat = hyb_upper (one lstsq per population member, all
    orbital pairs at once) and recover V via a PSD-safe eigh factorisation.
    This replaces the previous O(population * n_pairs) SLSQP loop.
    """
    population_size = eb.shape[0]
    n_eb = eb.shape[1]
    n_orb = hyb.shape[1]
    triu_i, triu_j = np.triu_indices(n_orb)
    tril_i, tril_j = np.tril_indices(n_orb, k=-1)

    v = np.zeros((population_size, n_eb, n_orb, n_orb), dtype=complex)
    for p in range(population_size):
        # G[m, b] = 1 / (z[m] - eb[p, b])
        G = 1.0 / (z[:, None] - eb[p, None, :])  # (M, n_eb)
        # Solve for all upper-triangle pairs simultaneously.
        # A_flat: (n_eb, n_triu)  where A_flat[b, k] ≈ A[b, triu_i[k], triu_j[k]]
        A_flat, _, _, _ = np.linalg.lstsq(G, hyb[:, triu_i, triu_j], rcond=None)
        A = np.zeros((n_eb, n_orb, n_orb), dtype=complex)
        A[:, triu_i, triu_j] = A_flat  # .T
        A[:, tril_i, tril_j] = np.conj(A[:, tril_j, tril_i])  # Hermitian completion
        if realvalue_v:
            A = A.real
        # PSD-safe V such that V[b]^H V[b] = A[b].
        lam, U = np.linalg.eigh(A)
        lam = np.clip(lam.real, 0.0, None)
        v[p] = np.sqrt(lam)[:, :, None] * np.conj(np.swapaxes(U, -1, -2))

    return v


def get_p0(z, hyb, eb, gamma, imag_only, realvalue_v):
    n_imp = np.shape(hyb)[1]
    n_b = len(eb)
    if n_imp == 1:
        return np.random.randn(n_b if realvalue_v else 2 * n_b)
    v0 = np.zeros((n_b, n_imp), dtype=complex)
    for i in range(n_imp):
        for j in range(i + 1):
            # p0 = 2*np.random.randn(1 if realvalue_v else 2)
            p0 = 2 * np.random.randn(n_b // n_imp if realvalue_v else 2 * n_b // n_imp)
            # fun = lambda p: cost_function(p, eb[[b_i + i]], z, hyb[i, j, :].reshape((1, 1, len(z))), gamma, imag_only, output='value')
            fun = lambda p: cost_function(
                p,
                eb[i::n_imp],
                z,
                hyb[:, i, j].reshape((len(z), 1, 1)),
                gamma=gamma,
                only_imag_part=imag_only,
                output="value",
            )
            if imag_only:
                jac = lambda p: cost_function(
                    p,
                    eb[i::n_imp],
                    z,
                    hyb[:, i, j].reshape((len(z), 1, 1)),
                    gamma=gamma,
                    only_imag_part=True,
                    output="gradient",
                )
                # Minimize cost function
                res = minimize(fun, p0, jac=jac, tol=1e-3)
            else:
                res = minimize(fun, p0, tol=1e-3)
            v = unroll(res.x, n_b // n_imp, 1).reshape((n_b // n_imp,))
            v0[i::n_imp, j] = v
            v0[j::n_imp, i] = np.conj(v)
    p0 = inroll(v0)
    # p0 = inroll(np.abs(v0))
    return p0 if not realvalue_v else p0[: n_b * n_imp]


def get_v(z, hyb, eb, gamma=0.0, imag_only=True, realvalue_v=False):
    """
    Return optimized hopping parameters.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh, just above the real axis.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    eb : array(B)
        Bath energies.
    gamma : float
        Regularization parameter

    Returns
    -------
    v : array(B, N)
        Hopping parameters.

    """
    n_w = len(z)
    n_imp = np.shape(hyb)[0]
    n_b = len(eb)
    # Initialize hopping parameters.
    # Treat complex-valued parameters,
    # by doubling the number of parameters.
    if realvalue_v:
        n = n_b * n_imp
    else:
        n = 2 * n_b * n_imp
    p0 = np.random.randn(n)
    # Define cost function as a function of a hopping parameter
    # vector.
    fun = lambda p: cost_function(
        p, eb, z, np.moveaxis(hyb, -1, 0), gamma, imag_only, output="value"
    )
    if imag_only:
        jac = lambda p: cost_function(
            p, eb, z, np.moveaxis(hyb, -1, 0), gamma, True, output="gradient"
        )
        # Minimize cost function
        res = minimize(fun, p0, jac=jac, tol=1e-12)
    else:
        res = minimize(fun, p0)
    # The solution
    p = res.x
    # Cost function value, with regularization.
    c = cost_function(
        p, eb, z, np.moveaxis(hyb, -1, 0), only_imag_part=imag_only, output="value"
    )
    # Convert hopping parameters to physical shape.
    v = unroll(p, n_b, n_imp)
    return v, c


def unroll(p, n_b, n_imp):
    """
    Return hybridization parameters as a matrix.

    Parameters
    ----------
    p : real array(K, S)
        Hybridization parameters as a stack of vectors.
    n_b : int
        Number of bath orbitals.
    n_imp : int
        Number of impurity orbitals.

    Returns
    -------
    v : complex array(S, n_b, n_imp)
        Hybridization parameters as a matrix.

    """
    onedimensional = len(p.shape) == 1
    p_c = p
    triu_rows, triu_columns = np.triu_indices(n_imp)
    r = p.shape[0]
    if r != n_b * len(triu_columns):
        # The real parts are the first r elements in p
        # the imaginary parts are the rest
        r //= 2
        p_c = p[:r] + 1j * p[r:]
    if onedimensional:
        # non_zero_indices = np.ix_(range(n_b), triu_rows, triu_columns)
        res = np.zeros((n_b, n_imp, n_imp), dtype=complex)
        res[:, triu_rows, triu_columns] = p_c.reshape((n_b, len(triu_columns)))
        return res
    # non_zero_indices = np.ix_(range(n_b), triu_rows, triu_columns, range(p.shape[1]))
    res = np.zeros((n_b, n_imp, n_imp, p.shape[1]), dtype=complex)
    # print(f"{res[non_zero_indices].shape=}")
    res[:, triu_rows, triu_columns] = p_c.reshape((n_b, len(triu_columns), p.shape[1]))
    return np.moveaxis(res, -1, 0)


def inroll(v):
    """
    Return hybridization parameters as a vector.

    Parameters
    ----------
    v : complex array(..., n_b, n_imp, n_imp)
        Hybridization parameters as a matrix.

    Returns
    -------
    p : real array(..., K)
        Hybridization parameters as a stack of vectors.

    """
    triu_rows, triu_columns = np.triu_indices(v.shape[-1])
    cplx = np.any(np.abs(v.imag)) > 0

    # res_shape = v.shape[:-3] + (v.shape[-3] * v.shape[-2] * v.shape[-1],)
    if cplx:
        return np.moveaxis(
            np.append(
                v[..., triu_rows, triu_columns].real,
                v[..., triu_rows, triu_columns].imag,
                axis=-1,
            ).reshape(v.shape[:-3] + (-1,)),
            0,
            -1,
        )
    return np.moveaxis(
        v[..., triu_rows, triu_columns].real.reshape(v.shape[:-3] + (-1,)), 0, -1
    )


def inroll_C(C):
    """Pack a Hermitian (n_imp × n_imp) matrix into a real vector.

    Diagonal entries are real; off-diagonal upper-triangle entries contribute
    both real and imaginary parts.  Returns a 1-D real array of length
    n_imp*(n_imp+1)//2 (real-symmetric) or n_imp^2 (complex Hermitian).
    """
    n_imp = C.shape[0]
    triu_i, triu_j = np.triu_indices(n_imp)
    diag_mask = triu_i == triu_j
    off_mask = ~diag_mask
    entries = C[triu_i, triu_j]
    if np.any(np.abs(entries.imag) > 1e-14):
        # real: all upper-triangle real parts; imag: off-diagonal imaginary parts only
        return np.concatenate([entries.real, entries[off_mask].imag])
    return entries.real.copy()


def unroll_C(p_C, n_imp):
    """Unpack a real vector (from inroll_C) into a Hermitian (n_imp × n_imp) matrix."""
    triu_i, triu_j = np.triu_indices(n_imp)
    n_triu = len(triu_i)
    off_mask = triu_i != triu_j
    n_off = int(np.sum(off_mask))
    C = np.zeros((n_imp, n_imp), dtype=complex)
    if len(p_C) == n_triu:  # real-symmetric
        C[triu_i, triu_j] = p_C
        C[triu_j, triu_i] = p_C  # symmetric
    else:  # complex Hermitian: n_triu real + n_off imaginary
        entries = p_C[:n_triu].astype(complex)
        entries[off_mask] += 1j * p_C[n_triu:]
        C[triu_i, triu_j] = entries
        C[triu_j[off_mask], triu_i[off_mask]] = np.conj(entries[off_mask])
        np.fill_diagonal(C, C.diagonal().real)
    return C


def merge_bath_states(ebs, vs):
    n_imp = vs.shape[1]
    sorted_idx = np.unravel_index(
        np.argsort(np.linalg.norm(vs, axis=(-2, -1))), vs.shape[:-2]
    )
    vs = vs[sorted_idx]
    ebs = ebs[sorted_idx]

    zeroth_moments = np.conj(np.transpose(vs, (0, 2, 1))) @ vs
    first_moments = ebs[..., np.newaxis, np.newaxis] * zeroth_moments
    A = np.sum(zeroth_moments, axis=0)
    # Use eigh-based pseudoinverse: safe when A is rank-deficient (zero-coupling orbitals).
    lam_A, U_A = np.linalg.eigh(A)
    tol = np.max(np.abs(lam_A)) * n_imp * np.finfo(float).eps * 1e4
    inv_lam = np.where(
        np.abs(lam_A) > tol, 1.0 / np.where(np.abs(lam_A) > tol, lam_A, 1.0), 0.0
    )
    Eb = (U_A * inv_lam) @ (np.conj(U_A.T) @ np.sum(first_moments, axis=0))
    eb = np.mean(np.linalg.eigvals(Eb).real)
    return eb.real[None], A[None]


def calc_moments(f, x, max_moment):
    if max_moment < 0:
        return np.zeros((0, f.shape[1].f.shape[2]))
    moments = [sp.integrate.simpson(f, x, axis=0)]
    # moments = [np.trapezoid(f, x, axis=0)]
    for m in range(1, max_moment + 1):
        moments.append(sp.integrate.simpson((x**m)[:, None, None] * f, x, axis=0))
        # moments.append(np.trapezoid((x**m)[:, None, None] * f, x, axis=0))
    return np.array(moments)


def cost_function(
    p,
    n_eb,
    z,
    hyb,
    gamma,
    only_imag_part,
    output,
    regularization_mode,
    max_moment: int,
):
    """
    Return cost function value.

    Since the imaginary part of the hybridization function,
    from one bath state, is more local in energy
    than the real part, the default is to only fit to the imaginary part.

    Parameters
    ----------
    p : real array(K)
        Hopping parameters.
    eb : array(B)
        Bath energies.
    z : complex array(M)
        Energy mesh, just above the real axis.
    hyb : complex array(N,N,M)
        RSPt hybridization functions.
    only_imag_part : bool
        If should consider only the imaginary part of the
        hybridization functions, or consider both real and
        imaginary part.
    gamma : float
        Regularization parameter
    output : str
        'value and gradient', 'value' or 'gradient'
    regularization_mode : str
        'L1', 'L2', 'sigmoid'
        Type of regularization.

    Returns
    -------
    c : float
        Cost function value.

    """
    # Dimensions. Help variables.
    n_w = len(z)
    n_imp = np.shape(hyb)[1]
    n_b = n_imp * n_eb

    # Number of data points to fit to.
    m = n_imp * n_imp * n_w
    assert hyb.size == m

    eb = np.repeat(p[:, :n_eb], n_imp)
    # Convert hopping parameters to physical shape.
    v = unroll(p[n_eb:], n_b, n_imp)

    hyb_model = get_hyb(z, eb, v)

    diff = hyb_model - hyb
    diff *= (1 / n_imp**2 * np.ones((n_imp, n_imp)) + np.eye(n_imp))[None, ...]
    if only_imag_part:
        diff = diff.imag
        loss = 1 / 2 * diff**2
    else:
        loss = 1 / 2 * np.abs(diff) ** 2

    model_moments = calc_moments(hyb_model, np.real(z), max_moment=max_moment)
    hyb_moments = calc_moments(hyb, np.real(z), max_moment=max_moment)
    moment_errors = 1 / 2 * np.abs(model_moments - hyb_moments) ** 2
    # Cost function value,
    # sum over two impurity orbital indices and
    # one energy index.
    c = 1 / m * np.sum(loss) + 1 / (2 * n_imp**2) * np.sum(moment_errors)
    # Add regularization terms
    if regularization_mode == "L1":
        c += gamma / len(p) * np.sum(np.abs(p))
    elif regularization_mode == "L2":
        c += gamma / (2 * len(p)) * np.sum(p**2)
    elif regularization_mode == "sigmoid":
        # Regularization with derivative being the sigmoid function.
        # This acts as a smoothened version of L1-regularization.
        # Delta determine the sharpness of the sigmoid function.
        delta = 0.1
        c += gamma / len(p) * np.sum(delta * np.log(np.cosh(p / delta)))

    else:
        raise NotImplementedError

    if output == "value":
        # Return only the cost function value.
        return c

    # Calculate gradient here...
    if not only_imag_part:
        raise RuntimeError(
            ("Gradient for fit to complex hybridization is not implemented yet...")
        )

    # Partial derivatives of the cost function with respect to the
    # real and imaginary part of the hopping parameters.
    dcdv_re = np.zeros((n_b, n_imp), dtype=float)
    dcdv_im = np.zeros((n_b, n_imp), dtype=float)
    if True:
        # Calculate the gradient using some numpy broadcasting trixs.
        # complex array(B,M)
        green_b = 1 / (z - np.atleast_2d(eb).T)
        # Loop over all impurity orbitals
        for r in range(n_imp):
            # Sum over columns of the hybridization matrix,
            # not being equal to column r.
            # Also sum over rows of the hybridization matrix,
            # not being equal to row r.
            for j in list(range(r)) + list(range(r + 1, n_imp)):
                # complex array(B,1)
                v_matrix = np.atleast_2d(v[:, j]).T
                # diff[r,j,:], real array(M)
                # Sum real array(B,M) along energy axis.
                dcdv_re[:, r] += np.sum(
                    diff[:, r, j] * np.imag(v_matrix * green_b), axis=1
                )
                dcdv_im[:, r] += np.sum(
                    diff[:, r, j] * np.imag(-1j * v_matrix * green_b), axis=1
                )
                # Sum real array(B,M) along energy axis.
                dcdv_re[:, r] += np.sum(
                    diff[:, j, r] * np.imag(v_matrix.conj() * green_b), axis=1
                )
                dcdv_im[:, r] += np.sum(
                    diff[:, j, r] * np.imag(1j * v_matrix.conj() * green_b), axis=1
                )
            # complex array(B,1)
            v_matrix = np.atleast_2d(v[:, r]).T
            # diff[r,r,:], real array(M)
            # Add contribution from case with i=j=r
            # Sum real array(B,M) along energy axis.
            dcdv_re[:, r] += np.sum(
                diff[:, r, r] * np.imag(2 * v_matrix.real * green_b), axis=1
            )
            dcdv_im[:, r] += np.sum(
                diff[:, r, r] * np.imag(2 * v_matrix.imag * green_b), axis=1
            )
    else:
        # Calculate the gradient without any numpy broadcasting trixs.
        # Loop over all impurity orbitals
        for r in range(n_imp):
            # Sum over columns of the hybridization matrix,
            # not being equal to column r.
            for j in list(range(r)) + list(range(r + 1, n_imp)):
                # Loop over all bath states
                for b in range(n_b):
                    # Sum over energies
                    dcdv_re[b, r] += np.sum(
                        diff[:, r, j] * np.imag(v[b, j] / (z - eb[b]))
                    )
                    dcdv_im[b, r] += np.sum(
                        diff[:, r, j] * np.imag(-1j * v[b, j] / (z - eb[b]))
                    )
            # Sum over rows of the hybridization matrix,
            # not being equal to row r.
            for i in list(range(r)) + list(range(r + 1, n_imp)):
                # Loop over all bath states
                for b in range(n_b):
                    # Sum over energies
                    dcdv_re[b, r] += np.sum(
                        diff[:, i, r] * np.imag(v[b, i].conj() / (z - eb[b]))
                    )
                    dcdv_im[b, r] += np.sum(
                        diff[:, i, r] * np.imag(1j * v[b, i].conj() / (z - eb[b]))
                    )
            # Add contribution from case with i=j=r
            # Loop over all bath states
            for b in range(n_b):
                # Sum over energies
                dcdv_re[b, r] += np.sum(
                    diff[:, r, r] * np.imag(2 * v[b, r].real / (z - eb[b]))
                )
                dcdv_im[b, r] += np.sum(
                    diff[:, r, r] * np.imag(2 * v[b, r].imag / (z - eb[b]))
                )

    # Divide with normalization factor
    dcdv_re /= m
    dcdv_im /= m
    # Complex-valued matrix.
    dcdv = dcdv_re + 1j * dcdv_im
    # Convert to real-valued vector.
    dcdp = inroll(dcdv)
    # Add regularization contribution to the gradient.
    if regularization_mode == "L1":
        # L1-regularization
        dcdp += gamma / len(p) * np.sign(p)
    elif regularization_mode == "L2":
        # L2-regularization
        dcdp += gamma / len(p) * p
    elif regularization_mode == "sigmoid":
        dcdp += gamma / len(p) * np.tanh(p / delta)
    else:
        raise RuntimeError(
            f"Regularization mode {regularization_mode} not implemented."
        )

    if output == "gradient":
        return dcdp
    elif output == "value and gradient":
        return c, dcdp
    else:
        raise RuntimeError("Output option {output} not possible.")
    return None


def merge_ebs(ebs):
    """
    Returns the bath energies, merged into a 1d array.

    Parameters
    ----------
    ebs : tuple(K)
        Bath energies, for each energy window.
        Each element contains an array of bath states.

    Returns
    -------
    eb : array
        All the bath states as a one dimensional array.

    """
    eb = np.hstack(ebs)
    # eb = ebs.flatten()
    return eb


def merge_vs(vs):
    """
    Returns the hopping parameters, merged into a 2d array.

    Parameters
    ----------
    vs : tuple(K)
        Hopping parameters, for each energy window.
        Each element contains an array(B,N) of hopping parameters.

    Returns
    -------
    v : array
        All the hopping parameters as a two dimensional array(Btot, N),
        where Btot is the number of all the bath states.
    """
    # v = vs.reshape(vs.shape[0]*vs.shape[1], vs.shape[-1])
    v = np.vstack(vs)
    return v


def get_v_and_eb(
    w,
    delta,
    hyb,
    eb,
    eb_bounds,
    gamma,
    imag_only,
    realvalue_v,
    scale_function,
    v_guess=None,
):
    n_imp = np.shape(hyb)[1]
    n_b = len(eb) * n_imp
    z = w + 1j * delta
    if v_guess is None:
        v0 = get_p0(
            z,
            hyb,
            np.repeat(eb, n_imp),
            gamma,
            imag_only,
            realvalue_v,
        )
    else:
        v0 = inroll(v_guess)

    def fun(p):
        return cost_function(
            p,
            len(eb),
            z,
            hyb,
            gamma,
            imag_only,
            "value",
            "L2",
            3,
        )

    bounds = [
        eb_bounds[i] if i < len(eb) else (None, None) for i in range(len(eb) + len(v0))
    ]

    res = minimize(fun, np.append(eb, v0), bounds=bounds, tol=1e-6)

    p = res.x
    # merge_duplicate_bath_states(p[: len(eb)], p[len(eb) :], n_imp, delta)
    c = res.fun
    return (
        unroll(p[len(eb) :], n_b, n_imp),
        np.repeat(p[: len(eb)], n_imp),
        c,
    )


def merge_overlapping_bath_states(ebs, vs, delta):
    n_imp = vs.shape[2]
    sorted_idx = np.argsort(ebs)
    ebs = ebs[sorted_idx]
    vs = vs[sorted_idx]
    e_diff = np.diff(ebs)
    # delta is the HWHM of the Lorentzian broadening.  Two bath states whose
    # separation is less than delta have overlapping Lorentzians with no local
    # minimum between them and are indistinguishable by the fit.
    split_indices = 1 + np.nonzero(e_diff > delta)[0]
    A_merged = np.empty((0, n_imp, n_imp), dtype=vs.dtype, order="F")
    eb_merged = np.empty((0), dtype=float, order="F")
    for v_g, eb_g in zip(np.split(vs, split_indices), np.split(ebs, split_indices)):
        if v_g.shape[0] == 1:
            Am = (np.conj(np.transpose(v_g, (0, -1, -2))) @ v_g).reshape(
                (1, n_imp, n_imp), order="F"
            )
            em = eb_g
        else:
            em, Am = merge_bath_states(eb_g, v_g)
        eb_merged = np.append(eb_merged, em, axis=0)
        A_merged = np.append(A_merged, Am, axis=0)
    lam, U = np.linalg.eigh(A_merged)
    lam = np.clip(lam, 0.0, None)
    # V such that V^H V = A_merged, safe for rank-deficient A.
    return eb_merged, np.sqrt(lam)[:, :, None] * np.conj(np.swapaxes(U, -1, -2))


def inroll_a(p, n_eb, n_imp):
    n_elem = n_imp * (n_imp + 1) // 2
    pop_size = p.shape[0]
    real_val = p.shape[1] == n_eb * n_elem
    A_flat = p
    A = np.zeros((pop_size, n_eb, n_imp, n_imp), dtype=complex)
    upper_indices = np.triu_indices(n_imp)
    lower_indices = np.tril_indices(n_imp, k=0)
    for i_eb in range(n_eb):
        offset = i_eb * n_elem
        A_indices = np.ix_(range(pop_size), [i_eb], upper_indices[0], upper_indices[1])
        A[A_indices] = A_flat[:, offset : offset + n_elem]
        if not real_val:
            A[A_indices] += (
                1j
                * A_flat[
                    :,
                    n_eb * n_elem + offset : n_eb * n_elem + offset + n_elem,
                ]
            )
        A_lower_indices = np.ix_(
            range(pop_size), [i_eb], lower_indices[0], lower_indices[1]
        )
        A[A_lower_indices] = np.conj(A[A_indices])
    return A


def unroll_a(A):
    pop_size, n_eb, n_imp, _ = A.shape
    n_elem = n_imp * (n_imp + 1) // 2
    upper_indices = np.triu_indices(n_imp)
    A_flat = A[:, :, upper_indices[0], upper_indices[1]].real.reshape(
        (pop_size, n_eb * n_elem)
    )
    if np.max(np.abs(A.imag)) > 0:
        A_flat = np.append(
            A_flat,
            A[:, :, upper_indices[0], upper_indices[1]].imag.reshape(
                (pop_size, n_eb * n_elem)
            ),
            axis=1,
        )

    return A_flat


def moment_weights(w, max_moment):
    r"""
    Weights for the scaled spectral moments used in the cost function.

    Returns W_mn such that
        (W_mn.T @ f)[n] == (1/M) * sum_m (w[m] / w_scale)^n * f[m]
    i.e. the sample mean of (w/w_scale)^n * f.  Dividing by M keeps the
    moment contribution O(1) regardless of the mesh size, matching the
    scale of the old Simpson-integral formulation.  w_scale = max(|w|)
    keeps the normalised frequency in [-1, 1] so high-order terms don't
    blow up.

    Pre-computing W_mn outside the optimisation loop avoids repeated
    exponentiation inside the cost function.

    Parameters
    ----------
    w : array(M)
        Real frequency mesh.
    max_moment : int
        Number of moments (powers 0 .. max_moment - 1).

    Returns
    -------
    W_mn : array(M, max_moment)
    """
    w_scale = np.max(np.abs(w))
    return np.pow(w[:, None] / w_scale, np.arange(max_moment)[None, :]) / len(w)


def _varpro_inner_solve(eb, z, hyb, realvalue_v):
    """
    For fixed bath energies, find optimal PSD residues and constant shift via
    lstsq + projection.

    Solves hyb ≈ C + sum_k A_k / (z - eb[k]) simultaneously:
    - A_k are Hermitian PSD (residues) — obtained by eigh + clip
    - C is Hermitian (constant shift) — Hermitized but not PSD-constrained

    Returns (A_psd, V, G, C) where G[m, k] = 1 / (z[m] - eb[k]).
    """
    n_bath = len(eb)
    n_imp = hyb.shape[1]
    M = len(z)

    G = 1.0 / (z[:, None] - eb[None, :])  # (M, n_bath)
    # Augment with a column of ones to simultaneously solve for the constant C.
    G_aug = np.hstack([G, np.ones((M, 1))])  # (M, n_bath + 1)

    X, _, _, _ = np.linalg.lstsq(
        G_aug, hyb.reshape(M, n_imp * n_imp), rcond=None
    )  # (n_bath + 1, n_imp^2)

    A = X[:n_bath].reshape(n_bath, n_imp, n_imp)
    C_raw = X[n_bath].reshape(n_imp, n_imp)

    A = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))  # Hermitize residues
    C = 0.5 * (C_raw + np.conj(C_raw.T))  # Hermitize C (no PSD constraint)
    if realvalue_v:
        A = A.real
        C = C.real

    lam, U = np.linalg.eigh(A)
    lam = np.clip(lam.real, 0.0, None)
    A_psd = (U * lam[:, None, :]) @ np.conj(np.swapaxes(U, -1, -2))
    V = np.sqrt(lam)[:, :, None] * np.conj(
        np.swapaxes(U, -1, -2)
    )  # (n_bath, n_imp, n_imp)

    return A_psd, V, G, C


def _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, realvalue_v):
    """
    VARPRO cost and its gradient w.r.t. bath energies.

    For each eb proposal, solves for optimal PSD residues A and constant shift
    C via `_varpro_inner_solve`, evaluates the fit cost, and returns the partial
    gradient w.r.t. eb (treating A_psd and C as fixed — valid by the VARPRO
    theorem for unconstrained lstsq; approximate after PSD projection).

    Returns (cost, grad_eb, V, C).
    """
    A_psd, V, G, C = _varpro_inner_solve(eb, z, hyb, realvalue_v)

    max_moment = W_mn.shape[1]
    hyb_model = np.einsum("mk, kij -> mij", G, A_psd) + C[None]  # (M, n_imp, n_imp)
    diff = hyb - hyb_model

    w2 = weight_array**2
    N = diff.size
    c = np.sum(w2[:, None, None] * 0.5 * np.abs(diff) ** 2) / N

    moment_diff = np.einsum("mn, mij -> nij", W_mn, diff)  # (max_moment, n_imp, n_imp)
    P = moment_diff[0].size * max_moment
    c += np.sum(0.5 * np.abs(moment_diff) ** 2) / P

    # Partial VARPRO gradient w.r.t. eb: C is treated as fixed, so only the
    # A_psd / (z - eb)^2 term contributes (C has no eb dependence at fixed A).
    dGdeb = G**2  # (M, n_bath)
    conj_diff_A = np.einsum("mij, kij -> mk", np.conj(diff), A_psd)  # (M, n_bath)
    grad = -np.real(np.einsum("m, mk, mk -> k", w2, dGdeb, conj_diff_A)) / N

    WdG = np.einsum("mn, mk -> kn", W_mn, dGdeb)  # (n_bath, max_moment)
    conj_mdf_A = np.einsum(
        "nij, kij -> kn", np.conj(moment_diff), A_psd
    )  # (n_bath, max_moment)
    grad -= np.real(np.einsum("kn, kn -> k", WdG, conj_mdf_A)) / P

    return c, grad, V, C


def get_v_and_eb_varpro_basin_hopping(
    w,
    delta,
    hyb,
    ebs,
    eb_restrictions,
    gamma,
    regularization,
    weight_function,
    realvalue_v,
    max_moment=3,
):
    """
    VARPRO basin-hopping: optimise only over bath energies, with residues
    solved analytically via lstsq + PSD projection at each step.

    The search space shrinks from n_bath*(1 + n_imp^2) to n_bath, with each
    evaluation costing one lstsq + eigh solve.  After basin-hopping, a final
    SLSQP polish refines both energies and hoppings jointly with the analytic
    Jacobian.
    """
    n_imp = hyb.shape[1]

    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr
    weight_array = weight_function(w)
    W_mn = moment_weights(w, max_moment)

    initial_costs = np.array(
        [
            _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, realvalue_v)[0]
            for eb in ebs
        ]
    )
    mean_cost = np.mean(initial_costs)
    stddev_cost = np.std(initial_costs)
    T = max(stddev_cost, 1e-3 * abs(mean_cost))

    x0 = ebs[np.argmin(initial_costs)]

    def _fg(eb):
        c, g, _, _ = _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, realvalue_v)
        return float(c), g

    res = basinhopping(
        _fg,
        x0,
        niter=150,
        T=T,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "jac": True,
            "tol": 1e-6,
            "options": {"maxiter": 500},
            "bounds": eb_restrictions,
        },
        disp=False,
    )

    # Extract hoppings and C for the converged bath energies, then merge.
    _, V_opt, _, C_opt = _varpro_inner_solve(res.x, z, hyb, realvalue_v)
    eb_merged, v_merged = merge_overlapping_bath_states(res.x, V_opt, delta)

    # Final SLSQP polish over eb, V, and C jointly with the analytic Jacobian.
    n_eb_merged = len(eb_merged)
    p_C0 = inroll_C(C_opt)
    n_C = len(p_C0)
    p0 = np.concatenate([eb_merged, inroll(v_merged), p_C0])
    bounds_polish = (
        eb_restrictions[:n_eb_merged] + [(None, None)] * (len(p0) - n_eb_merged)
        if eb_restrictions is not None
        else None
    )
    res_polish = minimize(
        vectorized_cost_function,
        p0,
        method="SLSQP",
        jac=vectorized_jacobian,
        tol=1e-8,
        options={"maxiter": 1000},
        args=(n_eb_merged, z, hyb, gamma, regularization, weight_array, W_mn, n_C),
        bounds=bounds_polish,
    )

    p = res_polish.x
    eb_final = p[:n_eb_merged]
    v_final = unroll(p[n_eb_merged:-n_C], n_eb_merged, n_imp)
    C_final = unroll_C(p[-n_C:], n_imp)
    c_final = float(
        vectorized_cost_function(
            p, n_eb_merged, z, hyb, gamma, None, weight_array, W_mn, n_C
        )
    )

    return v_final, eb_final, C_final, c_final


def get_v_and_eb_multiple_optimizations(
    w, delta, hyb, ebs, vs, gamma, regularization, weight_function=None, max_moment=3
):
    assert (
        vs.shape[0] == ebs.shape[0]
    ), f"population size must match between eb and v. {ebs.shape[0]} != {vs.shape[0]}"
    population_size = ebs.shape[0]
    n_imp = np.shape(hyb)[1]
    n_eb = ebs.shape[1]
    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr

    weight_array = (
        weight_function(w) if weight_function is not None else np.ones_like(w)
    )
    W_mn = moment_weights(w, max_moment)

    v0_flat = inroll(vs)
    initial_guesses = np.append(np.moveaxis(ebs, 0, -1), v0_flat, axis=0)
    initial_costs = vectorized_cost_function(
        initial_guesses, n_eb, z, hyb, gamma, regularization, weight_array, W_mn
    )
    mean_cost = np.mean(initial_costs)
    stddev_cost = np.std(initial_costs, mean=mean_cost)
    best_cost = np.inf
    best_v = None
    best_eb = None
    sort_indices = np.argsort(initial_costs)
    initial_guesses = initial_guesses[:, sort_indices]
    for column in range(min(10, population_size)):
        res = minimize(
            vectorized_cost_function,
            initial_guesses[:, column],
            tol=1e-3,
            method="SLSQP",
            jac=vectorized_jacobian,
            args=(n_eb, z, hyb, gamma, regularization, weight_array, W_mn),
        )

        p = res.x
        eb_merged, v_merged = merge_overlapping_bath_states(
            p[:n_eb], unroll(p[n_eb:], n_eb, n_imp), delta
        )
        c = vectorized_cost_function(
            np.append(eb_merged, inroll(v_merged)),
            eb_merged.shape[0],
            z,
            hyb,
            gamma,
            regularization,
            weight_array,
            W_mn,
        )
        if abs(c) < best_cost:
            best_v = v_merged
            best_eb = eb_merged
            best_cost = c
    return (
        best_v,
        best_eb,
        best_cost,
    )


def get_v_and_eb_basin_hopping(
    w,
    delta,
    hyb,
    ebs,
    eb_restrictions,
    vs,
    gamma,
    regularization,
    weight_function,
    realvalue_v=True,
    max_moment=3,
):
    assert (
        vs.shape[0] == ebs.shape[0]
    ), f"population size must match between eb and v. {ebs.shape[0]} != {vs.shape[0]}"
    population_size = ebs.shape[0]
    n_imp = np.shape(hyb)[1]
    n_eb = ebs.shape[1]
    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr

    weight_array = weight_function(w)
    W_mn = moment_weights(w, max_moment)

    # Seed C for each population member via VARPRO inner solve at the given eb.
    C_seeds = [_varpro_inner_solve(eb, z, hyb, realvalue_v)[3] for eb in ebs]
    C_flat_all = np.array([inroll_C(C) for C in C_seeds]).T  # (n_C, population_size)
    n_C = C_flat_all.shape[0]

    v0_flat = inroll(vs)
    initial_guesses = np.vstack([np.moveaxis(ebs, 0, -1), v0_flat, C_flat_all])
    initial_costs = vectorized_cost_function(
        initial_guesses, n_eb, z, hyb, gamma, regularization, weight_array, W_mn, n_C
    )
    mean_cost = np.mean(initial_costs)
    stddev_cost = np.std(initial_costs, mean=mean_cost)
    T = max(stddev_cost, 1e-3 * abs(mean_cost))
    best_cost = np.inf
    best_v = None
    best_eb = None
    best_C = None
    sort_indices = np.argsort(initial_costs)
    initial_guesses = initial_guesses[:, sort_indices]
    for column in range(min(1, population_size)):
        guess = initial_guesses[:, column]

        res = basinhopping(
            vectorized_cost_function,
            guess,
            niter=150,
            T=T,
            minimizer_kwargs={
                "tol": 1e-6,
                "jac": vectorized_jacobian,
                "method": "SLSQP",
                "options": {"maxiter": 500},
                "args": (n_eb, z, hyb, gamma, regularization, weight_array, W_mn, n_C),
                "bounds": (
                    eb_restrictions + [(None, None)] * (guess.shape[0] - n_eb)
                    if eb_restrictions is not None
                    else None
                ),
            },
            disp=True,
        )

        p = res.x
        eb_merged, v_merged = merge_overlapping_bath_states(
            p[:n_eb], unroll(p[n_eb:-n_C], n_eb, n_imp), delta
        )
        C_result = unroll_C(p[-n_C:], n_imp)
        c = vectorized_cost_function(
            np.concatenate([eb_merged, inroll(v_merged), inroll_C(C_result)]),
            eb_merged.shape[0],
            z,
            hyb,
            gamma,
            None,
            weight_array,
            W_mn,
            n_C,
        )
        if abs(c) < best_cost:
            best_v = v_merged
            best_eb = eb_merged
            best_C = C_result
            best_cost = c
    return best_v, best_eb, best_C, best_cost


def get_v_and_eb_differential_evolution(
    w,
    delta,
    hyb,
    ebs,
    eb_restrictions,
    gamma,
    regularization,
    weight_function,
    realvalue_v=True,
    max_moment=3,
):
    """
    VARPRO differential evolution: optimise only over bath energies, with
    residues and constant shift C solved analytically at each evaluation.

    After DE convergence a final SLSQP polish refines eb, V, and C jointly.
    """
    n_imp = hyb.shape[1]
    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr
    weight_array = weight_function(w)
    W_mn = moment_weights(w, max_moment)

    def varpro_cost(eb):
        c, _, _, _ = _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, realvalue_v)
        return float(c)

    res = differential_evolution(
        varpro_cost,
        Bounds(
            lb=[e_r[0] for e_r in eb_restrictions],
            ub=[e_r[1] for e_r in eb_restrictions],
        ),
        init=ebs,
        atol=1e-6,
        maxiter=10000,
        polish=False,
    )

    # Extract hoppings and C for the converged bath energies, then merge.
    _, V_opt, _, C_opt = _varpro_inner_solve(res.x, z, hyb, realvalue_v)
    eb_merged, v_merged = merge_overlapping_bath_states(res.x, V_opt, delta)

    # Final SLSQP polish over eb, V, and C jointly with the analytic Jacobian.
    n_eb_merged = len(eb_merged)
    p_C0 = inroll_C(C_opt)
    n_C = len(p_C0)
    p0 = np.concatenate([eb_merged, inroll(v_merged), p_C0])
    bounds_polish = (
        eb_restrictions[:n_eb_merged] + [(None, None)] * (len(p0) - n_eb_merged)
        if eb_restrictions is not None
        else None
    )
    res_polish = minimize(
        vectorized_cost_function,
        p0,
        method="SLSQP",
        jac=vectorized_jacobian,
        tol=1e-8,
        options={"maxiter": 1000},
        args=(n_eb_merged, z, hyb, gamma, regularization, weight_array, W_mn, n_C),
        bounds=bounds_polish,
    )

    p = res_polish.x
    eb_final = p[:n_eb_merged]
    v_final = unroll(p[n_eb_merged:-n_C], n_eb_merged, n_imp)
    C_final = unroll_C(p[-n_C:], n_imp)
    c_final = float(
        vectorized_cost_function(
            p, n_eb_merged, z, hyb, gamma, None, weight_array, W_mn, n_C
        )
    )
    return v_final, eb_final, C_final, c_final


def calc_diff(eb, v, z, hyb, C=None):
    hyb_model = get_hyb_2(z, eb, v, C=C)
    return hyb[np.newaxis] - hyb_model


def calc_moment_diff(diff, W_mn):
    r"""
    Integrate the difference between fitted and exact hybridization function
    (calculate the difference between the integrals).
    Prameters:
    ==========
    diff: np.ndarray((..., n_w, n_imp, n_imp)) - $$\Delta(\omega) - \tilde{\Delta}(\omega)$$
    W_mn: np.ndarray((n_w, max_moment)) - Integration weights for moments
    Returns:
    ========
    M: np.ndarray((..., max_moment, n_imp, n_imp)) - The integrals $$\int \omega^n (\Delta(\omega) - \tilde{\Delta}(\omega)) \mathrm{d}\omega$$
    """
    return np.einsum("mn, ...mij -> ...nij", W_mn, diff)


def _unroll_C_batch(p_C, n_imp):
    """Vectorised unroll_C: p_C (n_C, S) → C_arr (S, n_imp, n_imp)."""
    triu_i, triu_j = np.triu_indices(n_imp)
    n_triu = len(triu_i)
    off_mask = triu_i != triu_j
    S = p_C.shape[1]
    if p_C.shape[0] == n_triu:  # real symmetric
        C_arr = np.zeros((S, n_imp, n_imp))
        C_arr[:, triu_i, triu_j] = p_C.T
        C_arr[:, triu_j, triu_i] = p_C.T
    else:  # complex Hermitian
        off_i, off_j = triu_i[off_mask], triu_j[off_mask]
        C_arr = np.zeros((S, n_imp, n_imp), dtype=complex)
        entries = p_C[:n_triu].T.astype(complex)
        entries[:, off_mask] += 1j * p_C[n_triu:].T
        C_arr[:, triu_i, triu_j] = entries
        C_arr[:, off_j, off_i] = np.conj(entries[:, off_mask])
    return C_arr


def vectorized_cost_function(
    p,
    n_eb,
    z,
    hyb,
    gamma,
    regularization="L1",
    weight_array=None,
    W_mn=None,
    n_C=0,
):
    one_dim = len(p.shape) == 1
    if one_dim:
        p_batched = p[:, None]
    else:
        p_batched = p
    w = z.real
    n_w = len(z)
    n_imp = hyb.shape[1]
    eb = np.moveaxis(p_batched[:n_eb], 0, -1)

    n_v_end = p_batched.shape[0] - n_C if n_C else p_batched.shape[0]
    p_v = p_batched[n_eb:n_v_end]
    v = unroll(p_v, n_eb, n_imp)

    # Build C: (n_imp, n_imp) for one_dim, (S, n_imp, n_imp) for batched.
    C = None
    if n_C:
        C_arr = _unroll_C_batch(p_batched[-n_C:], n_imp)  # (S, n_imp, n_imp)
        C = C_arr[0] if one_dim else C_arr

    diff = hyb[np.newaxis] - get_hyb_2(z, eb, v, C=C)  # (S, M, N, N)

    if weight_array is None:
        weight_array = np.ones_like(w)

    c = (1 / (n_w * n_imp * n_imp)) * np.sum(
        0.5 * weight_array[None, :, None, None] * np.abs(diff) ** 2, axis=(1, 2, 3)
    )

    if W_mn is not None:
        moment_diff = np.einsum("mn, ...mij -> ...nij", W_mn, diff)
        c += (1 / (n_imp * n_imp * moment_diff.shape[1])) * np.sum(
            0.5 * np.abs(moment_diff) ** 2, axis=(1, 2, 3)
        )

    # Regularization applies only to V parameters, not eb or C.
    n_v = p_v.shape[0]
    if regularization is None or regularization.lower() == "none":
        pass
    elif regularization.lower() == "l1":
        c += (gamma / n_v) * np.sum(np.abs(p_v), axis=0)
    elif regularization.lower() == "l2":
        c += (gamma / n_v) * np.sum(p_v**2, axis=0)
    else:
        raise RuntimeError(f"Unknown regularization mode {regularization}")

    return c[0].item() if one_dim else c


def vectorized_jacobian(
    p,
    n_eb,
    z,
    hyb,
    gamma,
    regularization="L1",
    weight_array=None,
    W_mn=None,
    n_C=0,
):
    one_dim = len(p.shape) == 1
    if one_dim:
        p = p[:, None]
    J = np.zeros_like(p)
    popsize = p.shape[1]
    n_w = len(z)
    n_imp = hyb.shape[1]
    eb = np.moveaxis(p[:n_eb], 0, -1)

    if weight_array is None:
        weight_array = np.ones_like(z.real)

    triu_rows, triu_cols = np.triu_indices(n_imp)
    n_v_end = p.shape[0] - n_C if n_C else p.shape[0]
    p_v = p[n_eb:n_v_end]
    realvalued = p_v.shape[0] == n_eb * len(triu_cols)

    v = unroll(p_v, n_eb, n_imp)  # (S, n_eb, n_imp, n_imp)

    C = None
    if n_C:
        C_arr = _unroll_C_batch(p[-n_C:], n_imp)  # (S, n_imp, n_imp)
        C = C_arr[0] if one_dim else C_arr

    diff = hyb[np.newaxis] - get_hyb_2(z, eb, v, C=C)  # (S, M, N, N)

    G = 1.0 / (z[np.newaxis, :, np.newaxis] - eb[:, np.newaxis, :])  # (S, M, n_eb)
    A = np.conj(np.transpose(v, (0, 1, 3, 2))) @ v  # (S, n_eb, N, N)
    diff_W = diff * weight_array[None, :, None, None]

    dhyb_deb = (
        A[:, np.newaxis, :, :, :] * (G**2)[:, :, :, np.newaxis, np.newaxis]
    )  # (S, M, n_eb, N, N)
    J_eb = -np.einsum("smxy, smbxy -> sb", np.conj(diff_W), dhyb_deb).real
    J[:n_eb, :] = np.moveaxis(J_eb, 0, -1) / (n_w * n_imp * n_imp)

    if W_mn is not None:
        moment_diff = np.einsum("mn, ...mij -> ...nij", W_mn, diff)
        dmoment_deb = -np.einsum("mn, smbxy -> snbxy", W_mn, dhyb_deb)
        J_moment_eb = np.einsum(
            "snxy, snbxy -> sb", np.conj(moment_diff), dmoment_deb
        ).real
        J[:n_eb, :] += np.moveaxis(J_moment_eb, 0, -1) / (
            n_imp * n_imp * moment_diff.shape[1]
        )

    # --- V gradient ---
    S_term = -np.einsum("smxy, smb -> sbxy", np.conj(diff_W), G)
    if W_mn is not None:
        WG = np.einsum("mn, smb -> snb", W_mn, G)
        S_mom = -np.einsum("snxy, snb -> sbxy", np.conj(moment_diff), WG)
        S_total = S_term / (n_w * n_imp * n_imp) + S_mom / (
            n_imp * n_imp * moment_diff.shape[1]
        )
    else:
        S_total = S_term / (n_w * n_imp * n_imp)

    J_R = np.zeros((popsize, n_eb, n_imp, n_imp), dtype=float)
    J_I = np.zeros((popsize, n_eb, n_imp, n_imp), dtype=float)
    for m in range(n_imp):
        for n in range(n_imp):
            if m > n:
                continue
            term_R = np.sum(
                S_total[:, :, n, :] * v[:, :, m, :]
                + S_total[:, :, :, n] * np.conj(v[:, :, m, :]),
                axis=-1,
            )
            J_R[:, :, m, n] = np.real(term_R)
            if not realvalued:
                term_I = np.sum(
                    S_total[:, :, n, :] * (-1j * v[:, :, m, :])
                    + S_total[:, :, :, n] * (1j * np.conj(v[:, :, m, :])),
                    axis=-1,
                )
                J_I[:, :, m, n] = np.real(term_I)

    J_R_flat = J_R[:, :, triu_rows, triu_cols].reshape((popsize, -1), order="C")
    n_real = n_eb * len(triu_cols)
    J[n_eb : n_eb + n_real, :] = np.moveaxis(J_R_flat, 0, -1)
    if not realvalued:
        J_I_flat = J_I[:, :, triu_rows, triu_cols].reshape((popsize, -1), order="C")
        J[n_eb + n_real : n_v_end, :] = np.moveaxis(J_I_flat, 0, -1)

    # Regularization on V params only.
    n_v = p_v.shape[0]
    if regularization is None or regularization.lower() == "none":
        pass
    elif regularization.lower() == "l1":
        J[n_eb:n_v_end] += (gamma / n_v) * np.sign(p_v)
    elif regularization.lower() == "l2":
        J[n_eb:n_v_end] += (gamma / n_v) * 2 * p_v
    else:
        raise RuntimeError(f"Unknown regularization mode {regularization}")

    # --- C gradient ---
    if n_C:
        # For a Hermitian C, the independent parameters are:
        #   Re(C[p,q]) for all upper-tri (p,q) pairs, and Im(C[p,q]) for off-diagonal.
        # Each parameter affects both (p,q) and (q,p) elements of the model, so both
        # rows/columns of diff contribute.  We sum both contributions explicitly rather
        # than assuming diff is Hermitian (which fails when hyb is non-Hermitian).
        off_mask = triu_rows != triu_cols
        off_i, off_j = triu_rows[off_mask], triu_cols[off_mask]
        n_triu = len(triu_rows)

        weighted_diff = np.einsum(
            "m, smij -> sij", weight_array, diff
        )  # (S, n_imp, n_imp)
        N = n_w * n_imp * n_imp

        # For triu (p,q): upper[k] = weighted_diff[p,q], lower[k] = weighted_diff[q,p].
        # d(cost)/d(Re(C[p,q])) = -(1/N)*Re(upper + I(p!=q)*lower)
        upper = weighted_diff[:, triu_rows, triu_cols]  # (S, n_triu)
        lower = weighted_diff[
            :, triu_cols, triu_rows
        ]  # (S, n_triu) — transposed indices
        sum_RL = upper.copy()
        sum_RL[:, off_mask] += lower[:, off_mask]
        J_C_real = -(1.0 / N) * np.real(np.moveaxis(sum_RL, 0, -1))  # (n_triu, S)

        if W_mn is not None:
            P = n_imp * n_imp * moment_diff.shape[1]
            W_sum = W_mn.sum(axis=0)  # (max_moment,)
            weighted_mdiff = np.einsum(
                "n, snij -> sij", W_sum, moment_diff
            )  # (S, n_imp, n_imp)
            mupper = weighted_mdiff[:, triu_rows, triu_cols]
            mlower = weighted_mdiff[:, triu_cols, triu_rows]
            sum_mRL = mupper.copy()
            sum_mRL[:, off_mask] += mlower[:, off_mask]
            J_C_real -= (1.0 / P) * np.real(np.moveaxis(sum_mRL, 0, -1))

        J[n_v_end : n_v_end + n_triu, :] = J_C_real

        if n_C > n_triu:  # complex Hermitian: imaginary gradient for off-diagonal
            # d(cost)/d(Im(C[p,q])) = -(1/N)*Im(upper - lower)  for p < q
            J_C_imag = -(1.0 / N) * np.imag(
                np.moveaxis(upper[:, off_mask] - lower[:, off_mask], 0, -1)
            )
            if W_mn is not None:
                J_C_imag -= (1.0 / P) * np.imag(
                    np.moveaxis(mupper[:, off_mask] - mlower[:, off_mask], 0, -1)
                )
            J[n_v_end + n_triu :, :] = J_C_imag

    return J[:, 0] if one_dim else J

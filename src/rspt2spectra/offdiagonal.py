#!/usr/bin/env python3

"""

offdiagonal
===========

This module contains functions to treat
off-diagonal hybridization functions.

"""

import itertools
from functools import partial
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
from importlib.metadata import version
import matplotlib.pyplot as plt

scipy_newer_than_3_16 = all(
    int(installed_version) >= int(v_3_16)
    for installed_version, v_3_16 in zip(version("scipy").split(".")[:2], ("3", "16"))
)


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

    np.multiply.outer(v.conj(), v)
    # Loop over all bath energies
    for b, e in enumerate(eb):
        # Add contributions from each bath
        hyb[:] += (
            np.outer(v[b].conj(), v[b])[np.newaxis, ...]
            * (1 / (z - e))[:, np.newaxis, np.newaxis]
        )

    return hyb


def get_hyb_2(z, eb, v):
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

    Returns
    -------
    hyb : array(S, M, N,N)
        Hybridization functions.

    """
    A = np.conj(np.transpose(v, (0, 1, 3, 2))) @ v  # (S, N_B, N, N)
    G = 1.0 / (z[np.newaxis, :, np.newaxis] - eb[:, np.newaxis, :])  # (S, M, N_B)
    return np.einsum("smb,sbij->smij", G, A)


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
    population_size = eb.shape[0]
    n_eb = eb.shape[1]
    n_orb = hyb.shape[1]
    v = np.zeros((population_size, n_eb, n_orb, n_orb), dtype=complex)
    x0 = rng.uniform(
        low=0.1,
        high=0.1,
        # size=(population_size, n_eb if realvalue_v else 2 * n_eb),
        size=(population_size, n_eb if realvalue_v else 2 * n_eb, n_orb, n_orb),
    )
    for p, i, j in itertools.product(
        range(population_size), range(n_orb), range(n_orb)
    ):
        if j < i:
            continue
        res = minimize(
            lambda x, *args: vectorized_cost_function(np.append(eb[p], x), *args),
            x0[p, :, i, j].reshape((-1,)),
            # jac=lambda x, *args: vectorized_jacobian(np.append(eb[p], x), *args)[n_eb:],
            method="SLSQP",
            args=(n_eb, z, hyb[np.ix_(np.arange(len(z)), [i], [j])], gamma, "L1"),
            tol=1e-3,
        )
        v[p, :, i, j] = res.x[:n_eb]
        if not realvalue_v:
            v[p, :, i, j] += 1j * res.x[n_eb:]

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
    V = sp.linalg.cho_factor(A, lower=False)
    Eb = sp.linalg.cho_solve(V, np.sum(first_moments, axis=0))
    eb = np.sum(np.linalg.eigvals(Eb).real) / Eb.size
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
    # Delta gives Half width at half maximum
    # If peaks are closer together than delta/5, combine them.
    # delta/5 is half width at ~96% of maximum. If peaks are too closely spaced,
    # they start overlapping way too much for the fitting procedure to make much sense.
    split_indices = 1 + np.nonzero(e_diff > delta / 5)[0]
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
    return eb_merged, np.linalg.cholesky(A_merged, upper=True)


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


def get_v_and_eb_multiple_optimizations(w, delta, hyb, ebs, vs, gamma, regularization):
    assert (
        vs.shape[0] == ebs.shape[0]
    ), f"population size must match between eb and v. {ebs.shape[0]} != {vs.shape[0]}"
    population_size = ebs.shape[0]
    n_imp = np.shape(hyb)[1]
    n_eb = ebs.shape[1]
    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr

    weight_array = np.ones_like(w)
    simpson_weights = sp.integrate.simpson(np.eye(len(w)), w, axis=0)
    max_moment = 10
    W_mn = (
        simpson_weights[:, None]
        * np.pow(w[:, None], np.arange(max_moment)[None, :])
        / (w[-1] - w[0])
    )

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
            # jac=vectorized_jacobian,
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
    simpson_weights = sp.integrate.simpson(np.eye(len(w)), w, axis=0)
    max_moment = 10
    W_mn = (
        simpson_weights[:, None]
        * np.pow(w[:, None], np.arange(max_moment)[None, :])
        / (w[-1] - w[0])
    )

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
    for column in range(min(1, population_size)):
        guess = initial_guesses[:, column]

        res = basinhopping(
            vectorized_cost_function,
            guess,
            niter=150,
            # niter_success=50,
            T=stddev_cost,
            minimizer_kwargs={
                "tol": 1e-6,
                # "jac": vectorized_jacobian,
                "method": "SLSQP",
                "options": {
                    "maxiter": 500,
                },
                "args": (n_eb, z, hyb, gamma, regularization, weight_array, W_mn),
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
            p[:n_eb], unroll(p[n_eb:], n_eb, n_imp), delta
        )
        c = vectorized_cost_function(
            np.append(eb_merged, inroll(v_merged)),
            eb_merged.shape[0],
            z,
            hyb,
            gamma,
            None,
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


def get_v_and_eb_differential_evolution(
    w, delta, hyb, ebs, eb_restrictions, vs, gamma, regularization, weight_function
):
    n_eb = ebs.shape[1]
    n_imp = hyb.shape[1]
    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr

    weight_array = weight_function(w)
    simpson_weights = sp.integrate.simpson(np.eye(len(w)), w, axis=0)
    max_moment = 10
    W_mn = (
        simpson_weights[:, None]
        * np.pow(w[:, None], np.arange(max_moment)[None, :])
        / (w[-1] - w[0])
    )

    v0_flat = inroll(vs)
    initial_guesses = np.append(np.moveaxis(ebs, 0, -1), v0_flat, axis=0)

    polish_func = partial(
        minimize,
        args=(n_eb, z, hyb, gamma, regularization, weight_array, W_mn),
        # jac=vectorized_jacobian,
        method="SLSQP",
    )

    res = differential_evolution(
        vectorized_cost_function,
        Bounds(
            lb=[e_r[0] for e_r in eb_restrictions] + [-10] * v0_flat.shape[0],
            ub=[e_r[1] for e_r in eb_restrictions] + [10] * v0_flat.shape[0],
        ),
        atol=1e-6,
        args=(n_eb, z, hyb, gamma, regularization, weight_array, W_mn),
        init=initial_guesses.T,
        maxiter=10000,
        vectorized=True,
        updating="deferred",
        polish=polish_func,
        # disp=True,
        # mutation=(0.75, 1.5),
        # recombination=0.5,
    )

    p = res.x
    eb_merged, v_merged = merge_overlapping_bath_states(
        p[:n_eb], unroll(p[n_eb:], n_eb, n_imp), delta
    )
    c = vectorized_cost_function(
        np.append(eb_merged[:, None], inroll(v_merged)[:, None]),
        eb_merged.shape[0],
        z,
        hyb,
        gamma,
        regularization,
        weight_array,
        W_mn,
    )
    return (
        v_merged,
        eb_merged,
        c,
    )


def calc_diff(eb, v, z, hyb):
    hyb_model = get_hyb_2(z, eb, v)

    # Difference between original and model hybridization functions
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


def vectorized_cost_function(
    p,
    n_eb,
    z,
    hyb,
    gamma,
    regularization="L1",
    weight_array=None,
    W_mn=None,
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

    v = unroll(p_batched[n_eb:], n_eb, n_imp)
    diff = hyb[np.newaxis] - get_hyb_2(z, eb, v)  # (S, M, N, N)

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

    if regularization is None or regularization.lower() == "none":
        pass
    elif regularization.lower() == "l1":
        c += (gamma / p_batched.shape[0]) * np.sum(np.abs(p_batched), axis=0)
    elif regularization.lower() == "l2":
        c += (gamma / p_batched.shape[0]) * np.sum(p_batched**2, axis=0)
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
    realvalued = p.shape[0] - n_eb == n_eb * len(triu_cols)

    v = unroll(p[n_eb:], n_eb, n_imp)  # (S, n_eb, n_imp, n_imp)

    diff = hyb[np.newaxis] - get_hyb_2(z, eb, v)  # (S, M, N, N)

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

    S = -np.einsum("smxy, smb -> sbxy", np.conj(diff_W), G)

    if W_mn is not None:
        WG = np.einsum("mn, smb -> snb", W_mn, G)
        S_mom = -np.einsum("snxy, snb -> sbxy", np.conj(moment_diff), WG)
        S_total = S / (n_w * n_imp * n_imp) + S_mom / (
            n_imp * n_imp * moment_diff.shape[1]
        )
    else:
        S_total = S / (n_w * n_imp * n_imp)

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
        J[n_eb + n_real :, :] = np.moveaxis(J_I_flat, 0, -1)

    if regularization is None or regularization.lower() == "none":
        pass
    elif regularization.lower() == "l1":
        J += (gamma / p.shape[0]) * np.sign(p)
    elif regularization.lower() == "l2":
        J += (gamma / p.shape[0]) * 2 * p
    else:
        raise RuntimeError(f"Unknown regularization mode {regularization}")

    return J[:, 0] if one_dim else J

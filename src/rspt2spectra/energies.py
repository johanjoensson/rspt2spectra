#!/usr/bin/env python3

"""

energies
========

This module contains functions which are useful to
process on-site energy data generated by the RSPt software.

"""

import numpy as np
import subprocess
import sys
from math import pi
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pylab as plt

from . import hybridization
from .constants import eV


def get_h0(e_onsite, eb, vb, spinpol):
    """
    Return non-interacting Hamiltonian matrix.
    """
    no = len(e_onsite)
    nc, nb = np.shape(eb)
    norb = nc // 2 if spinpol else nc
    # Initialize the full Hamiltonian, including spin
    h = np.zeros(2 * norb * (1 + nb) * np.array([1, 1]), dtype=complex)

    # On-site energies of correlated orbitals
    if no == 2 * norb:
        for i in range(no):
            h[i, i] = e_onsite[i]
    elif no == norb:
        for i in range(no):
            h[i, i] = e_onsite[i]
            h[no + i, no + i] = e_onsite[i]

    # Bath energies
    if spinpol:
        for j in range(nb):
            for i in range(nc):
                k = nc + i + nc * j
                h[k, k] = eb[i, j]
    else:
        for j in range(nb):
            for i in range(nc):
                k = 2 * nc + i + 2 * nc * j
                h[k, k] = eb[i, j]
                h[k + no, k + no] = eb[i, j]

    # Hybridization hoppings
    if spinpol:
        for j in range(nb):
            for i in range(nc):
                k = nc + i + nc * j
                h[k, i] = vb[i, j]
                h[i, k] = np.conj(vb[i, j])
    else:
        for j in range(nb):
            for i in range(nc):
                k = 2 * nc + i + 2 * nc * j
                h[k, i] = vb[i, j]
                h[i, k] = np.conj(vb[i, j])
                h[k + no, i + no] = vb[i, j]
                h[i + no, k + no] = np.conj(vb[i, j])

    # Make sure Hamiltonian is hermitian
    assert np.all(h == np.conj(h.T))
    return h


def integrate(x, y, xmin=None, xmax=None):
    """
    Return numerical integration value.

    """
    if (xmin is not None) or (xmax is not None):
        if xmin is None:
            mask = x < xmax
        elif xmax is None:
            mask = xmin < x
        else:
            mask = (xmin < x) & (x < xmax)
        x = x[mask]
        y = y[mask]
    return np.trapz(y, x)


def cog(x, y, xmin=None, xmax=None):
    """
    Return the center of gravity, within the [xmin,xmax] window.

    Parameters
    ----------
    x : (N) array
    y : (N) array
    xmin : float
        Lower limit of window.
    xmax : float
        Upper limit of window.
    """
    return integrate(x, y * x, xmin, xmax) / integrate(x, y, xmin, xmax)


def plot_pdos0_2(w, pdos1, pdos2, nc, spinpol, xlim):
    """
    Plot non-interacting PDOS.
    """
    plt.figure()
    for i in range(nc):
        plt.plot(w, pdos1[i, :], c="C" + str(i), label=str(i))
        plt.plot(w, pdos2[i, :], "--", c="C" + str(i))
    plt.legend()
    plt.ylabel("PDOS$_0$")
    plt.xlim(xlim)
    plt.title("Orbital resolved PDOS$_0$. -:RSPt, --:discrete with e_rspt")
    plt.show()
    if spinpol:
        # Plot non-interacting PDOS
        # Down spin with negative sign
        plt.figure()
        for i in range(nc // 2):
            # Down spin
            plt.plot(w, -pdos1[i, :], c="C" + str(i), label=str(i))
            plt.plot(w, -pdos2[i, :], "--", c="C" + str(i))
            # Up spin
            plt.plot(w, pdos1[nc // 2 + i, :], c="C" + str(i))
            plt.plot(w, pdos2[nc // 2 + i, :], "--", c="C" + str(i))
        plt.legend()
        plt.ylabel("PDOS$_0$")
        plt.xlim(xlim)
        plt.title("Orbital resolved PDOS$_0$. -:RSPt, --:discrete with e_rspt")
        plt.show()


def plot_pdos0_3(w, p0d_rspt, p0d_initial, p0d, nc, spinpol, xlim):
    """
    Plot non-interacting PDOS.
    """
    norb = nc // 2 if spinpol else nc
    # Trace
    plt.figure()
    plt.plot(w, np.sum(p0d_rspt, axis=0), label="p0d_rspt")
    plt.plot(w, np.sum(p0d_initial, axis=0), label="p0d_initial")
    plt.plot(w, np.sum(p0d, axis=0), label="p0d")
    plt.legend()
    plt.ylabel("PDOS$_0$")
    plt.xlim(xlim)
    plt.title("Trace")
    plt.show()
    if spinpol:
        # Down spin with negative sign
        plt.figure()
        for i in range(norb):
            # Down spin
            plt.plot(w, -p0d_rspt[i, :], c="C" + str(i), label=str(i))
            plt.plot(w, -p0d[i, :], "--", c="C" + str(i))
            # Up spin
            plt.plot(w, p0d_rspt[norb + i, :], c="C" + str(i))
            plt.plot(w, p0d[norb + i, :], "--", c="C" + str(i))
        plt.legend()
        plt.ylabel("PDOS$_0$")
        plt.xlim(xlim)
        plt.title("Orbital resolved PDOS$_0$. -:p0d_rspt, --:p0d")
        plt.show()

    # Orbital resolved
    fig, axes = plt.subplots(nc, figsize=(6, 10), sharex=True)
    # Plot calculated PDOS (using e_rspt)
    for ax, y in zip(axes, p0d_initial):
        ax.plot(w, y, "-b", label="p0d_initial")
    # Plot calculated PDOS (using e_0d)
    for ax, y in zip(axes, p0d):
        ax.plot(w, y, "-g", label="p0d")
    # Plot original non-interacting PDOS
    for t, ax in enumerate(axes):
        ax.plot(w, p0d_rspt[t], "-r", label="p0d_rspt")
        ax.set_ylabel(str(t))
    # Figure design
    axes[-1].set_xlabel(r"$\omega$  (eV)")
    axes[0].legend(loc=2)
    axes[0].set_xlim(xlim)
    for i, ax in enumerate(axes):
        ax.grid()
    axes[0].set_title(r"Orbital resolved PDOS$_0$")
    plt.subplots_adjust(
        left=0.15, bottom=0.11, right=0.99, top=0.95, hspace=0, wspace=0
    )
    plt.show()

    if spinpol:
        fig, axes = plt.subplots(norb, figsize=(6, 6), sharex=True)
        # Plot calculated PDOS (using e_rspt)
        for i, ax in enumerate(axes):
            ax.plot(w, -p0d_initial[i, :], "-b", label=r"$\epsilon_\mathrm{rspt}$")
            ax.plot(w, p0d_initial[norb + i, :], "-b")
        # Plot calculated PDOS (using e)
        for i, ax in enumerate(axes):
            ax.plot(w, -p0d[i, :], "-g", label=r"$\epsilon$")
            ax.plot(w, p0d[norb + i, :], "-g")
        # Plot original PDOS
        for i, ax in enumerate(axes):
            ax.plot(w, -p0d_rspt[i], "-r", label="RSPt")
            ax.plot(w, p0d_rspt[norb + i], "-r")
            ax.set_ylabel(str(i))
        # Figure design
        axes[-1].set_xlabel(r"$\omega$  (eV)")
        axes[0].legend(loc=2)
        axes[0].set_xlim(xlim)
        plt.subplots_adjust(
            left=0.15, bottom=0.11, right=0.99, top=0.98, hspace=0, wspace=0
        )
        plt.show()


def plot_pdos0_4(w, p0d_rspt, p0d, p0_rspt, p0, norb, spinpol, xlim):
    """ """
    plt.figure()
    plt.plot(w, np.sum(p0d_rspt, axis=0), label="p0d_rspt")
    plt.plot(w, np.sum(p0d, axis=0), label="p0d")
    plt.plot(w, np.sum(p0_rspt, axis=0), label="p0_rspt")
    plt.plot(w, np.sum(p0, axis=0), label="p0")
    plt.legend()
    plt.ylabel("PDOS$_0$")
    plt.xlim(xlim)
    plt.show()
    if spinpol:
        fig, axes = plt.subplots(norb, figsize=(6, 6), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(w, -p0d[i, :], "--g", label=r"$\epsilon_{0,d}$")
            ax.plot(w, p0d[norb + i, :], "--g")
            ax.plot(w, -p0[i, :], "-g", label=r"$\epsilon_{0}$")
            ax.plot(w, p0[norb + i, :], "-g")

            ax.plot(w, -p0d_rspt[i], "--r", label="p0d_RSPt")
            ax.plot(w, p0d_rspt[norb + i], "--r")
            ax.plot(w, -p0_rspt[i], "-r", label="p0_RSPt")
            ax.plot(w, p0_rspt[norb + i], "-r")

            ax.set_ylabel(str(i))
        # Figure design
        axes[-1].set_xlabel(r"$\omega$  (eV)")
        axes[0].legend(loc=2)
        axes[0].set_xlim(xlim)
        plt.subplots_adjust(
            left=0.15, bottom=0.11, right=0.99, top=0.98, hspace=0, wspace=0
        )
        plt.show()


def plot_pdos0_from_off_diagonal_hyb(w, p0d_rspt, p0_rspt, p0d, xlim):
    plt.figure()
    plt.plot(w, np.sum(p0d_rspt, axis=0), label="p0d_rspt")
    plt.plot(w, np.sum(p0_rspt, axis=0), label="p0_rspt")
    plt.plot(w, np.sum(p0d, axis=0), label="p0d")
    plt.legend()
    plt.ylabel("PDOS$_0$")
    plt.xlim(xlim)
    plt.show()


def plot_pdos_2(w, p_rspt, p, xlim):
    """
    Plot interactive PDOSes.
    """
    nc = np.shape(p)[0]
    # Plot trace
    plt.figure()
    plt.plot(w, np.sum(p_rspt, axis=0), label="p_rspt")
    plt.plot(w, np.sum(p, axis=0), label="p")
    plt.legend()
    plt.ylabel("PDOS")
    plt.xlim(xlim)
    plt.show()
    # Plot orbital resolved PDOS
    fig, axes = plt.subplots(nrows=nc, sharex=True, sharey=True)
    for i in range(nc):
        axes[i].plot(w, p_rspt[i, :], label="p_rspt")
        axes[i].plot(w, p[i, :], label="p")
    axes[0].legend()
    axes[-1].set_xlabel("energy  (eV)")
    plt.xlim(xlim)
    plt.subplots_adjust(hspace=0)
    plt.show()


def plot_pdos_3(w, p_rspt, p_rspt_alg1, p_rspt_alg2, off_diag_hyb, spinpol):
    """
    Plot interactive PDOS.
    """
    nc = np.shape(p_rspt)[0]
    norb = nc // 2 if spinpol else nc
    plt.figure()
    plt.plot(w, np.sum(p_rspt, axis=0), label="RSPt")
    if off_diag_hyb:
        plt.plot(w, np.sum(p_rspt_alg1, axis=0), "--", label="RSPt, alg1")
    plt.plot(w, np.sum(p_rspt_alg2, axis=0), label="RSPt, alg2")
    plt.legend()
    plt.ylabel("PDOS")
    plt.xlim(xlim)
    plt.show()

    if spinpol:
        plt.figure()
        plt.plot(w, -np.sum(p_rspt[:norb, :], axis=0), c="C0", label="RSPt")
        if off_diag_hyb:
            plt.plot(
                w,
                -np.sum(p_rspt_alg1[:norb, :], axis=0),
                "--",
                c="C1",
                label="RSPt, alg1",
            )
        plt.plot(w, -np.sum(p_rspt_alg2[:norb, :], axis=0), c="C2", label="RSPt, alg2")
        plt.plot(w, np.sum(p_rspt[norb:, :], axis=0), c="C0")
        if off_diag_hyb:
            plt.plot(w, np.sum(p_rspt_alg1[norb:, :], axis=0), "--", c="C1")
        plt.plot(w, np.sum(p_rspt_alg2[norb:, :], axis=0), c="C2")
        plt.legend()
        plt.ylabel("PDOS")
        plt.xlim(xlim)
        plt.show()

        plt.figure()
        for i in range(norb):
            plt.plot(w, -p_rspt[i, :], c="C" + str(i), label=str(i))
            plt.plot(w, p_rspt[norb + i, :], c="C" + str(i))
        plt.legend()
        plt.ylabel("PDOS")
        plt.xlim(xlim)
        plt.show()


def plot_pdos_5(es, w, eim, p_rspt, hyb, sigmaM, spinpol):
    """
    Plot interactive PDOSes.

    Parameters
    ----------
    es : list
        Contains on-site energy sets: [e_rspt, e0d, e0, e]
    w : (N) array
    eim : float
    p_rspt : (M,N) array
    hyb : (M,N) array
    sigmaM : (M,M,N) array
    spinpol : boolean

    """
    labels = [
        r"$\epsilon_\mathrm{rspt}$",
        r"$\epsilon_{0,d}$",
        r"$\epsilon_0$",
        r"$\epsilon$",
    ]
    # Plot trace.
    # Verify different on-site energies by looking at
    # the interacting PDOS
    plt.figure()
    plt.plot(w, np.sum(p_rspt, axis=0), "k", label="RSPt")
    for en, label in zip(es, labels):
        if nc == no:
            tmp = np.diagonal(energies.pdos(w, eim, en, hyb, sigmaM)).T
        elif nc == 2 * no:
            tmp = np.diagonal(energies.pdos(w, eim, 2 * list(en), hyb, sigmaM)).T
        plt.plot(w, np.sum(tmp, axis=0), label=label)
    plt.legend()
    plt.xlabel("energy  (eV)")
    plt.xlim(xlim)
    plt.show()
    # Plot up and down spin
    if spinpol:
        plt.figure()
        plt.plot(w, -np.sum(p_rspt[:norb, :], axis=0), c="k", label="RSPt")
        plt.plot(w, np.sum(p_rspt[norb:, :], axis=0), c="k")
        for k, (en, label) in enumerate(zip(es, labels)):
            if nc == no:
                tmp = np.diagonal(energies.pdos(w, eim, en, hyb, sigmaM)).T
            elif nc == 2 * no:
                tmp = np.diagonal(energies.pdos(w, eim, 2 * list(en), hyb, sigmaM)).T
            plt.plot(w, -np.sum(tmp[:norb, :], axis=0), c="C" + str(k), label=label)
            plt.plot(w, np.sum(tmp[norb:, :], axis=0), c="C" + str(k))
        plt.legend()
        plt.xlabel("energy  (eV)")
        plt.xlim(xlim)
        plt.show()


def get_pdos0_eig_weight(e, b, v, w, eim, pdos_method="0"):
    """
    Return non-interacting PDOS for correlated orbitals.

    Eigenvalues and weights of non-interacting Hamiltonian
    is also returned.

    Parameters
    ----------
    e : (N) ndarray
        On-site energies of correlated orbitals.
    b : (N,M) ndarray
        On-site energies of bath orbitals.
    v : (N,M) ndarray
        Hopping elements.
    w : (K) ndarray
        Real energy mesh.
    eim : float
        Distance above real axis.
    pdos_method : str
        '0' or '1'
    """
    if pdos_method == "0":
        # Calculate the hamiltonian
        ham = h_d0(e, b, v)
        # Calculate the PDOS (using the Hamiltonian)
        pdos = pdos_d0_1(w, eim, ham)
        # Calculate eigenvalues and weights of Hamiltonian
        eig, weight = eig_and_weight1(ham)
    elif pdos_method == "1":
        # Calculate the hybridization function
        hyb = hybridization.hyb_d(w + 1j * eim, b, v)
        # Calulate the PDOS (using the hybridization function)
        pdos = pdos(w, eim, e, hyb)
        eig, weight = None, None
        print
        "Warning: eigenvalues and weights are not calculated"
    return pdos, eig, weight


def get_e0(w, eim, pdos_default, eb, vb, nc, no, bounds, wmin0, wmax0, verbose_text):
    """
    Return adjusted on-site energies.

    Due to the discretization approximation of the hybridization function,
    it is justified to try to compensate for this introduced error by
    adjusting the on-site energies.
    """
    e0 = np.zeros(no)
    if nc == no:
        for i in range(no):
            res = minimize_scalar(
                get_deviation,
                bounds=bounds,
                args=(eb[i], vb[i], w, eim, pdos_default[i], wmin0[i], wmax0[i]),
                method="bounded",
            )
            e0[i] = res["x"]
            if verbose_text:
                print("deviation(e0d) =", res.fun)
    else:
        for i in range(no):
            res = minimize_scalar(
                get_deviation_magnetic_nonSpinPol,
                bounds=bounds,
                args=(
                    [eb[i], eb[i + no]],
                    [vb[i], vb[i + no]],
                    w,
                    eim,
                    [pdos_default[i], pdos_default[i + no]],
                    wmin0[i],
                    wmax0[i],
                ),
                method="bounded",
            )
            e0[i] = res["x"]
            if verbose_text:
                print("deviation(e0d) =", res.fun)
    return e0


def get_e(w, eim, p_rspt, hyb, sigmaM, e0, wmin, wmax, verbose_text):
    """
    Return adjusted on-site energies.

    Due to the discretization approximation of the hybridization function,
    it is justified to try to compensate for this introduced error by
    adjusting the on-site energies.

    """
    mask = np.logical_and(wmin < w, w < wmax)
    nc = np.shape(p_rspt)[0]
    no = len(e0)
    avgErspt = np.zeros(nc)
    for i in range(nc):
        avgErspt[i] = cog(w[mask], p_rspt[i, mask])
    hybM_mask = np.zeros((nc, nc, len(w[mask])), dtype=complex)
    for i in range(nc):
        hybM_mask[i, i, :] = hyb[i, mask]
    sigmaM_mask = sigmaM[:, :, mask]
    # Trial energies
    e_start = e0.copy()
    # Optimize epsilon by fitting to interacting PDOS,
    # while keeping bath parameters fix
    res = minimize(
        get_deviation_using_self_energy,
        e_start,
        args=(avgErspt, w[mask], eim, hybM_mask, sigmaM_mask),
        method="SLSQP",
        bounds=[(wmin, wmax)] * no,
        options={"maxiter": 100, "disp": True},
    )
    e = res.x
    if verbose_text:
        print(
            "deviation(e_start) = ",
            get_deviation_using_self_energy(
                e_start, avgErspt, w[mask], eim, hybM_mask, sigmaM_mask
            ),
        )
        print("devation(e) = ", res.fun)
        print("nit = ", res.nit)
        print("success:", res.success)
        print("message:", res.message)
    return e


def get_deviation(e, eb, vb, w, eim, pdos_default, wmin, wmax):
    """
    Returns deviation of the center of gravity between default PDOS
    and the constructed non-intercating PDOS.

    Parameters
    ----------
    e : float
        On-site energy.
    eb : (N) array
        Bath energies.
    vb : (N) array
        hopping parameters.
    w : (M) array
        energy mesh.
    eim : float
        Distance above real energy axis.
    pdos_default : (M) array
        Default PDOS for comparison.
    wmin : float
        Lower limit of energy window.
    wmax : float
        Upper limit of energy window.

    Returns
    -------
    dev : float

    """
    h = h_d0(e, eb, vb)
    mask = np.logical_and(wmin < w, w < wmax)
    pdos0 = pdos_d0_1(w[mask], eim, h)
    return np.abs(cog(w[mask], pdos0) - cog(w[mask], pdos_default[mask]))


def get_deviation_magnetic_nonSpinPol(e, ebs, vbs, w, eim, pdos_defaults, wmin, wmax):
    """
    Returns deviation of center of gravity between default PDOS
    and the constructed non-intercating PDOSself.

    Usage: Magnetic simulations with non-spin polarized DFT functional
    combined with U (DFT+U) or DMFT (DFT+DMFT).
    In these scenarios, the spin-polarization is
    introduced by the self-energy.

    Parameters
    ----------
    e : float
    ebs : (K,N) array
    vbs : (K,N) array
    w : (M) array
    eim : float
    pdos_defaults : (K,M) array
    wmin : float
    wmax : float

    Returns
    -------
    dev : float

    """
    dev = 0
    # Loop over spin
    for eb, vb, pdos_default in zip(ebs, vbs, pdos_defaults):
        dev += err(e, eb, vb, w, eim, pdos_default, wmin, wmax) ** 2
    return dev


def get_deviation_using_self_energy(e, cog_rspt, w, eim, hyb, sig):
    """
    Return devation function.

    Usage: function to minimize in order to find optimal
    impurity energies for the finite impurity model.

    Parameters
    ----------
    e : (N) array
        on-site energies.
    cog_rspt : (M) array
        Center of gravity of interacting PDOS, within certain energy window.
    w : (K) array
        Energy mesh.
    eim : float
        Distance above real axis.
    hyb : (M,M,K) array
        Discrete hybridization function.
    sig : (M,M,K) array
        RSPt dynamic self-energy.
    """
    nc = len(cog_rspt)
    # Calculate approximative PDOS
    if len(e) == nc:
        p = np.diagonal(pdos(w, eim, e, hyb, sig))
    elif 2 * len(e) == nc:
        p = np.diagonal(pdos(w, eim, 2 * list(e), hyb, sig))
    # Sum of deviations of average energies
    s = 0
    for i in range(nc):
        s += abs(cog(w, p[:, i]) - cog_rspt[i]) ** 2
    return s


# Parse RSPt's out-file


def parse_matrices(out_file="out", search_phrase="Local hamiltonian"):
    """
    Return matrices and corresponding labels.

    Parameters
    ----------
    out_file : str
        File to read.
    search_phrase : str
        Search phrase for matrix.

    Returns
    -------
    hs : list
        List of matrices.
    labels : list
        List of labels.

    """
    with open(out_file, "r") as f:
        data = f.read()
    lines = data.splitlines()
    h_ids = []
    for i, line in enumerate(lines):
        if search_phrase in line:
            h_ids.append(i)
    # Store matrices
    hs = []
    # Store labels
    labels = []
    for h_id in h_ids:
        labels.append(lines[h_id].split()[1])
        # Real and imaginary part of matrix
        hr = []
        hi = []
        r_empty = False
        r_id = h_id + 2
        imag = 1
        # Loop until get empty line
        while r_empty is False:
            if len(lines[r_id].split()) == 0:
                r_empty = True
            elif lines[r_id].split()[0][:4] == "Imag":
                imag = 1j
                r_id += 1
            else:
                if imag == 1:
                    hr.append([float(c) for c in lines[r_id].split()])
                else:
                    hi.append([float(c) for c in lines[r_id].split()])
                r_id += 1
        hr = np.array(hr)
        hi = np.array(hi)
        h = hr + hi * 1j
        hs.append(h)
    return hs, labels


def print_matrix(x, space=7, ndecimals=3, fmt="f", cutoff=True):
    """
    Return string representation of matrix for printing.

    Parameters
    ----------
    x : (M,N) array
        Matrix to convert to a string.
    space : int
        Space for each number.
    ndecimals : int
        Number of decimals.
    fmt : {'f', 'E'}
        Print format keyword.
    cutoff : boolean
        If True, small numbers are presented as 0.

    Returns
    -------
    s : str
        String representation of matrix.

    Examples
    --------
    >>> from rspt2spectra.energies import print_matrix
    >>> from numpy.random import rand
    >>> x = rand(5,4)
    >>> print(x)
    [[ 0.84266211  0.51373679  0.62017691  0.14055559]
    [ 0.63183783  0.06084673  0.05167614  0.16491208]
    [ 0.55515508  0.47868486  0.79075186  0.4892547 ]
    [ 0.40485259  0.65460802  0.62777336  0.71200114]
    [ 0.54512609  0.18695706  0.6019384   0.85743096]]
    >>> print(print_matrix(x))
      0.843  0.514  0.620  0.141
      0.632  0.061  0.052  0.165
      0.555  0.479  0.791  0.489
      0.405  0.655  0.628  0.712
      0.545  0.187  0.602  0.857

    """
    fmt_f = "{:" + str(space) + "." + str(ndecimals) + "f}"
    fmt_e = "{:" + str(space) + "." + str(ndecimals) + "E}"
    fmt_int = "{:" + str(space) + "d}"

    if fmt == "f":
        fmt_s = fmt_f
    elif fmt == "E":
        fmt_s = fmt_e
    else:
        sys.exit("Wrong format given. Check variable fmt")
    if cutoff:
        s = []
        for row in x:
            rowl = []
            for item in row:
                if np.abs(item) < 0.5 * 10 ** (-ndecimals):
                    rowl.append(fmt_int.format(0))
                else:
                    rowl.append(fmt_s.format(item))
            s.append("".join(rowl))
        s = "\n".join(s)
    else:
        s = "\n".join(["".join([fmt_s.format(item) for item in row]) for row in x])

    return s


# ----------------------------------------------------------
# Functions related to impurity PDOS
# For example there is a function calculating the PDOS
# using the single-particle Hamiltonian $H_0$.
# And another using the hybridization function.


def lorentzian(w, wc, eim):
    """
    Return lorentzian with center at wc and with width
    given by eim.

    """
    return 1 / pi * eim / ((w - wc) ** 2 + eim**2)


def get_mu(path="out"):
    """
    Return the chemical potential.

    Parsed from file.
    If possible, the function greps for the
    'green_mu ' keyword, otherwise it will
    grep for the 'fermi energy' keyword.

    Parameters
    ----------
    path : str
        Filename of file to parse.

    """
    try:
        mu_value = subprocess.check_output("grep 'green_mu ' " + path, shell=True)
        mu_value = float(mu_value.split()[2])
    except subprocess.CalledProcessError:
        mu_value = subprocess.check_output("grep 'fermi energy' " + path, shell=True)
        mu_value = float(mu_value.split()[3])
    return mu_value


def pdos(w, eim, e, hyb, sig=0):
    r"""
    Return impurity projected density of states (PDOS).

    Parameters
    ----------
    w : (M) array
        Energy mesh :math:`\omega`.
    eim : float
        Distance :math:`\delta` above real-energy axis.
    e : {(N) array, (N,N) array}
        If (N) array: diagonal on-site energies
        If (N,N) array: full on-site matrix
    hyb : {(N,M) array, (N,N,M) array}
        Hybridization function :math:`\Delta(\omega+i\delta)`.
        If (N,M) array: diagonal hybridization function.
        If (N,N,M) array: full hybridization function.
    sig : {(N,M) array, (N,N,M) array}
        Self-energy :math:`\Sigma(\omega+i\delta)`.
        If (N,M) array: If equal dimensions, static self-energy,
        otherwise treated as diagonal but dynamical.
        If (N,N,M) array: full and dynamical self-energy.

    Returns
    -------
    dos : (N,N,M) ndarray
        Calculated PDOS.

    .. math:: PDOS_{a,b}(\omega) = ((  (\omega + i \delta)\delta_{i,j}
                                     - e_{i,j}
                                     - \Delta_{i,j}(\omega+i\delta)
                                     - \Sigma_{i,j}(\omega+i\delta)
                                    )^{-1})_{a,b}

    """
    e = np.array(e)
    # Number of correlated orbitals
    n = np.shape(e)[0]
    nw = len(w)
    if isinstance(sig, int) and sig == 0:
        sig = np.zeros((n, nw))
    # If everything is diagonal
    diag = e.ndim == 1 and hyb.ndim == 2 and np.shape(sig)[0] != np.shape(sig)[1]
    if diag:
        g = np.zeros((n, nw), dtype=complex)
        for i in range(n):
            g[i, :] = 1.0 / (w[:] + 1j * eim - e[i] - hyb[i, :] - sig[i, :])
    else:
        # Transform everything to off-diagonal
        # Make on-site energy 2d
        e = e if e.ndim == 2 else np.diag(e)
        # Make hybridization 3d
        if hyb.ndim == 2:
            tmp = np.zeros((n, n, nw), dtype=complex)
            for i in range(n):
                tmp[i, i, :] = hyb[i, :]
            hyb = tmp
        # Make self-energy 3d
        if sig.ndim == 2:
            tmp = np.zeros((n, n, nw), dtype=complex)
            if np.shape(sig)[0] == np.shape(sig)[1]:
                for i in range(nw):
                    tmp[:, :, i] = sig
            else:
                for i in range(n):
                    tmp[i, i, :] = sig[i, :]
            sig = tmp
        assert nw == np.shape(hyb)[2] and nw == np.shape(sig)[2]
        g = np.zeros((n, n, nw), dtype=complex)
        for i, x in enumerate(w):
            g[:, :, i] = np.linalg.inv(
                (x + 1j * eim) * np.eye(n) - e[:, :] - hyb[:, :, i] - sig[:, :, i]
            )
    dos = -1 / pi * np.imag(g)
    return dos


def pdos_d0_1(w, eim, hd0):
    """
    Return non-interacting impurity PDOS for
    the first orbital of each impurity type.

    Parameters
    ----------
    w : (N) array
        Energy mesh.
    eim : float
        Distance above real-energy axis.
    hd0 : {(N,N) array, (M,N,N) array}
        Single-particle Hamiltonian,
        can contain several impurity types
        (e.g. eg and t2g).

    """
    if len(np.shape(hd0)) == 2:
        hd0 = [hd0]
    # number of types
    n = np.shape(hd0)[0]
    pdos = np.zeros((n, len(w)))
    # loop over impurity types
    for t in range(n):
        eig, v = np.linalg.eigh(hd0[t])
        for e, weight in zip(eig, np.abs(v[0, :]) ** 2):
            pdos[t, :] += weight * lorentzian(w, e, eim)
    if n == 1:
        return pdos[0]
    else:
        return pdos


def eig_and_weight1(h0):
    """
    Return eigenvalues and weights for the first orbital,
    for each type.

    Parameters
    ----------
    h0 : (..., M, M) array
        Single-particle Hamiltonian,
        can contain several independent types
        (e.g. eg and t2g)

    Returns
    -------
    eig : (...,M) ndarray
        The eigenvalues.
        If one type, (M,) ndarray
        If many types, (N,M) ndarray
    """
    h0 = np.atleast_3d(h0)
    # number of types
    (nt, nh) = np.shape(h0)[:-1]
    eig = np.zeros((nt, nh))
    weight = np.zeros((nt, nh))
    # loop over types
    for t in range(nt):
        e, v = np.linalg.eigh(h0[t])
        eig[t, :] = e
        weight[t, :] = np.abs(v[0, :]) ** 2
    if nt == 1:
        return eig[0, :], weight[0, :]
    else:
        return eig, weight


def h_d0(e, eb=None, v=None):
    """
    Return single-particle Hamiltonian.

    Many independent impurity types,
    (e.g. eg and t2g), are possible.

    Parameters
    ----------
    e : (N) array
        Impurity energy level.
    eb : (N,M) array
        Bath energies.
    v : (N,M) array
        Hopping strengths.

    """
    # Transform to eventually one bigger dimension
    # in order to conveniently treat one or many
    # impurity orbitals
    e = np.atleast_1d(e)
    if (eb is None and v is None) or (len(eb) == 0 and len(v) == 0):
        # no bath states
        eb = [[]] * len(e)
        v = [[]] * len(e)
    else:
        eb = np.atleast_2d(eb)
        v = np.atleast_2d(v)
    # number of types and bath states
    (nt, nb) = np.shape(eb)
    ht = []
    # loop over the types
    for t in range(nt):
        # create the sub-block Hamiltonian
        h = np.atleast_2d(np.zeros((1 + nb, 1 + nb), dtype=complex))
        h[0, 0] = e[t]
        for i, e_bath in enumerate(eb[t]):
            h[1 + i, 1 + i] = e_bath
        for i, vb in enumerate(v[t]):
            h[1 + i, 0] = vb
            # it could actually be complex conjugate
            # here but then H needs to be defined as complex
            h[0, 1 + i] = np.conj(vb)
        ht.append(h)
    if nt == 1:
        return ht[0]
    else:
        return ht

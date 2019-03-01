#!/usr/bin/env python3

"""

hybridization
=============

This module contains functions which are useful to
process hybridization data generated by the RSPt software.

"""

import numpy as np
import matplotlib.pylab as plt
from math import pi
from itertools import chain

from .energies import cog
from .constants import eV

def plot_hyb(filename, xlim, spinpol, norb, nc):
    """
    Plot hybridization functions from file.
    """
    x = np.loadtxt(filename)
    wp = x[:, 0]*eV
    mask = np.logical_and(xlim[0] < wp, wp < xlim[1])
    wp = wp[mask]
    # Total hybridization
    plt.figure()
    hyb_tot = -1/pi*x[mask, 1]*eV
    plt.plot(wp, hyb_tot)
    plt.xlabel('$\omega$   (eV)')
    plt.ylabel(r'$\frac{-1}{\pi}$ Im $\Delta(\omega)$')
    plt.xlim(xlim)
    plt.grid(color='0.9')
    plt.title("Total hybridzation function")
    plt.show()
    if spinpol:
        # Spin up and down
        plt.figure()
        hyb_dn = -1/pi*x[mask, 2]*eV
        plt.plot(wp, -hyb_dn, c='C0')
        hyb_up = -1/pi*x[mask, 3]*eV
        plt.plot(wp, hyb_up, c='C0')
        plt.xlabel('$\omega$   (eV)')
        plt.ylabel(r'$\frac{-1}{\pi}$ Im $\Delta(\omega)$')
        plt.xlim(xlim)
        plt.grid(color='0.9')
        plt.show()
        # Orbital resolved
        fig = plt.figure()
        for i in range(norb):
            hyb_dn = -1/pi*x[mask, 4+i]*eV
            hyb_up = -1/pi*x[mask, 4+norb+i]*eV
            plt.plot(wp, hyb_dn + hyb_up, c='C' + str(i), label=str(i))
        plt.ylabel(r'$\frac{-1}{\pi}$ Im $\Delta(\omega)$')
        plt.grid(color='0.9')
        plt.legend()
        plt.xlabel('$\omega$   (eV)')
        plt.xlim(xlim)
        plt.tight_layout()
        plt.show()
        # Orbital and spin resolved
        fig = plt.figure()
        for i in range(norb):
            hyb_dn = -1/pi*x[mask, 4+i]*eV
            # Plot down spin with minus sign
            plt.plot(wp, -hyb_dn, c='C' + str(i), label=str(i))
            hyb_up = -1 / pi * x[mask, 4+norb+i]*eV
            plt.plot(wp, hyb_up, c='C' + str(i))
        plt.ylabel(r'$\frac{-1}{\pi}$ Im $\Delta(\omega)$')
        plt.grid(color='0.9')
        plt.legend()
        plt.xlabel('$\omega$   (eV)')
        plt.xlim(xlim)
        plt.tight_layout()
        plt.show()
        # Orbital and spin resolved
        fig, axes = plt.subplots(nrows=norb, sharex=True, sharey=True)
        for i, ax in enumerate(axes):
            hyb_dn = -1/pi*x[mask, 4+i]*eV
            # Plot down spin with thinner line
            ax.plot(wp, hyb_dn, lw=0.7, c='C' + str(i))
            hyb_up = -1/pi*x[mask, 4+norb+i]*eV
            ax.plot(wp, hyb_up, c='C' + str(i), label=str(i))
            ax.grid(color='0.9')
            ax.legend()
        axes[-1].set_ylabel(r'$\frac{-1}{\pi}$ Im $\Delta(\omega)$')
        axes[-1].set_xlabel('$\omega$   (eV)')
        axes[-1].set_xlim(xlim)
        plt.tight_layout()
        plt.show()
    else:
        # Orbital resolved
        fig = plt.figure()
        for i in range(nc):
            hyb_i = -1/pi*x[mask, 4+i]*eV
            plt.plot(wp, hyb_i, label=str(i))
        plt.ylabel(r'$\frac{-1}{\pi}$ Im $\Delta(\omega)$')
        plt.grid(color='0.9')
        plt.legend()
        plt.xlabel('$\omega$   (eV)')
        plt.xlim(xlim)
        plt.tight_layout()
        plt.title("Orbital resolved hybridzation function")
        plt.show()

def plot_discrete_hyb(w,hyb_im,hyb_im_rspt,eb,vb,wborder,nc,spinpol,xlim):
    """
    Plot discretized hybridization functions.
    """
    nb = np.shape(eb)[1]
    fig, axarr = plt.subplots(nc, figsize=(7, 8), sharex=True,
                              sharey=True)
    # Loop over non-equivalent correlated spin-orbitals
    for i, ax in enumerate(axarr):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, nb)))
        ax_v = ax.twinx()
        ax_v.set_ylabel('$v_{b}$  (eV)', color='r')
        ax_v.tick_params('y', colors='r')
        # Loop over bath states
        for ebath, v, wb, c in zip(eb[i], vb[i], wborder[i], color):
            ax_v.plot([ebath, ebath], [0, v], c='r')
            ax_v.plot([wb[0], wb[0]], [0, v], '--', c=c)
            ax_v.plot([wb[1], wb[1]], [0, v], '--', c=c)
        # Plot discretized hybridization function
        ax.plot(w, -1/pi*hyb_im[i, :], c='0.8', label='discrete')
        v_max_plot = 1.07 * np.max(vb[i])
        ax_v.set_ylim([0, v_max_plot])
        ax.grid(color='0.92')
    axarr[int(nc / 2)].set_ylabel(r'-$\frac{1}{\pi}\mathrm{Im}'
                                  '\Delta(\omega)$      (eV)')
    # Plot continues hybridization function
    for i, ax in enumerate(axarr):
        hyb_max_plot = 1.2 * np.max(-1/pi*hyb_im_rspt[i])
        ax.set_ylim([0, hyb_max_plot])
        ax.plot(w, -1/pi*hyb_im_rspt[i], '-k', label='RSPt')
    axarr[-1].set_xlabel('E   (eV)')
    axarr[0].set_xlim(xlim)
    axarr[0].legend(loc='best')
    axarr[0].set_title("Orbital resolved hybridization")
    plt.subplots_adjust(left=0.17, bottom=0.10, right=0.83,
                        top=0.95, hspace=0.0, wspace=0)
    plt.show()

    if spinpol:
        fig, axarr = plt.subplots(norb, figsize=(6, 7), sharex=True)
        # Loop over axes
        for i, ax in enumerate(axarr):
            ax_v = ax.twinx()
            ax_v.set_ylabel('$v_{b}$  (eV)', color='r')
            ax_v.tick_params('y', colors='r')

            for j, s in zip([i, norb + i], [-1, 1]):
                color = iter(plt.cm.rainbow(np.linspace(0, 1, nb)))
                # Loop over bath states
                for ebath, v, wb, c in zip(eb[j], vb[j],
                                           wborder[j], color):
                    ax_v.plot([ebath, ebath], [0, s * v], c='r')
                    ax_v.plot([wb[0], wb[0]], [0, s * v], '--', c=c)
                    ax_v.plot([wb[1], wb[1]], [0, s * v], '--', c=c)
            # Plot discretized hybridization function
            ax.plot(w, --1/pi*hyb_im[i, :],
                    c='0.8', label='discrete')
            ax.plot(w, -1/pi*hyb_im[norb + i, :], c='0.8')
            v_max_dn = 1.1 * np.max(vb[i])
            v_max_up = 1.1 * np.max(vb[norb + i])
            v_max = max(v_max_dn, v_max_up)
            ax_v.set_ylim([-v_max, v_max])
            ax.grid(color='0.92')
        axarr[2].set_ylabel(r'-$\frac{1}{\pi}\mathrm{Im}'
                            '\Delta(\omega)$      (eV)')
        # Plot RSPt hybridization function
        for i, ax in enumerate(axarr):
            hyb_max_dn = 1.4 * np.max(-1/pi*hyb_im_rspt[i])
            hyb_max_up = 1.4 * np.max(-1/pi*hyb_im_rspt[norb + i])
            hyb_max = max(hyb_max_dn, hyb_max_up)
            ax.set_ylim([-hyb_max, hyb_max])
            ax.plot(w, --1/pi*hyb_im_rspt[i], '-r', label='RSPt')
            ax.plot(w, -1/pi*hyb_im_rspt[norb + i], '-r')
        axarr[-1].set_xlabel('E   (eV)')
        axarr[0].set_xlim(xlim)
        axarr[0].legend(loc='best')
        plt.subplots_adjust(left=0.17, bottom=0.10, right=0.83,
                            top=0.98, hspace=0.0, wspace=0)
        plt.show()
        # New figure
        fig, axarr = plt.subplots(norb, figsize=(6, 7), sharex=True)
        # Loop over axes
        for i, ax in enumerate(axarr):
            # Down spin has negative sign
            ax.plot(w, --1/pi*hyb_im[i, :], c='0.8')
            ax.plot(w, --1/pi*hyb_im_rspt[i], '-r')
            # Up spin has positive sign
            ax.plot(w, -1/pi*hyb_im[norb + i, :],
                    c='0.8', label='discrete')
            ax.plot(w, -1/pi*hyb_im_rspt[norb + i],
                    '-r', label='RSPt')
            y_max_dn = 1.4 * np.min(-1/pi*hyb_im_rspt[i])
            y_max_up = 1.4 * np.max(-1/pi*hyb_im_rspt[norb + i])
            y_max = max(y_max_dn, y_max_up)
            ax.text(0, 2, i, fontsize=8,
                    fontweight='normal',
                    bbox={'facecolor': 'white', 'alpha': 0.9,
                          'pad': 2})
            ax.set_ylim([-y_max, y_max])
        axarr[0].set_ylabel(r'-$\frac{1}{\pi}\mathrm{Im}'
                            '\Delta(\omega)$  (eV)')
        axarr[-1].set_xlabel('E   (eV)')
        axarr[0].set_xlim(xlim)
        axarr[0].legend(loc=1)
        plt.subplots_adjust(left=0.16, bottom=0.16, right=0.99,
                            top=0.99, hspace=0, wspace=0)
        plt.show()


def plot_hyb_off_diagonal(w,hyb,nc,xlim):
    """
    Plot off-diagonal elements of hybridization functions.

    Parameters
    ----------
    w : (N) array
        Energy mesh.
    hyb : (M,M,N) array
        Matrix of hybridization function.
    nc : int
        Number of non-equivalent correlated spin-orbitals.
    xlim : list
        Plotting window.

    """
    plt.figure()
    for i in range(nc):
        plt.plot(w, -np.imag(hybM_rspt[i, i, :]), '-k')
    for i in range(nc):
        for j in range(nc):
            if i != j:
                plt.plot(w, -np.imag(hybM_rspt[i, j, :]), '-r')
    plt.plot([], [], '-k', label='diagonal')
    plt.plot([], [], '-r', label='off-diagonal')
    plt.legend()
    plt.xlim(xlim)
    plt.show()


def get_v_simple(w, hyb, eb, width=0.5):
    """
    Return hopping parameters, in a simple fashion.

    Integrate hybridization weight within a fixed energy
    window around peak.
    This weight is given to the bath state.

    The energy window, determined by width,
    should be big enough to capture the main part
    of the hybridization in that area, but small enough so
    no weight is contributing to different bath states.
    """
    # Hopping strength parameters
    vb = []
    # Number of different correlated orbitals,
    # e.g. e_g and t_2g
    norb = len(eb)
    # Loop over correlated obitals
    for i in range(norb):
        vb.append([])
        # Loop over bath states
        for e in eb[i]:
            mask = np.logical_and(e - width / 2 < w, w < e + width / 2)
            vb[i].append(
                np.sqrt(np.trapz(-hyb[mask, i], w[mask]) / np.pi))
    return np.array(vb)

def get_vb_and_eb(w, hyb, wborder):
    """
    Return hopping and bath energy parameters.

    Extract the hopping strengths by integrating the negative part of
    the hybridization function.
    Extract the bath energies by calculating the center of gravity of the
    negative part of the hybridization function.

    Parameters
    ----------
    w : (K) array
        Energy vector.
    hyb : (N,K) array
        The imaginary part of the continous hybridization function.
        Orbitals on seperate rows.
    wborder : (N,M,2) array
        Energy borders for the bath energies.

    Returns
    -------
    vb : (N,M) array
        List of lists of hopping parameters for different
        correlated orbitals.
    eb : (N,M) array
        Bath energies.

    """
    nb = len(wborder[0,:,0])
    print(np.shape(wborder))
    # Check so that energy windows do not overlap
    # Loop over correlated orbitals
    for i in range(np.shape(wborder)[0]):
        s = np.argsort(wborder[i,:,0])
        for j in range(nb):
            assert wborder[i,s[j],0] < wborder[i,s[j],1]
            if j < nb-1:
                assert wborder[i,s[j],1] <= wborder[i,s[j+1],0]
    vb = np.zeros(np.shape(wborder)[:2],dtype=np.float)
    eb = np.zeros_like(vb)
    # Loop over correlated orbitals
    for i in range(np.shape(wborder)[0]):
        # Loop over bath states
        for j in range(len(wborder[i])):
            kmin = np.argmin(np.abs(w - wborder[i,j][0]))
            kmax = np.argmin(np.abs(w - wborder[i,j][1]))
            vb[i,j] = np.sqrt(np.trapz(-hyb[i, kmin:kmax], w[kmin:kmax])/pi)
            eb[i,j] = cog(w[kmin:kmax],-hyb[i, kmin:kmax])
    return vb, eb


def get_vb_and_new_eb(w, hyb, eb, accept1=0.1, accept2=0.5, nbig=20):
    """
    Return hopping parameters and new bath energies.

    Integrate hybridization spectral weight within an energy
    window around peak.
    This weight is given to the bath state.
    New bath energies are calculated from center of gravity.

    Parameters
    ----------
    w : (K) array
        Energy vector.
    hyb : (N,K) vector
        The imaginary part of hybridization function.
        Orbitals on seperate rows.
    eb : (N,M) array
        List of list of bath energies for different
        correlated orbitals.
    accept1 : float
        Parameter to determine the energy window.
    accept2 : float
        Parameter to determine the energy window.
    nbig : int
        Parameter to determine the energy window.

    Returns
    -------
    vb : (N,M) array
        List of lists of hopping parameters for different
        correlated orbitals.
    eb_new : (N,M) array
        New bath energies equal to the center of gravity of the hybridization
        function within the energy windows.
    wborder : (N,M,2) array
        Integration energy window borders for each
        bath state.

    """
    vb = np.zeros_like(eb)
    eb_new = np.zeros_like(eb)
    wborder = np.zeros(np.shape(vb) + (2,))
    # Loop over correlated orbitals
    for i in range(np.shape(eb)[0]):
        # Get energy window border indices
        kmins, kmaxs = get_border_index(w, -hyb[i, :], eb[i,:],
                                        accept1, accept2, nbig)
        # Loop over bath states
        for j in range(len(eb[i,:])):
            kmin = kmins[j]
            kmax = kmaxs[j]
            wborder[i,j,:] = [w[kmin], w[kmax]]
            vb[i,j] = np.sqrt(np.trapz(-hyb[i, kmin:kmax], w[kmin:kmax])/pi)
            eb_new[i,j] = cog(w[kmin:kmax],-hyb[i, kmin:kmax])
    return vb, eb_new, wborder


def get_border_index(x, y, eb, accept1, accept2, nbig):
    r"""
    Return the lower/left and upper/right (integration)
    limit indices.

    First, the left and the right limits are determined
    independently of each other.
    For both limits, three criteria is used to determine
    the limit position.
    1) Look for intensity drop to `accept1` times the value
    in `y` at the bath energy.
    2) Look for intensity minimum in `y` between neighbouring
    bath energy, which is lower than `accept2` times the
    value of `y` at the bath energy.
    3) Pick limit halfway to the other bath energy.

    - Criterion one is initially tried.
    - If it is successful, it returns the limit position.
    - If not successful, the second criterion is tried.
    - If it is successful, it returns the limit position.
    - If not successful, the third criterion is used as a
      final resort.

    To avoid energy windows to overlap,
    the borders of the energy windows are checked
    and in the case of an overlap,
    instead the mean of the overlapping border energies are
    used as border.

    If there are more than a big number of bath states,
    the edge bath states are specially treated.
    This is to avoid the integration windows for the edge bath
    states to become unreasonably big.
    Without special treatment, this problem arises if the
    hybridization intensity at the edge bath state is
    small (but not very small).
    Then the criteria p1 or p2 will be satisfied,
    but at an energy really far away from the bath location.
    Instead, with the special treatment, the border for
    the edge bath state is determined by the distance to
    the nearest bath state.

    Parameters
    ----------
    x : (N) array
        Representing the energy mesh.
    y : (N) array
        Corresponding values to the energies in x.
    eb : (M) array
        Array of bath energies
    accept1 : float
        A procentage parameter used for criterion 1.
        This value times the value of y at the bath
        location sets accepted minimum value of y.
    accept2 : float
        A procentage parameter used for criterion 2.
        This value times the value of y at the bath
        location sets accepted minimum value of y.
    nbig : int
        If the number of bath states exceeds this
        number, the edge bath states are specially
        treated.

    Returns
    -------
    kmins_new : list
        List of left limit indices.
    kmaxs_new : list
        List of left limit indices.

    """
    # Index vectors to be returned
    kmins = []
    kmaxs = []
    # Loop over the bath energies
    for e_index, e in enumerate(eb):
        # The index corresponding to the bath energy
        kc = np.argmin(np.abs(x - e))
        # Accepted peak intensity, according to criterion
        # 1 and 2, respectively
        p1, p2 = accept1 * y[kc], accept2 * y[kc]
        other_peaks = np.delete(np.array(eb), e_index)
        # Find left border
        if np.any(other_peaks < e):
            left_peak_index = np.argmin(np.abs(
                e - other_peaks[other_peaks < e]))
            k_left = np.argmin(np.abs(
                other_peaks[other_peaks < e][left_peak_index] - x))
        else:
            k_left = 0
        # Check if bath state is an edge bath state
        if k_left == 0 and len(eb) > nbig:
            # Pick point at distance equal to the
            # Distance to the bath energy to the right
            de = np.min(other_peaks - e)
            kmin = np.argmin(np.abs(x - (e - de)))
        else:
            # Look for intensity lower than p1
            for k in np.arange(kc, k_left, -1):
                if y[k] < p1:
                    kmin = k
                    break
            else:
                # Another bath energy was reached.
                # Therefore, look for intensity minimum
                # between them, which should be lower than p2
                for k in np.arange(kc, k_left + 2, -1):
                    if (y[k - 1] < y[k] and y[k - 1] < y[k - 2]) and y[k] < p2:
                        kmin = k - 1
                        break
                else:
                    # There is no intensity minimum.
                    if k_left == 0 and len(other_peaks) > 0:
                        # Pick point at distance equal to the
                        # distance to the bath energy to the
                        # right
                        de = np.min(other_peaks - e)
                        kmin = np.argmin(np.abs(x - (e - de)))
                    else:
                        # Pick point halfway between them.
                        kmin = np.argmin(np.abs(
                            x - (x[k_left] + x[kc]) / 2))

        # find right border
        if np.any(other_peaks > e):
            right_peak_index = np.argmin(np.abs(
                e - other_peaks[other_peaks > e]))
            k_right = np.argmin(np.abs(
                other_peaks[other_peaks > e][right_peak_index] - x))
        else:
            k_right = len(x) - 1
        # Check if bath state is an edge bath state
        if k_right == len(x) - 1 and len(eb) > nbig:
            # Pick point at distance equal to the
            # distance to the bath energy to the left
            de = np.min(e - other_peaks)
            kmax = np.argmin(np.abs(x - (e + de)))
        else:
            # look for intensity lower than p1
            for k in np.arange(kc, k_right):
                if y[k] < p1:
                    kmax = k
                    break
            else:
                # Another bath energy was reached.
                # Therefore, look for intensity minimum
                # between them, which should be lower than p2
                for k in np.arange(kc, k_right - 2):
                    if (y[k + 1] < y[k] and y[k + 1] < y[k + 2]) and y[k] < p2:
                        kmax = k + 1
                        break
                else:
                    # There is no intensity minimum.
                    if k_right == len(x) - 1 and len(other_peaks) > 0:
                        # Pick point at distance equal to the
                        # distance to the bath energy to the
                        # left
                        de = np.min(e - other_peaks)
                        kmax = np.argmin(np.abs(x - (e + de)))
                    else:
                        # Pick point halfway between them.
                        kmax = np.argmin(np.abs(
                            x - (x[kc] + x[k_right]) / 2))
        kmins.append(kmin)
        kmaxs.append(kmax)
    # copy the lists
    kmins_new = list(kmins)
    kmaxs_new = list(kmaxs)
    # check for overlaps, by looping over the bath energies
    for e_index, (e, kmin, kmax) in enumerate(zip(eb, kmins, kmaxs)):
        # loop over the other bath energies
        for i in chain(range(e_index), range(e_index + 1, len(eb))):
            # check for overlap to the left
            if eb[i] < e and kmin < kmaxs[i]:
                kmins_new[e_index] = np.argmin(np.abs(
                    x - (x[kmin] + x[kmaxs[i]]) / 2.))
            # check for overlap to the right
            if eb[i] > e and kmax > kmins[i]:
                kmaxs_new[e_index] = np.argmin(np.abs(
                    x - (x[kmax] + x[kmins[i]]) / 2.))
    return kmins_new, kmaxs_new


def hyb_d(z, eb, vb):
    """
    return the hybridization function at points z.

    Several independent impurity orbitals,
    e.g. e_g and t_2g, possible.

    Parameters
    ----------
    z : (M) array
        Vector containing points in the complex plane.
    eb : (N,K) array
        Bath energies.
    vb : (N,K) array
        Hopping parameters.

    Returns
    -------
    d : {(M) ndarray, (N,M) ndarray}
        Hybridization function.

    """
    eb = np.atleast_2d(eb)
    vb = np.atleast_2d(vb)
    # Number of different impurity orbitals
    (norb, nb) = np.shape(eb)
    # Hybridization function
    d = np.zeros((norb, len(z)), dtype=np.complex)
    # Loop over correlated obitals
    for i in range(norb):
        # Loop over bath states
        for e, v in zip(eb[i], vb[i]):
            d[i, :] += np.abs(v) ** 2 / (z - e)
    if norb == 1:
        return d[0, :]
    else:
        return d

"""
Visualize a hybridization fit and the contribution of each bath state.

`plot_hyb_fit` produces one figure per inequivalent block: a top panel with
the original hybridization function (filled) and the fitted one (curve,
including the constant shift C), followed by one panel per bath state showing
its contribution in the selected bath geometry. For the star geometry a bath
state is a fitted level; for chain-type geometries it is a chain site, and
the panel for distance i overlays the sites i steps from the impurity of
every chain (e.g. the occupied and unoccupied chains).

Chain-site contributions use eigenmode-site attribution: with
``H_bath = U diag(lam) U^dagger`` the fitted hybridization is a sum of poles
``w_n / (z - lam_n)`` with ``w_n = (v^dagger u_n)(u_n^dagger v)``, and site j
receives the fraction ``|u_n[j]|^2`` of pole n. The site contributions sum
exactly to the pole part of the fit and have non-negative spectral weight on
the diagonal.
"""

import numpy as np

from rspt2spectra.block_structure import build_matrix
from rspt2spectra.edchain import build_H_bath_v
from rspt2spectra.offdiagonal import get_hyb_2

# Fixed categorical hue order (validated palette); color follows the orbital.
_ORBITAL_COLORS = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_ORIG_OFFDIAG_COLOR = "#b0afa9"
_FIT_OFFDIAG_COLOR = "#52514e"
_CHAIN_LINESTYLES = ["-", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

_CHAIN_GEOMETRIES = ("chain", "single_chain", "linked_chain", "peeled_linked_chain")


def bath_distances(H_bath, v, tol=1e-10):
    """Classify bath orbitals by hopping distance from the impurity.

    Parameters
    ----------
    H_bath : (n_bath, n_bath) np.ndarray
        Bath Hamiltonian of one block, in any geometry.
    v : (n_bath, n_orb) np.ndarray
        Impurity-bath hopping of the same block.
    tol : float, default 1e-10
        Matrix elements with magnitude below tol are treated as zero.

    Returns
    -------
    depth : (n_bath,) np.ndarray of int
        Number of hops from the impurity (1 = couples directly through v);
        -1 for bath orbitals with no path to the impurity.
    component : (n_bath,) np.ndarray of int
        Connected component of the bath-bath hopping graph (a chain id for
        the chain geometries).
    """
    n_bath = H_bath.shape[0]
    adjacency = np.abs(H_bath) > tol
    np.fill_diagonal(adjacency, False)

    depth = np.full(n_bath, -1, dtype=int)
    frontier = np.linalg.norm(v, axis=1) > tol
    d = 1
    while np.any(frontier):
        depth[frontier] = d
        frontier = np.any(adjacency[frontier], axis=0) & (depth == -1)
        d += 1

    component = np.full(n_bath, -1, dtype=int)
    c = 0
    for start in range(n_bath):
        if component[start] != -1:
            continue
        members = np.zeros(n_bath, dtype=bool)
        members[start] = True
        frontier = members.copy()
        while np.any(frontier):
            frontier = np.any(adjacency[frontier], axis=0) & ~members
            members |= frontier
        component[members] = c
        c += 1
    return depth, component


def site_resolved_hyb(z, H_bath, v, groups):
    """Split the hybridization of one block into per-site-group contributions.

    Parameters
    ----------
    z : (M,) complex np.ndarray
        Energy mesh.
    H_bath : (n_bath, n_bath) np.ndarray
        Hermitian bath Hamiltonian.
    v : (n_bath, n_orb) np.ndarray
        Impurity-bath hopping; the full hybridization is
        ``v^dagger (z - H_bath)^{-1} v``.
    groups : list of array_like of int
        Groups of bath-orbital indices (e.g. the orbitals of one chain site).

    Returns
    -------
    contributions : (n_groups, M, n_orb, n_orb) np.ndarray
        Per-group hybridization contributions; they sum to the full
        hybridization when the groups cover all coupled bath orbitals.
    """
    lam, U = np.linalg.eigh(H_bath)
    B = np.conj(U.T) @ v  # (n_modes, n_orb), row n = u_n^dagger v
    w = np.conj(B)[:, :, None] * B[:, None, :]  # (n_modes, n_orb, n_orb)
    site_weight = np.abs(U) ** 2  # (n_bath, n_modes)
    g = np.stack([np.sum(site_weight[list(group)], axis=0) for group in groups])
    poles = 1.0 / (z[:, None] - lam[None, :])  # (M, n_modes)
    return np.einsum("gn,mn,nij->gmij", g, poles, w)


def _star_panels(z, ebs, vs):
    """Per-level contributions and labels for the star geometry."""
    order = np.argsort(ebs, kind="stable")
    contributions = [
        get_hyb_2(z, ebs[k : k + 1][np.newaxis], vs[k : k + 1][np.newaxis])[0]
        for k in order
    ]
    labels = [[(f"$\\varepsilon_b = {ebs[k]: .3f}$", "-")] for k in order]
    return contributions, labels


def _chain_panels(z, H_bath, v):
    """Per-distance contributions (chains overlaid) and labels."""
    depth, component = bath_distances(H_bath, v)
    coupled = depth > 0
    chain_label = {}
    chains = np.unique(component[coupled])
    for c in chains:
        members = component == c
        if len(chains) == 1:
            chain_label[c] = ""
        else:
            occ = np.mean(np.real(np.diag(H_bath))[members]) < 0
            chain_label[c] = "occ" if occ else "unocc"

    groups = []
    panel_of_group = []
    labels = []
    for i, d in enumerate(np.unique(depth[coupled])):
        labels.append([])
        for c in chains:
            group = np.where((depth == d) & (component == c))[0]
            if len(group) == 0:
                continue
            groups.append(group)
            panel_of_group.append(i)
            style = _CHAIN_LINESTYLES[
                np.searchsorted(chains, c) % len(_CHAIN_LINESTYLES)
            ]
            name = f"site {d}"
            if chain_label[c]:
                name += f" ({chain_label[c]})"
            labels[-1].append((name, style))

    group_hyb = site_resolved_hyb(z, H_bath, v, groups)
    contributions = [
        [group_hyb[g] for g in range(len(groups)) if panel_of_group[g] == i]
        for i in range(len(labels))
    ]
    return contributions, labels


def _offdiag_norm(hyb_block):
    """Return the summed magnitude of the off-diagonal elements, (M,)."""
    mask = ~np.eye(hyb_block.shape[-1], dtype=bool)
    return np.sum(np.abs(hyb_block[:, mask]), axis=-1)


def _annotate_shift(fig, C):
    """Print the fitted constant shift C on the figure."""
    if C.shape == (1, 1):
        text = f"C = {np.real(C[0, 0]): .4f}"
    else:
        rows = [
            "  ".join(f"{np.real(x): .3f}{np.imag(x):+.3f}i" for x in row) for row in C
        ]
        text = "C =\n" + "\n".join(rows)
    fig.text(
        0.99,
        0.995,
        text,
        ha="right",
        va="top",
        fontsize=7,
        family="monospace",
        color=_FIT_OFFDIAG_COLOR,
    )


def plot_hyb_fit(
    w,
    eim,
    hyb,
    ebs_star,
    vs_star,
    cs_star,
    H_local_Q,
    block_structure,
    bath_geometry,
    peel_weight=0.05,
):
    """Plot the hybridization fit and per-bath-state contributions per block.

    Parameters
    ----------
    w : (M,) np.ndarray
        Real frequency mesh.
    eim : float
        Imaginary offset; everything is evaluated at ``w + 1j*eim``.
    hyb : (M, n_orb, n_orb) np.ndarray
        Original hybridization function in the fitting basis.
    ebs_star, vs_star, cs_star : list of np.ndarray
        Fitted star bath energies, hoppings and constant shifts per
        inequivalent block (from :func:`rspt2spectra.hyb_fit.fit_hyb`).
    H_local_Q : (n_orb, n_orb) np.ndarray
        Local Hamiltonian in the fitting basis; anchors the chain geometries.
    block_structure : BlockStructure
        Block partition of the hybridization function.
    bath_geometry : str
        Geometry the bath states are shown in; anything outside
        ``{"chain", "single_chain", "linked_chain", "peeled_linked_chain"}``
        is treated as star.
    peel_weight : float, default 0.05
        Passed through to the peeled linked chain construction.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        One figure per fitted inequivalent block.
    """
    # matplotlib is an optional dependency; import it only when plotting.
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    z = w + 1j * eim
    chain_geometry = bath_geometry in _CHAIN_GEOMETRIES
    if chain_geometry:
        H_shift = build_matrix(cs_star, block_structure)
        H_baths, vs_geom = build_H_bath_v(
            H_local_Q + H_shift,
            ebs_star,
            vs_star,
            bath_geometry,
            block_structure,
            verbose=False,
            extra_verbose=False,
            peel_weight=peel_weight,
        )

    figs = []
    for b, (ebs, vs, C) in enumerate(zip(ebs_star, vs_star, cs_star)):
        if len(ebs) == 0:
            continue
        orbs = block_structure.blocks[block_structure.inequivalent_blocks[b]]
        n_orb = len(orbs)
        orig = hyb[np.ix_(np.arange(len(w)), orbs, orbs)]
        fit = get_hyb_2(z, ebs[np.newaxis], vs[np.newaxis], C)[0]

        if chain_geometry:
            contributions, labels = _chain_panels(z, H_baths[b], vs_geom[b])
        else:
            contributions, labels = _star_panels(z, ebs, vs)
            contributions = [[c] for c in contributions]

        total = np.sum([np.sum(c, axis=0) for c in contributions], axis=0) + C
        if not np.allclose(total, fit, atol=1e-6):
            print(
                f"WARNING: contributions of block {b} do not sum to the star "
                f"fit (max dev {np.max(np.abs(total - fit)):.3e}); the "
                f"{bath_geometry} geometry changed the hybridization."
            )

        n_rows = 1 + len(contributions)
        fig, axes = plt.subplots(
            n_rows,
            2,
            sharex=True,
            figsize=(9, 1.4 * n_rows + 1.2),
            squeeze=False,
            gridspec_kw={"height_ratios": [2] + [1] * len(contributions)},
        )
        fig.suptitle(f"Hybridization fit, orbitals {orbs} ({bath_geometry})")
        colors = [_ORBITAL_COLORS[i % len(_ORBITAL_COLORS)] for i in range(n_orb)]

        ax_im, ax_re = axes[0]
        for i, color in enumerate(colors):
            ax_im.fill_between(w, -orig[:, i, i].imag, color=color, alpha=0.3, lw=0)
            ax_re.fill_between(w, orig[:, i, i].real, color=color, alpha=0.3, lw=0)
            ax_im.plot(w, -fit[:, i, i].imag, color=color, lw=1.5)
            ax_re.plot(w, fit[:, i, i].real, color=color, lw=1.5)
        if n_orb > 1:
            ax_im.plot(
                w, _offdiag_norm(orig), color=_ORIG_OFFDIAG_COLOR, lw=1.2, ls="--"
            )
            ax_im.plot(w, _offdiag_norm(fit), color=_FIT_OFFDIAG_COLOR, lw=1.2, ls="--")
        ax_im.set_ylabel(r"$-\mathrm{Im}\,\Delta$")
        ax_re.set_ylabel(r"$\mathrm{Re}\,\Delta$")

        for (ax_im, ax_re), panel, names in zip(axes[1:], contributions, labels):
            for contribution, (_, style) in zip(panel, names):
                for i, color in enumerate(colors):
                    ax_im.plot(
                        w, -contribution[:, i, i].imag, color=color, lw=1.5, ls=style
                    )
                    ax_re.plot(
                        w, contribution[:, i, i].real, color=color, lw=1.5, ls=style
                    )
                if n_orb > 1:
                    ax_im.plot(
                        w,
                        _offdiag_norm(contribution),
                        color=_FIT_OFFDIAG_COLOR,
                        lw=1.2,
                        ls="--",
                    )
            label = "\n".join(name for name, _ in names)
            ax_im.text(
                0.02,
                0.9,
                label,
                transform=ax_im.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            ax_im.set_ylabel(r"$-\mathrm{Im}\,\Delta$")
            ax_re.set_ylabel(r"$\mathrm{Re}\,\Delta$")

        handles = [
            Patch(facecolor=colors[i], alpha=0.3, label=f"orb {orbs[i]} (RSPt)")
            for i in range(n_orb)
        ] + [
            Line2D([], [], color=colors[i], lw=1.5, label=f"orb {orbs[i]} (fit)")
            for i in range(n_orb)
        ]
        if n_orb > 1:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=_FIT_OFFDIAG_COLOR,
                    lw=1.2,
                    ls="--",
                    label=r"$\sum_{i \neq j} |\Delta_{ij}|$",
                )
            )
        chain_styles = {
            style: name.split("(")[-1].rstrip(")")
            for names in labels
            for name, style in names
            if "(" in name
        }
        for style, name in chain_styles.items():
            handles.append(Line2D([], [], color="black", lw=1.5, ls=style, label=name))
        axes[0, 0].legend(handles=handles, fontsize=7, loc="best")

        for ax in axes[-1]:
            ax.set_xlabel(r"$\omega$")
        for ax in axes.flat:
            ax.grid(color="0.9", lw=0.5)
            ax.set_axisbelow(True)
        _annotate_shift(fig, C)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        figs.append(fig)
    return figs

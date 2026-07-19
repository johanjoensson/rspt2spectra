"""
Visualize a hybridization fit and the contribution of each bath state.

`plot_hyb_fit` produces one figure per inequivalent block: a top panel with
the original hybridization function (filled) and the fitted one (curve,
including the constant shift C), followed by one panel per bath state showing
its contribution in the selected bath geometry. For the star geometry a bath
state is a fitted level; for chain-type geometries it is a chain site, and
the panel for distance i overlays the sites i steps from the impurity of the
occupied and unoccupied chains.

Chain-site contributions use eigenmode-site attribution: with
``H_bath = U diag(lam) U^dagger`` the fitted hybridization is a sum of poles
``w_n / (z - lam_n)`` with ``w_n = (v^dagger u_n)(u_n^dagger v)``, and site j
receives the fraction ``|u_n[j]|^2`` of pole n. The site contributions sum
exactly to the pole part of the fit and have non-negative spectral weight on
the diagonal.

The chain topology (which bath orbital belongs to which chain and how far it
sits from the impurity) is recovered from the bath Hamiltonian itself by
:func:`bath_topology`, so the same code handles every geometry
:func:`rspt2spectra.edchain.build_H_bath_v` can produce.
"""

from collections import namedtuple

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
# Color follows the bath entity for the per-site panels.
_KIND_COLORS = {
    "occ": "#2a78d6",  # occupied chain, blue
    "unocc": "#eb6834",  # unoccupied chain, orange
    "coupling": "#4a3aa7",  # coupling state, violet
    "chain": "#52514e",  # a single, sign-mixed chain, gray
    "spoke": "#52514e",  # a direct impurity spoke, gray
}
_KIND_LABELS = {
    "occ": "occupied chain",
    "unocc": "unoccupied chain",
    "coupling": "coupling state",
    "chain": "bath chain",
}
_ORIG_OFFDIAG_COLOR = "#b0afa9"
_FIT_OFFDIAG_COLOR = "#52514e"

_CHAIN_GEOMETRIES = ("chain", "single_chain", "linked_chain", "peeled_linked_chain")

# A bath entity groups the bath orbitals that share one panel curve.
# kind : one of the keys of _KIND_COLORS.
# key  : chain distance from the impurity (occ/unocc/coupling/chain) or, for a
#        spoke, its energy (used to order and label the per-spoke panels).
# indices : the bath-orbital indices carried by the entity.
BathEntity = namedtuple("BathEntity", ["kind", "key", "indices"])


def _bfs_depth(adjacency, seed):
    """Breadth-first distance (1-based) from the seeded nodes over ``adjacency``."""
    n = adjacency.shape[0]
    depth = np.full(n, -1, dtype=int)
    frontier = seed.copy()
    d = 1
    while np.any(frontier):
        depth[frontier] = d
        nxt = np.any(adjacency[frontier], axis=0) & (depth == -1)
        frontier = nxt
        d += 1
    return depth


def _connected_components(adjacency, mask):
    """Label the connected components of ``adjacency`` restricted to ``mask``."""
    n = adjacency.shape[0]
    component = np.full(n, -1, dtype=int)
    c = 0
    for start in range(n):
        if not mask[start] or component[start] != -1:
            continue
        stack = [start]
        component[start] = c
        while stack:
            u = stack.pop()
            for w in np.where(adjacency[u] & mask & (component == -1))[0]:
                component[w] = c
                stack.append(w)
        c += 1
    return component


def bath_topology(H_bath, v, n_orb, tol=1e-10):
    """Classify the bath orbitals of one block into chain entities.

    The bath orbitals are grouped into *sites*: blocks of ``n_orb`` consecutive
    orbitals when the bath size is a multiple of ``n_orb`` (every chain builder
    in :mod:`rspt2spectra.edchain` emits such sites), otherwise single orbitals
    (a fallback for the multi-orbital linked chains, whose bath size need not be
    a multiple of ``n_orb``). Each site becomes a :class:`BathEntity`:

    - a **spoke** if it has no bath-bath coupling (a direct impurity level);
    - the **coupling** site of a linked chain, i.e. a site that couples to the
      impurity and to at least two other impurity-coupled sites;
    - otherwise a chain site, tagged **occ** / **unocc** when its connected
      chain has a definite energy sign, or **chain** when the chain spans both
      signs (a single chain).

    Parameters
    ----------
    H_bath : (n_bath, n_bath) np.ndarray
        Bath Hamiltonian of one block, in any geometry.
    v : (n_bath, n_orb) np.ndarray
        Impurity-bath hopping of the same block.
    n_orb : int
        Number of impurity orbitals of the block.
    tol : float, default 1e-10
        Matrix elements with magnitude below tol are treated as zero.

    Returns
    -------
    entities : list of BathEntity
        Bath entities in display order: the coupling site, then spokes by
        ascending energy, then chain sites by ascending distance. Their index
        sets partition ``range(n_bath)``.
    """
    n_bath = H_bath.shape[0]
    energies = np.real(np.diag(H_bath))
    block_mode = n_orb > 0 and n_bath % n_orb == 0
    step = n_orb if block_mode else 1
    n_sites = n_bath // step

    def site_slice(i):
        return slice(i * step, (i + 1) * step)

    adjacency = np.zeros((n_sites, n_sites), dtype=bool)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if np.linalg.norm(H_bath[site_slice(i), site_slice(j)]) > tol:
                adjacency[i, j] = adjacency[j, i] = True
    couples = np.array([np.linalg.norm(v[site_slice(i)]) > tol for i in range(n_sites)])
    depth = _bfs_depth(adjacency, couples)
    site_energy = np.array([np.mean(energies[site_slice(i)]) for i in range(n_sites)])
    is_spoke = ~np.any(adjacency, axis=1)

    # The coupling site of a linked chain sits between the occupied and the
    # unoccupied chain: it couples to the impurity and to both chain heads.
    is_coupling = np.zeros(n_sites, dtype=bool)
    if block_mode:
        for i in range(n_sites):
            if depth[i] == 1 and not is_spoke[i]:
                neighbours = np.where(adjacency[i])[0]
                if np.sum(depth[neighbours] == 1) >= 2:
                    is_coupling[i] = True

    branch_mask = ~is_spoke & ~is_coupling
    component = _connected_components(adjacency, branch_mask)
    branch_kind = {}
    for c in np.unique(component[branch_mask]):
        e = site_energy[component == c]
        if np.all(e < 0):
            branch_kind[c] = "occ"
        elif np.all(e > 0):
            branch_kind[c] = "unocc"
        else:
            branch_kind[c] = "chain"

    sites = []
    for i in range(n_sites):
        indices = list(range(i * step, (i + 1) * step))
        if is_spoke[i]:
            sites.append(BathEntity("spoke", site_energy[i], indices))
        elif is_coupling[i]:
            sites.append(BathEntity("coupling", depth[i], indices))
        else:
            sites.append(BathEntity(branch_kind[component[i]], depth[i], indices))

    chain_sites = [s for s in sites if s.kind not in ("spoke", "coupling")]
    coupling_sites = [s for s in sites if s.kind == "coupling"]
    spokes = sorted((s for s in sites if s.kind == "spoke"), key=lambda s: s.key)
    chain_sites.sort(key=lambda s: s.key)
    return coupling_sites + spokes + chain_sites


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
    """Per-level (label, curves) panels for the star geometry.

    Each curve is ``(contribution, None)``; the ``None`` color makes the
    plotting loop fall back to one color per orbital.
    """
    order = np.argsort(ebs, kind="stable")
    panels = []
    for k in order:
        contribution = get_hyb_2(z, ebs[k : k + 1][np.newaxis], vs[k : k + 1][np.newaxis])[0]
        panels.append((f"$\\varepsilon_b = {ebs[k]: .3f}$", [(contribution, None)]))
    return panels


def _chain_panels(z, H_bath, v, n_orb):
    """Per-entity (label, curves) panels for a chain-type geometry.

    Returns the panels and the set of chain-entity kinds present (for the
    legend). Each curve is ``(contribution, color)`` with the color following
    the bath entity.
    """
    entities = bath_topology(H_bath, v, n_orb)
    contributions = site_resolved_hyb(z, H_bath, v, [e.indices for e in entities])
    indexed = list(zip(entities, contributions))

    panels = []
    present_kinds = []
    # The coupling state and the direct spokes sit closest to the impurity, so
    # they are shown before the chains that extend away from it.
    for e, hyb in indexed:
        if e.kind == "coupling":
            panels.append(("coupling state", [(hyb, _KIND_COLORS["coupling"])]))
            if "coupling" not in present_kinds:
                present_kinds.append("coupling")
    # Direct spokes: one panel each, labelled by energy.
    for e, hyb in indexed:
        if e.kind == "spoke":
            label = f"$\\varepsilon_b = {e.key: .3f}$"
            panels.append((label, [(hyb, _KIND_COLORS["spoke"])]))
    # Chain sites: one panel per distance, occupied and unoccupied overlaid.
    chain = [(e, hyb) for e, hyb in indexed if e.kind in ("occ", "unocc", "chain")]
    for distance in sorted({e.key for e, _ in chain}):
        curves = []
        for e, hyb in chain:
            if e.key == distance:
                curves.append((hyb, _KIND_COLORS[e.kind]))
                if e.kind not in present_kinds:
                    present_kinds.append(e.kind)
        panels.append((f"site {int(distance)}", curves))
    return panels, present_kinds


def _offdiag_norm(hyb_block):
    """Return the summed magnitude of the off-diagonal elements, (M,)."""
    mask = ~np.eye(hyb_block.shape[-1], dtype=bool)
    return np.sum(np.abs(hyb_block[:, mask]), axis=-1)


def _annotate_shift(fig, C):
    """Print the fitted constant shift C on the figure."""
    if C.shape == (1, 1):
        text = f"C = {np.real(C[0, 0]): .4f}"
    else:
        rows = ["  ".join(f"{np.real(x): .3f}{np.imag(x):+.3f}i" for x in row) for row in C]
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
            panels, present_kinds = _chain_panels(z, H_baths[b], vs_geom[b], n_orb)
        else:
            panels, present_kinds = _star_panels(z, ebs, vs), []

        total = C + np.sum([np.sum([c for c, _ in curves], axis=0) for _, curves in panels], axis=0)
        if not np.allclose(total, fit, atol=1e-6):
            print(
                f"WARNING: contributions of block {b} do not sum to the star "
                f"fit (max dev {np.max(np.abs(total - fit)):.3e}); the "
                f"{bath_geometry} geometry changed the hybridization."
            )

        figs.append(
            _draw_block_figure(
                plt,
                Line2D,
                Patch,
                w,
                orig,
                fit,
                panels,
                present_kinds,
                orbs,
                C,
                bath_geometry,
            )
        )
    return figs


def _draw_block_figure(plt, Line2D, Patch, w, orig, fit, panels, present_kinds, orbs, C, bath_geometry):
    """Render one block figure: top fit panel plus one panel per bath entity."""
    n_orb = len(orbs)
    colors = [_ORBITAL_COLORS[i % len(_ORBITAL_COLORS)] for i in range(n_orb)]
    n_rows = 1 + len(panels)
    fig, axes = plt.subplots(
        n_rows,
        2,
        sharex=True,
        figsize=(9, 1.4 * n_rows + 1.2),
        squeeze=False,
        gridspec_kw={"height_ratios": [2] + [1] * len(panels)},
    )
    fig.suptitle(f"Hybridization fit, orbitals {orbs} ({bath_geometry})")

    ax_im, ax_re = axes[0]
    for i, color in enumerate(colors):
        ax_im.fill_between(w, -orig[:, i, i].imag, color=color, alpha=0.3, lw=0)
        ax_re.fill_between(w, orig[:, i, i].real, color=color, alpha=0.3, lw=0)
        ax_im.plot(w, -fit[:, i, i].imag, color=color, lw=1.5)
        ax_re.plot(w, fit[:, i, i].real, color=color, lw=1.5)
    if n_orb > 1:
        ax_im.plot(w, _offdiag_norm(orig), color=_ORIG_OFFDIAG_COLOR, lw=1.2, ls="--")
        ax_im.plot(w, _offdiag_norm(fit), color=_FIT_OFFDIAG_COLOR, lw=1.2, ls="--")
    ax_im.set_ylabel(r"$-\mathrm{Im}\,\Delta$")
    ax_re.set_ylabel(r"$\mathrm{Re}\,\Delta$")

    for (ax_im, ax_re), (label, curves) in zip(axes[1:], panels):
        for contribution, curve_color in curves:
            for i, orbital_color in enumerate(colors):
                color = orbital_color if curve_color is None else curve_color
                ax_im.plot(w, -contribution[:, i, i].imag, color=color, lw=1.5)
                ax_re.plot(w, contribution[:, i, i].real, color=color, lw=1.5)
            if n_orb > 1:
                ax_im.plot(
                    w,
                    _offdiag_norm(contribution),
                    color=_FIT_OFFDIAG_COLOR,
                    lw=1.2,
                    ls="--",
                )
        ax_im.text(0.02, 0.9, label, transform=ax_im.transAxes, ha="left", va="top", fontsize=8)
        ax_im.set_ylabel(r"$-\mathrm{Im}\,\Delta$")
        ax_re.set_ylabel(r"$\mathrm{Re}\,\Delta$")

    handles = [Patch(facecolor=colors[i], alpha=0.3, label=f"orb {orbs[i]} (RSPt)") for i in range(n_orb)] + [
        Line2D([], [], color=colors[i], lw=1.5, label=f"orb {orbs[i]} (fit)") for i in range(n_orb)
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
    for kind in present_kinds:
        if kind in _KIND_LABELS:
            handles.append(Line2D([], [], color=_KIND_COLORS[kind], lw=1.5, label=_KIND_LABELS[kind]))
    axes[0, 0].legend(handles=handles, fontsize=7, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel(r"$\omega$")
    for ax in axes.flat:
        ax.grid(color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    _annotate_shift(fig, C)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig

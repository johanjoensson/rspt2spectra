"""Tests for the per-bath-state hybridization contribution plots.

Uses analytic star models with known poles (no stochastic fit), so the
site-resolved contributions can be checked exactly against the analytic
hybridization function.
"""

import numpy as np
import pytest

from rspt2spectra.edchain import build_H_bath_v
from rspt2spectra.h0 import flatten_star_levels, prepare_hyb_fit
from rspt2spectra.offdiagonal import get_hyb_2
from rspt2spectra.plot import _chain_panels, _star_panels, bath_distances

W = np.linspace(-4, 4, 401)
EIM = 0.1
Z = W + 1j * EIM


def one_orbital_model():
    """1-orbital analytic star model: 2 occupied + 2 unoccupied poles."""
    poles = [(-2.0, 0.6), (-1.0, 0.4), (0.5, 0.3), (1.5, 0.5)]
    H_loc = np.array([[0.1]], dtype=complex)
    hyb = np.zeros((len(Z), 1, 1), dtype=complex)
    for e, v in poles:
        hyb[:, 0, 0] += v**2 / (Z - e)
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(
        hyb, H_loc, verbose=False
    )
    ebs = np.array([e for e, _ in poles])
    vs = np.array([[[v * Q[0, 0]]] for _, v in poles], dtype=complex)
    ebs_flat, vs_flat = flatten_star_levels(ebs, vs)
    return phase_hyb, H_local_Q, block_structure, ebs_flat, vs_flat


def two_orbital_model():
    """2-orbital model, orbitals coupled through the local Hamiltonian."""
    H_loc = np.array([[-0.5, 0.2], [0.2, 0.3]], dtype=complex)
    poles = [
        (-2.0, np.array([0.5, 0.0], dtype=complex)),
        (-1.5, np.array([0.0, 0.4], dtype=complex)),
        (0.8, np.array([0.0, 0.2], dtype=complex)),
        (1.0, np.array([0.3, 0.0], dtype=complex)),
    ]
    hyb = np.zeros((len(Z), 2, 2), dtype=complex)
    for e, v in poles:
        hyb += np.conj(v[None, :, None]) * v[None, None, :] / (Z - e)[:, None, None]
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(
        hyb, H_loc, verbose=False
    )
    ebs = np.array([e for e, _ in poles])
    vs = np.array([v @ Q for _, v in poles]).reshape((len(poles), 1, 2))
    ebs_flat, vs_flat = flatten_star_levels(ebs, vs)
    return phase_hyb, H_local_Q, block_structure, ebs_flat, vs_flat


def block_hyb(z, H_bath, v):
    """Direct evaluation of v^dagger (z - H_bath)^-1 v."""
    G = np.linalg.inv(z[:, None, None] * np.eye(H_bath.shape[0])[None] - H_bath[None])
    return np.conj(v.T)[None] @ G @ v[None]


def geometry_block(geometry, model):
    _, H_local_Q, block_structure, ebs, vs = model()
    H_baths, vs_geom = build_H_bath_v(
        H_local_Q,
        [ebs],
        [vs],
        geometry,
        block_structure,
        verbose=False,
        extra_verbose=False,
    )
    return H_baths[0], vs_geom[0], ebs, vs


def test_bath_distances_star():
    H_bath, v, ebs, _ = geometry_block("star", one_orbital_model)
    depth, component = bath_distances(H_bath, v)
    # Every star state couples directly and is its own component.
    assert np.all(depth == 1)
    assert len(np.unique(component)) == len(ebs)


def test_bath_distances_double_chain():
    H_bath, v, _, _ = geometry_block("chain", one_orbital_model)
    depth, component = bath_distances(H_bath, v)
    # Occupied and unoccupied chains: two components of two sites each,
    # at distances 1 and 2 from the impurity.
    assert len(np.unique(component)) == 2
    for c in np.unique(component):
        assert sorted(depth[component == c]) == [1, 2]


def test_bath_distances_single_chain():
    H_bath, v, _, _ = geometry_block("single_chain", one_orbital_model)
    depth, component = bath_distances(H_bath, v)
    assert len(np.unique(component)) == 1
    assert sorted(depth) == [1, 2, 3, 4]


@pytest.mark.parametrize("geometry", ["chain", "single_chain", "linked_chain"])
@pytest.mark.parametrize("model", [one_orbital_model, two_orbital_model])
def test_chain_contributions_sum_to_full_hyb(geometry, model):
    H_bath, v, ebs, vs = geometry_block(geometry, model)
    contributions, labels = _chain_panels(Z, H_bath, v)
    assert len(contributions) == len(labels)
    total = np.sum([np.sum(panel, axis=0) for panel in contributions], axis=0)
    # The site contributions telescope exactly to the geometry's
    # hybridization, which in turn matches the star-model poles.
    assert np.allclose(total, block_hyb(Z, H_bath, v), atol=1e-10)
    assert np.allclose(
        total, get_hyb_2(Z, ebs[np.newaxis], vs[np.newaxis])[0], atol=1e-8
    )
    # Each site contribution carries non-negative spectral weight.
    for panel in contributions:
        for contribution in panel:
            diag = np.diagonal(contribution, axis1=1, axis2=2)
            assert np.all(-diag.imag > -1e-12)


def test_double_chain_panels_overlay_occ_and_unocc():
    H_bath, v, _, _ = geometry_block("chain", one_orbital_model)
    _, labels = _chain_panels(Z, H_bath, v)
    assert len(labels) == 2  # two distances
    for i, names in enumerate(labels):
        assert sorted(name for name, _ in names) == [
            f"site {i + 1} (occ)",
            f"site {i + 1} (unocc)",
        ]


def test_star_panels_sum_to_fit():
    _, _, _, ebs, vs = one_orbital_model()
    C = np.array([[0.25]], dtype=complex)
    contributions, _labels = _star_panels(Z, ebs, vs)
    assert len(contributions) == len(ebs)
    total = np.sum(contributions, axis=0) + C
    assert np.allclose(
        total, get_hyb_2(Z, ebs[np.newaxis], vs[np.newaxis], C)[0], atol=1e-12
    )


@pytest.mark.parametrize(
    "geometry, model, n_panels",
    [
        # Star: one panel per fitted level.
        ("star", two_orbital_model, 4),
        # Double chain, 1 orbital: two sites per chain, overlaid per distance.
        ("chain", one_orbital_model, 2),
        # Double chain, 2 orbitals: each chain is a single 2-orbital site.
        ("chain", two_orbital_model, 1),
    ],
)
def test_plot_hyb_fit_layout(geometry, model, n_panels):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from rspt2spectra.plot import plot_hyb_fit  # noqa: PLC0415

    phase_hyb, H_local_Q, block_structure, ebs, vs = model()
    n_orb = phase_hyb.shape[-1]
    C = np.zeros((n_orb, n_orb), dtype=complex)
    figs = plot_hyb_fit(
        W,
        EIM,
        phase_hyb,
        [ebs],
        [vs],
        [C],
        H_local_Q,
        block_structure,
        geometry,
    )
    assert len(figs) == 1
    # Top panel plus one row per bath state/site, two columns each.
    assert len(figs[0].axes) == 2 * (1 + n_panels)
    plt.close("all")

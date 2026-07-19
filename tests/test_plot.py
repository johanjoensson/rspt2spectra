"""Tests for the per-bath-state hybridization contribution plots.

Uses analytic star models with known poles (no stochastic fit), so the
site-resolved contributions can be checked exactly against the analytic
hybridization function and the recovered chain topology is deterministic.
"""

import numpy as np
import pytest

from rspt2spectra.edchain import build_H_bath_v
from rspt2spectra.h0 import flatten_star_levels, prepare_hyb_fit
from rspt2spectra.offdiagonal import get_hyb_2
from rspt2spectra.plot import (
    _chain_panels,
    _star_panels,
    bath_topology,
    site_resolved_hyb,
)

W = np.linspace(-4, 4, 401)
EIM = 0.1
Z = W + 1j * EIM


def one_orbital_model():
    """1-orbital analytic star model: 3 occupied + 3 unoccupied poles."""
    poles = [(-2.5, 0.6), (-1.5, 0.5), (-0.7, 0.4), (0.4, 0.3), (1.0, 0.45), (1.8, 0.5)]
    H_loc = np.array([[0.1]], dtype=complex)
    hyb = np.zeros((len(Z), 1, 1), dtype=complex)
    for e, v in poles:
        hyb[:, 0, 0] += v**2 / (Z - e)
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(hyb, H_loc, verbose=False)
    ebs = np.array([e for e, _ in poles])
    vs = np.array([[[v * Q[0, 0]]] for _, v in poles], dtype=complex)
    ebs_flat, vs_flat = flatten_star_levels(ebs, vs)
    return phase_hyb, H_local_Q, block_structure, ebs_flat, vs_flat


def two_orbital_model():
    """2-orbital model, orbitals coupled through the local Hamiltonian."""
    H_loc = np.array([[-0.5, 0.2], [0.2, 0.3]], dtype=complex)
    poles = [
        (-2.5, np.array([0.5, 0.1], dtype=complex)),
        (-1.5, np.array([0.1, 0.4], dtype=complex)),
        (-0.7, np.array([0.3, 0.2], dtype=complex)),
        (0.4, np.array([0.2, 0.3], dtype=complex)),
        (1.0, np.array([0.4, 0.1], dtype=complex)),
        (1.8, np.array([0.15, 0.35], dtype=complex)),
    ]
    hyb = np.zeros((len(Z), 2, 2), dtype=complex)
    for e, v in poles:
        hyb += np.conj(v[None, :, None]) * v[None, None, :] / (Z - e)[:, None, None]
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(hyb, H_loc, verbose=False)
    ebs = np.array([e for e, _ in poles])
    vs = np.array([v @ Q for _, v in poles]).reshape((len(poles), 1, 2))
    ebs_flat, vs_flat = flatten_star_levels(ebs, vs)
    return phase_hyb, H_local_Q, block_structure, ebs_flat, vs_flat


def peel_model():
    """1-orbital model with one dominant sharp mode on a smooth background.

    The dominant mode is peeled off as a direct spoke while the background
    forms a linked double chain, so the peeled geometry has chain sites, a
    coupling site and a spoke at once.
    """
    poles = [
        (-2.2, 0.35),
        (-1.3, 0.3),
        (-0.5, 0.3),
        (0.2, 1.8),
        (0.9, 0.3),
        (1.7, 0.35),
        (2.4, 0.3),
    ]
    H_loc = np.array([[0.05]], dtype=complex)
    hyb = np.zeros((len(Z), 1, 1), dtype=complex)
    for e, v in poles:
        hyb[:, 0, 0] += v**2 / (Z - e)
    Q, phase_hyb, H_local_Q, block_structure = prepare_hyb_fit(hyb, H_loc, verbose=False)
    ebs = np.array([e for e, _ in poles])
    vs = np.array([[[v * Q[0, 0]]] for _, v in poles], dtype=complex)
    ebs_flat, vs_flat = flatten_star_levels(ebs, vs)
    return phase_hyb, H_local_Q, block_structure, ebs_flat, vs_flat


def block_hyb(z, H_bath, v):
    """Direct evaluation of v^dagger (z - H_bath)^-1 v."""
    G = np.linalg.inv(z[:, None, None] * np.eye(H_bath.shape[0])[None] - H_bath[None])
    return np.conj(v.T)[None] @ G @ v[None]


def geometry_block(geometry, model, peel_weight=0.3):
    _, H_local_Q, block_structure, ebs, vs = model()
    n_orb = H_local_Q.shape[0]
    H_baths, vs_geom = build_H_bath_v(
        H_local_Q,
        [ebs],
        [vs],
        geometry,
        block_structure,
        verbose=False,
        extra_verbose=False,
        peel_weight=peel_weight,
    )
    return H_baths[0], vs_geom[0], ebs, vs, n_orb


def kinds_at(entities, kind):
    """Distances of the entities of a given kind, sorted."""
    return sorted(int(e.key) for e in entities if e.kind == kind)


def test_topology_star_is_all_spokes():
    # A purely diagonal bath (star form) couples every orbital directly.
    H_bath = np.diag([-2.0, -1.0, 0.5, 1.5]).astype(complex)
    v = np.ones((4, 1), dtype=complex)
    entities = bath_topology(H_bath, v, 1)
    assert [e.kind for e in entities] == ["spoke"] * 4
    # Spokes are ordered by energy and labelled by it.
    assert [round(float(e.key), 1) for e in entities] == [-2.0, -1.0, 0.5, 1.5]


def test_topology_double_chain():
    H_bath, v, _, _, n_orb = geometry_block("chain", one_orbital_model)
    entities = bath_topology(H_bath, v, n_orb)
    # Occupied and unoccupied chains, three sites each, no coupling or spokes.
    assert kinds_at(entities, "occ") == [1, 2, 3]
    assert kinds_at(entities, "unocc") == [1, 2, 3]
    assert not any(e.kind in ("coupling", "spoke") for e in entities)


def test_topology_single_chain():
    H_bath, v, _, _, n_orb = geometry_block("single_chain", one_orbital_model)
    entities = bath_topology(H_bath, v, n_orb)
    # A single, sign-mixed chain: one gray entity per site, no occ/unocc split.
    assert all(e.kind == "chain" for e in entities)
    assert kinds_at(entities, "chain") == [1, 2, 3, 4, 5, 6]


def test_topology_linked_chain():
    H_bath, v, _, _, n_orb = geometry_block("linked_chain", one_orbital_model)
    entities = bath_topology(H_bath, v, n_orb)
    # A linked chain has exactly one coupling site between two labelled chains.
    assert sum(e.kind == "coupling" for e in entities) == 1
    assert kinds_at(entities, "occ")
    assert kinds_at(entities, "unocc")


def test_topology_peeled_linked_chain():
    H_bath, v, _, _, n_orb = geometry_block("peeled_linked_chain", peel_model)
    entities = bath_topology(H_bath, v, n_orb)
    # The sharp mode is peeled off as a spoke; the rest forms a linked chain.
    assert sum(e.kind == "spoke" for e in entities) == 1
    assert sum(e.kind == "coupling" for e in entities) == 1
    assert kinds_at(entities, "occ")
    assert kinds_at(entities, "unocc")
    # The index sets of all entities partition the bath.
    covered = sorted(i for e in entities for i in e.indices)
    assert covered == list(range(H_bath.shape[0]))


@pytest.mark.parametrize(
    "geometry",
    ["chain", "single_chain", "linked_chain", "peeled_linked_chain"],
)
@pytest.mark.parametrize("model", [one_orbital_model, two_orbital_model])
def test_chain_contributions_sum_to_full_hyb(geometry, model):
    H_bath, v, ebs, vs, n_orb = geometry_block(geometry, model)
    entities = bath_topology(H_bath, v, n_orb)
    contributions = site_resolved_hyb(Z, H_bath, v, [e.indices for e in entities])
    total = np.sum(contributions, axis=0)
    # The site contributions telescope exactly to the geometry's
    # hybridization, which in turn matches the star-model poles.
    assert np.allclose(total, block_hyb(Z, H_bath, v), atol=1e-10)
    assert np.allclose(total, get_hyb_2(Z, ebs[np.newaxis], vs[np.newaxis])[0], atol=1e-8)
    # Each entity contribution carries non-negative spectral weight.
    for contribution in contributions:
        diag = np.diagonal(contribution, axis1=1, axis2=2)
        assert np.all(-diag.imag > -1e-12)


@pytest.mark.parametrize("geometry", ["chain", "single_chain", "linked_chain", "peeled_linked_chain"])
def test_chain_panels_sum_to_full_hyb(geometry):
    H_bath, v, _, _, n_orb = geometry_block(geometry, one_orbital_model)
    panels, _kinds = _chain_panels(Z, H_bath, v, n_orb)
    total = np.sum([np.sum([c for c, _ in curves], axis=0) for _, curves in panels], axis=0)
    assert np.allclose(total, block_hyb(Z, H_bath, v), atol=1e-10)


def test_panel_labels_are_unique_and_single_line():
    # A linked chain exercises site panels, a coupling panel and (via peel)
    # spoke panels; every label must be a single, distinct line.
    for model, geometry in [
        (one_orbital_model, "linked_chain"),
        (peel_model, "peeled_linked_chain"),
    ]:
        H_bath, v, _, _, n_orb = geometry_block(geometry, model)
        panels, _kinds = _chain_panels(Z, H_bath, v, n_orb)
        labels = [label for label, _ in panels]
        assert all("\n" not in label for label in labels)
        assert len(labels) == len(set(labels))

    # The double chain overlays occupied and unoccupied sites in one panel.
    H_bath, v, _, _, n_orb = geometry_block("chain", one_orbital_model)
    panels, kinds = _chain_panels(Z, H_bath, v, n_orb)
    assert [label for label, _ in panels] == ["site 1", "site 2", "site 3"]
    assert set(kinds) == {"occ", "unocc"}
    for _label, curves in panels:
        assert len(curves) == 2  # occupied and unoccupied overlaid


def test_linked_chain_has_coupling_panel():
    H_bath, v, _, _, n_orb = geometry_block("linked_chain", one_orbital_model)
    panels, kinds = _chain_panels(Z, H_bath, v, n_orb)
    labels = [label for label, _ in panels]
    assert "coupling state" in labels
    assert "coupling" in kinds


def test_star_panels_sum_to_fit():
    _, _, _, ebs, vs = one_orbital_model()
    C = np.array([[0.25]], dtype=complex)
    panels = _star_panels(Z, ebs, vs)
    assert len(panels) == len(ebs)
    # Star panels carry one orbital-coloured curve each (color sentinel None).
    for _label, curves in panels:
        assert len(curves) == 1
        assert curves[0][1] is None
    total = C + np.sum([curves[0][0] for _, curves in panels], axis=0)
    assert np.allclose(total, get_hyb_2(Z, ebs[np.newaxis], vs[np.newaxis], C)[0], atol=1e-12)


@pytest.mark.parametrize(
    "geometry, model, n_panels",
    [
        # Star: one panel per fitted level.
        ("star", two_orbital_model, 6),
        # Double chain, 1 orbital: three sites, occ + unocc overlaid per site.
        ("chain", one_orbital_model, 3),
        # Double chain, 2 orbitals: two 2-orbital sites per chain.
        ("chain", two_orbital_model, 2),
        # Linked chain: three site panels plus the coupling panel.
        ("linked_chain", one_orbital_model, 4),
        # Peeled: three site panels, the coupling panel and one spoke panel.
        ("peeled_linked_chain", peel_model, 5),
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
        peel_weight=0.3,
    )
    assert len(figs) == 1
    # Top panel plus one row per bath state/site, two columns each.
    assert len(figs[0].axes) == 2 * (1 + n_panels)
    plt.close("all")

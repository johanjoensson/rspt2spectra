"""A core shell with no hybridization must not enlarge the bath pool of the valence shell."""

import numpy as np
from types import SimpleNamespace

from rspt2spectra.hyb_fit import HYB_WEIGHT_REL_TOL, get_state_per_inequivalent_block, hybridization_weight


def _structure(n_blocks):
    blocks = [[i] for i in range(n_blocks)]
    empty = [[] for _ in blocks]
    return SimpleNamespace(
        blocks=blocks,
        identical_blocks=[[i] for i in range(n_blocks)],
        transposed_blocks=empty,
        particle_hole_blocks=empty,
        particle_hole_transposed_blocks=empty,
        inequivalent_blocks=list(range(n_blocks)),
    )


def _hyb(weights, w):
    hyb = np.zeros((len(w), len(weights), len(weights)), complex)
    for i, weight in enumerate(weights):
        hyb[:, i, i] = -0.5j * weight
    return hyb


def test_a_block_with_negligible_weight_gets_nothing_and_does_not_enlarge_the_pool():
    w = np.linspace(-5, 5, 201)
    bs = _structure(3)
    states = get_state_per_inequivalent_block(bs, 4, _hyb([1.0, 1.0, 1e-9], w), w, np.ones_like, delta=0.1)
    old = get_state_per_inequivalent_block(bs, 4, _hyb([1.0, 1.0, 1e-9], w), w, np.ones_like, delta=0.1, rel_tol=0)
    assert states[2] == 0
    assert states[0] + states[1] == 8
    assert old[2] >= 1, "rel_tol=0 is today's behaviour: any positive weight is covered"
    assert states[0] + states[1] < old.sum() + 1 and HYB_WEIGHT_REL_TOL > 0


def test_equal_blocks_are_unchanged():
    w = np.linspace(-5, 5, 201)
    bs = _structure(2)
    states = get_state_per_inequivalent_block(bs, 3, _hyb([1.0, 1.0], w), w, np.ones_like, delta=0.1)
    np.testing.assert_array_equal(states, [3, 3])


def test_a_core_shell_at_two_percent_gets_no_bath_at_the_default_tolerance():
    """Ni 2p next to 3d in NiO: about 2% of the valence weight in the fit window."""
    w = np.linspace(-5, 5, 201)
    bs = _structure(3)
    hyb = _hyb([1.0, 0.8, 0.02], w)
    states = get_state_per_inequivalent_block(bs, 5, hyb, w, np.ones_like, delta=0.1)
    assert states[2] == 0
    assert states[0] + states[1] == 10, "the pool is nBaths times the hybridizing blocks only"
    loose = get_state_per_inequivalent_block(bs, 5, hyb, w, np.ones_like, delta=0.1, rel_tol=1e-3)
    assert loose[2] >= 1


def test_hybridization_weight_is_what_the_allocation_compares():
    w = np.linspace(-5, 5, 201)
    hyb = _hyb([1.0, 3.0], w)
    assert np.isclose(hybridization_weight(hyb, w), 0.5 * 4.0 * 10.0)
    assert np.isclose(hybridization_weight(hyb, w, lambda x: np.where(x < 0, 1.0, 0.0)), 0.5 * 4.0 * 5.0, rtol=2e-2)

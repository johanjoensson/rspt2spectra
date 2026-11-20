"""Tests for warm-starting a hybridization fit from a previous bath."""

import numpy as np
import pytest

from rspt2spectra.bath_optimizer import get_hyb
from rspt2spectra.block_structure import build_block_structure
from rspt2spectra.hyb_fit import fit_block, fit_hyb
from rspt2spectra.symmetries import trivial_symmetry

W = np.linspace(-5, 5, 351)
DELTA = 0.1
Z = W + 1j * DELTA
SYM = trivial_symmetry(1)


def _three_peaks(s):
    """Three peaks fitted by two bath states: which pair to merge is a near tie.

    ``s`` shifts weight from the peak nearest E_F to the outermost one. Around
    s = 1.0 the two-state global minimum flips from ``[-1.20, -0.40]`` to
    ``[-2.00, -1.20]`` while the two costs agree to ~0.1% -- a near-degenerate pair
    of basins like the ones that make a cold refit jump between DMFT iterations.
    """
    eb = np.array([-2.0, -1.2, -0.4])
    v = np.array([0.7 * s, 0.7, 0.7 / s]).reshape(-1, 1)
    return get_hyb(Z, eb, v)


def _fit(s, bath_guess=None, **kwargs):
    info = {}
    eb, _, _ = fit_block(
        _three_peaks(s),
        W,
        DELTA,
        2,
        gamma=0.0,
        sym=SYM,
        comm=None,
        verbose=False,
        weight_fun=np.ones_like,
        bath_guess=bath_guess,
        warm_start_info=info,
        **kwargs,
    )
    return eb, info


@pytest.fixture(scope="module")
def previous():
    eb, info = _fit(0.998)
    assert info["status"] == "cold"
    return eb


def test_cold_refit_jumps_basin_across_a_near_tie(previous):
    # The failure mode the warm start exists for: a tiny change of the hybridization
    # sends a cold fit to a different, almost equally good bath.
    eb, _ = _fit(1.002)
    assert np.max(np.abs(np.sort(eb) - np.sort(previous))) > 0.5


def test_warm_refit_holds_the_basin_across_a_near_tie(previous):
    eb, info = _fit(1.002, bath_guess=previous)
    assert info["status"] == "held"
    np.testing.assert_allclose(np.sort(eb), np.sort(previous), atol=1e-12)
    # What was given up is within the margin.
    assert info["cost held"] <= min(info["cost refined"], info["cost global"]) / (
        1 - 0.05
    )


def test_a_clearly_better_minimum_replaces_the_warm_fit(previous):
    cold, _ = _fit(1.3)
    eb, info = _fit(1.3, bath_guess=previous)
    assert info["status"] == "replaced"
    assert info["cost global"] < 0.95 * min(info["cost held"], info["cost refined"])
    np.testing.assert_allclose(np.sort(eb), np.sort(cold), atol=1e-6)


@pytest.mark.parametrize(
    ("tol", "s", "status"), [(0.0, 1.002, "replaced"), (1.0, 1.3, "held")]
)
def test_the_margin_decides(previous, tol, s, status):
    # tol=0: the cheapest candidate wins. tol=1: the previous energies are never moved.
    _, info = _fit(s, bath_guess=previous, warm_accept_tol=tol)
    assert info["status"] == status


def test_held_keeps_the_energies_where_the_refinement_would_move_them(previous):
    # A flat valley: the refined minimum drifts with the hybridization, the margin holds the bath.
    eb_held, info = _fit(1.02, bath_guess=previous, warm_accept_tol=0.5)
    assert info["status"] == "held"
    np.testing.assert_allclose(np.sort(eb_held), np.sort(previous), atol=1e-12)
    eb_moved, info_moved = _fit(1.02, bath_guess=previous, warm_accept_tol=0.0)
    assert info_moved["cost held"] == pytest.approx(info["cost held"])
    # The held cost is the cost of the same objective the other candidates are priced with.
    assert info["cost held"] >= min(info["cost refined"], info["cost global"]) - 1e-12


def test_a_guess_with_another_state_count_only_seeds_the_global_fit(previous):
    cold, _ = _fit(1.002)
    eb, info = _fit(1.002, bath_guess=previous[:1])
    assert info["status"] == "state count changed"
    assert "cost warm" not in info
    np.testing.assert_allclose(np.sort(eb), np.sort(cold), atol=1e-6)


def test_fit_hyb_reports_one_outcome_per_inequivalent_block(previous):
    hyb = _three_peaks(1.002)
    bs = build_block_structure(hyb, mat=np.zeros(hyb.shape[1:]), tol=1e-6, w=W)
    info = []
    ebs, _, _ = fit_hyb(
        W,
        DELTA,
        hyb,
        2,
        bs,
        gamma=0.0,
        verbose=False,
        ebs_guess=[previous],
        warm_start_info=info,
    )
    assert len(info) == len(bs.inequivalent_blocks) == 1
    assert info[0]["status"] == "held"
    np.testing.assert_allclose(np.sort(ebs[0]), np.sort(previous), atol=1e-12)

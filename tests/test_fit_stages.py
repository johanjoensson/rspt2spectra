"""The stages of the h0 driver a caller composes itself: basis, fit, assemble."""

import numpy as np
import pytest

from rspt2spectra.fit_options import FitOptions
from rspt2spectra.h0 import bath_hybridization, fit_bath, fit_fingerprint, fit_window, prepare_hyb_fit


def _system(coupling):
    """Two weakly coupled pairs of orbitals; ``coupling`` joins them."""
    w = np.linspace(-3, 3, 121)
    hyb = np.zeros((len(w), 4, 4), dtype=complex)
    for i, width in enumerate([0.30, 0.35, 0.10, 0.12]):
        hyb[:, i, i] = 0.1 / (w - 0.5 * i + 1j * width)
    hyb[:, 1, 2] = hyb[:, 2, 1] = coupling * hyb[:, 1, 1]
    H = np.diag([0.0, 0.2, 1.0, 1.4]).astype(complex)
    H[0, 1] = H[1, 0] = 0.15
    H[2, 3] = H[3, 2] = 0.15
    H[1, 2] = H[2, 1] = coupling
    return w, hyb, H


def test_neglected_couplings_is_set_only_when_the_tolerance_removed_something():
    w, hyb, H = _system(1e-3)
    assert not prepare_hyb_fit(hyb, H, tol=1e-6, verbose=False, w=w).neglected_couplings
    assert prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w).neglected_couplings


def test_a_raised_tolerance_that_removes_nothing_is_not_reported_as_neglecting():
    w, hyb, H = _system(0.3)
    basis = prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w)
    assert len(basis.block_structure.blocks) == 1
    assert not basis.neglected_couplings


def test_the_fingerprint_follows_the_inputs_and_not_the_run():
    w, hyb, H = _system(1e-3)
    a = prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w, x_lim=(-2, 2))
    b = prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w, x_lim=(-2, 2))
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != prepare_hyb_fit(hyb * 1.0001, H, tol=1e-2, verbose=False, w=w, x_lim=(-2, 2)).fingerprint
    assert a.fingerprint != prepare_hyb_fit(hyb, H, tol=2e-2, verbose=False, w=w, x_lim=(-2, 2)).fingerprint
    assert a.fingerprint != prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w, x_lim=(-1, 2)).fingerprint


def test_the_fingerprint_ignores_the_coulomb_tensor_when_the_basis_does():
    _, hyb, H = _system(1e-3)
    u4 = np.zeros((4,) * 4, dtype=complex)
    u4[0, 1, 0, 1] = 1.0
    off = fit_fingerprint(hyb, H, u4, None, 1e-6, np.inf, None)
    assert off == fit_fingerprint(hyb, H, None, None, 1e-6, np.inf, None)
    assert off != fit_fingerprint(hyb, H, u4, None, 1e-6, 2.0, None)


def test_the_window_runs_over_the_support_and_stops_at_the_fermi_level_for_a_valence_bath():
    w = np.linspace(-3, 3, 601)
    hyb = np.zeros((len(w), 1, 1), dtype=complex)
    hyb[(w > -2) & (w < 1.5), 0, 0] = 0.1
    lo, hi = fit_window(w, hyb, valence_bath_only=False)
    assert lo == pytest.approx(-1.99, abs=0.02) and hi == pytest.approx(1.49, abs=0.02)
    assert fit_window(w, hyb, valence_bath_only=True)[1] == 0


def test_bath_hybridization_of_one_level():
    w = np.linspace(-1, 1, 11)
    got = bath_hybridization(w, 0.1, np.array([[0.5]]), np.array([[0.3 + 0.1j]]))
    expected = abs(0.3 + 0.1j) ** 2 / (w + 0.1j - 0.5)
    assert np.allclose(got[:, 0, 0], expected)


def test_fit_bath_returns_a_stored_fit_sorted_and_flat_without_fitting(monkeypatch):
    w, hyb, H = _system(1e-3)
    basis = prepare_hyb_fit(hyb, H, tol=1e-2, verbose=False, w=w)
    blocks = basis.block_structure.inequivalent_blocks
    sizes = [len(basis.block_structure.blocks[b]) for b in blocks]
    stored = (
        [np.array([0.5, -0.5]) for _ in blocks],
        [np.array([[[0.1] * n], [[0.2] * n]], dtype=complex) for n in sizes],
        [np.zeros((n, n), dtype=complex) for n in sizes],
    )
    monkeypatch.setattr("rspt2spectra.h0.fit_hyb", lambda *a, **k: pytest.fail("fitted despite a stored fit"))
    fit = fit_bath(basis, w, 0.05, 1, FitOptions(), stored_fit=stored, verbose=False)
    assert fit.warm_start_info is None
    for eb, vb, n in zip(fit.ebs_star, fit.vs_star, sizes):
        assert list(eb) == [-0.5, 0.5]  # sorted
        assert vb.shape == (2, 1, n)
        assert np.allclose(vb[0], 0.2) and np.allclose(vb[1], 0.1)  # followed the energies

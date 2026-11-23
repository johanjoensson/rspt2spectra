"""A bath fit must not depend on the orbital basis it is carried out in.

Fitting ``U^dagger Delta U`` instead of ``Delta`` is the same physics in other coordinates, so
the fitted model, rotated back, must be the same: same cost, same pole energies, same singular
values of the hoppings and the same model hybridization.  The fitting basis is chosen from the
data and can reorder near-degenerate orbitals from one DMFT iteration to the next; a fit that is
not covariant turns every such relabelling into a different impurity model and keeps the DMFT
loop from converging (a LaMnO3 run alternated between two models whose fifth hopping was 0.04
and 0.0003).
"""

import numpy as np
import pytest

from rspt2spectra.bath_optimizer import get_v_and_eb_varpro_held
from rspt2spectra.hyb_fit import fit_block
from rspt2spectra.symmetries import trivial_symmetry

N_IMP = 5
W = np.linspace(-2.0, 0.0, 401)
HELD = -0.4
DELTA = 0.05
Z = W + 1j * DELTA
SYM = trivial_symmetry(N_IMP)


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _hybridization():
    """A strong pole with one weakly coupled direction, plus a small second pole.

    One bath level fits it with a residue whose smallest eigenvalue is ~0.0015 against ~0.03:
    the shape of the LaMnO3 fit, where the weak direction's hopping (~0.04) was kept in some
    bases and lost (~0.002, at a 10% higher cost) in others.
    """
    rng = np.random.default_rng(3)
    hyb = np.zeros((len(Z), N_IMP, N_IMP), dtype=complex)
    for e, scale in (
        (HELD, (0.03, 0.02, 0.006, 0.004, 0.0015)),
        (-1.2, (0.002,) * N_IMP),
    ):
        U = _random_unitary(rng, N_IMP)
        A = U @ np.diag(scale) @ np.conj(U.T)
        hyb += A[None] / (Z - e)[:, None, None]
    return hyb


def _frames():
    rng = np.random.default_rng(11)
    perm = np.eye(N_IMP)[np.roll(np.arange(N_IMP), 1)].astype(complex)
    return [
        ("identity", np.eye(N_IMP, dtype=complex)),
        ("permutation", perm),
        ("unitary", _random_unitary(rng, N_IMP)),
    ]


def _model(eb, v, C):
    """The fitted hybridization ``C + sum_k V_k^dagger V_k / (z - eb_k)``."""
    A = np.conj(np.swapaxes(v, -1, -2)) @ v
    return C[None] + np.einsum("mk, kij -> mij", 1.0 / (Z[:, None] - eb[None, :]), A)


def _in_frame(hyb, U):
    return np.conj(U.T)[None] @ hyb @ U[None]


def _invariants(eb, v, C, cost, U):
    sv = np.linalg.svd(v.reshape(-1, v.shape[-1]), compute_uv=False)
    model = (
        U[None] @ _model(eb, v, C) @ np.conj(U.T)[None]
    )  # back in the reference basis
    return np.sort(eb), sv, model, cost


def _assert_same(results):
    (eb0, sv0, model0, cost0), *rest = results
    for eb, sv, model, cost in rest:
        assert cost == pytest.approx(cost0, rel=1e-6)
        assert np.allclose(eb, eb0, atol=1e-6)
        assert np.allclose(sv, sv0, atol=1e-5)
        assert np.max(np.abs(model - model0)) < 1e-5 * np.max(np.abs(model0))


def test_held_energies_give_the_same_hoppings_in_every_basis():
    hyb = _hybridization()
    results = []
    for _, U in _frames():
        v, eb, C, cost = get_v_and_eb_varpro_held(
            W,
            DELTA,
            _in_frame(hyb, U),
            np.array([HELD]),
            [(W[0], W[-1])],
            0.0,
            None,
            np.ones_like,
            SYM,
        )
        results.append(_invariants(eb, v, C, cost, U))
    _assert_same(results)


def test_the_full_fit_is_the_same_in_every_basis():
    hyb = _hybridization()
    results = []
    for _, U in _frames():
        eb, v, C = fit_block(
            _in_frame(hyb, U),
            W,
            DELTA,
            1,
            gamma=0.0,
            sym=SYM,
            comm=None,
            verbose=False,
            weight_fun=np.ones_like,
        )
        results.append(_invariants(eb, v, C, 0.0, U))
    _assert_same(results)

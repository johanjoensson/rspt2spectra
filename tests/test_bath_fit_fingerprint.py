"""A stored star fit is reused only for the same fit request."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rspt2spectra import FitOptions
from rspt2spectra.h0 import bath_fit_fingerprint

W = np.linspace(-2, 2, 21)
BASIS = SimpleNamespace(fingerprint="abc")


def _key(w=W, eim=0.1, n_baths=3, options=FitOptions(), basis=BASIS):
    return bath_fit_fingerprint(basis, w, eim, n_baths, options)


def test_same_request_same_key():
    assert _key() == _key()


@pytest.mark.parametrize(
    "changed",
    [
        {"n_baths": 4},
        {"eim": 0.2},
        {"w": W + 1e-3},
        {"basis": SimpleNamespace(fingerprint="abd")},
        {"options": replace(FitOptions(), gamma=0.02)},
        {"options": replace(FitOptions(), weight_function="unit", weight=3.0)},
        {"options": replace(FitOptions(), weight_w0=0.5)},
        {"options": replace(FitOptions(), fit_unocc=True)},
    ],
)
def test_a_changed_fit_input_changes_the_key(changed):
    assert _key(**changed) != _key()


@pytest.mark.parametrize("changed", [{"warm_start": False}, {"freeze_bath_energies": True}, {"bath_geometry": "star"}])
def test_how_the_fit_is_started_or_arranged_does_not(changed):
    assert _key(options=replace(FitOptions(), **changed)) == _key()

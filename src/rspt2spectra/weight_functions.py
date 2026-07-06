r"""
Weight functions for the hybridization fit.

Each factory takes a center :math:`w_0` and a decay parameter :math:`c` and
returns a callable ``f(w)`` used to weight the fit residual as a function of
energy. The available factories are collected in the `weight_functions`
dict, keyed by the names accepted by ``build_h0 --weight-function``.
"""

import numpy as np


def unit(_, __):
    """Return a weight function that is one everywhere."""
    return np.ones_like


def exponential(w0, c):
    r"""Return :math:`\exp(-c\,|w - w_0|)`."""
    return lambda w: np.exp(-c * np.abs(w - w0))


def gaussian(w0, c):
    r"""Return :math:`\exp(-c\,|w - w_0|^2 / 2)`."""
    return lambda w: np.exp(-c / 2 * np.abs(w - w0) ** 2)


def sqrtgauss(w0, c):
    r"""Return :math:`\sqrt{|w - w_0|}\,\exp(-c\,|w - w_0|^2/2)`, peak-normalized."""
    return lambda w: np.sqrt(np.abs(w - w0)) * np.exp(-c / 2 * np.abs(w - w0) ** 2) * (2 * c * np.e) ** (1 / 4)


def lingauss(w0, c):
    r"""Return :math:`|w - w_0|\,\exp(-c\,|w - w_0|^2/2)`, peak-normalized."""
    return lambda w: np.abs(w - w0) * np.exp(-c / 2 * (w - w0) ** 2) * np.sqrt(c * np.e)


def quadgauss(w0, c):
    r"""Return :math:`(w - w_0)^2\,\exp(-c\,|w - w_0|^2/2)`, peak-normalized."""
    return lambda w: ((w - w0) ** 2) * np.exp(-c / 2 * (w - w0) ** 2) * (c * np.e) / 2


def step(w0, _):
    r"""Return a step: 1 if :math:`w < w_0`, 0.5 at :math:`w_0`, 0 above."""
    return lambda w: 1 - np.heaviside(w - w0, 0.5)


weight_functions = dict(
    unit=unit,
    exponential=exponential,
    gaussian=gaussian,
    sqrtgauss=sqrtgauss,
    lingauss=lingauss,
    quadgauss=quadgauss,
    step=step,
)
"""Mapping from weight-function name to factory, as used by ``build_h0``."""

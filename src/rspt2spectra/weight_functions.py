import numpy as np


def unit(_, __):
    """
    Return all ones.
    """
    return lambda w: np.ones_like(w)


def exponential(w0, c):
    r"""
    Return $ exp(-e|w - w_0|) $
    """
    return lambda w: np.exp(-c * np.abs(w - w0))


def gaussian(w0, c):
    r"""
    Return $exp(-\frac{e|w -w_0|^2}{2})$
    """
    return lambda w: np.exp(-c / 2 * np.abs(w - w0) ** 2)


# This should be scaled so that the maximum value is 1.0
# However, I don't have the energy to calculate what the max acually is...
def rspt(w0, c):
    r"""
    Return $\frac{|w-w_0|}{1+e*|w-w_0|^3}$
    """
    assert c > 0

    def f(w):
        res = np.abs(w - w0) / (1 + c * np.abs(w - w0)) ** 3
        return res / np.max(res)

    return f


def sqrtgauss(w0, c):
    r"""
    Return $\sqrt{|w-w_0|}exp(-\frac{e|w-w_0|^2}{2})$
    """

    return (
        lambda w: np.sqrt(np.abs(w - w0))
        * np.exp(-c / 2 * np.abs(w - w0) ** 2)
        * (2 * c * np.e) ** (1 / 4)
    )


def lingauss(w0, c):
    r"""
    Return $|w-w_0| exp{-\frac{e|w-w_0|^2}{2}}$
    """

    return lambda w: np.abs(w - w0) * np.exp(-c / 2 * (w - w0) ** 2) * np.sqrt(c * np.e)


def quadgauss(w0, c):
    r"""
    Return $|w-w_0| exp{-\frac{e|w-w_0|^2}{2}}$
    """
    return lambda w: ((w - w0) ** 2) * np.exp(-c / 2 * (w - w0) ** 2) * (c * np.e) / 2


def step(w0, _):
    r"""
    Returns the step function (1-heaviside function)
    1 if w<w_0, 0.5 if w==w_0, 0 otherwise.
    """
    return lambda w: 1 - np.heaviside(w - w0, 0.5)


weight_functions = dict(
    unit=unit,
    exponential=exponential,
    gaussian=gaussian,
    # rspt=rspt,
    sqrtgauss=sqrtgauss,
    lingauss=lingauss,
    quadgauss=quadgauss,
    step=step,
)

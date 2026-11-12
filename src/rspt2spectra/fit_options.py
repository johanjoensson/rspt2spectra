"""The options of the Delta(omega) -> h0 fit, and the words they are spelled with.

:class:`FitOptions` is the one place that lists what steers the fit: the bath geometry, the
regularisation and weight of the least-squares problem, which couplings are neglected and how the
fitting basis may densify the Coulomb tensor.  It is a plain dataclass, so scripts and wrappers
build it directly; :func:`parse_fit_tokens` reads the same options from the whitespace-separated
tokens of a solver line (``peeled gamma 0.01 symm_prec 0.05``) and hands back every token it does
not own, so a caller that shares the line with other consumers can pass the remainder on.
"""

import re
from dataclasses import dataclass, fields

import numpy as np

from .weight_functions import weight_functions

#: The default ``symm_prec``: only numerically negligible couplings (relative 1e-6, never below an
#: absolute 1e-6 Ry) are dropped.  A value above it asks for truncation.
DEFAULT_SYMM_PREC = 1e-6

#: The default ``u_densify``: the fitting basis may make the Coulomb tensor at most this many times
#: denser (number of non-negligible elements) than it is in the basis the caller hands over.
DEFAULT_U_DENSIFY = 2.0

#: Bath geometries :func:`rspt2spectra.edchain.build_H_bath_v` can build.
BATH_GEOMETRIES = ("star", "chain", "linked_chain", "peeled_linked_chain")

# Spellings accepted for a geometry on a solver line.
_GEOMETRY_TOKENS = {**{name: name for name in BATH_GEOMETRIES}, "peeled": "peeled_linked_chain"}


@dataclass(frozen=True)
class FitOptions:
    """What steers the fit of a hybridization function with bath states."""

    bath_geometry: str = "peeled_linked_chain"
    #: Fit the unoccupied bath states too (default: the occupied ones only).
    fit_unocc: bool = False
    #: Hold the bath energies at the previous fit and re-solve only the hoppings.
    freeze_bath_energies: bool = False
    #: Seed the fit with the previous fit's bath energies.
    warm_start: bool = True
    #: A warm fit is kept unless a global search beats its cost by more than this relative margin.
    warm_tol: float = 0.05
    #: Tikhonov regularisation of the hopping amplitudes.
    gamma: float = 0.01
    #: Name in :data:`rspt2spectra.weight_functions.weight_functions`.
    weight_function: str = "unit"
    #: Decay/steepness factor of the weight function.
    weight: float = 2.0
    #: Centre of the weight function.
    weight_w0: float = 0.0
    #: Couplings weaker than this (relative to the strongest diagonal hybridization and to the
    #: crystal-field spread, never below an absolute 1e-6 Ry) are neglected.
    symm_prec: float = DEFAULT_SYMM_PREC
    #: How many times denser the fitting basis may make the Coulomb tensor; ``inf`` disables it.
    u_densify: float = DEFAULT_U_DENSIFY

    def __post_init__(self):
        """Refuse values and combinations the fit cannot honour."""
        if self.bath_geometry not in BATH_GEOMETRIES:
            raise ValueError(f"bath_geometry must be one of {', '.join(BATH_GEOMETRIES)}, got {self.bath_geometry!r}")
        if self.weight_function not in weight_functions:
            raise ValueError(
                f"weight_function must be one of {', '.join(weight_functions)}, got {self.weight_function!r}"
            )
        if not 0 <= self.warm_tol <= 1:
            raise ValueError(f"warm_tol must be between 0 and 1, got {self.warm_tol!r}")
        if self.freeze_bath_energies and not self.warm_start:
            raise ValueError(
                "freeze_bath_energies and cold_fit contradict each other: freezing holds the previous "
                "iteration's bath energies, a cold fit ignores them."
            )
        if not np.isfinite(self.symm_prec) or self.symm_prec < 0:
            raise ValueError(f"symm_prec must be a finite number that is not negative, got {self.symm_prec!r}")
        if np.isnan(self.u_densify) or self.u_densify < 1:
            raise ValueError(f"u_densify must be at least 1 (or inf), got {self.u_densify!r}")

    @property
    def neglects_couplings(self):
        """Whether ``symm_prec`` asks for truncation, as opposed to dropping negligible couplings."""
        return self.symm_prec > DEFAULT_SYMM_PREC

    def to_attrs(self):
        """Return the options as a flat ``{name: value}`` record, for provenance in an archive."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def describe(self):
        """``[(label, text)]`` rows for a run header, in print order."""
        return [(label, getattr(self, name)) for name, label in _LABELS]


# Header label of every option, in print order.
_LABELS = (
    ("bath_geometry", "Bath geometry"),
    ("fit_unocc", "Fit unoccupied states"),
    ("freeze_bath_energies", "Freeze bath energies"),
    ("warm_start", "Warm-start bath fit"),
    ("warm_tol", "Warm-start margin"),
    ("gamma", "Fit regularization gamma"),
    ("weight_function", "Fitting weight function"),
    ("weight", "Fitting weight factor"),
    ("weight_w0", "Fitting weight center w0"),
    ("symm_prec", "Neglected coupling (relative)"),
    ("u_densify", "Max. Coulomb densification"),
)


def _number(option, token):
    """Read a float from a solver-line token; Fortran writes ``1d-2`` for ``1e-2``."""
    try:
        return float(re.sub(r"(?<=[0-9.])[dD](?=[+-]?\d)", "e", token))
    except ValueError:
        raise ValueError(f"'{option}' needs a number, got {token!r}") from None


#: Tokens that take the next token as a value, and how that value is read.
_VALUED = {
    "warm_tol": ("warm_tol", _number),
    "gamma": ("gamma", _number),
    "weight": ("weight", _number),
    "weight_w0": ("weight_w0", _number),
    "symm_prec": ("symm_prec", _number),
    "symmetry_prec": ("symm_prec", _number),
    "u_densify": ("u_densify", _number),
}

#: Tokens that set a flag.
_FLAGS = {
    "fit_unocc": ("fit_unocc", True),
    "fit_occ": ("fit_unocc", False),
    "freeze_bath_energies": ("freeze_bath_energies", True),
    "cold_fit": ("warm_start", False),
}


def parse_fit_tokens(tokens):
    """Read the fit options out of solver-line tokens.

    Parameters
    ----------
    tokens : sequence of str
        The whitespace-separated words, comments already removed.

    Returns
    -------
    options : FitOptions
    unused : list of str
        Every token that is not a fit option, in order, together with the values that follow the
        ones this function does not know.  A caller that shares the line with other option groups
        parses those from here.

    Raises
    ------
    ValueError
        A fit option without a value, with a value out of range, or contradicting another.
    """
    values = {}
    unused = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        word = token.lower()
        if word in _GEOMETRY_TOKENS:
            values["bath_geometry"] = _GEOMETRY_TOKENS[word]
        elif word in _FLAGS:
            name, value = _FLAGS[word]
            values[name] = value
        elif word in weight_functions:
            values["weight_function"] = word
        elif word in _VALUED:
            if i + 1 >= len(tokens):
                raise ValueError(f"'{token}' on the solver line needs a value")
            name, read = _VALUED[word]
            values[name] = read(token, tokens[i + 1])
            i += 1
        else:
            unused.append(token)
        i += 1
    return FitOptions(**values), unused

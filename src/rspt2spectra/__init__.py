"""Turn RSPt real-frequency hybridization functions into impurity Hamiltonians.

This package owns the full pipeline
RSPt files -> Delta(omega) analysis -> block partition -> bath fit
-> bath geometry (star / chains / linked double chain) -> h0 matrix.
The resulting h0 can be consumed by any many-body impurity solver
(e.g. the impurityModel repository); this package does not depend on any
solver.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rspt2spectra")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"

#!/usr/bin/env python3
"""
Read RSPt output and turn real-frequency hybridization functions into
non-interacting impurity Hamiltonians (h0).

This package owns the full pipeline
RSPt files -> Delta(omega) analysis -> block partition -> bath fit
-> bath geometry (star / chains / linked double chain) -> h0 matrix.
The resulting h0 can be consumed by any many-body impurity solver
(e.g. the impurityModel repository); this package does not depend on any
solver. Some support for interfacing to the Quanty software is also
provided.
"""

from . import constants
from . import slater
from . import h2Quanty
from . import plotSpectra
from . import soc
from . import d4h
from . import dc
from . import hybridization
from . import energies
from . import orbitals
from . import unitarytransform
from . import readfile
from . import offdiagonal
from . import hyb_fit
from . import block_structure
from . import edchain
from . import h0
from . import natural_orbitals
from . import utils

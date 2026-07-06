rspt2spectra
============

``rspt2spectra`` reads output from the FP-LMTO DFT(+DMFT) code
`RSPt <http://fplmto-rspt.org/>`_ and turns real-frequency hybridization
functions into finite non-interacting Anderson impurity Hamiltonians (h0),
including bath orbitals.

The package owns the full pipeline::

   RSPt files -> Delta(omega) analysis -> block partition -> bath fit
              -> bath geometry (star / chains / linked double chain) -> h0

The resulting h0 can be consumed by any many-body impurity solver, e.g. the
`impurityModel <https://github.com/johanjoensson/impurityModel>`_ software
(see :mod:`rspt2spectra.h2imp` for writing files in its operator format);
this package does not depend on any solver.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

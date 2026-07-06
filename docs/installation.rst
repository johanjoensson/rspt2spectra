Installation
============

``rspt2spectra`` requires Python 3.11 or later and depends only on
``numpy`` and ``scipy`` at runtime.

Install from a clone of the repository::

   git clone https://github.com/johanjoensson/rspt2spectra.git
   cd rspt2spectra
   pip install .

Optional extras
---------------

MPI-parallel hybridization fits (recommended for production runs)::

   pip install .[mpi]

This requires an MPI implementation (e.g. OpenMPI) to be installed on the
system. Without ``mpi4py`` everything still works, just serially.

Plotting support for ``build_h0 --plot``::

   pip install .[plot]

For development (tests, coverage, linting)::

   pip install -e .[dev]

For building this documentation::

   pip install -e .[docs]
   sphinx-build -b html docs docs/_build/html

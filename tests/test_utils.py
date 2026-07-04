"""Tests for rspt2spectra.utils."""

import numpy as np

from rspt2spectra.utils import block_diagonalize_hyb


def test_block_diagonalize_hyb():
    hyb = np.zeros((2, 2, 2), dtype=complex)
    hyb[:, 0, 1] = 1.0 + 1j
    hyb[:, 1, 0] = 1.0 - 1j
    hyb[:, 0, 0] = 2.0
    hyb[:, 1, 1] = 2.0

    phase_hyb, Q_full = block_diagonalize_hyb(hyb)
    assert phase_hyb.shape == (2, 2, 2)
    assert Q_full.shape == (2, 2)
    assert np.allclose(phase_hyb[:, 0, 1], 0, atol=1e-10)
    assert np.allclose(phase_hyb[:, 1, 0], 0, atol=1e-10)

import numpy as np
import pytest

from rspt2spectra.readfile import parse_fermi_energy, parse_matrices


def test_parse_matrices_single_block_at_eof(tmp_path):
    # The matrix block ending exactly at end-of-file must parse cleanly.
    (tmp_path / "out").write_text(
        "Cluster cl Local hamiltonian\n real part\n -1.5  0.2\n  0.2 -0.7\n imag part\n  0.0  0.1\n -0.1  0.0\n"
    )
    hs = parse_matrices(out_file="out", search_phrase="Local hamiltonian", prefix=str(tmp_path))
    assert list(hs) == ["cl"]
    expected = np.array([[-1.5, 0.2], [0.2, -0.7]]) + 1j * np.array([[0.0, 0.1], [-0.1, 0.0]])
    assert np.allclose(hs["cl"], expected)


def test_parse_matrices_multiple_blocks(tmp_path):
    (tmp_path / "out").write_text(
        "Cluster a Local hamiltonian\n"
        " real part\n"
        "  1.0\n"
        " imag part\n"
        "  0.0\n"
        "Cluster b Local hamiltonian\n"
        " real part\n"
        "  2.0\n"
        " imag part\n"
        "  0.5\n"
        "trailing text\n"
    )
    hs = parse_matrices(out_file="out", search_phrase="Local hamiltonian", prefix=str(tmp_path))
    assert np.allclose(hs["a"], [[1.0]])
    assert np.allclose(hs["b"], [[2.0 + 0.5j]])


def test_parse_fermi_energy(tmp_path):
    (tmp_path / "out").write_text(" some preamble\n fermi energy =  6.7596104733184E-01\n more output\n")
    assert parse_fermi_energy(prefix=str(tmp_path)) == pytest.approx(0.67596104733184)


def test_parse_fermi_energy_takes_the_last_scf_cycle(tmp_path):
    # RSPt prints one line per SCF cycle; the converged value is the last.
    (tmp_path / "out").write_text(
        " fermi energy =  1.0000000000000E-01\n fermi energy =  2.0000000000000E-01\n fermi energy =  3.0E-01\n"
    )
    assert parse_fermi_energy(prefix=str(tmp_path)) == pytest.approx(0.3)


def test_parse_fermi_energy_ignores_derived_restatements(tmp_path):
    # CrI3 runs also print "lda_efsave (fermi energy from lda):", which is not the value
    # the local Hamiltonian is referenced to.
    (tmp_path / "out").write_text(
        " fermi energy =  2.7652035816712E-01\n   lda_efsave (fermi energy from lda):   0.99999999\n"
    )
    assert parse_fermi_energy(prefix=str(tmp_path)) == pytest.approx(0.27652035816712)


def test_parse_fermi_energy_missing_raises(tmp_path):
    (tmp_path / "out").write_text(" nothing useful here\n")
    with pytest.raises(RuntimeError, match="fermi energy"):
        parse_fermi_energy(prefix=str(tmp_path))

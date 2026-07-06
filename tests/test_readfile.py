import numpy as np

from rspt2spectra.readfile import parse_matrices


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

import numpy as np
import pytest

from rspt2spectra.readfile import list_cluster_labels, parse_cluster_basis, parse_fermi_energy, parse_matrices


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


def _write_fcc_ni_style_green_inp(path):
    # green.inp defines the cluster as "0102010103" (numeric-site fallback ID, since the
    # explicit "idNi" prefix is lowercase and does not describe the site-derived digits); RSPt's
    # "Local hamiltonian"/hyb/pdos *output* for it is labelled "0102010103-obs". Basis tag 3
    # (Eg+T2g) with l=2, taken verbatim from impmod_tests/FCC_Ni/base/green.inp.
    path.write_text("cluster\n 1 0 idNi\n 1 2 1 1 3\n")


def test_parse_cluster_basis_matches_despite_obs_suffix(tmp_path):
    _write_fcc_ni_style_green_inp(tmp_path / "green.inp")
    # The unsuffixed label (what green.inp defines) and the "-obs" one (what "out" and the
    # hybridization filenames actually carry) must resolve to the same cluster.
    assert parse_cluster_basis("0102010103", prefix=str(tmp_path)) == (False, 3, 2)
    assert parse_cluster_basis("0102010103-obs", prefix=str(tmp_path)) == (False, 3, 2)


def test_parse_cluster_basis_lowercase_id_prefix_is_not_honored(tmp_path):
    # RSPt's own "Id" keyword match is case-sensitive: on impmod_tests/FCC_Ni/base, a lowercase
    # "idNi" in green.inp is not honored by RSPt either -- "out" labels the cluster
    # "0102010103-obs" (its numeric-site fallback ID), not "Ni-obs". Matching case-insensitively
    # here would recognize a label RSPt itself does not and silently mismatch the real output.
    _write_fcc_ni_style_green_inp(tmp_path / "green.inp")
    assert parse_cluster_basis("Ni", prefix=str(tmp_path)) is None
    assert parse_cluster_basis("0102010103-obs", prefix=str(tmp_path)) == (False, 3, 2)


def test_parse_cluster_basis_returns_none_when_unmatched(tmp_path):
    # Must not be confused with a genuine "basis tag 0, already spherical" match.
    _write_fcc_ni_style_green_inp(tmp_path / "green.inp")
    assert parse_cluster_basis("does-not-exist", prefix=str(tmp_path)) is None


def test_parse_cluster_basis_returns_none_when_file_missing(tmp_path):
    assert parse_cluster_basis("cl", prefix=str(tmp_path)) is None


def test_list_cluster_labels(tmp_path):
    (tmp_path / "green.inp").write_text("cluster\n 1 Ida\n 1 2 1 1 0\ncluster\n 1 Idb\n 1 2 1 1 3\n")
    assert list_cluster_labels(prefix=str(tmp_path)) == ["a", "b"]


def test_list_cluster_labels_missing_file(tmp_path):
    assert list_cluster_labels(prefix=str(tmp_path)) == []

import numpy as np
import pytest

from rspt2spectra.dat import extract_dat, match_columns, split_header_columns


def _write_dat_pair(tmp_path, dataname, cluster, w, matrices, indexmap):
    """Write real-/imag- .dat files with an indexmap header.

    ``matrices`` is (N, n_orb, n_orb) complex; ``indexmap`` maps orbital
    pairs to 1-based data columns (0 = element not stored).
    """
    n_cols = 1 + int(np.max(indexmap))
    data = np.zeros((len(w), n_cols), dtype=complex)
    data[:, 0] = w
    for i in range(indexmap.shape[0]):
        for j in range(indexmap.shape[1]):
            if indexmap[i, j] == 0:
                continue
            data[:, indexmap[i, j] - 1] = matrices[:, i, j]

    header_lines = ["#   Energy      orbitals", "# indexmap"]
    header_lines += ["# " + "  ".join(str(v) for v in row) for row in indexmap]
    header = "\n".join(header_lines) + "\n"

    for part, fname in (
        (np.real, f"real-{dataname}-{cluster}.dat"),
        (np.imag, f"imag-{dataname}-{cluster}.dat"),
    ):
        # Both files carry the real energy mesh in the energy column.
        rows = part(data)
        rows[:, 0] = w
        body = "\n".join("  ".join(f"{v: .12e}" for v in row) for row in rows)
        (tmp_path / fname).write_text(header + body + "\n")


def test_split_and_match_columns():
    cols = split_header_columns("#   Energy      Total     orbitals\n")
    assert cols == ["Energy", "Total", "orbitals"]
    matched = match_columns(cols)
    assert matched == {"energy": 0, "total": 1, "orbitals": 2}


def test_extract_dat_indexmap_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    n_w, n_orb = 11, 2
    w = np.linspace(-5, 5, n_w)
    matrices = rng.normal(size=(n_w, n_orb, n_orb)) + 1j * rng.normal(
        size=(n_w, n_orb, n_orb)
    )
    # Element (0,1)/(1,0) not stored; diagonal in columns 2 and 3 (1-based).
    indexmap = np.array([[2, 0], [0, 3]])
    matrices[:, 0, 1] = 0
    matrices[:, 1, 0] = 0
    _write_dat_pair(tmp_path, "hyb", "0102010100", w, matrices, indexmap)

    dat = extract_dat("hyb", "0102010100", prefix=str(tmp_path))

    assert np.allclose(dat.w, w)
    assert dat.orbitals.shape == (n_w, n_orb, n_orb)
    assert np.allclose(dat.orbitals, matrices)
    # Decoupled diagonal indexmap -> one block per orbital.
    assert dat.blocks == [[0], [1]]
    assert np.allclose(dat.sum, np.sum(np.diagonal(matrices, axis1=1, axis2=2), axis=1))


def test_extract_dat_imag_only(tmp_path):
    rng = np.random.default_rng(3)
    n_w = 7
    w = np.linspace(-1, 1, n_w)
    matrices = 1j * rng.normal(size=(n_w, 1, 1))
    indexmap = np.array([[2]])
    _write_dat_pair(tmp_path, "hyb", "cl", w, matrices, indexmap)
    (tmp_path / "real-hyb-cl.dat").unlink()

    dat = extract_dat("hyb", "cl", prefix=str(tmp_path))
    assert np.allclose(dat.w, w)
    assert np.allclose(dat.orbitals, matrices)


def test_extract_dat_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_dat("hyb", "nope", prefix=str(tmp_path))

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rspt2spectra.op_printer import write_h0_file

GOLDEN = Path(__file__).parent / "golden_h0_v1_index.h0"

# Pinned in impurityModel's doc/h0_file_format.md. The two repos implement the format
# independently, so a drifting writer shows up here rather than in a downstream calculation.
GOLDEN_SHA256 = "eb35f454283638e588e3944b3dc979f1872b9993134f776a8ab99279ab1b2b76"


def _index_encoded(n=8):
    # Values encode their own indices, so any permutation, transpose or conjugation error
    # in the writer changes the bytes.
    h = np.array([[(10 * i + j) + 0.5j * (i - j) for j in range(n)] for i in range(n)], dtype=complex)
    h = 0.5 * (h + h.conj().T)
    np.fill_diagonal(h, h.diagonal().real)
    return h


def test_golden_fixture_hash_is_the_one_the_spec_pins():
    digest = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    assert digest == GOLDEN_SHA256, "the shared fixture drifted from the one impurityModel tests"


def test_writer_reproduces_the_golden_fixture(tmp_path):
    header = json.loads(GOLDEN.read_text().splitlines()[1])
    out = tmp_path / "written.h0"
    write_h0_file(
        out,
        _index_encoded(),
        impurity_orbitals={0: [0, 1, 2, 3]},
        unit="eV",
        energy_reference="fermi",
        fermi_energy=0.0,
        spin_ordering="down_first",
        shell_layout="single",
        valence_bath=[4, 5],
        conduction_bath=[6, 7],
        basis="spherical",
        contains_soc=False,
        producer=header["producer"],
    )

    written = out.read_text().splitlines()
    golden = GOLDEN.read_text().splitlines()
    assert written[0] == golden[0]
    # Header key order is not part of the format; the content is.
    assert json.loads(written[1]) == {**header, "drop_tolerance": 1e-12}
    assert written[3:] == golden[3:]


def test_writer_rejects_a_non_hermitian_hamiltonian(tmp_path):
    h = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="not Hermitian"):
        write_h0_file(tmp_path / "bad.h0", h, impurity_orbitals={0: [0]}, unit="eV")


def test_writer_rejects_non_finite(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        write_h0_file(tmp_path / "bad.h0", np.array([[np.nan]]), impurity_orbitals={0: [0]}, unit="eV")


def test_writer_drops_pairwise(tmp_path):
    h = np.diag([1000.0, 1.0, 1.0]).astype(complex)
    h[1, 2] = h[2, 1] = 1e-10  # below 1e-12 * 1000
    out = tmp_path / "drop.h0"
    write_h0_file(out, h, impurity_orbitals={0: [0]}, unit="eV")

    pairs = {(int(line.split()[0]), int(line.split()[1])) for line in out.read_text().splitlines()[3:]}
    assert (1, 2) not in pairs and (2, 1) not in pairs


def test_amplitudes_round_trip_exactly(tmp_path):
    # The fixed-point legacy writer lost these; repr does not.
    h = np.zeros((2, 2), dtype=complex)
    h[0, 0] = -4.126921785541874
    h[0, 1] = 1.2345678901234567e-10 + 6.599434672757963e-12j
    h[1, 0] = np.conj(h[0, 1])

    out = tmp_path / "small.h0"
    write_h0_file(out, h, impurity_orbitals={0: [0]}, unit="eV", drop_tolerance=0.0)

    got = np.zeros((2, 2), dtype=complex)
    for line in out.read_text().splitlines()[3:]:
        i, j, re_part, im_part = line.split()
        got[int(i), int(j)] = complex(float(re_part), float(im_part))
    np.testing.assert_array_equal(got, h)

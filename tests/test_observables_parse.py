from pathlib import Path

import numpy as np
import pytest

from rspt2spectra.readfile import parse_matrices, parse_observables
from rspt2spectra.rspt_basis import cluster_rotation, spherical_operators

REAL_RUN = Path("/home/johan/Programming/impmod_tests/superlattice/LMO_stressed_bulk/observables")


def _dump(title, m):
    def rows(a):
        return "\n".join("  " + "  ".join(f"{x: .8f}" for x in r) for r in a)

    return f" {title}\n Real part:\n{rows(m.real)}\n Imaginary part:\n{rows(m.imag)}\n"


def test_the_dump_is_parsed_per_cluster_including_both_dipole_groups(tmp_path):
    rng = np.random.default_rng(0)
    m = {k: rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) for k in "abcdefghijk"}
    text = " Observables for cluster ID Mn5\n"
    for title, key in [("Lz: (ml)", "a"), ("Sz: (ms)", "b"), ("S_spinaxis: (msaxis)", "c"), ("L^+: (mlp)", "d"),
                       ("S^+: (msp)", "e")]:
        text += _dump(title, m[key])
    text += " Extra observables:\n"
    for title, key in [("(Dipole) Dx", "f"), ("        Dy", "g"), ("        Dz", "h"),
                       ("(Dipole) Dp", "i"), ("        Dm", "j"), ("        Dz", "k")]:
        text += _dump(title, m[key])
    text += " Observables for cluster ID Other\n" + _dump("Lz: (ml)", m["a"])
    (tmp_path / "out").write_text(text)
    got = parse_observables(prefix=str(tmp_path))
    assert set(got) == {"Mn5", "Other"} and set(got["Other"]) == {"ml"}
    np.testing.assert_allclose(got["Mn5"]["msp"], m["e"], atol=1e-8)
    np.testing.assert_allclose(got["Mn5"]["dxyz"], [m["f"], m["g"], m["h"]], atol=1e-8)
    np.testing.assert_allclose(got["Mn5"]["dpm"], [m["i"], m["j"], m["k"]], atol=1e-8)


def test_spherical_operators_obey_the_angular_momentum_algebra():
    ops = spherical_operators([(2, 3), (1, 1)])
    lz, lp, sz, sp = (ops[k] for k in ("lz", "l_plus", "sz", "s_plus"))
    np.testing.assert_allclose(lz @ lp - lp @ lz, lp, atol=1e-12)
    np.testing.assert_allclose(sz @ sp - sp @ sz, sp, atol=1e-12)
    assert lz.shape == (16, 16) and np.allclose(sz, np.diag(sz).real * np.eye(16))


@pytest.mark.skipif(not (REAL_RUN / "out").exists(), reason="needs the LMO observables run")
def test_rotating_rspts_cubic_harmonic_dump_with_the_hamiltonian_rotation_gives_spherical_operators():
    # Mn5 is in real cubic harmonics (Cf flag); the same T that rotates H must turn the printed
    # operators into the canonical spherical ones.
    obs = parse_observables(prefix=str(REAL_RUN))["Mn5"]
    qs = parse_matrices(search_phrase="Transformation to the local cf basis:", prefix=str(REAL_RUN))
    T = cluster_rotation([(2, 3)], spin_resolved=True) @ qs["Mn5"]
    ref = spherical_operators([(2, 3)])
    for printed, name in [("ml", "lz"), ("ms", "sz"), ("mlp", "l_plus"), ("msp", "s_plus")]:
        np.testing.assert_allclose(T @ obs[printed] @ T.conj().T, ref[name], atol=1e-6)
    assert obs["dxyz"].shape == (3, 10, 10)


def test_a_ragged_or_incomplete_dump_is_an_error_not_a_silent_misread(tmp_path):
    good = _dump("Lz: (ml)", np.eye(2))
    (tmp_path / "out").write_text(" Observables for cluster ID A\n" + good.replace("   1.00000000   0.00000000", "   1.00000000", 1))
    with pytest.raises(ValueError, match="square"):
        parse_observables(prefix=str(tmp_path))
    two = " Observables for cluster ID A\n" + _dump("(Dipole) Dx", np.eye(2)) + _dump("Dy", np.eye(2))
    (tmp_path / "out").write_text(two + " Observables for cluster ID B\n")
    with pytest.raises(ValueError, match="3 components|expected 3"):
        parse_observables(prefix=str(tmp_path))


@pytest.mark.parametrize("shells, kept", [([(2, 3)], False), ([(1, 1), (2, 3)], True)])
def test_a_dipole_without_a_core_valence_block_is_not_written(tmp_path, shells, kept):
    """RSPt's Dxyz of a d-only cluster is the valence-valence optical matrix: no XAS transition, so build_h0
    leaves it out and impurityModel falls back to its Gaunt dipole."""
    from rspt2spectra.scripts.build_h0 import _read_operators

    n = sum(2 * (2 * l + 1) for l, _ in shells)
    rng = np.random.default_rng(1)
    text = " Observables for cluster ID A\n" + _dump("Lz: (ml)", np.eye(n)) + " Extra observables:\n"
    for title in ("(Dipole) Dx", "        Dy", "        Dz"):
        text += _dump(title, rng.normal(size=(n, n)))
    (tmp_path / "out").write_text(text)
    ops, _ = _read_operators("A", str(tmp_path), np.eye(n), n, shells, True, False)
    assert ("dipole" in ops) is kept
    assert "lz" in ops

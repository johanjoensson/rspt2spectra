import numpy as np
import pytest

from rspt2spectra.block_structure import (
    BlockStructure,
    build_block_structure,
    build_greens_function,
    build_matrix,
    get_equivalent_blocks,
    get_equivalent_orbs,
    print_block_structure,
)


def test_build_block_structure_matrix():
    mat = np.zeros((7, 7))
    b1 = np.array([[1.0, 0.5], [0.5, 2.0]])
    mat[0:2, 0:2] = b1
    # Identical
    mat[2:4, 2:4] = b1

    b3 = np.array([[3.0]])
    mat[4:5, 4:5] = b3

    # Negative identical (particle-hole without transpose)
    b4 = -b1
    mat[5:7, 5:7] = b4

    bs = build_block_structure(G=None, mat=mat)

    assert len(bs.blocks) == 4
    assert bs.blocks[0] == [0, 1]
    assert bs.blocks[1] == [2, 3]
    assert bs.blocks[2] == [4]
    assert bs.blocks[3] == [5, 6]

    assert bs.identical_blocks[0] == [0, 1]
    assert bs.inequivalent_blocks == [0, 2]


def test_get_equivalent_blocks():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3], [4]],
        identical_blocks=[[0, 1], [], [2]],
        transposed_blocks=[[], [], []],
        particle_hole_blocks=[[], [], []],
        particle_hole_transposed_blocks=[[], [], []],
        inequivalent_blocks=[0, 2],
    )

    eq_blocks = get_equivalent_blocks(bs)
    assert eq_blocks == [[0, 1], [2]]

    eq_orbs = get_equivalent_orbs(bs)
    assert eq_orbs == [[0, 1, 2, 3], [4]]


def test_build_matrix():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[1.0, 0.5], [0.5, 2.0]])
    mat = build_matrix([b1], bs)
    assert np.allclose(mat[0:2, 0:2], b1)
    assert np.allclose(mat[2:4, 2:4], b1)


def test_print_block_structure(capsys):
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    print_block_structure(bs)
    captured = capsys.readouterr()
    assert "0   0   +   +" in captured.out
    assert "+   +   0   0" in captured.out


def test_build_block_structure_G():
    # Particle-hole equivalence relates a block at +w to another at -w, so the
    # test needs a mesh symmetric about zero and G_3(w) = -conj(G_0(-w)).
    w = np.array([-1.0, 0.0, 1.0])
    G = np.zeros((len(w), 7, 7), dtype=complex)
    b1 = np.array(
        [
            [[1.0 - 0.2j, 0.5], [0.5, 2.0 - 0.1j]],
            [[1.5 - 0.4j, 0.6], [0.6, 2.5 - 0.3j]],
            [[2.0 - 0.6j, 0.7], [0.7, 3.0 - 0.5j]],
        ]
    )
    G[:, 0:2, 0:2] = b1
    # Identical
    G[:, 2:4, 2:4] = b1

    b3 = np.array([[[3.0]], [[3.5]], [[4.0]]])
    G[:, 4:5, 4:5] = b3

    # Particle-hole partner of block 0.
    G[:, 5:7, 5:7] = -np.conj(b1[::-1])

    bs = build_block_structure(G=G, w=w)
    assert len(bs.blocks) == 4
    assert bs.identical_blocks[0] == [0, 1]
    # Check that particle-hole blocks for block 0 include block 3
    assert 3 in bs.particle_hole_blocks[0] or 3 in bs.particle_hole_transposed_blocks[0]


def test_particle_hole_detection_needs_the_frequency_mirror():
    # The same data compared at equal frequency indices instead of mirrored ones
    # finds nothing: Kramers-Kronig makes that condition unsatisfiable, which is
    # why the relation used to be detected essentially never.
    w = np.array([-1.0, 0.0, 1.0])
    G = np.zeros((len(w), 4, 4), dtype=complex)
    b1 = np.array(
        [
            [[1.0 - 0.2j, 0.5], [0.5, 2.0 - 0.1j]],
            [[1.5 - 0.4j, 0.6], [0.6, 2.5 - 0.3j]],
            [[2.0 - 0.6j, 0.7], [0.7, 3.0 - 0.5j]],
        ]
    )
    G[:, 0:2, 0:2] = b1
    G[:, 2:4, 2:4] = -np.conj(b1[::-1])

    assert 1 in build_block_structure(G=G, w=w).particle_hole_blocks[0]

    # Without the mesh the relation cannot be tested at all, and is skipped.
    with pytest.warns(RuntimeWarning, match="particle-hole"):
        bs_no_w = build_block_structure(G=G)
    assert bs_no_w.particle_hole_blocks == [[], []]


def test_build_greens_function():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[[1.0, 0.5], [0.5, 2.0]]])
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], b1)


def test_build_greens_function_transposed():
    # Block 1 is the transpose of block 0 (swap last two orbital axes).
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[1], [0]],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    # Shape: (n_freq=3, n_orb_block=2, n_orb_block=2) — asymmetric so transpose is visible.
    b1 = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], b1.swapaxes(-2, -1))


def test_build_greens_function_particle_hole():
    # Block 1 is the particle-hole partner of block 0: G_1(w) = -conj(G_0(-w)).
    # The sign and the conjugation are not cosmetic -- without them the partner
    # has Im G > 0 and is not a retarded function.
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[1], [0]],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], -np.conj(b1[::-1, :, :]))


def test_build_greens_function_particle_hole_transposed():
    # Block 1 is the particle-hole partner of block 0 and orbitally transposed:
    # G_1(w) = -conj(G_0(-w))^T.
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[1], [0]],
        inequivalent_blocks=[0],
    )
    b1 = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], -np.conj(b1[::-1]).swapaxes(-2, -1))

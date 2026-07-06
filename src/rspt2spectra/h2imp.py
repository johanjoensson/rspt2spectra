#!/usr/bin/env python3

"""Interface to the impurityModel repository operator format."""

import pickle

from rspt2spectra import op_printer


def matrixToIOp(mat):
    r"""Return the non-zero matrix elements in impurityModel operator format.

    The output format is ``{((i, "c"), (j, "a")): mat[i, j]}``.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix representation of the operator

    Returns
    -------
    res : dict
        Operator dictionary {(i, j) : val}
    """
    rows, columns = mat.shape
    res = {}
    for i in range(rows):
        for j in range(columns):
            if abs(mat[i, j]) > 0:
                res[((i, "c"), (j, "a"))] = mat[i, j]
    return res


def write_to_file(d, filename="h0_Op", save_as_dict=False):
    """Write the operator to disk, pickled or in impurityModel text format."""
    if save_as_dict:
        op_printer.write_operator_to_file([d], filename + ".dict")
    else:
        with open(filename + ".pickle", "wb") as handle:
            pickle.dump(d, handle)

"""Parse matrices printed in RSPt ``out`` files."""

import numpy as np


def parse_matrices(out_file="out", search_phrase="Local hamiltonian", prefix="."):
    """Extract labelled complex matrices from an RSPt ``out`` file.

    RSPt prints matrices as a label line containing ``search_phrase``,
    followed by a real block and an imaginary block of whitespace-separated
    numbers.

    Parameters
    ----------
    out_file : str, default "out"
        Name of the RSPt output file.
    search_phrase : str, default "Local hamiltonian"
        Phrase identifying the matrices to extract; the cluster label is
        taken as the second word on the matching line.
    prefix : str, default "."
        Directory holding the output file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from cluster label to complex matrix.
    """
    labels = []
    matrices = []
    with open(f"{prefix}/{out_file}", "r") as f:
        it = iter(f)
        for line in it:
            if search_phrase not in line:
                continue
            labels.append(line.split()[1])
            cursor = line
            while "real" not in cursor.lower():
                cursor = next(it)
            cursor = next(it)
            real_rows = []
            while "imag" not in cursor.lower():
                real_rows.append([float(num) for num in cursor.split()])
                cursor = next(it)
            imag_rows = []
            for _ in range(len(real_rows)):
                cursor = next(it)
                imag_rows.append([float(num) for num in cursor.split()])
            matrices.append(np.array(real_rows) + 1j * np.array(imag_rows))

    return dict(zip(labels, matrices))

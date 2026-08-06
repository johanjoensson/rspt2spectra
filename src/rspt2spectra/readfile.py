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


def parse_fermi_energy(out_file="out", prefix="."):
    """Extract the Fermi energy from an RSPt ``out`` file.

    RSPt prints ``fermi energy =  6.7596104733184E-01`` once per SCF cycle; the **last**
    occurrence is the converged one, so that is the one returned.

    The value matters because RSPt's printed ``Local hamiltonian`` is an *absolute* energy
    while the hybridization function it is combined with is written on a mesh whose zero is
    the Fermi level. Subtracting it is what puts the impurity and bath blocks on one energy
    scale; without it the impurity level sits several eV above the entire bath.

    Parameters
    ----------
    out_file : str, default "out"
        Name of the RSPt output file.
    prefix : str, default "."
        Directory holding the output file.

    Returns
    -------
    float
        The Fermi energy, in whatever unit the file uses (Rydberg for a default RSPt run).

    Raises
    ------
    RuntimeError
        If the file contains no Fermi energy line.
    """
    value = None
    with open(f"{prefix}/{out_file}", "r") as f:
        for line in f:
            lowered = line.lower()
            if "fermi energy" not in lowered or "=" not in line:
                continue
            # Skip derived restatements such as "lda_efsave (fermi energy from lda):".
            if lowered.strip().startswith("fermi energy"):
                value = float(line.split("=", 1)[1].split()[0])

    if value is None:
        raise RuntimeError(
            f"Could not find a 'fermi energy =' line in {prefix}/{out_file}. "
            "It is needed to put the local Hamiltonian on the same energy zero as the "
            "hybridization function."
        )
    return value


def _normalized_cluster_label(label):
    """Strip RSPt's observable-output ``-obs`` suffix and case-fold a cluster label.

    RSPt labels the ``Local hamiltonian``/hybridization/pdos output for a cluster with an
    ``-obs`` suffix that never appears in the cluster's own ``green.inp`` definition (e.g.
    ``out`` says ``ID 0102010103-obs``, ``green.inp`` defines ``0102010103``). Both spellings
    name the same cluster, so matching must fold one onto the other.
    """
    stripped = label[: -len("-obs")] if label.lower().endswith("-obs") else label
    return stripped.lower()


def list_cluster_labels(inp_file="green.inp", prefix="."):
    """List every cluster label defined in a green.inp file, for a "cluster not found" message.

    Parameters
    ----------
    inp_file : str, default "green.inp"
        Name of the RSPt input file.
    prefix : str, default "."
        Directory holding the input file.

    Returns
    -------
    list[str]
        The cluster labels found, in file order. Empty if the file is missing or defines none.
    """
    try:
        with open(f"{prefix}/{inp_file}", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    labels = []
    for i in range(len(lines)):
        if not lines[i].strip().lower().startswith("cluster") or i + 1 >= len(lines):
            continue
        parsed_id = _parse_cluster_id(lines, i)
        if parsed_id:
            labels.append(parsed_id)
    return labels


def _parse_cluster_id(lines, i):
    """Parse the cluster label for the ``cluster`` block starting at line ``i``.

    Returns ``""`` if the block is malformed (no ``ntot`` on the following line).
    """
    ntot_line = lines[i + 1].strip()
    code_part = ntot_line.split("!")[0].split("#")[0].strip()
    words = code_part.split()
    if not words:
        return ""
    try:
        int(words[0])
    except ValueError:
        return ""

    for word in words[1:]:
        # RSPt's own "Id" keyword match is case-sensitive: a lowercase "idNi" in green.inp is
        # not honored by RSPt either, which silently falls back to its numeric cluster ID
        # instead (confirmed on impmod_tests/FCC_Ni/base -- "idNi" is defined, but RSPt's own
        # "out" labels the cluster "0102010103-obs", not "Ni-obs"). Matching case-insensitively
        # here would recognize a label RSPt itself does not, and break the match.
        if word.startswith("Id"):
            return word[2:]

    parsed_id = ""
    if i + 2 < len(lines):
        first_site_line = lines[i + 2].split("!")[0].split("#")[0]
        first_site_parts = first_site_line.split()
        if len(first_site_parts) >= 5:
            for x in first_site_parts[:5]:
                val = abs(int(x))
                parsed_id += f"{val:03d}" if val >= 100 else f"{val:02d}"
    return parsed_id


def parse_cluster_basis(cluster_label, inp_file="green.inp", prefix="."):
    """Parse green.inp file to determine if Cf flag or a non-spherical basis was used for the given cluster.

    Parameters
    ----------
    cluster_label : str
        The label of the cluster to check.
    inp_file : str, default "green.inp"
        Name of the RSPt input file.
    prefix : str, default "."
        Directory holding the input file.

    Returns
    -------
    tuple[bool, int, int] or None
        ``(has_cf_flag, basis_tag, l_val)`` for the matched cluster -- ``has_cf_flag`` is True
        if the data is in the CF basis, ``basis_tag`` is nonzero for a non-spherical basis, and
        ``l_val`` is the shell's angular momentum (``-1`` if it could not be determined). Returns
        ``None`` if the input file is missing or no cluster in it matches ``cluster_label`` --
        this is deliberately *not* the same value as a genuine "basis tag 0" match, since a
        caller must not treat an unresolved cluster as "already spherical".
    """
    try:
        with open(f"{prefix}/{inp_file}", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    target = _normalized_cluster_label(cluster_label)
    for i in range(len(lines)):
        line = lines[i].strip()
        line_lower = line.lower()

        # Look for the cluster initialization block
        if line_lower.startswith("cluster"):
            if i + 1 >= len(lines):
                continue

            ntot_line = lines[i + 1].strip()
            # Handle comments
            code_part = ntot_line.split("!")[0].split("#")[0].strip()

            words = code_part.split()
            if not words:
                continue

            try:
                ntot = int(words[0])
            except ValueError:
                continue

            parsed_id = _parse_cluster_id(lines, i)

            cfflag_local = False
            for word in words[1:]:
                if word.lower() == "cf":
                    cfflag_local = True
                    break

            if _normalized_cluster_label(parsed_id) == target:
                cfflag = cfflag_local
                basis_tag = 0
                l_val = -1
                for j in range(ntot):
                    if i + 2 + j < len(lines):
                        site_line = lines[i + 2 + j].split("!")[0].split("#")[0]
                        site_parts = site_line.split()
                        # id(1) = type, id(2) = l, id(3) = e, id(4) = site/tail, id(5) = basis tag
                        if len(site_parts) >= 5:
                            try:
                                if l_val == -1:
                                    l_val = int(site_parts[1])
                                b = int(site_parts[4])
                                if b != 0:
                                    basis_tag = b
                            except ValueError:
                                pass
                return cfflag, basis_tag, l_val

    return None

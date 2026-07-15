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
    bool
        True if the data for the cluster is in the CF basis or uses a non-spherical basis (b!=0), False otherwise.
    """
    try:
        with open(f"{prefix}/{inp_file}", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return False
    
    for i in range(len(lines)):
        line = lines[i].strip()
        line_lower = line.lower()
        
        # Look for the cluster initialization block
        if line_lower.startswith("cluster"):
            if i + 1 >= len(lines):
                continue
                
            ntot_line = lines[i+1].strip()
            # Handle comments
            code_part = ntot_line.split("!")[0].split("#")[0].strip()
            
            words = code_part.split()
            if not words:
                continue
                
            try:
                ntot = int(words[0])
            except ValueError:
                continue

            parsed_id = ""
            for word in words[1:]:
                if word.startswith("Id"):
                    parsed_id = word[2:]
                    break
                    
            cfflag_local = False
            for word in words[1:]:
                if word.lower() == "cf":
                    cfflag_local = True
                    break
                    
            if not parsed_id and i + 2 < len(lines):
                first_site_line = lines[i + 2].split("!")[0].split("#")[0]
                first_site_parts = first_site_line.split()
                if len(first_site_parts) >= 5:
                    for x in first_site_parts[:5]:
                        val = abs(int(x))
                        if val >= 100:
                            parsed_id += f"{val:03d}"
                        else:
                            parsed_id += f"{val:02d}"
            
            if parsed_id.lower() == cluster_label.lower():
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

    return False, 0, -1

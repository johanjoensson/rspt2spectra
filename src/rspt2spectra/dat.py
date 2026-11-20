"""Read RSPt ``real-*``/``imag-*`` .dat files into structured arrays.

Vendored from the ``pyRSPthon`` package (same author) so that
``rspt2spectra`` is installable without external RSPt tooling.
"""

import itertools
import logging
import re
from collections import namedtuple

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)

RSPtDAT = namedtuple("RSPtDAT", ["w", "sum", "up", "down", "orbitals", "blocks"])
"""Contents of a pair of ``real-*``/``imag-*`` RSPt .dat files.

Attributes
----------
w : (N,) np.ndarray
    Real energy mesh.
sum : (N,) np.ndarray
    Trace/total of the quantity.
up, down : (N,) np.ndarray or None
    Spin-resolved sums, when available.
orbitals : (N, n_orb, n_orb) or (N, n_col) np.ndarray or None
    Orbital-resolved data; a matrix per energy point when the file
    carries an indexmap.
blocks : list[list[int]] or None
    Groups of orbitals coupled by the indexmap.
"""

# Column-label patterns, tried in order. More specific patterns (tot+, spin up)
# must come before the generic ones (total, up).
_COLUMN_PATTERNS = (
    ("energy", ("energy", "frequency")),
    ("up", ("tot+", "spin up")),
    ("down", ("tot-", "spin dn", "spin down")),
    ("total", ("total", "sum", "tot")),
    ("sx", ("sx",)),
    ("sy", ("sy",)),
    ("sz", ("sz",)),
    ("lx", ("lx",)),
    ("ly", ("ly",)),
    ("lz", ("lz",)),
    ("jx", ("jx",)),
    ("jy", ("jy",)),
    ("jz", ("jz",)),
    ("orbitals", ("orbitals",)),
    ("up", ("up",)),
    ("down", ("down", "dn")),
)


def split_header_columns(header):
    """Split a '#'-prefixed RSPt column header line into column labels."""
    header = header.strip().strip("#")
    header = header.replace(",", "  ")
    return re.split("  +", header.strip())


def match_columns(columns):
    """Map RSPt header column labels to column indices.

    Parameters
    ----------
    columns : list of str
        Column labels from the header line.

    Returns
    -------
    dict[str, int]
        Keys among: energy, total, up, down, sx..sz, lx..lz, jx..jz,
        orbitals (index of the first orbital column). Only matched labels
        are present.
    """
    matched = {}
    for i, label in enumerate(columns):
        label_l = label.lower()
        if not label_l:
            continue
        for name, needles in _COLUMN_PATTERNS:
            if any(needle in label_l for needle in needles):
                # First match wins; both for the pattern and for the column.
                matched.setdefault(name, i)
                break
        else:
            logger.warning("Unknown column label %r", label)
    return matched


def _read_dat_header(fname):
    """Read the header line and optional indexmap block of a real-/imag- dat file."""
    indexmap = None
    with open(fname, "r") as f:
        header = next(f)
        if "indexmap" in next(f, ""):
            indexmap = []
            for line in f:
                if line[0] != "#":
                    break
                row = [int(i) for i in line.strip("#").split()]
                indexmap.append(row)
    return header, indexmap


def extract_dat(dataname, cluster, prefix="."):
    """Read a pair of ``real-/imag-<dataname>-<cluster>.dat`` RSPt output files.

    Either file may be missing, in which case the corresponding part is
    taken to be zero. When the header carries an indexmap, the orbital
    columns are unfolded into an ``(N, n_orb, n_orb)`` matrix per energy
    point.

    Parameters
    ----------
    dataname : str
        Base name of the quantity, e.g. ``"hyb"`` or ``"sig"``.
    cluster : str
        RSPt cluster label, e.g. ``"0102010100"``.
    prefix : str, default "."
        Directory holding the .dat files.

    Returns
    -------
    RSPtDAT
        The parsed data; see `RSPtDAT` for the field layout.
    """
    if prefix != "" and prefix[-1] != "/":
        prefix = prefix + "/"
    realname = f"{prefix}real-{dataname}-{cluster}.dat"
    imagname = f"{prefix}imag-{dataname}-{cluster}.dat"
    try:
        header, indexmap = _read_dat_header(realname)
    except FileNotFoundError:
        try:
            header, indexmap = _read_dat_header(imagname)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find either {realname} or {imagname}."
            ) from None
    if indexmap is not None:
        indexmap = np.array(indexmap, dtype=int)

    columns = split_header_columns(header)
    cols = match_columns(columns)
    if "energy" not in cols:
        raise RuntimeError(
            f"{realname}, {imagname} do not contain an energy mesh! "
            "(The header does not include 'Energy' or 'Frequency')"
        )
    e_col = cols["energy"]

    try:
        re_dat = np.loadtxt(realname)
    except FileNotFoundError:
        logger.info("Could not find file %s. Setting real part to 0.", realname)
        re_dat = None
    try:
        im_dat = np.loadtxt(imagname)
    except FileNotFoundError:
        logger.info("Could not find file %s. Setting imaginary part to 0.", imagname)
        im_dat = None
    if re_dat is None:
        dat = 1j * im_dat
        # The energy mesh is real, even when read from the imaginary-part file
        dat[:, e_col] = dat[:, e_col].imag
    elif im_dat is None:
        dat = re_dat.astype(complex)
    else:
        dat = re_dat + 1j * im_dat
        dat[:, e_col] = dat[:, e_col].real

    # Columns beyond the labeled ones hold orbital-resolved data
    orb_start = cols.get("orbitals")
    if orb_start is None and len(columns) < dat.shape[1]:
        orb_start = len(columns)

    orb_data = None
    if indexmap is not None:
        orb_data = np.zeros(
            (dat.shape[0], indexmap.shape[0], indexmap.shape[1]), dtype=complex
        )
        for i, j in itertools.product(
            range(indexmap.shape[0]), range(indexmap.shape[1])
        ):
            if indexmap[i, j] == 0:
                continue
            orb_data[:, i, j] = dat[:, indexmap[i, j] - 1]
    elif orb_start is not None and orb_start < dat.shape[1]:
        orb_data = dat[:, orb_start:]

    if "total" in cols:
        sum_data = dat[:, cols["total"]]
    elif orb_data is not None and orb_data.ndim == 3:
        sum_data = np.sum(np.diagonal(orb_data, axis1=1, axis2=2), axis=1)
    elif orb_data is not None:
        sum_data = np.sum(orb_data, axis=1)
    else:
        raise RuntimeError(
            f"{realname}, {imagname} do not contain either summed up data "
            "(header field 'total') or orbital resolved data (indexmap in header)"
        )

    # With spin polarization but no explicit spin columns, the diagonal
    # orbital entries hold the two spin channels: first half down, second half up.
    up_data = None
    dn_data = None
    if "down" in cols:
        dn_data = dat[:, cols["down"]]
    elif orb_data is not None and orb_data.shape[1] % 2 == 0 and orb_data.shape[1] > 0:
        logger.info(
            "%s, %s do not contain any spin down projected data. "
            "Assuming I can sum the first half of the diagonal orbital terms.",
            realname,
            imagname,
        )
        if orb_data.ndim == 3:
            dn_data = np.sum(
                np.diagonal(orb_data, axis1=1, axis2=2)[:, : orb_data.shape[1] // 2],
                axis=1,
            )
        else:
            dn_data = np.sum(orb_data[:, : orb_data.shape[1] // 2], axis=1)
    if "up" in cols:
        up_data = dat[:, cols["up"]]
    elif orb_data is not None and orb_data.shape[1] % 2 == 0 and orb_data.shape[1] > 0:
        logger.info(
            "%s, %s do not contain any spin up projected data. "
            "Assuming I can sum the second half of the diagonal orbital terms.",
            realname,
            imagname,
        )
        if orb_data.ndim == 3:
            up_data = np.sum(
                np.diagonal(orb_data, axis1=1, axis2=2)[:, orb_data.shape[1] // 2 :],
                axis=1,
            )
        else:
            up_data = np.sum(orb_data[:, orb_data.shape[1] // 2 :], axis=1)

    blocks = None
    if indexmap is not None:
        n_blocks, block_idxs = sp.sparse.csgraph.connected_components(
            csgraph=sp.sparse.csr_matrix(indexmap > 0),
            directed=False,
            return_labels=True,
        )
        blocks = [[] for _ in range(n_blocks)]
        for orb_i, block_i in enumerate(block_idxs):
            blocks[block_i].append(orb_i)
    return RSPtDAT(
        w=dat[:, e_col].real,
        sum=sum_data,
        up=up_data,
        down=dn_data,
        orbitals=orb_data,
        blocks=blocks,
    )

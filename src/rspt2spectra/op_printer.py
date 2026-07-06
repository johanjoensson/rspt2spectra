"""
Write second-quantization operators to text files readable by impurityModel.

Each operator is a dict ``{((i, "c"), (j, "a")): amplitude}``; it is written
as one line per element: the two spin-orbital indices followed by the real
and imaginary parts of the amplitude.
"""


def key_to_string(key):
    """Format an operator key ``((i, "c"), (j, "a"))`` as ``"  i   j"``."""
    (state1, _), (state2, _) = key
    return f"{state1:3d} {state2:3d}"


def value_to_string(value):
    """Format a complex amplitude as ``"re im"`` with 15 decimals."""
    return f"{value.real: .15f} {value.imag: .15f}"


def key_value_to_string(key, value):
    """Format one operator element as a full output line."""
    return key_to_string(key) + " " + value_to_string(value) + "\n"


def write_operator_to_file(operator, filename):
    """Write a sequence of operators to ``filename``, one block per operator.

    Parameters
    ----------
    operator : iterable of dict
        Operators in the ``{((i, "c"), (j, "a")): amplitude}`` format.
    filename : str
        Output file; overwritten if it exists. Operator blocks are separated
        by blank lines.
    """
    strings = []
    for op in operator:
        s = ""
        for key, value in op.items():
            s += key_value_to_string(key, value)
        strings.append(s)
    with open(filename, "w") as f:
        f.write("\n".join(strings))

"""

op_printer
==========

This module contains functions for printing operators to files. The operators
are written in a format readable by impurityModel

"""


def key_to_string(key):
    ((state1, _), (state2, _)) = key
    return f"{state1:3d} {state2:3d}"


def value_to_string(value):
    return f"{value.real: .15f} {value.imag: .15f}"


def key_value_to_string(key, value):
    return key_to_string(key) + " " + value_to_string(value) + "\n"


def write_operator_to_file(operator, filename):
    with open(filename, "w+"):
        pass

    strings = []
    for op in operator:
        s = ""
        for key, value in op.items():
            s += key_value_to_string(key, value)
        strings.append(s)
    with open(filename, "a") as f:
        f.write("\n".join(strings))

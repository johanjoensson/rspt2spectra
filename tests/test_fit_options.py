import math

import pytest

from rspt2spectra.fit_options import DEFAULT_SYMM_PREC, DEFAULT_U_DENSIFY, FitOptions, parse_fit_tokens


def test_defaults_are_what_a_bare_solver_line_gives():
    options, unused = parse_fit_tokens([])
    assert options == FitOptions()
    assert unused == []
    assert options.gamma == 0.01
    assert options.bath_geometry == "peeled_linked_chain"
    assert options.symm_prec == DEFAULT_SYMM_PREC and options.u_densify == DEFAULT_U_DENSIFY
    assert not options.neglects_couplings


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["star"], {"bath_geometry": "star"}),
        (["peeled"], {"bath_geometry": "peeled_linked_chain"}),
        (["Linked_Chain"], {"bath_geometry": "linked_chain"}),
        (["fit_unocc"], {"fit_unocc": True}),
        (["fit_unocc", "fit_occ"], {"fit_unocc": False}),
        (["cold_fit"], {"warm_start": False}),
        (["freeze_bath_energies"], {"freeze_bath_energies": True}),
        (["warm_tol", "0.2"], {"warm_tol": 0.2}),
        (["gamma", "1d-3"], {"gamma": 1e-3}),
        (
            ["gaussian", "weight", "4", "weight_w0", "-1.5"],
            {"weight_function": "gaussian", "weight": 4.0, "weight_w0": -1.5},
        ),
        (["symm_prec", "0.05"], {"symm_prec": 0.05}),
        (["symmetry_prec", "5d-2"], {"symm_prec": 0.05}),
        (["u_densify", "inf"], {"u_densify": math.inf}),
    ],
)
def test_tokens_set_the_option_they_name(tokens, expected):
    options, unused = parse_fit_tokens(tokens)
    assert unused == []
    for name, value in expected.items():
        assert getattr(options, name) == value


def test_tokens_that_are_not_fit_options_are_returned_in_order_with_their_values():
    options, unused = parse_fit_tokens(["slater_min", "1e-3", "star", "gf_method", "lanczos", "no_mv", "gamma", "0.1"])
    assert options.bath_geometry == "star" and options.gamma == 0.1
    assert unused == ["slater_min", "1e-3", "gf_method", "lanczos", "no_mv"]


@pytest.mark.parametrize("tokens", [["gamma"], ["symm_prec"], ["u_densify"], ["warm_tol"]])
def test_a_valued_token_without_a_value_is_an_error(tokens):
    with pytest.raises(ValueError, match="needs a value"):
        parse_fit_tokens(tokens)


@pytest.mark.parametrize(
    "tokens",
    [["symm_prec", "-1"], ["symm_prec", "nan"], ["symm_prec", "x"], ["u_densify", "0.5"], ["warm_tol", "2"]],
)
def test_values_out_of_range_are_errors(tokens):
    with pytest.raises(ValueError):
        parse_fit_tokens(tokens)


def test_freezing_and_cold_fitting_contradict_each_other():
    with pytest.raises(ValueError, match="contradict"):
        parse_fit_tokens(["freeze_bath_energies", "cold_fit"])
    with pytest.raises(ValueError, match="contradict"):
        parse_fit_tokens(["cold_fit", "freeze_bath_energies"])


def test_raising_symm_prec_asks_for_truncation():
    assert FitOptions(symm_prec=0.05).neglects_couplings
    assert not FitOptions(symm_prec=DEFAULT_SYMM_PREC).neglects_couplings


def test_attrs_and_description_cover_every_option():
    options = FitOptions(symm_prec=0.05)
    attrs = options.to_attrs()
    assert set(attrs) == set(FitOptions.__dataclass_fields__)
    assert attrs["symm_prec"] == 0.05
    assert len(options.describe()) == len(attrs)

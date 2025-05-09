import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from scipy.signal import find_peaks, peak_widths
from .offdiagonal import get_hyb, get_v_and_eb
import warnings


def v_opt(a, b, _):
    return a if abs(a[-1]) <= abs(b[-1]) else b


def fit_block(
    hyb,
    w,
    delta,
    bath_states_per_orbital,
    gamma,
    imag_only,
    realvalue_v,
    comm,
    verbose,
    weight_fun,
    bath_guess=None,
    v_guess=None,
):
    rng = np.random.default_rng()

    hyb_trace = -np.imag(np.sum(np.diagonal(hyb, axis1=1, axis2=2), axis=1))
    hyb_trace[hyb_trace < 0] = 0
    n_orb = hyb.shape[1]
    peaks, info = find_peaks(
        hyb_trace,
    )
    scores = weight_fun(w[peaks]) * hyb_trace[peaks]
    normalised_scores = scores / np.sum(scores)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, left_ips, right_ips = peak_widths(hyb_trace, peaks, rel_height=0.8)

    if verbose:
        print("Peak positions:    ", ", ".join(f"{el: ^16.3f}" for el in w[peaks]))
        print(
            "Peak intervals:    ",
            ", ".join(
                f"[{w[int(max(0, el))]: >6.3f}, {w[int(min(len(w) - 1, er))]: >6.3f}]"
                for el, er in zip(left_ips, right_ips)
            ),
        )
        print(
            "Peak scores:       ",
            ", ".join(f"{el: ^16.3f}" for el in normalised_scores),
        )
    min_cost = np.inf
    eb_best = None
    v_best = None
    for _ in range(max(1000 // comm.size, 10) if comm is not None else 1000):
        # for _ in range():
        if bath_guess is None and v_guess is None:
            if len(peaks) > 0:
                bath_index = rng.choice(
                    np.arange(len(peaks)),
                    size=min(len(peaks), bath_states_per_orbital),
                    p=normalised_scores,
                    replace=False,
                )
                bath_energies = w[peaks[bath_index]]
                bounds = [
                    (
                        w[max(0, int(np.floor(left_ips[i])))],
                        w[min(len(w) - 1, int(np.ceil(right_ips[i])))],
                    )
                    for i in bath_index
                ]
            else:
                bath_energies = []
                bounds = []
        else:
            bath_energies = bath_guess[::n_orb]
            bounds = [
                (
                    max(eb - (w[1] - w[0]) / 2, w[0]),
                    min(eb + (w[1] - w[0]) / 2, w[-1]),
                )
                for eb in bath_energies
            ]

        if v_guess is not None:
            v_guess = np.append(
                v_guess,
                np.random.rand(
                    max((bath_states_per_orbital - len(bath_energies)) * n_orb, 0),
                    n_orb,
                ),
                axis=0,
            )
        bath_energies = np.append(
            bath_energies,
            rng.uniform(
                low=w[0],
                high=w[-1],
                size=max(bath_states_per_orbital - len(bath_energies), 0),
            ),
        )
        bounds.extend([(w[0], w[-1])] * max(bath_states_per_orbital - len(bounds), 0))

        v, eb, cost = get_v_and_eb(
            w,
            delta,
            hyb,
            bath_energies,
            eb_bounds=bounds,
            gamma=gamma,
            imag_only=imag_only,
            realvalue_v=realvalue_v,
            scale_function=weight_fun,
            v_guess=v_guess,
        )
        bath_guess = None
        v_guess = None
        if abs(cost) < min_cost:
            eb_best = eb
            v_best = v
            min_cost = abs(cost)
    if comm is not None:
        bath_energies, v, _ = comm.allreduce(
            (eb_best, v_best, min_cost), op=MPI.Op.Create(v_opt, commute=True)
        )

    return bath_energies, v

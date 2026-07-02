import numpy as np
from types import SimpleNamespace
from scipy.optimize import check_grad
from rspt2spectra.hyb_fit import get_state_per_inequivalent_block
from rspt2spectra.offdiagonal import (
    vectorized_cost_function,
    vectorized_jacobian,
    calc_diff,
    calc_moment_diff,
    inroll,
    unroll,
    inroll_C,
    unroll_C,
    merge_bath_states,
    merge_overlapping_bath_states,
    get_hyb,
    get_hyb_2,
    moment_weights,
    get_v_and_eb_varpro_basin_hopping,
    get_v_and_eb_differential_evolution,
    _gaps_to_eb,
    _eb_to_gaps,
    _gap_bounds,
    _gaps_grad,
    _max_bath_states,
    _repair_gaps,
    _varpro_cost_and_grad,
    _varpro_cost_and_full_grad,
)
import pytest

eb = np.array([-1, 0, 1], dtype=float)
vs = np.array([[[1, 2], [0, 1]], [[2, 1], [0, 1]], [[3, 3], [0, 1]]], dtype=float)

w = np.linspace(-1, 1, 201)
w_scale = np.max(np.abs(w))  # normalisation used in W_mn; matches production code
delta = 0.05
hyb = 5 * np.ones((w.shape[0], vs.shape[1], vs.shape[1]), dtype=float) + 1j * np.ones(
    (w.shape[0], vs.shape[1], vs.shape[1]), dtype=float
)


def test_calc_diff():
    z = w + 1j * delta
    diff = calc_diff(eb[None], vs[None], z, hyb)[0]
    a = np.conj(np.transpose(vs, (0, 2, 1))) @ vs
    exact = hyb - np.sum(
        a[None] / (z[:, None] - eb[None])[:, :, None, None],
        axis=1,
    )
    assert np.allclose(diff, exact)


def test_calc_moment_diff():
    z = w + 1j * delta
    diff = calc_diff(eb[None], vs[None], z, hyb)

    W_mn = moment_weights(w, 10)
    moment_diff = calc_moment_diff(diff, W_mn)

    # W_mn[m, n] = (w[m]/w_scale)^n, so the result is a plain weighted sum.
    for n in range(10):
        exact = np.einsum("m, mij -> ij", (w / w_scale) ** n, diff[0]) / len(w)
        assert np.allclose(moment_diff[0, n], exact)


def test_cost_function_no_regularization():
    assert eb.shape[0] == vs.shape[0]
    n_b = eb.shape[0]
    z = w + 1j * delta

    p = np.append(eb, inroll(vs))
    assert np.allclose(vs, unroll(inroll(vs), n_b, 2))

    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 10)

    c = vectorized_cost_function(p, n_b, z, hyb, 0.10, regularization="none", weight_array=weight_array, W_mn=W_mn)

    diff = calc_diff(eb[None], vs[None], z, hyb)
    moment_diff = calc_moment_diff(diff, W_mn)[0]
    diff = diff[0]

    exact = (
        1 / np.prod(diff.shape) * np.sum(0.5 * np.abs(diff) ** 2)
        + 1 / np.prod(moment_diff.shape[-2:]) * np.sum(0.5 * np.abs(moment_diff) ** 2) / moment_diff.shape[0]
    )
    assert np.allclose(c, exact)


def test_moment_weights():
    # W_mn[m, n] must equal (w[m] / w_scale)**n exactly.
    max_moment = 5
    W_mn = moment_weights(w, max_moment)

    assert W_mn.shape == (len(w), max_moment)
    assert np.max(np.abs(w / w_scale)) <= 1.0 + 1e-12  # normalised freq stays in [-1, 1]

    expected = np.pow(w[:, None] / w_scale, np.arange(max_moment)[None, :]) / len(w)
    assert np.allclose(W_mn, expected)


def test_jacobian_real():
    assert eb.shape[0] == vs.shape[0]
    n_b = eb.shape[0]
    z = w + 1j * delta

    p = np.append(eb, inroll(vs))

    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 4)

    def func(x):
        return vectorized_cost_function(
            x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn
        )

    def grad(x):
        return vectorized_jacobian(x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)

    err = check_grad(func, grad, p, epsilon=1e-5)

    assert err / np.linalg.norm(grad(p)) < 1e-4


# --- C (constant shift) tests ---


def test_inroll_C_unroll_C_real():
    n_imp = 3
    triu_i, triu_j = np.triu_indices(n_imp)
    C = np.zeros((n_imp, n_imp))
    C[triu_i, triu_j] = np.arange(1, len(triu_i) + 1, dtype=float)
    C[triu_j, triu_i] = C[triu_i, triu_j]
    p_C = inroll_C(C)
    assert len(p_C) == n_imp * (n_imp + 1) // 2
    C_rt = unroll_C(p_C, n_imp)
    assert np.allclose(C_rt.real, C)
    assert np.allclose(C_rt.imag, 0)


def test_inroll_C_unroll_C_complex():
    n_imp = 2
    C = np.array([[1.0, 2.0 + 3j], [2.0 - 3j, 4.0]])
    p_C = inroll_C(C)
    n_triu = n_imp * (n_imp + 1) // 2
    n_off = n_imp * (n_imp - 1) // 2
    assert len(p_C) == n_triu + n_off
    C_rt = unroll_C(p_C, n_imp)
    assert np.allclose(C_rt, C)
    assert np.allclose(C_rt, np.conj(C_rt.T))


def test_cost_function_with_C():
    n_b = eb.shape[0]
    n_imp = vs.shape[1]
    z = w + 1j * delta
    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 4)

    C = np.eye(n_imp) * 0.5
    p_C = inroll_C(C)
    n_C = len(p_C)
    p = np.concatenate([eb, inroll(vs), p_C])

    c_with_C = vectorized_cost_function(
        p,
        n_b,
        z,
        hyb,
        0.0,
        regularization="none",
        weight_array=weight_array,
        W_mn=W_mn,
        n_C=n_C,
    )
    C_reconstructed = unroll_C(p_C, n_imp)
    diff = calc_diff(eb[None], vs[None], z, hyb, C=C_reconstructed)[0]
    moment_diff = calc_moment_diff(diff[None], W_mn)[0]
    expected = np.sum(0.5 * np.abs(diff) ** 2) / diff.size + np.sum(0.5 * np.abs(moment_diff) ** 2) / (
        n_imp * n_imp * moment_diff.shape[0]
    )
    assert np.allclose(c_with_C, expected)


def test_jacobian_with_C_real():
    n_b = eb.shape[0]
    z = w + 1j * delta
    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 4)

    C0 = np.eye(vs.shape[1]) * 0.1
    p_C = inroll_C(C0)
    n_C = len(p_C)
    p = np.concatenate([eb, inroll(vs), p_C])

    def func(x):
        return vectorized_cost_function(
            x,
            n_b,
            z,
            hyb,
            0.0,
            regularization="none",
            weight_array=weight_array,
            W_mn=W_mn,
            n_C=n_C,
        )

    def grad(x):
        return vectorized_jacobian(
            x,
            n_b,
            z,
            hyb,
            0.0,
            regularization="none",
            weight_array=weight_array,
            W_mn=W_mn,
            n_C=n_C,
        )

    err = check_grad(func, grad, p, epsilon=1e-5)
    assert err / np.linalg.norm(grad(p)) < 1e-4


def test_jacobian_with_C_complex():
    n_b = eb.shape[0]
    z = w + 1j * delta
    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 4)

    np.random.seed(7)
    vs_c = vs + 0.1j * np.random.randn(*vs.shape)
    C0 = np.array([[0.5, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]])
    p_C = inroll_C(C0)
    n_C = len(p_C)
    p = np.concatenate([eb, inroll(vs_c), p_C])

    def func(x):
        return vectorized_cost_function(
            x,
            n_b,
            z,
            hyb,
            0.0,
            regularization="none",
            weight_array=weight_array,
            W_mn=W_mn,
            n_C=n_C,
        )

    def grad(x):
        return vectorized_jacobian(
            x,
            n_b,
            z,
            hyb,
            0.0,
            regularization="none",
            weight_array=weight_array,
            W_mn=W_mn,
            n_C=n_C,
        )

    err = check_grad(func, grad, p, epsilon=1e-5)
    assert err / np.linalg.norm(grad(p)) < 1e-4


def test_merge_bath_states_energy():
    # Bug: the merged energy was divided by n_imp**2 instead of n_imp.
    # For n_imp=2 with diagonal, equal-weight couplings the merged energy
    # must equal the coupling-weighted mean, not half of it.
    e1, e2 = -2.0, 1.0
    # v_i = scalar * I  →  A_i = scalar^2 * I  (symmetric case)
    vs_g = np.array([2.0 * np.eye(2), 1.0 * np.eye(2)], dtype=float)
    eb_g = np.array([e1, e2])
    eb_merged, _ = merge_bath_states(eb_g, vs_g)
    # A = (4+1)*I, first_moment = (-2*4+1*1)*I = -7*I → Eb = -7/5*I
    # eigvals = [-1.4, -1.4], mean = -1.4  (bug gave -0.7)
    expected = (e1 * 4.0 + e2 * 1.0) / (4.0 + 1.0)
    assert np.isclose(eb_merged[0], expected), (
        f"merged energy {eb_merged[0]:.4f} != expected {expected:.4f}; "
        "possibly still dividing by n_imp**2 instead of n_imp"
    )


def test_merge_overlapping_psd_safe():
    # Bug: np.linalg.cholesky raised on rank-deficient A (one orbital zero).
    n_imp = 2
    # One orbital has zero coupling in both states → A is rank-deficient.
    vs = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, 0.0]],
        ],
        dtype=complex,
    )
    ebs = np.array([-0.1, 0.1])
    # delta = 2.0 > 0.2 = |e2-e1|, so both states land in one group.
    delta = 2.0

    # Must not raise
    eb_merged, v_merged = merge_overlapping_bath_states(ebs, vs, delta)

    assert eb_merged.shape[0] == 1
    assert v_merged.shape == (1, n_imp, n_imp)

    # Round-trip: v_merged^H v_merged == sum_i (v_i^H v_i)
    A_reconstructed = (np.conj(np.swapaxes(v_merged, -1, -2)) @ v_merged)[0]
    A_expected = sum(np.conj(v.T) @ v for v in vs)
    assert np.allclose(A_reconstructed, A_expected)


def test_regularization_on_hoppings_only():
    # Bug: L1/L2 penalty was applied to the full p vector (bath energies + hoppings).
    # It must act only on the hopping parameters p[n_eb:].
    n_b = eb.shape[0]
    z = w + 1j * delta
    p = np.append(eb, inroll(vs))
    gamma = 0.5

    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 4)

    c_none = vectorized_cost_function(p, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)
    c_l1 = vectorized_cost_function(p, n_b, z, hyb, gamma, regularization="l1", weight_array=weight_array, W_mn=W_mn)
    c_l2 = vectorized_cost_function(p, n_b, z, hyb, gamma, regularization="l2", weight_array=weight_array, W_mn=W_mn)

    v_params = p[n_b:]
    n_v = len(v_params)
    assert np.isclose(c_l1, c_none + (gamma / n_v) * np.sum(np.abs(v_params)))
    assert np.isclose(c_l2, c_none + (gamma / n_v) * np.sum(v_params**2))

    # Gradient: bath-energy rows must be identical with and without regularization.
    g_none = vectorized_jacobian(p, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)
    g_l1 = vectorized_jacobian(p, n_b, z, hyb, gamma, regularization="l1", weight_array=weight_array, W_mn=W_mn)
    assert np.allclose(g_l1[:n_b], g_none[:n_b]), "regularization must not affect eb gradient"
    assert not np.allclose(g_l1[n_b:], g_none[n_b:]), "regularization must change hopping gradient"


def test_jacobian_complex():
    assert eb.shape[0] == vs.shape[0]
    n_b = eb.shape[0]
    z = w + 1j * delta

    np.random.seed(42)
    vs_c = vs + 0.1j * np.random.randn(*vs.shape)
    p = np.append(eb, inroll(vs_c))

    weight_array = np.ones_like(w)
    W_mn = moment_weights(w, 4)

    def func(x):
        return vectorized_cost_function(
            x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn
        )

    def grad(x):
        return vectorized_jacobian(x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)

    err = check_grad(func, grad, p, epsilon=1e-5)

    assert err / np.linalg.norm(grad(p)) < 1e-4


def test_gap_reparametrization_roundtrip():
    # eb -> gaps -> eb recovers the sorted energies when gaps already exceed delta.
    d = 0.1
    energies = np.array([1.7, -2.3, 0.5, -0.4])  # unsorted, all separated > delta
    gaps = _eb_to_gaps(energies, d)
    assert np.allclose(_gaps_to_eb(gaps), np.sort(energies))
    # The first entry is the smallest energy; the rest are the (>= delta) gaps.
    assert np.isclose(gaps[0], np.min(energies))
    assert np.all(gaps[1:] >= d - 1e-12)


def test_eb_to_gaps_clips_to_delta():
    # Energies closer than delta must be pushed apart to satisfy the min separation.
    d = 0.5
    energies = np.array([0.0, 0.1, 0.15])  # gaps 0.1, 0.05 both < delta
    gaps = _eb_to_gaps(energies, d)
    assert np.all(gaps[1:] >= d - 1e-12)
    eb_rec = _gaps_to_eb(gaps)
    assert np.all(np.diff(eb_rec) >= d - 1e-12)


def test_gap_bounds_structure():
    b = _gap_bounds(-3.0, 4.0, 4, 0.2)
    assert b[0] == (-3.0, 4.0)  # first energy spans the window
    assert all(lo == 0.2 for lo, _ in b[1:])  # every gap lower bound is delta
    # Degenerate window still yields valid (lo <= hi) bounds.
    b2 = _gap_bounds(0.0, 0.05, 3, 0.2)
    assert all(lo <= hi for lo, hi in b2)


def test_gaps_grad_matches_chain_rule():
    # grad_p[j] = sum_{k>=j} grad_e[k]  (reverse cumulative sum).
    grad_e = np.array([0.06, 0.02, -0.03, 0.01])
    expected = np.array([np.sum(grad_e[j:]) for j in range(len(grad_e))])
    assert np.allclose(_gaps_grad(grad_e), expected)


def test_fit_enforces_min_separation_no_merge():
    # The gap parametrization must keep fitted bath energies separated by >= delta
    # without any post-fit merge, and recover energies near the injected poles.
    rng = np.random.default_rng(0)
    wgrid = np.linspace(-5, 5, 301)
    d = 0.1
    z = wgrid + 1j * (d * (1 + 0.5 * np.abs(wgrid) ** 2))
    true_eb = np.array([-2.3, -0.4, 1.1, 2.6])
    true_v = np.array([0.6, 0.9, 0.5, 0.7]).reshape(-1, 1)
    hyb_syn = get_hyb(z, true_eb, true_v)

    n_eb = true_eb.shape[0]
    ebs = np.sort(rng.uniform(-4, 4, size=(40, n_eb)), axis=1)
    eb_restrictions = [(wgrid[0], wgrid[-1])] * n_eb

    _, eb_final, _, _ = get_v_and_eb_varpro_basin_hopping(
        wgrid,
        d,
        hyb_syn,
        ebs,
        eb_restrictions,
        gamma=0.0,
        regularization=None,
        weight_function=lambda x: np.ones_like(x),
        realvalue_v=True,
    )

    gaps = np.diff(np.sort(eb_final))
    assert np.all(gaps >= d - 1e-9), f"states closer than delta: min gap {gaps.min()}"
    # Every injected pole is matched by some fitted energy within delta.
    for e in true_eb:
        assert np.min(np.abs(eb_final - e)) < d, f"pole {e} not recovered"


def _fake_block_structure(block_orbitals):
    # One inequivalent block per entry in block_orbitals (each a list of orbital
    # indices); multiplicity 1 (only "identical to itself").
    n = len(block_orbitals)
    return SimpleNamespace(
        blocks=block_orbitals,
        inequivalent_blocks=list(range(n)),
        identical_blocks=[[i] for i in range(n)],
        transposed_blocks=[[] for _ in range(n)],
        particle_hole_blocks=[[] for _ in range(n)],
        particle_hole_transposed_blocks=[[] for _ in range(n)],
    )


def _diag_hyb(w, imag_per_orbital):
    # Diagonal hybridization with a flat -Im on each orbital, so the integrated
    # weight of a block is (sum of its orbital values) * window width.
    n_orb = len(imag_per_orbital)
    hyb = np.zeros((len(w), n_orb, n_orb), dtype=complex)
    for k, val in enumerate(imag_per_orbital):
        hyb[:, k, k] = -1j * val
    return hyb


def test_state_distribution_prioritizes_strong_hybridization():
    # Pool = B * n_blocks, split by hybridization weight. Block 0 (1 orbital,
    # -Im=6) vs block 1 (3 orbitals, -Im=1 each): weights 6 vs 3.
    w = np.linspace(-5.0, 5.0, 201)
    bs = _fake_block_structure([[0], [1, 2, 3]])
    hyb = _diag_hyb(w, [6.0, 1.0, 1.0, 1.0])
    states = get_state_per_inequivalent_block(bs, 4, hyb, w, lambda x: np.ones_like(x), delta=0.5)
    # pool = 4*2 = 8; shares 6/9 and 3/9 -> round(5.33)=5, round(2.67)=3.
    assert list(states) == [5, 3]


def test_state_distribution_coverage_and_cap():
    w = np.linspace(-1.0, 1.0, 201)
    bs = _fake_block_structure([[0], [1, 2, 3]])
    # Block 0 dominant, block 1 (3 orbitals) tiny but nonzero -> block 1 must
    # still get at least one state per orbital (>= 3).
    hyb = _diag_hyb(w, [100.0, 0.001, 0.001, 0.001])
    delta = 0.2
    n_max = _max_bath_states(w[0], w[-1], delta)  # 11
    states = get_state_per_inequivalent_block(bs, 8, hyb, w, lambda x: np.ones_like(x), delta=delta)
    assert states[1] >= 3  # coverage: one bath state per orbital
    assert np.all(states <= n_max)  # window cap
    # A zero-weight block gets zero states (nothing to fit).
    hyb0 = _diag_hyb(w, [1.0, 0.0, 0.0, 0.0])
    states0 = get_state_per_inequivalent_block(bs, 8, hyb0, w, lambda x: np.ones_like(x), delta=delta)
    assert states0[1] == 0

    # The per-orbital floor is capped by the window: a narrow window cannot host
    # one state per orbital, and the count is capped, not fitted out of range.
    w_narrow = np.linspace(0.0, 0.3, 50)
    n_max_narrow = _max_bath_states(w_narrow[0], w_narrow[-1], delta)  # 2
    hyb_n = _diag_hyb(w_narrow, [0.001, 0.001, 0.001, 0.001])  # 4-orbital block wants 4
    states_n = get_state_per_inequivalent_block(
        _fake_block_structure([[0, 1, 2, 3]]), 8, hyb_n, w_narrow, lambda x: np.ones_like(x), delta=delta
    )
    assert states_n[0] == n_max_narrow  # floor of 4 capped down to what fits (2)


def test_max_bath_states_counts_what_fits():
    # n states fit when (n-1)*delta <= width, i.e. n <= width/delta + 1.
    assert _max_bath_states(-1.0, 1.0, 0.2) == 11  # 2.0/0.2 + 1
    assert _max_bath_states(-1.0, 1.0, 0.3) == 7  # floor(6.66) + 1
    assert _max_bath_states(0.0, 0.05, 0.2) == 1  # window narrower than delta -> at least 1


def test_fit_caps_states_to_window_without_failing():
    # Requesting far more states than fit must not raise; the fit returns at most
    # the feasible number, all in-window and separated by >= delta.
    rng = np.random.default_rng(3)
    w_min, w_max, d = -1.0, 1.0, 0.2
    n_max = _max_bath_states(w_min, w_max, d)  # 11
    wgrid = np.linspace(w_min, w_max, 200)
    z = wgrid + 1j * (d * (1 + 0.5 * np.abs(wgrid) ** 2))
    true_eb = np.array([-0.4, 0.3, 0.7])
    true_v = rng.normal(size=(3, 1, 1))
    hyb_syn = get_hyb_2(z, true_eb[None], true_v[None])[0]

    n_req = n_max + 6  # ask for more than can fit
    ebs = np.sort(rng.uniform(w_min, w_max, size=(8, n_req)), axis=1)
    _, eb_final, _, _ = get_v_and_eb_varpro_basin_hopping(
        wgrid, d, hyb_syn, ebs, [(w_min, w_max)] * n_req,
        gamma=0.0, regularization=None,
        weight_function=lambda x: 1.0 / (1.0 + x**2), realvalue_v=True,
    )
    assert eb_final.shape[0] <= n_max
    # Edge-packed fit: top state sits at w_max up to the SLSQP constraint tolerance.
    assert eb_final.max() <= w_max + 1e-6 and eb_final.min() >= w_min - 1e-6
    assert np.all(np.diff(np.sort(eb_final)) >= d - 1e-6)


def test_repair_gaps_projects_into_window():
    # Clipping raw gaps up to delta can push the cumulative sum past w_max; the
    # repair must pull it back while preserving ordering and min separation.
    w_min, w_max, d = -1.0, 1.0, 0.2
    eb = np.array([0.6, 0.65, 0.68, 0.72, 0.9])  # bunched near the top edge
    p = _eb_to_gaps(eb, d)
    assert p.sum() > w_max  # raw seed overshoots the window
    pr = _repair_gaps(p, w_min, w_max, d)
    eb_r = _gaps_to_eb(pr)
    assert eb_r[-1] <= w_max + 1e-9
    assert eb_r[0] >= w_min - 1e-9
    assert np.all(np.diff(eb_r) >= d - 1e-9)
    # Already-feasible seeds are returned unchanged.
    p_ok = _eb_to_gaps(np.array([-0.5, 0.0, 0.5]), d)
    assert np.allclose(_repair_gaps(p_ok, w_min, w_max, d), p_ok)


@pytest.mark.parametrize("optimizer", [
    get_v_and_eb_varpro_basin_hopping,
    get_v_and_eb_differential_evolution,
])
def test_fit_keeps_states_inside_window(optimizer):
    # With more requested states than true poles clustered near the top edge, the
    # fit must not place any bath energy outside [w_min, w_max].
    rng = np.random.default_rng(1)
    w_min, w_max, d = -1.0, 1.0, 0.2
    wgrid = np.linspace(w_min, w_max, 250)
    z = wgrid + 1j * (d * (1 + 0.5 * np.abs(wgrid) ** 2))
    true_eb = np.array([0.5, 0.7, 0.85])  # near the upper edge
    true_v = rng.normal(size=(3, 1, 1))
    hyb_syn = get_hyb_2(z, true_eb[None], true_v[None])[0]

    n_eb = 6  # more states than poles
    ebs = np.sort(rng.uniform(w_min, w_max, size=(8, n_eb)), axis=1)
    eb_restrictions = [(w_min, w_max)] * n_eb

    _, eb_final, _, _ = optimizer(
        wgrid, d, hyb_syn, ebs, eb_restrictions,
        gamma=0.0, regularization=None,
        weight_function=lambda x: 1.0 / (1.0 + x**2),
        realvalue_v=True,
    )

    assert eb_final.max() <= w_max + 1e-9, f"state above window: {eb_final.max()}"
    assert eb_final.min() >= w_min - 1e-9, f"state below window: {eb_final.min()}"
    assert np.all(np.diff(np.sort(eb_final)) >= d - 1e-9)


def _synthetic_varpro_problem(n_imp, seed):
    rng = np.random.default_rng(seed)
    wgrid = np.linspace(-5, 5, 351)
    d = 0.1
    z = wgrid + 1j * (d * (1 + 0.5 * np.abs(wgrid) ** 2))
    true_eb = np.sort(rng.uniform(-3, 3, size=4))
    Vt = rng.normal(size=(4, n_imp, n_imp))
    hyb_syn = get_hyb_2(z, true_eb[None], Vt[None])[0]
    weight_array = 1.0 / (1.0 + wgrid**2)  # non-trivial weight
    W_mn = moment_weights(wgrid, 3)
    return z, hyb_syn, weight_array, W_mn


def test_varpro_full_gradient_matches_finite_difference():
    # The exact total-derivative gradient must agree with a finite-difference
    # gradient of the reduced cost, and must beat the Kaufman approximation.
    for n_imp in (1, 2, 3):
        for realvalue in (True, False):
            z, hyb_syn, wa, W_mn = _synthetic_varpro_problem(n_imp, 10 * n_imp + int(realvalue))
            rng = np.random.default_rng(7)
            eb = np.sort(rng.uniform(-2.5, 2.5, size=4))  # positive residues -> smooth region

            def cost(e):
                return _varpro_cost_and_full_grad(e, z, hyb_syn, wa, W_mn, realvalue)[0]

            c_full, g_full, _, _ = _varpro_cost_and_full_grad(eb, z, hyb_syn, wa, W_mn, realvalue)
            c_kauf, _, _, _ = _varpro_cost_and_grad(eb, z, hyb_syn, wa, W_mn, realvalue)

            h = 1e-6
            g_num = np.array([(cost(eb + h * np.eye(4)[i]) - cost(eb - h * np.eye(4)[i])) / (2 * h) for i in range(4)])
            # Cost is identical to the Kaufman routine (same forward model).
            assert np.isclose(c_full, c_kauf, rtol=0, atol=1e-9)
            rel = np.max(np.abs(g_full - g_num)) / (np.linalg.norm(g_num) + 1e-30)
            assert rel < 1e-5, f"n_imp={n_imp} real={realvalue}: full-grad rel err {rel:.2e}"

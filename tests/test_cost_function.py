import numpy as np
from scipy.optimize import check_grad
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
    get_hyb_2,
    moment_weights,
)


eb = np.array([-1, 0, 1], dtype=float)
vs = np.array([[[1, 2], [0, 1]], [[2, 1], [0, 1]], [[3, 3], [0, 1]]], dtype=float)

w = np.linspace(-1, 1, 201)
w_scale = np.max(np.abs(w))  # normalisation used in W_mn; matches production code
delta = 0.05
hyb = 5 * np.ones((w.shape[0], vs.shape[1], vs.shape[1]), dtype=float) + 1j * np.ones((w.shape[0], vs.shape[1], vs.shape[1]), dtype=float)


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

    exact = 1 / np.prod(diff.shape) * np.sum(0.5 * np.abs(diff)**2) + 1 / np.prod(
        moment_diff.shape[-2:]
    ) * np.sum(0.5 * np.abs(moment_diff)**2) / moment_diff.shape[0]
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
        return vectorized_cost_function(x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)

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
        p, n_b, z, hyb, 0.0, regularization="none",
        weight_array=weight_array, W_mn=W_mn, n_C=n_C,
    )
    C_reconstructed = unroll_C(p_C, n_imp)
    diff = calc_diff(eb[None], vs[None], z, hyb, C=C_reconstructed)[0]
    moment_diff = calc_moment_diff(diff[None], W_mn)[0]
    expected = (
        np.sum(0.5 * np.abs(diff) ** 2) / diff.size
        + np.sum(0.5 * np.abs(moment_diff) ** 2) / (n_imp * n_imp * moment_diff.shape[0])
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
            x, n_b, z, hyb, 0.0, regularization="none",
            weight_array=weight_array, W_mn=W_mn, n_C=n_C,
        )

    def grad(x):
        return vectorized_jacobian(
            x, n_b, z, hyb, 0.0, regularization="none",
            weight_array=weight_array, W_mn=W_mn, n_C=n_C,
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
            x, n_b, z, hyb, 0.0, regularization="none",
            weight_array=weight_array, W_mn=W_mn, n_C=n_C,
        )

    def grad(x):
        return vectorized_jacobian(
            x, n_b, z, hyb, 0.0, regularization="none",
            weight_array=weight_array, W_mn=W_mn, n_C=n_C,
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
    vs = np.array([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.5, 0.0], [0.0, 0.0]],
    ], dtype=complex)
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

    c_none = vectorized_cost_function(
        p, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn
    )
    c_l1 = vectorized_cost_function(
        p, n_b, z, hyb, gamma, regularization="l1", weight_array=weight_array, W_mn=W_mn
    )
    c_l2 = vectorized_cost_function(
        p, n_b, z, hyb, gamma, regularization="l2", weight_array=weight_array, W_mn=W_mn
    )

    v_params = p[n_b:]
    n_v = len(v_params)
    assert np.isclose(c_l1, c_none + (gamma / n_v) * np.sum(np.abs(v_params)))
    assert np.isclose(c_l2, c_none + (gamma / n_v) * np.sum(v_params ** 2))

    # Gradient: bath-energy rows must be identical with and without regularization.
    g_none = vectorized_jacobian(
        p, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn
    )
    g_l1 = vectorized_jacobian(
        p, n_b, z, hyb, gamma, regularization="l1", weight_array=weight_array, W_mn=W_mn
    )
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
        return vectorized_cost_function(x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)

    def grad(x):
        return vectorized_jacobian(x, n_b, z, hyb, 0.0, regularization="none", weight_array=weight_array, W_mn=W_mn)

    err = check_grad(func, grad, p, epsilon=1e-5)

    assert err / np.linalg.norm(grad(p)) < 1e-4

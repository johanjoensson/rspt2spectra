#!/usr/bin/env python3

"""Fit discrete bath models to (off-diagonal) hybridization functions.

The active optimizers are `get_v_and_eb_varpro_basin_hopping` and
`get_v_and_eb_differential_evolution`: both search only over bath energies
(reparametrized as gaps to enforce ordering and minimum separation) while the
hopping residues and a constant Hermitian shift are solved analytically at
each step (VARPRO), followed by a joint SLSQP polish using the analytic
Jacobian of `vectorized_cost_function`.
"""

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    basinhopping,
    differential_evolution,
    minimize,
)


def get_hyb(z, eb, v):
    """
    Return the hybridization functions, as a rank 3 tensor.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh.
    eb : array(B)
        Bath energies.
    v : array(B, N)
        Hopping parameters.

    Returns
    -------
    hyb : array(M, N,N,)
        Hybridization functions.

    """
    n_w = len(z)
    n_imp = v.shape[1]
    hyb = np.zeros((n_w, n_imp, n_imp), dtype=complex)

    # Loop over all bath energies
    for b, e in enumerate(eb):
        # Add contributions from each bath
        hyb[:] += np.outer(v[b].conj(), v[b])[np.newaxis, ...] * (1 / (z - e))[:, np.newaxis, np.newaxis]

    return hyb


def get_hyb_2(z, eb, v, C=None):
    """
    Return the hybridization functions, as a rank 3 tensor.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh.
    eb : array(S, N_B)
        Bath energies.
    v : array(S, N_B, N, N)
        Hopping parameters.
    C : array(N, N), optional
        Constant Hermitian shift added to every frequency point.

    Returns
    -------
    hyb : array(S, M, N,N)
        Hybridization functions.

    """
    A = np.conj(np.transpose(v, (0, 1, 3, 2))) @ v  # (S, N_B, N, N)
    G = 1.0 / (z[np.newaxis, :, np.newaxis] - eb[:, np.newaxis, :])  # (S, M, N_B)
    result = np.einsum("smb,sbij->smij", G, A)
    if C is not None:
        # single (n_imp, n_imp) broadcasts over S and M; batched (S, n_imp, n_imp) over M only.
        result = result + (C[np.newaxis, np.newaxis] if C.ndim == 2 else C[:, np.newaxis])
    return result


def unroll(p, n_b, n_imp):
    """
    Return hybridization parameters as a matrix.

    Parameters
    ----------
    p : real array(K, S)
        Hybridization parameters as a stack of vectors.
    n_b : int
        Number of bath orbitals.
    n_imp : int
        Number of impurity orbitals.

    Returns
    -------
    v : complex array(S, n_b, n_imp)
        Hybridization parameters as a matrix.

    """
    onedimensional = len(p.shape) == 1
    p_c = p
    triu_rows, triu_columns = np.triu_indices(n_imp)
    r = p.shape[0]
    if r != n_b * len(triu_columns):
        # The real parts are the first r elements in p
        # the imaginary parts are the rest
        r //= 2
        p_c = p[:r] + 1j * p[r:]
    if onedimensional:
        # non_zero_indices = np.ix_(range(n_b), triu_rows, triu_columns)
        res = np.zeros((n_b, n_imp, n_imp), dtype=complex)
        res[:, triu_rows, triu_columns] = p_c.reshape((n_b, len(triu_columns)))
        return res
    # non_zero_indices = np.ix_(range(n_b), triu_rows, triu_columns, range(p.shape[1]))
    res = np.zeros((n_b, n_imp, n_imp, p.shape[1]), dtype=complex)
    res[:, triu_rows, triu_columns] = p_c.reshape((n_b, len(triu_columns), p.shape[1]))
    return np.moveaxis(res, -1, 0)


def inroll(v):
    """
    Return hybridization parameters as a vector.

    Parameters
    ----------
    v : complex array(..., n_b, n_imp, n_imp)
        Hybridization parameters as a matrix.

    Returns
    -------
    p : real array(..., K)
        Hybridization parameters as a stack of vectors.

    """
    triu_rows, triu_columns = np.triu_indices(v.shape[-1])
    cplx = np.any(np.abs(v.imag)) > 0

    # res_shape = v.shape[:-3] + (v.shape[-3] * v.shape[-2] * v.shape[-1],)
    if cplx:
        return np.moveaxis(
            np.append(
                v[..., triu_rows, triu_columns].real,
                v[..., triu_rows, triu_columns].imag,
                axis=-1,
            ).reshape(v.shape[:-3] + (-1,)),
            0,
            -1,
        )
    return np.moveaxis(v[..., triu_rows, triu_columns].real.reshape(v.shape[:-3] + (-1,)), 0, -1)


def inroll_C(C):
    """Pack a Hermitian (n_imp x n_imp) matrix into a real vector.

    Diagonal entries are real; off-diagonal upper-triangle entries contribute
    both real and imaginary parts.  Returns a 1-D real array of length
    n_imp*(n_imp+1)//2 (real-symmetric) or n_imp^2 (complex Hermitian).
    """
    n_imp = C.shape[0]
    triu_i, triu_j = np.triu_indices(n_imp)
    diag_mask = triu_i == triu_j
    off_mask = ~diag_mask
    entries = C[triu_i, triu_j]
    if np.any(np.abs(entries.imag) > 1e-14):
        # real: all upper-triangle real parts; imag: off-diagonal imaginary parts only
        return np.concatenate([entries.real, entries[off_mask].imag])
    return entries.real.copy()


def unroll_C(p_C, n_imp):
    """Unpack a real vector (from inroll_C) into a Hermitian (n_imp x n_imp) matrix."""
    triu_i, triu_j = np.triu_indices(n_imp)
    n_triu = len(triu_i)
    off_mask = triu_i != triu_j
    C = np.zeros((n_imp, n_imp), dtype=complex)
    if len(p_C) == n_triu:  # real-symmetric
        C[triu_i, triu_j] = p_C
        C[triu_j, triu_i] = p_C  # symmetric
    else:  # complex Hermitian: n_triu real + n_off imaginary
        entries = p_C[:n_triu].astype(complex)
        entries[off_mask] += 1j * p_C[n_triu:]
        C[triu_i, triu_j] = entries
        C[triu_j[off_mask], triu_i[off_mask]] = np.conj(entries[off_mask])
        np.fill_diagonal(C, C.diagonal().real)
    return C


def merge_bath_states(ebs, vs):
    r"""Merge a group of bath states into one effective state.

    The merged coupling matrix ``A`` preserves the total zeroth spectral
    moment, :math:`A = \sum_k V_k^\dagger V_k`, and the merged energy is the
    coupling-weighted mean solved from the first moments,
    :math:`E_b = A^{+} \sum_k e_k V_k^\dagger V_k` (pseudo-inverse; safe for
    rank-deficient ``A``), averaged over its eigenvalues.

    Parameters
    ----------
    ebs : (n,) np.ndarray
        Bath energies to merge.
    vs : (n, n_imp, n_imp) np.ndarray
        Hopping matrices of the states.

    Returns
    -------
    eb : (1,) np.ndarray
        The merged bath energy.
    A : (1, n_imp, n_imp) np.ndarray
        The merged coupling matrix :math:`V^\dagger V`.
    """
    n_imp = vs.shape[1]
    sorted_idx = np.unravel_index(np.argsort(np.linalg.norm(vs, axis=(-2, -1))), vs.shape[:-2])
    vs = vs[sorted_idx]
    ebs = ebs[sorted_idx]

    zeroth_moments = np.conj(np.transpose(vs, (0, 2, 1))) @ vs
    first_moments = ebs[..., np.newaxis, np.newaxis] * zeroth_moments
    A = np.sum(zeroth_moments, axis=0)
    # Use eigh-based pseudoinverse: safe when A is rank-deficient (zero-coupling orbitals).
    lam_A, U_A = np.linalg.eigh(A)
    tol = np.max(np.abs(lam_A)) * n_imp * np.finfo(float).eps * 1e4
    inv_lam = np.where(np.abs(lam_A) > tol, 1.0 / np.where(np.abs(lam_A) > tol, lam_A, 1.0), 0.0)
    Eb = (U_A * inv_lam) @ (np.conj(U_A.T) @ np.sum(first_moments, axis=0))
    eb = np.mean(np.linalg.eigvals(Eb).real)
    return eb.real[None], A[None]


def merge_overlapping_bath_states(ebs, vs, delta):
    """Merge bath states closer than the broadening into single states.

    Two bath states whose separation is less than ``delta`` (the HWHM of the
    Lorentzian broadening) have overlapping Lorentzians with no local minimum
    between them and are indistinguishable by the fit; each such group is
    collapsed with `merge_bath_states`.

    Parameters
    ----------
    ebs : (n,) np.ndarray
        Bath energies.
    vs : (n, n_imp, n_imp) np.ndarray
        Hopping matrices.
    delta : float
        Minimum separation below which states are merged.

    Returns
    -------
    eb_merged : (m,) np.ndarray
        Merged bath energies, sorted ascending.
    v_merged : (m, n_imp, n_imp) np.ndarray
        Hopping factors ``V`` with ``V^H V`` equal to the merged coupling
        matrices (PSD-safe factorization; rank-deficient groups are fine).
    """
    n_imp = vs.shape[2]
    sorted_idx = np.argsort(ebs)
    ebs = ebs[sorted_idx]
    vs = vs[sorted_idx]
    e_diff = np.diff(ebs)
    # delta is the HWHM of the Lorentzian broadening.  Two bath states whose
    # separation is less than delta have overlapping Lorentzians with no local
    # minimum between them and are indistinguishable by the fit.
    split_indices = 1 + np.nonzero(e_diff > delta)[0]
    A_merged = np.empty((0, n_imp, n_imp), dtype=vs.dtype, order="F")
    eb_merged = np.empty((0), dtype=float, order="F")
    for v_g, eb_g in zip(np.split(vs, split_indices), np.split(ebs, split_indices)):
        if v_g.shape[0] == 1:
            Am = (np.conj(np.transpose(v_g, (0, -1, -2))) @ v_g).reshape((1, n_imp, n_imp), order="F")
            em = eb_g
        else:
            em, Am = merge_bath_states(eb_g, v_g)
        eb_merged = np.append(eb_merged, em, axis=0)
        A_merged = np.append(A_merged, Am, axis=0)
    lam, U = np.linalg.eigh(A_merged)
    lam = np.clip(lam, 0.0, None)
    # V such that V^H V = A_merged, safe for rank-deficient A.
    return eb_merged, np.sqrt(lam)[:, :, None] * np.conj(np.swapaxes(U, -1, -2))


def moment_weights(w, max_moment):
    r"""
    Weights for the scaled spectral moments used in the cost function.

    Returns W_mn such that
        (W_mn.T @ f)[n] == (1/M) * sum_m (w[m] / w_scale)^n * f[m]
    i.e. the sample mean of (w/w_scale)^n * f.  Dividing by M keeps the
    moment contribution O(1) regardless of the mesh size, matching the
    scale of the old Simpson-integral formulation.  w_scale = max(|w|)
    keeps the normalised frequency in [-1, 1] so high-order terms don't
    blow up.

    Pre-computing W_mn outside the optimisation loop avoids repeated
    exponentiation inside the cost function.

    Parameters
    ----------
    w : array(M)
        Real frequency mesh.
    max_moment : int
        Number of moments (powers 0 .. max_moment - 1).

    Returns
    -------
    W_mn : array(M, max_moment)
    """
    w_scale = np.max(np.abs(w))
    return np.pow(w[:, None] / w_scale, np.arange(max_moment)[None, :]) / len(w)


def _gaps_to_eb(p):
    """Reconstruct sorted absolute bath energies from the gap parametrization.

    p = [e_0, g_1, ..., g_{n-1}] -> eb = cumsum(p) = [e_0, e_0+g_1, ...].
    """
    return np.cumsum(p)


def _eb_to_gaps(eb, delta):
    """Inverse of `_gaps_to_eb`: sorted energies -> [first energy, gaps].

    Gaps are clipped to be at least `delta` so that a seed built from arbitrary
    energies already satisfies the minimum-separation constraint.
    """
    p = np.diff(np.sort(eb), prepend=0.0)
    if p.shape[0] > 1:
        p[1:] = np.maximum(p[1:], delta)
    return p


def _gap_bounds(w_min, w_max, n, delta):
    """Box bounds for the gap parametrization.

    The first energy lives in the frequency window `[w_min, w_max]`; every
    subsequent gap lives in `[delta, window width]`.  The `delta` lower bound is
    what enforces the minimum separation (and hence the ordering) of the states.
    """
    upper = max(w_max - w_min, delta)
    return [(w_min, w_max)] + [(delta, upper)] * (n - 1)


def _gaps_grad(grad_e):
    """Map a gradient w.r.t. absolute energies to one w.r.t. gap parameters.

    Since eb = cumsum(p), de_k/dp_j = 1 for j <= k, so
    grad_p[j] = sum_{k>=j} grad_e[k] -- a reverse cumulative sum.
    """
    return np.cumsum(grad_e[::-1])[::-1]


def _max_bath_states(w_min, w_max, delta):
    """Largest number of states that fit in `[w_min, w_max]` separated by `delta`.

    The gap parametrization can place `n` ordered states separated by at least
    `delta` iff ``(n-1)*delta <= w_max - w_min``, i.e. ``n <= (w_max-w_min)/delta
    + 1``.  Callers cap their requested count at this value instead of failing,
    so a too-large request is honoured as "as many as reasonably fit".
    """
    return max(1, int(np.floor((w_max - w_min) / delta + 1e-9)) + 1)


def _gap_sum_upper(n_eb, total_len, w_max):
    """Linear inequality keeping the largest bath energy at or below `w_max`.

    The top energy is ``e_{n-1} = sum(p[:n_eb])`` (cumulative sum of the gap
    vector), so the constraint is ``w_max - sum(p[:n_eb]) >= 0``.  Box bounds on
    the gaps alone cannot cap this sum, hence the explicit constraint.  Returned
    as an SLSQP-style dict; `total_len` covers polish vectors that also carry V/C.
    """
    grad = np.zeros(total_len)
    grad[:n_eb] = -1.0
    return {
        "type": "ineq",
        "fun": lambda p: w_max - np.sum(p[:n_eb]),
        "jac": lambda p: grad,
    }


def _repair_gaps(p, w_min, w_max, delta):
    """Project a gap seed onto the feasible region (in-window, min-separated).

    Clipping raw energies to gaps >= delta can push the cumulative sum past
    `w_max`; this pulls it back by first lowering the leading energy toward
    `w_min`, then shrinking the surplus gap slack above `delta` proportionally.
    Assumes feasibility (n states fit in the window, see `_max_bath_states`),
    under which the repair always succeeds.
    """
    p = np.array(p, dtype=float)
    p[0] = np.clip(p[0], w_min, w_max)
    if p.shape[0] > 1:
        p[1:] = np.maximum(p[1:], delta)
    excess = p.sum() - w_max
    if excess <= 0:
        return p
    cut0 = min(excess, p[0] - w_min)
    p[0] -= cut0
    excess -= cut0
    if excess > 0 and p.shape[0] > 1:
        slack = p[1:] - delta
        total_slack = slack.sum()
        if total_slack > 0:
            p[1:] -= slack * (min(excess, total_slack) / total_slack)
    return p


def _gap_slsqp_polish(gap_x, z, hyb, gamma, regularization, weight_array, W_mn, realvalue_v, gap_bounds):
    """SLSQP refinement of a gap-parametrized bath fit over eb, V and C jointly.

    `gap_x` is the converged gap vector [first energy, gaps].  The eb block stays
    gap-parametrized during the polish (via local cost/Jacobian wrappers around
    the shared vectorized functions) so the minimum-separation constraint cannot
    be violated.  Returns (v_final, eb_final, C_final, c_final).
    """
    n_eb = len(gap_x)
    n_imp = hyb.shape[1]

    eb_opt = _gaps_to_eb(gap_x)
    _, V_opt, _, C_opt = _varpro_inner_solve(eb_opt, z, hyb, realvalue_v)

    p_C0 = inroll_C(C_opt)
    n_C = len(p_C0)
    p0 = np.concatenate([gap_x, inroll(V_opt), p_C0])
    bounds = gap_bounds + [(None, None)] * (len(p0) - n_eb)
    w_max = gap_bounds[0][1]

    def _cost(p):
        p_abs = np.concatenate([_gaps_to_eb(p[:n_eb]), p[n_eb:]])
        return vectorized_cost_function(p_abs, n_eb, z, hyb, gamma, regularization, weight_array, W_mn, n_C)

    def _jac(p):
        p_abs = np.concatenate([_gaps_to_eb(p[:n_eb]), p[n_eb:]])
        J = vectorized_jacobian(p_abs, n_eb, z, hyb, gamma, regularization, weight_array, W_mn, n_C)
        J[:n_eb] = _gaps_grad(J[:n_eb])
        return J

    res = minimize(
        _cost,
        p0,
        method="SLSQP",
        jac=_jac,
        tol=1e-8,
        options={"maxiter": 1000},
        bounds=bounds,
        constraints=[_gap_sum_upper(n_eb, len(p0), w_max)],
    )

    p = res.x
    eb_final = _gaps_to_eb(p[:n_eb])
    v_final = unroll(p[n_eb:-n_C], n_eb, n_imp)
    C_final = unroll_C(p[-n_C:], n_imp)
    c_final = float(
        vectorized_cost_function(
            np.concatenate([eb_final, p[n_eb:]]),
            n_eb,
            z,
            hyb,
            gamma,
            None,
            weight_array,
            W_mn,
            n_C,
        )
    )
    return v_final, eb_final, C_final, c_final


def _varpro_inner_solve(eb, z, hyb, realvalue_v):
    """Find optimal PSD residues and constant shift for fixed bath energies.

    Uses lstsq + projection.

    Solves hyb ≈ C + sum_k A_k / (z - eb[k]) simultaneously:
    - A_k are Hermitian PSD (residues) — obtained by eigh + clip
    - C is Hermitian (constant shift) — Hermitized but not PSD-constrained

    Returns (A_psd, V, G, C) where G[m, k] = 1 / (z[m] - eb[k]).
    """
    n_bath = len(eb)
    n_imp = hyb.shape[1]
    M = len(z)

    G = 1.0 / (z[:, None] - eb[None, :])  # (M, n_bath)
    # Augment with a column of ones to simultaneously solve for the constant C.
    G_aug = np.hstack([G, np.ones((M, 1))])  # (M, n_bath + 1)

    X, _, _, _ = np.linalg.lstsq(G_aug, hyb.reshape(M, n_imp * n_imp), rcond=None)  # (n_bath + 1, n_imp^2)

    A = X[:n_bath].reshape(n_bath, n_imp, n_imp)
    C_raw = X[n_bath].reshape(n_imp, n_imp)

    A = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))  # Hermitize residues
    C = 0.5 * (C_raw + np.conj(C_raw.T))  # Hermitize C (no PSD constraint)
    if realvalue_v:
        A = A.real
        C = C.real

    lam, U = np.linalg.eigh(A)
    lam = np.clip(lam.real, 0.0, None)
    A_psd = (U * lam[:, None, :]) @ np.conj(np.swapaxes(U, -1, -2))
    V = np.sqrt(lam)[:, :, None] * np.conj(np.swapaxes(U, -1, -2))  # (n_bath, n_imp, n_imp)

    return A_psd, V, G, C


def _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, realvalue_v):
    """
    VARPRO cost and its gradient w.r.t. bath energies.

    For each eb proposal, solves for optimal PSD residues A and constant shift
    C via `_varpro_inner_solve`, evaluates the fit cost, and returns the partial
    gradient w.r.t. eb (treating A_psd and C as fixed — valid by the VARPRO
    theorem for unconstrained lstsq; approximate after PSD projection).

    Returns (cost, grad_eb, V, C).
    """
    A_psd, V, G, C = _varpro_inner_solve(eb, z, hyb, realvalue_v)

    max_moment = W_mn.shape[1]
    hyb_model = np.einsum("mk, kij -> mij", G, A_psd) + C[None]  # (M, n_imp, n_imp)
    diff = hyb - hyb_model

    w2 = weight_array**2
    N = diff.size
    c = np.sum(w2[:, None, None] * 0.5 * np.abs(diff) ** 2) / N

    moment_diff = np.einsum("mn, mij -> nij", W_mn, diff)  # (max_moment, n_imp, n_imp)
    P = moment_diff[0].size * max_moment
    c += np.sum(0.5 * np.abs(moment_diff) ** 2) / P

    # Partial VARPRO gradient w.r.t. eb: C is treated as fixed, so only the
    # A_psd / (z - eb)^2 term contributes (C has no eb dependence at fixed A).
    dGdeb = G**2  # (M, n_bath)
    conj_diff_A = np.einsum("mij, kij -> mk", np.conj(diff), A_psd)  # (M, n_bath)
    grad = -np.real(np.einsum("m, mk, mk -> k", w2, dGdeb, conj_diff_A)) / N

    WdG = np.einsum("mn, mk -> kn", W_mn, dGdeb)  # (n_bath, max_moment)
    conj_mdf_A = np.einsum("nij, kij -> kn", np.conj(moment_diff), A_psd)  # (n_bath, max_moment)
    grad -= np.real(np.einsum("kn, kn -> k", WdG, conj_mdf_A)) / P

    return c, grad, V, C


def _psd_frechet_factors(A_h):
    """Eigendecomposition and Daleckii-Krein factors of the PSD projection.

    For a Hermitian ``A_h = U diag(lam) U^H`` the projection onto the PSD cone is
    ``Pi(A) = U diag(max(lam, 0)) U^H``.  Its Frechet derivative in a Hermitian
    direction ``H`` is ``Pi'(A)[H] = U (Psi ∘ (U^H H U)) U^H`` where
    ``Psi_ab = (f(lam_a) - f(lam_b)) / (lam_a - lam_b)`` with ``f(x)=max(x,0)``
    (and ``Psi_aa = f'(lam_a)`` on the diagonal / for degenerate eigenvalues).

    Returns ``(U, lam_clipped, A_psd, Psi)``.
    """
    lam, U = np.linalg.eigh(A_h)  # (..., n), (..., n, n)
    lam_c = np.clip(lam, 0.0, None)
    A_psd = (U * lam_c[..., None, :]) @ np.conj(np.swapaxes(U, -1, -2))

    fp = (lam > 0).astype(float)  # f'(lam): subgradient, 0 at the boundary
    li = lam[..., :, None]
    lj = lam[..., None, :]
    denom = li - lj
    num = np.clip(li, 0.0, None) - np.clip(lj, 0.0, None)
    tol = 1e-9
    degenerate = np.abs(denom) <= tol
    safe_denom = np.where(degenerate, 1.0, denom)
    Psi = np.where(degenerate, 0.5 * (fp[..., :, None] + fp[..., None, :]), num / safe_denom)
    return U, lam_c, A_psd, Psi


def _varpro_cost_and_full_grad(eb, z, hyb, weight_array, W_mn, realvalue_v):
    """VARPRO cost and the *exact* total-derivative gradient w.r.t. bath energies.

    Unlike `_varpro_cost_and_grad` (which uses the Kaufman simplification -- it
    treats the analytically solved residues/shift as fixed), this propagates the
    full dependence of the inner solve on eb: the derivative of the lstsq
    solution ``X = Phi^+ Y`` (pseudoinverse-derivative formula), the
    Hermitization, and the PSD projection (Frechet derivative).  It matches a
    finite-difference gradient of the reduced cost to machine precision (away
    from PSD active-set boundaries, where the cost is only sub-differentiable).

    Returns (cost, grad_eb, V, C) with the same conventions as
    `_varpro_cost_and_grad`.
    """
    n = len(eb)
    n_imp = hyb.shape[1]
    M = len(z)
    q = n_imp * n_imp
    max_moment = W_mn.shape[1]

    # --- forward pass (mirrors _varpro_inner_solve, via normal equations so the
    #     pseudoinverse derivative below is consistent with X) ---
    G = 1.0 / (z[:, None] - eb[None, :])  # (M, n)
    Gp = G**2  # dG[:, k]/deb_k lives on column k only
    Phi = np.hstack([G, np.ones((M, 1))])  # (M, n+1)
    Y = hyb.reshape(M, q)

    Gram_inv = np.linalg.inv(np.conj(Phi.T) @ Phi)  # (n+1, n+1)
    Pinv = Gram_inv @ np.conj(Phi.T)  # (n+1, M)
    X = Pinv @ Y  # (n+1, q)
    Rres = Y - Phi @ X  # (M, q), unweighted lstsq residual

    A_raw = X[:n].reshape(n, n_imp, n_imp)
    C_raw = X[n].reshape(n_imp, n_imp)
    A_h = 0.5 * (A_raw + np.conj(np.swapaxes(A_raw, -1, -2)))
    C_h = 0.5 * (C_raw + np.conj(C_raw.T))
    if realvalue_v:
        A_h = A_h.real
        C_h = C_h.real

    U, lam_c, A_psd, Psi = _psd_frechet_factors(A_h)

    # --- cost (identical to _varpro_cost_and_grad) ---
    hyb_model = np.einsum("mk, kij -> mij", G, A_psd) + C_h[None]
    diff = hyb - hyb_model
    w2 = weight_array**2
    N = diff.size
    c = np.sum(w2[:, None, None] * 0.5 * np.abs(diff) ** 2) / N
    moment_diff = np.einsum("mn, mij -> nij", W_mn, diff)
    P = moment_diff[0].size * max_moment
    c += np.sum(0.5 * np.abs(moment_diff) ** 2) / P

    # Cost gradient w.r.t. the model: dc = Re sum_m <Gbar_m, d(hyb_model)_m>.
    Gbar = -(1.0 / N) * w2[:, None, None] * np.conj(diff) - (1.0 / P) * np.einsum(
        "mn, nij -> mij", W_mn, np.conj(moment_diff)
    )  # (M, n_imp, n_imp)

    # Explicit (Kaufman) part: only the k-th pole's G varies.
    GpGbar = np.einsum("mk, mij -> kij", Gp, Gbar)  # (n, n_imp, n_imp)
    grad_expl = np.real(np.einsum("kij, kij -> k", GpGbar, A_psd))

    # Implicit part: pull back the cost gradient through A_psd(eb) and C(eb).
    QA = np.einsum("mk, mij -> kij", G, Gbar)  # (n, n_imp, n_imp)
    QC = np.sum(Gbar, axis=0)  # (n_imp, n_imp)

    PG = Pinv @ Gp  # (n+1, n)
    RG = np.conj(Gp).T @ Rres  # (n, q)
    # dX[k] = -PG[:, k] (x) X[k]  +  Gram_inv[:, k] (x) RG[k].
    dX = -np.einsum("ak, kq -> kaq", PG, X[:n]) + np.einsum("ak, kq -> kaq", Gram_inv[:, :n], RG)  # (n, n+1, q)
    dA_raw = dX[:, :n, :].reshape(n, n, n_imp, n_imp)  # (k, j, i, j')
    dC_raw = dX[:, n, :].reshape(n, n_imp, n_imp)  # (k, i, j)
    dA_h = 0.5 * (dA_raw + np.conj(np.swapaxes(dA_raw, -1, -2)))
    dC_h = 0.5 * (dC_raw + np.conj(np.swapaxes(dC_raw, -1, -2)))
    if realvalue_v:
        dA_h = dA_h.real
        dC_h = dC_h.real

    # PSD Frechet: dA_psd[k, j] = U_j (Psi_j ∘ (U_j^H dA_h[k, j] U_j)) U_j^H.
    UH = np.conj(np.swapaxes(U, -1, -2))  # (j, a, b)
    Mmat = np.einsum("jab, kjbc, jcd -> kjad", UH, dA_h, U) * Psi[None]
    dA_psd = np.einsum("jab, kjbc, jcd -> kjad", U, Mmat, UH)

    grad_impl = np.real(np.einsum("jab, kjab -> k", QA, dA_psd) + np.einsum("ab, kab -> k", QC, dC_h))

    grad = grad_expl + grad_impl
    V = np.sqrt(lam_c)[:, :, None] * UH
    return float(c), grad, V, C_h


def get_v_and_eb_varpro_basin_hopping(
    w,
    delta,
    hyb,
    ebs,
    eb_restrictions,
    gamma,
    regularization,
    weight_function,
    realvalue_v,
    max_moment=3,
    full_gradient=True,
):
    """Fit bath energies with VARPRO basin-hopping.

    Optimizes only over bath energies, with residues solved analytically via
    lstsq + PSD projection at each step.

    The search space shrinks from n_bath*(1 + n_imp^2) to n_bath, with each
    evaluation costing one lstsq + eigh solve.  After basin-hopping, a final
    SLSQP polish refines both energies and hoppings jointly with the analytic
    Jacobian.

    ``full_gradient`` (default True) uses the exact total-derivative gradient of
    the reduced cost (`_varpro_cost_and_full_grad`), which propagates the eb
    dependence through the analytic inner solve.  Set it False to fall back to
    the cheaper Kaufman approximation (`_varpro_cost_and_grad`), which treats the
    solved residues as fixed and gives a less accurate search direction.
    """
    grad_fun = _varpro_cost_and_full_grad if full_gradient else _varpro_cost_and_grad
    n_eb = ebs.shape[1]

    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr
    weight_array = weight_function(w)
    W_mn = moment_weights(w, max_moment)

    # Reparametrize bath energies as [first energy, gaps]; gaps >= delta keep the
    # states sorted and separated by at least the broadening, so no post-fit merge
    # is needed and reorder-equivalent configurations collapse to one.
    w_min, w_max = eb_restrictions[0]
    # Cap (don't fail) at the number of states that fit in the window separated by
    # delta; a larger request is honoured as "as many as reasonably fit".
    n_max = _max_bath_states(w_min, w_max, delta)
    if n_eb > n_max:
        n_eb = n_max
        ebs = ebs[:, :n_eb]
    gap_bounds = _gap_bounds(w_min, w_max, n_eb, delta)
    gap_seeds = np.array([_repair_gaps(_eb_to_gaps(eb, delta), w_min, w_max, delta) for eb in ebs])

    initial_costs = np.array(
        [_varpro_cost_and_grad(_gaps_to_eb(p), z, hyb, weight_array, W_mn, realvalue_v)[0] for p in gap_seeds]
    )
    mean_cost = np.mean(initial_costs)
    stddev_cost = np.std(initial_costs)
    T = max(stddev_cost, 1e-3 * abs(mean_cost))

    x0 = gap_seeds[np.argmin(initial_costs)]

    def _fg(p):
        eb = _gaps_to_eb(p)
        c, g, _, _ = grad_fun(eb, z, hyb, weight_array, W_mn, realvalue_v)
        return float(c), _gaps_grad(g)

    # L-BFGS-B (box bounds only) explores: it gives markedly better minima here
    # than a constrained SLSQP, and has no incentive to push a pole out of the
    # window since out-of-window poles get ~zero residue (no gradient pull). Box
    # bounds do not strictly cap the largest energy (a cumulative sum can
    # overshoot w_max), so the final SLSQP polish -- which carries the explicit
    # sum(gaps) <= w_max constraint -- projects any stray pole back into the
    # window and re-optimizes, guaranteeing the returned energies are in-window.
    res = basinhopping(
        _fg,
        x0,
        niter=150,
        T=T,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "jac": True,
            "tol": 1e-6,
            "options": {"maxiter": 500},
            "bounds": gap_bounds,
        },
        disp=False,
    )

    # No merge: the gap constraint already guarantees separation >= delta.  A
    # final SLSQP polish refines eb (still gap-parametrized), V and C jointly.
    return _gap_slsqp_polish(
        res.x,
        z,
        hyb,
        gamma,
        regularization,
        weight_array,
        W_mn,
        realvalue_v,
        gap_bounds,
    )


def get_v_and_eb_differential_evolution(
    w,
    delta,
    hyb,
    ebs,
    eb_restrictions,
    gamma,
    regularization,
    weight_function,
    realvalue_v=True,
    max_moment=3,
):
    """Fit bath energies with VARPRO differential evolution.

    Optimizes only over bath energies, with residues and constant shift C
    solved analytically at each evaluation.

    After DE convergence a final SLSQP polish refines eb, V, and C jointly.
    """
    n_eb = ebs.shape[1]
    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr
    weight_array = weight_function(w)
    W_mn = moment_weights(w, max_moment)

    # Gap parametrization [first energy, gaps]; gaps >= delta enforce ordering and
    # minimum separation, removing the need to merge overlapping states afterwards.
    w_min, w_max = eb_restrictions[0]
    # Cap (don't fail) at the number of states that fit in the window separated by
    # delta; a larger request is honoured as "as many as reasonably fit".
    n_max = _max_bath_states(w_min, w_max, delta)
    if n_eb > n_max:
        n_eb = n_max
        ebs = ebs[:, :n_eb]
    gap_bounds = _gap_bounds(w_min, w_max, n_eb, delta)
    gap_seeds = np.array([_repair_gaps(_eb_to_gaps(eb, delta), w_min, w_max, delta) for eb in ebs])

    def varpro_cost(p):
        eb = _gaps_to_eb(p)
        c, _, _, _ = _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, realvalue_v)
        return float(c)

    # Linear constraint sum(gaps) <= w_max keeps the largest energy in the window;
    # box bounds on individual gaps cannot enforce this on their cumulative sum.
    res = differential_evolution(
        varpro_cost,
        Bounds(
            lb=[b[0] for b in gap_bounds],
            ub=[b[1] for b in gap_bounds],
        ),
        constraints=(LinearConstraint(np.ones((1, n_eb)), -np.inf, w_max),),
        init=gap_seeds,
        atol=1e-6,
        maxiter=10000,
        polish=False,
    )

    # No merge: the gap constraint already guarantees separation >= delta.  A
    # final SLSQP polish refines eb (still gap-parametrized), V and C jointly.
    return _gap_slsqp_polish(
        res.x,
        z,
        hyb,
        gamma,
        regularization,
        weight_array,
        W_mn,
        realvalue_v,
        gap_bounds,
    )


def calc_diff(eb, v, z, hyb, C=None):
    """Return the residual ``hyb - model`` for bath parameters (eb, v, C)."""
    hyb_model = get_hyb_2(z, eb, v, C=C)
    return hyb[np.newaxis] - hyb_model


def calc_moment_diff(diff, W_mn):
    r"""Return the spectral-moment residuals of the fit.

    Parameters
    ----------
    diff : (..., n_w, n_imp, n_imp) np.ndarray
        Residual :math:`\Delta(\omega) - \tilde{\Delta}(\omega)`.
    W_mn : (n_w, max_moment) np.ndarray
        Moment weights from `moment_weights`.

    Returns
    -------
    (..., max_moment, n_imp, n_imp) np.ndarray
        The weighted moments of the residual.
    """
    return np.einsum("mn, ...mij -> ...nij", W_mn, diff)


def _unroll_C_batch(p_C, n_imp):
    """Vectorised unroll_C: p_C (n_C, S) → C_arr (S, n_imp, n_imp)."""
    triu_i, triu_j = np.triu_indices(n_imp)
    n_triu = len(triu_i)
    off_mask = triu_i != triu_j
    S = p_C.shape[1]
    if p_C.shape[0] == n_triu:  # real symmetric
        C_arr = np.zeros((S, n_imp, n_imp))
        C_arr[:, triu_i, triu_j] = p_C.T
        C_arr[:, triu_j, triu_i] = p_C.T
    else:  # complex Hermitian
        off_i, off_j = triu_i[off_mask], triu_j[off_mask]
        C_arr = np.zeros((S, n_imp, n_imp), dtype=complex)
        entries = p_C[:n_triu].T.astype(complex)
        entries[:, off_mask] += 1j * p_C[n_triu:].T
        C_arr[:, triu_i, triu_j] = entries
        C_arr[:, off_j, off_i] = np.conj(entries[:, off_mask])
    return C_arr


def vectorized_cost_function(
    p,
    n_eb,
    z,
    hyb,
    gamma,
    regularization="L1",
    weight_array=None,
    W_mn=None,
    n_C=0,
):
    r"""Weighted least-squares cost of a bath-parametrized hybridization model.

    The cost is the weighted mean-square residual between ``hyb`` and the
    model :math:`\sum_b V_b^\dagger V_b / (z - e_b) + C`, plus a scaled
    spectral-moment penalty (when ``W_mn`` is given) and L1/L2 regularization
    of the hopping parameters only (bath energies and C are not penalized).

    Parameters
    ----------
    p : (n_p,) or (n_p, S) np.ndarray
        Parameter vector(s): ``n_eb`` bath energies, then packed hoppings
        (see `unroll`), then optionally ``n_C`` packed constant-shift
        parameters (see `unroll_C`). A 2-D array evaluates a population of
        ``S`` parameter vectors at once.
    n_eb : int
        Number of bath energies at the start of ``p``.
    z : (M,) np.ndarray
        Complex frequency mesh.
    hyb : (M, n_imp, n_imp) np.ndarray
        Hybridization function to fit.
    gamma : float
        Regularization strength.
    regularization : {"L1", "L2", "none", None}
        Regularization type applied to the hopping parameters.
    weight_array : (M,) np.ndarray, optional
        Pointwise fit weights; defaults to uniform.
    W_mn : (M, max_moment) np.ndarray, optional
        Moment weights from `moment_weights`; omit to skip the moment term.
    n_C : int, default 0
        Number of constant-shift parameters at the end of ``p``.

    Returns
    -------
    float or (S,) np.ndarray
        The cost, scalar for a 1-D ``p``.
    """
    one_dim = len(p.shape) == 1
    p_batched = p[:, None] if one_dim else p
    w = z.real
    n_w = len(z)
    n_imp = hyb.shape[1]
    eb = np.moveaxis(p_batched[:n_eb], 0, -1)

    n_v_end = p_batched.shape[0] - n_C if n_C else p_batched.shape[0]
    p_v = p_batched[n_eb:n_v_end]
    v = unroll(p_v, n_eb, n_imp)

    # Build C: (n_imp, n_imp) for one_dim, (S, n_imp, n_imp) for batched.
    C = None
    if n_C:
        C_arr = _unroll_C_batch(p_batched[-n_C:], n_imp)  # (S, n_imp, n_imp)
        C = C_arr[0] if one_dim else C_arr

    diff = hyb[np.newaxis] - get_hyb_2(z, eb, v, C=C)  # (S, M, N, N)

    if weight_array is None:
        weight_array = np.ones_like(w)

    c = (1 / (n_w * n_imp * n_imp)) * np.sum(
        0.5 * weight_array[None, :, None, None] * np.abs(diff) ** 2, axis=(1, 2, 3)
    )

    if W_mn is not None:
        moment_diff = np.einsum("mn, ...mij -> ...nij", W_mn, diff)
        c += (1 / (n_imp * n_imp * moment_diff.shape[1])) * np.sum(0.5 * np.abs(moment_diff) ** 2, axis=(1, 2, 3))

    # Regularization applies only to V parameters, not eb or C.
    n_v = p_v.shape[0]
    if regularization is None or regularization.lower() == "none":
        pass
    elif regularization.lower() == "l1":
        c += (gamma / n_v) * np.sum(np.abs(p_v), axis=0)
    elif regularization.lower() == "l2":
        c += (gamma / n_v) * np.sum(p_v**2, axis=0)
    else:
        raise RuntimeError(f"Unknown regularization mode {regularization}")

    return c[0].item() if one_dim else c


def vectorized_jacobian(
    p,
    n_eb,
    z,
    hyb,
    gamma,
    regularization="L1",
    weight_array=None,
    W_mn=None,
    n_C=0,
):
    """Analytic gradient of `vectorized_cost_function`.

    Takes the same arguments as `vectorized_cost_function` and returns the
    gradient with respect to ``p``, shape ``(n_p,)`` for a 1-D ``p`` or
    ``(n_p, S)`` for a population. Verified against finite differences in
    the test suite for both real and complex hoppings.
    """
    one_dim = len(p.shape) == 1
    if one_dim:
        p = p[:, None]
    J = np.zeros_like(p)
    popsize = p.shape[1]
    n_w = len(z)
    n_imp = hyb.shape[1]
    eb = np.moveaxis(p[:n_eb], 0, -1)

    if weight_array is None:
        weight_array = np.ones_like(z.real)

    triu_rows, triu_cols = np.triu_indices(n_imp)
    n_v_end = p.shape[0] - n_C if n_C else p.shape[0]
    p_v = p[n_eb:n_v_end]
    realvalued = p_v.shape[0] == n_eb * len(triu_cols)

    v = unroll(p_v, n_eb, n_imp)  # (S, n_eb, n_imp, n_imp)

    C = None
    if n_C:
        C_arr = _unroll_C_batch(p[-n_C:], n_imp)  # (S, n_imp, n_imp)
        C = C_arr[0] if one_dim else C_arr

    diff = hyb[np.newaxis] - get_hyb_2(z, eb, v, C=C)  # (S, M, N, N)

    G = 1.0 / (z[np.newaxis, :, np.newaxis] - eb[:, np.newaxis, :])  # (S, M, n_eb)
    A = np.conj(np.transpose(v, (0, 1, 3, 2))) @ v  # (S, n_eb, N, N)
    diff_W = diff * weight_array[None, :, None, None]

    dhyb_deb = A[:, np.newaxis, :, :, :] * (G**2)[:, :, :, np.newaxis, np.newaxis]  # (S, M, n_eb, N, N)
    J_eb = -np.einsum("smxy, smbxy -> sb", np.conj(diff_W), dhyb_deb).real
    J[:n_eb, :] = np.moveaxis(J_eb, 0, -1) / (n_w * n_imp * n_imp)

    if W_mn is not None:
        moment_diff = np.einsum("mn, ...mij -> ...nij", W_mn, diff)
        dmoment_deb = -np.einsum("mn, smbxy -> snbxy", W_mn, dhyb_deb)
        J_moment_eb = np.einsum("snxy, snbxy -> sb", np.conj(moment_diff), dmoment_deb).real
        J[:n_eb, :] += np.moveaxis(J_moment_eb, 0, -1) / (n_imp * n_imp * moment_diff.shape[1])

    # --- V gradient ---
    S_term = -np.einsum("smxy, smb -> sbxy", np.conj(diff_W), G)
    if W_mn is not None:
        WG = np.einsum("mn, smb -> snb", W_mn, G)
        S_mom = -np.einsum("snxy, snb -> sbxy", np.conj(moment_diff), WG)
        S_total = S_term / (n_w * n_imp * n_imp) + S_mom / (n_imp * n_imp * moment_diff.shape[1])
    else:
        S_total = S_term / (n_w * n_imp * n_imp)

    J_R = np.zeros((popsize, n_eb, n_imp, n_imp), dtype=float)
    J_I = np.zeros((popsize, n_eb, n_imp, n_imp), dtype=float)
    for m in range(n_imp):
        for n in range(n_imp):
            if m > n:
                continue
            term_R = np.sum(
                S_total[:, :, n, :] * v[:, :, m, :] + S_total[:, :, :, n] * np.conj(v[:, :, m, :]),
                axis=-1,
            )
            J_R[:, :, m, n] = np.real(term_R)
            if not realvalued:
                term_I = np.sum(
                    S_total[:, :, n, :] * (-1j * v[:, :, m, :]) + S_total[:, :, :, n] * (1j * np.conj(v[:, :, m, :])),
                    axis=-1,
                )
                J_I[:, :, m, n] = np.real(term_I)

    J_R_flat = J_R[:, :, triu_rows, triu_cols].reshape((popsize, -1), order="C")
    n_real = n_eb * len(triu_cols)
    J[n_eb : n_eb + n_real, :] = np.moveaxis(J_R_flat, 0, -1)
    if not realvalued:
        J_I_flat = J_I[:, :, triu_rows, triu_cols].reshape((popsize, -1), order="C")
        J[n_eb + n_real : n_v_end, :] = np.moveaxis(J_I_flat, 0, -1)

    # Regularization on V params only.
    n_v = p_v.shape[0]
    if regularization is None or regularization.lower() == "none":
        pass
    elif regularization.lower() == "l1":
        J[n_eb:n_v_end] += (gamma / n_v) * np.sign(p_v)
    elif regularization.lower() == "l2":
        J[n_eb:n_v_end] += (gamma / n_v) * 2 * p_v
    else:
        raise RuntimeError(f"Unknown regularization mode {regularization}")

    # --- C gradient ---
    if n_C:
        # For a Hermitian C, the independent parameters are:
        #   Re(C[p,q]) for all upper-tri (p,q) pairs, and Im(C[p,q]) for off-diagonal.
        # Each parameter affects both (p,q) and (q,p) elements of the model, so both
        # rows/columns of diff contribute.  We sum both contributions explicitly rather
        # than assuming diff is Hermitian (which fails when hyb is non-Hermitian).
        off_mask = triu_rows != triu_cols
        n_triu = len(triu_rows)

        weighted_diff = np.einsum("m, smij -> sij", weight_array, diff)  # (S, n_imp, n_imp)
        N = n_w * n_imp * n_imp

        # For triu (p,q): upper[k] = weighted_diff[p,q], lower[k] = weighted_diff[q,p].
        # d(cost)/d(Re(C[p,q])) = -(1/N)*Re(upper + I(p!=q)*lower)
        upper = weighted_diff[:, triu_rows, triu_cols]  # (S, n_triu)
        lower = weighted_diff[:, triu_cols, triu_rows]  # (S, n_triu) — transposed indices
        sum_RL = upper.copy()
        sum_RL[:, off_mask] += lower[:, off_mask]
        J_C_real = -(1.0 / N) * np.real(np.moveaxis(sum_RL, 0, -1))  # (n_triu, S)

        if W_mn is not None:
            P = n_imp * n_imp * moment_diff.shape[1]
            W_sum = W_mn.sum(axis=0)  # (max_moment,)
            weighted_mdiff = np.einsum("n, snij -> sij", W_sum, moment_diff)  # (S, n_imp, n_imp)
            mupper = weighted_mdiff[:, triu_rows, triu_cols]
            mlower = weighted_mdiff[:, triu_cols, triu_rows]
            sum_mRL = mupper.copy()
            sum_mRL[:, off_mask] += mlower[:, off_mask]
            J_C_real -= (1.0 / P) * np.real(np.moveaxis(sum_mRL, 0, -1))

        J[n_v_end : n_v_end + n_triu, :] = J_C_real

        if n_C > n_triu:  # complex Hermitian: imaginary gradient for off-diagonal
            # d(cost)/d(Im(C[p,q])) = -(1/N)*Im(upper - lower)  for p < q
            J_C_imag = -(1.0 / N) * np.imag(np.moveaxis(upper[:, off_mask] - lower[:, off_mask], 0, -1))
            if W_mn is not None:
                J_C_imag -= (1.0 / P) * np.imag(np.moveaxis(mupper[:, off_mask] - mlower[:, off_mask], 0, -1))
            J[n_v_end + n_triu :, :] = J_C_imag

    return J[:, 0] if one_dim else J

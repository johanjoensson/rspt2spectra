#!/usr/bin/env python3

"""Fit discrete bath models to (off-diagonal) hybridization functions.

The active optimizers are `get_v_and_eb_varpro_basin_hopping` and
`get_v_and_eb_differential_evolution`: both search only over bath energies
(reparametrized as gaps to enforce ordering and minimum separation) while the
hopping residues and a constant Hermitian shift are solved analytically at
each step (VARPRO), followed by a joint SLSQP polish using the analytic
Jacobian of `vectorized_cost_function`.
"""

from dataclasses import replace

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


def free_window(w_min, w_max, delta, sym):
    """Bounds for the bath energies the optimizer actually varies.

    Without particle-hole symmetry these are the fit window itself.  With it,
    only the positive half of the bath is free and each pole ``e`` implies a
    partner at ``-e``, so ``e`` must fit in the largest sub-window symmetric
    about the Fermi level.  The lower bound keeps the mirrored poles separated by
    at least the broadening, exactly as the ``delta`` gap bound does for
    neighbouring poles: ``delta/2`` separates ``+e`` from ``-e``, and ``delta``
    when an unpaired pole sits at zero between them.

    Parameters
    ----------
    w_min, w_max : float
        Edges of the fit window.
    delta : float
        Broadening / minimum state separation.
    sym : rspt2spectra.symmetries.BlockSymmetry
        Detected symmetry of the block.

    Returns
    -------
    (float, float)
        Lower and upper bound for the free bath energies.
    """
    if not sym.particle_hole:
        return w_min, w_max
    half = min(w_max, -w_min)
    lo = delta if sym.zero_pole else 0.5 * delta
    return lo, max(half, lo)


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


def max_bath_states(w_min, w_max, delta, sym=None):
    """Largest honourable state count for a block, accounting for symmetry.

    Without particle-hole symmetry this is `_max_bath_states` over the fit
    window.  With it the count refers to the *full* mirrored bath, so it is
    twice the number of poles that fit in the positive half (plus one for an
    unpaired pole at zero when the count is odd).

    Parameters
    ----------
    w_min, w_max : float
        Edges of the fit window.
    delta : float
        Broadening / minimum state separation.
    sym : rspt2spectra.symmetries.BlockSymmetry, optional
        Detected symmetry; ``None`` means unconstrained.

    Returns
    -------
    int
        Maximum number of bath states.
    """
    if sym is None or not sym.particle_hole:
        return _max_bath_states(w_min, w_max, delta)
    # `free_window`'s lower bound depends on whether there is an unpaired pole at
    # E_F, but that is only decided once this cap has been applied -- so the cap
    # is taken without one, which is the permissive choice, and the extra pole
    # always fits between the innermost mirrored pair.
    lo, hi = free_window(w_min, w_max, delta, replace(sym, zero_pole=False))
    return 2 * _max_bath_states(lo, hi, delta) + 1


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


def _project_residues(basis, A):
    """Orthogonally project Hermitian matrices onto the symmetry-allowed subspace."""
    if basis.shape[0] == 0:
        return np.zeros_like(A)
    coef = np.einsum("pij, ...ji -> ...p", basis, A).real
    return np.einsum("...p, pij -> ...ij", coef, basis)


def _project_residues_adjoint(basis, T):
    """Adjoint of `_project_residues` in the ``Re sum_xy T_xy dA_xy`` pairing.

    The Jacobian pairs a sensitivity ``T`` with a Hermitian direction ``dA`` as
    ``Re sum_xy T_xy dA_xy`` (see the V-gradient block below), under which the
    adjoint of the projection is ``T -> sum_p Re(sum_xy T_xy S_p,xy) S_p^T``.
    """
    if basis.shape[0] == 0:
        return np.zeros_like(T)
    beta = np.einsum("pij, ...ij -> ...p", basis, T).real
    return np.einsum("...p, pji -> ...ij", beta, basis)


def _project_shift_adjoint(basis, weighted_diff):
    """Pull a constant-shift sensitivity back through the projection.

    `vectorized_jacobian` expresses the ``C`` gradient through the (unconjugated)
    weighted residual rather than through an explicit sensitivity matrix, so the
    adjoint takes the matching form.
    """
    if basis.shape[0] == 0:
        return np.zeros_like(weighted_diff)
    beta = np.einsum("pij, ...ij -> ...p", basis, np.conj(weighted_diff)).real
    return np.einsum("...p, pij -> ...ij", beta, basis)


def _project_free_residues(sym, A_free, n_eb):
    """Project the freely parametrized residues onto their allowed subspaces.

    The mirrored poles may use the whole allowed subspace, but an unpaired pole
    sitting at the Fermi level is its own mirror partner, so its residue has to
    satisfy ``A = A^T`` on its own: it lives in the transpose-even sector alone.
    `_varpro_inner_solve` gets this for free by building the zero-pole column in
    the even sector; the polish parametrizes a free ``V`` instead, so the
    restriction has to be applied explicitly here.
    """
    out = _project_residues(sym.basis, A_free[..., :n_eb, :, :])
    if sym.particle_hole and sym.zero_pole:
        zero = _project_residues(sym.even_basis, A_free[..., n_eb : n_eb + 1, :, :])
        out = np.concatenate([out, zero], axis=-3)
    return out


def _project_free_residues_adjoint(sym, T_free, n_eb):
    """Adjoint of `_project_free_residues`, in the Jacobian's pairing convention."""
    out = _project_residues_adjoint(sym.basis, T_free[..., :n_eb, :, :])
    if sym.particle_hole and sym.zero_pole:
        zero = _project_residues_adjoint(sym.even_basis, T_free[..., n_eb : n_eb + 1, :, :])
        out = np.concatenate([out, zero], axis=-3)
    return out


def _expand_residues(A_free, sym, n_eb):
    """Mirror the free residues onto the full pole set (identity without p-h)."""
    if not sym.particle_hole:
        return A_free
    A_pos = A_free[..., :n_eb, :, :]
    blocks = [np.swapaxes(A_pos, -1, -2)[..., ::-1, :, :]]
    if sym.zero_pole:
        blocks.append(A_free[..., n_eb : n_eb + 1, :, :])
    blocks.append(A_pos)
    return np.concatenate(blocks, axis=-3)


def _fold_residue_sensitivity(T_full, sym, n_eb):
    """Adjoint of `_expand_residues`: fold a per-pole sensitivity onto the free poles."""
    if not sym.particle_hole:
        return T_full
    offset = n_eb + (1 if sym.zero_pole else 0)
    T_free = T_full[..., offset:, :, :] + np.swapaxes(T_full[..., :n_eb, :, :], -1, -2)[..., ::-1, :, :]
    if sym.zero_pole:
        T_free = np.concatenate([T_free, T_full[..., n_eb : n_eb + 1, :, :]], axis=-3)
    return T_free


def _fold_energy_sensitivity(J_full, sym, n_eb):
    """Fold a per-pole energy gradient onto the free energies, along the last axis."""
    if not sym.particle_hole:
        return J_full
    offset = n_eb + (1 if sym.zero_pole else 0)
    return J_full[..., offset:] - J_full[..., :n_eb][..., ::-1]


def n_residue_blocks(n_eb, sym):
    """Return the number of independently parametrized residues for ``n_eb`` free energies."""
    if sym is None or not sym.particle_hole:
        return n_eb
    return n_eb + (1 if sym.zero_pole else 0)


def _c_basis(sym):
    """Basis the constant shift is restricted to.

    Particle-hole symmetry forces ``C = -C^T``, i.e. the constant shift lives in
    the transpose-odd sector alone; a transpose-symmetric (real) residue space
    therefore pins ``C`` to zero, as it must for a particle-hole symmetric
    hybridization.
    """
    return sym.odd_basis if sym.particle_hole else sym.basis


def _free_residue_factors(V_full, sym, n_free):
    """Select the independently parametrized hopping factors from a full bath."""
    if not sym.particle_hole:
        return V_full
    offset = n_free + (1 if sym.zero_pole else 0)
    parts = [V_full[offset:]]
    if sym.zero_pole:
        parts.append(V_full[n_free : n_free + 1])
    return np.concatenate(parts, axis=0)


def _expand_residue_factors(V_free, sym, n_free):
    """Mirror hopping factors onto the full pole set.

    ``A_{-e} = A_{+e}^T = conj(V)^dagger conj(V)``, so the mirrored factor is the
    complex conjugate of the one that was fitted.
    """
    if not sym.particle_hole:
        return V_free
    V_pos = V_free[:n_free]
    blocks = [np.conj(V_pos)[::-1]]
    if sym.zero_pole:
        blocks.append(V_free[n_free : n_free + 1])
    blocks.append(V_pos)
    return np.concatenate(blocks, axis=0)


def _gap_slsqp_polish(gap_x, z, hyb, gamma, regularization, weight_array, W_mn, sym, gap_bounds):
    """SLSQP refinement of a gap-parametrized bath fit over eb, V and C jointly.

    `gap_x` is the converged gap vector [first energy, gaps].  The eb block stays
    gap-parametrized during the polish (via local cost/Jacobian wrappers around
    the shared vectorized functions) so the minimum-separation constraint cannot
    be violated.  The symmetry constraints are applied inside the shared cost and
    Jacobian, so the polish cannot undo the symmetry the search established.
    Returns (v_final, eb_final, C_final, c_final) over the *full* pole set.
    """
    n_eb = len(gap_x)
    n_imp = hyb.shape[1]
    n_blocks = n_residue_blocks(n_eb, sym)

    eb_opt = _gaps_to_eb(gap_x)
    _, V_opt, _, C_opt = _varpro_inner_solve(eb_opt, z, hyb, sym)

    p_C0 = inroll_C(C_opt)
    n_C = len(p_C0)
    p0 = np.concatenate([gap_x, inroll(_free_residue_factors(V_opt, sym, n_eb)), p_C0])
    bounds = gap_bounds + [(None, None)] * (len(p0) - n_eb)
    w_max = gap_bounds[0][1]

    def _cost(p):
        p_abs = np.concatenate([_gaps_to_eb(p[:n_eb]), p[n_eb:]])
        return vectorized_cost_function(p_abs, n_eb, z, hyb, gamma, regularization, weight_array, W_mn, n_C, sym=sym)

    def _jac(p):
        p_abs = np.concatenate([_gaps_to_eb(p[:n_eb]), p[n_eb:]])
        J = vectorized_jacobian(p_abs, n_eb, z, hyb, gamma, regularization, weight_array, W_mn, n_C, sym=sym)
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
    eb_free = _gaps_to_eb(p[:n_eb])
    v_free = unroll(p[n_eb:-n_C], n_blocks, n_imp)
    C_final = _project_residues(_c_basis(sym), unroll_C(p[-n_C:], n_imp))
    c_final = float(
        vectorized_cost_function(
            np.concatenate([eb_free, p[n_eb:]]),
            n_eb,
            z,
            hyb,
            gamma,
            None,
            weight_array,
            W_mn,
            n_C,
            sym=sym,
        )
    )
    v_final = _expand_residue_factors(_project_factors(sym, v_free, n_eb), sym, n_eb)
    return v_final, expand_eb(eb_free, sym), C_final, c_final


def _project_factors(sym, v, n_eb):
    """Re-factor hoppings so that ``V^dagger V`` is exactly the projected residue.

    The polish parametrizes a free ``V`` and the model uses ``P_S(V^dagger V)``,
    so the ``V`` handed back to the caller must be re-derived from the projected
    residue -- otherwise the reported bath would not reproduce the fitted model.
    """
    A = _project_free_residues(sym, np.conj(np.swapaxes(v, -1, -2)) @ v, n_eb)
    lam, U = np.linalg.eigh(A)
    lam = np.clip(lam.real, 0.0, None)
    return np.sqrt(lam)[:, :, None] * np.conj(np.swapaxes(U, -1, -2))


def expand_eb(eb_free, sym):
    """Expand the freely optimized bath energies into the full pole set.

    Without particle-hole symmetry this is the identity.  With it, ``eb_free``
    holds only the positive half of a mirrored bath and the negative partners
    (plus an optional unpaired pole at zero) are generated here.

    Parameters
    ----------
    eb_free : (n_free,) array
        The energies the optimizer varies.
    sym : rspt2spectra.symmetries.BlockSymmetry
        Detected symmetry of the block.

    Returns
    -------
    (n_full,) np.ndarray
        Ascending pole energies of the model.
    """
    eb_free = np.asarray(eb_free, dtype=float)
    if not sym.particle_hole:
        return eb_free
    parts = [-eb_free[..., ::-1]]
    if sym.zero_pole:
        parts.append(np.zeros(eb_free.shape[:-1] + (1,)))
    parts.append(eb_free)
    return np.concatenate(parts, axis=-1)


def _fold_grad(grad_full, sym, n_free):
    """Map a gradient w.r.t. the full pole set back to the free parameters."""
    return _fold_energy_sensitivity(grad_full, sym, n_free)


def _sym_sectors(eb_free, z, sym):
    """Design matrices of the symmetry-adapted least-squares problem.

    Returns one sector without particle-hole symmetry and two (transpose-even
    and transpose-odd) with it.  Each sector is
    ``(basis, Phi, dPhi, n_pole_columns)`` where ``Phi[:, k]`` is the frequency
    factor multiplying the ``k``-th coefficient matrix and ``dPhi[:, k]`` its
    derivative with respect to ``eb_free[k]`` (pole columns only; the trailing
    constant / zero-pole column does not depend on any free parameter).
    """
    M = len(z)
    if not sym.particle_hole:
        G = 1.0 / (z[:, None] - eb_free[None, :])
        Phi = np.hstack([G, np.ones((M, 1))])
        return [(sym.basis, Phi, G**2, len(eb_free))]

    n_free = len(eb_free)
    Gp = 1.0 / (z[:, None] - eb_free[None, :])  # pole at +e
    Gm = 1.0 / (z[:, None] + eb_free[None, :])  # pole at -e
    sectors = []

    even = sym.even_basis
    cols = [Gp + Gm]
    if sym.zero_pole:
        cols.append((1.0 / z)[:, None])
    Phi_even = np.hstack(cols) if cols else np.zeros((M, 0), dtype=complex)
    sectors.append((even, Phi_even, Gp**2 - Gm**2, n_free))

    odd = sym.odd_basis
    Phi_odd = np.hstack([Gp - Gm, np.ones((M, 1))])
    sectors.append((odd, Phi_odd, Gp**2 + Gm**2, n_free))
    return sectors


def _sector_coefficients(basis, Phi, hyb):
    """Solve one sector's real least-squares problem.

    ``Tr(S_p S_q) = delta_pq`` is real, so the normal equations decouple across
    ``p`` and every coefficient column solves the same real system.  Solving the
    stacked real problem ``[Re Phi; Im Phi] a = [Re c; Im c]`` is equivalent and
    better conditioned than forming ``Re(Phi^H Phi)`` explicitly.
    """
    if basis.shape[0] == 0 or Phi.shape[1] == 0:
        return np.zeros((Phi.shape[1], basis.shape[0]))
    c = np.einsum("pij, mji -> mp", basis, hyb)
    Phi_r = np.vstack([Phi.real, Phi.imag])
    c_r = np.vstack([c.real, c.imag])
    a, *_ = np.linalg.lstsq(Phi_r, c_r, rcond=None)
    return a


def _assemble_from_coefficients(coefs, sectors, sym, n_free, n_imp):
    """Build the residue stack and the constant shift from sector coefficients.

    Accepts an optional leading batch axis on ``coefs`` so the same assembly is
    used for the forward solve and for its derivative.
    """
    if not sym.particle_hole:
        a = coefs[0]
        basis = sectors[0][0]
        A = np.einsum("...bp, pij -> ...bij", a[..., :n_free, :], basis)
        C = np.einsum("...p, pij -> ...ij", a[..., n_free, :], basis)
        return A, C

    a_even, a_odd = coefs
    even, odd = sectors[0][0], sectors[1][0]
    batch = a_odd.shape[:-2]
    A_pos = np.zeros(batch + (n_free, n_imp, n_imp), dtype=complex)
    if even.shape[0]:
        A_pos = A_pos + np.einsum("...bp, pij -> ...bij", a_even[..., :n_free, :], even)
    if odd.shape[0]:
        A_pos = A_pos + np.einsum("...bp, pij -> ...bij", a_odd[..., :n_free, :], odd)

    # A_{-e} = A_{+e}^T.  The residues are Hermitian, so this is a plain
    # transpose (equivalently a complex conjugation), not a conjugate transpose.
    blocks = [np.swapaxes(A_pos, -1, -2)[..., ::-1, :, :]]
    if sym.zero_pole:
        A_zero = np.zeros(batch + (1, n_imp, n_imp), dtype=complex)
        if even.shape[0]:
            A_zero = np.einsum("...p, pij -> ...ij", a_even[..., n_free, :], even)[..., None, :, :]
        blocks.append(A_zero)
    blocks.append(A_pos)

    C = np.zeros(batch + (n_imp, n_imp), dtype=complex)
    if odd.shape[0]:
        C = np.einsum("...p, pij -> ...ij", a_odd[..., n_free, :], odd)
    return np.concatenate(blocks, axis=-3), C


def _varpro_inner_solve(eb_free, z, hyb, sym):
    """Find optimal PSD residues and constant shift for fixed bath energies.

    Solves ``hyb ~= C + sum_k A_k / (z - eb[k])`` with the residues restricted to
    the symmetry-allowed subspace ``sym.basis``:

    - ``A_k`` are Hermitian, PSD (via eigh + clip) and lie in ``sym.basis``;
    - ``C`` is Hermitian and lies in ``sym.basis`` (no PSD constraint);
    - with particle-hole symmetry the poles are mirrored, ``A_{-e} = A_{+e}^T``,
      and ``eb_free`` holds only the positive half.

    Writing ``A_b = sum_p a_bp S_p`` with real ``a`` turns this into a real
    least-squares problem that decouples across ``p``; see
    :mod:`rspt2spectra.symmetries`.  Note this is the *constrained* optimum, not
    a projection of the unconstrained one -- those differ because the Gram matrix
    ``Phi^H Phi`` is complex off-diagonal while the coefficients are real.

    Parameters
    ----------
    eb_free : (n_free,) np.ndarray
        Bath energies the optimizer varies.
    z : (M,) np.ndarray
        Complex frequency mesh.
    hyb : (M, n_imp, n_imp) np.ndarray
        Hybridization block to fit.
    sym : rspt2spectra.symmetries.BlockSymmetry
        Symmetry constraints to impose.

    Returns
    -------
    A_psd : (n_full, n_imp, n_imp) np.ndarray
        PSD residues at the full pole set.
    V : (n_full, n_imp, n_imp) np.ndarray
        Hopping factors with ``V^dagger V == A_psd``.
    G : (M, n_full) np.ndarray
        ``1 / (z - eb_full)``.
    C : (n_imp, n_imp) np.ndarray
        Constant Hermitian shift.
    """
    eb_free = np.asarray(eb_free, dtype=float)
    n_free = len(eb_free)
    n_imp = hyb.shape[1]

    sectors = _sym_sectors(eb_free, z, sym)
    coefs = [_sector_coefficients(basis, Phi, hyb) for basis, Phi, _, _ in sectors]
    A, C = _assemble_from_coefficients(coefs, sectors, sym, n_free, n_imp)

    lam, U = np.linalg.eigh(A)
    lam = np.clip(lam.real, 0.0, None)
    A_psd = (U * lam[:, None, :]) @ np.conj(np.swapaxes(U, -1, -2))
    V = np.sqrt(lam)[:, :, None] * np.conj(np.swapaxes(U, -1, -2))  # (n_full, n_imp, n_imp)

    eb_full = expand_eb(eb_free, sym)
    G = 1.0 / (z[:, None] - eb_full[None, :])
    return A_psd, V, G, C


def _varpro_cost_and_grad(eb_free, z, hyb, weight_array, W_mn, sym):
    """
    VARPRO cost and its gradient w.r.t. bath energies.

    For each eb proposal, solves for optimal PSD residues A and constant shift
    C via `_varpro_inner_solve`, evaluates the fit cost, and returns the partial
    gradient w.r.t. eb (treating A_psd and C as fixed — valid by the VARPRO
    theorem for the unconstrained solve; approximate after PSD projection).

    The gradient is folded back onto the free parameters, so with particle-hole
    symmetry it has the length of the positive half of the bath.

    Returns (cost, grad_eb, V, C).
    """
    A_psd, V, G, C = _varpro_inner_solve(eb_free, z, hyb, sym)

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

    return c, _fold_grad(grad, sym, len(eb_free)), V, C


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


def _varpro_cost_and_full_grad(eb_free, z, hyb, weight_array, W_mn, sym):
    """VARPRO cost and the *exact* total-derivative gradient w.r.t. bath energies.

    Unlike `_varpro_cost_and_grad` (which uses the Kaufman simplification -- it
    treats the analytically solved residues/shift as fixed), this propagates the
    full dependence of the inner solve on eb: the derivative of the symmetry
    constrained coefficients ``a = N^-1 b`` (with ``N = Re(Phi^H Phi)``) and the
    PSD projection (Frechet derivative).  It matches a finite-difference gradient
    of the reduced cost to machine precision (away from PSD active-set
    boundaries, where the cost is only sub-differentiable).

    Returns (cost, grad_eb, V, C) with the same conventions as
    `_varpro_cost_and_grad`, i.e. the gradient is folded onto the free parameters.
    """
    eb_free = np.asarray(eb_free, dtype=float)
    n_free = len(eb_free)
    n_imp = hyb.shape[1]
    max_moment = W_mn.shape[1]

    # --- forward pass, keeping what the derivative needs per sector ---
    sectors = _sym_sectors(eb_free, z, sym)
    coefs = []
    solves = []
    for basis, Phi, _, _ in sectors:
        a = _sector_coefficients(basis, Phi, hyb)
        coefs.append(a)
        if basis.shape[0] == 0 or Phi.shape[1] == 0:
            solves.append(None)
            continue
        c_p = np.einsum("pij, mji -> mp", basis, hyb)
        N_inv = np.linalg.pinv((np.conj(Phi.T) @ Phi).real)
        solves.append((N_inv, c_p - Phi @ a))

    A_h, C_h = _assemble_from_coefficients(coefs, sectors, sym, n_free, n_imp)
    U, lam_c, A_psd, Psi = _psd_frechet_factors(A_h)

    eb_full = expand_eb(eb_free, sym)
    G = 1.0 / (z[:, None] - eb_full[None, :])

    # --- cost (identical to _varpro_cost_and_grad) ---
    hyb_model = np.einsum("mk, kij -> mij", G, A_psd) + C_h[None]
    diff = hyb - hyb_model
    w2 = weight_array**2
    N = diff.size
    cost = np.sum(w2[:, None, None] * 0.5 * np.abs(diff) ** 2) / N
    moment_diff = np.einsum("mn, mij -> nij", W_mn, diff)
    P = moment_diff[0].size * max_moment
    cost += np.sum(0.5 * np.abs(moment_diff) ** 2) / P

    # Cost gradient w.r.t. the model: dc = Re sum_m <Gbar_m, d(hyb_model)_m>.
    Gbar = -(1.0 / N) * w2[:, None, None] * np.conj(diff) - (1.0 / P) * np.einsum(
        "mn, nij -> mij", W_mn, np.conj(moment_diff)
    )  # (M, n_imp, n_imp)

    # Explicit (Kaufman) part: only the pole positions vary, at fixed residues.
    GpGbar = np.einsum("mk, mij -> kij", G**2, Gbar)  # (n_full, n_imp, n_imp)
    grad_expl = _fold_grad(np.real(np.einsum("kij, kij -> k", GpGbar, A_psd)), sym, n_free)

    # Implicit part: pull the cost gradient back through a(eb) and C(eb).
    QA = np.einsum("mk, mij -> kij", G, Gbar)  # (n_full, n_imp, n_imp)
    QC = np.sum(Gbar, axis=0)  # (n_imp, n_imp)

    # Only column k of Phi depends on eb_free[k], so with r = c - Phi a the
    # coefficient derivative is
    #   da/dk = N^-1 e_k (dphi_k^H r)  -  N^-1 (dphi_k^H Phi)^T a_k.
    das = []
    for (_basis, Phi, dphi, _), solve, a in zip(sectors, solves, coefs, strict=True):
        if solve is None:
            das.append(np.zeros((n_free,) + a.shape))
            continue
        N_inv, r = solve
        U_mat = (np.conj(dphi.T) @ Phi).real  # (n_free, K)
        W1 = (np.conj(dphi.T) @ r).real  # (n_free, p)
        da = np.einsum("jk, kp -> kjp", N_inv[:, :n_free], W1)
        da -= np.einsum("kj, kp -> kjp", U_mat @ N_inv, a[:n_free])
        das.append(da)

    dA_h, dC_h = _assemble_from_coefficients(das, sectors, sym, n_free, n_imp)

    # PSD Frechet: dA_psd[k, j] = U_j (Psi_j o (U_j^H dA_h[k, j] U_j)) U_j^H.
    UH = np.conj(np.swapaxes(U, -1, -2))  # (j, a, b)
    Mmat = np.einsum("jab, kjbc, jcd -> kjad", UH, dA_h, U) * Psi[None]
    dA_psd = np.einsum("jab, kjbc, jcd -> kjad", U, Mmat, UH)

    grad_impl = np.real(np.einsum("jab, kjab -> k", QA, dA_psd) + np.einsum("ab, kab -> k", QC, dC_h))

    V = np.sqrt(lam_c)[:, :, None] * UH
    return float(cost), grad_expl + grad_impl, V, C_h


def get_v_and_eb_varpro_basin_hopping(
    w,
    delta,
    hyb,
    ebs,
    eb_restrictions,
    gamma,
    regularization,
    weight_function,
    sym,
    max_moment=3,
    full_gradient=True,
    rng=None,
    optimize_bath_energies=True,
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

    ``rng`` seeds :func:`scipy.optimize.basinhopping`'s random displacement walk.
    Pass a :class:`numpy.random.Generator` (the caller's per-rank one) to make the
    fit reproducible -- with ``rng=None`` basin-hopping draws from the unseeded
    process-global RNG, so two fits of the *same* hybridization return different
    bath energies whenever the block has competing local minima.

    ``optimize_bath_energies`` (default ``True``): set ``False`` to freeze the bath
    energies at ``ebs[0]`` and solve only the hoppings and the constant offset, by
    least squares (one :func:`_varpro_inner_solve`). No basin-hopping, no polish --
    the returned energies are exactly the ones passed in (clipped into the window).

    ``sym`` is the block's :class:`rspt2spectra.symmetries.BlockSymmetry`.  With
    particle-hole symmetry ``ebs`` holds only the positive half of the bath and
    the returned energies are the full mirrored set.
    """
    grad_fun = _varpro_cost_and_full_grad if full_gradient else _varpro_cost_and_grad
    n_eb = ebs.shape[1]

    delta_arr = delta * (1 + 0.5 * np.abs(w) ** 2)
    z = w + 1j * delta_arr
    weight_array = weight_function(w)
    W_mn = moment_weights(w, max_moment)

    if not optimize_bath_energies:
        lo, hi = free_window(*eb_restrictions[0], delta, sym)
        eb_fixed = np.sort(np.clip(np.asarray(ebs[0], dtype=float), lo, hi))
        _, V, _, C = _varpro_inner_solve(eb_fixed, z, hyb, sym)
        cost = _varpro_cost_and_grad(eb_fixed, z, hyb, weight_array, W_mn, sym)[0]
        return V, expand_eb(eb_fixed, sym), C, float(cost)

    # Reparametrize bath energies as [first energy, gaps]; gaps >= delta keep the
    # states sorted and separated by at least the broadening, so no post-fit merge
    # is needed and reorder-equivalent configurations collapse to one.
    # With particle-hole symmetry only the positive half of the bath is free, and
    # it lives in the largest sub-window symmetric about the Fermi level.
    lo, hi = free_window(*eb_restrictions[0], delta, sym)
    # Cap (don't fail) at the number of states that fit in the window separated by
    # delta; a larger request is honoured as "as many as reasonably fit".
    n_max = _max_bath_states(lo, hi, delta)
    if n_eb > n_max:
        n_eb = n_max
        ebs = ebs[:, :n_eb]
    gap_bounds = _gap_bounds(lo, hi, n_eb, delta)
    gap_seeds = np.array([_repair_gaps(_eb_to_gaps(eb, delta), lo, hi, delta) for eb in ebs])

    initial_costs = np.array(
        [_varpro_cost_and_grad(_gaps_to_eb(p), z, hyb, weight_array, W_mn, sym)[0] for p in gap_seeds]
    )
    mean_cost = np.mean(initial_costs)
    stddev_cost = np.std(initial_costs)
    T = max(stddev_cost, 1e-3 * abs(mean_cost))

    x0 = gap_seeds[np.argmin(initial_costs)]

    def _fg(p):
        eb = _gaps_to_eb(p)
        c, g, _, _ = grad_fun(eb, z, hyb, weight_array, W_mn, sym)
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
        rng=rng,
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
        sym,
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
    sym,
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
    # With particle-hole symmetry only the positive half of the bath is free, and
    # it lives in the largest sub-window symmetric about the Fermi level.
    lo, hi = free_window(*eb_restrictions[0], delta, sym)
    # Cap (don't fail) at the number of states that fit in the window separated by
    # delta; a larger request is honoured as "as many as reasonably fit".
    n_max = _max_bath_states(lo, hi, delta)
    if n_eb > n_max:
        n_eb = n_max
        ebs = ebs[:, :n_eb]
    gap_bounds = _gap_bounds(lo, hi, n_eb, delta)
    gap_seeds = np.array([_repair_gaps(_eb_to_gaps(eb, delta), lo, hi, delta) for eb in ebs])

    def varpro_cost(p):
        eb = _gaps_to_eb(p)
        c, _, _, _ = _varpro_cost_and_grad(eb, z, hyb, weight_array, W_mn, sym)
        return float(c)

    # Linear constraint sum(gaps) <= w_max keeps the largest energy in the window;
    # box bounds on individual gaps cannot enforce this on their cumulative sum.
    res = differential_evolution(
        varpro_cost,
        Bounds(
            lb=[b[0] for b in gap_bounds],
            ub=[b[1] for b in gap_bounds],
        ),
        constraints=(LinearConstraint(np.ones((1, n_eb)), -np.inf, hi),),
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
        sym,
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
    sym=None,
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
    sym : rspt2spectra.symmetries.BlockSymmetry, optional
        Symmetry to impose on the model.  ``None`` leaves the model
        unconstrained.  Otherwise the residues are projected onto the allowed
        subspace, the constant shift onto `_c_basis`, and with particle-hole
        symmetry ``n_eb`` counts only the positive half of the mirrored bath.

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

    n_blocks = n_residue_blocks(n_eb, sym)
    n_v_end = p_batched.shape[0] - n_C if n_C else p_batched.shape[0]
    p_v = p_batched[n_eb:n_v_end]
    v = unroll(p_v, n_blocks, n_imp)

    # Build C: (n_imp, n_imp) for one_dim, (S, n_imp, n_imp) for batched.
    C = None
    if n_C:
        C_arr = _unroll_C_batch(p_batched[-n_C:], n_imp)  # (S, n_imp, n_imp)
        C = C_arr[0] if one_dim else C_arr

    if sym is None:
        model = get_hyb_2(z, eb, v, C=C)
    else:
        A_raw = np.conj(np.swapaxes(v, -1, -2)) @ v
        A = _expand_residues(_project_free_residues(sym, A_raw, n_eb), sym, n_eb)
        G = 1.0 / (z[None, :, None] - expand_eb(eb, sym)[:, None, :])
        model = np.einsum("smb, sbij -> smij", G, A)
        if C is not None:
            C_p = _project_residues(_c_basis(sym), C)
            model = model + (C_p[np.newaxis, np.newaxis] if C_p.ndim == 2 else C_p[:, np.newaxis])
    diff = hyb[np.newaxis] - model  # (S, M, N, N)

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
    sym=None,
):
    """Analytic gradient of `vectorized_cost_function`.

    Takes the same arguments as `vectorized_cost_function` and returns the
    gradient with respect to ``p``, shape ``(n_p,)`` for a 1-D ``p`` or
    ``(n_p, S)`` for a population. Verified against finite differences in
    the test suite for both real and complex hoppings, with and without an
    imposed symmetry.
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
    n_blocks = n_residue_blocks(n_eb, sym)
    n_v_end = p.shape[0] - n_C if n_C else p.shape[0]
    p_v = p[n_eb:n_v_end]
    realvalued = p_v.shape[0] == n_blocks * len(triu_cols)

    v = unroll(p_v, n_blocks, n_imp)  # (S, n_blocks, n_imp, n_imp)

    C = None
    if n_C:
        C_arr = _unroll_C_batch(p[-n_C:], n_imp)  # (S, n_imp, n_imp)
        C = C_arr[0] if one_dim else C_arr

    A_raw = np.conj(np.transpose(v, (0, 1, 3, 2))) @ v  # (S, n_blocks, N, N)
    if sym is None:
        A = A_raw
        eb_full = eb
        C_used = C
    else:
        A = _expand_residues(_project_free_residues(sym, A_raw, n_eb), sym, n_eb)  # (S, n_full, N, N)
        eb_full = expand_eb(eb, sym)
        C_used = None if C is None else _project_residues(_c_basis(sym), C)

    G = 1.0 / (z[np.newaxis, :, np.newaxis] - eb_full[:, np.newaxis, :])  # (S, M, n_full)
    model = np.einsum("smb, sbij -> smij", G, A)
    if C_used is not None:
        model = model + (C_used[np.newaxis, np.newaxis] if C_used.ndim == 2 else C_used[:, np.newaxis])
    diff = hyb[np.newaxis] - model  # (S, M, N, N)

    diff_W = diff * weight_array[None, :, None, None]

    dhyb_deb = A[:, np.newaxis, :, :, :] * (G**2)[:, :, :, np.newaxis, np.newaxis]  # (S, M, n_full, N, N)
    J_eb = -np.einsum("smxy, smbxy -> sb", np.conj(diff_W), dhyb_deb).real
    J_eb = J_eb / (n_w * n_imp * n_imp)

    if W_mn is not None:
        moment_diff = np.einsum("mn, ...mij -> ...nij", W_mn, diff)
        dmoment_deb = -np.einsum("mn, smbxy -> snbxy", W_mn, dhyb_deb)
        J_moment_eb = np.einsum("snxy, snbxy -> sb", np.conj(moment_diff), dmoment_deb).real
        J_eb = J_eb + J_moment_eb / (n_imp * n_imp * moment_diff.shape[1])
    if sym is not None:
        J_eb = _fold_energy_sensitivity(J_eb, sym, n_eb)
    J[:n_eb, :] = np.moveaxis(J_eb, 0, -1)

    # --- V gradient ---
    S_term = -np.einsum("smxy, smb -> sbxy", np.conj(diff_W), G)
    if W_mn is not None:
        WG = np.einsum("mn, smb -> snb", W_mn, G)
        S_mom = -np.einsum("snxy, snb -> sbxy", np.conj(moment_diff), WG)
        S_total = S_term / (n_w * n_imp * n_imp) + S_mom / (n_imp * n_imp * moment_diff.shape[1])
    else:
        S_total = S_term / (n_w * n_imp * n_imp)
    if sym is not None:
        # Fold the mirrored poles back onto the free residues, then pull the
        # sensitivity through the projection before it reaches V.
        S_total = _project_free_residues_adjoint(sym, _fold_residue_sensitivity(S_total, sym, n_eb), n_eb)

    J_R = np.zeros((popsize, n_blocks, n_imp, n_imp), dtype=float)
    J_I = np.zeros((popsize, n_blocks, n_imp, n_imp), dtype=float)
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
    n_real = n_blocks * len(triu_cols)
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
        if sym is not None:
            weighted_diff = _project_shift_adjoint(_c_basis(sym), weighted_diff)
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
            if sym is not None:
                weighted_mdiff = _project_shift_adjoint(_c_basis(sym), weighted_mdiff)
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

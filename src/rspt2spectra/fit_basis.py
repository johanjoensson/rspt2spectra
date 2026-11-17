"""The fitting-basis search: which orbital basis makes the hybridization function and H most block diagonal.

The caller hands over the hybridization on the fit window, the local Hamiltonian and (optionally) the
Coulomb tensor; :func:`block_diagonalize_hyb` returns the basis, chosen among symmetry-adapted
localized orbital candidates rotated in SO(3) (:func:`_tier1_candidates`) and a joint Jacobi
diagonalisation, such that the block partition at the requested tolerance is as fine as possible and
off-diagonal couplings are suppressed without making the Coulomb tensor much denser.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize
from scipy.sparse.csgraph import connected_components

from .block_structure import _g_scale, _mat_scale, _min_broadening, _relative, relative_couplings
from .utils import _rotate_u4, count_u_terms


def build_angular_momentum_operators(l):
    """Return the ``(2l+1) x (2l+1)`` angular momentum matrices ``(Lx, Ly, Lz)``.

    The basis is the complex spherical harmonics ``|l, m>`` with ``m = -l, ..., l``.
    """
    m = np.arange(-l, l + 1, dtype=float)
    n = 2 * l + 1
    Lz = np.diag(m).astype(complex)
    Lp = np.zeros((n, n), dtype=complex)
    for i, mm in enumerate(m[:-1]):
        Lp[i + 1, i] = np.sqrt(l * (l + 1) - mm * (mm + 1))
    Lm = Lp.conj().T
    return 0.5 * (Lp + Lm), -0.5j * (Lp - Lm), Lz


def complex_to_real_harmonics_unitary(l):
    """Return the unitary from complex spherical harmonics to real harmonics.

    The complex harmonics are ``|l, m>`` (``m = -l..l``, Condon-Shortley phase); the real ones are ordered
    ``m = -l..-1`` (sin terms), ``m = 0``, ``m = 1..l`` (cos terms).
    For ``l = 2``, columns correspond to ``d_xy, d_yz, d_z2, d_xz, d_x2-y2``.
    """
    s = 1.0 / np.sqrt(2.0)
    n = 2 * l + 1
    V = np.zeros((n, n), dtype=complex)
    V[l, l] = 1.0
    for m in range(1, l + 1):
        a, b = l - m, l + m
        V[[a, b], a] = [1j * s, -1j * s * (-1) ** m]
        V[[a, b], b] = [s, s * (-1) ** m]
    return V


_ZYZ_SPEC_CACHE = {}


def _zyz_spec(l):
    """Return the cached ``(m_vals, evals_y, evecs_y)`` for fast spectral ZYZ rotation at angular momentum ``l``."""
    if l not in _ZYZ_SPEC_CACHE:
        if l == 0.5:
            m_vals = np.array([-0.5, 0.5], dtype=float)
            sy = np.array([[0, 1j], [-1j, 0]], dtype=complex) / 2.0
            evals_y, evecs_y = np.linalg.eigh(sy)
        else:
            _, Ly, _ = build_angular_momentum_operators(l)
            m_vals = np.arange(-l, l + 1, dtype=float)
            evals_y, evecs_y = np.linalg.eigh(Ly)
        _ZYZ_SPEC_CACHE[l] = (m_vals, evals_y, evecs_y)
    return _ZYZ_SPEC_CACHE[l]


def _zyz_matrix(l, angles):
    """Single ``(2l+1) x (2l+1)`` ZYZ rotation ``exp(-i alpha Lz) exp(-i beta Ly) exp(-i gamma Lz)``."""
    m_vals, evals_y, evecs_y = _ZYZ_SPEC_CACHE.get(l) or _zyz_spec(l)
    a, b, g = float(angles[0]), float(angles[1]), float(angles[2])
    ea = np.exp(-1j * a * m_vals)
    eb = np.exp(-1j * b * evals_y)
    eg = np.exp(-1j * g * m_vals)
    By = (evecs_y * eb[None, :]) @ evecs_y.conj().T
    return ea[:, None] * By * eg[None, :]


def _zyz_matrix_and_grad(l, angles):
    """Single ZYZ rotation matrix and its derivatives ``(dD/dalpha, dD/dbeta, dD/dgamma)``."""
    m_vals, evals_y, evecs_y = _ZYZ_SPEC_CACHE.get(l) or _zyz_spec(l)
    a, b, g = float(angles[0]), float(angles[1]), float(angles[2])
    ea = np.exp(-1j * a * m_vals)
    eb = np.exp(-1j * b * evals_y)
    eg = np.exp(-1j * g * m_vals)
    By = (evecs_y * eb[None, :]) @ evecs_y.conj().T
    D = ea[:, None] * By * eg[None, :]
    dD_da = -1j * m_vals[:, None] * D
    dD_dg = -1j * D * m_vals[None, :]
    dBy = (evecs_y * (-1j * evals_y * eb)[None, :]) @ evecs_y.conj().T
    dD_db = ea[:, None] * dBy * eg[None, :]
    return D, (dD_da, dD_db, dD_dg)


def _zyz_batch(l, angles_batch):
    """Vectorized ZYZ rotation matrices for ``angles_batch`` of shape ``(N, 3)``."""
    m_vals, evals_y, evecs_y = _ZYZ_SPEC_CACHE.get(l) or _zyz_spec(l)
    a = angles_batch[:, 0:1]
    b = angles_batch[:, 1:2]
    g = angles_batch[:, 2:3]
    ea = np.exp(-1j * a * m_vals[None, :])
    eb = np.exp(-1j * b * evals_y[None, :])
    eg = np.exp(-1j * g * m_vals[None, :])
    By = np.einsum("ij,nj,kj->nik", evecs_y, eb, evecs_y.conj())
    return ea[:, :, None] * By * eg[:, None, :]


def _canonical_zyz_angles(theta):
    """Wrap ZYZ Euler angles into ``[-pi, pi] x [0, pi] x [-pi, pi]`` per 3-angle block."""
    theta = np.asarray(theta, dtype=float).copy()
    for offset in range(0, len(theta), 3):
        a, b, g = theta[offset : offset + 3]
        b = (b + np.pi) % (2.0 * np.pi) - np.pi
        if b < 0.0:
            b = -b
            a += np.pi
            g += np.pi
        a = (a + np.pi) % (2.0 * np.pi) - np.pi
        g = (g + np.pi) % (2.0 * np.pi) - np.pi
        theta[offset : offset + 3] = [a, b, g]
    return theta


def _shell_l(to_spherical):
    """Return the ``l`` of a single, complete, spin-resolved shell described by ``to_spherical``, else ``None``.

    Only a square unitary map whose size is ``2 (2 l + 1)`` qualifies.  Partial sets (t2g only,
    Irr subsets) have no complete shell to rotate; whether a larger square matrix is really *one*
    shell is not decided here -- :func:`_tier1_candidates` verifies that the interaction is
    invariant under the rotations it builds.
    """
    if to_spherical is None:
        return None
    S = np.asarray(to_spherical)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return None
    n = S.shape[0]
    if n % 2 or (n // 2) % 2 == 0:
        return None
    if np.max(np.abs(np.conj(S.T) @ S - np.eye(n))) > 1e-8:
        return None
    return (n // 2 - 1) // 2


def _so3_rotation(l, theta_orb, theta_spin=None):
    """Return the ``2(2l+1)`` dimensional axis-angle rotation ``D_spin (x) D_l`` on [spin down, spin up] x m blocks."""
    Lx, Ly, Lz = build_angular_momentum_operators(l)
    D = expm(-1j * (theta_orb[0] * Lx + theta_orb[1] * Ly + theta_orb[2] * Lz))
    if theta_spin is None:
        return np.kron(np.eye(2), D)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    sz = np.array([[-1, 0], [0, 1]], dtype=complex)
    Ds = expm(-0.5j * (theta_spin[0] * sx + theta_spin[1] * sy + theta_spin[2] * sz))
    return np.kron(Ds, D)


def _zyz_shell_rotation(l, theta, tied=False):
    """Return the ``2(2l+1)`` dimensional ZYZ Euler rotation ``D_spin (x) D_l`` on [spin down, spin up] x m blocks."""
    Do = _zyz_matrix(l, theta[:3])
    if tied:
        Ds = _zyz_matrix(0.5, theta[:3])
    elif len(theta) == 6:
        Ds = _zyz_matrix(0.5, theta[3:])
    else:
        Ds = np.eye(2, dtype=complex)
    return np.kron(Ds, Do)


def _so3_start_angles():
    """Deterministic starting ZYZ Euler angle triples; the identity comes first."""
    quarter, half = np.pi / 4, np.pi / 2
    return [
        np.zeros(3),
        np.array([0.0, half, 0.0]),
        np.array([half, half, 0.0]),
        np.array([0.0, 0.0, half]),
        np.array([0.0, 0.0, quarter]),
        np.array([0.0, quarter, 0.0]),
        np.array([quarter, quarter, quarter]),
        np.array([0.6, 0.8, -0.5]),
    ]


_AXIS_SEED = 20261112
_AXIS_N_SAMPLES = 600
_AXIS_N_REFINE = 6


def _off_diagonal_weight(mats, Q):
    R = np.conj(Q.T)[None] @ mats @ Q[None]
    return float(np.sum(np.abs(R) ** 2) - np.sum(np.abs(np.einsum("mii->mi", R)) ** 2))


def _compress_hermitian_set(mats, tol=1e-15):
    """Compress a stack of Hermitian matrices into an orthogonal principal set.

    The set has the same off-diagonal weight and the same Jacobi subproblems as the stack.
    """
    n = mats.shape[-1]
    if len(mats) == 0:
        return mats
    tr = np.einsum("mii->m", mats).real / n
    M0 = mats - tr[:, None, None] * np.eye(n)
    if len(M0) <= 4:
        return M0
    gram = np.einsum("mij,nji->mn", M0, M0).real
    evals, evecs = np.linalg.eigh(gram)
    scale = max(float(evals[-1]), 1e-300)
    keep = evals > tol * scale
    if not np.any(keep):
        return M0[:1]
    return np.einsum("mk,mij->kij", evecs[:, keep], M0)


def _cubic_field_operator(l):
    """Octahedral crystal-field operator ``O40 + 5 O44`` on angular momentum ``l``."""
    Lx, Ly, Lz = build_angular_momentum_operators(l)
    Lp, Lm = Lx + 1j * Ly, Lx - 1j * Ly
    L2 = l * (l + 1)
    Lz2 = Lz @ Lz
    O40 = 35 * Lz2 @ Lz2 - (30 * L2 - 25) * Lz2 + (3 * L2**2 - 6 * L2) * np.eye(2 * l + 1)
    O44 = 0.5 * (np.linalg.matrix_power(Lp, 4) + np.linalg.matrix_power(Lm, 4))
    return O40 + 5 * O44


def _cubic_f_harmonics():
    """Cubic harmonics of an f shell (a2u, t1u, t2u): real harmonics diagonalising the cubic field.

    The cubic crystal field is ``O40 + 5 O44`` (Stevens operators).  It is diagonalised inside each
    set of real harmonics it couples (the parity classes of the cube's mirror planes), which
    leaves the degenerate t levels in a fixed, symmetric choice of orbitals.
    """
    l = 3
    T = complex_to_real_harmonics_unitary(l)
    M = np.conj(T.T) @ _cubic_field_operator(l) @ T
    M = 0.5 * (M + np.conj(M.T))
    n_classes, labels = connected_components(np.abs(M) > 1e-9, directed=False)
    V = np.zeros_like(M)
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        _, vecs = np.linalg.eigh(M[np.ix_(idx, idx)])
        V[np.ix_(idx, idx)] = vecs
    return T @ V


def _coupled_j_basis(l):
    """``|j, m_j>`` states of a shell, in [spin down, spin up] x ``m`` ordering (columns by ``L.S`` then ``J_z``)."""
    Lops = build_angular_momentum_operators(l)
    n = 2 * l + 1
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, 1j], [-1j, 0]], dtype=complex) / 2
    sz = np.diag([-0.5, 0.5]).astype(complex)
    LS = sum(np.kron(sp, Lc) for sp, Lc in zip((sx, sy, sz), Lops))
    Jz = np.kron(sz, np.eye(n)) + np.kron(np.eye(2), Lops[2])
    _, vecs = np.linalg.eigh(LS + 1e-3 * Jz)
    return vecs


def _cubic_jj_basis(l):
    """Octahedral crystal-field basis (``Gamma_6, Gamma_7, Gamma_8``) within each ``j = l +- 1/2`` manifold."""
    Wj = _coupled_j_basis(l)
    V_cub = np.kron(np.eye(2), _cubic_field_operator(l))
    n_minus = 2 * l
    n_tot = 2 * (2 * l + 1)
    U = np.zeros((n_tot, n_tot), dtype=complex)
    for sl in (slice(0, n_minus), slice(n_minus, n_tot)):
        W_sub = Wj[:, sl]
        M = np.conj(W_sub.T) @ V_cub @ W_sub
        M = 0.5 * (M + np.conj(M.T))
        n_comp, labels = connected_components(np.abs(M) > 1e-9, directed=False)
        V_sub = np.zeros_like(M)
        for c in range(n_comp):
            idx = np.flatnonzero(labels == c)
            _, vecs = np.linalg.eigh(M[np.ix_(idx, idx)])
            V_sub[np.ix_(idx, idx)] = vecs
        U[:, sl] = W_sub @ V_sub
    return U


def _components(mask):
    return connected_components(mask, directed=False, return_labels=True)


def _partition_key(sizes):
    """Order on partitions, smaller is better: largest block, then sum of squared sizes, then more blocks."""
    sizes = list(sizes)
    return (max(sizes), sum(x * x for x in sizes), -len(sizes))


def _block_threshold(c, k):
    """Smallest ``tol`` at which the blocks of ``c > tol`` have at most ``k`` orbitals each (``0`` if always)."""
    n = c.shape[0]
    rows, cols = np.triu_indices(n, 1)
    values = c[rows, cols]
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in np.argsort(-values, kind="stable"):
        ri, rj = find(rows[e]), find(cols[e])
        if ri == rj:
            continue
        if size[ri] + size[rj] > k:
            return float(values[e])
        parent[rj] = ri
        size[ri] += size[rj]
    return 0.0


def _coupling_matrix(Q, hyb, mat, tol, mat_spread=None):
    G = np.conj(Q.T)[None] @ hyb @ Q[None]
    if mat_spread is None:
        H = None if mat is None else np.conj(Q.T) @ mat @ Q
        return relative_couplings(G, H, tol)
    if G.ndim == 2:
        G = G[None]
    parts = [_relative(np.max(np.abs(G), axis=0), _g_scale(G), tol)]
    if mat is not None:
        H = np.conj(Q.T) @ mat @ Q
        parts.append(_relative(H, max(mat_spread, _min_broadening(G)), tol))
    c = np.maximum.reduce(parts)
    return np.maximum(c, c.T)


def _quality(c, tol):
    """``(partition key, largest coupling between blocks)`` of a coupling matrix at ``tol``."""
    _, labels = _components(c > tol)
    sizes = np.bincount(labels)
    between = c[labels[:, None] != labels[None, :]]
    return _partition_key(sizes), (float(between.max()) if between.size else 0.0)


def _margin_better(new, old, tol):
    """Whether a coupling left between blocks is clearly smaller; couplings far below ``tol`` do not matter."""
    return old > 0.1 * tol and new < 0.95 * old


def _basis_metrics(Q, mats, total_off, hyb, mat, tol, mat_spread=None):
    """Full blockiness and off-diagonal metrics of a candidate basis ``Q``."""
    c = _coupling_matrix(Q, hyb, mat, tol, mat_spread=mat_spread)
    key, between = _quality(c, tol)
    b_max = int(key[0])
    sub_thresholds = tuple(float(_block_threshold(c, k)) for k in range(b_max - 1, 0, -1))
    next_split = sub_thresholds[0] if sub_thresholds else 0.0
    off_ratio = _off_diagonal_weight(mats, Q) / max(total_off, 1e-300)
    return {
        "key": key,
        "margin": between,
        "b_max": b_max,
        "sub_thresholds": sub_thresholds,
        "next_split": next_split,
        "off_ratio": off_ratio,
    }


def _blockiness_better(cand, best, tol, gain=0.05):
    """Whether ``cand`` has clearly better intra-block blockiness or off-diagonal suppression than ``best``.

    Meant for two bases that give the same partition at ``tol`` with ``b_max > 1``.
    """
    if max(cand["next_split"], best["next_split"]) <= 0.1 * tol and max(cand["off_ratio"], best["off_ratio"]) <= 1e-6:
        return False
    split_better = best["next_split"] > 0.1 * tol and cand["next_split"] < (1.0 - gain) * best["next_split"]
    split_not_worse = cand["next_split"] <= (1.0 + gain) * best["next_split"] + 0.1 * tol
    off_better = best["off_ratio"] > 1e-6 and cand["off_ratio"] < (1.0 - gain) * best["off_ratio"]
    off_not_worse = cand["off_ratio"] <= (1.0 + gain) * best["off_ratio"] + 1e-6
    return (split_better and off_not_worse) or (off_better and split_not_worse)


def _frame_better(cand, best, tol):
    """Comparison between two Euler-angle frames of the same localized orbital shape (equal U-terms)."""
    if cand["key"] != best["key"]:
        return cand["key"] < best["key"]
    if _margin_better(cand["margin"], best["margin"], tol):
        return True
    if _margin_better(best["margin"], cand["margin"], tol):
        return False
    if cand["b_max"] > 1:
        if _blockiness_better(cand, best, tol, gain=0.05):
            return True
        if _blockiness_better(best, cand, tol, gain=0.05):
            return False
        if cand["next_split"] <= best["next_split"] * 1.01 + 0.1 * tol and cand["off_ratio"] < best["off_ratio"] - 1e-6:
            return True
    return False


def _frame_starts(n_par):
    """Deterministic ZYZ Euler angle frames to start from: identity, structured ones, and a seeded cloud.

    A fixed seed makes every rank and every DMFT iteration scan the same cloud, so the choice only
    changes when the data does.
    """
    rng = np.random.default_rng(_AXIS_SEED)
    starts = [np.zeros(n_par)]
    starts += [np.concatenate([a, np.zeros(n_par - 3)]) for a in _so3_start_angles()[1:]]
    for _ in range(_AXIS_N_SAMPLES):
        # uniform on the rotation group: alpha and gamma uniform, cos(beta) uniform
        sample = rng.uniform(-np.pi, np.pi, n_par)
        sample[1::3] = np.arccos(rng.uniform(-1.0, 1.0, n_par // 3))
        starts.append(sample)
    return starts


def _zyz_weight_minimising_frame(l, mats_sph, W, starts, tied=False, improvement=1e-9):
    """Find the ZYZ Euler-angle frame minimising ``_off_diagonal_weight(mats_sph, D(theta) @ W)``.

    A vectorized scan of the start frames picks the few that BFGS (analytic gradient) then refines.
    """
    P = _compress_hermitian_set(mats_sph)
    total_norm2 = float(np.sum(np.abs(P) ** 2))
    if total_norm2 <= 0:
        return np.zeros(len(starts[0]))

    starts_arr = np.asarray(starts, dtype=float)
    n_par = starts_arr.shape[1]
    d_orb = 2 * l + 1
    n = 2 * d_orb

    Do_b = _zyz_batch(l, starts_arr[:, :3])
    if tied:
        Ds_b = _zyz_batch(0.5, starts_arr[:, :3])
        D_b = np.einsum("nab,ncd->nacbd", Ds_b, Do_b).reshape(-1, n, n)
    elif n_par == 6:
        Ds_b = _zyz_batch(0.5, starts_arr[:, 3:])
        D_b = np.einsum("nab,ncd->nacbd", Ds_b, Do_b).reshape(-1, n, n)
    else:
        D_b = np.zeros((len(starts_arr), n, n), dtype=complex)
        D_b[:, :d_orb, :d_orb] = Do_b
        D_b[:, d_orb:, d_orb:] = Do_b

    Q_b = D_b @ W[None, :, :]
    diag_scan = np.einsum("nai,mab,nbi->nmi", Q_b.conj(), P, Q_b).real
    scores = (total_norm2 - np.sum(diag_scan**2, axis=(1, 2))) / total_norm2

    def loss_and_grad(theta):
        Do, dDo = _zyz_matrix_and_grad(l, theta[:3])
        if tied:
            Ds, dDs = _zyz_matrix_and_grad(0.5, theta[:3])
            D = np.kron(Ds, Do)
        elif n_par == 6:
            Ds, dDs = _zyz_matrix_and_grad(0.5, theta[3:])
            D = np.kron(Ds, Do)
        else:
            Ds = np.eye(2, dtype=complex)
            D = np.kron(Ds, Do)

        Q = D @ W
        MQ = P @ Q[None, :, :]
        diag_R = np.einsum("ai,mai->mi", Q.conj(), MQ).real
        loss = (total_norm2 - float(np.sum(diag_R**2))) / total_norm2
        G_Q = (-4.0 / total_norm2) * np.einsum("mij,mj->ij", MQ, diag_R)
        G_D = (G_Q @ np.conj(W.T)).reshape(2, d_orb, 2, d_orb)
        G_orb = np.einsum("ac,abcd->bd", Ds.conj(), G_D)
        grad_orb = [np.real(np.vdot(G_orb, dDo_k)) for dDo_k in dDo]
        if tied:
            G_sp = np.einsum("bd,abcd->ac", Do.conj(), G_D)
            grad = np.array([go + np.real(np.vdot(G_sp, dDs_k)) for go, dDs_k in zip(grad_orb, dDs)])
        elif n_par == 6:
            G_sp = np.einsum("bd,abcd->ac", Do.conj(), G_D)
            grad_sp = [np.real(np.vdot(G_sp, dDs_k)) for dDs_k in dDs]
            grad = np.array(grad_orb + grad_sp)
        else:
            grad = np.array(grad_orb)
        return loss, grad

    best_theta, best_cost = starts_arr[0], float(scores[0])
    for k in np.argsort(scores, kind="stable")[:_AXIS_N_REFINE]:
        res = minimize(
            loss_and_grad,
            starts_arr[k],
            jac=True,
            method="BFGS",
            options={"gtol": 1e-12, "maxiter": 300},
        )
        if res.fun < best_cost - improvement:
            best_theta, best_cost = res.x, float(res.fun)
    return _canonical_zyz_angles(best_theta)


def _partition_frames(l, W, tied, g_sub, mat_sph, tol, mat_spread, starts, anchors=()):
    """Frames that give the finest block partition at ``tol`` or smallest sub-block splitting thresholds.

    Uses a vectorized batch scan on a peak-frequency subset to select the most promising starting
    frames, then refines the single-linkage threshold ``_block_threshold(c, target)`` with Nelder--Mead.
    Retains the best refined frames both when they cross ``tol`` and when they improve the sub-block
    splitting threshold above ``tol``.
    """

    def coupling(theta):
        q_sph = _zyz_shell_rotation(l, theta, tied=tied) @ W
        return _coupling_matrix(q_sph, g_sub, mat_sph, tol, mat_spread=mat_spread)

    # Fast vectorized pre-screening of the random cloud on up to 12 peak frequencies of g_sub
    starts_arr = np.asarray(starts, dtype=float)
    n_par = starts_arr.shape[1]
    d_orb = 2 * l + 1
    n = 2 * d_orb
    Do_b = _zyz_batch(l, starts_arr[:, :3])
    if tied:
        Ds_b = _zyz_batch(0.5, starts_arr[:, :3])
        D_b = np.einsum("nab,ncd->nacbd", Ds_b, Do_b).reshape(-1, n, n)
    elif n_par == 6:
        Ds_b = _zyz_batch(0.5, starts_arr[:, 3:])
        D_b = np.einsum("nab,ncd->nacbd", Ds_b, Do_b).reshape(-1, n, n)
    else:
        D_b = np.zeros((len(starts_arr), n, n), dtype=complex)
        D_b[:, :d_orb, :d_orb] = Do_b
        D_b[:, d_orb:, d_orb:] = Do_b
    Q_b = D_b @ W[None, :, :]

    if len(g_sub) > 12:
        frob = np.linalg.norm(g_sub, axis=(1, 2))
        top_w = np.argsort(-frob)[:12]
        g_fast = g_sub[top_w]
    else:
        g_fast = g_sub
    # G_b: (N_starts, n_w_fast, n, n)
    G_b = np.einsum("nai,wab,nbj->nwij", Q_b.conj(), g_fast, Q_b)
    max_G_b = np.max(np.abs(G_b), axis=1)
    g_scales = np.max(np.abs(np.einsum("nwii->nwi", G_b)), axis=(1, 2))
    if mat_sph is not None:
        H_b = np.einsum("nai,ab,nbj->nij", Q_b.conj(), mat_sph, Q_b)
        abs_H_b = np.abs(H_b)
        min_broad = np.min(np.mean(np.abs(np.einsum("nwii->nwi", G_b).imag), axis=1), axis=1)

    fast_scores = []
    for idx_s in range(len(starts_arr)):
        c_g = _relative(max_G_b[idx_s], float(g_scales[idx_s]), tol)
        if mat_sph is not None:
            h_sc = max(mat_spread, float(min_broad[idx_s]))
            c_h = _relative(abs_H_b[idx_s], h_sc, tol)
            c_s = np.maximum(c_g, c_h)
        else:
            c_s = c_g
        c_s = np.maximum(c_s, c_s.T)
        key, _ = _quality(c_s, tol)
        fast_scores.append((key, _block_threshold(c_s, max(key[0] - 1, 1))))

    top_scan_idx = sorted(range(len(starts_arr)), key=lambda i: fast_scores[i])[:24]
    candidate_starts = list(anchors) + [starts_arr[i] for i in top_scan_idx]

    scan = []
    for t in candidate_starts:
        c = coupling(t)
        key, _ = _quality(c, tol)
        scan.append((key, _block_threshold(c, max(key[0] - 1, 1))))
    order = sorted(range(len(candidate_starts)), key=lambda i: scan[i])
    best = candidate_starts[order[0]]
    frames = [best]
    largest = scan[order[0]][0][0]
    if largest <= 1:
        return frames

    targets = [largest - step for step in range(1, min(largest, 5)) if largest - step >= 1]
    for target in targets:
        best_target_fun = _block_threshold(coupling(best), target)
        best_target_x = None
        seeds = [best] + [candidate_starts[i] for i in order[:_PARTITION_N_REFINE]]
        for p in seeds:
            res = minimize(
                lambda t, tgt=target: _block_threshold(coupling(t), tgt),
                p,
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": _PARTITION_MAXITER},
            )
            if res.fun < best_target_fun - 1e-6:
                best_target_fun = float(res.fun)
                best_target_x = _canonical_zyz_angles(res.x)
            if res.fun <= tol:
                break
        if best_target_x is not None:
            frames.append(best_target_x)
            if best_target_fun <= tol:
                best = best_target_x
            else:
                break
        else:
            break
    return frames


_PARTITION_N_REFINE = (
    8  # 3 missed the partition frame for 2 of 12 start clouds on the two-plus-three test problem; 8 for none
)
_PARTITION_MAXITER = 400
_JD_MAX_FREQ = 50
_JD_MAX_SWEEPS = 100
_JD_SKIP = 1e-8  # relative to the O(1) scaled matrices; couplings below are numerical noise
_JD_REL_DECREASE = 1e-10  # stop when a sweep removes less than this fraction of the off-diagonal weight
_JD_MAT_WEIGHT = 0.1  # weight of H relative to the whole hyb set: H breaks hyb degeneracies, hyb sets the basis
_JD_MAT_FLOOR = 1e-3  # Ry; crystal-field spread below this is noise when weighting H against hyb


def _subsample_indices(hyb_win, max_freq=_JD_MAX_FREQ):
    """Select up to ``max_freq`` frequency indices from ``hyb_win``.

    Combines spectral-weight quantiles, off-diagonal pair maxima, and uniform coverage of the window.
    """
    n_w = len(hyb_win)
    if n_w <= max_freq:
        return np.arange(n_w, dtype=int)
    frob = np.linalg.norm(hyb_win, axis=(1, 2))
    total_frob = float(np.sum(frob))
    if total_frob <= 0:
        return np.unique(np.linspace(0, n_w - 1, max_freq).astype(int))

    # 1. Peak indices of non-zero pairwise magnitudes |hyb_win[:, i, j]|
    abs_h = np.abs(hyb_win)
    pair_max = np.max(abs_h, axis=0)
    coupled = pair_max > 0
    # strongest pairs first, so a truncation below drops the weakest peaks
    peak_idx = np.argmax(abs_h, axis=0)[coupled][np.argsort(-pair_max[coupled], kind="stable")]

    # 2. Spectral-weight quantiles
    cdf = np.cumsum(frob) / total_frob
    n_quant = max(max_freq // 2, 10)
    quant_idx = np.searchsorted(cdf, np.linspace(0.0, 1.0, n_quant))
    quant_idx = np.clip(quant_idx, 0, n_w - 1)

    # 3. Uniform anchors across the window
    n_unif = max(max_freq - len(np.unique(np.concatenate([peak_idx, quant_idx]))), 10)
    unif_idx = np.linspace(0, n_w - 1, n_unif).astype(int)

    combined = np.unique(np.concatenate([peak_idx, quant_idx, unif_idx]))
    if len(combined) > max_freq:
        # The peaks are what the partition is decided on: keep them, fill up evenly from the rest.
        _, first = np.unique(peak_idx, return_index=True)
        peaks = peak_idx[np.sort(first)][:max_freq]
        rest = np.setdiff1d(combined, peaks)
        n_fill = min(max_freq - len(peaks), len(rest))
        fill = rest[np.unique(np.linspace(0, len(rest) - 1, n_fill).astype(int))] if n_fill > 0 else rest[:0]
        combined = np.union1d(peaks, fill)
    return combined


def _tier1_candidates(mats, hyb, mat, u4, to_spherical, budget, tol, custom_shapes=None, details=None, rejected=None):
    """Rigid-rotation bases of the correlated shell: ``[(label, Q)]``, one per shape.

    A rotation of the shell leaves a rotationally invariant Coulomb tensor exactly as sparse as it
    is.  Each *shape* below is a fixed unitary ``W`` from the spherical harmonics to the orbitals
    (what the orbitals look like); the candidate basis is ``Q = S D(frame) W`` where ``S = to_spherical``
    maps the spherical harmonics to the fit orbitals and ``D`` rotates orbital (and, if the matrices
    couple spins, spin) space.  The rotated tensor does not depend on the frame, so a shape's cost is
    known up front and shapes beyond ``budget`` are not searched.

    * input orbitals (``W = S^dagger``): the basis the fit orbitals arrive in, rotated;
    * complex harmonics: axial symmetry (a C3, C4 or C6 axis, an orbital moment);
    * real harmonics: mirror planes, D2h/D4h/octahedral fields;
    * cubic f harmonics (``l = 3``): the octahedral f levels;
    * ``|j, m_j>`` (spin-coupled matrices only): strong spin-orbit coupling;
    * cubic j-j (spin-coupled ``l >= 2``): octahedral levels (Gamma_6/Gamma_7/Gamma_8) inside ``j = l +- 1/2``.
    """
    l = _shell_l(to_spherical)
    if l is None:
        return []
    S = np.asarray(to_spherical, dtype=complex)
    n = S.shape[0]
    half = n // 2

    def frame(theta, tied=False):
        return _zyz_shell_rotation(l, theta, tied=tied)

    # The interaction must survive a generic rotation of both frames, else this is not the shell
    # we think it is (or U is not rotationally invariant) and the premise of the tier is gone.
    u_op = u4 - u4.transpose(1, 0, 2, 3) - u4.transpose(0, 1, 3, 2) + u4.transpose(1, 0, 3, 2)
    scale = max(float(np.max(np.abs(u_op))), 1e-300)
    for angles in ([0.7, 1.1, 0.4, -0.5, 0.9, 0.3], [-0.3, 0.8, 1.2, 0.8, 0.6, 1.0]):
        probe = S @ frame(np.array(angles)) @ np.conj(S.T)
        if np.max(np.abs(_rotate_u4(u_op, probe) - u_op)) > 1e-8 * scale:
            return []

    total = _off_diagonal_weight(mats, np.eye(n, dtype=complex))
    mat_spread = None if mat is None else _mat_scale(mat)
    if total <= 0:
        Q_id = np.eye(n, dtype=complex)
        if details is not None:
            m_id = _basis_metrics(Q_id, mats, 1.0, hyb, mat, tol, mat_spread=mat_spread)
            details["input orbitals"] = (np.zeros(3), m_id)
        return [("input orbitals", Q_id)]

    # Express matrices in the spherical frame once so inner loops do not multiply by S
    mats_sph = np.conj(S.T)[None] @ mats @ S[None]
    hyb_sph = np.conj(S.T)[None] @ hyb @ S[None]
    mat_sph = None if mat is None else np.conj(S.T) @ mat @ S
    spin_coupled = float(np.sum(np.abs(mats_sph[:, :half, half:]) ** 2)) > 1e-12 * float(np.sum(np.abs(mats) ** 2))

    I2 = np.eye(2)
    shapes = [
        ("input orbitals", np.conj(S.T), False),
        ("complex harmonics", np.eye(n, dtype=complex), False),
    ]
    if l >= 1:
        shapes.append(("real harmonics", np.kron(I2, complex_to_real_harmonics_unitary(l)), False))
    if l == 3:
        shapes.append(("cubic f harmonics", np.kron(I2, _cubic_f_harmonics()), False))
    if spin_coupled:
        shapes.append(("j-j coupled", _coupled_j_basis(l), True))
        if l in (2, 3):
            shapes.append(("cubic j-j", _cubic_jj_basis(l), True))
    if custom_shapes:
        for item in custom_shapes:
            if len(item) == 2:
                shapes.append((item[0], np.asarray(item[1], dtype=complex), False))
            else:
                shapes.append((item[0], np.asarray(item[1], dtype=complex), bool(item[2])))

    sub = _subsample_indices(hyb_sph, _JD_MAX_FREQ)
    g_sub = hyb_sph[sub]
    starts = {3: _frame_starts(3), 6: _frame_starts(6)}

    out = []
    evaluated_shapes = []
    for label, W, tied in shapes:
        n_par = 3 if (tied or not spin_coupled) else 6
        terms = count_u_terms(_rotate_u4(u4, S @ W))
        if terms > budget:
            if rejected is not None:
                # Not searched, but its weight-minimising frame says what partition the budget costs.
                theta = _zyz_weight_minimising_frame(l, mats_sph, W, starts[n_par], tied=tied)
                m = _basis_metrics(
                    frame(theta, tied) @ W, mats_sph, total, hyb_sph, mat_sph, tol, mat_spread=mat_spread
                )
                rejected.append({"label": label, "terms": terms, "key": m["key"]})
            continue
        # Reuse result if W (and tied) is identical to an already-searched shape (e.g. input == complex harmonics)
        reused = None
        for prev_W, prev_tied, prev_res in evaluated_shapes:
            if prev_tied == tied and np.allclose(W, prev_W, atol=1e-12):
                reused = prev_res
                break
        if reused is not None:
            Q_cand, best_theta, best_m = reused
            if details is not None:
                details[label] = (best_theta, best_m)
            out.append((label, Q_cand))
            continue

        def make_q_sph(theta, W=W, tied=tied):
            return frame(theta, tied) @ W

        theta_w = _zyz_weight_minimising_frame(l, mats_sph, W, starts[n_par], tied=tied)
        anchor_thetas = [theta_w, np.zeros(n_par)]
        if mat_sph is not None and _off_diagonal_weight(mat_sph[None], np.eye(n, dtype=complex)) > 1e-12:
            theta_h = _zyz_weight_minimising_frame(l, mat_sph[None], W, starts[n_par][:64], tied=tied)
            anchor_thetas.append(theta_h)

        best_m, best_q_sph, best_theta = None, None, None
        for theta in anchor_thetas:
            q_sph = make_q_sph(theta)
            m = _basis_metrics(q_sph, mats_sph, total, hyb_sph, mat_sph, tol, mat_spread=mat_spread)
            if best_m is None or _frame_better(m, best_m, tol):
                best_m, best_q_sph, best_theta = m, q_sph, theta

        # Run the Nelder-Mead partition threshold search only when blocks of size > 1 remain at tol
        # (or significant inter-block couplings remain)
        if best_m["b_max"] > 1 or best_m["margin"] > 0.1 * tol:
            part_thetas = _partition_frames(
                l, W, tied, g_sub, mat_sph, tol, mat_spread, starts[n_par], anchors=anchor_thetas
            )
            for theta in part_thetas:
                q_sph = make_q_sph(theta)
                m = _basis_metrics(q_sph, mats_sph, total, hyb_sph, mat_sph, tol, mat_spread=mat_spread)
                if _frame_better(m, best_m, tol):
                    best_m, best_q_sph, best_theta = m, q_sph, theta

        Q_cand = S @ best_q_sph
        canon_theta = _canonical_zyz_angles(best_theta)
        evaluated_shapes.append((W, tied, (Q_cand, canon_theta, best_m)))
        if details is not None:
            details[label] = (canon_theta, best_m)
        out.append((label, Q_cand))
    return out


def _choose_basis(mats, hyb_window, mat, u4, to_spherical, max_u_densify, tol, info, custom_shapes=None):
    """Pick the unitary that blocks best at ``tol`` among the bases whose Coulomb tensor stays within budget.

    Candidates are the rigid-rotation bases of :func:`_tier1_candidates`, the free joint rotation, and,
    when that is too dense, the free rotation restricted to the budget and started from the best rigid
    one.  The finest partition at ``tol`` wins.  On a tied partition a named shape (rigid rotation) always
    beats a free/budgeted rotation: a generic rotation inside degenerate levels spreads the ground state
    over more determinants, which neither a sparser Coulomb tensor nor finer splitting below ``tol`` makes
    up for.  The free rotations therefore only win by a strictly finer partition, or when no named shape
    fits the budget.  Among rigid rotations, when unsplit blocks (size > 1) remain, a >= 5% gain in the
    next sub-block splitting threshold or off-diagonal weight wins; when blockiness is comparable (or all
    blocks are size 1), the sparser Coulomb tensor wins.
    """
    n = mats.shape[-1]
    n_ref = count_u_terms(u4)
    if not np.isfinite(max_u_densify) or max_u_densify * n_ref >= n**4:
        budget = n**4
    else:
        budget = max(int(np.floor(max_u_densify * n_ref * (1 + 1e-12) + 1e-9)), n_ref)
    info["u_terms_input"] = n_ref
    total_off = max(_off_diagonal_weight(mats, np.eye(n, dtype=complex)), 1e-300)
    mat_spread = None if mat is None else _mat_scale(mat)

    best = {}

    def consider(tier, label, Q, euler_angles=None, metrics=None):
        terms = count_u_terms(_rotate_u4(u4, Q))
        if terms > budget:
            return False
        if metrics is None:
            metrics = _basis_metrics(Q, mats, total_off, hyb_window, mat, tol, mat_spread=mat_spread)
        cand = {
            "tier": tier,
            "label": label,
            "Q": Q,
            "terms": terms,
            "euler_angles": euler_angles,
            **metrics,
        }
        if best:
            if cand["key"] != best["key"]:
                if cand["key"] > best["key"]:
                    return False
            elif best["tier"] == "axis rotation" and tier != "axis rotation":
                # Same partition as a named shape: the named shape stays. A free/budgeted rotation mixes orbitals
                # the symmetry keeps apart (degenerate levels, spin), which spreads the ground state over more
                # determinants; neither fewer Coulomb terms nor finer splitting below tol inside blocks pays for
                # that (FM Gd: free Jacobi splits its 3- and 2-blocks 4x further and needs 3x the determinants).
                return False
            else:
                # Same partition at tol, among rigid rotations
                if cand["b_max"] == 1:
                    if terms < best["terms"]:
                        if _margin_better(best["margin"], cand["margin"], tol):
                            return False
                    elif terms > best["terms"]:
                        if not (best["margin"] > 0.1 * tol and cand["margin"] < 0.95 * best["margin"]):
                            return False
                    else:
                        if not _margin_better(cand["margin"], best["margin"], tol):
                            return False
                else:
                    # Unsplit blocks of size > 1 remain at tol: compare intra-block blockiness & off-diag weight
                    if _blockiness_better(cand, best, tol, gain=0.05):
                        pass
                    elif _blockiness_better(best, cand, tol, gain=0.05):
                        return False
                    elif terms != best["terms"]:
                        if terms > best["terms"]:
                            return False
                    elif not _margin_better(cand["margin"], best["margin"], tol):
                        return False
        best.clear()
        best.update(cand)
        return True

    tier1_details = {}
    rejected = []
    for label, Q in _tier1_candidates(
        mats,
        hyb_window,
        mat,
        u4,
        to_spherical,
        budget,
        tol,
        custom_shapes=custom_shapes,
        details=tier1_details,
        rejected=rejected,
    ):
        euler_angles, metrics = tier1_details.get(label, (None, None))
        consider("axis rotation", label, Q, euler_angles=euler_angles, metrics=metrics)
    start = best["Q"] if best else np.eye(n, dtype=complex)
    Q_free = _joint_diagonalize(mats)
    if not consider("free rotation", "", Q_free):
        free_terms = count_u_terms(_rotate_u4(u4, Q_free))
        if best and free_terms > budget:
            m_free = _basis_metrics(Q_free, mats, total_off, hyb_window, mat, tol, mat_spread=mat_spread)
            rejected.append({"label": "free rotation", "terms": free_terms, "key": m_free["key"]})
        consider(
            "budgeted free rotation",
            "",
            _joint_diagonalize(mats, Q0=start, u4=u4, budget=budget),
        )
    if not best:
        consider("input basis", "", np.eye(n, dtype=complex))
    info.update(
        tier=best["tier"],
        shape=best["label"],
        u_terms=best["terms"],
        off_diag_ratio=best["off_ratio"],
        sub_thresholds=best["sub_thresholds"],
        euler_angles=best["euler_angles"],
        # what the Coulomb budget excluded although it would block more finely: [(label, terms, largest block, blocks)]
        finer_beyond_budget=[
            (r["label"], r["terms"], int(r["key"][0]), int(-r["key"][2]))
            for r in sorted(rejected, key=lambda r: r["key"])
            if r["key"] < best["key"]
        ],
    )
    return best["Q"]


def _joint_diagonalize(mats, max_sweeps=_JD_MAX_SWEEPS, skip=_JD_SKIP, Q0=None, u4=None, budget=None):
    """Approximately diagonalize a set of Hermitian matrices with one unitary.

    Complex Jacobi iteration after Cardoso and Souloumiac: every sweep rotates each
    orbital pair by the angle that minimises the summed squared off-diagonal weight of
    all matrices.  The start is the identity and every rotation is non-increasing, so
    the result is never less diagonal than the input.  A pair whose couplings vanish in
    every matrix is skipped, which keeps decoupled orbitals (spin up/down, separate
    blocks) from being mixed.
    """
    A = np.array(mats, dtype=complex)
    n = A.shape[-1]
    Q = np.eye(n, dtype=complex)
    if Q0 is not None:
        Q = np.array(Q0, dtype=complex)
        A = np.conj(Q.T)[None] @ A @ Q[None]
    constrained = u4 is not None and budget is not None
    if constrained:
        U_now = _rotate_u4(u4, Q)

    h_spin = n // 2
    paired = (
        constrained
        and n % 2 == 0
        and np.max(np.abs(A[:, :h_spin, :h_spin] - A[:, h_spin:, h_spin:])) < _JD_SKIP
        and np.max(np.abs(A[:, :h_spin, h_spin:])) < _JD_SKIP
    )

    def off_weight():
        return float(np.sum(np.abs(A) ** 2) - np.sum(np.abs(np.einsum("mii->mi", A)) ** 2))

    previous = off_weight()
    for _ in range(max_sweeps):
        rotated = False
        limit = h_spin if paired else n
        pairs = [(p, q) for p in range(limit - 1) for q in range(p + 1, limit)]
        if constrained:
            pairs.sort(key=lambda pq: -float(np.sum(np.abs(A[:, pq[0], pq[1]]) ** 2)))
        for p, q in pairs:
            a_pq = A[:, p, q]
            if np.sum(np.abs(a_pq) ** 2) < skip**2:
                continue
            h = np.stack(
                [
                    (A[:, p, p] - A[:, q, q]).real,
                    2 * a_pq.real,
                    2 * a_pq.imag,
                ]
            )
            _, eigvecs = np.linalg.eigh(h @ h.T)
            x, y, z = eigvecs[:, -1]
            if x < 0:
                x, y, z = -x, -y, -z
            r = np.sqrt(x * x + y * y + z * z)
            c = np.sqrt((x + r) / (2 * r))
            sn = (y - 1j * z) / np.sqrt(2 * r * (x + r)) if (x + r) > 0 else 0.0
            if abs(sn) < 1e-15:
                continue
            moves = [(p, q), (p + h_spin, q + h_spin)] if paired else [(p, q)]
            if constrained:
                J = np.eye(n, dtype=complex)
                for pp, qq in moves:
                    J[pp, pp] = J[qq, qq] = c
                    J[qq, pp] = sn
                    J[pp, qq] = -np.conj(sn)
                U_trial = _rotate_u4(U_now, J)
                if count_u_terms(U_trial) > budget:
                    continue
                U_now = U_trial
            rotated = True
            for pp, qq in moves:
                for M in (A, Q[None]):
                    cp = M[:, :, pp].copy()
                    cq = M[:, :, qq].copy()
                    M[:, :, pp] = c * cp + sn * cq
                    M[:, :, qq] = -np.conj(sn) * cp + c * cq
                rp = A[:, pp, :].copy()
                rq = A[:, qq, :].copy()
                A[:, pp, :] = c * rp + np.conj(sn) * rq
                A[:, qq, :] = -sn * rp + c * rq
        if not rotated:
            break
        current = off_weight()
        if previous - current <= _JD_REL_DECREASE * previous:
            break
        previous = current
    return Q


def block_diagonalize_hyb(
    hyb,
    mat=None,
    w=None,
    x_lim=None,
    *,
    u4=None,
    to_spherical=None,
    max_u_densify=2.0,
    tol=1e-6,
    info=None,
    custom_shapes=None,
):
    """Rotate the hybridization function (and optionally a matrix) to a near-block-diagonal basis.

    The unitary ``Q`` jointly and approximately block-diagonalizes and diagonalizes the Hermitian and
    anti-Hermitian parts of the hybridization function in the fit window together with ``mat``
    (typically the local Hamiltonian), keeping the rotated Coulomb tensor ``u4`` within ``max_u_densify``.
    """
    n_freq, n_orb, _ = hyb.shape
    idx = np.arange(n_freq)
    full_window = idx
    if x_lim is not None:
        assert w is not None, "x_lim needs the frequency mesh w"
        in_window = np.flatnonzero((w >= x_lim[0]) & (w <= x_lim[1]))
        if len(in_window) > 0:
            idx = full_window = in_window
    if not (np.all(np.isfinite(hyb[full_window])) and (mat is None or np.all(np.isfinite(mat)))):
        raise ValueError("block_diagonalize_hyb: the hybridization and the local Hamiltonian must be finite")
    if u4 is not None and not np.all(np.isfinite(u4)):
        raise ValueError("block_diagonalize_hyb: the Coulomb tensor must be finite")
    if len(idx) > _JD_MAX_FREQ:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, _JD_MAX_FREQ).astype(int))]
    h = hyb[idx]
    herm = 0.5 * (h + np.conj(np.transpose(h, (0, 2, 1))))
    anti = (h - np.conj(np.transpose(h, (0, 2, 1)))) / 2j
    scale = max(
        float(np.max(np.sqrt(np.mean(np.abs(np.einsum("wii->wi", h)) ** 2, axis=0)))),
        1e-10,
    )
    mats = [herm / scale, anti / scale]
    mats = np.concatenate(mats, axis=0)
    if mat is not None:
        mat = 0.5 * (mat + np.conj(mat.T))
        ev = np.linalg.eigvalsh(mat)
        mat_scale = max(float(ev[-1] - ev[0]), _JD_MAT_FLOOR)
        weight = _JD_MAT_WEIGHT * np.sqrt(len(mats))
        mats = np.concatenate([mats, (weight * mat / mat_scale)[None]], axis=0)
    if u4 is not None and np.isfinite(max_u_densify):
        window = hyb[full_window]
        Q = _choose_basis(
            mats,
            window,
            mat,
            u4,
            to_spherical,
            max_u_densify,
            tol,
            info if info is not None else {},
            custom_shapes=custom_shapes,
        )
    else:
        Q = _joint_diagonalize(mats)

    overlap = np.round(np.abs(Q) ** 2, 6)
    rows, cols = linear_sum_assignment(-overlap)
    Q = Q[:, cols[np.argsort(rows)]]
    Q_full = np.zeros_like(Q)
    for column in range(n_orb):
        j = np.argmax(np.round(np.abs(Q[:, column]), 6))
        Q_full[:, column] = Q[:, column] * abs(Q[j, column]) / Q[j, column]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full

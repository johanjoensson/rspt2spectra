"""The fitting-basis search: which orbital basis makes the hybridization function and H most block diagonal.

The caller hands over the hybridization on the fit window, the local Hamiltonian and (optionally) the
Coulomb tensor; :func:`block_diagonalize_hyb` returns the basis, chosen among symmetry-adapted
candidates (:func:`_tier1_candidates`) and a joint Jacobi diagonalisation, such that the block partition at the
requested tolerance is as fine as possible without making the Coulomb tensor much denser.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize

from .utils import _rotate_u4, count_u_terms


def _angular_momentum_ops(l):
    """``(Lx, Ly, Lz)`` for angular momentum ``l`` in complex harmonics, ``m = -l..l``."""
    m = np.arange(-l, l + 1)
    n = 2 * l + 1
    Lz = np.diag(m).astype(complex)
    Lp = np.zeros((n, n), dtype=complex)
    for i, mm in enumerate(m[:-1]):
        Lp[i + 1, i] = np.sqrt(l * (l + 1) - mm * (mm + 1))
    Lm = Lp.conj().T
    return (Lp + Lm) / 2, (Lp - Lm) / 2j, Lz


def _shell_l(to_spherical):
    """The ``l`` of a single, complete, spin-resolved shell described by ``to_spherical``, else ``None``.

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
    """The ``2(2l+1)`` dimensional rotation ``D_spin (x) D_l`` on [spin down, spin up] x m blocks."""
    Lx, Ly, Lz = _angular_momentum_ops(l)
    D = expm(-1j * (theta_orb[0] * Lx + theta_orb[1] * Ly + theta_orb[2] * Lz))
    if theta_spin is None:
        return np.kron(np.eye(2), D)
    # Pauli matrices in the [down, up] ordering
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    sz = np.array([[-1, 0], [0, 1]], dtype=complex)
    Ds = expm(-0.5j * (theta_spin[0] * sx + theta_spin[1] * sy + theta_spin[2] * sz))
    return np.kron(Ds, D)


def _so3_start_angles():
    """Deterministic starting rotation vectors; the identity comes first."""
    quarter, half = np.pi / 4, np.pi / 2
    return [
        np.zeros(3),
        np.array([half, 0, 0]),
        np.array([0, half, 0]),
        np.array([0, 0, half]),
        np.array([0, 0, quarter]),
        np.array([0, quarter, 0]),
        np.array([0.6, 0.6, 0.6]),
    ]


_AXIS_SEED = 20261112
_AXIS_N_SAMPLES = 600
_AXIS_N_REFINE = 6


def _off_diagonal_weight(mats, Q):
    R = np.conj(Q.T)[None] @ mats @ Q[None]
    return float(np.sum(np.abs(R) ** 2) - np.sum(np.abs(np.einsum("mii->mi", R)) ** 2))


def _real_harmonics(l):
    """Unitary from complex (``m = -l..l``) to real harmonics: columns ``m = -l..-1, 0, 1..l`` (sin, 1, cos)."""
    s = 1 / np.sqrt(2)
    n = 2 * l + 1
    V = np.zeros((n, n), dtype=complex)
    V[l, l] = 1
    for m in range(1, l + 1):
        a, b = l - m, l + m
        V[[a, b], a] = [1j * s, -1j * s * (-1) ** m]
        V[[a, b], b] = [s, s * (-1) ** m]
    return V


def _cubic_f_harmonics():
    """Cubic harmonics of an f shell (a2u, t1u, t2u): real harmonics diagonalising the cubic field.

    The cubic crystal field is ``O40 + 5 O44`` (Stevens operators).  It is diagonalised inside each
    set of real harmonics it couples (the parity classes of the cube's mirror planes), which
    leaves the degenerate t levels in a fixed, symmetric choice of orbitals.
    """
    from scipy.sparse.csgraph import connected_components

    l = 3
    Lx, Ly, Lz = _angular_momentum_ops(l)
    Lp, Lm = Lx + 1j * Ly, Lx - 1j * Ly
    L2 = l * (l + 1)
    Lz2 = Lz @ Lz
    O40 = 35 * Lz2 @ Lz2 - (30 * L2 - 25) * Lz2 + (3 * L2**2 - 6 * L2) * np.eye(2 * l + 1)
    O44 = (np.linalg.matrix_power(Lp, 4) + np.linalg.matrix_power(Lm, 4)) / 2
    T = _real_harmonics(l)
    M = np.conj(T.T) @ (O40 + 5 * O44) @ T
    M = 0.5 * (M + np.conj(M.T))
    n_classes, labels = connected_components(np.abs(M) > 1e-9, directed=False)
    V = np.zeros_like(M)
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        _, vecs = np.linalg.eigh(M[np.ix_(idx, idx)])
        V[np.ix_(idx, idx)] = vecs
    return T @ V


def _coupled_j_basis(l):
    """``|j, m_j>`` states of a shell, in the [spin down, spin up] x ``m`` ordering (columns by ``L.S`` then ``J_z``)."""
    Lops = _angular_momentum_ops(l)
    n = 2 * l + 1
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, 1j], [-1j, 0]]) / 2
    sz = np.diag([-0.5, 0.5]).astype(complex)
    LS = sum(np.kron(sp, Lc) for sp, Lc in zip((sx, sy, sz), Lops))
    Jz = np.kron(sz, np.eye(n)) + np.kron(np.eye(2), Lops[2])
    _, vecs = np.linalg.eigh(LS + 1e-3 * Jz)
    return vecs


def _components(mask):
    from scipy.sparse.csgraph import connected_components

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


def _coupling_matrix(Q, hyb, mat, tol):
    from rspt2spectra.block_structure import relative_couplings

    G = np.conj(Q.T)[None] @ hyb @ Q[None]
    H = None if mat is None else np.conj(Q.T) @ mat @ Q
    return relative_couplings(G, H, tol)


def _quality(c, tol):
    """``(partition key, largest coupling between blocks)`` of a coupling matrix at ``tol``."""
    _, labels = _components(c > tol)
    sizes = np.bincount(labels)
    between = c[labels[:, None] != labels[None, :]]
    return _partition_key(sizes), (float(between.max()) if between.size else 0.0)


def _margin_better(new, old, tol):
    """Whether a coupling left between blocks is clearly smaller; couplings far below ``tol`` do not matter."""
    return old > 0.1 * tol and new < 0.95 * old


def _basis_quality(Q, hyb, mat, tol):
    return _quality(_coupling_matrix(Q, hyb, mat, tol), tol)


def _frame_starts(n_par):
    """Deterministic frames to start from: identity, structured ones, and a seeded random cloud.

    A fixed seed makes every rank and every DMFT iteration scan the same cloud, so the choice only
    changes when the data does.
    """
    rng = np.random.default_rng(_AXIS_SEED)
    starts = [np.zeros(n_par)]
    starts += [np.concatenate([a, np.zeros(n_par - 3)]) for a in _so3_start_angles()[1:]]
    starts += [rng.uniform(-np.pi, np.pi, size=n_par) for _ in range(_AXIS_N_SAMPLES)]
    return starts


def _weight_minimising_frame(cost, starts, improvement=1e-9):
    """Frame minimising a smooth cost: best few scan points refined with BFGS, a new minimum wins by a margin."""
    scores = np.array([cost(t) for t in starts])
    best_theta, best_cost = None, np.inf
    for k in np.argsort(scores, kind="stable")[:_AXIS_N_REFINE]:
        res = minimize(cost, starts[k], method="BFGS", options={"gtol": 1e-10, "maxiter": 300})
        if res.fun < best_cost - improvement:
            best_theta, best_cost = res.x, res.fun
    return best_theta


def _partition_frames(coupling, starts, tol):
    """Frames that give the finest block partition at ``tol``.

    Scores the scan by partition (and by how far the next finer one is), then keeps asking for blocks
    one orbital smaller: Nelder--Mead on the threshold at which that is reached, which is zero-gradient
    almost everywhere for a plain weight but a clean piecewise-smooth target.  Returns the frames in the
    order found (best scan point first, finest partition last).
    """
    scan = []
    for t in starts:
        c = coupling(t)
        key, _ = _quality(c, tol)
        scan.append((key, _block_threshold(c, max(key[0] - 1, 1))))
    order = sorted(range(len(starts)), key=lambda i: scan[i])
    best = starts[order[0]]
    frames = [best]
    largest = scan[order[0]][0][0]
    for _ in range(4):
        if largest <= 1:
            break
        target = largest - 1
        improved = False
        for p in [best] + [starts[i] for i in order[:_PARTITION_N_REFINE]]:
            res = minimize(
                lambda t: _block_threshold(coupling(t), target),
                p,
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": _PARTITION_MAXITER},
            )
            if res.fun <= tol:
                best, improved = res.x, True
                largest = _quality(coupling(best), tol)[0][0]
                frames.append(best)
                break
        if not improved:
            break
    return frames


_PARTITION_N_REFINE = 3
_PARTITION_MAXITER = 400


def _tier1_candidates(mats, hyb, mat, u4, to_spherical, budget, tol):
    """Rigid-rotation bases of the correlated shell: ``[(label, Q)]``, one per orbital shape.

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
    * ``|j, m_j>`` (spin-coupled matrices only): strong spin-orbit coupling.

    Each shape's frame is chosen to give the finest block partition at ``tol`` among the frame that
    minimises the off-diagonal weight, the identity frame, and frames found by searching the partition
    directly; ties go to the smaller coupling between blocks.

    Returns ``[]`` when there is no complete single shell, or when ``u4`` is not invariant under the
    rotations (several shells, a different ``l``, a Coulomb tensor without full rotational symmetry);
    the caller then falls back to the free rotation.
    """
    l = _shell_l(to_spherical)
    if l is None:
        return []
    S = np.asarray(to_spherical, dtype=complex)
    n = S.shape[0]
    half = n // 2

    def frame(theta, tied=False):
        if tied:
            return _so3_rotation(l, theta[:3], theta[:3])
        return _so3_rotation(l, theta[:3], theta[3:] if len(theta) == 6 else None)

    # The interaction must survive a generic rotation of both frames, else this is not the shell
    # we think it is (or U is not rotationally invariant) and the premise of the tier is gone.
    # Only the operator has to be invariant: elements that vanish by the Pauli principle (c_i^+ c_i^+ = 0)
    # may be absent from the tensor, so it is antisymmetrized in each index pair before the test.
    u_op = u4 - u4.transpose(1, 0, 2, 3) - u4.transpose(0, 1, 3, 2) + u4.transpose(1, 0, 3, 2)
    scale = max(float(np.max(np.abs(u_op))), 1e-300)
    # two probes about non-parallel axes: one rotation only generates a one-parameter subgroup
    for angles in ([0.7, -1.1, 0.4, -0.5, 0.9, 0.3], [-0.3, 0.2, 1.2, 0.8, -0.6, 1.0]):
        probe = S @ frame(np.array(angles)) @ np.conj(S.T)
        if np.max(np.abs(_rotate_u4(u_op, probe) - u_op)) > 1e-8 * scale:
            return []

    total = _off_diagonal_weight(mats, np.eye(n, dtype=complex))
    if total <= 0:
        return [("input orbitals", np.eye(n, dtype=complex))]
    # spin-off-diagonal weight in the spherical frame, where the spin blocks are the two halves
    mats_sph = np.conj(S.T)[None] @ mats @ S[None]  # S is <input|spherical>
    spin_coupled = float(np.sum(np.abs(mats_sph[:, :half, half:]) ** 2)) > 1e-12 * float(np.sum(np.abs(mats) ** 2))

    I2 = np.eye(2)
    shapes = [("input orbitals", np.conj(S.T), False), ("complex harmonics", np.eye(n, dtype=complex), False)]
    if l >= 1:
        shapes.append(("real harmonics", np.kron(I2, _real_harmonics(l)), False))
    if l == 3:
        shapes.append(("cubic f harmonics", np.kron(I2, _cubic_f_harmonics()), False))
    if spin_coupled:
        shapes.append(("j-j coupled", _coupled_j_basis(l), True))

    # sub-sampled frequencies for the search; the shapes are ranked on the whole window
    sub = np.unique(np.linspace(0, len(hyb) - 1, min(len(hyb), _JD_MAX_FREQ)).astype(int))
    g_sub = hyb[sub]
    starts = {3: _frame_starts(3), 6: _frame_starts(6)}

    out = []
    for label, W, tied in shapes:
        if count_u_terms(_rotate_u4(u4, S @ W)) > budget:
            continue
        n_par = 3 if (tied or not spin_coupled) else 6

        def make_q(theta, W=W, tied=tied):
            return S @ frame(theta, tied) @ W

        def cost(theta):
            return _off_diagonal_weight(mats, make_q(theta)) / total

        def coupling(theta):
            return _coupling_matrix(make_q(theta), g_sub, mat, tol)

        thetas = [_weight_minimising_frame(cost, starts[n_par]), np.zeros(n_par)]
        thetas += _partition_frames(coupling, starts[n_par], tol)
        best, best_q = None, None
        for theta in thetas:
            if theta is None:
                continue
            Q = make_q(theta)
            key, margin = _basis_quality(Q, hyb, mat, tol)
            if best is None or key < best[0] or (key == best[0] and _margin_better(margin, best[1], tol)):
                best, best_q = (key, margin), Q
        out.append((label, best_q))
    return out


def _choose_basis(mats, hyb_window, mat, u4, to_spherical, max_u_densify, tol, info):
    """Pick the unitary that blocks best at ``tol`` among the bases whose Coulomb tensor stays within budget.

    Candidates are the rigid-rotation bases of :func:`_tier1_candidates`, the free joint rotation, and,
    when that is too dense, the free rotation restricted to the budget and started from the best rigid
    one.  The best partition wins; among equal partitions the sparser Coulomb tensor, then the smaller
    coupling left between blocks, then the order above.
    """
    n = mats.shape[-1]
    n_ref = count_u_terms(u4)
    if not np.isfinite(max_u_densify) or max_u_densify * n_ref >= n**4:
        budget = n**4  # a tensor cannot hold more elements than that
    else:
        # the epsilon keeps floor(1.13 * 100) at 113 rather than 112
        budget = max(int(np.floor(max_u_densify * n_ref * (1 + 1e-12) + 1e-9)), n_ref)  # input basis always available
    info["u_terms_input"] = n_ref

    best = {}

    def consider(tier, label, Q):
        terms = count_u_terms(_rotate_u4(u4, Q))
        if terms > budget:
            return False
        key, margin = _basis_quality(Q, hyb_window, mat, tol)
        if best:
            if key != best["key"]:
                if key > best["key"]:
                    return False
            elif terms != best["terms"]:
                if terms > best["terms"]:
                    return False
            elif not _margin_better(margin, best["margin"], tol):
                return False
        best.update(tier=tier, label=label, Q=Q, key=key, terms=terms, margin=margin)
        return True

    for label, Q in _tier1_candidates(mats, hyb_window, mat, u4, to_spherical, budget, tol):
        consider("axis rotation", label, Q)
    start = best["Q"] if best else np.eye(n, dtype=complex)
    Q_free = _joint_diagonalize(mats)
    if not consider("free rotation", "", Q_free):
        # also when the free rotation fits but blocks worse: the budgeted search refines the best rigid basis
        consider("budgeted free rotation", "", _joint_diagonalize(mats, Q0=start, u4=u4, budget=budget))
    if not best:
        consider("input basis", "", np.eye(n, dtype=complex))
    info.update(tier=best["tier"], shape=best["label"], u_terms=best["terms"])
    return best["Q"]


_JD_MAX_FREQ = 50
_JD_MAX_SWEEPS = 100
_JD_SKIP = 1e-8  # relative to the O(1) scaled matrices; couplings below are numerical noise
_JD_REL_DECREASE = 1e-10  # stop when a sweep removes less than this fraction of the off-diagonal weight
_JD_MAT_WEIGHT = 0.1  # weight of H relative to the whole hyb set: H breaks hyb degeneracies, hyb sets the basis
_JD_MAT_FLOOR = 1e-3  # Ry; crystal-field spread below this is noise when weighting H against hyb


def _joint_diagonalize(mats, max_sweeps=_JD_MAX_SWEEPS, skip=_JD_SKIP, Q0=None, u4=None, budget=None):
    """Approximately diagonalize a set of Hermitian matrices with one unitary.

    Complex Jacobi iteration after Cardoso and Souloumiac: every sweep rotates each
    orbital pair by the angle that minimises the summed squared off-diagonal weight of
    all matrices.  The start is the identity and every rotation is non-increasing, so
    the result is never less diagonal than the input.  A pair whose couplings vanish in
    every matrix is skipped, which keeps decoupled orbitals (spin up/down, separate
    blocks) from being mixed.

    Parameters
    ----------
    mats : ndarray of shape (n_mat, n, n)
        Hermitian matrices, already scaled to be of order one.
    max_sweeps : int
        Maximum number of Jacobi sweeps.
    skip : float
        Pairs with total coupling weight below this are left alone.
    Q0 : ndarray of shape (n, n), optional
        Start from this unitary instead of the identity.
    u4, budget : ndarray, int, optional
        With both given, a pair rotation is only applied if the number of non-negligible
        elements of ``u4`` rotated by the accumulated ``Q`` stays within ``budget``; the
        pairs are then visited in order of decreasing coupling, so the budget goes to the
        rotations that remove the most off-diagonal weight.  Without them the sweep is the
        plain, unconstrained one.

    Returns
    -------
    Q : ndarray of shape (n, n)
        Unitary with ``Q^dagger mats[m] Q`` as diagonal as possible.
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

    # A spin-degenerate problem (identical spin blocks, no spin coupling) must keep identical rotations
    # for both spins: each move is then applied to both blocks at once and checked against the budget once.
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
            # h[2] = i(a_qp - a_pq) = 2 Im(a_pq) for Hermitian a_qp = conj(a_pq)
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
            # A <- J^H A J and Q <- Q J, with J = [[c, -conj(sn)], [sn, c]] on (p, q)
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
    hyb, mat=None, w=None, x_lim=None, *, u4=None, to_spherical=None, max_u_densify=2.0, tol=1e-6, info=None
):
    """Rotate the hybridization function (and optionally a matrix) to a near-diagonal basis.

    The unitary ``Q`` jointly and approximately diagonalizes the Hermitian and
    anti-Hermitian parts of the hybridization function at (up to 50) frequencies in the
    fit window together with ``mat`` (typically the local Hamiltonian).  Off-diagonal
    couplings that no common rotation can remove stay in the result; they are small
    when the system is close to symmetric and are what the block threshold then
    decides about.

    Parameters
    ----------
    hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The hybridization matrix.
    mat : ndarray of shape (n_orb, n_orb), optional
        Matrix to diagonalize together with ``hyb``, e.g. the local Hamiltonian.
    w : ndarray of shape (n_freq,), optional
        Frequency mesh of ``hyb``; needed for ``x_lim``.
    x_lim : tuple of float, optional
        ``(lo, hi)`` window; only frequencies inside it are used.
    u4 : ndarray of shape (n_orb,) * 4, optional
        The Coulomb tensor in the input basis.  When given (and ``max_u_densify`` is finite) the
        basis is chosen with its sparsity in mind, in this order:

        1. a rigid rotation of the shell (SO(3)), which cannot densify a rotationally invariant
           ``u4``, if it separates the orbitals into blocks as well as a free rotation does;
        2. the free joint rotation, if the rotated ``u4`` has at most ``max_u_densify`` times
           as many non-negligible elements as ``u4``;
        3. otherwise the free rotation again, but only as far as that budget allows.

        Without ``u4`` the free rotation is used unconstrained.
    to_spherical : ndarray of shape (n_orb, n_sph), optional
        Map from the input orbitals to complex spherical harmonics; needed for step 1, which is
        skipped for partial or multi-shell sets.
    max_u_densify : float, default 2.0
        Allowed growth of the number of Coulomb-tensor terms; ``inf`` disables the constraint.
    tol : float, default 1e-6
        The ``symm_prec`` threshold, used to compare how well bases separate the orbitals.
    info : dict, optional
        Filled with ``tier``, ``u_terms_input`` and ``u_terms`` when ``u4`` is used.

    Returns
    -------
    phase_hyb : ndarray of shape (n_freq, n_orb, n_orb)
        The hybridization function in the rotated basis.
    Q_full : ndarray of shape (n_orb, n_orb)
        The transformation matrix.  Column ``j`` is the orbital that overlaps most with
        the input orbital ``j``, so orbital labels are stable between calls.
    """
    n_freq, n_orb, _ = hyb.shape
    idx = np.arange(n_freq)
    full_window = idx
    if x_lim is not None:
        assert w is not None, "x_lim needs the frequency mesh w"
        in_window = np.flatnonzero((w >= x_lim[0]) & (w <= x_lim[1]))
        if len(in_window) > 0:
            idx = full_window = in_window
    if len(idx) > _JD_MAX_FREQ:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, _JD_MAX_FREQ).astype(int))]
    h = hyb[idx]
    if not (np.all(np.isfinite(h)) and (mat is None or np.all(np.isfinite(mat)))):
        raise ValueError("block_diagonalize_hyb: the hybridization and the local Hamiltonian must be finite")
    if u4 is not None and not np.all(np.isfinite(u4)):
        raise ValueError("block_diagonalize_hyb: the Coulomb tensor must be finite")
    herm = 0.5 * (h + np.conj(np.transpose(h, (0, 2, 1))))
    anti = (h - np.conj(np.transpose(h, (0, 2, 1)))) / 2j
    scale = max(float(np.max(np.sqrt(np.mean(np.abs(np.einsum("wii->wi", h)) ** 2, axis=0)))), 1e-10)
    mats = [herm / scale, anti / scale]
    mats = np.concatenate(mats, axis=0)
    if mat is not None:
        mat = 0.5 * (mat + np.conj(mat.T))
        ev = np.linalg.eigvalsh(mat)
        mat_scale = max(float(ev[-1] - ev[0]), _JD_MAT_FLOOR)
        # H gets a tenth of the weight of the whole set of frequencies: it breaks degeneracies of the
        # hybridization and tie-breaks the basis, but must not drag it away from the hybridization's.
        # Weight 0 would leave the rotation arbitrary wherever the hybridization is degenerate.
        weight = _JD_MAT_WEIGHT * np.sqrt(len(mats))
        mats = np.concatenate([mats, (weight * mat / mat_scale)[None]], axis=0)
    if u4 is not None and np.isfinite(max_u_densify):
        window = hyb[full_window]
        Q = _choose_basis(mats, window, mat, u4, to_spherical, max_u_densify, tol, info if info is not None else {})
    else:
        Q = _joint_diagonalize(mats)

    # Label stability: column j is the rotated orbital closest to input orbital j.
    # Overlaps are rounded so that round-off cannot decide a tie (a 45 degree mixing of two
    # degenerate orbitals has equal overlaps); ties then resolve by orbital index, always.
    overlap = np.round(np.abs(Q) ** 2, 6)
    rows, cols = linear_sum_assignment(-overlap)
    Q = Q[:, cols[np.argsort(rows)]]
    Q_full = np.zeros_like(Q)
    for column in range(n_orb):
        j = np.argmax(np.round(np.abs(Q[:, column]), 6))
        Q_full[:, column] = Q[:, column] * abs(Q[j, column]) / Q[j, column]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full

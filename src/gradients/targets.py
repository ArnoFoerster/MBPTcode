"""Target-energy partials (dE_X/dF_pq, dE_X/d(pq|rs)) for the gradient engine.

Each target returns (gammaF, Gamma4) as raw coefficient tensors in the
conventions of grad_engine (no symmetrization; the engine's FD gates define
correctness). Index map for boson pairs: I = i * nvirt + (a - nocc).
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def _pair_views(mat, nocc, norb):
    """(n_ov, n_ov) -> (i, a, j, b) view with a, b as virtual offsets."""
    nvirt = norb - nocc
    return mat.reshape(nocc, nvirt, nocc, nvirt)


def rpa_partials(qb, nocc, norb):
    """Partials of E_c^dRPA at fixed t (t-variational, no Z needed).

    E_c = Tr[A G^A] + Tr[B G^B],  G^A = sinh^2 t, G^B = sinh t cosh t;
    A_IJ = d_ij F_ab - d_ab F_ij + 2(ia|jb),  B_IJ = 2(ia|jb).
    """
    nvirt = norb - nocc
    GA = _pair_views(qb.gamma_A(), nocc, norb)
    GB = _pair_views(qb.gamma_B(), nocc, norb)

    gammaF = np.zeros((norb, norb))
    # dA/dF_ab = d_ij: sum over i of the (ia),(ib) diagonal-in-i block
    gammaF[nocc:, nocc:] += np.einsum('iaib->ab', GA)
    # dA/dF_ij = -d_ab
    gammaF[:nocc, :nocc] -= np.einsum('iaja->ij', GA)

    Gamma4 = np.zeros((norb,) * 4)
    Gamma4[:nocc, nocc:, :nocc, nocc:] += 2.0 * (GA + GB)
    return gammaF, Gamma4


def chain_AB(XA, XB, nocc, norb, gammaF, Gamma4):
    """Accumulate E += sum XA_IJ A_IJ + XB_IJ B_IJ into (F, ERI) partials.

    A_IJ = d_ij F_ab - d_ab F_ij + 2(ia|jb),  B_IJ = 2(ia|jb).
    """
    nvirt = norb - nocc
    XA4 = XA.reshape(nocc, nvirt, nocc, nvirt)
    XB4 = XB.reshape(nocc, nvirt, nocc, nvirt)
    gammaF[nocc:, nocc:] += np.einsum('iaib->ab', XA4)
    gammaF[:nocc, :nocc] -= np.einsum('iaja->ij', XA4)
    Gamma4[:nocc, nocc:, :nocc, nocc:] += 2.0 * (XA4 + XB4)


def qp_partials(qp, p, w):
    """Partials of the diagonal qb-EOM QP energy eps^QP_p at its root w.

    Returns (gammaF, Gamma4, t_grad); t_grad is the RHS driver of the Z
    equation (None for TDA screening, which has no t).
    Hellmann-Feynman on eps = R^T H R with the closed-form eigenvector.

    A static <p|Sigma_x - v_xc|p> shift is REFUSED here: it moves the root and
    is no function of (F, ERI), so a caller reading these three alone would
    return a force missing the whole of d(shift)/dR. A caller that assembles
    that derivative asks `qp_partials_with_shift`, which hands back its weight.
    """
    if qp.delta.any():
        raise NotImplementedError(
            'these partials carry the (F, ERI) dependence of the root and this '
            'one also carries a static <p|Sigma_x - v_xc|p> shift, whose '
            'nuclear derivative is a term of its own: the shift weight is the '
            'quasiparticle weight rp^2 at that root. A force built from these '
            'three alone would be missing it. Ask `qp_partials_with_shift`, '
            'which returns that weight, and assemble it.')
    return qp_partials_with_shift(qp, p, w)[:3]


def qp_partials_with_shift(qp, p, w):
    """(gammaF, Gamma4, t_grad, weight) for a caller that differentiates the
    static shift itself.

    <p|Sigma_x - v_xc|p> sits in the (p, p) element of the upfolded matrix and
    nowhere else -- the 2h1p/2p1h poles are built from the reference's own
    eigenvalues and carry no shift -- so d eps^QP_p / d delta_p is the
    quasiparticle weight rp^2 of the root, the coefficient eps_p itself enters
    with. It is returned rather than folded in because the shift is not an
    (F, ERI) partial: a caller that differentiates it too must fold that term
    in separately, and a caller that does not must use `qp_partials`, which
    refuses.
    """
    nocc, norb = qp.nocc, qp.norb
    nvirt = norb - nocc
    qb = qp.qb
    rp, Rh_nu, Rp_nu = qp.eigvec_diag(p, w)
    U = qp.U
    Rh = Rh_nu @ U.T                     # (j, I) boson-original basis
    Rp = Rp_nu @ U.T                     # (b, I)

    gammaF = np.zeros((norb, norb))
    Gamma4 = np.zeros((norb,) * 4)

    gammaF[p, p] += rp ** 2
    gammaF[:nocc, :nocc] += Rh @ Rh.T
    gammaF[nocc:, nocc:] += Rp @ Rp.T

    # Abar block: T_IJ = -sum_j R_jI R_jJ + sum_b R_bI R_bJ
    T = -Rh.T @ Rh + Rp.T @ Rp
    XA, XB = qb.abar_AB_adjoint(T)
    chain_AB(XA, XB, nocc, norb, gammaF, Gamma4)

    # couplings: deps/dW^J_pj = 2 rp R_jJ, deps/dW^J_pb = 2 rp R_bJ,
    # W^J_pq = sqrt(2) sum_I V^I_pq [e^t]_IJ, V^I_pq = (pq|ia)
    Ch = 2.0 * rp * Rh                   # (j, J)
    Cp = 2.0 * rp * Rp                   # (b, J)
    dV_h = np.sqrt(2.0) * Ch @ qb.exp_t  # (j, I)
    dV_p = np.sqrt(2.0) * Cp @ qb.exp_t  # (b, I)
    Gamma4[p, :nocc, :nocc, nocc:] += dV_h.reshape(nocc, nocc, nvirt)
    Gamma4[p, nocc:, :nocc, nocc:] += dV_p.reshape(nvirt, nocc, nvirt)

    if qp.screening == 'tda':
        return gammaF, Gamma4, None, rp ** 2

    # t-gradient: Abar part + coupling part (Tr[e^t K^T])
    t_grad = qb.abar_t_gradient(T)
    Vb = qp.Vbare                        # (norb, norb, n_ov)
    K = np.sqrt(2.0) * (np.einsum('jI,jJ->IJ', Vb[p, :nocc, :], Ch, optimize=True)
                        + np.einsum('bI,bJ->IJ', Vb[p, nocc:, :], Cp, optimize=True))
    t_grad += qb.linear_expt_t_gradient(K)
    return gammaF, Gamma4, t_grad, rp ** 2


def solve_Z(qb, t_grad, tol=1e-10, verbose=False):
    """Solve t_grad + dJ/dt-adjoint(Z) = 0 for symmetric Z, in the t eigenbasis.

    dJ/dt-adjoint(Z) = L_{e^t}(sym(P e^t Z)) - L_{e^-t}(sym(M e^-t Z)) costs six
    (n_ov)^3 products as written. Rotate to the eigenbasis of t, where the two
    Frechet maps L_{e^{+-t}} are Hadamard products with the divided-difference
    matrices Phi^{+-}, and where the amplitude equation e^t P e^t = e^-t M e^-t
    makes the two operands row scalings of ONE matrix:

        Qhat_1 = (Phat E) Zhat,   Qhat_2 = E^2 Qhat_1,   E = diag(e^w)

    One product per iteration instead of six, and the diagonal of the map is
    then available in closed form as a preconditioner. The map is the Hessian
    of E_c with respect to t, so it is symmetric, and positive definite at a
    stable RPA solution -- conjugate gradients, which carries four vectors,
    rather than a restarted Krylov method carrying thirty of length (n_ov)^2.
    """
    n = t_grad.shape[0]
    w, V = qb.wt, qb.Vt
    e = np.exp(w)
    dw = w[:, None] - w[None, :]
    scale = np.maximum(np.abs(w[:, None]), np.abs(w[None, :])) + 1.0
    small = np.abs(dw) < 1e-12 * scale
    den = np.where(small, 1.0, dw)
    mid = 0.5 * (w[:, None] + w[None, :])
    phi_p = np.where(small, np.exp(mid), (e[:, None] - e[None, :]) / den)
    phi_m = np.where(small, -np.exp(-mid),
                     (1.0 / e[:, None] - 1.0 / e[None, :]) / den)
    e2 = e ** 2

    PE = (V.T @ (qb.A + qb.B) @ V) * e[None, :]
    diag_map = 0.5 * (phi_p * (np.diag(PE)[:, None] + np.diag(PE)[None, :])
                      - phi_m * (e2[:, None] * np.diag(PE)[:, None]
                                 + e2[None, :] * np.diag(PE)[None, :]))
    guard = np.where(np.abs(diag_map) < 1e-10, 1.0, diag_map)

    def mv(v):
        Y = PE @ v.reshape(n, n)
        S1 = 0.5 * (Y + Y.T)
        Y2 = e2[:, None] * Y
        S2 = 0.5 * (Y2 + Y2.T)
        return (phi_p * S1 - phi_m * S2).ravel()

    def prec(v):
        return (v.reshape(n, n) / guard).ravel()

    rhs = -(V.T @ (0.5 * (t_grad + t_grad.T)) @ V).ravel()
    op = LinearOperator((n * n, n * n), matvec=mv)
    Mop = LinearOperator((n * n, n * n), matvec=prec)
    z, info = cg(op, rhs, M=Mop, rtol=0.0,
                 atol=tol * max(1.0, np.linalg.norm(rhs)), maxiter=4000)
    res = np.linalg.norm(mv(z) - rhs)
    if info != 0 or res > 1e-6 * max(1.0, np.linalg.norm(rhs)):
        raise RuntimeError(f"Z solve failed: info={info}, res={res:.3e}")
    Z = V @ (0.5 * (z.reshape(n, n) + z.reshape(n, n).T)) @ V.T
    if verbose:
        print(f"    [Z] res={res:.3e} |Z|={np.abs(Z).max():.3e}")
    return 0.5 * (Z + Z.T)


def add_z_contribution(qb, Z, nocc, norb, gammaF, Gamma4):
    """Accumulate the sum_IJ Z_IJ J_IJ term's (F, ERI) partials at fixed t."""
    ZA, ZB = qb.dJ_dAB_adjoint(Z)
    chain_AB(ZA, ZB, nocc, norb, gammaF, Gamma4)

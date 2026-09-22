"""Nuclear derivatives of the ISDF factors, as adjoint contractions.

The space-time route returns sensitivities with respect to the factors,
(eps_bar, X_bar, D_bar), and this module turns the X and D parts into forces.
The derivative tensor dX/dR is never formed, only its contraction against a
given adjoint.

X depends on the geometry three times over, through the fit and the
collocation, through the points translating with their atom, and through the
local frames turning with the environment. All three are needed to gate below
1e-4.

An axis of a frame carries a pure gauge sign, and no sign convention is
continuous everywhere. `continued_frames` therefore carries the convention over
from a reference and differentiates that.

Derivative-integral signs are measured, not transcribed. pyscf's int2c2e_ip1,
int3c2e_ip1 and int3c2e_ip2 are all +(grad .), so every nuclear derivative
built from them carries a minus.
"""
from dataclasses import dataclass

import numpy as np
from src.Base.constants import (ORBITAL_MULTIPLIER_MAX_ITER,
                                ORBITAL_MULTIPLIER_TOL,
                                THREE_CENTER_BLOCK_BYTES)
from src.Base.dispersion import refuse_dispersion_under_rpa
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import scf as pyscf_scf
from pyscf.grad import rhf as grad_rhf

from src.Base.environment import dresses_interaction
from src.Base.isdf_jk import ISDFJK, mean_field_skeleton_force, range_coulomb
from src.Base.pcm_derivatives import reaction_field_fock_skeleton
from src.Base.pyscf_interface import (aux_metric_inverse, fock_mo,
                                      response_kernel)
from src.Base.separable_ri import (DEFAULT_PAIR_TOL, DEFAULT_REGULARIZATION,
                                   _FRAME_DECAY, atomic_frames, atomic_points,
                                   aux_metric_sqrt, build_D_F, fit_M_stable,
                                   subshells, test_set_D, test_set_layout)
from src.gradients.multipliers import solve_orbital_multipliers
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.LinearResponse.rpa_energy import (
    exchange_channels, exx_double_counting, rsh_split, xc_hybrid_coeff)


def point_layout(mol, radii_by_element, origin_by_element=None):
    """(pts_local, atom_of_point): the atom-local clouds and their owners.

    The half of `separable_ri.molecular_points_covariant` that a derivative
    needs and that one does not return. Production stacks the placed points and
    keeps only the coordinates, while a chain rule on r_g = p_g F_i + R_i needs
    p_g and i separately. The two share the `atomic_points` call and the
    stacking order, which is what fixes the row order of X.
    """
    pts_local, owner = [], []
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        use_origin = (origin_by_element or {}).get(sym, False)
        p = atomic_points(radii_by_element[sym], centre=(0.0, 0.0, 0.0),
                          origin=use_origin)
        pts_local.append(p)
        owner.append(np.full(len(p), ia, dtype=int))
    return pts_local, np.concatenate(owner)


def continued_frames(mol, frames_ref, decay=_FRAME_DECAY):
    """`atomic_frames` with each axis matched to a reference frame.

    Axis k is the raw axis with the largest |overlap| with frames_ref[i, k],
    signed to make that overlap positive, so the convention varies continuously
    with the geometry instead of following `eigh`'s arbitrary output sign. A
    proper rotation is restored at the end, since matching may leave an odd
    number of flips.
    """
    raw = atomic_frames(mol, decay=decay)[0]
    out = np.empty_like(raw)
    for i in range(mol.natm):
        taken = []
        for k in range(3):
            ov = raw[i] @ frames_ref[i, k]
            cand = [j for j in range(3) if j not in taken]
            j = cand[int(np.argmax(np.abs(ov[cand])))]
            taken.append(j)
            out[i, k] = raw[i, j] * (1.0 if ov[j] >= 0 else -1.0)
        if np.linalg.det(out[i]) < 0:
            # the reference is a proper rotation, so an odd number of flips
            # means the least-determined axis should carry the parity
            ov = np.abs([out[i, k] @ frames_ref[i, k] for k in range(3)])
            out[i, int(np.argmin(ov))] *= -1.0
    return out


def frame_adjoint(mol, W, decay=_FRAME_DECAY, frames=None, degenerate=None):
    """(natm, 3) gradient of sum_i sum_ab W[i,a,b] F_i[a,b], F_i the atomic frame.

    F_i's rows are the eigenvectors of T_i = sum_j w_j dhat_j dhat_j^T, sorted
    by descending eigenvalue and sign-fixed. With ordering and signs frozen,
    Tbar_i is the standard non-degenerate eigenvector adjoint, symmetrized
    because dT is symmetric, and T_i's own chain to the coordinates closes it.
    """
    coords = np.asarray(mol.atom_coords())
    natm = len(coords)
    if frames is None or degenerate is None:
        f, d = atomic_frames(mol, decay=decay)
        frames = f if frames is None else frames
        degenerate = d if degenerate is None else degenerate
    grad = np.zeros((natm, 3))
    for i in range(natm):
        if not np.any(W[i]):
            continue
        if degenerate[i]:
            raise ValueError(
                f"atom {i} has a degenerate local frame: its axes are not a "
                f"differentiable function of the geometry, and the construction "
                f"takes a branch there. Displace the reference, or exclude the "
                f"atom's points from the adjoint.")
        d = coords - coords[i]
        r = np.linalg.norm(d, axis=1)
        keep = r > 1e-8
        idx = np.flatnonzero(keep)
        dj = d[keep] / r[keep, None]
        w = np.exp(-r[keep] / decay)
        T = np.einsum('j,ja,jb->ab', w, dj, dj)
        lam, V = np.linalg.eigh(T)
        # Match each frame axis to its eigenvector by overlap, not by
        # eigenvalue order. Continued frames may be ordered differently from
        # eigh's, and an order-based match would pair the wrong axes.
        vbar = np.zeros((3, 3))                       # vbar[a] for eigenvector a
        taken = []
        for k in range(3):
            ov = V.T @ frames[i, k]
            cand = [j for j in range(3) if j not in taken]
            a = cand[int(np.argmax(np.abs(ov[cand])))]
            taken.append(a)
            vbar[a] = np.sign(ov[a]) * W[i, k]
        Tbar = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                gap = lam[a] - lam[b]
                if abs(gap) < 1e-10:
                    raise ValueError(
                        f"atom {i}: eigenvalue gap {gap:.2e} is too small for a "
                        f"frame derivative; the construction is on a branch.")
                Tbar += ((vbar[a] @ V[:, b]) / gap) * np.outer(V[:, a], V[:, b])
        Tbar = 0.5 * (Tbar + Tbar.T)
        # T = sum_j w(r_j) dhat_j dhat_j^T,  d_j = R_j - R_i
        for n, j in enumerate(idx):
            dh, rj, wj = dj[n], r[j], w[n]
            q = Tbar @ dh
            g = (-wj / decay) * float(dh @ q) * dh + (2.0 * wj / rj) * (q - float(dh @ q) * dh)
            grad[j] += g
            grad[i] -= g
    return grad


def basis_centre_forces(basis_mol, coords, bar):
    """(centre gradient, dE/dr_g) for sum_{g,mu} bar[g,mu] chi_mu(r_g).

    Splits the two ways a collocation depends on the geometry, so that several
    bases on the same points can each contribute their own centre term while
    sharing one point chain.

    pyscf's GTOval_ip_sph is +d(chi)/dr, and chi_mu(r) depends on a nuclear
    coordinate only through its own centre, where the derivative is -d(chi)/dr.
    """
    g = basis_mol.eval_gto('GTOval_ip_sph', coords)        # (3, M, nbas_ao)
    centre = np.zeros((basis_mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(basis_mol.aoslice_by_atom()):
        centre[ia] -= np.einsum('xgm,gm->x', g[:, :, p0:p1], bar[:, p0:p1])
    return centre, np.einsum('xgm,gm->gx', g, bar)         # (natm,3), (M,3)


def point_chain(mol, P, pts_local, atom_of_point, frames=None,
                decay=_FRAME_DECAY, with_frames=True):
    """(natm, 3) from an adjoint P[g] = dE/dr_g on the interpolation points.

    The points translate with the atom that owns them and turn with that atom's
    frame: r_g = p_g F_i + R_i, so dE/dF_i[a,b] = sum_{g in i} p_g[a] P[g,b].
    """
    grad = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        grad[ia] += P[atom_of_point == ia].sum(axis=0)
    if with_frames:
        W = np.zeros((mol.natm, 3, 3))
        for ia in range(mol.natm):
            W[ia] = pts_local[ia].T @ P[atom_of_point == ia]
        grad += frame_adjoint(mol, W, decay=decay, frames=frames)
    return grad


def collocation_adjoint(mol, coords, Xao_bar, pts_local, atom_of_point,
                        frames=None, decay=_FRAME_DECAY, with_frames=True):
    """(natm, 3) gradient of sum_{g,mu} Xao_bar[g,mu] chi_mu(r_g).

    The orbital-basis case: AO centres plus the shared point chain.
    `Xao_bar` is the adjoint on the AO-basis collocation; from an adjoint on
    X_mo = X_ao C it is X_bar C^T.
    """
    centre, P = basis_centre_forces(mol, coords, Xao_bar)
    return centre + point_chain(mol, P, pts_local, atom_of_point, frames=frames,
                                decay=decay, with_frames=with_frames)


# ---------------------------------------------------------------------------
# the fit: adjoint of the regularized least-squares estimator
# ---------------------------------------------------------------------------

def fit_adjoint(D, F, M_bar, regularization=DEFAULT_REGULARIZATION):
    """(D_bar, F_bar) of sum_{beta k} M_bar[beta,k] M[beta,k], M from `fit_M`.

    `fit_M` row-balances D, forms a regularized Gram matrix and solves

        s_k = ||D[k,:]||,   d = 1/s,   Dt = d D
        G   = Dt Dt^T + reg I
        M   = (F Dt^T) G^-1 d

    so the reverse pass is that read backwards, with the inverse contributing
    Gbar = -G^-1 Ginv_bar G^-1. No dM/dR is ever formed, and G is the matrix the
    forward pass already factorized.

    Nothing here forms G^-1. G is floored by `regularization` and reaches a
    large condition number on a production basis, and the adjoint applies its
    inverse twice, so an explicit inverse loses most of double precision. G is
    symmetric positive definite, so one Cholesky factorization serves every
    solve. `fit_adjoint_conditioning` reports the amplification.
    """
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    s = np.where(s == 0.0, 1.0, s)
    d = 1.0 / s
    Dt = D * d[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    cho = scipy.linalg.cho_factor(G, lower=True)
    A = F @ Dt.T
    B = scipy.linalg.cho_solve(cho, A.T).T      # A G^-1, G symmetric

    B_bar = M_bar * d[None, :]
    d_bar = np.einsum('bk,bk->k', M_bar, B)
    A_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
    Y = scipy.linalg.cho_solve(cho, A.T @ B_bar)          # G^-1 K
    G_bar = -scipy.linalg.cho_solve(cho, Y.T).T           # -G^-1 K G^-1
    F_bar = A_bar @ Dt
    Dt_bar = A_bar.T @ F + (G_bar + G_bar.T) @ Dt

    D_bar = d[:, None] * Dt_bar
    d_bar = d_bar + np.einsum('kr,kr->k', Dt_bar, D)
    s_bar = -d_bar * d ** 2                    # d = 1/s
    D_bar += (s_bar / s)[:, None] * D          # s = ||D[k,:]||
    return D_bar, F_bar


def fit_adjoint_conditioning(D, regularization=DEFAULT_REGULARIZATION):
    """(cond(G), reg, amplification) for the Gram matrix the fit adjoint inverts.

    The adjoint's error scales as cond(G)^2 because G^-1 enters twice, so this
    is the number that says whether a converged-looking fit gradient can be
    trusted.
    """
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    s = np.where(s == 0.0, 1.0, s)
    Dt = D * (1.0 / s)[:, None]
    G = Dt @ Dt.T
    G[np.diag_indices_from(G)] += regularization
    w = np.linalg.eigvalsh(G)
    cond = float(w.max() / max(w.min(), 1e-300))
    return cond, regularization, cond ** 2


# ---------------------------------------------------------------------------
# the grid itself: the fit error differentiated in the interpolation points
# ---------------------------------------------------------------------------

def fit_error_coulomb_grad(mol, auxmol, coords, layout=None, F=None, V=None,
                           l_max_second=2, pair_tol=DEFAULT_PAIR_TOL,
                           regularization=DEFAULT_REGULARIZATION):
    """(E, dE/dr_g) for `separable_ri.fit_error_coulomb`, the grid's objective.

    E = ||M D - F||_V, and of the three only D knows where the points are.
    E^2 = Tr[R^T V R] with R = M D - F reverses as R_bar = 2 V R,
    M_bar = R_bar D^T, D_bar = M^T R_bar plus the fit's own adjoint, and

        dE/dr_g = (1/2E) sum_c D_bar[g,c] dD[g,c]/dr_g,

    one GTOval_ip_sph evaluation. Row g of D depends on r_g alone, so the points
    do not couple and the result is (n_k, 3), not a Jacobian.

    E comes from `fit_M_stable`, so value and gradient are the same realization
    of the estimator. `layout` freezes which AO pairs the test set holds, since
    screening is a discrete choice and a derivative is only defined at fixed
    column set. It has to travel with its F, whose columns are those same pairs.

    The fit adjoint applies G^-1 twice, so the accuracy here follows cond(G).
    See `fit_adjoint_conditioning`.
    """
    if (layout is None) != (F is None):
        raise ValueError('layout and F describe the same columns and must be '
                         'passed together; F built by screening at these '
                         'coordinates does not match a frozen layout.')
    if V is None:
        V = auxmol.intor('int2c2e', aosym='s1')
    if layout is None:
        layout = test_set_layout(mol, coords, l_max_second=l_max_second,
                                 pair_tol=pair_tol)
        F = build_D_F(mol, auxmol, coords, l_max_second=l_max_second,
                      pair_tol=pair_tol)[1]
    D = test_set_D(mol, auxmol, coords, layout)
    M = fit_M_stable(D, F, regularization)
    R = M @ D - F
    E = float(np.sqrt(np.einsum('br,bc,cr->', R, V, R)))

    R_bar = 2.0 * (V @ R)                                  # d(E^2)/dR
    D_bar = M.T @ R_bar                                    # R's explicit D
    D_bar += fit_adjoint(D, F, R_bar @ D.T, regularization)[0]   # and M's

    mu, nu, w = layout
    nao = mol.nao_nr()
    ao = mol.eval_gto('GTOval_sph', coords)
    g = mol.eval_gto('GTOval_ip_sph', coords)              # (3, nk, nao)
    gaux = auxmol.eval_gto('GTOval_ip_sph', coords)
    pair_bar = D_bar[:, :len(mu)] * w[None, :]

    # Fold the pair adjoint onto the AO index before meeting the gradients.
    # Each column contributes chi_mu' chi_nu + chi_mu chi_nu'. The direct
    # einsum over the column list instead materializes (3, n_k, n_pair), which
    # is gigabytes on a molecule.
    S = np.zeros((len(coords), nao))
    for first, second in ((mu, nu), (nu, mu)):
        order = np.argsort(first, kind='stable')
        edges = np.searchsorted(first[order], np.arange(nao + 1))
        for m in range(nao):
            cols = order[edges[m]:edges[m + 1]]
            if len(cols):
                S[:, m] += np.einsum('kc,kc->k', pair_bar[:, cols], ao[:, second[cols]])
    P = (np.einsum('xkm,km->kx', g, S)
         + np.einsum('kc,xkc->kx', D_bar[:, len(mu):], gaux))
    return E, P / (2.0 * E)


def atomic_radii_adjoint(radii, P, origin=False):
    """{shell: dE/dr_i} from an adjoint P[g] = dE/dr_g on one atom's points.

    The reverse of `atomic_points`, walking the point list in that function's
    order. r_g = r_i u_g with u_g a unit direction, so dE/dr_i sums u_g . P[g]
    over the sub-shell. The cusp sample sits at the nucleus and carries no
    radius.
    """
    shells, out, k = subshells(), {}, (1 if origin else 0)
    for name, rs in radii.items():
        u = shells[name]
        grad = np.empty(len(np.atleast_1d(rs)))
        for i in range(len(grad)):
            grad[i] = np.einsum('gx,gx->', u, P[k:k + len(u)])
            k += len(u)
        out[name] = grad
    return out


# ---------------------------------------------------------------------------
# the D factor: D = M^T V^(1/2), all the way back to the nuclei
# ---------------------------------------------------------------------------

def sqrtm_adjoint(V, Vh_bar, tol=1e-12):
    """V_bar for an adjoint on V^(1/2), V symmetric positive semidefinite.

    The Frechet derivative L of the square root solves Vh L + L Vh = E, so in
    the eigenbasis it is the Hadamard multiplier 1/(sqrt(w_i) + sqrt(w_j)).
    That multiplier is symmetric, hence the map is self-adjoint and the reverse
    pass is the same operation. The truncation mirrors `separable_factors`,
    which drops eigenvalues below tol x max before taking the root.
    """
    w, U = np.linalg.eigh(V)
    keep = w > tol * w.max()
    Uk, sk = U[:, keep], np.sqrt(w[keep])
    E = Uk.T @ (0.5 * (Vh_bar + Vh_bar.T)) @ Uk
    return Uk @ (E / (sk[:, None] + sk[None, :])) @ Uk.T


def two_centre_adjoint(auxmol, V_bar, omega=0.0):
    """(natm, 3) gradient of sum_PQ V_bar[P,Q] (P|Q), V_bar symmetrized here.

    omega != 0 takes the integrals with the erf-attenuated operator, the
    long-range channel of a range-separated hybrid.

    pyscf's int2c2e_ip1 is +(grad P|Q), so the nuclear derivative is minus it,
    and both index positions contribute equally once V_bar is symmetric.
    """
    with auxmol.with_range_coulomb(omega):
        v1 = auxmol.intor('int2c2e_ip1', comp=3)           # +(grad P|Q)
    Vb = 0.5 * (V_bar + V_bar.T)
    t = -2.0 * np.einsum('xPQ,PQ->xP', v1, Vb, optimize=True)
    grad = np.zeros((auxmol.natm, 3))
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        grad[ia] += t[:, q0:q1].sum(axis=1)
    return grad


def shell_blocks(mol, per_ao_bytes, max_bytes):
    """Shell ranges whose slab, `per_ao_bytes` for each AO in it, fits the cap.

    Always yields at least one shell, so a single shell wider than the cap is
    still attempted rather than silently skipped.
    """
    ao_loc = mol.ao_loc_nr()
    blocks, sh0 = [], 0
    while sh0 < mol.nbas:
        sh1 = sh0 + 1
        # int() because ao_loc_nr() is int32 while a slab is nao*naux*8 bytes or
        # more. The product wraps negative on a production system, the test then
        # passes for every block, and the loop asks for the whole tensor.
        while (sh1 < mol.nbas
               and int(ao_loc[sh1 + 1] - ao_loc[sh0]) * per_ao_bytes <= max_bytes):
            sh1 += 1
        blocks.append((sh0, sh1))
        sh0 = sh1
    return blocks


def three_centre_adjoint(mol, auxmol, G3, omega=0.0, max_bytes=THREE_CENTER_BLOCK_BYTES):
    """(natm, 3) gradient of sum_{mn P} G3[m,n,P] (mn|P).

    omega != 0 takes the erf-attenuated operator, set on both molecules since
    `aux_e2` reads the range parameter from the combined environment.

    G3 is not symmetric in (m, n), because the ISDF test set pairs every AO with
    only the low-l ones, so the two orbital positions are contracted separately.
    The derivative integrals are contracted where they are made, blocked over
    the shells of the differentiated index, since a whole (3, nao, nao, naux)
    tensor is gigabytes at production size.
    """
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    t_first, t_second = np.zeros((3, nao)), np.zeros((3, nao))
    t_aux = np.zeros((3, naux))
    per_ao = 3 * int(nao) * int(naux) * 8
    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        for sh0, sh1 in shell_blocks(mol, per_ao, max_bytes):
            a0, a1 = ao_loc[sh0], ao_loc[sh1]
            sl = (sh0, sh1, 0, mol.nbas, 0, auxmol.nbas)
            d1 = pyscf_df.incore.aux_e2(
                mol, auxmol, intor='int3c2e_ip1', aosym='s1', comp=3,
                shls_slice=sl).reshape(3, a1 - a0, nao, naux)
            # minus, for the same reason as the two-centre case
            t_first[:, a0:a1] = -np.einsum('xmnP,mnP->xm', d1, G3[a0:a1],
                                           optimize=True)
            t_second[:, a0:a1] = -np.einsum('xnmP,mnP->xn', d1, G3[:, a0:a1],
                                            optimize=True)
            del d1
            d2 = pyscf_df.incore.aux_e2(
                mol, auxmol, intor='int3c2e_ip2', aosym='s1', comp=3,
                shls_slice=sl).reshape(3, a1 - a0, nao, naux)
            t_aux -= np.einsum('xmnP,mnP->xP', d2, G3[a0:a1], optimize=True)
            del d2
    grad = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        grad[ia] += t_first[:, p0:p1].sum(axis=1) + t_second[:, p0:p1].sum(axis=1)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        grad[ia] += t_aux[:, q0:q1].sum(axis=1)
    return grad


@dataclass(frozen=True)
class GaugeAdjoint:
    """One gauge of one fit: the D adjoint, its environment, and any adjoint a
    term outside D put on the same metric.

    d_bar: (nk, naux) adjoint on this gauge's factor.
    environment: what dresses this gauge's metric, None for the bare one.
    root_bar: an adjoint on the metric root R = (V + vtilde)^(1/2) from a term
        that reads R without going through D. It is added to D's own root
        adjoint before the Frechet solve, so the two halves cannot drift apart.
    kernel_bar: an adjoint on vtilde alone, which does not reach V and so cannot
        join `root_bar`. It goes to the environment's own adjoint instead.

    Both extras are refused on a bare gauge, which has no root and no kernel.
    """

    d_bar: np.ndarray
    environment: object = None
    root_bar: np.ndarray = None
    kernel_bar: np.ndarray = None


def dfactor_adjoint(mol, auxmol, coords, D_bar, layout, pts_local,
                    atom_of_point, frames=None, decay=_FRAME_DECAY,
                    regularization=DEFAULT_REGULARIZATION, with_frames=True,
                    environment=None):
    """(natm, 3) gradient of sum_{gP} D_bar[g,P] D[g,P], D = M^T V_env^(1/2)."""
    return dfactor_adjoint_gauges(mol, auxmol, coords,
                                  [(D_bar, environment)], layout, pts_local,
                                  atom_of_point, frames=frames, decay=decay,
                                  regularization=regularization,
                                  with_frames=with_frames)


def dfactor_adjoint_gauges(mol, auxmol, coords, gauges, layout, pts_local,
                           atom_of_point, frames=None, decay=_FRAME_DECAY,
                           regularization=DEFAULT_REGULARIZATION,
                           with_frames=True):
    """(natm, 3) gradient of sum over `gauges` of sum_{gP} D_bar[g,P] D[g,P].

    The whole D branch. Split D into the fit and the metric root, run the fit's
    Z-vector (`fit_adjoint`), then land the test-set adjoints on integrals.

        D[g,P] = sum_Q M[Q,g] Vh[Q,P]      Mbar = Vh Dbar^T,  Vhbar = M Dbar
        Mbar  --fit_adjoint-->  (Dtest_bar, F_bar)
        Dtest  columns are collocations: AO pairs, then auxiliaries
        F      columns are w_c V^-1 (mu nu|.) for pairs, and exactly the
               identity for auxiliaries, which therefore contributes nothing
        Vh     contributes through the square-root Frechet adjoint, and V^-1
               inside F contributes a second term to V_bar

    Each gauge is a `GaugeAdjoint`, or the bare (D_bar, environment) pair. A
    solvated target carries two gauges of one fit, since the self-energy screens
    bare while the kernel keeps the cavity. Everything below the metric root is
    common to them, so the fit is built once and each gauge contributes its own
    adjoints to shared accumulators. Both bases collocate on the same points, so
    their centre terms differ but the point and frame chain is shared.
    """
    mu, nu, wc = layout
    npair = len(mu)
    ao = mol.eval_gto('GTOval_sph', coords)
    V = auxmol.intor('int2c2e', aosym='s1')

    D_test = test_set_D(mol, auxmol, coords, layout)
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    e3c = e3c.reshape(mol.nao_nr(), mol.nao_nr(), auxmol.nao_nr())
    g_cols = e3c[mu, nu, :].T                              # (naux, npair)
    F = np.hstack([np.linalg.solve(V, g_cols) * wc[None, :],
                   np.eye(auxmol.nao_nr())])
    M = fit_M_stable(D_test, F, regularization)

    V_bar = np.zeros_like(V)
    G3 = np.zeros_like(e3c)
    ao_bar = np.zeros_like(ao)
    aux_bar = np.zeros((len(coords), auxmol.nao_nr()))
    env_grad = None
    for gauge in gauges:
        gauge = gauge if isinstance(gauge, GaugeAdjoint) else GaugeAdjoint(*gauge)
        D_bar, environment = gauge.d_bar, gauge.environment
        dressed = dresses_interaction(environment, auxmol)
        if not dressed and (gauge.root_bar is not None
                            or gauge.kernel_bar is not None):
            raise ValueError(
                f'{environment!r} dresses nothing, so this gauge has no '
                'vtilde and no dressed root for the extra adjoints to land '
                'on; they belong to the gauge that carries the environment')
        V_gauge = V + environment.aux_kernel(auxmol) if dressed else V
        Vh = aux_metric_sqrt(auxmol, environment, V=V)

        M_bar = Vh @ D_bar.T                               # (naux, nk)
        Vh_bar = M @ D_bar                                 # (naux, naux)
        if gauge.root_bar is not None:
            Vh_bar = Vh_bar + gauge.root_bar
        Dtest_bar, F_bar = fit_adjoint(D_test, F, M_bar, regularization)

        # --- the metric: the root of the dressed gauge, and the bare V^-1
        # inside F's pair columns. Y is the adjoint on V_env and reaches V and
        # vtilde alike.
        Y = sqrtm_adjoint(V_gauge, Vh_bar)
        Fp_bar = F_bar[:, :npair]
        VinvFb = np.linalg.solve(V, Fp_bar)                # (naux, npair)
        V_bar += Y - F[:, :npair] @ VinvFb.T
        g_bar = VinvFb * wc[None, :]                       # adjoint on (mu nu|P)
        np.add.at(G3, (mu, nu), g_bar.T)
        if dressed:
            # V_gauge = V + vtilde, so Y is the adjoint on BOTH; a term that
            # differentiated vtilde alone adds only here.
            kernel_bar = (Y if gauge.kernel_bar is None
                          else Y + gauge.kernel_bar)
            k_grad = environment.aux_kernel_adjoint(auxmol, kernel_bar)
            env_grad = k_grad if env_grad is None else env_grad + k_grad

        # --- the test set's collocations, sharing one point chain
        pair_bar = Dtest_bar[:, :npair] * wc[None, :]
        np.add.at(ao_bar.T, mu, (pair_bar * ao[:, nu]).T)  # chi_mu's slot
        np.add.at(ao_bar.T, nu, (pair_bar * ao[:, mu]).T)  # chi_nu's slot
        aux_bar += Dtest_bar[:, npair:]

    grad = two_centre_adjoint(auxmol, V_bar) + three_centre_adjoint(mol, auxmol, G3)
    if env_grad is not None:
        grad = grad + env_grad
    c_ao, P = basis_centre_forces(mol, coords, ao_bar)
    c_aux, P_aux = basis_centre_forces(auxmol, coords, aux_bar)
    grad += c_ao + c_aux
    grad += point_chain(mol, P + P_aux, pts_local, atom_of_point, frames=frames,
                        decay=decay, with_frames=with_frames)
    return grad


# ---------------------------------------------------------------------------
# the orbital-energy chain, without any norb^4 tensor
# ---------------------------------------------------------------------------

def require_no_range_separation(mf, what):
    """Refuse a derivative that would silently drop the long-range exchange.

    No skeleton here calls it any more, since `fock_partial_skeleton`, its
    fitted twin, `qp_xc_correction_skeleton` and `isdf_exchange_skeleton` all
    carry every channel. It remains as the refusal a single-channel route owes
    its caller.

    The forward quantities are safe on a range-separated hybrid, since `v_xc` is
    the SCF's own `get_veff - get_j`. A skeleton that weights one full-range K
    build by `xc_hybrid_coeff` is not, because that scalar is alpha and the
    whole beta K_lr(omega) term goes missing without announcing itself.
    """
    omega, alpha, beta = rsh_split(mf)
    if omega != 0.0 and beta != 0.0:
        raise NotImplementedError(
            f'{what} does not support the range-separated hybrid '
            f'{mf.xc!r} (omega={omega:g}, alpha={alpha:g}, beta={beta:g}): '
            f'the skeleton derivative carries one full-range exchange build '
            f'weighted by alpha, so the beta K_lr(omega) term would be '
            f'silently dropped. ENERGIES on this reference are correct -- '
            f'v_xc comes from the SCF itself -- so use them, or finite '
            f'differences of them, until the split lands.')


def fock_partial_Y(mf, gamma, nocc):
    """Orbital-rotation gradient of Tr[gamma F], gamma a symmetric MO partial.

    Tr[gamma F] depends on the coefficients through the two-sided transform and
    through the density inside F, and both come out as

        Y[u,p] = 2 (F gamma)[u,p] + 4 delta_{p occupied} (C^T G(gamma_ao) C)[u,p]

    with G(x) the closed-shell response kernel and gamma_ao = C gamma C^T. The
    same expression serves the target's own partial and the multiplier's. It
    needs a Coulomb and an exchange build and nothing else, so with a fitted or
    ISDF mean field it is cubic and no norb^4 tensor appears.
    """
    C = mf.mo_coeff
    gs = 0.5 * (gamma + gamma.T)
    g_ao = C @ gs @ C.T
    G = response_kernel(mf)(g_ao)
    F_mo = fock_mo(mf)
    Y = 2.0 * (F_mo @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ G @ C)[:, :nocc]
    return Y


def qp_xc_correction(mf, states=None, reaction_field=None):
    """`GW.qp_solve.static_exchange_diagonal` at exchange='mf', all states by
    default: the gradient chain and `calc_qp_energy` differentiate and evaluate
    one function, so there is nothing here but the default."""
    states = (np.arange(np.shape(mf.mo_coeff)[-1]) if states is None
              else states)
    return static_exchange_diagonal(mf, mf.mol, states, exchange='mf',
                                    reaction_field=reaction_field)


def _delta_o_response(mf):
    """x -> d(Sigma_x - v_xc)/dD . x, the response of the correction's operator.

    `v_xc` here is `get_veff - get_j`, which on a hybrid already contains
    -a_x K/2, so the operator is

        O[D] = -((1 - a_x)/2) K - v_xc^DFT

    and its response, using f_xc(x) = G_KS(x) - J(x) + (a_x/2) K(x), is

        dO(x) = J(x) - K(x)/2 - G_KS(x).

    On Hartree-Fock that is identically zero, as it must be. Getting the a_x
    bookkeeping wrong gives -K(x)/2, which coincides with the right answer on a
    pure functional and is wrong by the whole gradient on a hybrid. Test a
    hybrid, not just PBE.
    """
    # continuum-free: O itself carries no reaction field, so neither may its
    # response, and the J - K/2 - G identity holds only for the gas-phase G
    kern = response_kernel(mf, environment=False)

    def apply(x):
        vj, vk = mf.get_jk(mf.mol, x, hermi=1)
        return vj - 0.5 * vk - kern(x)
    return apply


def qp_xc_correction_Y(mf, weights, nocc):
    """Orbital-rotation gradient of sum_p w_p <p|Sigma_x - v_xc|p>.

    The same shape as `fock_partial_Y`, with the Fock replaced by this
    operator: two terms, one from the two-sided MO transform and one from the
    operator's own density dependence. It must enter the Lagrangian BEFORE the
    multiplier solve, because it shares Lambda with every other contribution.
    """
    C = mf.mo_coeff
    gs = np.diag(np.asarray(weights, float))
    g_ao = C @ gs @ C.T
    dm = mf.make_rdm1()
    o_ao = -0.5 * mf.get_k(mf.mol, dm) - (mf.get_veff(mf.mol, dm)
                                          - mf.get_j(mf.mol, dm))
    o_mo = C.T @ o_ao @ C
    Y = 2.0 * (o_mo @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ _delta_o_response(mf)(g_ao) @ C)[:, :nocc]
    return Y


def qp_xc_correction_skeleton(mf, weights, nocc):
    """(natm, 3) of d/dR sum_p w_p <p|Sigma_x - v_xc|p>, coefficients fixed.

    The operator is -((1 - a_x)/2) K - v_xc^DFT (see `_delta_o_response`), so
    the exchange half is `fock_partial_skeleton`'s K structure at weight
    -(1 - a_x) with no Coulomb term, and the rest is minus `xc_skeleton`. There
    is no one-electron part. On Hartree-Fock both halves vanish.
    """
    is_ks, _ = xc_hybrid_coeff(mf)
    if not is_ks:
        return np.zeros((mf.mol.natm, 3))
    C = mf.mo_coeff
    gs = np.diag(np.asarray(weights, float))
    # Sigma_x is the FULL-RANGE exact exchange whatever the reference is, and
    # v_xc carries the reference's own channels, so Sigma_x - v_xc is one
    # full-range unit minus each of them. On a global hybrid that collapses to
    # the single weight 1 - a_x.
    delta = [(0.0, 1.0)] + [(o, -w) for o, w in exchange_channels(mf)]
    # A density-fitted mean field must go through the fitted skeleton, for the
    # same reason `fock_partial_skeleton` routes there. pyscf's conventional and
    # density-fitted derivative J/K do not share a convention, and the mismatch
    # shows up as a sharply degraded translational invariance.
    if isinstance(getattr(mf, 'with_df', None), ISDFJK):
        # Sigma_x is the interpolated exchange here, because that is what the
        # SCF built. Taking it from the auxiliary basis would differentiate one
        # operator while the energy carries another.
        return (isdf_fock_partial_exchange(mf, gs, channels=delta)
                - xc_skeleton(mf, gs))
    if getattr(mf, 'with_df', None) is not None:
        grad = fock_partial_skeleton_df(mf, _auxmol_of(mf), gs, nocc,
                                        coulomb=False, channels=delta)
        return grad - xc_skeleton(mf, gs)
    g_ao = C @ gs @ C.T
    D = mf.make_rdm1()
    grad = np.zeros((mf.mol.natm, 3))
    for omega, weight in delta:
        if weight != 0.0:
            grad = grad + exchange_channel_skeleton(mf, g_ao, D, omega, weight)
    return grad - xc_skeleton(mf, gs)


def exx_double_counting_skeleton(mf, mol=None):
    """(natm, 3) of d/dR (E_x^exact - E_xc), MO coefficients held fixed.

    Taken as a difference of two pyscf gradients at one density. Both gradient
    objects read the same mo_coeff/mo_energy/mo_occ, so the one-electron,
    overlap, Coulomb and nuclear terms are bit-identical and cancel, leaving
    full exchange against a_x-exchange-plus-xc. Weighting derivative K builds by
    hand instead is where the coefficients go wrong.

    A continuum is stripped off the Kohn-Sham side first. The reaction field is
    the same functional of the same density in both members, so it belongs to
    neither, and leaving it in would leave -dE_PCM/dR behind. The mean field's
    own force carries it once, in `FactorChain.mean_field_gradient`.

    Each member goes through `isdf_jk.mean_field_skeleton_force`, the one
    dispatch that answers with the force of the energy a given mean field
    reported.
    """
    mol = mf.mol if mol is None else mol
    if not xc_hybrid_coeff(mf)[0]:
        return np.zeros((mol.natm, 3))
    ks = mf.undo_solvent() if hasattr(mf, 'undo_solvent') else mf
    hf = pyscf_scf.RHF(mol)
    with_df = getattr(ks, 'with_df', None)
    if with_df is not None:
        hf = hf.density_fit(auxbasis=with_df.auxbasis)
    hf.mo_coeff, hf.mo_energy = mf.mo_coeff, mf.mo_energy
    hf.mo_occ, hf.converged = mf.mo_occ, True
    refuse_dispersion_under_rpa(mf, 'the Kohn-Sham-to-Hartree-Fock skeleton')
    if isinstance(with_df, ISDFJK):
        # Both sides must be the same fit, so the Hartree-Fock partner is given
        # the very same ISDFJK object. The points, collocation and fit matrix
        # are then bit-identical and everything but the exchange fraction
        # cancels. A fresh fit would leave its own realization in the
        # difference.
        hf.with_df = with_df
    return (np.asarray(mean_field_skeleton_force(hf))
            - np.asarray(mean_field_skeleton_force(ks)))


def exx_double_counting_Y(mf, nocc):
    """Orbital-rotation gradient of (E_x^exact - E_xc).

    One term, where `qp_xc_correction_Y` needs two. The functional derivative of
    this energy with respect to the density is exactly the operator
    `qp_xc_correction` carries, dE/dD = Sigma_x - v_xc, and an energy does not
    depend on the coefficients through the states as <p|O[D]|p> does.

    The factor and the occupied restriction follow `fock_partial_Y`.
    """
    if not xc_hybrid_coeff(mf)[0]:
        return np.zeros((np.shape(mf.mo_coeff)[-1],) * 2)
    C = mf.mo_coeff
    dm = mf.make_rdm1()
    o_ao = -0.5 * mf.get_k(mf.mol, dm) - (mf.get_veff(mf.mol, dm)
                                          - mf.get_j(mf.mol, dm))
    Y = np.zeros((C.shape[1], C.shape[1]))
    Y[:, :nocc] = 4.0 * (C.T @ o_ao @ C)[:, :nocc]
    return Y


def solve_lambda(mf, Y_E, nocc, tol=ORBITAL_MULTIPLIER_TOL,
                 max_iter=ORBITAL_MULTIPLIER_MAX_ITER, verbose=False):
    """Symmetric zero-diagonal Lambda with antisym(Y_E + Y[Lambda]) = 0.

    The cubic route's entry to `solve_orbital_multipliers`: the matvec is
    `fock_partial_Y` -- a Coulomb and an exchange build -- instead of the
    four-index tensor, and the orbital energies come from the mean field.
    """
    return solve_orbital_multipliers(
        lambda Lam: fock_partial_Y(mf, Lam, nocc),
        Y_E, mf.mo_energy, tol=tol, max_iter=max_iter, verbose=verbose)


def _auxmol_of(mf):
    """The auxiliary molecule behind a density-fitted mean field."""
    aux = getattr(mf.with_df, 'auxmol', None)
    if aux is None:
        aux = pyscf_df.addons.make_auxmol(mf.mol, auxbasis=mf.with_df.auxbasis)
    return aux


def reaction_field_skeleton(mf, gamma):
    """(natm, 3) the ground-state continuum's entry in the skeleton of Tr[gamma F].

    V_PCM(eps_static) is part of this mean field's Fock, so a folded Fock
    partial owes it a derivative term of the same kind as the Coulomb one, with
    the relaxed density on one side and the SCF density on the other. Zero
    without a continuum, which is why callers add it unconditionally.
    """
    pcm = getattr(mf, 'with_solvent', None)
    if pcm is None:
        return 0.0
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return reaction_field_fock_skeleton(pcm, mf.make_rdm1(), g_ao)


# ---------------------------------------------------------------------------
# the interpolated exchange: E_K = -(a_x/4) sum_PQ Z_PQ W_PQ^2, W = X D X^T
# ---------------------------------------------------------------------------

def isdf_exchange_adjoints(X, Z, dm, a_x, dm_other=None):
    """(X_bar, Z_bar) of the ISDF exchange form

        E = -(a_x/4) sum_PQ Z_PQ (X dm X^T)_PQ (X dm_other X^T)_PQ

    dm_other = None is the SCF's own E_K = -(a_x/4) Tr[dm K[dm]]. W enters
    squared, so W_bar is -(a_x/2) Z .* W and X, sitting on both sides, collects
    twice that.

    A folded Fock partial needs the two-density form. Its exchange half is
    -(gamma vk[D] + D vk[gamma]), and K is linear in the density it is built
    from, so the same contraction serves with W^2 replaced by W1 W2.
    """
    W1 = X @ dm @ X.T
    other = dm if dm_other is None else dm_other
    W2 = W1 if dm_other is None else X @ other @ X.T
    Z_bar = -0.25 * a_x * (W1 * W2)
    X_bar = -0.5 * a_x * ((Z * W2) @ (X @ dm) + (Z * W1) @ (X @ other))
    return X_bar, Z_bar


def isdf_fock_partial_exchange(mf, gamma, channels=None):
    """(natm, 3) exchange half of a folded Fock partial, from ISDF factors.

    The folded two-particle density Gamma_abcd = gamma_ab D_cd - gamma_ac D_bd/2
    contributes -(w/2) Tr[gamma K[D]] per channel, while the SCF energy is
    -(a_x/4) Tr[D K[D]]. So the bilinear coefficient is twice the SCF's, not
    eight times.

    A finite difference cannot fix that factor, since the reference and the
    analytic side share it. What pins it is agreement with
    `fock_partial_skeleton_df` at the same gamma.

    `channels` overrides the reference's own, because
    `qp_xc_correction_skeleton` needs Sigma_x - v_xc, one full-range unit minus
    each of the reference's channels.
    """
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return isdf_exchange_skeleton(mf, dm=g_ao, dm_other=mf.make_rdm1(),
                                  prefactor=2.0, channels=channels)


def isdf_exchange_skeleton(mf, dm=None, dm_other=None, prefactor=1.0,
                           channels=None):
    """(natm, 3) of d/dR E_K^ISDF with the density matrix held fixed.

    The whole ISDF branch of the force: the collocation's adjoint, the fit's
    Z-vector, and the derivative integrals underneath them. Z = M^T V M is
    differentiated as it is built in `z_mode='dense'`. `z_mode='factored'` holds
    L = V^(1/2) M instead, and its Z differs by the modes the square root
    truncates, which is negligible on a bare Coulomb metric.
    """
    with_df = mf.with_df
    mol, auxmol, crd = with_df.mol, with_df.auxmol, with_df.coords
    dm = mf.make_rdm1() if dm is None else np.asarray(dm)
    channels = exchange_channels(mf) if channels is None else list(channels)
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    reg = with_df.regularization

    # The fit is rebuilt, not taken from the mean field, so that one
    # realization of the estimator is differentiated. `fit_adjoint` reverses
    # `fit_M_stable` on a frozen column set, while `fit_M_streaming` builds its
    # Gram matrix from the unscreened test set. The two cannot be mixed.
    layout = test_set_layout(mol, crd, l_max_second=with_df.l_max_second)
    mu, nu, wc = layout
    npair = len(mu)
    D_test = test_set_D(mol, auxmol, crd, layout)
    V = auxmol.intor('int2c2e', aosym='s1')
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e',
                                 aosym='s1').reshape(nao, nao, naux)
    F = np.hstack([np.linalg.solve(V, e3c[mu, nu, :].T) * wc[None, :],
                   np.eye(naux)])
    M = fit_M_stable(D_test, F, reg)
    X = mol.eval_gto('GTOval_sph', crd)

    # One channel per exchange operator, sharing the fit. The ISDF route keeps
    # the bare M and builds Z_w = M^T V_w M, which is what holds
    # K_SR + K_LR = K_bare, so a range-separated hybrid adds metrics and not
    # fits. W = X dm X^T is the same for every channel and is never rebuilt.
    X_bar = np.zeros((len(crd), nao))
    M_bar = np.zeros_like(M)
    V_bar = np.zeros_like(V)
    attenuated = []
    for omega, weight in channels:
        if weight == 0.0:
            continue
        if omega == 0.0:
            V_w = V
        else:
            with range_coulomb(mol, auxmol, omega):
                V_w = auxmol.intor('int2c2e', aosym='s1')
        xb, Z_bar = isdf_exchange_adjoints(X, M.T @ (V_w @ M), dm,
                                           prefactor * weight, dm_other)
        # Z = M^T V M: M sits on both sides of a symmetric adjoint, V between.
        X_bar += xb
        M_bar += 2.0 * (V_w @ M) @ Z_bar
        if omega == 0.0:
            V_bar += M @ Z_bar @ M.T
        else:
            attenuated.append((omega, M @ Z_bar @ M.T))
    Dtest_bar, F_bar = fit_adjoint(D_test, F, M_bar, reg)

    # F's pair columns are w_c V^-1 (mu nu|P), so they reach the metric a
    # second time and the three-centre integrals once; its auxiliary block is
    # exactly the identity and contributes nothing.
    VinvFb = np.linalg.solve(V, F_bar[:, :npair])
    V_bar -= F[:, :npair] @ VinvFb.T
    G3 = np.zeros_like(e3c)
    np.add.at(G3, (mu, nu), (VinvFb * wc[None, :]).T)

    # One collocation adjoint for both slots: the exchange's own X and the
    # test set's AO pairs are the same evaluation on the same points.
    ao_bar = X_bar.copy()
    pair_bar = Dtest_bar[:, :npair] * wc[None, :]
    np.add.at(ao_bar.T, mu, (pair_bar * X[:, nu]).T)
    np.add.at(ao_bar.T, nu, (pair_bar * X[:, mu]).T)

    # The attenuated metrics reach the fit nowhere else -- F is built on the
    # BARE operator -- so they contribute through their own derivative alone.
    grad = (two_centre_adjoint(auxmol, V_bar)
            + three_centre_adjoint(mol, auxmol, G3))
    for omega, vb in attenuated:
        grad = grad + two_centre_adjoint(auxmol, vb, omega=omega)
    c_ao, P = basis_centre_forces(mol, crd, ao_bar)
    c_aux, P_aux = basis_centre_forces(auxmol, crd, Dtest_bar[:, npair:])
    grad += c_ao + c_aux
    pts_local, owner = point_layout(mol, with_df.grid_radii,
                                    with_df.grid_origins)
    return grad + point_chain(mol, P + P_aux, pts_local, owner,
                              frames=atomic_frames(mol)[0], with_frames=True)


def fock_partial_skeleton(mf, gamma, nocc):
    """(natm, 3) two-electron skeleton for a folded Fock partial, tensor-free.

    Folding Tr[gamma F] gives the separable two-particle density

        Gamma_abcd = gamma_ab D_cd - (1/2) gamma_ac D_bd,   D the SCF density,

    whose four-permutation sum contracts against derivative integrals as
    Coulomb- and exchange-shaped terms only,

        dE_2/dR_A = sum_{a in A} [ 2 (gamma.vj[D] + D.vj[gamma])
                                   - (gamma.vk[D] + D.vk[gamma]) ]_a

    gamma and D enter E_2 symmetrically, which forces one coefficient per
    channel. A free least-squares fit cannot recover them, since the four
    contractions are linearly dependent and return unphysical weights that
    still fit exactly. Constraining by that symmetry gives 2 and -1 exactly.

    A density-fitted mean field is routed to `fock_partial_skeleton_df`
    instead, and must be. pyscf's conventional and density-fitted derivative
    J/K do not share a convention, and feeding the latter through the weights
    below gives a gradient large enough to ruin a geometry while looking
    plausible.
    """
    is_ks, a_x = xc_hybrid_coeff(mf)
    if getattr(mf, 'with_df', None) is not None:
        # An interpolated mean field splits the two halves. Its Coulomb is
        # fitted and integral-direct, so the fitted skeleton differentiates it.
        # Its exchange comes from the ISDF factors, which the auxiliary-basis
        # skeleton knows nothing about.
        if isinstance(mf.with_df, ISDFJK):
            g = fock_partial_skeleton_df(mf, _auxmol_of(mf), gamma, nocc,
                                         channels=(), coulomb=True)
            g = g + isdf_fock_partial_exchange(mf, gamma)
        else:
            g = fock_partial_skeleton_df(mf, _auxmol_of(mf), gamma, nocc,
                                         channels=exchange_channels(mf))
        return (g + (xc_skeleton(mf, gamma) if is_ks else 0.0)
                + reaction_field_skeleton(mf, gamma))
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    D = mf.make_rdm1()
    vj_D = grad_rhf.get_jk(mf.mol, D)[0]
    vj_g = grad_rhf.get_jk(mf.mol, g_ao)[0]
    # The Coulomb half is never range-separated and is built once. The exchange
    # half is one build per channel, so a range-separated hybrid carries its
    # erf-attenuated term instead of dropping it.
    t = 2.0 * (np.einsum('ab,xab->xa', g_ao, vj_D, optimize=True)
               + np.einsum('ab,xab->xa', D, vj_g, optimize=True))
    grad = np.zeros((mf.mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
        grad[ia] = t[:, p0:p1].sum(axis=1)
    for omega, weight in (exchange_channels(mf) if is_ks else [(0.0, 1.0)]):
        if weight != 0.0:
            grad = grad + exchange_channel_skeleton(mf, g_ao, D, omega, weight)
    if is_ks:
        grad = grad + xc_skeleton(mf, gamma)
    return grad + reaction_field_skeleton(mf, gamma)


def xc_skeleton(mf, gamma):
    """(natm, 3) of d/dR Tr[gamma v_xc], MO coefficients held fixed.

    The exchange-correlation half of the Kohn-Sham Fock partial. It appears
    once, not twice. The Coulomb and exchange terms come from a four-index
    object in which gamma and the density enter symmetrically, while v_xc is
    not bilinear in the density.

    `pyscf.hessian.rks._get_vxc_deriv1` supplies dv_xc/dR including the
    density's response to the basis functions moving, which is the skeleton
    definition. Its convention was measured against a finite difference, not
    assumed.
    """
    from pyscf.hessian import rks as _hrks
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    vmat = _hrks._get_vxc_deriv1(mf.Hessian(), C, mf.mo_occ,
                                 getattr(mf, 'max_memory', 4000))
    return np.einsum('axij,ij->ax', np.asarray(vmat), g_ao)


def eps_chain_gradient(mf, eps_bar, nocc, Y_extra=None, verbose=False):
    """(natm, 3) of sum_pq gamma_pq dF_pq/dR, with no norb^4 tensor anywhere.

    The dense engine's Lagrangian, rebuilt on Coulomb/exchange builds. Fold the
    adjoint into a Fock partial, solve for the canonical-condition multiplier,
    read the orthonormality multiplier off the stationary orbital gradient, and
    contract the skeleton. `Y_extra` carries any orbital-rotation dependence
    that is not expressible as an (F, ERI) partial.

    eps_bar: the adjoint on the orbital energies, gamma = diag(eps_bar), or a
        full symmetric MO Fock partial gamma_pq = dE/dF_pq.
    """
    # circular import: grad_engine also imports from isdf_derivatives
    from src.gradients.grad_engine import one_electron_skeleton
    C = mf.mo_coeff
    eps_bar = np.asarray(eps_bar, float)
    gamma = np.diag(eps_bar) if eps_bar.ndim == 1 else 0.5 * (eps_bar + eps_bar.T)
    Y_E = fock_partial_Y(mf, gamma, nocc)
    if Y_extra is not None:
        Y_E = Y_E + Y_extra
    Lam, res = solve_lambda(mf, Y_E, nocc, verbose=verbose)
    Y_tot = Y_E + fock_partial_Y(mf, Lam, nocc)
    Mmat = -0.25 * (Y_tot + Y_tot.T)
    gamma_tot = gamma + Lam
    g2 = fock_partial_skeleton(mf, gamma_tot, nocc)
    g1 = one_electron_skeleton(mf.mol, mf, [C @ gamma_tot @ C.T],
                               [C @ Mmat @ C.T])[0]
    return g1 + g2, {'stationarity': np.abs(Y_tot - Y_tot.T).max(),
                     'multiplier_res': res}


def exchange_channel_skeleton(mf, g_ao, D, omega, weight):
    """(natm, 3) of weight * d/dR of one exchange channel, four-centre.

    omega = 0 is the ordinary full-range build, and a nonzero one is the
    erf-attenuated channel of a range-separated hybrid.

    The long-range channel is not density fitted, and that is forced. Its
    auxiliary metric (P|erf(omega r)/r|Q) goes rank-deficient, and the
    derivative of a truncated pseudo-inverse is not -V^-1 dV V^-1, so a fitted
    skeleton is wrong there at any threshold. Four-centre derivative integrals
    have no metric and no such failure, and the term costs one exchange build
    either way.

    `pyscf.grad.rhf.get_jk` is used directly, not `mf.Gradients()`, which on a
    fitted mean field returns the fitted derivative J/K.
    """
    mol = mf.mol
    with mol.with_range_coulomb(omega):
        vk_D = grad_rhf.get_jk(mol, D)[1]
        vk_g = grad_rhf.get_jk(mol, g_ao)[1]
    t = -weight * (np.einsum('ac,xac->xa', g_ao, vk_D, optimize=True)
                   + np.einsum('ad,xad->xa', D, vk_g, optimize=True))
    grad = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        grad[ia] = t[:, p0:p1].sum(axis=1)
    return grad


def fock_partial_skeleton_df(mf, auxmol, gamma, nocc, channels=((0.0, 1.0),),
                             coulomb=True):
    """(natm, 3) two-electron skeleton for a density-fitted mean field.

    pyscf's conventional derivative J/K and its density-fitted one do not share
    a convention, so this skeleton is built from the three- and two-centre
    adjoints directly, both gated against finite differences of the integrals.

    With (ab|cd) = sum_PQ J[ab,P] Vinv[P,Q] J[cd,Q] and the folded density
    Gamma_abcd = gamma_ab D_cd - gamma_ac D_bd / 2,

        E_2 = a^T Vinv b - (1/2) sum_PQ Vinv[P,Q] T[P,Q]
        a_P = sum_ab gamma_ab J[ab,P],   b_P = sum_cd D_cd J[cd,P]
        T_PQ = tr[gamma J^P D J^Q]      (symmetric)

    whose adjoints are

        Jbar[ab,P] = gamma_ab (Vinv b)_P + D_ab (Vinv a)_P - [gamma K^P D]_ab
        Vbar       = -(Vinv a)(Vinv b)^T + (1/2) Vinv T Vinv      (symmetrized)

    with K^P = sum_Q Vinv[P,Q] J^Q. Storage is three-index throughout.

    `channels` is `exchange_channels(mf)`, and a range-separated hybrid's
    second channel carries its own density fit. `coulomb=False` leaves the
    exchange alone, which is what the Sigma_x - v_xc correction needs. On a
    Kohn-Sham reference the XC half comes from `xc_skeleton`, which the caller
    adds.
    """
    mol = mf.mol
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    D = mf.make_rdm1()
    nao, naux = mol.nao_nr(), auxmol.nao_nr()

    def _fit(omega):
        """(J, V) of the density fit under the operator of one channel.

        A long-range channel cannot reuse the Coulomb fit, since
        (ab|erf(omega r)/r|cd) has its own three- and two-centre integrals.
        """
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            j = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e',
                                       aosym='s1').reshape(nao, nao, naux)
            v = auxmol.intor('int2c2e', aosym='s1')
        return j, v

    J, V = _fit(0.0)
    a = np.einsum('ab,abP->P', g_ao, J, optimize=True)
    b = np.einsum('cd,cdP->P', D, J, optimize=True)
    # A direct solve on the Coulomb metric, which is full rank and well
    # conditioned. Routing it through the regularized inverse costs an order of
    # magnitude of precision for no benefit, and only the long-range metric
    # needs the regularization.
    Va, Vb = np.linalg.solve(V, a), np.linalg.solve(V, b)

    # GEMMs, not einsums. A three-operand contraction carrying the auxiliary
    # batch index falls off BLAS and runs an order of magnitude slower for
    # identical flops.
    def _sandwich(Tn):
        """g_ao Tn[P] D for every P, as two large GEMMs.

        The einsum falls off BLAS on the batch index, and a Python loop over
        naux issues hundreds of GEMMs too small to amortize their call
        overhead. Folding the batch index into the free dimension keeps one
        matrix product a side.
        """
        left = (g_ao @ Tn.transpose(1, 0, 2).reshape(nao, naux * nao))
        left = left.reshape(nao, naux, nao).transpose(1, 0, 2)
        return (left.reshape(naux * nao, nao) @ D).reshape(naux, nao, nao)

    # The Coulomb half is never range-separated and is built once. The exchange
    # half is built once per channel, each with its own fit and its own
    # derivative integrals.
    c = 1.0 if coulomb else 0.0
    grad = np.zeros((mol.natm, 3))
    if c:
        Jbar_c = (g_ao[:, :, None] * Vb[None, None, :]
                  + D[:, :, None] * Va[None, None, :])
        grad += (three_centre_adjoint(mol, auxmol, Jbar_c)
                 + two_centre_adjoint(auxmol, -np.outer(Va, Vb)))
    for omega, weight in channels:
        if weight == 0.0:
            continue
        if omega != 0.0:
            grad += exchange_channel_skeleton(mf, g_ao, D, omega, weight)
            continue
        Jw, Vw = J, V
        # The LONG-RANGE metric is numerically singular, so it is inverted on
        # its numerical range and never solved through.
        if omega == 0.0:
            Kw = np.linalg.solve(Vw, Jw.reshape(-1, naux).T).reshape(naux, nao,
                                                                     nao)
        else:
            Kw = (aux_metric_inverse(Vw)
                  @ Jw.reshape(-1, naux).T).reshape(naux, nao, nao)
        GJ = _sandwich(Jw.transpose(2, 0, 1))
        GK = _sandwich(Kw)
        T = GJ.reshape(naux, -1) @ Jw.reshape(-1, naux)
        if omega == 0.0:
            VTV = np.linalg.solve(Vw, np.linalg.solve(Vw, T).T).T
        else:
            Vwi = aux_metric_inverse(Vw)
            VTV = Vwi @ T @ Vwi
        grad += weight * (
            three_centre_adjoint(mol, auxmol, -GK.transpose(1, 2, 0),
                                 omega=omega)
            + two_centre_adjoint(auxmol, 0.5 * VTV, omega=omega))
    return grad

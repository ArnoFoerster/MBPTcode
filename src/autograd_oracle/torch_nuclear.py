"""The solvated BSE@GW chain from the NUCLEAR COORDINATES, in torch.

Everything below the factors that `torch_chain` starts from, so that
autograd differentiates the whole surface with respect to the atoms:

    coords -> integrals (torch_integrals)            S, T, V_nuc, (mu nu|P), (P|Q)
           -> cavity: points, switching, S, D, K, R  (pyscf's PCM, ported)
           -> Q = sym(K^-1 R), vtilde = A_P Q A_P^T   two dielectric constants:
                                                       static for the SCF, optical for W
           -> DF-RHF Fock with the reaction field; the converged density is a
              FIXED POINT differentiated by the implicit function theorem
              (`SCFFixedPoint`), then (eps, C) by one generalized eigensolve
           -> ISDF: collocation on frozen local points and frames, the
              regularized least-squares fit, D = M^T (V [+ vtilde])^(1/2)
           -> X_mo = X_ao C, and `torch_chain.forward` to Omega.

Discrete choices are read off production and frozen: the retained cavity
points, the pair layout, the interpolation points and frames, the quasiparticle
window, residue sets, Newton seeds and grids.
"""
import math

import numpy as np
import torch

from src.autograd_oracle import torch_chain as tc
from src.autograd_oracle import torch_integrals as ti

torch.set_default_dtype(torch.float64)


class Constants:
    """Everything the torch chain needs besides the coordinates, as read off
    the production objects by `gate_nuclear.py`."""

    def __init__(self, z):
        g = lambda k: torch.as_tensor(np.asarray(z[k], float))
        self.mol = ti.ShellTable.from_npz(z, 'mol')
        self.aux = ti.ShellTable.from_npz(z, 'aux')
        self.charges = g('charges')
        self.nelec = int(z['nelec'])
        # cavity
        self.owner = np.asarray(z['cav_owner'], int)
        self.normals, self.weights = g('cav_normals'), g('cav_weights')
        self.r_vdw, self.xi = g('cav_r_vdw'), g('cav_xi')
        self.R_J, self.R_in_J, self.R_sw_J = g('cav_R_J'), g('cav_R_in_J'), g('cav_R_sw_J')
        self.eps_opt, self.eps_static = float(z['eps_opt']), float(z['eps_static'])
        self.method = str(z['pcm_method'])
        self.solvated = bool(z['solvated'])
        # ISDF
        self.layout = (np.asarray(z['lay_mu'], int), np.asarray(z['lay_nu'], int), g('lay_w'))
        self.pts_local = g('pts_local')                 # (M, 3), stacked atom by atom
        self.pt_owner = np.asarray(z['pt_owner'], int)
        self.frames = g('frames')                       # (natm, 3, 3), rows are axes
        # the geometry the frames were frozen at, which they turn with
        self.ref_coords = g('ref_coords') if 'ref_coords' in z.files else None
        self.regularization = float(z['regularization'])
        self.D_ref = g('D_ref')


def switch_h(x):
    """York-Karplus switching polynomial, clamped to [0, 1]."""
    xc = torch.clamp(x, 0.0, 1.0)
    return xc ** 3 * (10.0 - 15.0 * xc + 6.0 * xc * xc)


def cavity(coords, c):
    """(grid, xi, F, A, S, D) of the frozen cavity point set at `coords`."""
    grid = coords[c.owner] + c.r_vdw[:, None] * c.normals
    natm = coords.shape[0]
    riJ = torch.sqrt(((grid[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    diJ = (riJ - c.R_in_J[None, :]) / c.R_sw_J[None, :]
    own = torch.as_tensor(c.owner)[:, None] == torch.arange(natm)[None, :]
    fiJ = torch.where(own, torch.ones_like(diJ), switch_h(diJ))
    swf = fiJ.prod(1)
    F = swf
    A = c.weights * c.r_vdw ** 2 * swf
    npts = grid.shape[0]
    eye = torch.eye(npts)
    d = grid[:, None, :] - grid[None, :, :]
    r2 = (d * d).sum(-1)
    rij = torch.sqrt(r2 + eye)                       # 1 on the diagonal, exact off it
    xi_ij = c.xi[:, None] * c.xi[None, :] / torch.sqrt(c.xi[:, None] ** 2 + c.xi[None, :] ** 2)
    xr = xi_ij * rij * (1.0 - eye)
    S = torch.erf(xr) / rij
    S = S * (1.0 - eye) + torch.diag(c.xi * math.sqrt(2.0 / math.pi) / F)
    nrij = (d * c.normals[None, :, :]).sum(-1)       # (r_i - r_j) . n_j
    D = (S * nrij / rij ** 2
         - 2.0 * xr / math.sqrt(math.pi) * torch.exp(-xr * xr) * nrij / rij ** 3)
    D = D * (1.0 - eye) + torch.diag(-c.xi * math.sqrt(2.0 / math.pi) / (2.0 * c.r_vdw))
    return grid, F, A, S, D


def pcm_matrices(S, D, A, eps, method):
    """(K, R) of pyscf's PCM for one dielectric constant."""
    npts = S.shape[0]
    eye = torch.eye(npts)
    m = method.upper()
    if m in ('C-PCM', 'CPCM'):
        f = (eps - 1.0) / eps
        return S, -f * eye
    if m == 'COSMO':
        f = (eps - 1.0) / (eps + 0.5)
        return S, -f * eye
    f = (eps - 1.0) / (eps + 1.0)
    DA = D * A[None, :]
    DAS = DA @ S
    if m in ('IEF-PCM', 'IEFPCM'):
        K = S - f / (2.0 * math.pi) * DAS
    elif m == 'SS(V)PE':
        K = S - f / (4.0 * math.pi) * (DAS + DAS.T)
    else:
        raise ValueError(method)
    R = -f * (eye - DA / (2.0 * math.pi))
    return K, R


def response(K, R):
    """Q = sym(K^-1 R): q_sym = Q v, the same symmetrization pyscf and production use."""
    KiR = torch.linalg.solve(K, R)
    return 0.5 * (KiR + KiR.T)


# ---------------------------------------------------------------------------
# the density-fitted mean field in the reaction field
# ---------------------------------------------------------------------------

def df_jk(e3, V, D):
    """Coulomb-metric density-fitted J and K of a symmetric density."""
    nao, naux = e3.shape[0], e3.shape[2]
    rho = torch.einsum('klP,kl->P', e3, D)
    J = torch.einsum('ijP,P->ij', e3, torch.linalg.solve(V, rho))
    B = torch.einsum('ikP,kl->ilP', e3, D).reshape(nao * nao, naux)
    Bt = torch.linalg.solve(V, B.T).T.reshape(nao, nao, naux)
    K = torch.einsum('ilP,ljP->ij', Bt, e3)
    return J, K


def fock_and_energy(D, ints):
    """(F, E_elec, e_pcm) of the DF-RHF in PCM; ints = (h, S, e3, V, Ag, vn, Q).

    Q is None in the gas phase. The density is symmetrized on entry so that
    antisymmetric perturbations are annihilated (see `SCFFixedPoint`).
    """
    h, S, e3, V, Ag, vn, Q = ints
    D = 0.5 * (D + D.T)
    J, K = df_jk(e3, V, D)
    veff = J - 0.5 * K
    F = h + veff
    e_elec = torch.einsum('ij,ij->', D, h) + 0.5 * torch.einsum('ij,ij->', D, veff)
    e_pcm = torch.zeros(())
    if Q is not None:
        v = vn - torch.einsum('ijL,ij->L', Ag, D)
        q = Q @ v
        F = F - torch.einsum('ijL,L->ij', Ag, q)
        e_pcm = 0.5 * (q @ v)
    return F, e_elec + e_pcm, e_pcm


def geneigh(F, S):
    """(eps, C) with F C = S C eps, C^T S C = 1, through a Cholesky reduction."""
    L = torch.linalg.cholesky(S)
    X = torch.linalg.solve_triangular(L, F, upper=False)
    Fp = torch.linalg.solve_triangular(L, X.T, upper=False).T
    w, U = torch.linalg.eigh(0.5 * (Fp + Fp.T))
    C = torch.linalg.solve_triangular(L.T, U, upper=True)
    return w, C


def roothaan_map(D, ints, nocc):
    """g(D) = 2 C_occ C_occ^T from the Fock matrix of D: the SCF fixed-point map."""
    F, _, _ = fock_and_energy(D, ints)
    _, C = geneigh(F, ints[1])
    Co = C[:, :nocc]
    return 2.0 * Co @ Co.T


class SCFFixedPoint(torch.autograd.Function):
    """D* as a differentiable function of the integrals, by the implicit
    function theorem on D = g(D, ints): the forward returns the converged
    density handed in, the backward solves (1 - dg/dD)^T lambda = v with a
    dense Jacobian from autograd and pushes lambda through g onto the integrals.
    """

    @staticmethod
    def forward(ctx, D_star, nocc, *ints):
        ctx.nocc = int(nocc)
        ctx.n_ints = len(ints)
        ctx.save_for_backward(D_star, *[t for t in ints if t is not None])
        ctx.none_mask = [t is None for t in ints]
        return D_star.clone()

    @staticmethod
    def backward(ctx, v):
        saved = list(ctx.saved_tensors)
        D_star = saved.pop(0)
        ints = []
        for is_none in ctx.none_mask:
            ints.append(None if is_none else saved.pop(0))
        n = D_star.shape[0]
        ints_const = [None if t is None else t.detach() for t in ints]
        with torch.enable_grad():
            Dv = D_star.detach().clone().requires_grad_(True)
            Jd = torch.autograd.functional.jacobian(
                lambda Dm: roothaan_map(Dm, ints_const, ctx.nocc), Dv).reshape(n * n, n * n)
            lam = torch.linalg.solve((torch.eye(n * n) - Jd).T, v.reshape(-1)).reshape(n, n)
            leaves = [None if t is None else t.detach().clone().requires_grad_(True)
                      for t in ints]
            g = roothaan_map(D_star.detach(), leaves, ctx.nocc)
            live = [t for t in leaves if t is not None]
            grads = torch.autograd.grad(g, live, grad_outputs=lam, allow_unused=True)
        out, k = [], 0
        for t in leaves:
            if t is None:
                out.append(None)
            else:
                out.append(grads[k] if grads[k] is not None else torch.zeros_like(t))
                k += 1
        return (None, None, *out)


# ---------------------------------------------------------------------------
# the ISDF factors
# ---------------------------------------------------------------------------

def kabsch(ref, coords):
    """Proper Q with (coords - c) ~ (ref - c0) Q, through torch's own SVD."""
    a = (ref - ref.mean(0)).T @ (coords - coords.mean(0))
    u, _, vh = torch.linalg.svd(a)
    d = 1.0 if float(torch.det(u @ vh)) >= 0.0 else -1.0
    return u @ torch.diag(torch.tensor([1.0, 1.0, d])) @ vh


def isdf_points(coords, c):
    """r_g = p_g F_i Q + R_i on the frozen local clouds and frames, Q the body
    frame of `coords` against the reference (`body_frame.BodyFrame`)."""
    frames = c.frames if c.ref_coords is None \
        else c.frames @ kabsch(c.ref_coords, coords)
    fr = frames[c.pt_owner]                          # (M, 3, 3)
    return torch.einsum('gk,gkl->gl', c.pts_local, fr) + coords[c.pt_owner]


def fit_M(Dt, F, reg):
    """`separable_ri.fit_M_stable`: balanced, Tikhonov-regularized least squares."""
    s = torch.sqrt((Dt * Dt).sum(1))
    s = torch.where(s == 0.0, torch.ones_like(s), s)
    d = 1.0 / s
    Dts = Dt * d[:, None]
    G = Dts @ Dts.T + reg * torch.eye(Dt.shape[0])
    L = torch.linalg.cholesky(G)
    X = torch.cholesky_solve((F @ Dts.T).T, L)       # (M, naux)
    return X.T * d[None, :]                           # (naux, M)


def metric_sqrt(V):
    w, U = torch.linalg.eigh(0.5 * (V + V.T))
    keep = w > 1e-12 * w.max()
    return (U[:, keep] * torch.sqrt(w[keep])) @ U[:, keep].T


def isdf_factors(ao_pts, aux_pts, e3, V, c, vtilde):
    """(Mfit, D_dressed, D_bare_or_None) on the frozen layout."""
    mu, nu, w = c.layout
    pairs = ao_pts[:, mu] * ao_pts[:, nu] * w[None, :]
    Dt = torch.cat([pairs, aux_pts], 1)
    naux = V.shape[0]
    F = torch.cat([torch.linalg.solve(V, e3[mu, nu, :].T) * w[None, :], torch.eye(naux)], 1)
    Mfit = fit_M(Dt, F, c.regularization)
    if vtilde is None:
        return Mfit, Mfit.T @ metric_sqrt(V), None
    return Mfit, Mfit.T @ metric_sqrt(V + vtilde), Mfit.T @ metric_sqrt(V)


# ---------------------------------------------------------------------------
# the whole surface
# ---------------------------------------------------------------------------

def surface(coords, c, fz, target='bse', qp_orb=None, pins=None):
    """(Omega or eps^QP, E_0, info) at `coords`, all differentiable.

    pins: a dict to receive intermediates for the pinning table.
    """
    info = {}
    bas, aux = ti.Basis(c.mol, coords), ti.Basis(c.aux, coords)
    S, T = ti.overlap_kinetic(bas)
    Vn = ti.nuclear_attraction(bas, c.charges, coords)
    h = T + Vn
    e3 = ti.three_centre(bas, aux)
    V = ti.two_centre(aux)
    nocc = c.nelec // 2
    if c.solvated:
        grid, Fsw, A, Sc, Dc = cavity(coords, c)
        K_s, R_s = pcm_matrices(Sc, Dc, A, c.eps_static, c.method)
        K_o, R_o = pcm_matrices(Sc, Dc, A, c.eps_opt, c.method)
        Q_s, Q_o = response(K_s, R_s), response(K_o, R_o)
        Ag = ti.pair_potential_on_charges(bas, grid, c.xi)
        vn = ti.nuclear_potential_on_charges(coords, c.charges, grid, c.xi)
        Aaux = ti.single_potential_on_charges(aux, grid, c.xi)
        vtilde = Aaux @ Q_o @ Aaux.T
        ints = (h, S, e3, V, Ag, vn, Q_s)
    else:
        vtilde, ints = None, (h, S, e3, V, None, None, None)
    D_star = SCFFixedPoint.apply(c.D_ref, nocc, *ints)
    F, e_elec, e_pcm = fock_and_energy(D_star, ints)
    eps, C = geneigh(F, ints[1])
    rr = coords[:, None, :] - coords[None, :, :]
    dist = torch.sqrt((rr * rr).sum(-1) + torch.eye(coords.shape[0]))
    e_nuc = 0.5 * ((c.charges[:, None] * c.charges[None, :]) * (1.0 - torch.eye(coords.shape[0])) / dist).sum()
    E0 = e_elec + e_nuc

    crd = isdf_points(coords, c)
    ao_pts, aux_pts = bas.values(crd), aux.values(crd)
    Mfit, Dd, Db = isdf_factors(ao_pts, aux_pts, e3, V, c, vtilde)
    X_mo = ao_pts @ C
    value, chain_info = tc.forward(X_mo, Dd, Db, eps, fz, target=target, qp_orb=qp_orb)
    if pins is not None:
        pins.update(S=S, T=T, Vn=Vn, e3=e3, V=V, F=F, eps=eps, C=C, e_tot=E0,
                    e_pcm=e_pcm, D_star=D_star, Mfit=Mfit, Dd=Dd, Db=Db, X_mo=X_mo,
                    resid=(roothaan_map(D_star, ints, nocc) - D_star))
        if c.solvated:
            pins.update(K_s=K_s, R_s=R_s, K_o=K_o, R_o=R_o, Q_o=Q_o, vtilde=vtilde,
                        grid=grid, Fsw=Fsw, A=A)
    info.update(chain_info)
    return value, E0, info

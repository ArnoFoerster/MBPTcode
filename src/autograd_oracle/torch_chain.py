"""torch.autograd oracle for the solvated ISDF / contour-deformation BSE@GW chain.

The production chain (`src/gradients/excited_state.py`) is a hand-written
reverse pass through

    factors (eps, X_mo, D_dressed, D_bare)
      -> proj(tau) = -2 D^T [G^o(tau) * G^v(tau)] D           (both gauges, static axis)
      -> W(0) = [1 - chi0(0)]^-1, Eq. (18) shift Delta eps_p    (Duchemin 2018)
      -> proj(tau) on the GW axis, bare gauge; chi0(i.nu) on the CD quadrature
      -> wc[nu, q] = B_p[:,q]^T [W(i.nu) - I] B_p[:,q]
      -> Sigma^CD(w) = integral term + explicit residues (C_ov)
      -> Newton root w* of w = eps_p + xc_p + Sigma^CD(w), Z; the root is w* + Delta eps_p
      -> eps^QP on the window, eps + Delta eps outside it
      -> dense Casida on the dressed factor with W_aux = W_dressed(0)
      -> Omega_n

This file evaluates the SAME function in torch, from the same frozen discrete
choices (quasiparticle set, residue sets, Newton seeds, grids), so that
torch.autograd's adjoints on (eps, X, D_dressed, D_bare) can be compared with
the production ones entry by entry. Nothing here is streamed or tiled: it is
an oracle for small systems, not a route.

The Newton root is differentiated in one of two ways, both exact:
  'newton'  a few Newton steps unrolled from the production root, slope by
            autograd (create_graph). At a converged root one step already
            carries the implicit-function-theorem derivative -f_theta / f_w.
  'ift'     w* + Z [F(w*, theta) - F(w*, theta).detach()]: value w*, gradient
            Z dF/dtheta, the closed form production uses.
"""
import math
import os
import time

import numpy as np
import torch

from src.Base.constants import RESIDUE_ON_CONTOUR_TOL
from src.autograd_oracle.frozen import FrozenConventions, dump, load  # noqa: F401

torch.set_default_dtype(torch.float64)

_T0 = time.perf_counter()


def trace(msg):
    if os.environ.get('TORCH_CHAIN_TRACE'):
        print(f'    [torch {time.perf_counter() - _T0:7.2f} s] {msg}', flush=True)


def as_t(a, requires_grad=False):
    t = torch.as_tensor(np.asarray(a, dtype=float)).clone()
    return t.requires_grad_(True) if requires_grad else t


# ---------------------------------------------------------------------------
# the space-time polarizability and the three-index objects
# ---------------------------------------------------------------------------

def polarizability_tau(X, D, eps, nocc, mu, tau):
    """proj(tau) = -2 D^T [G^o * G^v] D, (ntau, naux, naux); mu cancels in Pi."""
    Xo, Xv = X[:, :nocc], X[:, nocc:]
    eo, ev = eps[:nocc] - mu, eps[nocc:] - mu
    out = []
    for t in tau:
        Go = (Xo * torch.exp(eo * t)) @ Xo.T
        Gv = (Xv * torch.exp(-ev * t)) @ Xv.T
        out.append(-2.0 * (D.T @ ((Go * Gv) @ D)))
    return torch.stack(out)


def chi0_from_proj(proj, cosft):
    """chi0 on the frequency axis: fixed transform weights (nfreq, ntau)."""
    return torch.einsum('kt,tPQ->kPQ', cosft, proj)


def three_index_slice(X, D, p):
    """B_p[P, q] = sum_k D[k,P] X[k,p] X[k,q], (naux, norb)."""
    return D.T @ (X[:, p:p + 1] * X)


def three_index_ov(X, D, nocc):
    """C_ov[P, (i,a)] = sum_k D[k,P] X[k,i] X[k,a], (naux, nocc*nvirt), i major."""
    Xo, Xv = X[:, :nocc], X[:, nocc:]
    pair = (Xo[:, :, None] * Xv[:, None, :]).reshape(X.shape[0], -1)
    return D.T @ pair


def ov_energies(eps, nocc):
    return (eps[nocc:][None, :] - eps[:nocc][:, None]).reshape(-1)


def static_screening(X, D, eps, nocc, mu, w_tau, w_cosft):
    """(A, W): A[Q,p] = sum_k D[k,Q] X[k,p]^2 and W = [1 - chi0(0)]^-1."""
    proj = polarizability_tau(X, D, eps, nocc, mu, w_tau)
    chi0 = torch.einsum('t,tPQ->PQ', w_cosft[0], proj)
    eye = torch.eye(chi0.shape[0])
    return D.T @ (X ** 2), torch.linalg.inv(eye - chi0)


def reaction_field_shift(A_d, W_d, A_b, W_b, nocc):
    """Duchemin 2018 Eq. (18): -/+ (1/2) <pp|W[v+vt] - W[v]|pp>, occupied / virtual."""
    sd = ((W_d @ A_d) * A_d).sum(0)
    sb = ((W_b @ A_b) * A_b).sum(0)
    sign = torch.ones(A_d.shape[1])
    sign[:nocc] = -1.0
    return 0.5 * sign * (sd - sb)


# ---------------------------------------------------------------------------
# the contour-deformation self-energy and its Newton root
# ---------------------------------------------------------------------------

def screened_contraction(chi0, Bp):
    """wc[k, q] = Bp[:,q]^T [W(i.nu_k) - I] Bp[:,q], W = [1 - chi0]^-1."""
    eye = torch.eye(chi0.shape[-1])
    WtB = torch.linalg.solve(eye - chi0, Bp.expand(chi0.shape[0], *Bp.shape))
    return torch.einsum('Pq,kPq->kq', Bp, WtB - Bp)


def sigma_int(w, eps, wc, nu, wt):
    """-(1/pi) int dnu sum_q (w - eps_q)/[(w - eps_q)^2 + nu^2] wc[nu, q].

    The quadrature of the Lorentzian plus its singular part in closed form,
    the function `contour_deformation.cd_integral_weights` integrates:
    wc_q(0) h(nu), h = a^2/(a^2 + nu^2), a = 4 x the median node, is taken
    out and integrated exactly ((pi/2) sign(de) a/(a + |de|)), with wc_q(0)
    extrapolated in nu^2 from the two smallest nodes. Written out here
    rather than imported, so autograd differentiates it independently.
    """
    de = w - eps
    g = de[None, :] / (de[None, :] ** 2 + nu[:, None] ** 2)
    plain = (wt @ (g * wc).sum(1))
    nu_np = nu.detach().cpu().numpy()
    k1, k2 = (int(k) for k in np.argsort(nu_np)[:2])
    n1, n2 = float(nu_np[k1]) ** 2, float(nu_np[k2]) ** 2
    wc0 = (n2 * wc[k1] - n1 * wc[k2]) / (n2 - n1)
    a = 4.0 * float(np.median(nu_np))
    h = wt * a * a / (a * a + nu ** 2)
    side = torch.where(torch.abs(de) < RESIDUE_ON_CONTOUR_TOL,
                       torch.zeros_like(de), torch.sign(de))
    err = 0.5 * math.pi * side * a / (a + torch.abs(de)) - h @ g
    return -(plain + (err * wc0).sum()) / math.pi


def sigma_res(w, eps, Bp, C_ov, d, residues):
    """sum_(q, weight) weight Bp[:,q]^T [W(|eps_q - w|) - I] Bp[:,q], explicit W."""
    s = torch.zeros(())
    if not residues:
        return s
    eye = torch.eye(C_ov.shape[0])
    for q, weight in residues:
        freq = torch.abs(eps[q] - w)
        f = 1.0 / (freq - d) - 1.0 / (freq + d)
        chi0 = 2.0 * (C_ov * f) @ C_ov.T
        b = Bp[:, q]
        y = torch.linalg.solve(eye - chi0, b)
        s = s + weight * (b @ y - b @ b)
    return s


def qp_root(p, eps, xc_p, Bp, wc, nu, wt, C_ov, d, residues, w0, mode='newton',
            steps=3):
    """(w*, Z, |f(w*)|) of w = eps_p + xc_p + Sigma^CD(w), residue set frozen.

    xc_p carries <Sigma_x - v_xc>_pp (a constant here). The Eq. (18) shift is
    not part of this equation: production adds it to the root afterwards.
    """
    def sigma(w):
        return sigma_int(w, eps, wc, nu, wt) + sigma_res(w, eps, Bp, C_ov, d,
                                                          residues)

    if mode == 'newton':
        w = torch.tensor(float(w0), requires_grad=True)
        for _ in range(steps):
            S = sigma(w)
            f = w - eps[p] - xc_p - S
            dS = torch.autograd.grad(S, w, create_graph=True)[0]
            w = w - f / (1.0 - dS)
        S = sigma(w)
        dS = torch.autograd.grad(S, w, create_graph=True)[0]
        resid = (w - eps[p] - xc_p - S).detach().abs()
        return w, 1.0 / (1.0 - dS), resid
    if mode == 'ift':
        wl = torch.tensor(float(w0), requires_grad=True)
        S = sigma(wl)
        dS = torch.autograd.grad(S, wl, create_graph=True)[0]
        z = 1.0 / (1.0 - dS)
        F = eps[p] + xc_p + S
        w = wl.detach() + z.detach() * (F - F.detach())
        return w, z, (wl.detach() - F.detach()).abs()
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# the BSE on the dressed factor
# ---------------------------------------------------------------------------

def bse_spectrum(X, D, eps_qp, W_aux, nocc, kappa, tda):
    """Every Casida root, ascending, from (X, D, eps^QP, W_aux) -- `bse_isdf.bse_blocks`
    + `bse_solve` in torch."""
    Xo, Xv = X[:, :nocc], X[:, nocc:]
    no, nv = Xo.shape[1], Xv.shape[1]
    n_ov = no * nv
    B_ov = torch.einsum('kP,ki,ka->Pia', D, Xo, Xv).reshape(-1, n_ov)
    B_oo = torch.einsum('kP,ki,kj->Pij', D, Xo, Xo)
    B_vv = torch.einsum('kP,ka,kb->Pab', D, Xv, Xv)
    v = B_ov.T @ B_ov
    W_ijab = torch.einsum('Pij,PQ,Qab->ijab', B_oo, W_aux, B_vv)
    dd = ov_energies(eps_qp, nocc)
    A = torch.diag(dd) + kappa * v - W_ijab.permute(0, 2, 1, 3).reshape(n_ov, n_ov)
    if tda:
        return torch.linalg.eigvalsh(0.5 * (A + A.T))
    B_vo = torch.einsum('kP,ka,ki->Pai', D, Xv, Xo)
    W_ajbi = torch.einsum('Paj,PQ,Qbi->ajbi', B_vo, W_aux, B_vo)
    Bm = kappa * v - W_ajbi.permute(3, 0, 1, 2).reshape(n_ov, n_ov)
    AmB = 0.5 * ((A - Bm) + (A - Bm).T)
    L = torch.linalg.cholesky(AmB)
    Mh = L.T @ (A + Bm) @ L
    w2 = torch.linalg.eigvalsh(0.5 * (Mh + Mh.T))
    return torch.sqrt(w2)


# ---------------------------------------------------------------------------
# the whole chain
# ---------------------------------------------------------------------------

def forward(X, Dd, Db, eps, fz, target='bse', qp_orb=None, root_mode='newton',
            newton_steps=3):
    """(value, info) of the frozen chain at the factors.

    target: 'bse'    Omega_state of the BSE@GW on the window
            'qp'     eps^QP of `qp_orb`
    Db is None in the gas phase: one factor serves both the self-energy and
    the kernel, and there is no shift.
    """
    nocc, mu = fz.nocc, fz.mu
    norb = eps.shape[0]
    gw_tau, gw_cosft = as_t(fz.gw_tau), as_t(fz.gw_cosft)
    nu, wt = as_t(fz.nu), as_t(fz.wt)
    w_tau, w_cosft = as_t(fz.w_tau), as_t(fz.w_cosft)
    trace('static screening, dressed')
    A_d, W_d = static_screening(X, Dd, eps, nocc, mu, w_tau, w_cosft)
    if Db is None:
        shift = torch.zeros(norb)
        D_sigma = Dd
    else:
        trace('static screening, bare')
        A_b, W_b = static_screening(X, Db, eps, nocc, mu, w_tau, w_cosft)
        shift = reaction_field_shift(A_d, W_d, A_b, W_b, nocc)
        D_sigma = Db
    info = {'shift': shift, 'roots': {}, 'z': {}, 'newton_resid': {}}

    states = fz.states if target != 'qp' else [int(qp_orb)]
    roots, full = {}, {}
    if states:
        trace('proj(tau) on the GW axis')
        proj = polarizability_tau(X, D_sigma, eps, nocc, mu, gw_tau)
        trace('chi0(i.nu)')
        chi0 = chi0_from_proj(proj, gw_cosft)
        d = ov_energies(eps, nocc)
        need_cov = any(fz.residues.get(p) for p in states)
        C_ov = three_index_ov(X, D_sigma, nocc) if need_cov else None
        for p in states:
            trace(f'state {p}: slice, wc')
            Bp = three_index_slice(X, D_sigma, p)
            wc = screened_contraction(chi0, Bp)
            trace(f'state {p}: Newton')
            xc_p = float(fz.xc_const[p])
            w, z, resid = qp_root(p, eps, xc_p, Bp, wc, nu, wt, C_ov, d,
                                  fz.residues.get(p, []), fz.roots[p],
                                  mode=root_mode, steps=newton_steps)
            roots[p] = w
            full[p] = w + shift[p]
            info['roots'][p], info['z'][p] = w, z
            info['newton_resid'][p] = float(resid)
    if target == 'qp':
        return full[int(qp_orb)], info

    # eps^QP: the window's roots; outside it the mean field plus the frozen
    # scissor, a constant, and in a continuum the orbital's own Eq. (18) shift
    cols = []
    for p in range(norb):
        if p in roots:
            cols.append(full[p])
        elif fz.outside_window[p] and Db is not None:
            cols.append(eps[p] + float(fz.scissor[p]) + shift[p])
        elif fz.outside_window[p]:
            cols.append(eps[p] + float(fz.scissor[p]))
        else:
            cols.append(eps[p])
    eps_qp = torch.stack(cols)
    info['eps_qp'] = eps_qp
    trace('Casida')
    om = bse_spectrum(X, Dd, eps_qp, W_d, nocc, fz.kappa, fz.bse_tda)
    info['omega'] = om
    return om[fz.state], info


def adjoints(X, Dd, Db, eps, fz, **kw):
    """(value, info, {eps, x, d, d_bare}) -- torch.autograd's adjoints of the
    target with respect to the four factors."""
    leaves = [eps, X, Dd] + ([] if Db is None else [Db])
    for t in leaves:
        t.requires_grad_(True)
    value, info = forward(X, Dd, Db, eps, fz, **kw)
    trace('autograd.grad')
    grads = torch.autograd.grad(value, leaves)
    trace('done')
    out = {'eps': grads[0].numpy(), 'x': grads[1].numpy(), 'd': grads[2].numpy(),
           'd_bare': None if Db is None else grads[3].numpy()}
    return float(value), info, out


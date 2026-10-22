"""Nuclear force of a density-fitted correlated energy on an RHF reference,
for any method that hands over its derivatives -- the method-general engine
(the dense reference: every tensor whole, small molecules).

A correlated energy E_c on RHF orbitals is described to this module by

    gamma_pq = dE_c / df_pq           (active blocks, the Fock matrix)
    Gamma^P_pq = dE_c / dB^P_pq       (the RI factors of the correlation fit,
                                       B = J X^T, X = L^-1, V = L L^T)
    Y_extra (nmo, nmo)                dE_c / dkappa_rp of any further orbital
                                       dependence (a reaction field's LR term)

with E_c depending on B only through sum_P B^P_pq B^P_rs. MP2 is here
(`mp2_pieces`); ADC(2) in src.SingleReference.ADC.eeADC.ee_gradient.

The orbitals. For C -> C (1 + kappa) (kappa general), dL/dkappa_rp = Y_rp,

    Y_rp = sum_qP Gs2^P_pq B^P_rq + 2 (f gamma)_rp + [p occupied] 4 G[gamma_AO]_rp
           + Y_extra_rp,

Gs2 = Gamma + Gamma^T over (pq), G[D] = J[D] - K[D]/2 in the SCF's JK fit.
L = E_c + sum z_ai f_ai + sum z'_ci f_ci (frozen core c) is made stationary by
the Z-vector (src.Base.utils.cphf) and z'_ci = (Y_ic - Y_ci) / (e_c - e_i);
gamma then carries (z + z^T)/2 and (z' + z'^T)/2, gamma_AO = C gamma C^T is
dE_c/dh (the relaxed correlation density), and Y is symmetric (returned as a
check). The overlap enters through W = (Y + Y^T)/4.

The force of E_c at fixed orbitals, the orbitals' response through W:
    sum gamma_AO dh/dR - sum W_AO dS/dR             (one-electron skeleton)
  + sum gamma_AO G'[D_HF]                           (the SCF's JK fit)
  + 2 sum Gamma3 dJ - sum Gamma2 dV                 (the correlation fit)
    Gamma3_pq,Q = (1/2) sum_P Gs^P_pq X_PQ,  Gamma2 = K^T Gamma3,  K = J V^-1,
with Gs = Gs2 / 2 and D_HF = 2 C_o C_o^T. dh/dR is the VACUUM core Hamiltonian's:
an external field (src.Base.solvation_loop) adds its own derivative through
the medium (src.Base.reaction_field), so the engine is the same in the gas
phase and in a fixed field.
"""
import numpy as np
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import scf

from src.Base.constants import CORRELATION_CPHF_TOL, CPHF_MAX_CYCLE
from src.Base.numerics import Numerics
from src.Base.utils import cphf
from src.gradients.df_assembly import two_electron_skeleton_df
from src.gradients.grad_engine import one_electron_skeleton

PAIR = (0, 3, 2, 1)


class RIFactors:
    """Whole RI factors of every MO pair in one fit: J (nmo, nmo, naux)
    = (pq|P), V the metric, X = L^-1 (V = L L^T), B = J X^T."""

    def __init__(self, mol, mf, auxbasis):
        self.auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis)
        J = pyscf_df.incore.aux_e2(mol, self.auxmol, intor='int3c2e', aosym='s1')
        J = J.reshape(mol.nao, mol.nao, self.auxmol.nao)
        C = np.asarray(mf.mo_coeff)
        self.J = np.einsum('mp,nq,mnP->pqP', C, C, J, optimize=True)
        self.V, self.X = whitening(self.auxmol)
        self.B = np.einsum('pqQ,PQ->pqP', self.J, self.X, optimize=True)


def whitening(auxmol):
    """(V, X = L^-1) of the auxiliary metric V = L L^T."""
    V = auxmol.intor('int2c2e')
    Lc = scipy.linalg.cholesky(V, lower=True)
    return V, scipy.linalg.solve_triangular(Lc, np.eye(Lc.shape[0]), lower=True)


def doubles_denominator(eo, ev, rows=None):
    """D_kcld = e_c + e_d - e_k - e_l, k over `rows` (default eo) and l over eo."""
    rows = eo if rows is None else rows
    return (ev[None, :, None, None] + ev[None, None, None, :]
            - rows[:, None, None, None] - eo[None, None, :, None])


def amplitude_fock(lam, A):
    """dE/df of an amplitude A = N / D (D = e_a + e_b - e_i - e_j) whose
    adjoint over D is lam = (dE/dA) / D, symmetric under the pair swap:
    (oo, vv)."""
    oo = (np.einsum('kcld,jcld->kj', lam, A)
          + np.einsum('lckd,lcjd->kj', lam, A))
    vv = -(np.einsum('kcld,kbld->cb', lam, A)
           + np.einsum('kdlc,kdlb->cb', lam, A))
    return oo, vv


def tile_amplitude_fock(lam, A):
    """amplitude_fock on one tile of rows, the swap symmetry of lam and A
    putting both terms of each block on the tile's first index."""
    oo = 2.0 * np.einsum('lckd,lcjd->kj', lam, A, optimize=True)
    vv = -2.0 * np.einsum('kcld,kbld->cb', lam, A, optimize=True)
    return oo, vv


def mp2_pieces(B, eps, nf, nocc):
    """(e_mp2, gamma (nmo, nmo), Gamma (nmo, nmo, naux)) of closed-shell
    DF-MP2 with nf frozen core orbitals: E = -sum I tau, tau = (2 I - I~) / D,
    D = e_a + e_b - e_i - e_j > 0."""
    nmo, naux = B.shape[0], B.shape[2]
    o, v = slice(nf, nocc), slice(nocc, nmo)
    L = B[o, v]
    I = np.einsum('iaP,jbP->iajb', L, L)
    D = doubles_denominator(eps[o], eps[v])
    tau = (2.0 * I - I.transpose(PAIR)) / D
    e = -float(np.sum(I * tau))
    g_oo, g_vv = amplitude_fock(-I / D, tau)
    gamma = np.zeros((nmo, nmo))
    gamma[o, o], gamma[v, v] = 0.5 * (g_oo + g_oo.T), 0.5 * (g_vv + g_vv.T)
    Gam = np.zeros((nmo, nmo, naux))
    Gam[o, v] = -4.0 * np.einsum('iajb,jbP->iaP', tau, L)
    return e, gamma, Gam


def _jk(mf):
    """get_jk of mf's own interaction (its JK fit), no environment."""
    m = scf.RHF(mf.mol).density_fit(auxbasis=mf.with_df.auxbasis)
    m.with_df = mf.with_df
    return m.get_jk


def relax(mf, mol, B, gamma0, Gam, nf, Y_extra=None, tol=CORRELATION_CPHF_TOL,
          max_cycle=CPHF_MAX_CYCLE):
    """(gamma, Y): the relaxed gamma (MO, multipliers included) and the
    orbital gradient of the stationary Lagrangian (module docstring)."""
    YG = np.einsum('pqP,rqP->rp', Gam + Gam.transpose(1, 0, 2), B, optimize=True)
    if Y_extra is not None:
        YG = YG + Y_extra
    return relax_gradient(mf, mol, YG, gamma0, nf, tol, max_cycle)


def relax_gradient(mf, mol, YG, gamma0, nf, tol=CORRELATION_CPHF_TOL,
                   max_cycle=CPHF_MAX_CYCLE):
    """relax from the Gamma part of the orbital gradient, YG_rp = sum_qP
    Gs2^P_pq B^P_rq (+ Y_extra), however it was assembled."""
    C = np.asarray(mf.mo_coeff)
    eps = np.asarray(mf.mo_energy, float)
    nmo, nocc = C.shape[1], mol.nelectron // 2
    get_jk = _jk(mf)

    def Y_of(gam):
        vj, vk = get_jk(mol, C @ gam @ C.T, hermi=1)
        Y = YG + 2.0 * eps[:, None] * gam
        Y[:, :nocc] += 4.0 * (C.T @ (vj - 0.5 * vk) @ C)[:, :nocc]
        return Y

    gamma = gamma0.copy()
    if nf:
        zc = ((YG[nf:nocc, :nf].T - YG[:nf, nf:nocc])
              / (eps[:nf, None] - eps[None, nf:nocc]))
        gamma[:nf, nf:nocc] += 0.5 * zc
        gamma[nf:nocc, :nf] += 0.5 * zc.T
    Y0 = Y_of(gamma)
    Xvo = Y0[nocc:, :nocc] - Y0[:nocc, nocc:].T
    Co, Cv = C[:, :nocc], C[:, nocc:]

    def fvind(z):
        z = z.reshape(nmo - nocc, nocc)
        dm = Cv @ z @ Co.T
        vj, vk = get_jk(mol, dm + dm.T, hermi=1)
        return (Cv.T @ (2.0 * vj - vk) @ Co).ravel()
    e_ai = eps[nocc:, None] - eps[None, :nocc]
    z = cphf.solve(fvind, e_ai, -Xvo, numerics=Numerics(
        scf={'cphf_tol': tol, 'cphf_max_cycle': max_cycle}))
    z = z.reshape(nmo - nocc, nocc)
    gamma[nocc:, :nocc] += 0.5 * z
    gamma[:nocc, nocc:] += 0.5 * z.T
    return gamma, Y_of(gamma)


def _jk_skeleton(mol, mf, Dc, Dhf):
    """(auxmol, Gamma3, Gamma2) of sum Dc G'[Dhf] in the SCF's JK fit for
    two_electron_skeleton_df: Gamma4 = Dc (x) Dhf - (1/2) exchange,
    symmetrized over both pairs and the pair swap."""
    auxmol = pyscf_df.addons.make_auxmol(mol, mf.with_df.auxbasis)
    J = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    J = J.reshape(mol.nao, mol.nao, auxmol.nao)
    V = auxmol.intor('int2c2e')
    K = np.linalg.solve(V, J.reshape(-1, auxmol.nao).T).T.reshape(J.shape)
    cD = np.einsum('mnQ,mn->Q', K, Dc)
    cP = np.einsum('mnQ,mn->Q', K, Dhf)
    G3 = 0.5 * (Dc[:, :, None] * cP + Dhf[:, :, None] * cD)
    ex = np.einsum('ml,lsQ,sn->mnQ', Dc, K, Dhf, optimize=True)
    G3 -= 0.25 * (ex + ex.transpose(1, 0, 2))
    G2 = np.einsum('mnP,mnQ->PQ', K, G3, optimize=True)
    return auxmol, G3, 0.5 * (G2 + G2.T)


def correlation_force(mol, mf, ri, gamma0, Gam, nf, Y_extra=None):
    """(force (natm, 3) of E_c, gamma (MO, relaxed), max |Y - Y^T|)."""
    C = np.asarray(mf.mo_coeff)
    nocc = mol.nelectron // 2
    gamma, Y = relax(mf, mol, ri.B, gamma0, Gam, nf, Y_extra)
    asym = float(np.max(np.abs(Y - Y.T)))
    W = 0.25 * (Y + Y.T)
    gamma_ao, W_ao = C @ gamma @ C.T, C @ W @ C.T
    f1 = one_electron_skeleton(mol, mf, [gamma_ao], [-W_ao])[0]
    Dhf = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    auxjk, G3jk, G2jk = _jk_skeleton(mol, mf, gamma_ao, Dhf)
    fjk = two_electron_skeleton_df(mol, auxjk, [G3jk], [G2jk])[0]
    Gs = 0.5 * (Gam + Gam.transpose(1, 0, 2))
    G3 = 0.5 * np.einsum('pqP,PQ->pqQ', Gs, ri.X, optimize=True)
    K = np.linalg.solve(ri.V, ri.J.reshape(-1, ri.V.shape[0]).T).T.reshape(ri.J.shape)
    G2 = np.einsum('pqP,pqQ->PQ', K, G3, optimize=True)
    G3_ao = np.einsum('mp,nq,pqQ->mnQ', C, C, G3, optimize=True)
    fri = two_electron_skeleton_df(mol, ri.auxmol, [G3_ao], [0.5 * (G2 + G2.T)])[0]
    return f1 + fjk + fri, gamma, asym


def hf_force(mf):
    """pyscf's force of the SCF energy E_vac[D] + tr(h_ext D) with the
    external operator's own derivative left out (its vacuum hcore_generator),
    nuclear repulsion included. The mean field's own gradient class
    (mf.nuc_grad_method()): RHF or DF-RHF, and for a Kohn-Sham reference the
    RKS class with its exchange-correlation term (RHF's class there would
    differentiate J - K/2 and no XC)."""
    g = mf.nuc_grad_method()
    g.verbose = 0
    return g.kernel()


def mp2_force(mf, mol, frozen, auxbasis, excited=None):
    """(dE/dR (natm, 3), (e_mp2, gamma (MO, relaxed), max |Y - Y^T|, extra)) of
    E_HF + E_MP2 on the dense reference; excited(C, nocc, nmo) -> (gamma,
    Gamma, Y_extra, extra) adds an excitation's pieces in the full MO layout."""
    C = np.asarray(mf.mo_coeff)
    nmo, nocc = C.shape[1], mol.nelectron // 2
    ri = RIFactors(mol, mf, auxbasis)
    e2, gamma, Gam = mp2_pieces(ri.B, mf.mo_energy, frozen, nocc)
    Y_extra = extra = None
    if excited is not None:
        g1, G1, Y_extra, extra = excited(C, nocc, nmo)
        gamma, Gam = gamma + g1, Gam + G1
    fc, gam, asym = correlation_force(mol, mf, ri, gamma, Gam, frozen, Y_extra)
    return hf_force(mf) + fc, (e2, gam, asym, extra)

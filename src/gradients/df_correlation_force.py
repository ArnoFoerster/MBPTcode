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
(`mp2_pieces`); ADC(2) in src.gradients.adc.ee_gradient.

The orbitals. For C -> C (1 + kappa) (kappa general), dL/dkappa_rp = Y_rp,

    Y_rp = sum_qP Gs2^P_pq B^P_rq + 2 (f gamma)_rp + [p occupied] 4 G[gamma_AO]_rp
           + Y_extra_rp,

Gs2 = Gamma + Gamma^T over (pq), G[D] = J[D] - K[D]/2 in the SCF's JK fit.
L = E_c + sum Lambda_pq f_pq over the virtual-occupied and frozen-core-active
pairs is made stationary by the one Z-vector entry of every gradient
(`multipliers.solve_orbital_multipliers`, Y[Lambda] from
`multipliers.rotation_gradient`); the active occupied and virtual pairs need
no multiplier, so Y is symmetric there by construction (returned as a check).
gamma then carries Lambda, gamma_AO = C gamma C^T is dE_c/dh (the relaxed
correlation density), and the overlap enters through W = (Y + Y^T)/4.

The force of E_c at fixed orbitals, the orbitals' response through W:
    sum gamma_AO dh/dR - sum W_AO dS/dR             (one-electron skeleton)
  + sum gamma_AO G'[D_HF]                           (the SCF's JK fit,
                                                     `skeleton_tiles.fitted_fock_skeleton`)
  + 2 sum Gamma3 dJ - sum Gamma2 dV                 (the correlation fit)
    Gamma3_pq,Q = (1/2) sum_P Gs^P_pq X_PQ,  Gamma2 = K^T Gamma3,  K = J V^-1,
with Gs = Gs2 / 2 and D_HF = 2 C_o C_o^T. dh/dR is the VACUUM core Hamiltonian's:
an external field (src.environments.solvation_loop) adds its own derivative through
the medium (src.environments.reaction_field), so the engine is the same in the gas
phase and in a fixed field.
"""
import numpy as np
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import scf

from src.Base.pyscf.pyscf_interface import auxmol_of
from src.SingleReference.common.df_amplitudes import (PAIR, amplitude_fock,
                                                      doubles_denominator)
from src.gradients.df_assembly import fit_coefficients, two_electron_skeleton_df
from src.gradients.grad_engine import one_electron_skeleton
from src.gradients.multipliers import rotation_gradient, solve_orbital_multipliers
from src.gradients.skeleton_tiles import fitted_fock_skeleton


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


def jk_fit(mf):
    """The auxiliary basis of mf's JK fit. A correlation force takes the
    Fock skeleton of that fit, so a mean field without one is refused."""
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError('the DF correlation force needs a density-fitted mean '
                         'field (mf.density_fit()): its JK skeleton is the fit\'s, '
                         f'and this {type(mf).__name__} has none')
    return with_df.auxbasis


def scf_kernel(mf):
    """G(x) = J[x] - K[x]/2 of mf's own interaction (its JK fit, or the exact
    integrals when it has none), no environment, built serially whatever the
    mean field's distribution."""
    with_df = getattr(mf, 'with_df', None)
    m = scf.RHF(mf.mol)
    if with_df is not None:
        m = m.density_fit(auxbasis=with_df.auxbasis)
        m.with_df = with_df

    def kernel(x):
        vj, vk = m.get_jk(mf.mol, x, hermi=1)
        return vj - 0.5 * vk
    return kernel


def relax(mf, mol, B, gamma0, Gam, nf, Y_extra=None, numerics=None, flags=None):
    """(gamma, Y): the relaxed gamma (MO, multipliers included) and the
    orbital gradient of the stationary Lagrangian (module docstring)."""
    YG = np.einsum('pqP,rqP->rp', Gam + Gam.transpose(1, 0, 2), B, optimize=True)
    if Y_extra is not None:
        YG = YG + Y_extra
    return relax_gradient(mf, mol, YG, gamma0, nf, numerics, flags)


def relax_gradient(mf, mol, YG, gamma0, nf, numerics=None, flags=None):
    """relax from the Gamma part of the orbital gradient, YG_rp = sum_qP
    Gs2^P_pq B^P_rq (+ Y_extra), however it was assembled: the multipliers of
    the virtual-occupied and frozen-core-active pairs (module docstring),
    their tolerances from `numerics`, a short solve flagged into `flags`."""
    C = np.asarray(mf.mo_coeff)
    eps = np.asarray(mf.mo_energy, float)
    nmo, nocc = C.shape[1], mol.nelectron // 2
    kernel, fock = scf_kernel(mf), np.diag(eps)

    def Y_of(gam):
        return YG + rotation_gradient(C, fock, gam, nocc, kernel)
    free = np.zeros((nmo, nmo), bool)
    free[:nocc, nocc:] = True
    free[:nf, nf:nocc] = True
    Lam, _ = solve_orbital_multipliers(
        lambda L: rotation_gradient(C, fock, L, nocc, kernel), Y_of(gamma0), eps,
        numerics=numerics, free=free, flags=flags)
    gamma = gamma0 + Lam
    return gamma, Y_of(gamma)


def correlation_force(mol, mf, ri, gamma0, Gam, nf, Y_extra=None, numerics=None,
                      flags=None):
    """(force (natm, 3) of E_c, gamma (MO, relaxed), max |Y - Y^T|)."""
    C = np.asarray(mf.mo_coeff)
    nocc = mol.nelectron // 2
    gamma, Y = relax(mf, mol, ri.B, gamma0, Gam, nf, Y_extra, numerics, flags)
    asym = float(np.max(np.abs(Y - Y.T)))
    W = 0.25 * (Y + Y.T)
    gamma_ao, W_ao = C @ gamma @ C.T, C @ W @ C.T
    f1 = one_electron_skeleton(mol, mf, [gamma_ao], [-W_ao])[0]
    Co = C[:, :nocc]
    jk_fit(mf)                     # refuses a mean field with no JK fit
    auxjk = auxmol_of(mf, mol)
    fjk = fitted_fock_skeleton(mol, auxjk, gamma_ao, 2.0 * Co @ Co.T,
                               occ=np.sqrt(2.0) * Co, exchange=1.0)
    Gs = 0.5 * (Gam + Gam.transpose(1, 0, 2))
    G3 = 0.5 * np.einsum('pqP,PQ->pqQ', Gs, ri.X, optimize=True)
    G2 = np.einsum('pqP,pqQ->PQ', fit_coefficients(ri.J, ri.V), G3, optimize=True)
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


def mp2_force(mf, mol, frozen, auxbasis, excited=None, numerics=None, flags=None):
    """(dE/dR (natm, 3), (e_mp2, gamma (MO, relaxed), max |Y - Y^T|, extra)) of
    E_HF + E_MP2 on the dense reference; excited(C, nocc, nmo) -> (gamma,
    Gamma, Y_extra, extra) adds an excitation's pieces in the full MO layout.
    numerics: the caller's (the Z-vector's tolerances); flags: the list a
    short Z-vector solve appends its flag to."""
    jk_fit(mf)                     # refuses a mean field with no JK fit
    C = np.asarray(mf.mo_coeff)
    nmo, nocc = C.shape[1], mol.nelectron // 2
    ri = RIFactors(mol, mf, auxbasis)
    e2, gamma, Gam = mp2_pieces(ri.B, mf.mo_energy, frozen, nocc)
    Y_extra = extra = None
    if excited is not None:
        g1, G1, Y_extra, extra = excited(C, nocc, nmo)
        gamma, Gam = gamma + g1, Gam + G1
    fc, gam, asym = correlation_force(mol, mf, ri, gamma, Gam, frozen, Y_extra,
                                      numerics, flags)
    return hf_force(mf) + fc, (e2, gam, asym, extra)

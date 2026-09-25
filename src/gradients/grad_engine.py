"""Orbital-response and gradient-assembly engine for the Toelle-route pilot.

A correlation-type target energy E_X is handed over as its exact partials
w.r.t. the MO-basis Fock matrix and ERIs,

    E_X ~ sum_pq gammaF_pq F_pq + (1/2-free convention) sum Gamma4_pqrs (pq|rs),

i.e. gammaF = dE_X/dF_pq and Gamma4 = dE_X/d(pq|rs) as raw coefficient
tensors (no symmetrization assumed; contraction targets are symmetric).
The engine then:

  1. folds the Fock dependence into pure (h, ERI) densities
     [F_pq = h_pq + sum_k 2(pq|kk) - (pk|qk)],
  2. builds the orbital-rotation gradient Y_up (dE = sum_up X_up Y_up for
     dC = C X),
  3. solves the multiplier equation antisym(Y_E + Y_fold(Lambda)) = 0 with
     the closed-form CPHF-structured matvec
        Y[fold(Lambda)] = 2 F Lambda + occupied-column 2e-response terms,
  4. forms M = -(Y_tot + Y_tot^T)/2,
  5. contracts the resulting densities against skeleton derivative integrals,
     in the AO basis for exact ERIs or through a three-index auxiliary
     factorization for RI ones.

Step 1 is the Hartree-Fock Fock and step 3 the Hartree-Fock response kernel, so
a Kohn-Sham reference is routed to `kohn_sham_correlation_gradients`, which
takes both from the mean field itself.

Total gradient = the mean field's own force + dL/dR.
"""
import numpy as np
from pyscf import ao2mo
from pyscf import grad as _pyscf_grad  # registers mf.Gradients()

from src.Base.constants import DERIV_BLOCK_BYTES, ORBITAL_MULTIPLIER_TOL
from src.Base.eri_blocks import MOEriBlocks
from src.SingleReference.LinearResponse.rpa_energy import xc_hybrid_coeff
from src.gradients.df_assembly import df_densities, two_electron_skeleton_df
from src.gradients.isdf_derivatives import (fock_partial_Y,
                                            fock_partial_skeleton, solve_lambda)
from src.gradients.multipliers import solve_orbital_multipliers


# ---------------------------------------------------------------------------
# Fock folding and orbital gradient
# ---------------------------------------------------------------------------

def fold_fock(gammaF, nocc, Gamma4=None):
    """(gamma_h, Gamma4) equivalent of a dE/dF_pq partial.

    Accumulates into Gamma4 when one is given, so the assembly never holds more
    than one (norb^4) tensor -- at nao 96 that is the difference between 0.7 GB
    and 2.7 GB, and it is the array that decides where the dense route stops.
    """
    norb = gammaF.shape[0]
    if Gamma4 is None:
        Gamma4 = np.zeros((norb,) * 4)
    # F_pq = h_pq + sum_k [2 (pq|kk) - (pk|qk)]
    for k in range(nocc):
        Gamma4[:, :, k, k] += 2.0 * gammaF
        Gamma4[:, k, :, k] -= gammaF
    return gammaF.copy(), Gamma4


def orbital_Y(gamma_h, Gamma4, h_mo, eri_mo):
    """Y_up such that dE = sum_up X_up Y_up under dC = C X (X arbitrary)."""
    Y = h_mo @ (gamma_h + gamma_h.T)                          # h-part: h(gamma + gamma^T)
    Y += np.einsum('uqrs,pqrs->up', eri_mo, Gamma4, optimize=True)
    Y += np.einsum('qurs,qprs->up', eri_mo, Gamma4, optimize=True)
    Y += np.einsum('qrus,qrps->up', eri_mo, Gamma4, optimize=True)
    Y += np.einsum('qrsu,qrsp->up', eri_mo, Gamma4, optimize=True)
    return Y


def _y_fold_lambda(Lam, F_mo, eri_mo, nocc):
    """Y[fold(Lambda)] in closed form (Lambda symmetric, zero diagonal).

    = 2 F Lambda
      + occupied columns k:  4 VL_uk - 2 W2_uk
    with VL_uk = sum_qr (qr|uk) Lambda_qr, W2_uk = sum_qr (qu|rk) Lambda_qr.
    """
    norb = F_mo.shape[0]
    Y = 2.0 * (F_mo @ Lam)
    occ = slice(0, nocc)
    VL = np.einsum('qruk,qr->uk', eri_mo[:, :, :, occ], Lam, optimize=True)
    W2 = np.einsum('qurk,qr->uk', eri_mo[:, :, :, occ], Lam, optimize=True)
    Y[:, occ] += 4.0 * VL - 2.0 * W2
    return Y


def solve_multipliers(Y_E, F_mo, eri_mo, nocc, eps,
                      tol=ORBITAL_MULTIPLIER_TOL, verbose=False):
    """Solve antisym(Y_E + Y[fold(Lambda)]) = 0 for symmetric zero-diag Lambda.

    The dense route's entry to `solve_orbital_multipliers`: the matvec is the
    closed-form `_y_fold_lambda` built from the four-index MO integral tensor.
    Returns (Lambda, res_norm).
    """
    return solve_orbital_multipliers(
        lambda Lam: _y_fold_lambda(Lam, F_mo, eri_mo, nocc),
        Y_E, eps, tol=tol, verbose=verbose)


# ---------------------------------------------------------------------------
# skeleton derivative contraction (AO basis, one integral sweep)
# ---------------------------------------------------------------------------

def ao_backtransform(Gamma4, C):
    """MO two-particle coefficient tensor -> AO, one index at a time."""
    G = np.tensordot(C, Gamma4, axes=(1, 0))          # (u, q, r, s)
    G = np.tensordot(C, G, axes=(1, 1)).transpose(1, 0, 2, 3)
    G = np.tensordot(C, G, axes=(1, 2)).transpose(1, 2, 0, 3)
    return np.tensordot(C, G, axes=(1, 3)).transpose(1, 2, 3, 0)


def _permutation_sum(G_ao):
    """Gbar_abcd collecting the four index positions a derivative can sit in.

    d(pq|rs)/dR_A picks up one term per centre; each becomes a first-index
    contraction of (grad mu nu|lam sig) once the tensor is permuted, so all
    four fold into a single sweep of int2e_ip1.
    """
    return (G_ao + G_ao.transpose(1, 0, 2, 3)
            + G_ao.transpose(2, 3, 0, 1) + G_ao.transpose(3, 2, 0, 1))


def _shell_blocks(mol, max_bytes):
    """Shell ranges whose (grad mu nu|lam sig) block fits in max_bytes."""
    ao_loc = mol.ao_loc_nr()
    nao = mol.nao
    per_ao = 3 * nao ** 3 * 8
    blocks = []
    sh0 = 0
    while sh0 < mol.nbas:
        sh1 = sh0 + 1
        # int() because ao_loc_nr() is int32 and per_ao is 3 nao^3 * 8 bytes:
        # 3.6e7 at 114 basis functions, 1.4e8 at 180. The product overflows
        # int32 past 60 and 15 AOs respectively, wraps negative, and the test
        # then passes for every block -- so the loop runs to the end of the
        # molecule and asks for the whole tensor instead of a chunk of it.
        while (sh1 < mol.nbas
               and int(ao_loc[sh1 + 1] - ao_loc[sh0]) * per_ao <= max_bytes):
            sh1 += 1
        blocks.append((sh0, sh1))
        sh0 = sh1
    return blocks


def two_electron_skeleton(mol, Gbar_list, max_bytes=DERIV_BLOCK_BYTES):
    """sum_abcd Gbar_abcd d(ab|cd)/dR_A for every target, one integral sweep.

    The derivative integrals are contracted where they are made -- in the AO
    basis, blocked over the shells of the differentiated index -- so the cost
    is one N^4 pass per Cartesian direction instead of a full N^5 MO transform
    per atom and direction.
    """
    nao, nbas = mol.nao, mol.nbas
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()
    per_ao = np.zeros((len(Gbar_list), 3, nao))
    for sh0, sh1 in _shell_blocks(mol, max_bytes):
        p0, p1 = ao_loc[sh0], ao_loc[sh1]
        # pyscf's int2e_ip1 is -(grad mu nu|lam sig)
        blk = -mol.intor('int2e_ip1', comp=3,
                         shls_slice=(sh0, sh1, 0, nbas, 0, nbas, 0, nbas))
        blk = blk.reshape(3, p1 - p0, nao, nao, nao)
        for k, Gbar in enumerate(Gbar_list):
            per_ao[k, :, p0:p1] = np.einsum('xabcd,abcd->xa', blk,
                                            Gbar[p0:p1], optimize=True)
    out = np.zeros((len(Gbar_list), mol.natm, 3))
    for iatm in range(mol.natm):
        p0, p1 = aoslices[iatm, 2:]
        out[:, iatm, :] = per_ao[:, :, p0:p1].sum(axis=2)
    return out


def one_electron_skeleton(mol, mf, gamma_ao_list, M_ao_list):
    """sum_ab gamma_ab dh_ab/dR_A + sum_ab M_ab dS_ab/dR_A for every target."""
    hgen = mf.Gradients().hcore_generator(mol)
    s1 = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    out = np.zeros((len(gamma_ao_list), mol.natm, 3))
    for iatm in range(mol.natm):
        p0, p1 = aoslices[iatm, 2:]
        dh_all = hgen(iatm)
        for xyz in range(3):
            dS = np.zeros((mol.nao, mol.nao))
            dS[p0:p1, :] += s1[xyz, p0:p1, :]
            dS[:, p0:p1] += s1[xyz, p0:p1, :].T
            for k, (g_ao, M_ao) in enumerate(zip(gamma_ao_list, M_ao_list)):
                out[k, iatm, xyz] = (np.einsum('ab,ab->', g_ao, dh_all[xyz])
                                     + np.einsum('ab,ab->', M_ao, dS))
    return out


# ---------------------------------------------------------------------------
# top-level correlation-gradient driver
# ---------------------------------------------------------------------------

def kohn_sham_correlation_gradients(mol, mf, target_list, eri_mo, auxmol=None,
                                    verbose=False):
    """dL_X/dR with the Fock partial taken on the REFERENCE'S OWN Fock.

    `fold_fock` writes F_pq = h_pq + sum_k [2(pq|kk) - (pk|qk)], which is the
    Hartree-Fock Fock, and `_y_fold_lambda` is its response kernel. A Kohn-Sham
    orbital energy is h + 2J - a_x K + v_xc, so folding dE/dF that way
    differentiates a different one-body matrix from the one that produced the
    orbital energies the target was built on, and the multiplier equation then
    enforces a stationarity the reference does not satisfy.

    The Fock partial and the multiplier therefore go through `fock_partial_Y`
    and `fock_partial_skeleton`, which carry the reference's exchange fraction,
    its exchange-correlation potential and its response kernel; only the
    target's explicit (pq|rs) partial goes through the four-index sweep. On a
    Hartree-Fock reference this reproduces the folded route bitwise, which is
    why that route is kept rather than replaced, it being the oracle every
    cubic gate is measured from.

    The exchange-correlation double counting of the TARGET's own energy is not
    here: E_0 = E_HF[rho] rather than E_KS is the surface's declaration, and it
    reaches this assembly as the Y_extra of its target and is added to the
    skeleton by its caller.
    """
    C = mf.mo_coeff
    nocc = mol.nelectron // 2
    h_mo = C.T @ mf.get_hcore() @ C
    gam, Ms, four, diags = [], [], [], []
    for target in target_list:
        gammaF, Gamma4 = target[0], target[1]
        Y_E = orbital_Y(np.zeros_like(gammaF), Gamma4, h_mo, eri_mo)
        Y_E = Y_E + fock_partial_Y(mf, gammaF, nocc)
        if len(target) > 2 and target[2] is not None:
            Y_E = Y_E + target[2]
        Lam, res = solve_lambda(mf, Y_E, nocc, verbose=verbose)
        Y_tot = Y_E + fock_partial_Y(mf, Lam, nocc)
        Mmat = -0.25 * (Y_tot + Y_tot.T)
        gam.append(0.5 * (gammaF + gammaF.T) + Lam)
        Ms.append(Mmat)
        four.append(Gamma4)
        diags.append({'stationarity': np.abs(Y_tot - Y_tot.T).max(),
                      'multiplier_res': res, 'Lambda': Lam, 'M': Mmat})
    if auxmol is None:
        Gbar = [_permutation_sum(ao_backtransform(G4, C)) for G4 in four]
        G = two_electron_skeleton(mol, Gbar)
    else:
        three, two = zip(*(df_densities(G4, C, mol, auxmol) for G4 in four))
        G = two_electron_skeleton_df(mol, auxmol, three, two)
    G += one_electron_skeleton(mol, mf, [C @ g @ C.T for g in gam],
                               [C @ m @ C.T for m in Ms])
    return [g + fock_partial_skeleton(mf, gt, nocc)
            for g, gt in zip(G, gam)], diags


def correlation_gradients(mol, mf, target_list, eri_mo=None, auxmol=None,
                          verbose=False):
    """dL_X/dR for several targets sharing one skeleton-derivative sweep.

    target_list: list of (gammaF, Gamma4_E) MO-partial pairs, optionally
    (gammaF, Gamma4_E, Y_extra) to add an orbital-rotation gradient that no
    (F, ERI) partial can express. Returns
    (list of G_corr (natm,3), list of diagnostics). Add the mean field's own
    force for totals.
    eri_mo:  the MO integrals the partials were built from; recomputed here if
             omitted, which repeats an N^5 transform the caller already paid for.
    auxmol:  auxiliary basis for the factorized skeleton contraction. Only
             consistent when eri_mo is the RI four-index tensor from the SAME
             auxiliary basis (`Base.eri_blocks.df_eri_mo`) -- differentiating a
             fitted energy with exact derivative integrals is not a gradient.

    A Kohn-Sham reference is routed to `kohn_sham_correlation_gradients`: the
    fold below is the Hartree-Fock Fock and differentiates the wrong one-body
    matrix on one.
    """
    if isinstance(eri_mo, MOEriBlocks):
        # The skeleton sweep contracts a four-index density against the whole
        # tensor; blocks exist exactly so that tensor is never formed.
        eri_mo = eri_mo.require_dense('correlation_gradients')
    C = mf.mo_coeff
    eps = mf.mo_energy
    norb = C.shape[1]
    nocc = mol.nelectron // 2
    if eri_mo is None:
        eri_mo = ao2mo.general(mol, (C,) * 4, compact=False).reshape((norb,) * 4)
    if xc_hybrid_coeff(mf)[0]:
        return kohn_sham_correlation_gradients(mol, mf, target_list, eri_mo,
                                               auxmol=auxmol, verbose=verbose)
    h_mo = C.T @ mf.get_hcore() @ C
    F_mo = np.diag(eps)

    dens = []
    diags = []
    for target in target_list:
        # a target may carry an extra orbital-rotation gradient of its own:
        # anything depending on C that is NOT expressible through (F, ERI)
        # partials, such as an ISDF collocation X_mo = X_ao C, whose adjoint
        # gives Y_up += sum_g X_mo[g,u] Xbar[g,p]. It has to enter HERE, before
        # the multiplier solve, because it shares Lambda with everything else.
        gammaF, Gamma4_E = target[0], target[1]
        Y_extra = target[2] if len(target) > 2 else None
        Gamma4 = Gamma4_E.copy()
        gh_E, _ = fold_fock(gammaF, nocc, Gamma4)
        Y_E = orbital_Y(gh_E, Gamma4, h_mo, eri_mo)
        if Y_extra is not None:
            Y_E = Y_E + Y_extra
        Lam, res = solve_multipliers(Y_E, F_mo, eri_mo, nocc, eps, verbose=verbose)
        Y_tot = Y_E + _y_fold_lambda(Lam, F_mo, eri_mo, nocc)
        stat = np.abs(Y_tot - Y_tot.T).max()
        if verbose:
            print(f"    [stationarity] |antisym Y_tot| = {stat:.3e}")
        # dL under dC = C X: X_up coefficient Y_tot,up + M_pu + M_up = 0 with
        # M symmetric and antisym(Y_tot) = 0 enforced => M = -Y_tot/2.
        Mmat = -0.25 * (Y_tot + Y_tot.T)
        gh_L, _ = fold_fock(Lam, nocc, Gamma4)
        dens.append((gh_E + gh_L, Gamma4, Mmat))
        diags.append({'stationarity': stat, 'multiplier_res': res,
                      'Lambda': Lam, 'M': Mmat})

    # Back-transform ONCE per target and contract the derivative integrals in
    # the AO basis: the MO transform is what used to run per atom and direction.
    gam_ao = [C @ g @ C.T for g, _, _ in dens]
    M_ao = [C @ m @ C.T for _, _, m in dens]
    if auxmol is None:
        Gbar = [_permutation_sum(ao_backtransform(G4, C)) for _, G4, _ in dens]
        G = two_electron_skeleton(mol, Gbar)
    else:
        three, two = zip(*(df_densities(G4, C, mol, auxmol) for _, G4, _ in dens))
        G = two_electron_skeleton_df(mol, auxmol, three, two)
    G += one_electron_skeleton(mol, mf, gam_ao, M_ao)
    return list(G), diags


def correlation_gradient(mol, mf, gammaF, Gamma4_E, eri_mo=None, auxmol=None,
                         verbose=False):
    """Single-target wrapper around correlation_gradients."""
    Gs, diags = correlation_gradients(mol, mf, [(gammaF, Gamma4_E)],
                                      eri_mo=eri_mo, auxmol=auxmol,
                                      verbose=verbose)
    return Gs[0], diags[0]

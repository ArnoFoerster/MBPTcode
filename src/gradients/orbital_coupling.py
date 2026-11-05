"""The mean-field orbitals' derivative overlap A_pq = <phi_p | d phi_q / dR>
contracted with a fixed weight, for every method: the occupied-virtual block
(`ov_coupling`, the coupled-perturbed rotation), the canonical within-block
term between two excited states (`configuration_coupling`) and its
symmetric-gauge part (`symmetric_gauge_coupling`), and the fixed-coefficient
basis motion they are built on (`sigma_contraction`).

Writing A = A^sym + kappa with A^sym the symmetric-gauge part, the canonical
condition gives, for p != q inside one block,

    kappa_pq = [ dF_pq - eps_q dS_pq ] / (eps_q - eps_p) + dS_pq / 2

with dF and dS the derivatives in the FIXED reference MO basis. Contracting
a weight with that turns every energy denominator into a FIXED weight, and the
density response inside dF meets a fixed matrix, so ONE Z-vector replaces the
3 natm coupled-perturbed solves the nuclear perturbations would each need.
The excited-state derivative couplings (`src.gradients.gwbse.derivative_coupling`)
and the fragment-diabatic gradients (`src.gradients.fragment_diabatic`) read
these. Only the mean field enters (its orbitals, energies and response
kernel): nothing here depends on the excited-state method.
"""
import numpy as np

from src.Base.numerics.constants import (EXCITED_SPIN_FACTOR,
                                         ORBITAL_MULTIPLIER_DEGENERACY_TOL,
                                         ORBITAL_MULTIPLIER_RESIDUAL_TOL)
from src.Base.pyscf.pyscf_interface import response_kernel
from src.gradients.mean_field_response import (fock_partial_skeleton,
                                               solve_lambda)
from src.gradients.one_electron import one_electron_skeleton


def _transition_weight(goo, gvv, nocc, norb):
    """gamma_pq with sum_pq gamma_pq A_pq the configuration term.

    The occupied block is transposed against the virtual one, which is the
    minus sign of the transported RPA metric written as a single contraction.
    """
    gam = np.zeros((norb, norb))
    gam[:nocc, :nocc] = -np.asarray(goo, float).T
    gam[nocc:, nocc:] = np.asarray(gvv, float)
    return EXCITED_SPIN_FACTOR * gam


def sigma_contraction(mol, mo_coeff, weight, antisymmetrize):
    """(natm, 3) sum_pq weight_pq Sigma_pq, Sigma_pq = <phi_p | d phi_q / dR>
    at FIXED coefficients -- one `int1e_ipovlp` contraction, no response.

    A basis function moves only with the nucleus it sits on, and
    d(chi_nu)/dR_A = -grad(chi_nu) because chi_nu(r - R_A).

    antisymmetrize takes (Sigma - Sigma.T)/2, the within-block A^sym of the
    symmetric gauge. The ground-state coupling wants the raw Sigma instead,
    because there the coupled-perturbed rotation is supplied explicitly rather
    than fixed by a gauge choice.
    """
    c = np.asarray(mo_coeff, float)
    ip = mol.intor('int1e_ipovlp', comp=3)
    out = np.zeros((mol.natm, 3))
    for atom in range(mol.natm):
        lo, hi = mol.aoslice_by_atom()[atom][2:4]
        for x in range(3):
            sig = np.zeros_like(ip[x])
            sig[:, lo:hi] = -ip[x].T[:, lo:hi]
            mo = c.T @ sig @ c
            out[atom, x] = float((weight * (0.5 * (mo - mo.T) if antisymmetrize
                                            else mo)).sum())
    return out


def z_vector_contraction(mol, mf, nocc, w_f, w_s, numerics):
    """(natm, 3) sum_pq w_f_pq dF_pq + sum_pq w_s_pq dS_pq, FIXED MO basis.

    Both derivatives are what a displaced geometry gives with the reference
    coefficients held. dF carries the density response and dS does not, and
    the response meets a FIXED weight, so ONE Z-vector replaces the 3 natm
    coupled-perturbed solves the nuclear perturbations would each need.

    `solve_lambda` inverts exactly the operator this needs: for a symmetric
    Lambda, antisym(fock_partial_Y)[i,a] is
    -[2 (eps_a - eps_i) Lambda_ai + 4 Ghat[Lambda]_ai], so seeding an
    antisymmetric Y_E's occupied-virtual block with 4 Ghat solves for Z.

    The occupied-occupied rotation never enters: its antisymmetric part cancels
    out of the density derivative, leaving dD dependent on the
    occupied-virtual block and on dS_oo alone. numerics: the run's `Numerics`
    (the Z-vector's solve), required.
    """
    c = np.asarray(mf.mo_coeff, float)
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    kernel = response_kernel(mf)
    ghat = c.T @ kernel(c @ w_f @ c.T) @ c

    y_e = np.zeros((norb, norb))
    y_e[:nocc, nocc:] = 2.0 * ghat[nocc:, :nocc].T
    y_e[nocc:, :nocc] = -2.0 * ghat[nocc:, :nocc]
    z = solve_lambda(mf, y_e, nocc, numerics=numerics)[0]

    v = np.zeros((norb, norb))
    v[nocc:, :nocc] = z[nocc:, :nocc] * eps[None, :nocc]
    v = v + v.T
    ghat_z = c.T @ kernel(c @ z @ c.T) @ c

    oo = np.zeros((norb, norb), bool)
    oo[:nocc, :nocc] = True
    theta = w_f - z
    omega = w_s + v + np.where(oo, 2.0 * (ghat_z - ghat), 0.0)
    return (one_electron_skeleton(mol, mf, [c @ theta @ c.T],
                                  [c @ omega @ c.T])[0]
            + fock_partial_skeleton(mf, theta, nocc))


def _pair_matrix(block, nocc, norb):
    """The symmetric (norb, norb) matrix carrying an (nocc, nvir) weight twice.

    A sum over occupied-virtual pairs meets a SYMMETRIC derivative, so it is
    HALF of the full matrix contraction; callers pass half of this.
    """
    out = np.zeros((norb, norb))
    out[:nocc, nocc:] = np.asarray(block, float)
    return out + out.T


def configuration_coupling(mol, mf, nocc, goo, gvv, numerics):
    """(natm, 3) the configuration term between two EXCITED states.

    Writing A = A^sym + kappa, the canonical condition gives, for p != q inside
    one block,

        kappa_pq = [ dF_pq - eps_q dS_pq ] / (eps_q - eps_p) + dS_pq / 2

    in the FIXED reference MO basis. Contracting gamma with it turns every
    energy denominator into a FIXED weight, so nothing diverges as two orbitals
    approach each other -- what multiplies 1/(eps_q - eps_p) is dF, which
    vanishes with it:

        w_f = sym(g),                g_pq = gamma_pq / (eps_q - eps_p)
        w_s = sym(-g eps_q + gamma/2)

    Only the block-diagonal blocks of A enter: the state-to-state density of
    two singly excited states has no occupied-virtual block, so the one part of
    the rotation a gauge cannot fix never appears.

    A pair inside ORBITAL_MULTIPLIER_DEGENERACY_TOL has no canonical rotation:
    it is dropped when gamma vanishes on it (by symmetry) and refused when it
    does not, as `multipliers` refuses such a pair's right-hand side.
    numerics: the run's `Numerics` (the one Z-vector's solve), required.
    """
    c = np.asarray(mf.mo_coeff, float)
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    gam = _transition_weight(goo, gvv, nocc, norb)

    de = eps[None, :] - eps[:, None]
    block = np.zeros((norb, norb), bool)
    block[:nocc, :nocc] = block[nocc:, nocc:] = True
    np.fill_diagonal(block, False)
    degenerate = block & (np.abs(de) < ORBITAL_MULTIPLIER_DEGENERACY_TOL)
    if degenerate.any():
        bad = float(np.abs(gam[degenerate]).max())
        if bad > ORBITAL_MULTIPLIER_RESIDUAL_TOL:
            raise RuntimeError(
                f'a degenerate orbital pair carries state-to-state density '
                f'{bad:.2e}: its canonical rotation is undefined, so the '
                f'configuration term is too; symmetry-adapted treatment '
                f'needed')
    safe = np.where(degenerate, np.inf, de)
    g = np.where(block, gam / safe, 0.0)
    w_f = 0.5 * (g + g.T)
    w_s = np.where(block, -g * eps[None, :] + 0.5 * gam, 0.0)
    w_s = 0.5 * (w_s + w_s.T)
    return (z_vector_contraction(mol, mf, nocc, w_f, w_s, numerics)
            + symmetric_gauge_coupling(mol, mf, nocc, goo, gvv))


def symmetric_gauge_coupling(mol, mf, nocc, goo, gvv):
    """(natm, 3) the configuration term of `configuration_coupling` in the
    SYMMETRIC gauge: sum_pq gamma_pq A^sym_pq, A^sym = (Sigma - Sigma^T) / 2
    inside the occupied and the virtual block, no canonical rotation kappa.

    The orbitals of a reverse pass that keeps no canonical condition move in
    this gauge (a correlation force whose Fock derivative gamma is taken in a
    non-canonical basis, `df_correlation_force`): what such a pass leaves out
    of an orbital rotation is A^sym, not the canonical A.
    """
    c = np.asarray(mf.mo_coeff, float)
    gam = _transition_weight(goo, gvv, nocc, c.shape[1])
    return sigma_contraction(mol, c, gam, True)


def ov_coupling(mol, mf, nocc, w, numerics):
    """(natm, 3) sum_ia W_ia A_ia, A_ia = <phi_i | d phi_a / dR>, canonical.

    The occupied-virtual block of the derivative overlap, which is the
    coupled-perturbed rotation: `ground_state_coupling` is this with
    W = sqrt(2) (X - Y), and the localization response of
    `fragment_diabatic` reaches it with the occupied-virtual weight of the
    local orbitals' motion. One Z-vector for any W, solved on `numerics`,
    the run's.
    """
    w = np.asarray(w, float).reshape(nocc, -1)
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    u = w / (eps[None, nocc:] - eps[:nocc, None])
    # dF and dS are SYMMETRIC, so a sum over occupied-virtual pairs is half of
    # the full matrix contraction. Sigma is NOT, so its weight carries the
    # occupied-virtual entries alone and nothing on the transpose.
    w_f = 0.5 * _pair_matrix(u, nocc, norb)
    w_s = -0.5 * _pair_matrix(u * eps[None, nocc:], nocc, norb)
    w_sig = np.zeros((norb, norb))
    w_sig[:nocc, nocc:] = w
    return (z_vector_contraction(mol, mf, nocc, w_f, w_s, numerics)
            + sigma_contraction(mol, mf.mo_coeff, w_sig, False))

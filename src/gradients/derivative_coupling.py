"""Derivative couplings d_mn = <Psi_m | d/dR Psi_n> between two BSE roots.

The reference is `src.properties.nonadiabatic.derivative_couplings`, which
differences overlaps of the Casida vectors at displaced geometries and costs
6 natm + 1 GW-BSE solves. This route costs ONE GW-BSE solve.

TWO TERMS, AND NEITHER IS OPTIONAL. With
|Psi_n> = sum_ia X^n_ia(R) |Phi_ia(R)> + Y-part, the nuclear derivative hits the
amplitudes and the configurations separately:

    d_mn = <m| dH/dR |n> / (Omega_n - Omega_m)      amplitudes
         + sum_pq gamma^mn_pq A_pq                  configurations

with A_pq = <phi_p | d phi_q / dR>. Ou, Bellchambers, Furche and Subotnik
(J. Chem. Phys. 142, 064114 (2015)) reach the same split. The first term is the
excitation gradient's own reverse chain seeded with an interstate element
instead of a diagonal one, so the orbital relaxation and the Pulay terms arrive
with it (an excited-state chain's `interstate_gradient`).

THE TWO TERMS SHARE A GAUGE AND IT IS THE CANONICAL ONE. Neither is separately
invariant under an occupied-occupied or virtual-virtual rotation -- <m|dH|n> is
the derivative of a quantity that is identically zero, and what it measures is
how the orbital basis moves. The chain leaves the quasiparticle block DIAGONAL,
which is the canonical condition, so A must be the canonical derivative overlap
and not the symmetric-gauge one. The difference is not small: near-degenerate
virtuals rotate through each other at a rate 1/(eps_a - eps_b), and the
symmetric-gauge form can leave the majority of the coupling on the table.

NO DISPLACED SOLVE ANYWHERE. Writing A = A^sym + kappa with A^sym the
symmetric-gauge part, the canonical condition gives, for p != q inside one
block,

    kappa_pq = [ dF_pq - eps_q dS_pq ] / (eps_q - eps_p) + dS_pq / 2

with dF and dS the derivatives in the FIXED reference MO basis. Contracting
gamma with that turns every energy denominator into a FIXED weight, so the
assembled term carries no 1/(eps_q - eps_p) at all: what multiplies it is dF,
which vanishes with it. The density response inside dF then meets a fixed
matrix, so ONE Z-vector replaces the 3 natm coupled-perturbed solves the
nuclear perturbations would each need. `configuration_coupling` has the
assembly.

It cannot be routed through a moving-basis Fock-partial gradient, despite that
routine taking a full MO Fock partial: it differentiates in the MOVING
canonical basis, where the off-diagonal MO Fock is identically zero at every
geometry, so an off-diagonal gamma contributes nothing to it.

WHAT IS NOT HERE. The ground-to-excited coupling d_0n contracts the
occupied-virtual block of A, which is the coupled-perturbed rotation itself.
Everything ISC, rISC and internal conversion need BETWEEN excited states is
excited-excited.
"""
import numpy as np
from pyscf import gto

from src.Base.constants import NUCLEAR_FD_STEP
from src.gradients.grad_engine import one_electron_skeleton
from src.gradients.isdf_derivatives import (fock_partial_skeleton,
                                            response_kernel, solve_lambda)
from src.properties.nonadiabatic import mo_overlap
from src.properties.spin_orbit import (EXCITED_SPIN_FACTOR,
                                       GROUND_SPIN_FACTOR)


def state_to_state_density(nocc, xm, ym, xn, yn):
    """(gamma_oo, gamma_vv) between two singly excited states.

        gamma^vv_ab = sum_i (X^m_ia X^n_ib - Y^m_ia Y^n_ib)
        gamma^oo_ij = sum_a (X^m_ia X^n_ja - Y^m_ia Y^n_ja)

    Read off the transported RPA metric of `src.properties.nonadiabatic` at
    first order in the displacement, so the two routes share their convention
    for the minus sign and the spin factor rather than each choosing one. There
    is no occupied-virtual block: it would need a ground-state-like component
    the single-excitation ansatz does not carry.
    """
    x_m = np.asarray(xm, float).reshape(nocc, -1)
    y_m = np.asarray(ym, float).reshape(nocc, -1)
    x_n = np.asarray(xn, float).reshape(nocc, -1)
    y_n = np.asarray(yn, float).reshape(nocc, -1)
    return (x_m @ x_n.T - y_m @ y_n.T, x_m.T @ x_n - y_m.T @ y_n)


def displaced(mol, crd, atom, axis, delta):
    """`mol` with one Cartesian component of one nucleus moved, in Bohr."""
    c = np.array(crd, float)
    c[atom, axis] += delta
    return gto.M(atom=[(mol.atom_symbol(i), tuple(c[i])) for i in range(mol.natm)],
                 basis=mol.basis, unit='Bohr', charge=mol.charge,
                 spin=mol.spin, verbose=0)


def canonical_overlap_derivative(mol, mf, scf_factory, atom, axis,
                                 step=NUCLEAR_FD_STEP):
    """A_pq = <phi_p | d phi_q / dR> for the CANONICAL orbitals, by difference.

    Mean fields only -- no GW and no BSE. Not on the production path: this is
    the independent check that `configuration_coupling`'s Z-vector reproduces,
    and it is also the cheapest way to SEE the gauge.

    Each displaced solve returns its orbitals up to a sign, which is fixed by
    making the diagonal of the overlap positive; a within-block ROTATION needs
    no fixing, because it is exactly the canonical gauge this term carries.
    """
    crd = np.asarray(mol.atom_coords())
    out = []
    for sign in (-1.0, 1.0):
        md = displaced(mol, crd, atom, axis, sign * step)
        t = mo_overlap(mol, mf.mo_coeff, md, scf_factory(md).mo_coeff)
        out.append(t * np.sign(np.diag(t))[None, :])
    return (out[1] - out[0]) / (2 * step)


def _transition_weight(goo, gvv, nocc, norb):
    """gamma_pq with sum_pq gamma_pq A_pq the configuration term.

    The occupied block is transposed against the virtual one, which is the
    minus sign of the transported RPA metric written as a single contraction.
    """
    gam = np.zeros((norb, norb))
    gam[:nocc, :nocc] = -np.asarray(goo, float).T
    gam[nocc:, nocc:] = np.asarray(gvv, float)
    return EXCITED_SPIN_FACTOR * gam


def _sigma_contraction(mol, mo_coeff, weight, antisymmetrize):
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


def _z_vector_contraction(mol, mf, nocc, w_f, w_s):
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
    occupied-virtual block and on dS_oo alone.
    """
    c = np.asarray(mf.mo_coeff, float)
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    kernel = response_kernel(mf)
    ghat = c.T @ kernel(c @ w_f @ c.T) @ c

    y_e = np.zeros((norb, norb))
    y_e[:nocc, nocc:] = 2.0 * ghat[nocc:, :nocc].T
    y_e[nocc:, :nocc] = -2.0 * ghat[nocc:, :nocc]
    z = solve_lambda(mf, y_e, nocc)[0]

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


def configuration_coupling(mol, mf, nocc, goo, gvv):
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
    """
    c = np.asarray(mf.mo_coeff, float)
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    gam = _transition_weight(goo, gvv, nocc, norb)

    de = eps[None, :] - eps[:, None]
    block = np.zeros((norb, norb), bool)
    block[:nocc, :nocc] = block[nocc:, nocc:] = True
    np.fill_diagonal(block, False)
    safe = np.where(np.abs(de) < 1e-8, np.inf, de)
    g = np.where(block, gam / safe, 0.0)
    w_f = 0.5 * (g + g.T)
    w_s = np.where(block, -g * eps[None, :] + 0.5 * gam, 0.0)
    w_s = 0.5 * (w_s + w_s.T)
    return (_z_vector_contraction(mol, mf, nocc, w_f, w_s)
            + _sigma_contraction(mol, c, gam, True))


def ground_state_coupling(mol, mf, nocc, xn, yn):
    """(natm, 3) d_0n = <Psi_0 | d/dR Psi_n>, the S1 -> S0 coupling.

    ONE TERM, not two, and this is what makes it different from every
    excited-excited pair. <Psi_0|Phi_ia> = 0, so the amplitude derivative drops
    out identically and the whole coupling is the configurations moving:

        d_0n = sqrt(2) sum_ia (X^n - Y^n)_ia A_ia

    read off the transported RPA metric of `src.properties.nonadiabatic` at
    first order in the displacement.

    A_ia IS THE COUPLED-PERTURBED ROTATION, not a gauge choice. The
    excited-excited coupling touches only the block-diagonal blocks of A, whose
    within-block rotation a gauge fixes; the occupied-virtual block is
    physical, and

        A_ia = Sigma_ia + [ dF_ia - eps_a dS_ia ] / (eps_a - eps_i)

    So this needs a response solve where the other did not -- but it is still
    ONE Z-vector, because (X - Y) is a fixed vector and the energy denominator
    is absorbed into its weight. eps_a - eps_i never vanishes on a closed shell.
    """
    x = np.asarray(xn, float).reshape(nocc, -1)
    y = np.asarray(yn, float).reshape(nocc, -1)
    eps = np.asarray(mf.mo_energy, float)
    norb = len(eps)
    w = GROUND_SPIN_FACTOR * (x - y)
    u = w / (eps[None, nocc:] - eps[:nocc, None])
    # dF and dS are SYMMETRIC, so a sum over occupied-virtual pairs is half of
    # the full matrix contraction. Sigma is NOT, so its weight carries the
    # occupied-virtual entries alone and nothing on the transpose.
    w_f = 0.5 * _pair_matrix(u, nocc, norb)
    w_s = -0.5 * _pair_matrix(u * eps[None, nocc:], nocc, norb)
    w_sig = np.zeros((norb, norb))
    w_sig[:nocc, nocc:] = w
    return (_z_vector_contraction(mol, mf, nocc, w_f, w_s)
            + _sigma_contraction(mol, mf.mo_coeff, w_sig, False))


def analytic_coupling(chain, m, n, mol=None, mf=None):
    """(d_mn, diagnostics) in 1/Bohr between two BSE roots of `chain`.

    Antisymmetric in m <-> n: the interstate numerator is symmetric -- both
    orderings are averaged in the chain's interstate backward pass -- and the
    gap changes sign.
    """
    if m == n:
        raise ValueError('a derivative coupling needs two different roots')
    mol, mf = chain.mean_field(mol, mf)
    num, diags = chain.interstate_gradient(m, n, mol=mol, mf=mf)
    _, pieces = chain._forward(mol, mf)
    xn, yn = pieces[10], pieces[11]
    goo, gvv = state_to_state_density(chain.nocc, xn[:, m], yn[:, m],
                                      xn[:, n], yn[:, n])
    csf = configuration_coupling(mol, mf, chain.nocc, goo, gvv)
    amp = num / diags['gap']
    return amp + csf, dict(diags, branch_amplitude=float(np.abs(amp).max()),
                           branch_configuration=float(np.abs(csf).max()))


def analytic_ground_coupling(chain, n, mol=None, mf=None):
    """(d_0n, diagnostics) in 1/Bohr between the ground state and BSE root `n`.

    The S1 -> S0 internal-conversion coupling. One Z-vector and no displaced
    solve; see `ground_state_coupling` for why it has no amplitude term.
    """
    mol, mf = chain.mean_field(mol, mf)
    om, pieces = chain._forward(mol, mf)
    xn, yn = pieces[10], pieces[11]
    d = ground_state_coupling(mol, mf, chain.nocc, xn[:, n], yn[:, n])
    return d, {'omega_n': float(om[n]), 'gap': float(om[n]),
               'branch_configuration': float(np.abs(d).max())}

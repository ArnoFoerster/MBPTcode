"""The mean field's response to a gradient's adjoint, on every route.

A force reaches the nuclei through the orbitals as well as through its own
factors. This module carries that part: the orbital-rotation gradient of a
Fock partial (`fock_partial_Y`), the multiplier solve that makes it
stationary (`solve_lambda`), and the skeleton derivatives it folds into, with
the coefficients held fixed (`fock_partial_skeleton`, `xc_skeleton`,
`reaction_field_skeleton`). A method's own one-electron operators join through
the same pieces: the quasiparticle correction Sigma_x - v_xc in
`gradients.gwbse.static_correction`, the dRPA ground state's E_x - E_xc in
`gradients.rpa.exchange_double_counting`.
`eps_chain_gradient` assembles them into the force of an adjoint on the
orbital energies, and `relaxed_density_mo` stops before the skeleton and
returns the relaxed density instead.

Every piece is built from Coulomb and exchange builds, so no norb^4 tensor
appears on the dense, density-fitted or ISDF-K mean field. Each route picks
its skeleton here: four-centre derivative integrals on a dense mean field,
the fitted adjoints on a density-fitted one (`fock_partial_skeleton_df`),
and the interpolated exchange of `isdf_derivatives` on an ISDF-K one.

Every gradient route reads it: the dense, cubic and space-time chains
(`grad_engine`, `dense_surfaces`, `factor_chain`, `excited_state`,
`derivative_coupling`) and the correlated forces' multiplier
(`multipliers.rotation_gradient`).
"""
import numpy as np
from pyscf.grad import rhf as grad_rhf

from src.Base.factorization.isdf_jk import ISDFJK
from src.environments.pcm_derivatives import reaction_field_fock_skeleton
from src.Base.pyscf.pyscf_interface import fock_mo, response_kernel
from src.gradients.skeleton_tiles import fitted_fock_skeleton, xc_grid_skeleton
from src.Base.utils.parallel.mpi_grid import lockstep
from src.gradients.isdf_derivatives import (isdf_fock_partial_exchange,
                                            require_fitted_coulomb)
from src.gradients.multipliers import rotation_gradient, solve_orbital_multipliers
from src.gradients.one_electron import one_electron_skeleton
from src.Base.pyscf.pyscf_interface import (auxmol_of, exchange_channels, rsh_split,
                                            xc_hybrid_coeff)


# ---------------------------------------------------------------------------
# the orbital-energy chain, without any norb^4 tensor
# ---------------------------------------------------------------------------

def require_no_range_separation(mf, what):
    """Refuse a derivative that would silently drop the long-range exchange.

    The skeletons in this module (`fock_partial_skeleton`, its fitted twin,
    `qp_xc_correction_skeleton`) and `isdf_derivatives.isdf_exchange_skeleton`
    carry every channel and do not need it; it is the refusal a
    single-channel route owes its caller.

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


def fock_partial_Y(mf, gamma, nocc, environment=True):
    """Orbital-rotation gradient of Tr[gamma F], gamma a symmetric MO partial,
    on this mean field: `multipliers.rotation_gradient` with its MO Fock
    matrix and its response kernel G. Tr[gamma F] depends on the coefficients
    through the two-sided transform and through the density inside F; with a
    fitted or ISDF mean field it is cubic and no norb^4 tensor appears.

    environment=False takes G without the continuum's response (the ground
    state's surface charges frozen), for a density response in the frozen
    field (`relaxed_density_mo`).
    """
    return rotation_gradient(mf.mo_coeff, fock_mo(mf), gamma, nocc,
                             response_kernel(mf, environment=environment))


def solve_lambda(mf, Y_E, nocc, numerics=None, verbose=False,
                 environment=True, flags=None):
    """Symmetric zero-diagonal Lambda with antisym(Y_E + Y[Lambda]) = 0.

    The cubic route's entry to `solve_orbital_multipliers`: the matvec is
    `fock_partial_Y` -- a Coulomb and an exchange build -- instead of the
    four-index tensor, and the orbital energies come from the mean field.
    numerics: the caller's `Numerics` (the solve's tolerances); flags: a list
    a solve left short appends its flag to (`multipliers`).
    """
    return solve_orbital_multipliers(
        lambda Lam: fock_partial_Y(mf, Lam, nocc, environment=environment),
        Y_E, mf.mo_energy, numerics=numerics, verbose=verbose, flags=flags)


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
            require_fitted_coulomb(mf, 'the folded Fock partial skeleton')
            g = fock_partial_skeleton_df(mf, auxmol_of(mf), gamma, nocc,
                                         channels=(), coulomb=True)
            g = g + isdf_fock_partial_exchange(mf, gamma)
        else:
            g = fock_partial_skeleton_df(mf, auxmol_of(mf), gamma, nocc,
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
    """(natm, 3) of d/dR Tr[gamma v_xc^DFT], MO coefficients held fixed and the
    Becke grid moving with the atoms, as the energy's grid does.

    The exchange-correlation half of the Kohn-Sham Fock partial. It appears
    once, not twice. The Coulomb and exchange terms come from a four-index
    object in which gamma and the density enter symmetrically, while v_xc is
    not bilinear in the density.

    The grid's own motion is part of it. The fixed-grid derivative (the AOs
    and the density moving through the kernel) misses the points riding with
    their atoms and the Becke weights' response, which is the whole of an
    excitation force's translation residual on a Kohn-Sham reference, larger
    on a range-separated functional than on a global hybrid;
    `skeleton_tiles.xc_grid_skeleton` carries all three terms.
    """
    if mf.do_nlc():
        raise NotImplementedError(
            f'{mf.xc!r} carries a VV10 kernel, whose skeleton is not built')
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return xc_grid_skeleton(mf.mol, mf.grids, mf._numint, mf.xc, g_ao,
                            np.asarray(mf.make_rdm1()))


def eps_chain_gradient(mf, eps_bar, nocc, Y_extra=None, verbose=False,
                       numerics=None):
    """(natm, 3) of sum_pq gamma_pq dF_pq/dR, with no norb^4 tensor anywhere.

    The dense engine's Lagrangian, rebuilt on Coulomb/exchange builds. Fold the
    adjoint into a Fock partial, solve for the canonical-condition multiplier,
    read the orthonormality multiplier off the stationary orbital gradient, and
    contract the skeleton. `Y_extra` carries any orbital-rotation dependence
    that is not expressible as an (F, ERI) partial.

    eps_bar: the adjoint on the orbital energies, gamma = diag(eps_bar), or a
        full symmetric MO Fock partial gamma_pq = dE/dF_pq.
    """
    C = mf.mo_coeff
    eps_bar = np.asarray(eps_bar, float)
    gamma = np.diag(eps_bar) if eps_bar.ndim == 1 else 0.5 * (eps_bar + eps_bar.T)
    Y_E = fock_partial_Y(mf, gamma, nocc)
    if Y_extra is not None:
        Y_E = Y_E + Y_extra
    Lam, res = solve_lambda(mf, Y_E, nocc, numerics=numerics, verbose=verbose)
    Y_tot = Y_E + fock_partial_Y(mf, Lam, nocc)
    Mmat = -0.25 * (Y_tot + Y_tot.T)
    gamma_tot = gamma + Lam
    g2 = fock_partial_skeleton(mf, gamma_tot, nocc)
    g1 = one_electron_skeleton(mf.mol, mf, [C @ gamma_tot @ C.T],
                               [C @ Mmat @ C.T])[0]
    return g1 + g2, {'stationarity': np.abs(Y_tot - Y_tot.T).max(),
                     'multiplier_res': res}


def relaxed_density_mo(mf, eps_bar, nocc, Y_extra=None, environment=False,
                       numerics=None):
    """The orbital-relaxed density of a target, gamma + Lambda in the MO basis
    of `mf`: d target / d h = Tr[C (gamma + Lambda) C^T h] for any one-electron
    perturbation h, from the same Fock partial and orbital-rotation gradient
    `eps_chain_gradient` folds into a force.

    environment=False solves the multiplier with the response kernel of the
    gas phase on the mean field's orbitals: the continuum's surface charges
    stay those of the ground state, as a state-specific term needs (the
    classical E_SS already holds the solvent's whole linear answer to the
    density, and a responding kernel would count the mutual polarization a
    second time). environment=True is the force's own multiplier.
    """
    eps_bar = np.asarray(eps_bar, float)
    gamma = (np.diag(eps_bar) if eps_bar.ndim == 1
             else 0.5 * (eps_bar + eps_bar.T))
    Y_E = fock_partial_Y(mf, gamma, nocc, environment=environment)
    if Y_extra is not None:
        Y_E = Y_E + Y_extra
    Lam, res = solve_lambda(mf, Y_E, nocc, numerics=numerics,
                            environment=environment)
    return gamma + Lam, res


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

    with K^P = sum_Q Vinv[P,Q] J^Q. Nothing of the (nao, nao, naux) shape is
    formed: `skeleton_tiles.fitted_fock_skeleton` streams the three-centre
    integrals and their derivatives over fixed auxiliary tiles, the tiles
    divided over the current ranks and their (natm, 3) addends reduced once.
    Every omega = 0 channel is one exchange at the channels' summed weight;
    a long-range one is the four-centre `exchange_channel_skeleton`, rank 0's
    on every rank. The exchange term costs naux nao^2 nocc, what
    density-fitted exchange costs anywhere.

    `channels` is `exchange_channels(mf)`, and a range-separated hybrid's
    second channel carries its own density fit. `coulomb=False` leaves the
    exchange alone, which is what the Sigma_x - v_xc correction needs. On a
    Kohn-Sham reference the XC half comes from `xc_skeleton`, which the caller
    adds.
    """
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    D = mf.make_rdm1()
    full = 0.0
    for omega, weight in channels:
        if omega == 0.0:
            full += weight
    occ = mf.mo_occ
    grad = fitted_fock_skeleton(
        mf.mol, auxmol, g_ao, D,
        occ=C[:, occ > 0] * np.sqrt(occ[occ > 0]) if full else None,
        coulomb=coulomb, exchange=full)
    for omega, weight in channels:
        if omega != 0.0 and weight != 0.0:
            # pyscf's threaded derivative K: one rank's bits on all of them
            grad = grad + lockstep(exchange_channel_skeleton(mf, g_ao, D,
                                                             omega, weight))
    return grad

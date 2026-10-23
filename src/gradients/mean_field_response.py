"""The mean field's response to a gradient's adjoint, on every route.

A force reaches the nuclei through the orbitals as well as through its own
factors. This module carries that part: the orbital-rotation gradients of a
Fock partial and of the static correction Sigma_x - v_xc (`fock_partial_Y`,
`qp_xc_correction_Y`), the multiplier solve that makes them stationary
(`solve_lambda`), and the skeleton derivatives they fold into, with the
coefficients held fixed (`fock_partial_skeleton`, `xc_skeleton`,
`reaction_field_skeleton`, `exx_double_counting_skeleton`).
`eps_chain_gradient` assembles them into the force of an adjoint on the
orbital energies, and `relaxed_density_mo` stops before the skeleton and
returns the relaxed density instead.

Every piece is built from Coulomb and exchange builds, so no norb^4 tensor
appears on the dense, density-fitted or ISDF-K mean field. Each route picks
its skeleton here: four-centre derivative integrals on a dense mean field,
the fitted adjoints on a density-fitted one (`fock_partial_skeleton_df`),
and the interpolated exchange of `isdf_derivatives` on an ISDF-K one.

The dense, cubic and space-time chains (`grad_engine`, `dense_surfaces`,
`factor_chain`, `excited_state`, `derivative_coupling`) read it.
"""
import numpy as np
from pyscf import scf as pyscf_scf
from pyscf.grad import rhf as grad_rhf

from src.Base.numerics.constants import (ORBITAL_MULTIPLIER_MAX_ITER,
                                         ORBITAL_MULTIPLIER_TOL)
from src.Base.pyscf.dispersion import refuse_dispersion_under_rpa
from src.Base.factorization.distributed_df import distributed_fock
from src.Base.factorization.isdf_jk import ISDFJK
from src.gradients.isdf_mean_field import mean_field_skeleton_force
from src.environments.pcm_derivatives import reaction_field_fock_skeleton
from src.Base.pyscf.pyscf_interface import fock_mo, response_kernel
from src.gradients.skeleton_tiles import fitted_fock_skeleton, xc_grid_skeleton
from src.Base.utils.parallel.mpi_grid import lockstep
from src.gradients.isdf_derivatives import (auxmol_of,
                                            isdf_fock_partial_exchange,
                                            require_fitted_coulomb)
from src.gradients.multipliers import solve_orbital_multipliers
from src.gradients.one_electron import one_electron_skeleton
from src.SingleReference.GW.qp_solve import mean_field_static_operator
from src.SingleReference.LinearResponse.rpa_energy import (
    exchange_channels, rsh_split, xc_hybrid_coeff)


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
    """Orbital-rotation gradient of Tr[gamma F], gamma a symmetric MO partial.

    Tr[gamma F] depends on the coefficients through the two-sided transform and
    through the density inside F, and both come out as

        Y[u,p] = 2 (F gamma)[u,p] + 4 delta_{p occupied} (C^T G(gamma_ao) C)[u,p]

    with G(x) the closed-shell response kernel and gamma_ao = C gamma C^T. The
    same expression serves the target's own partial and the multiplier's. It
    needs a Coulomb and an exchange build and nothing else, so with a fitted or
    ISDF mean field it is cubic and no norb^4 tensor appears.

    environment=False takes G without the continuum's response (the ground
    state's surface charges frozen), for a density response in the frozen
    field (`relaxed_density_mo`).
    """
    C = mf.mo_coeff
    gs = 0.5 * (gamma + gamma.T)
    g_ao = C @ gs @ C.T
    G = response_kernel(mf, environment=environment)(g_ao)
    F_mo = fock_mo(mf)
    Y = 2.0 * (F_mo @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ G @ C)[:, :nocc]
    return Y


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
        with distributed_fock(mf, build=False):
            vj, vk = mf.get_jk(mf.mol, x, hermi=1)
        return vj - 0.5 * vk - kern(x)
    return apply


def weight_matrix(weights):
    """The symmetric MO weight of sum_pq w_pq <p|O|q>: diag(w) for a vector of
    diagonal weights, the symmetric part of a matrix otherwise."""
    w = np.asarray(weights, float)
    return np.diag(w) if w.ndim == 1 else 0.5 * (w + w.T)


def qp_xc_correction_matrix(mf, orbitals):
    """<p|Sigma_x - v_xc|q> over `orbitals`, the static correction's
    off-diagonal block, on the operator `qp_xc_correction_Y` differentiates."""
    c = np.asarray(mf.mo_coeff)[:, list(orbitals)]
    return c.T @ mean_field_static_operator(mf) @ c


def qp_xc_correction_Y(mf, weights, nocc):
    """Orbital-rotation gradient of sum_p w_p <p|Sigma_x - v_xc|p>, or of
    sum_pq w_pq <p|Sigma_x - v_xc|q> for a symmetric matrix of weights.

    The same shape as `fock_partial_Y`, with the Fock replaced by this
    operator: two terms, one from the two-sided MO transform and one from the
    operator's own density dependence. It must enter the Lagrangian before the
    multiplier solve, because it shares Lambda with every other contribution.
    """
    C = mf.mo_coeff
    gs = weight_matrix(weights)
    g_ao = C @ gs @ C.T
    o_mo = C.T @ mean_field_static_operator(mf) @ C
    Y = 2.0 * (o_mo @ gs)
    Y[:, :nocc] += 4.0 * (C.T @ _delta_o_response(mf)(g_ao) @ C)[:, :nocc]
    return Y


def qp_xc_correction_skeleton(mf, weights, nocc):
    """(natm, 3) of d/dR sum_p w_p <p|Sigma_x - v_xc|p>, coefficients fixed;
    a matrix of weights is sum_pq w_pq <p|Sigma_x - v_xc|q>.

    The operator is -((1 - a_x)/2) K - v_xc^DFT (see `_delta_o_response`), so
    the exchange half is `fock_partial_skeleton`'s K structure at weight
    -(1 - a_x) with no Coulomb term, and the rest is minus `xc_skeleton`. There
    is no one-electron part. On Hartree-Fock both halves vanish.
    """
    is_ks, _ = xc_hybrid_coeff(mf)
    if not is_ks:
        return np.zeros((mf.mol.natm, 3))
    C = mf.mo_coeff
    gs = weight_matrix(weights)
    # Sigma_x is the full-range exact exchange whatever the reference is, and
    # v_xc carries the reference's own channels, so Sigma_x - v_xc is one
    # full-range unit minus each of them. On a global hybrid that collapses to
    # the single weight 1 - a_x.
    delta = [(0.0, 1.0)] + [(o, -w) for o, w in exchange_channels(mf)]
    # A density-fitted mean field must go through the fitted skeleton, for the
    # same reason `fock_partial_skeleton` routes there. pyscf's conventional and
    # density-fitted derivative J/K do not share a convention, and the mismatch
    # shows up as a sharply degraded translational invariance.
    if isinstance(getattr(mf, 'with_df', None), ISDFJK):
        require_fitted_coulomb(mf, 'the Sigma_x - v_xc skeleton')
        # Sigma_x is the interpolated exchange here, because that is what the
        # SCF built and what the chain's own static exchange uses. Taking it
        # from the auxiliary basis would differentiate one operator while the
        # energy carries another.
        return (isdf_fock_partial_exchange(mf, gs, channels=delta)
                - xc_skeleton(mf, gs))
    if getattr(mf, 'with_df', None) is not None:
        grad = fock_partial_skeleton_df(mf, auxmol_of(mf), gs, nocc,
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
        # difference, and so would the other realization of it, so the
        # distributed SCF's handle travels with the fit.
        hf.with_df = with_df
        handles = getattr(mf, '_distributed', None)
        if handles is not None:
            for partner in (hf, ks):
                if getattr(partner, '_distributed', None) is None:
                    partner._distributed = handles
    return (np.asarray(mean_field_skeleton_force(hf))
            - np.asarray(mean_field_skeleton_force(ks)))


def exx_double_counting_Y(mf, nocc):
    """Orbital-rotation gradient of (E_x^exact - E_xc).

    One term, where `qp_xc_correction_Y` needs two. The functional derivative of
    this energy with respect to the density is the operator
    `qp_xc_correction_matrix` carries, dE/dD = Sigma_x - v_xc, and an energy
    does not depend on the coefficients through the states as <p|O[D]|p>
    does.

    The factor and the occupied restriction follow `fock_partial_Y`.
    """
    if not xc_hybrid_coeff(mf)[0]:
        return np.zeros((np.shape(mf.mo_coeff)[-1],) * 2)
    C = mf.mo_coeff
    o_ao = mean_field_static_operator(mf)
    Y = np.zeros((C.shape[1], C.shape[1]))
    Y[:, :nocc] = 4.0 * (C.T @ o_ao @ C)[:, :nocc]
    return Y


def solve_lambda(mf, Y_E, nocc, tol=ORBITAL_MULTIPLIER_TOL,
                 max_iter=ORBITAL_MULTIPLIER_MAX_ITER, verbose=False,
                 environment=True, flags=None):
    """Symmetric zero-diagonal Lambda with antisym(Y_E + Y[Lambda]) = 0.

    The cubic route's entry to `solve_orbital_multipliers`: the matvec is
    `fock_partial_Y` -- a Coulomb and an exchange build -- instead of the
    four-index tensor, and the orbital energies come from the mean field.
    flags: a list a solve left short appends its flag to (`multipliers`).
    """
    return solve_orbital_multipliers(
        lambda Lam: fock_partial_Y(mf, Lam, nocc, environment=environment),
        Y_E, mf.mo_energy, tol=tol, max_iter=max_iter, verbose=verbose,
        flags=flags)


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


def relaxed_density_mo(mf, eps_bar, nocc, Y_extra=None, environment=False):
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
    Lam, res = solve_lambda(mf, Y_E, nocc, environment=environment)
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

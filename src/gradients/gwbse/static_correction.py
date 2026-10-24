"""The quasiparticle energies' static correction Sigma_x - v_xc, differentiated:
its matrix over a set of orbitals (`qp_xc_correction_matrix`), the
orbital-rotation gradient of a weighted sum of its elements, which joins the
Lagrangian before the multiplier solve (`qp_xc_correction_Y`, the shape of
`mean_field_response.fock_partial_Y`), and its skeleton with the coefficients
held fixed (`qp_xc_correction_skeleton`). On Hartree-Fock both derivatives
vanish. The operator is the one the quasiparticle solve adds
(`qp_solve.mean_field_static_operator`).
"""
import numpy as np

from src.Base.factorization.distributed_df import distributed_fock
from src.Base.factorization.isdf_jk import ISDFJK
from src.Base.pyscf.pyscf_interface import response_kernel
from src.SingleReference.GW.qp_solve import mean_field_static_operator
from src.Base.pyscf.pyscf_interface import exchange_channels, xc_hybrid_coeff
from src.gradients.isdf_derivatives import (auxmol_of, isdf_fock_partial_exchange,
                                            require_fitted_coulomb)
from src.gradients.mean_field_response import (exchange_channel_skeleton,
                                               fock_partial_skeleton_df,
                                               xc_skeleton)


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

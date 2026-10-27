"""The dRPA ground state's exchange correction on a Kohn-Sham reference,
E_x^exact - E_xc of the reference density, differentiated: its skeleton with
the coefficients held fixed (`exx_double_counting_skeleton`) and its
orbital-rotation gradient (`exx_double_counting_Y`, which joins the
Lagrangian before the multiplier solve). Both vanish on Hartree-Fock.
"""
import numpy as np
from pyscf import scf as pyscf_scf

from src.Base.factorization.distributed_df import share_distributed
from src.Base.factorization.isdf_jk import ISDFJK
from src.Base.pyscf.dispersion import refuse_dispersion_under_rpa
from src.SingleReference.GW.qp_solve import mean_field_static_operator
from src.Base.pyscf.pyscf_interface import xc_hybrid_coeff
from src.gradients.isdf_mean_field import mean_field_skeleton_force


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

    Each member goes through `isdf_mean_field.mean_field_skeleton_force`, the one
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
        share_distributed(mf, hf, ks)
    return (np.asarray(mean_field_skeleton_force(hf))
            - np.asarray(mean_field_skeleton_force(ks)))


def exx_double_counting_Y(mf, nocc):
    """Orbital-rotation gradient of (E_x^exact - E_xc).

    One term, where `qp_xc_correction_Y` needs two. The functional derivative of
    this energy with respect to the density is the operator
    `qp_solve.mean_field_static_operator` builds, dE/dD = Sigma_x - v_xc, and an energy
    does not depend on the coefficients through the states as <p|O[D]|p>
    does.

    The factor and the occupied restriction follow
    `mean_field_response.fock_partial_Y`.
    """
    if not xc_hybrid_coeff(mf)[0]:
        return np.zeros((np.shape(mf.mo_coeff)[-1],) * 2)
    C = mf.mo_coeff
    o_ao = mean_field_static_operator(mf)
    Y = np.zeros((C.shape[1], C.shape[1]))
    Y[:, :nocc] = 4.0 * (C.T @ o_ao @ C)[:, :nocc]
    return Y

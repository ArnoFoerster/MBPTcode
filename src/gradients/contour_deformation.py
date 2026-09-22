"""Compatibility shim: the contour deformation's forward half lives in
`SingleReference.GW.contour_deformation` and its adjoints in
`gradients.contour_deformation_adjoint`; every name this module exports
resolves to one of those two, so a gradient caller reaches the same objects
under the names it has always used.

`ExplicitRealScreening` is the ADJOINT subclass, which is what this name has
always meant on the gradient side: the forward-only class of the same name is
`GW.real_screening.ExplicitRealScreening`.
"""
from src.Base.constants import (QP_POLE_OFFSET,  # noqa: F401
                                QP_POLE_OFFSET_MIN, QP_POLE_STRENGTH_MIN,
                                RESIDUE_ON_CONTOUR_TOL)
from src.SingleReference.GW.contour_deformation import (  # noqa: F401
    _need, qp_energy_cd as _qp_energy_cd, residue_pole_distance,
    residue_set, root_pole_distance, screening_applied,
    screening_contraction, sigma_cd, sigma_cd_slope,
    wc_explicit as _wc_explicit)
from src.SingleReference.GW.real_screening import (  # noqa: F401
    ov_energies as _ov_energies, screening_aux)
from src.SingleReference.LinearResponse.imaginary_frequency import \
    frequency_factor
from src.gradients.contour_deformation_adjoint import (  # noqa: F401
    ExplicitRealScreeningAdjoint as ExplicitRealScreening, _residue_backend,
    integral_term_backward, qp_energy_cd_backward, residue_terms_backward,
    screening_chain, sigma_cd_backward)


def _f_imaginary(d, nu):
    """Frequency factor of the particle-hole propagator at i.nu, and d(f)/d(d)."""
    return frequency_factor(d, nu, True, slopes=True)[:2]


def _f_real(d, omega, eta=0.0):
    """Frequency factor at a real omega, and its d and omega slopes."""
    return frequency_factor(d, omega, False, eta, slopes=True)


def qp_energy_cd(*args, **kwargs):
    """`GW.contour_deformation.qp_energy_cd` with this module's guard constants.

    The three guard numbers are read HERE, at call time, so that a caller which
    rebinds `contour_deformation.QP_POLE_OFFSET_MIN` (or either of the others)
    still moves the Newton it always moved.
    """
    kwargs.setdefault('offset_min', QP_POLE_OFFSET_MIN)
    kwargs.setdefault('z_min', QP_POLE_STRENGTH_MIN)
    kwargs.setdefault('linear_offset', QP_POLE_OFFSET)
    return _qp_energy_cd(*args, **kwargs)

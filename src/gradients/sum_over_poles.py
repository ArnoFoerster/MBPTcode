"""Compatibility shim: every name the pole model bound -- the four SOP
constants, `_denominators`, and the helpers borrowed from the contour
deformation included -- resolves to the forward objects of
`SingleReference.GW.sum_over_poles` or the reverse ones of
`gradients.sum_over_poles_adjoint`.
"""
from src.Base.constants import (SOP_CLEARANCE_MIN,  # noqa: F401
                                SOP_FIT_RCOND, SOP_FIT_STRIDE, SOP_N_POLES)
from src.SingleReference.GW.contour_deformation import residue_set  # noqa: F401
from src.SingleReference.GW.real_screening import (  # noqa: F401
    ov_energies as _ov_energies, screening_aux)
from src.SingleReference.GW.sum_over_poles import (  # noqa: F401
    compressible, denominators as _denominators, fit_poles, initial_poles,
    pole_amplitudes, pole_basis, pole_clearance, pole_pseudoinverse,
    qp_energy_sop, sigma_sop, sigma_sop_slope, sop_from_wc)
from src.gradients.contour_deformation_adjoint import (  # noqa: F401
    integral_term_backward, screening_chain)
from src.gradients.sum_over_poles_adjoint import (  # noqa: F401
    sigma_sop_backward, sop_partials)

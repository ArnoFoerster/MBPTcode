"""Analytic nuclear gradients: the adjoints of the correlation routes, and the
chains that carry them to the nuclei.

The forward physics lives in production -- `src.SingleReference`, `src.Base`
-- and every routine here differentiates THAT object rather than a second
copy of it. The route itself is the Lagrangian of Toelle (arXiv:2412.17085)
and Toelle, Kitsaras and Loos (arXiv:2507.02160), with the papers' iterative
BCH/truncated Taylor machinery replaced by exact closed forms.

REVERSE MODE, one module per forward.

  quasi_boson_adjoint
               the dense quasi-boson layer -- dRPA, the diagonal G0W0 solve and
               the four BSE variants -- as adjoint subclasses of production's
               own classes: the same amplitudes plus the Frechet maps
               (Daleckii-Krein, one eigh each). `RPA`, `QPqb` and `BSEqb` are
               those subclasses
  space_time_adjoint
               the ISDF/imaginary-time chain: adjoints of E_c^dRPA and
               Sigma_c(i.omega) w.r.t. (eps, X, D) at the forward pass's own
               cost; proj(tau) is the one whole object, frequencies stream in
               blocks, the self-energy lives in the auxiliary basis
  contour_deformation_adjoint
               the quasiparticle energy without an analytic continuation, and
               the push/adjoints half of the real-frequency screening backends
  sum_over_poles_adjoint
               the same quasiparticle with W modelled by M auxiliary poles, so
               Sigma_c(omega) is closed-form, the screening is never evaluated
               off the imaginary axis and there is no residue SET to freeze.
               Valence only -- `pole_clearance` says so
  bse_isdf     the BSE@GW eigenproblem on the ISDF factors, and the adjoints of
               the three-index blocks it is built from
  reaction_field_adjoint
               the continuum's Eq. (18) quasiparticle shift, built from the
               dressed and the bare factor
  solvated_rpa_energy
               the solvated dRPA fold, and the diagnostic that sweeps it
  isdf_derivatives
               nuclear derivatives of the ISDF factors themselves

ASSEMBLY, AND THE CHAINS BUILT ON IT.

  targets      (F, ERI, t) partials per target energy, and the Z-multiplier solve
  multipliers  the orbital-response (Z-vector) solve every route shares
  grad_engine  Fock folding, orbital response, skeleton assembly; the total
               gradient is pyscf's mean-field gradient plus this
  df_assembly  the same skeleton contraction through a three-index auxiliary
               factorization: 3 N^2 naux derivative integrals, no N^4 tensor
  factor_chain the frozen ISDF factorization every cubic chain differentiates
               (radii, pair layout, frames) and the three branches that carry
               adjoints on (eps, X_mo, D) to the nuclei
  isdf_mean_field
               the SCF force when the mean field's own exchange comes from those
               factors (src.Base.isdf_jk): one pyscf gradient of the functional
               with its exact exchange removed, plus the ISDF exchange skeleton.
               pyscf's gradient alone differentiates the FITTED interaction and
               is 4e-4 Ha/Bohr from that energy
  qp_space_time
               the streamed cubic quasiparticle gradient: contour deformation
               on space-time screening
  rpa_ground_state, excited_state
               E_HF + E_c^dRPA and the BSE@GW excited state, each a surface
               with its own energy and cubic gradient
  rpa_bse_surface, dense_surfaces
               the same surfaces on the DENSE quasi-boson route
  derivative_coupling, state_manifold
               d_mn = <Psi_m | d/dR Psi_n> between two BSE roots, and the root
               set the interstate quantities are read off

Two rules for anything built on this. Decide the QP set ONCE per surface: a
per-geometry Z > 0.5 filter makes the surface discontinuous. And converge the
SCF orbital gradient to 1e-11 -- the Lagrangian assumes F_offdiag = 0, and
symmetry hides the contamination on symmetric molecules.
"""
from src.Base.utils.time_frequency import sigma_transforms
from src.SingleReference.GW.contour_deformation import (
    cd_screening_contraction, qp_energy_cd, residue_pole_distance,
    residue_route_auto, sigma_cd, sigma_cd_slope)
from src.SingleReference.GW.imaginary_time import (selfenergy_block,
                                                   selfenergy_diag)
from src.SingleReference.GW.quasi_boson import build_rpa_AB, qp_energy_general
from src.SingleReference.GW.real_screening import real_frequency_weights
from src.SingleReference.GW.sum_over_poles import (compressible, fit_poles,
                                                   pole_amplitudes,
                                                   pole_clearance,
                                                   qp_energy_sop, sigma_sop,
                                                   sigma_sop_slope, sop_from_wc)
from src.SingleReference.LinearResponse.space_time import (
    laplace_representation_error, three_index_ov, three_index_slice)
from src.gradients.contour_deformation_adjoint import (
    ExplicitRealScreeningAdjoint as ExplicitRealScreening,
    LaplaceRealScreeningAdjoint as LaplaceRealScreening, qp_energy_cd_backward,
    sigma_cd_backward)
from src.gradients.df_assembly import (df_densities, df_eri_mo,
                                       two_electron_skeleton_df)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FactorChain
from src.gradients.grad_engine import correlation_gradient, correlation_gradients
from src.gradients.isdf_derivatives import isdf_exchange_skeleton
from src.gradients.isdf_mean_field import isdf_mean_field_gradient
from src.gradients.multipliers import solve_orbital_multipliers
from src.gradients.qp_space_time import qp_gradient_space_time
from src.gradients.quasi_boson_adjoint import (BSEqbAdjoint,
                                               BSEqbAdjoint as BSEqb,
                                               QPqbAdjoint,
                                               QPqbAdjoint as QPqb, RPAAdjoint,
                                               RPAAdjoint as RPA,
                                               frechet_funm_sym)
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.gradients.solvated_rpa_energy import solvated_correlation_energies
from src.gradients.space_time_adjoint import (chi0_backward, chi0_frequency,
                                              polarizability_backward,
                                              polarizability_tau,
                                              qp_energy_space_time,
                                              rpa_energy_and_adjoint,
                                              selfenergy_block_backward,
                                              selfenergy_diag_backward,
                                              three_index_ov_backward,
                                              three_index_slice_backward)
from src.gradients.sum_over_poles_adjoint import sigma_sop_backward, sop_partials
from src.gradients.targets import (add_z_contribution, chain_AB, qp_partials,
                                   qp_partials_with_shift, rpa_partials, solve_Z)

# `derivative_coupling` and `state_manifold` depend on `src.properties`, not
# present in this package, so those two modules are reached by importing them
# directly rather than through this namespace.

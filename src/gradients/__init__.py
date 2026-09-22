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
  contour_deformation_adjoint
               the quasiparticle energy without an analytic continuation, and
               the push/adjoints half of the real-frequency screening backends
  sum_over_poles_adjoint
               the same quasiparticle with W modelled by M auxiliary poles, so
               Sigma_c(omega) is closed-form, the screening is never evaluated
               off the imaginary axis and there is no residue SET to freeze.
               Valence only -- `pole_clearance` says so
  reaction_field_adjoint
               the continuum's Eq. (18) quasiparticle shift, built from the
               dressed and the bare factor

ASSEMBLY.

  targets      (F, ERI, t) partials per target energy, and the Z-multiplier solve
  multipliers  the orbital-response (Z-vector) solve every route shares
  grad_engine  Fock folding, orbital response, skeleton assembly; the total
               gradient is pyscf's mean-field gradient plus this
  df_assembly  the same skeleton contraction through a three-index auxiliary
               factorization: 3 N^2 naux derivative integrals, no N^4 tensor
  derivative_coupling, state_manifold
               d_mn = <Psi_m | d/dR Psi_n> between two BSE roots, and the root
               set the interstate quantities are read off

Two rules for anything built on this. Decide the QP set ONCE per surface: a
per-geometry Z > 0.5 filter makes the surface discontinuous. And converge the
SCF orbital gradient tightly -- the Lagrangian assumes F_offdiag = 0, and
symmetry hides the contamination on symmetric molecules.
"""
from src.gradients.bse_qb import BSEqb
from src.gradients.contour_deformation_adjoint import (
    ExplicitRealScreeningAdjoint as ExplicitRealScreening,
    LaplaceRealScreeningAdjoint as LaplaceRealScreening, qp_energy_cd_backward,
    sigma_cd_backward)
from src.gradients.df_assembly import (df_densities, df_eri_mo,
                                       two_electron_skeleton_df)
from src.gradients.grad_engine import correlation_gradient, correlation_gradients
from src.gradients.multipliers import solve_orbital_multipliers
from src.gradients.qb_core import RPA
from src.gradients.qp_qb import QPqb
from src.gradients.quasi_boson_adjoint import (BSEqbAdjoint, QPqbAdjoint,
                                               RPAAdjoint, frechet_funm_sym)
from src.gradients.sum_over_poles_adjoint import sigma_sop_backward, sop_partials
from src.gradients.targets import (add_z_contribution, chain_AB, qp_partials,
                                   qp_partials_with_shift, rpa_partials, solve_Z)

# `derivative_coupling` and `state_manifold` depend on `src.properties`, not
# present in this package, so those two modules are reached by importing them
# directly rather than through this namespace.

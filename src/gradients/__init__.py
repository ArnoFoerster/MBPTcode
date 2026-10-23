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
               (Daleckii-Krein, one eigh each): `RPAAdjoint`, `QPqbAdjoint`
               and `BSEqbAdjoint`
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
  mean_field_response
               the mean field's response on every route: the Fock partial's
               and Sigma_x - v_xc's orbital-rotation gradients, the multiplier
               solve and the skeletons they fold into
  df_assembly  the same skeleton contraction through a three-index auxiliary
               factorization: 3 N^2 naux derivative integrals, no N^4 tensor
  factor_chain the frozen ISDF factorization every cubic chain differentiates
               (radii, pair layout, frames) and the three branches that carry
               adjoints on (eps, X_mo, D) to the nuclei
  isdf_mean_field
               the SCF force when the mean field's own exchange comes from those
               factors (src.Base.factorization.isdf_jk): one pyscf gradient of the functional
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

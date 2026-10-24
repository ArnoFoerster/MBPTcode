"""Analytic nuclear gradients: the method-neutral core, and one subpackage per
method for its adjoints.

The forward physics lives in the electronic-structure layer
(`src.SingleReference`, `src.Base`), and every routine here differentiates
THAT object rather than a second copy of it. The GW-BSE route is the
Lagrangian of Toelle (arXiv:2412.17085) and Toelle, Kitsaras and Loos
(arXiv:2507.02160), with the papers' iterative BCH/truncated Taylor machinery
replaced by exact closed forms.

THE CORE, used by every method.

  surface      the surface protocol every property reads (the abstract base
               class and its ingredients), the finite-difference gradient
  surface_identity, evaluation
               what a force is a force of; one evaluation per geometry
  multipliers  the orbital-response (Z-vector) solve every route shares, and
               its orbital-rotation gradient
  mean_field_response
               the mean field's response on every route: the Fock partial's
               orbital-rotation gradient, the multiplier solve and the
               skeletons it folds into
  grad_engine, one_electron
               Fock folding, orbital response and skeleton assembly on the
               four-index route; the one-electron skeleton
  df_assembly  the same skeleton contraction through a three-index auxiliary
               factorization: 3 N^2 naux derivative integrals, no N^4 tensor
  skeleton_tiles
               the fitted Fock, interpolated exchange and moving-grid XC
               skeletons in fixed tiles over the ranks
  isdf_derivatives, isdf_fit_adjoint, isdf_mean_field
               nuclear derivatives of the ISDF factors, the adjoint of the
               row-distributed ISDF fit, and the SCF force when the mean
               field's own exchange comes from those factors
  df_correlation_force, df_correlation_force_tiled
               the force of a density-fitted correlated energy (MP2, ADC(2)),
               whole and tiled over ranks
  df_factors   the RI factors of a correlation fit the tiled force reads,
               the tiled MP2 force's default
  reference_mean_field, mean_field_surface, state_identity
               the reference mean field a chain stands on, its own surface, and
               the identity of orbitals and states across geometries
  state_models, solvated_surface
               the state models of state-specific solvation and the surface
               built from the medium, a state model and the loop

rpa/           the RPA ground state and the screening machinery GW-BSE builds
               on: the factor chain, the space-time adjoint, the chain
               settings, the solvated dRPA energy, the exchange
               double-counting correction
gwbse/         the GW-BSE adjoints and surfaces: the quasiparticle set, the
               contour and sum-over-poles adjoints, BSE on ISDF and its grid
               adjoint (`isdf_bse_adjoint`), the reverse
               fold, the conventions, the chain and its parts, the state
               manifold and derivative couplings, the fragment-diabatic BSE,
               the dense quasi-boson route, the surface factory, and the
               chains' own record bookkeeping (`diagnostics`)
adc/           ADC(2): its force, its state model and the fold's factors

Two rules for anything built on this. Decide the QP set ONCE per surface: a
per-geometry Z > 0.5 filter makes the surface discontinuous. And converge the
SCF orbital gradient to 1e-11 -- the Lagrangian assumes F_offdiag = 0, and
symmetry hides the contamination on symmetric molecules.
"""

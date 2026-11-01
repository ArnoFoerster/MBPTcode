# Warnings, flags and refusals

The code answers trouble in one of two ways, depending on where it comes from.

**A wrong input is refused.** A request that cannot give a meaningful answer
raises an error before any work is done, and the message names what is
wrong. Examples: an ADC(4) or higher calculation that does not say which
construction to use, an ISDF grid the shipped radii table has no row for,
EOM-CCSD from the coupled-cluster package (use pyscf's), a property that needs
an ingredient the surface does not provide (`MissingIngredient`), or two
different sets of numerical settings in one calculation.

**A numerical shortfall is warned about and flagged.** When a solver stops
short (an iteration runs out of cycles, a root is not found, a weight comes
out unphysical), the code keeps its last result, issues a Python warning, and
names the event in what it returns. The run goes on. Read the flags before
you trust a number: a flagged number is the best the solver reached, not a
converged one.

Some events are handed back by the solver itself, in the place the table
below gives (`info['converged']`, `out['no_root']`, ...). The others are
*named flags* (`src.Base.numerical_flags`): the solver warns and raises the
flag, and the function that returns the result collects every flag raised
inside it, on any thread or MPI rank, into one list of
`{'flag': name, 'message': text}` entries:

- `calc_qp_energy(..., out=out)`: `out['numerical_flags']`;
- an ADC solver: `solver.last_result['numerical_flags']` after `solve()`
  (every flag of this solver so far, since later solves reuse its
  amplitudes);
- an excited-state chain (`ExcitedStateChain`) and `DenseBSESurface`: the
  `numerical_flag_warnings` of their vertical-excitation and emission records,
  with the geometry each flag was first raised at; the dense surfaces' gradients
  also return them in their diagnostics, `diagnostics['numerical_flags']`;
- a photophysics stage: the stage record's `numerical_flags`, as
  `{flag: [messages]}`.

A flag raised outside all of these, by a solver called on its own, is
warned about and nothing more.

## The flags

| where | what stopped short | warning | flag |
|---|---|---|---|
| quasiparticle equation, every route | no root in the widest scan window | `QPNoRoot` | the root is NaN, named flag `'no_root'`; `out['no_root']` of the root finder |
| quasiparticle equation, every route | the root kept has pole strength Z outside (0, 1]; the root is kept as solved | `QPWeightOutside` | named flag `'z_outside'`; `out['z_outside']` of the root finder; on a gradient chain `qp_z_outside` |
| excited-state chain on the analytic continuation (`--residues pade`) | the Pade continuation of a quasiparticle-set member failed numerically; it takes the correction of its nearest solved member (a failed HOMO or LUMO is refused) | `QPContinuationFailed` | named flag `'qp_continuation_failed'`; the chain's `qp_diagnostics['continuation_failed']` |
| quasiparticle equation, a root followed from a seed | the largest pole strength has moved to another root | `RuntimeWarning` | the root is kept on its branch; `QPqb.branch_log`, on a gradient chain `qp_branch_warnings` |
| quasi-boson BSE, the set decided at the reference | an orbital's quasiparticle root has pole strength Z of 0.5 or less | `QPWindowMeanField` | named flag `'qp_window_mean_field'`; the BSE diagonal keeps that orbital's mean-field energy |
| evGW | the loop ran out of cycles | `EvGWNotConverged` | named flag `'evgw_not_converged'`; `info['converged']` False |
| qsGW | the loop ran out of cycles before a fixed point | `QsGWNotConverged` | named flag `'qsgw_not_converged'`; `info['converged']` False; the spectrum is the last iterate |
| screening on the imaginary axis | 1 - chi0(i nu) is not positive definite, so Cholesky fails; solved by LU | `ScreeningNotPositive` | named flag `'screening_not_positive_definite'` |
| quasiparticle Hamiltonian, a group of more than 24 orbitals | the GMRES on its Newton equation stopped above its tolerance; the last iterate is kept and the Newton goes on (its stop is the exact residual) | `SubspaceGMRESShort` | named flag `'qp_subspace_gmres_not_converged'` |
| sum-over-poles vector fitting | the least squares failed; solved by another LAPACK driver (gelss, then gelsy) | `VectorFitFallback` | named flag `'vector_fit_lstsq_fallback'` |
| dRPA correlation energy (imaginary axis, space-time and periodic) | det(1 - chi0) is not positive at a frequency; log\|det\| is used, and the energy is not the dRPA one | `NonPositiveDielectric` | named flag `'nonpositive_det'`; with a `flags` dict passed, `flags['nonpositive_det']`, the frequencies (and transfers) where it happened |
| periodic THC GW | no tabulated minimax grid reaches the asked accuracy; the largest grid runs | `MinimaxGridShort` | with `return_diagnostics=True`, `diag['numerical_flags']['minimax_tau_target_missed']`, the accuracy reached |
| Casida / BSE Davidson | a root above its residual tolerance, or a root an unsettled higher pair may lie below | `RuntimeWarning` | `info['davidson_flags']`: `'davidson_not_converged'`, `'davidson_unconfirmed'` |
| orbital response: the CPHF / Z-vector solve of a relaxed density (`Base.pyscf.cphf`) and the gradient layer's multiplier solve (`gradients.multipliers`) | the residual stayed above its tolerance; the last iterate is kept | `CphfNotConverged` | named flag `'orbital_response_not_converged'`, also in a `flags` list the caller hands the multiplier solve |
| CC amplitudes and Lambda (also pyscf's CCSD under the CC density) | out of iterations | `CCNotConverged` | `converged` False, and the named flag `'cc_not_converged'` |
| an iteration inside an ADC route (the CCD/CCSD or Brueckner amplitudes of Faddeev- and CC-ADC(3), the ladder CCD, a FLEX Lanczos resolvent, the Lanczos inverse square root of the Faddeev metric) | out of iterations or steps | `ADCNotConverged` | `out['converged']`, `info['converged']` or `solve_qp(...)['converged']` False, and the named flags `'cc_amplitudes_not_converged'`, `'brueckner_not_converged'`, `'ladder_amplitudes_not_converged'`, `'flex_resolvent_not_settled'`, `'inverse_sqrt_not_settled'` |
| coupled cluster as an active-space solver (`CoupledClusterSolver`, the `'ccsdt'` / `'ccsdtq'` backends of `active_space.solve`) | the Hartree-Fock reference in the active space, or the amplitudes, out of cycles; the last iterate's energy is returned | `ActiveSpaceCCNotConverged` | `solver.last_result`: `reference_converged`, `converged`, and in `numerical_flags` the named flags `'active_space_reference_not_converged'`, `'active_space_cc_not_converged'` |
| EOM-CC | unconverged roots | `UnconvergedEOMRoots` | `EOMResult.converged`, and the named flag `'eom_roots_not_converged'` |
| EOM-CC | a state whose left eigenvector was not found, so it cannot be paired with its right one; its transition densities are zero, and G0W@CC drops its pole from Sigma_c | `UnpairedEOMStates` | `EOMResult.unpaired_states`, and the named flag `'eom_states_unpaired'` |
| EOM-CC, non-Hermitian solve | eigenvalues with an imaginary part above `EOM_IMAG_TOL`; the real parts are kept | `ComplexEOMEigenvalues` | `EOMResult.complex_eigenvalues`, and the named flag `'eom_complex_eigenvalues'` |
| ground-state relaxation | the walk stopped before converging | `UnconvergedRelaxation` | `mf.relaxation`: `relaxation_converged`, `opt_grad_max` |
| excited-state walk | the walk did not converge | `WalkNotRelaxed` | `'walk_not_converged'` |
| refreeze loop | the energy still moves by more than 3 meV after 4 passes | `RefreezeNotConverged` | `refreeze='not converged'`, `'refreeze_not_converged'` |
| state-specific solvation (`environments.solvation_loop`) | the loop ran out of cycles (its last cycle is returned), or the state model returned no number (the result is NaN) | `SolvationNotConverged` | `LoopResult.flags`, beside the state model's own flags of that cycle: `'solvation_loop_not_converged'`, `'solvation_loop_stopped'` |
| a scheme-B minimum (`photophysics.state_specific.walk_vs_vertical`) | G_eq of the state at a walk's minimum on the walk's surface and on the vertical's there differ by more than the bar (3 meV after a refreeze, else ten loop tolerances) | `RuntimeWarning` | `walk_vs_vertical_flag` in the record, and the stage pipeline's `'walk_vs_vertical'` |
| photophysics driver (`photophysics.solvated`) | no memory budget given (no `--max-memory`, SLURM allocation or `PYSCF_MAX_MEMORY`): each rank keeps pyscf's default, and a tiled force whose estimate exceeds it is refused | `RuntimeWarning` | named flag `'memory_budget_unknown'`; the record's `run_flags` |
| a scheme-B T1 stage on the full BSE (`photophysics.stages.tda_single_point`) | the triplet walk stopped at the full kernel's instability (`CasidaInstability`, `CasidaBreakdown`): T1 is the full-BSE single point at the declared Tamm-Dancoff T1 minimum, not a minimum of the full kernel | `TdaSinglePoint` | `'tda_minimum_single_point'`, and `tda_single_point` (why, where the walk stopped, the minimum's sha256) in the record |
| a force on a loosely converged mean field | the SCF stopped above its force tolerance | `RuntimeWarning` | `scf_flags` |
| an SCF of the photophysics chain | out of SCF cycles; the stage goes on | `SCFNotConverged` | the stage's `numerical_flags`: `'scf_not_converged'`, the count |
| Franck-Condon rate integral | the time integrand had not decayed; the rate is computed on the truncated density | `FCIntegrandNotDecayed` | the rate table's `numerical_flags`: `'fc_integrand_not_decayed'` |
| Herzberg-Teller dipole derivative by finite differences (`dipole_derivative`) | along a mode the dipole changes mostly evenly in the step, or a displaced dipole points against mu0 (a phase flip between the displaced solves, or a dipole far from linear); the slope is kept | `TransitionDipolePhase` | `TransitionDipole.suspect_modes` |
| ISC/RISC rate chain on a Duschinsky density (`photophysics.rate_chain`) | a density below 1e-13 of its band's peak, the real-time integral's roundoff floor (a deeply activated crossing); the rate is kept and is not resolved | `FCDensityAtRoundoff` | the chain's `numerical_flags`: `'fc_density_at_roundoff'` (and `'fc_integrand_not_decayed'` as above) |
| Duschinsky model from two Hessians (`duschinsky_model`) | the overlap of the two mode sets is not orthogonal to 2e-2 (axis switching the alignment did not remove); its nearest orthogonal matrix is used | `DuschinskyNotOrthogonal` | `DuschinskyModel.orthogonality` |
| the photophysics rate table | the ISDF-vs-DF acceptance check failed at a stage the table reads; the table is computed | `AcceptanceCheckFailed` | the rate table's `numerical_flags`, under that stage: `'isdf_df_check_failed'` |
| ISDF points pivoted for the AO-product fit (`ao_product_fit.pivoted_points`) | the parent grid's Gram matrix has a numerical rank below the asked point count; the points up to that rank are kept | `ISDFPivotShort` | named flag `'isdf_pivot_rank_short'` |
| the tight SCF of a dense GW-BSE surface (`tight_rhf`) | out of SCF cycles; the last iterate is kept | `RuntimeWarning` | named flag `'scf_not_converged'`, also in `mf.flags` |
| finite-difference Hessian (`surface_hessian`, the mode Hessian) | the two halves differ by more than 1e-3 of the largest force constant (forces not reproducible, or a grid too coarse for the step); the Hessian is kept symmetrized | `HessianAsymmetric` | `info['hessian_asymmetric']`, `'hessian_asymmetric'` in `info['flags']`; a saddle check's record and its stage's `numerical_flags`: `'hessian_asymmetric'` |
| fragment diabats: the localization response | MINRES stopped above its tolerance; the last iterate is kept | `RuntimeWarning` | `'localization_response_not_converged'` |
| fragment diabats: a diabat's response | the solve stopped above its tolerance; the last iterate is kept | `RuntimeWarning` | `'diabat_response_not_converged'` |
| quasiparticle equation, a root followed from a seed in the finite-difference reference of the quasi-boson route (`quasi_boson.qp_energy_general`) | the followed root has pole strength Z at or below 0.5; it is kept on its branch | `QPLowWeightRoot` | named flag `'qp_followed_root_low_weight'` |
| upfolded BSE (`BSE.bse_upfolded`) | a root with a significant imaginary part (the dynamical BSE has no real root there): returned as NaN | `RuntimeWarning` | `'upfolded_bse_complex_roots'` in the `flags` list the caller hands in |
| upfolded BSE (`BSE.bse_upfolded`) | a root with Re <= 0 is left out of the returned excitation energies (the dense eigenvalues, or the Ritz values of the Davidson's final subspace) | `RuntimeWarning` | `'upfolded_bse_dropped_roots'` in the `flags` list the caller hands in |
| upfolded BSE, Davidson | out of cycles, or a linearly dependent correction above tolerance; the Ritz pairs are kept | `RuntimeWarning` | `'upfolded_bse_not_converged'` in the `flags` list the caller hands in; `refuse_unconverged=True` raises instead |
| excitation ADC(2) as a state model and its force (`gradients.adc.ee_state_model`, `ee_gradient`) | a root not converged (kept as solved); a root index the solve did not return (NaN energy, density and force, which a walk rejects); a followed root that overlaps the roots solved by less than 0.5, or two roots within 0.2 of each other in overlap (the best overlap is taken) | `FollowingWarning` | `info['flags']`: `'adc_roots_not_converged'`, `'adc_root_missing'`, `'root_following_lost'`, `'root_following_mixed'` |
| excitation ADC(2) force | the force re-solved the roots and its root is not the call's: under `follow='index'` its transition amplitude overlaps the call's root by less than 0.5; at the call's own field its omega differs from the call's | `FollowingWarning` | the force's `info['flags']`: `'force_root_order_differs'`, `'force_omega_mismatch'` |
| folded excitation ADC(2) (`eeADC.ee_fold.solve_folded`), each event's solve carrying on | a root, level or branch above its tolerance (`'root_not_converged'`, `'level_not_converged'`, `'branch_root_not_converged'`, `'roots_not_converged'`); model roots or a model branch not converged (`'model_roots_not_converged'`, `'model_branch_unsolved'`); a vector or level whose overlap with its previous one fell below 0.5, kept (`'root_crossing'`, `'level_crossing'`); a level or block of branches not degenerate in A_eff, re-solved per root (`'level_split'`, `'branches_split'`); two roots on one, the copy dropped (`'duplicate_root_dropped'`); fewer roots than asked (`'roots_missing'`); roots reaching the lowest doubles energy, solved per root (`'block_reaches_doubles'`), or lying above it, not checked by the count (`'roots_above_doubles_unchecked'`); a state below the roots that a warm solve's start lacked and only the full check's random-vector walk found (`'fold_root_crossed'`: earlier warm solves may have followed another state; the ADC(2) state model's `confirm` raises it at a walk's converged geometry, and the walk then ends not converged) | `RuntimeWarning` | `FoldResult.flags`, which the ADC(2) state model copies into `info['flags']`; at a walk's converged geometry `info['walk_flags']` |
| fragment partition of a singles operator, GW-BSE or ADC(2) (`fragments.partition`) | a block Davidson short of its tolerance (`'block_eigenpairs_not_converged'`); a Q-space lowest-root pass that stopped above tolerance (`'q_lowest_unconverged'`); the pole guard's Newton (`'pole_guard_newton_not_converged'`); the resolvent's conjugate gradients (`'resolvent_not_converged'`); the Feshbach root's Newton (`'feshbach_root_not_converged'`); the last iterate is kept each time | `RuntimeWarning` | `FragmentPartition.guard_flags` (`{name: detail}`) for the partition's solves, which every diabat gradient (`DiabaticGradient.gradient`, `element`, `root_gradient`) carries in its `flags`, each also a named flag; the `flags` dict handed to `FeshbachOracle.root` |

The photophysics stages gather every named flag raised inside them, and
their own events, into the stage record's `numerical_flags`, and the rate
table gathers those by stage, so it says which numbers stand on a flagged
stage.

G0W@CC (`calc_qp_energy(..., polarizability='CCSDT', out=out)`) records the
CC and EOM events above in `out['numerical_flags']`: a CC amplitude solve out
of iterations, unconverged EOM roots, and an unpaired or complex EOM state.

A matrix-free BSE solve can be made strict again:
`solve_bse_isdf(..., refuse_unconverged=True)` raises `DavidsonNotConverged`
instead of flagging.

## Example

An evGW loop cut off after two cycles returns its last iterate, warns, and
says so in its record:

```python
import warnings

from pyscf import dft, gto

from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.evGW import evgw_eigenvalues

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = dft.RKS(mol, xc='pbe0').density_fit().run()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    eps, info = evgw_eigenvalues(mf, mol, mode='casida',
                                 numerics=Numerics(evgw={'max_cycle': 2}))

print('converged:', info['converged'])        # False
for w in caught:
    print(type(w.message).__name__, w.message)
```

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
writes a named flag into the result or record it returns. The run goes on.
Read the flags before you trust a number: a flagged number is the best the
solver reached, not a converged one.

## The flags

| where | what stopped short | warning | flag |
|---|---|---|---|
| quasiparticle equation | no root in the widest scan window | `QPNoRoot` | the root is NaN, `'no_root'` |
| quasiparticle equation | the largest pole strength Z lies outside (0, 1] | `QPWeightOutside` | `'z_outside'`; on a gradient chain `qp_z_outside` |
| quasiparticle equation, a root followed from a seed | the followed root has pole strength Z of 0.5 or less | `QPLowWeightRoot` | the root is kept on its branch, `'qp_followed_root_low_weight'` |
| quasi-boson BSE | an orbital's quasiparticle root has pole strength Z of 0.5 or less | `QPWindowMeanField` | the BSE diagonal keeps that orbital's mean-field energy, `'qp_window_mean_field'` |
| evGW | the loop ran out of cycles | `EvGWNotConverged` | `info['converged']` False |
| qsGW | the loop ran out of cycles before a fixed point | `QsGWNotConverged` | `info['converged']` False; the spectrum is the last iterate |
| screening on the imaginary axis | 1 - chi0(i nu) is not positive definite, so Cholesky fails; solved by LU | `ScreeningNotPositive` | `'screening_not_positive_definite'` |
| sum-over-poles vector fitting | the least squares failed; solved by another LAPACK driver (gelss, then gelsy) | `VectorFitFallback` | `'vector_fit_lstsq_fallback'` |
| dRPA correlation energy (space-time and periodic) | det(1 - chi0) is not positive at a frequency; log\|det\| is used, and the energy is not the dRPA one | `NonPositiveDielectric` | `flags['nonpositive_det']`, the frequencies (and transfers) where it happened |
| periodic THC GW | no tabulated minimax grid reaches the asked accuracy; the largest grid runs | `MinimaxGridShort` | with `return_diagnostics=True`, `diag['numerical_flags']['minimax_tau_target_missed']`, the accuracy reached |
| Casida / BSE Davidson | a root above its residual tolerance, or a root an unsettled higher pair may lie below | `RuntimeWarning` | `info['davidson_flags']`: `'davidson_not_converged'`, `'davidson_unconfirmed'` |
| orbital response (CPHF, Z-vector) | the residual stayed above its tolerance | `CphfNotConverged` | `'orbital_response_not_converged'` |
| CC amplitudes and Lambda | out of iterations | `CCNotConverged` | `converged` False |
| an iteration inside an ADC route (the CCD/CCSD or Brueckner amplitudes of Faddeev- and CC-ADC(3), the ladder CCD, a FLEX Lanczos resolvent, the Lanczos inverse square root of the Faddeev metric) | out of iterations or steps | `ADCNotConverged` | `out['converged']`, `info['converged']` or `solve_qp(...)['converged']` False |
| EOM-CC | unconverged roots | `UnconvergedEOMRoots` | `EOMResult.converged` |
| ground-state relaxation | the walk stopped before converging | `UnconvergedRelaxation` | `mf.relaxation`: `relaxation_converged`, `opt_grad_max` |
| excited-state walk | the walk did not converge | `WalkNotRelaxed` | `'walk_not_converged'` |
| refreeze loop | the energy still moves by more than 3 meV after 4 passes | `RefreezeNotConverged` | `refreeze='not converged'`, `'refreeze_not_converged'` |
| state-specific solvation | the loop ran out of cycles | `SolvationNotConverged` | the loop's `flags` |
| a force on a loosely converged mean field | the SCF stopped above its force tolerance | `RuntimeWarning` | `scf_flags` |
| an SCF of the photophysics chain | out of SCF cycles; the stage goes on | `SCFNotConverged` | the stage's `numerical_flags`: `'scf_not_converged'`, the count |
| Franck-Condon rate integral | the time integrand had not decayed; the rate is computed on the truncated density | `FCIntegrandNotDecayed` | the rate table's `numerical_flags`: `'fc_integrand_not_decayed'` |
| the photophysics rate table | the ISDF-vs-DF acceptance check failed at a stage the table reads; the table is computed | `AcceptanceCheckFailed` | the rate table's `numerical_flags`, under that stage: `'isdf_df_check_failed'` |
| the tight SCF of a dense GW-BSE surface | out of SCF cycles; the last iterate is kept | `RuntimeWarning` | `mf.flags`: `'scf_not_converged'` |
| finite-difference Hessian (`surface_hessian`, the mode Hessian) | the two halves differ by more than 1e-3 of the largest force constant (forces not reproducible, or a grid too coarse for the step); the Hessian is kept symmetrized | `HessianAsymmetric` | `info['hessian_asymmetric']`, `'hessian_asymmetric'` in `info['flags']`; a saddle check's record and its stage's `numerical_flags`: `'hessian_asymmetric'` |
| fragment diabats: the localization response | MINRES stopped above its tolerance; the last iterate is kept | `RuntimeWarning` | `'localization_response_not_converged'` |
| fragment diabats: a diabat's response | the solve stopped above its tolerance; the last iterate is kept | `RuntimeWarning` | `'diabat_response_not_converged'` |

The photophysics drivers gather every flag of a stage into its record
(`numerical_flags`), so the rate table says which numbers stand on a flagged
stage.

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
from src.SingleReference import evgw_eigenvalues

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

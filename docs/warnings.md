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
| evGW | the loop ran out of cycles | `EvGWNotConverged` | `info['converged']` False |
| Casida / BSE Davidson | a root above its residual tolerance, or a root an unsettled higher pair may lie below | `RuntimeWarning` | `info['davidson_flags']`: `'davidson_not_converged'`, `'davidson_unconfirmed'` |
| orbital response (CPHF, Z-vector) | the residual stayed above its tolerance | `CphfNotConverged` | `'orbital_response_not_converged'` |
| CC amplitudes and Lambda | out of iterations | `CCNotConverged` | `converged` False |
| EOM-CC | unconverged roots | `UnconvergedEOMRoots` | `EOMResult.converged` |
| ground-state relaxation | the walk stopped before converging | `UnconvergedRelaxation` | `mf.relaxation`: `relaxation_converged`, `opt_grad_max` |
| excited-state walk | the walk did not converge | `WalkNotRelaxed` | `'walk_not_converged'` |
| refreeze loop | the energy still moves by more than 3 meV after 4 passes | `RefreezeNotConverged` | `refreeze='not converged'`, `'refreeze_not_converged'` |
| state-specific solvation | the loop ran out of cycles | `SolvationNotConverged` | the loop's `flags` |
| a force on a loosely converged mean field | the SCF stopped above its force tolerance | `RuntimeWarning` | `scf_flags` |

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

from src.SingleReference import evgw_eigenvalues

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz', verbose=0)
mf = dft.RKS(mol, xc='pbe0').density_fit().run()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    eps, info = evgw_eigenvalues(mf, mol, mode='casida', max_cycle=2)

print('converged:', info['converged'])        # False
for w in caught:
    print(type(w.message).__name__, w.message)
```

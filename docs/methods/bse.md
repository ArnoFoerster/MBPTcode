# BSE

The Bethe-Salpeter equation (BSE) gives neutral excitation energies, the
optical spectrum of a molecule, on top of a GW calculation. GW supplies the
quasiparticle energies, which set the energy of each electron-hole pair; the
BSE then lets the electron and the hole attract each other through the
screened Coulomb interaction W, which binds them into an exciton.

## Running it

```python
from pyscf import gto, dft

from src.Base.numerics.numerics import Numerics
from src.SingleReference.BSE.solve import solve_bse_isdf

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz')
mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
mf.kernel()

numerics = Numerics(isdf={'level': 'G1'}, bse={'nroots': 5})
omega, X, Y, info = solve_bse_isdf(mf, mol, mol.nelectron // 2,
                                   numerics=numerics)
```

`omega` holds the excitation energies in Hartree, `X` and `Y` the excitation
and de-excitation amplitudes of each root.

Every numerical choice comes in through one `Numerics` object
(`src.Base.numerics.numerics`): here the interpolation grid of the ISDF fit,
named by its accuracy level (`'G1'` keeps the three lowest roots within
8 meV of `solve_bse_df`; `'G2'` and `'G3'` are tighter), and the number of
roots. There is no default grid: a call that names none is refused.

Two functions solve the same equations and differ only in how the Coulomb
interaction is represented:

| function | interaction | use it for |
|---|---|---|
| `solve_bse_isdf` | interpolative separable density fitting (ISDF) | large molecules; the GW step then scales cubically |
| `solve_bse_df` | pyscf's own density fitting | smaller molecules, or as a reference |

`spin='triplet'` gives triplet excitations instead of singlets, and
`tda=True` the Tamm-Dancoff BSE (B = 0). `solve_bse(mf, solver=...,
integrals=...)` reaches the same drivers and the dense route from one call.
All three live in `src/SingleReference/BSE/solve.py`.

## Which GW

By default the quasiparticle energies come from G0W0 on the mean-field
orbitals. `self_consistency='evGW'` iterates the GW eigenvalues to self
consistency first and uses them both for the pair energies and for W. If you
pass your own quasiparticle energies with `qp=`, `screen_at` says whether W is
built from them (`'qp'`) or from the mean field (the G0W0 convention, the
default).

## How the equations are solved

The BSE matrix is never built. A Davidson iteration only ever needs the
matrix applied to a few trial vectors, and the factorized interaction makes
that product cheap. A few points the user may want to know:

- **Convergence.** The Davidson divides each correction by an estimate of the
  matrix diagonal. By default this estimate includes the electron-hole
  attraction, which usually needs noticeably fewer iterations;
  `Numerics(bse={'preconditioner': 'bare'})` uses the plain pair energies.
  Both give the same roots. A root that does not converge within the cycle
  cap is kept, warned about and named in `info['davidson_flags']`
  ([Warnings, flags and refusals](../warnings.md)).
- **Stability check.** The BSE as solved here assumes that the ground state is
  stable. With `probe=True` the code checks this after the solve and refuses
  the roots if the reference is unstable, the usual situation for some
  triplets on top of Hartree-Fock. The stand-alone drivers skip the check
  unless asked; the gradient chains run it after every full-BSE (not
  Tamm-Dancoff) matrix-free solve.
- **Memory and parallel runs.** The Davidson keeps as many trial vectors as
  the available memory allows. Under MPI the trial vectors are split over the
  ranks; the roots agree with a serial run to within the convergence
  tolerance.
- **Starting from the last geometry.** Along a geometry optimization,
  consecutive solves are nearly the same problem. An `ExcitedStateChain`
  therefore starts each matrix-free solve from the roots of the last geometry
  the walk accepted, carried into the new orbitals through their overlap, so
  sign flips and rotations of degenerate orbitals do not spoil the guess
  (`chain.davidson_warm_start`, a `DavidsonWarmStart`; set it to None to
  solve every geometry cold). It changes how many iterations a solve takes,
  not the roots it converges to; a step larger than 0.5 Bohr starts cold.

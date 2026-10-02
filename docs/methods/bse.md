# BSE

The Bethe-Salpeter equation (BSE) gives neutral excitation energies, the
optical spectrum of a molecule, on top of a GW calculation. GW supplies the
quasiparticle energies, which set the energy of each electron-hole pair; the
BSE then lets the electron and the hole attract each other through the
screened Coulomb interaction W, which binds them into an exciton.

## Running it

```python
from pyscf import gto, dft
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz')
mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
mf.kernel()

omega, X, Y, info = solve_bse_isdf(mf, mol, mol.nelectron // 2, nroots=5)
```

`omega` holds the excitation energies in Hartree, `X` and `Y` the excitation
and de-excitation amplitudes of each root.

Two functions solve the same equations and differ only in how the Coulomb
interaction is represented:

| function | interaction | use it for |
|---|---|---|
| `solve_bse_isdf` | interpolative separable density fitting (ISDF) | large molecules; the GW step then scales cubically |
| `solve_bse_df` | pyscf's own density fitting | smaller molecules, or as a reference |

`spin='triplet'` gives triplet excitations instead of singlets.

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
  `preconditioner='bare'` uses the plain pair energies. Both give the same
  roots.
- **Stability check.** The BSE as solved here assumes that the ground state is
  stable. With `probe=True` (the default) the code checks this after the
  solve and refuses the roots if the reference is unstable, the usual
  situation for some triplets on top of Hartree-Fock.
- **Memory and parallel runs.** The Davidson keeps as many trial vectors as
  the available memory allows. Under MPI the trial vectors are split over the
  ranks; the roots agree with a serial run to within the convergence
  tolerance.

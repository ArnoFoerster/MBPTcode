# Low-scaling factorization

**Low-scaling factorization** — the separable RI of Duchemin and Blase
([J. Chem. Phys. 150, 174120 (2019)](https://doi.org/10.1063/1.5090605)),
with optimized atomic interpolation grids. It backs the space-time GW route,
`solve_bse_isdf` (a BSE on the same factors), and ISDF-J/K for the SCF, which
replaces the density-fitted `cderi` and so removes the three-index tensor from
the memory budget.

## Augmented basis sets

The published fit is trained on each AO times the AOs up to d, plus the
auxiliary functions. On an augmented basis it misses the densities of the
near-null AO combinations the diffuse functions bring, and a BSE on such
factors finds a spurious low root: for CBP in aug-cc-pVDZ (LRC-wPBEh, every
AO combination kept) the lowest singlet comes out 2.6 eV below the
density-fitted reference, and adding grid points does not help.

The remedy is a fit trained on every AO product with the residual measured in
the Coulomb metric (`test_set='ao_products'`), on points chosen for it: a
pivoted Cholesky factorization picks `pivot_points` points from the atomic
grid of the level with every radial shell doubled, those whose AO products
the points already taken represent worst. On CBP and PXZ-TRZ in
aug-cc-pVDZ, 9000 to 10000 such points put the lowest three singlets and
triplets within 0.6 meV of the density-fitted reference, with 2.5 to 3 times
fewer points than the G3 grid and 6 to 10 times faster.

```python
from pyscf import gto, scf
from src.Base.numerics.numerics import Numerics
from src.SingleReference.BSE.solve import solve_bse_isdf

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
            basis='aug-cc-pvdz')
mf = scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-ri').run()
numerics = Numerics(isdf={'level': 'G3', 'test_set': 'ao_products',
                          'pivot_points': 800})
omega, X, Y, info = solve_bse_isdf(mf, mol, mol.nelectron // 2, qp=None,
                                   numerics=numerics)
```

The same `Numerics` serves the gradient chains: the points are chosen once,
at the reference geometry, and then move with their atoms, so the energy is
a smooth function of the geometry and the forces are its derivative. Under
MPI the points are divided over the processes in fixed blocks: the selection
keeps only its factor columns (about 16 GB over all processes for 20000 of
100000 candidate points) and never the candidates' whole Gram matrix, and
picks the same points on any number of processes; the fit and its force
divide the three-centre integrals over the processes, and change with their
number only in the last digits. The fit is not available on the
row-distributed fit (`fit='rows'`).

The ISDF-K mean field takes the same two settings from the same `Numerics`
(its parent is the grid of `numerics.scf.grid_level`): its exchange is built
on the AO-product fit, and where its parent and point count are those of the
post-SCF factors, the two share one selection and one fit. On formaldehyde in
aug-cc-pVDZ (LRC-wPBEh, 900 points) its orbital energies stay within 0.8 meV
of the density-fitted mean field, the dipole within 4e-4 D and the force
within 1e-4 Ha/Bohr; its total energy carries a bias of up to a few mHa on
large molecules, which excitation energies do not see.

# Analytic nuclear gradients

The code computes the forces on the nuclei for the states it computes the
energy of: the dRPA ground state, BSE@GW excited states, GW quasiparticle
energies, and the ADC(2) excited states with their MP2 ground state. The
forces are analytic: each is the exact derivative of the energy
the same calculation reports, not a finite difference. They follow the
Lagrangian formulation of Toelle,
[arXiv:2412.17085](https://arxiv.org/abs/2412.17085), and of Toelle, Kitsaras
and Loos, [arXiv:2507.02160](https://arxiv.org/abs/2507.02160), with the
iterative expansions of those papers replaced by closed forms. Geometry
optimization, vibrational analysis and rates ([properties](properties.md)) are
built on them.

## Running it

```python
from pyscf import dft, gto

from src.Base.declaration import Excitation, GroundState
from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.surface_factory import potential_energy_surface


def pbe0(mol):
    mf = dft.RKS(mol, xc='pbe0')
    mf.grids.level = 4                      # see below
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    mf.kernel()
    return mf


mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz')
s1 = potential_energy_surface(mol, pbe0, ground_state=GroundState('dft', 'pbe0'),
                              excitation=Excitation('singlet'),
                              numerics=Numerics(isdf={'level': 'G1'}))
grad, energy, info = s1.total_gradient(mol)
```

`grad` is dE/dR in Hartree/Bohr, one row per atom, `energy` the total energy
of the first singlet and `info['omega']` its excitation energy. The mean field
is passed as a function of the geometry, because every displaced geometry
needs its own; it must be converged tightly, since the gradient assumes the
orbitals are exactly stationary. A Kohn-Sham mean field should integrate its
exchange-correlation potential on pyscf's grid level 4: on the default level
3, range-separated hybrids such as LRC-wPBEh move the orbital energies in
small steps between nearby geometries, which GW carries into every
excitation energy and which spoils a force. The staged TADF run (`python -m
src.photophysics`) builds its mean fields on level 4 (`ScfNumerics.xc_grid`,
`--xc-grid`); the tests that check a force against finite differences use
level 5.
`numerics` names the ISDF interpolation grid by its accuracy level, as for
the [BSE](bse.md). `ground_state` says which energy the state
sits on (here the PBE0 total energy; `GroundState('rpa', 'hf')` gives the dRPA
ground state on Hartree-Fock), `excitation` which state (omit it for the
ground state), and `environment` an optional solvent or point charges.

On the dRPA ground state (`GroundState('rpa', 'hf')`) and the states built on
it, two routes compute the same gradient:

| keywords | cost | use it for |
|---|---|---|
| default (`chi0='space-time'`, `factorization='isdf'`) | cubic in system size | larger molecules |
| `chi0='dense-qb'`, `factorization='four-index'` | O(N^6) | small molecules, or as a reference |

## How it works

The energy is written as a Lagrangian that is stationary in every quantity
the calculation solves for, so its derivative needs only the explicit
dependence on the nuclei plus one orbital-response (Z-vector) equation. That
one solve serves every energy a calculation reports. Each way of evaluating
the GW self-energy ([GW](gw.md): Pade, contour deformation, sum over poles)
and each environment ([Environments](environments.md)) has its own
derivative, so the force always belongs to the energy that was actually
computed.

On the cubic route the Coulomb interaction is represented by an ISDF fit, and
the gradient differentiates that same fit, the one the SCF, GW and BSE steps
used. Under MPI the work of a force is split over the ranks in fixed pieces,
so the result does not depend on the number of ranks beyond the order of one
final sum (see [Threads and MPI](../parallel.md)).

Several excited states at one geometry, for example a singlet and a triplet,
can share one evaluation: the mean field, the factors, the screening and the
quasiparticle energies are computed once, then each spin is solved and each
force is taken from them. The numbers are the same as evaluating each state
on its own.

The same reverse pass gives the orbital-relaxed density of an excited state,
the derivative of its excitation energy with respect to the one-electron
Hamiltonian, which state-specific solvation reads
([Environments](environments.md)). It stops at the orbital response and does
not assemble the force it does not need.

## Interchangeable surfaces

Every surface derives from one abstract class,
`PotentialEnergySurface` in `src/gradients/surface.py`: it gives the total
energy and gradient at a geometry, its mean field, a label, and `refreeze`
(see [properties](properties.md)). A surface also declares the further
ingredients it provides (its excitation energy, every excited state solved
at a geometry, ...; `INGREDIENTS` in the same module lists them). A property
asks for the ingredients it needs, so it works with any surface that
provides them, whatever method is behind it, and a surface that lacks one is
refused at the call with `MissingIngredient`, naming what is missing.

The code that every method uses (the orbital response, the derivative
integrals, the state models, the force of a density-fitted correlated
energy) lives in `src/gradients/` itself; each method's own part lives in a
subpackage: `rpa/` (the dRPA ground state and its screening), `gwbse/`
(GW-BSE, built on `rpa/`) and `adc/` (ADC(2)). See
[Source layout](../layout.md).

## ADC(2) forces

The ADC(2) force runs on the folded, density-fitted route
(`src/gradients/adc/`): the MP2 ground state plus the ADC(2) excitation
energy, with one orbital-response solve, on whole tensors or on tiles split
over MPI ranks. It is used through the state models of state-specific
solvation ([Environments](environments.md)), in the gas phase and in a
continuum, and drives the geometry optimizations of `python -m
src.photophysics solvated`.

## Examples

`examples/16_numerical_hessian_from_gradient.py` builds a vibrational analysis
from the gradient by central differences, and
`examples/15_excited_state_geometry_optimization.py` relaxes an excited state.

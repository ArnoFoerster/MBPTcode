# Analytic nuclear gradients

The code computes the forces on the nuclei for the states it computes the
energy of: the dRPA ground state, BSE@GW excited states and GW quasiparticle
energies. The forces are analytic: each is the exact derivative of the energy
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
from src.properties.surfaces import potential_energy_surface


def pbe0(mol):
    mf = dft.RKS(mol, xc='pbe0')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz')
s1 = potential_energy_surface(mol, pbe0, ground_state=GroundState('dft', 'pbe0'),
                              excitation=Excitation('singlet'))
grad, energy, info = s1.total_gradient(mol)
```

`grad` is dE/dR in Hartree/Bohr, one row per atom, `energy` the total energy
of the first singlet and `info['omega']` its excitation energy. The mean field
is passed as a function of the geometry, because every displaced geometry
needs its own; it must be converged tightly, since the gradient assumes the
orbitals are exactly stationary. `ground_state` says which energy the state
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

`examples/16_numerical_hessian_from_gradient.py` builds a vibrational analysis
from the gradient by central differences, and
`examples/15_excited_state_geometry_optimization.py` relaxes an excited state.

"""The excitation force of an ISDF-K chain in point charges is the derivative
of its excitation energy.

`PointCharges.mean_field` wraps the factory's mean field with pyscf's
`qmmm.mm_charge` after `attach_isdf_gradient` has bound `mf.Gradients` to the
unwrapped object, so `mf.Gradients()` of the wrapped mean field answers for
the gas phase. The mean-field force builds its one-electron gradient from the
class (`isdf_mean_field.one_electron_gradient`); the Lagrangian half of
every correlated force, `grad_engine.one_electron_skeleton`, read
`mf.Gradients().hcore_generator` and contracted the relaxed densities with a
core Hamiltonian derivative missing the charges' field. The excitation
energy Omega carries no mean-field term, so its force sees the Lagrangian
alone. Water/cc-pVDZ B3LYP, ISDF-K on the G3 grid, three point charges, the
production realization (SOP residues, frontier set, Davidson, grid adjoint):
dOmega/dR within TOL of a five-point difference of Omega on two components.
"""
import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.environment import PointCharges
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.properties.excitations import SurfaceSpec, surface_of

pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Angstrom, charges in e: a field on every atom, no charge near one
CHARGE_COORDS = [[1.5, 0.3, 1.8], [-1.2, -1.6, 0.9], [0.4, 2.0, -1.9]]
CHARGES = [0.6, -0.5, 0.4]
#: Bohr; the five-point difference's h^4 error is far below TOL here
STEP = 2e-3
TOL = 2e-6
COMPONENTS = ((0, 2), (1, 1))


def isdf_b3lyp(mol):
    """A tightly converged ISDF-K B3LYP mean field on the G3 grid."""
    counts, n_start = resolve_isdf_grid('G3', BASIS, ['H', 'O'], auxbasis=AUX)
    mf = isdf_jk(dft.RKS(mol, xc='b3lyp'), auxbasis=AUX, counts=counts,
                 n_start=n_start)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    mf.kernel()
    return mf


def molecule(coords=None):
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
        mol.build(False, False)
    return mol


def five_point(energy, mol0, ia, k):
    """dE/dR of one Cartesian component, Ha/Bohr."""
    x0 = mol0.atom_coords()
    e = {}
    for s in (-2, -1, 1, 2):
        x = x0.copy()
        x[ia, k] += s * STEP
        e[s] = energy(molecule(x))
    return (e[-2] - 8.0 * e[-1] + 8.0 * e[1] - e[2]) / (12.0 * STEP)


def test_excitation_force_in_point_charges_is_the_energy_derivative():
    mol0 = molecule()
    spec = SurfaceSpec(GroundState('dft', 'b3lyp'),
                       environment=PointCharges(CHARGE_COORDS, CHARGES),
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf',
                       qp_states=QPStates(kind='frontier'),
                       numerics={'grid_accuracy': 'G3', 'bse_adjoint': 'grid',
                                 'nroots': 3})
    chain = surface_of(spec, Excitation('singlet', root=1), mol0, isdf_b3lyp)
    force = np.asarray(chain.excitation_gradient()[0])
    for ia, k in COMPONENTS:
        fd = five_point(chain.excitation, mol0, ia, k)
        print(f'[info] atom {ia} axis {k}: analytic {force[ia, k]:.9f} '
              f'five-point {fd:.9f} Ha/Bohr, residual '
              f'{force[ia, k] - fd:.2e}')
        assert abs(force[ia, k] - fd) < TOL, (ia, k, force[ia, k], fd)

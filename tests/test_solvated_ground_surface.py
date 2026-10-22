"""E_0's surface stands in the declaration's continuum.

`potential_energy_surface` with no state builds `MeanFieldSurface`, which
carries the declaration's environment, so a solvated adiabatic energy has a
ground surface to stand on: its mean field at every geometry is the
factory's relaxed in the static reaction field (PCM at eps_static), the one
an excited chain of the same declaration stands on. Water/cc-pVDZ PBE0,
ISDF-K on the G3 grid, toluene:

  * the ground surface of the spec builds, declares the continuum, and is
    comparable with the excited chain's surface (`compare_surfaces`);
  * its energy at R0 is the excited chain's reference mean field's, bit for
    bit; at a displaced geometry its mean field carries the PCM;
  * its force is the central difference of its energy to 1e-6 Ha/Bohr;
  * in the gas phase it is the bare factory's surface.
"""
from pyscf import dft, gto

from tests.pins import assert_same
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.solvent_screening import SolventScreening
from src.properties.excitations import (SurfaceSpec, ground_surface_of,
                                        surface_of)
from src.properties.surfaces import compare_surfaces

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
STEP = 1e-3
TOL = 1e-6


def isdf_pbe0(mol):
    counts, n_start = resolve_isdf_grid('G3', BASIS, ['H', 'O'], auxbasis=AUX)
    mf = isdf_jk(dft.RKS(mol, xc='pbe0'), auxbasis=AUX, counts=counts,
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


def spec(environment):
    return SurfaceSpec(GroundState('dft', 'pbe0'), environment=environment,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf',
                       qp_states=QPStates(kind='frontier'),
                       numerics={'grid_accuracy': 'G3', 'bse_adjoint': 'grid',
                                 'nroots': 4})


def test_the_solvated_ground_surface_is_the_chain_reference():
    mol0 = molecule()
    declared = spec(SolventScreening(mol0, solvent='toluene'))
    excited = surface_of(declared, Excitation('singlet', root=1), mol0,
                         isdf_pbe0)
    ground = ground_surface_of(declared, excited, mol0, isdf_pbe0)
    assert ground.physics.environment != 'gas'
    compare_surfaces(excited, ground)
    assert_same(ground.total_energy(), excited.mf0.e_tot)
    x = mol0.atom_coords().copy()
    x[1, 1] += 0.05
    moved = molecule(x)
    assert hasattr(ground.mean_field(moved)[1], 'with_solvent')
    force, e0, _ = ground.total_gradient()
    assert_same(e0, excited.mf0.e_tot)
    for ia, k in ((0, 2), (1, 1)):
        es = []
        for s in (+1, -1):
            y = mol0.atom_coords().copy()
            y[ia, k] += s * STEP
            es.append(ground.total_energy(molecule(y)))
        fd = (es[0] - es[1]) / (2 * STEP)
        assert abs(force[ia, k] - fd) < TOL, (ia, k, force[ia, k], fd)


def test_the_gas_phase_ground_surface_is_the_factory_one():
    mol0 = molecule()
    declared = spec(None)
    excited = surface_of(declared, Excitation('singlet', root=1), mol0,
                         isdf_pbe0)
    ground = ground_surface_of(declared, excited, mol0, isdf_pbe0)
    assert ground.physics.environment == 'gas'
    assert ground.mean_field()[1] is excited.mf0
    x = mol0.atom_coords().copy()
    x[1, 1] += 0.05
    moved = molecule(x)
    assert_same(ground.total_energy(moved), isdf_pbe0(moved).e_tot)
    assert not hasattr(ground.mean_field(moved)[1], 'with_solvent')

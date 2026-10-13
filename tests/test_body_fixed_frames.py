"""The frozen excited-state surface is invariant to rigid motion.

The interpolation clouds of the ISDF factors are turned by frames frozen at
the reference geometry R0. The frames turn with the Kabsch rotation of the
geometry onto R0 (`src.Base.body_frame`), and the force carries that
rotation's derivative. Held in the lab frame, a rigid rotation would move the
molecule against its own clouds: the energy would change under the rotation
and the force would carry a net torque.

Gated on C1 water/cc-pVDZ, Hartree-Fock references (no exchange-correlation
grid, whose pyscf quadrature is itself lab-fixed), at a rotation of 25 degrees
about a generic axis plus a translation:
  - the Kabsch rotation follows a rigid motion, is the identity bit for bit
    at its reference, and its adjoint is its derivative;
  - on every route (the contour-deformation chain for S1 and T1; the
    production declaration with SOP residues, Davidson and the row fit
    sliced, on a DF and on an ISDF-K mean field, for S0, S1 and T1; the ISDF
    dRPA ground state; the dense DF route) the energy is invariant, the force
    rotates with the molecule and carries no net force or torque;
  - the force is the derivative of its energy at the rotated geometry,
    along a generic direction and along a rigid rotation;
  - at R0 the energy and the points are the lab-frame convention's bit for
    bit;
  - negative control: with the frames held in the lab frame the energy moves
    under the rotation and the force carries a torque.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base import body_frame
from src.Base.body_frame import BodyFrame, aligned_displacement, kabsch_rotation
from src.Base.constants import ISDF_GRADIENT_FLOOR
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.isdf_jk import isdf_jk
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.gradients.state_manifold import StateManifold
from src.properties.excitations import SurfaceSpec, surface_of
from src.properties.optimize import translation_rotation_basis
from src.properties.surfaces import potential_energy_surface

BASIS = 'cc-pvdz'
#: Water with no symmetry, so no component of a force vanishes by symmetry.
H2O_C1 = 'O 0 0.05 0.117; H 0.1 0.757 -0.468; H 0 -0.757 -0.468'
KEYS = (('singlet', 0), ('triplet', 0))
AXIS, DEGREES = (1.0, 2.0, 3.0), 25.0
SHIFT = np.array([0.3, -0.2, 0.4])
#: Energy invariance, Ha: the rotated geometry repeats R0's arithmetic up to
#: the rotation of every integral, which moves the last bits only.
ENERGY_TOL = 1e-10
#: Net force and torque of an analytic force, Ha/Bohr: sums of the branches'
#: own rounding, far below the fit's floor.
RIGID_TOL = 1e-10
#: Step and five-point stencil of the finite differences, Bohr.
FD_STEP = 1e-4


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def isdf_rhf(mol):
    """An ISDF-K mean field converged past `rhf`: Omega is not variational in
    the orbitals, and at conv_tol_grad 1e-11 this SCF's residual puts 2e-8
    Ha/Bohr into the five-point difference of S1 along a rotation (9e-10 at
    1e-13)."""
    mf = isdf_jk(scf.RHF(mol), auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-13, 400
    mf.kernel()
    assert mf.converged
    return mf


def water():
    return gto.M(atom=H2O_C1, basis=BASIS, verbose=0)


def rotation(axis=AXIS, degrees=DEGREES):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx


def at(mol, coords):
    m = mol.copy()
    m.set_geom_(coords, unit='Bohr')
    m.build(False, False)
    return m


def moved(mol, rot=None, shift=SHIFT):
    """`mol` turned by `rot` about its centroid and translated by `shift`."""
    x = mol.atom_coords()
    c = x.mean(axis=0)
    rot = rotation() if rot is None else rot
    return at(mol, (x - c) @ rot.T + c + shift)


def rigid_part(mol, g):
    """|net force| and |torque| of the force `g`."""
    tr = translation_rotation_basis(mol.atom_coords()) @ np.ravel(g)
    return float(np.linalg.norm(tr[:3])), float(np.linalg.norm(tr[3:]))


def five_point(energy, mol, direction, h=FD_STEP):
    """dE/dlambda along `direction` (normalized) at `mol`."""
    d = direction / np.linalg.norm(direction)
    v = [energy(at(mol, mol.atom_coords() + k * h * d)) for k in (-2, -1, 1, 2)]
    return (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * h), d


def lab_frames(monkeypatch):
    """Frames held in the lab frame, with no rotation term."""
    monkeypatch.setattr(body_frame.BodyFrame, 'rotation',
                        lambda self, coords: None)
    monkeypatch.setattr(body_frame.BodyFrame, 'rotation_adjoint',
                        lambda self, coords, q_bar: np.zeros_like(coords))


# ------------------------------------------------------------- the rotation
@pytest.mark.parametrize('planar', [False, True])
def test_the_kabsch_rotation_follows_a_rigid_motion(planar):
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(5, 3))
    if planar:
        ref[:, 2] = 0.0
    frame = BodyFrame(ref)
    assert frame.rotation(ref) is None, 'the reference reads its own bits'
    rot = rotation()
    x = (ref - ref.mean(0)) @ rot.T + SHIFT
    q = frame.rotation(x)
    assert np.abs(q - rot.T).max() < 1e-13
    assert np.abs((ref - ref.mean(0)) @ q - (x - x.mean(0))).max() < 1e-13
    assert aligned_displacement(ref, x) < 1e-13


@pytest.mark.parametrize('planar', [False, True])
@pytest.mark.parametrize('where', ['reference', 'displaced'])
def test_the_rotation_adjoint_is_its_derivative(planar, where):
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(5, 3))
    if planar:
        ref[:, 2] = 0.0
    x = ref if where == 'reference' else ref + 0.2 * rng.normal(size=ref.shape)
    q_bar = rng.normal(size=(3, 3))
    g = BodyFrame(ref).rotation_adjoint(x, q_bar)
    h, fd = 1e-6, np.zeros_like(x)
    for i in range(len(x)):
        for a in range(3):
            d = np.zeros_like(x)
            d[i, a] = h
            fd[i, a] = (np.sum(q_bar * kabsch_rotation(ref, x + d))
                        - np.sum(q_bar * kabsch_rotation(ref, x - d))) / (2 * h)
    assert np.abs(g - fd).max() < 1e-8


def test_a_linear_reference_keeps_the_lab_frame():
    ref = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 2.9]])
    frame = BodyFrame(ref)
    assert frame.fixed
    assert frame.rotation(ref @ rotation().T) is None
    assert not np.any(frame.rotation_adjoint(ref, np.ones((3, 3))))


# ------------------------------------------------------------ the surfaces
def cd_route(spin):
    """The contour-deformation chain, total energy and force at `mol`."""
    mol = water()
    chain = ExcitedStateChain(mol, rhf, spin=spin, bse_tda=spin == 'triplet')

    def energy(m):
        return chain.energy(m)[0]

    def gradient(m):
        return chain.total_gradient(m)[0]
    return mol, energy, gradient


def production_surface(factory):
    """The production declaration on a Hartree-Fock reference."""
    spec = SurfaceSpec(
        GroundState('dft', 'hf'), environment=None, chi0='space-time',
        residues='sop', solver='davidson', factorization='isdf',
        qp_states=QPStates('admitted'),
        numerics={'grid_accuracy': 'G1', 'bse_adjoint': 'grid', 'nroots': 4,
                  'sliced': True, 'fit': 'rows'})
    mol = water()
    return mol, surface_of(spec, Excitation('singlet'), mol, factory)


def evaluated(surface, mol, gradients=KEYS):
    """One evaluation on `surface`'s frozen conventions, its own root
    following; {name: (energy, force)} for S0, S1 and T1."""
    ev = StateManifold(surface, states=KEYS).evaluate(mol, gradients=gradients)
    out = {'S0': (float(ev.mf.e_tot), ev.g0)}
    for name, k in zip(('S1', 'T1'), KEYS):
        out[name] = (float(ev.energy[k]),
                     None if k not in gradients else ev.gradient[k])
    return out


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_the_cd_chain_is_invariant_and_its_force_exact(spin):
    warnings.simplefilter('ignore')
    mol, energy, gradient = cd_route(spin)
    g0 = gradient(mol)
    e0 = energy(mol)
    force, torque = rigid_part(mol, g0)
    assert force < RIGID_TOL and torque < RIGID_TOL, (force, torque)
    m2 = moved(mol)
    assert abs(energy(m2) - e0) < ENERGY_TOL
    g2 = gradient(m2)
    assert np.abs(g2 - g0 @ rotation().T).max() < ISDF_GRADIENT_FLOOR
    rng = np.random.default_rng(0)
    for direction in (rng.normal(size=g2.shape),
                      translation_rotation_basis(m2.atom_coords())[4]
                      .reshape(-1, 3)):
        fd, d = five_point(energy, m2, direction)
        assert abs(fd - float((g2 * d).sum())) < ISDF_GRADIENT_FLOOR


@pytest.mark.parametrize('factory', [rhf, isdf_rhf], ids=['df', 'isdf-k'])
def test_the_production_surface_is_invariant(factory):
    """S0, S1 and T1 of the production declaration, on a DF and on an ISDF-K
    mean field."""
    warnings.simplefilter('ignore')
    mol, surface = production_surface(factory)
    here = evaluated(surface, mol)
    m2 = moved(mol)
    there = evaluated(surface, m2)
    rot = rotation()
    for name in ('S0', 'S1', 'T1'):
        (e0, g0), (e2, g2) = here[name], there[name]
        force, torque = rigid_part(mol, g0)
        assert force < RIGID_TOL and torque < RIGID_TOL, (name, force, torque)
        assert abs(e2 - e0) < ENERGY_TOL, (name, e2 - e0)
        assert np.abs(g2 - g0 @ rot.T).max() < ISDF_GRADIENT_FLOOR, name
    # the force at the rotated geometry is the derivative of its energy
    g2 = there['S1'][1]
    fd, d = five_point(
        lambda m: evaluated(surface, m, gradients=())['S1'][0], m2,
        translation_rotation_basis(m2.atom_coords())[5].reshape(-1, 3))
    assert abs(fd - float((g2 * d).sum())) < ISDF_GRADIENT_FLOOR


def test_the_drpa_ground_state_is_invariant():
    warnings.simplefilter('ignore')
    mol = water()
    chain = RPAGroundStateChain(mol, rhf)
    e0, g0 = chain.total_energy(mol), chain.total_gradient(mol)[0]
    force, torque = rigid_part(mol, g0)
    assert force < RIGID_TOL and torque < RIGID_TOL, (force, torque)
    m2 = moved(mol)
    assert abs(chain.total_energy(m2) - e0) < ENERGY_TOL
    assert np.abs(chain.total_gradient(m2)[0] - g0 @ rotation().T).max() \
        < ISDF_GRADIENT_FLOOR


def test_the_dense_df_route_is_invariant():
    """No interpolation grid, so nothing was lab-fixed: the scope check. The
    post-SCF fit in the mean field's own auxiliary basis, so the force
    differentiates the energy's one functional."""
    warnings.simplefilter('ignore')
    mol = water()
    surface = potential_energy_surface(
        mol, rhf, ground_state=GroundState('rpa', 'hf'),
        excitation=Excitation('singlet'), chi0='dense-qb', factorization='df',
        auxbasis=BASIS + '-ri')
    e0, g0 = surface.total_energy(mol), surface.total_gradient(mol)[0]
    force, torque = rigid_part(mol, g0)
    assert force < RIGID_TOL and torque < RIGID_TOL, (force, torque)
    m2 = moved(mol)
    assert abs(surface.total_energy(m2) - e0) < ENERGY_TOL


# ------------------------------------------- R0 and the lab-frame convention
def test_r0_is_the_lab_frame_convention_bit_for_bit(monkeypatch):
    warnings.simplefilter('ignore')
    mol, energy, _ = cd_route('singlet')
    chain_points = ExcitedStateChain(mol, rhf).coords(mol)
    e_body = energy(mol)
    lab_frames(monkeypatch)
    mol2, energy2, _ = cd_route('singlet')
    assert energy2(mol2) == e_body
    assert np.array_equal(ExcitedStateChain(mol2, rhf).coords(mol2),
                          chain_points)


def test_lab_frames_fail_the_invariance_gate(monkeypatch):
    """The negative control: lab-fixed frames move under a rotation."""
    warnings.simplefilter('ignore')
    lab_frames(monkeypatch)
    mol, energy, gradient = cd_route('singlet')
    e0 = energy(mol)
    _, torque = rigid_part(mol, gradient(mol))
    assert torque > 1e3 * RIGID_TOL, torque
    assert abs(energy(moved(mol)) - e0) > 1e3 * ENERGY_TOL


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

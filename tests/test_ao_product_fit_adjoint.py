"""The nuclear adjoint of the AO-product ISDF fit on frozen pivoted points
(`src.gradients.ao_product_fit_adjoint`), and the chain force built on it.

Gated on C1 water/cc-pVDZ, Hartree-Fock references, 120-200 points pivoted
from the G1 parent:
  * the fit's adjoint, collocation included, is the derivative of a bounded
    function of the factors, 0.5 |D^T rho|^2 with rho a fixed AO product
    density on the points, against a five-point difference (step 1e-3 Bohr)
    of fits on the selection carried by the body frame: within 1e-8 Ha/Bohr
    (measured 2.1e-10 in the bare gauge, 5.7e-10 dressed), in the bare gauge and in the gauge
    a continuum dresses;
  * three simulated ranks give the one-rank fit adjoint to rounding, the
    same bits on every rank;
  * the excited-state chain's S1 force on these factors is the derivative of
    its energy along a generic direction within ISDF_GRADIENT_FLOOR (measured
    3.6e-10), carries no net force or torque, and a rigid rotation and
    translation leave the energy (1e-10 Ha) and the rotated force unchanged;
  * on aug-cc-pVDZ (400 points from the G1 parent), the production
    declaration's S1 force with the sum-over-poles residues and with the
    explicit contour-deformation residues is the derivative of its energy
    within ISDF_GRADIENT_FLOOR (measured 1.4e-9 and 6.9e-10). Every residue
    route reaches the fit through `FactorChain.whole_fit_branches`; the
    Pade continuation has no force (its derivative is of order 1e11) and
    only selects states.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.factorization import ao_product_fit as apf
from src.Base.factorization.separable_ri import (atomic_frames, aux_metric_sqrt,
                                                 resolve_grid)
from src.Base.geometry.body_frame import BodyFrame
from src.Base.numerics.constants import (ISDF_GRADIENT_FLOOR,
                                         SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.numerics.numerics import Numerics, numerics_of
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.environments.solvent_screening import SolventScreening
from src.gradients.ao_product_fit_adjoint import ao_product_fit_adjoint_gauges
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_manifold import StateManifold
from src.gradients.isdf_derivatives import collocation_adjoint
from src.properties.excitations import SurfaceSpec, surface_of
from tests.test_body_fixed_frames import (ENERGY_TOL, RIGID_TOL, five_point,
                                          moved, rhf, rigid_part, rotation,
                                          water)

H2O_C1 = 'O 0 0.05 0.117; H 0.1 0.757 -0.468; H 0 -0.757 -0.468'
#: Force bar of the fit adjoint against the five-point difference, Ha/Bohr.
FIT_FD_TOL = 1e-8
FD_STEP = 1e-3


class Frozen:
    """A pivoted selection frozen at `mol` and carried by its body frame."""

    def __init__(self, mol, count):
        radii, origins, _, _ = resolve_grid(
            mol, Numerics(isdf={'level': 'G1'}).isdf)
        self.frames = atomic_frames(mol)[0]
        self.pts, self.owner, _ = apf.pivoted_points(mol, radii, origins,
                                                     self.frames, count)
        self.body = BodyFrame(mol.atom_coords())

    def coords(self, m):
        q = self.body.rotation(m.atom_coords())
        fr = self.frames if q is None else self.frames @ q
        return np.vstack([self.pts[i] @ fr[i] + m.atom_coord(i)
                          for i in range(m.natm)])


def at(mol, x):
    m = mol.copy()
    m.set_geom_(x, unit='Bohr')
    m.build(False, False)
    return m


@pytest.mark.parametrize('dressed', [False, True], ids=['bare', 'continuum'])
def test_the_fit_adjoint_is_its_derivative(dressed):
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    frozen = Frozen(mol, 120)
    c = np.random.default_rng(1).normal(size=mol.nao_nr())
    env0 = SolventScreening(mol, solvent='water') if dressed else None

    def value(m):
        """(0.5 |D^T rho|^2, D, X, rho, environment, auxmol) at `m`."""
        env = None if env0 is None else env0.for_geometry(m)
        aux = pyscf_df.addons.make_auxmol(m, auxbasis='cc-pvdz-ri')
        X, M, V = apf.ao_product_fit(m, aux, frozen.coords(m))
        D = M.T @ aux_metric_sqrt(aux, env, V=V)
        rho = (X @ c) ** 2
        w = D.T @ rho
        return 0.5 * w @ w, D, X, rho, env, aux

    _, D, X, rho, env, aux = value(mol)
    w = D.T @ rho
    d_bar = np.outer(rho, w)
    x_bar = 2.0 * ((D @ w) * (X @ c))[:, None] * c[None, :]
    crd = frozen.coords(mol)
    kw = dict(frames=frozen.frames, with_frames=False, body=frozen.body)
    g = (ao_product_fit_adjoint_gauges(mol, aux, crd, [(d_bar, env)],
                                       frozen.pts, frozen.owner, **kw)
         + collocation_adjoint(mol, crd, x_bar, frozen.pts, frozen.owner, **kw))
    x0 = mol.atom_coords()
    fd = np.zeros_like(g)
    for i in range(mol.natm):
        for a in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                x = x0.copy()
                x[i, a] += k * FD_STEP
                v.append(value(at(mol, x))[0])
            fd[i, a] = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * FD_STEP)
    assert np.abs(g - fd).max() < FIT_FD_TOL, np.abs(g - fd).max()
    assert np.abs(g.sum(axis=0)).max() < 1e-10 * np.abs(g).max()


def test_the_fit_adjoint_over_ranks():
    """The shell runs divided over three ranks: the one-rank force to
    rounding, the same bits on every rank."""
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    frozen = Frozen(mol, 120)
    aux = pyscf_df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
    crd = np.vstack([frozen.coords(mol)] * 6)       # several point tiles
    owner = np.concatenate([frozen.owner] * 6)
    pts = [np.vstack([p] * 6) for p in frozen.pts]
    order = np.argsort(owner, kind='stable')         # clouds in atom order
    crd, owner = crd[order], owner[order]
    d_bar = np.random.default_rng(2).normal(size=(len(crd), aux.nao_nr()))

    def force(comm):
        return ao_product_fit_adjoint_gauges(
            mol, aux, crd, [(d_bar, None)], pts, owner,
            frames=frozen.frames, body=frozen.body,
            numerics=Numerics().derive('a block per AO run',
                                       layout={'fit_gb': 1e-3}))
    serial = force(None)
    ranks = run_simulated(force, 3)
    for g in ranks:
        assert np.abs(g - serial).max() < 1e-9 * np.abs(serial).max()
        assert np.array_equal(g, ranks[0])


def test_the_chain_force_is_the_derivative_of_its_energy():
    warnings.simplefilter('ignore')
    mol = water()
    chain = ExcitedStateChain(mol, rhf, spin='singlet', numerics=Numerics(
        isdf={'level': 'G1', 'test_set': 'ao_products', 'pivot_points': 200}))
    assert chain.M == 200

    def energy(m):
        return chain.energy(m)[0]
    e0, g0 = energy(mol), chain.total_gradient(mol)[0]
    force, torque = rigid_part(mol, g0)
    assert force < RIGID_TOL and torque < RIGID_TOL, (force, torque)
    fd, d = five_point(energy, mol,
                       np.random.default_rng(0).normal(size=g0.shape))
    assert abs(fd - float((g0 * d).sum())) < ISDF_GRADIENT_FLOOR
    m2 = moved(mol)
    assert abs(energy(m2) - e0) < ENERGY_TOL
    g2 = chain.total_gradient(m2)[0]
    assert np.abs(g2 - g0 @ rotation().T).max() < ISDF_GRADIENT_FLOOR



def augmented_rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


@pytest.mark.parametrize('residues', ['sop', 'explicit'])
def test_the_production_force_on_an_augmented_basis(residues):
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='aug-cc-pvdz', verbose=0)
    spec = SurfaceSpec(
        GroundState('dft', 'hf'), environment=None, chi0='space-time',
        residues=residues, solver='davidson', factorization='isdf',
        qp_states=QPStates('admitted'),
        numerics=numerics_of({'grid_accuracy': 'G1', 'nroots': 4,
                              'test_set': 'ao_products', 'pivot_points': 400}))
    surface = surface_of(spec, Excitation('singlet'), mol, augmented_rhf)
    key = (('singlet', 0),)
    g = StateManifold(surface, states=key).evaluate(mol, gradients=key).gradient[key[0]]

    def energy(m):
        ev = StateManifold(surface, states=key).evaluate(m, gradients=())
        return float(ev.energy[key[0]])
    fd, d = five_point(energy, mol, np.random.default_rng(0).normal(size=g.shape))
    err = abs(fd - float((g * d).sum()))
    assert err < ISDF_GRADIENT_FLOOR, err


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

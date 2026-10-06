"""Every route's force is the derivative of the energy it reports, on the
ISDF fit every route runs.

The fit is `separable_ri.fit_M_streaming`'s estimator (the Gram matrix over
every product pair, F D^T over the screened pairs), which the ISDF-K SCF, the
space-time GW and the ISDF BSE run, and the gradient chain runs and
differentiates serially (the replicated fit, `fit_M_streaming` itself on the
frozen pair layout) and under ranks (the row fit, `fit_rows`). Water keeps
every pair, so the gates run on ethylene/cc-pVDZ (70 to 74 of 2304 pairs
screened, by the route's grid) and formaldehyde/cc-pVDZ (6 of 1444), both
with a seeded distortion so no symmetry zeroes a component, on a
density-fitted Hartree-Fock reference (no xc quadrature floor):

  (a) the dRPA ground-state force and the BSE@GW singlet force of the
      space-time route (contour-deformation residues) and of the SOP route
      (frontier states with the outside scissor, the grid BSE adjoint), each
      against a 4-point central difference of the chain's own energy:
      serially on the default fit, and on the row fit over 2, 3 and 8
      simulated ranks, every rank's force rank 0's and within
      `ISDF_GRADIENT_FLOOR` of the serial one;
  (b) the ISDF-K mean-field force (`isdf_mean_field_gradient`) against a
      Richardson central difference of the ISDF-K SCF energy: the row
      realization serially, and over 2, 3 and 8 simulated ranks on the
      distributed ISDF-K SCF, whose own tiles the skeleton reuses.

The chain meets its differences to a few 1e-9 Ha/Bohr, within
`ISDF_GRADIENT_FLOOR`. The SOP route on ethylene carries pole-basis rounding
noise (coincident fitted poles, ~1e-10 Ha of energy noise that a difference
divides by h; tests/test_sop_force_follows_its_energy.py), so it is gated at
h = 4e-3 and 1e-7. The ISDF-K SCF's points follow `atomic_frames` at every
geometry, whose curvature on the distorted formaldehyde needs Richardson
steps of 2.5e-4 and 5e-4.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (ISDF_GRADIENT_FLOOR,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.distributed_df import distributed_mean_field
from src.Base.distributed_isdf_jk import distributed_isdf_jk
from src.Base.isdf_jk import isdf_jk
from src.Base.utils.mpi_grid import distributed, run_simulated
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.isdf_derivatives import product_pairs
from src.gradients.isdf_mean_field import isdf_mean_field_gradient
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.excitations import SurfaceSpec, surface_of

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
MOLECULES = {
    'ethylene': ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
                 'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; '
                 'H 0.0 -0.923 -1.238'),
    'formaldehyde': ('C 0 0 -0.5395; O 0 0 0.6636; '
                     'H 0 0.9445 -1.1090; H 0 -0.9445 -1.1090'),
}
#: The seeded distortion, Bohr, of tests/test_sop_force_follows_its_energy.py.
DISTORTION = 0.05
DISTORTION_SEED = 7
#: The Cartesian components each difference takes: a hydrogen and two heavy
#: atoms, in and out of the undistorted plane.
COMPONENTS = {'ethylene': ((2, 1), (0, 2), (3, 2)),
              'formaldehyde': ((0, 2), (1, 2), (2, 1))}
#: (step, bar): the chain's routes at h = 1e-3 on the floor; SOP ethylene at
#: the pole-basis noise of its energy.
STEP, BAR = 1e-3, ISDF_GRADIENT_FLOOR
SOP_NOISY = {'ethylene': (4e-3, 1e-7)}
SIZES = (2, 3, 8)
#: The row fit's tile edge: ethylene's grid in 14 tiles, formaldehyde's in 10.
TILE = 64
#: The Davidson residual, below the fit's response (tests/test_chain_row_fit.py).
BSE_CONV_TOL = 1e-9
#: The Richardson steps of the ISDF-K SCF energy's difference, Bohr.
SCF_STEPS = (2.5e-4, 5e-4)
#: (atom, axis) of the mean-field gates: a carbon along the bond, a hydrogen.
MF_COMPONENTS = ((0, 2), (2, 1))
#: The SCF of the mean-field gates: its energy converged far below what a
#: difference at `SCF_STEPS` divides by h.
MF_CONV_TOL, MF_CONV_TOL_GRAD = 1e-13, 1e-10
#: The ISDF-K SCF geometries of tests/test_skeleton_tiles.py (C1-distorted).
MF_GEOMS = {
    'ethylene': ('C 0.02 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
                 'H 0 -0.9589 1.2321; H 0.03 0.9289 -1.2321; '
                 'H 0 -0.9289 -1.2021'),
    'formaldehyde': ('C 0 0 -0.5296; O 0 0.02 0.6763; H 0 0.9357 -1.1172; '
                     'H 0.02 -0.9557 -1.1172'),
}


def molecule(name):
    """`name`/cc-pVDZ with the seeded distortion."""
    mol = gto.M(atom=MOLECULES[name], basis=BASIS, verbose=0)
    rng = np.random.default_rng(DISTORTION_SEED)
    shift = rng.uniform(-DISTORTION, DISTORTION, (mol.natm, 3))
    return mol.set_geom_(mol.atom_coords() + shift, unit='Bohr', inplace=False)


def at(mol, atom, axis, step):
    crd = mol.atom_coords().copy()
    crd[atom, axis] += step
    return mol.set_geom_(crd, unit='Bohr', inplace=False)


def chain_scf(mol):
    """A density-fitted Hartree-Fock mean field converged for gradient work."""
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def sop_scf(mol):
    """The SOP gate's reference, tests/test_sop_force_follows_its_energy.py's."""
    mf = dft.RKS(mol, xc='hf').density_fit()
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


class Route:
    """(energy at a geometry, force at the reference) of one chain."""

    def __init__(self, chain, energy, force):
        self.chain, self.energy, self.force = chain, energy, force


def drpa(mol, **kw):
    chain = RPAGroundStateChain(mol, chain_scf, mf=chain_scf(mol), **kw)
    return Route(chain, lambda m: chain.energy(m)[0],
                 lambda: chain.total_gradient()[0])


def bse(mol, **kw):
    chain = ExcitedStateChain(mol, chain_scf, spin='singlet',
                              solver='davidson', bse_conv_tol=BSE_CONV_TOL,
                              mf=chain_scf(mol), **kw)
    return Route(chain, chain.excitation,
                 lambda: chain.excitation_gradient()[0])


def sop(mol, **kw):
    spec = SurfaceSpec(GroundState('dft', 'hf'), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf', qp_states=QPStates(kind='frontier'),
                       numerics=dict(kw, bse_adjoint='grid'))
    chain = surface_of(spec, Excitation('singlet', root=1, kernel='bse'), mol,
                       sop_scf)
    return Route(chain, chain.excitation,
                 lambda: chain.excitation_gradient()[0])


ROUTES = {'drpa': drpa, 'bse': bse, 'sop': sop}
ROW_FIT = dict(sliced=True, fit='rows', fit_block=TILE)


def difference(route, mol, components, h):
    """The 4-point central difference of the route's own energy."""
    out = {}
    for atom, axis in components:
        e = {k: route.energy(at(mol, atom, axis, k * h)) for k in (2, 1, -1, -2)}
        out[atom, axis] = (8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * h)
    return out


def misses(force, fd):
    return np.array([force[c] - v for c, v in fd.items()])


# ------------------------------------------------------ (a) the chain routes
@pytest.mark.parametrize('name', list(MOLECULES))
@pytest.mark.parametrize('route', list(ROUTES))
def test_the_chain_force_follows_its_energy(route, name):
    """Serially on the default (replicated) fit and on the row fit over 2,
    3 and 8 simulated ranks, against one difference of the serial energy."""
    h, bar = SOP_NOISY.get(name, (STEP, BAR)) if route == 'sop' else (STEP,
                                                                      BAR)
    mol = molecule(name)
    with distributed(None):
        serial = ROUTES[route](mol)
        assert serial.chain.fit == 'replicated' and not serial.chain.sliced
        screened = len(product_pairs(mol)[0]) - len(serial.chain.layout[0])
        assert screened > 0, 'the screen drops no pair: no gate of the fit'
        force = serial.force()
        fd = difference(serial, mol, COMPONENTS[name], h)
    lines = [f'{screened} pairs screened, serial '
             f'{np.array2string(misses(force, fd), precision=2)}']
    assert np.abs(misses(force, fd)).max() < bar, lines

    def rank(comm):
        return ROUTES[route](molecule(name), **ROW_FIT).force()

    for size in SIZES:
        out = run_simulated(rank, size)
        assert all(np.array_equal(f, out[0]) for f in out), size
        miss = misses(out[0], fd)
        apart = float(np.abs(out[0] - force).max())
        lines.append(f'rows at {size}: {np.array2string(miss, precision=2)}, '
                     f'serial {apart:.1e}')
        assert np.abs(miss).max() < bar, lines
        assert apart < ISDF_GRADIENT_FLOOR, lines
    print(f'\n{route} {name}, analytic - FD (h {h:g}): ' + '; '.join(lines))


# --------------------------------------------- (b) the ISDF-K mean field
def isdf_scf(name, xc, coords=None):
    """An unconverged ISDF-K mean field of `name` at `coords` (Bohr)."""
    mol = gto.M(atom=MF_GEOMS[name], basis=BASIS, verbose=0, max_memory=8000)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
    base = dft.RKS(mol, xc=xc) if xc != 'hf' else scf.RHF(mol)
    mf = isdf_jk(base, auxbasis=AUX)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = (MF_CONV_TOL,
                                                   MF_CONV_TOL_GRAD, 200)
    return mf


def scf_difference(name, xc):
    """Richardson central difference of the ISDF-K SCF energy."""
    R0 = isdf_scf(name, xc).mol.atom_coords()
    out = {}
    for atom, axis in MF_COMPONENTS:
        d = {}
        for h in SCF_STEPS:
            e = []
            for s in (1, -1):
                R = R0.copy()
                R[atom, axis] += s * h
                e.append(isdf_scf(name, xc, R).kernel())
            d[h] = (e[0] - e[1]) / (2 * h)
        out[atom, axis] = (4 * d[SCF_STEPS[0]] - d[SCF_STEPS[1]]) / 3
    return out


@pytest.mark.parametrize('xc', ['hf', 'pbe0'])
@pytest.mark.parametrize('name', list(MF_GEOMS))
def test_the_isdf_k_mean_field_force_follows_its_energy(name, xc):
    """The row realization of the ISDF-K mean-field force serially, and the
    distributed ISDF-K SCF's own at 2, 3 and 8 simulated ranks."""
    fd = scf_difference(name, xc)
    with distributed(None):
        mf = isdf_scf(name, xc)
        mf.kernel()
        force = isdf_mean_field_gradient(mf, fit='rows')
    lines = [f'serial rows {np.array2string(misses(force, fd), precision=2)}']
    assert np.abs(misses(force, fd)).max() < BAR, lines

    def rank(comm):
        mf = isdf_scf(name, xc)
        distributed_isdf_jk(mf, comm, tile=TILE)
        distributed_mean_field(mf)
        return isdf_mean_field_gradient(mf)

    for size in SIZES:
        out = run_simulated(rank, size)
        assert all(np.array_equal(f, out[0]) for f in out), size
        miss = misses(out[0], fd)
        apart = float(np.abs(out[0] - force).max())
        lines.append(f'{size} ranks {np.array2string(miss, precision=2)}, '
                     f'serial {apart:.1e}')
        assert np.abs(miss).max() < BAR, lines
        assert apart < ISDF_GRADIENT_FLOOR, lines
    print(f'\n{name}/{xc} ISDF-K mean field, analytic - FD: '
          + '; '.join(lines))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

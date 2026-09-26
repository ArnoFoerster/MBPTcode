"""An explicit polarizable shell inside a continuum: one coupled reaction field.

The cavity encloses the solute AND the explicit shell (`SolventScreening(...,
cavity_atoms=shell)`), for the ground state's PCM and for the optical
response alike, and polarizable sites at the shell (`ContinuumWithSites`)
respond together with the continuum: q = Q (U + G mu), mu = B (E - G^T q),
folded into vtilde = U Q U^T - E' M E'^T.

  1. The enclosing cavity: the surface grows by the shell's spheres, the
     surface's nuclear potential stays the solute's alone, a shell far away
     leaves the solvated SCF unchanged, and the ground state and the response
     share the shell (a PCM without it is refused).
  2. The coupled kernel: the dipole potential on the surface matches a finite
     difference of point charges, the closed form matches the full block
     solve of the coupled linear problem, the kernel is symmetric and
     negative semidefinite, differs from the sum of the members, and reduces
     exactly to each member alone (B = 0, eps = 1).
  3. Through the GW routes: the Casida and space-time Eq. (18) routes agree,
     the shell water adds polarization on top of the continuum, and B = 0
     reproduces the shell continuum alone bitwise.
  4. Refusals: a continuum that does not enclose the shell, a site outside
     the cavity, a frequency-dependent member in the dRPA factor, the routes
     without W, and a force.

Run as a script (`python tests/test_continuum_with_sites.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df, gto, scf

from src.Base.composite_environment import (ContinuumWithSites,
                                            dipole_surface_potential)
from src.Base.environment import attach_environment
from src.Base.polarizable_sites import PolarizableSites, site_field
from src.Base.solvent_screening import EnclosingPCM, SolventScreening
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.gradients.excited_state import ExcitedStateChain

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
#: a second water 4 A out, beyond the sites' 3.4 A clearance from the solute
SHELL = [('O', (0, 0, 4.0)), ('H', (0, 0.757, 4.586)), ('H', (0, -0.757, 4.586))]
#: one polarizable site per shell molecule, at its oxygen: water's alpha
SITE_ALPHA = 9.8
EPS_INF, EPS_S = 1.776, 78.355


def _continuum(mol, shell=SHELL, **kw):
    return SolventScreening(mol, eps=EPS_INF, eps_static=EPS_S,
                            cavity_atoms=shell, **kw)


def _factory(mol):
    return scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run(conv_tol=1e-11)


@pytest.fixture(scope='module')
def small():
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    continuum = _continuum(mol)
    sites = PolarizableSites([SHELL[0][1]], [SITE_ALPHA], mol=mol)
    return mol, continuum, sites


def test_the_cavity_encloses_the_shell(small):
    mol, continuum, _ = small
    plain = SolventScreening(mol, eps=EPS_INF, eps_static=EPS_S)
    assert continuum.ngrids > plain.ngrids
    assert not continuum.differentiable and plain.differentiable
    # the solute's nuclei alone on the surface: the shell is cavity, not charge
    pcm = continuum._pcm
    grid = gto.fakemol_for_charges(pcm.surface['grid_coords'],
                                   expnt=pcm.surface['charge_exp'] ** 2)
    nuclei = gto.fakemol_for_charges(mol.atom_coords())
    v_n = mol.atom_charges() @ gto.mole.intor_cross('int2c2e', nuclei, grid)
    assert np.abs(pcm.v_grids_n - v_n).max() < 1e-12
    # a shell 60 A away is a bubble the solute does not see
    far = _continuum(mol, shell=[('O', (0, 0, 60.0))])
    e_plain = plain.mean_field(mol, _factory).e_tot
    assert abs(far.mean_field(mol, _factory).e_tot - e_plain) < 1e-9
    near = continuum.mean_field(mol, _factory)
    assert isinstance(near.with_solvent, EnclosingPCM)
    assert abs(near.e_tot - e_plain) > 1e-6
    # the ground state and the response share the shell
    with pytest.raises(ValueError, match='enclosed shell differs'):
        continuum.check_ground_state_continuum(
            plain.mean_field(mol, _factory).with_solvent)
    moved = mol.copy()
    moved.set_geom_(mol.atom_coords() + 0.01, unit='Bohr')
    for other in (continuum.for_geometry(moved), continuum.static_partner()):
        assert [s for s, _ in other.cavity_atoms] == ['O', 'H', 'H']
        assert np.allclose([x for _, x in other.cavity_atoms],
                           [x for _, x in continuum.cavity_atoms])


def test_the_coupled_kernel(small):
    mol, continuum, sites = small
    composite = ContinuumWithSites(continuum, sites)
    G, M = composite.coupling()
    # G against a finite difference of point charges on the smeared surface
    surface, h = continuum._fakemol(), 1e-4
    G_fd = np.zeros_like(G)
    for j, site in enumerate(sites.coords):
        for x in range(3):
            d = np.zeros(3)
            d[x] = h / 2
            plus, minus = (gto.fakemol_for_charges(np.array([site + s * d]))
                           for s in (1, -1))
            G_fd[:, 3 * j + x] = (gto.mole.intor_cross('int2c2e', plus, surface)
                                  - gto.mole.intor_cross('int2c2e', minus,
                                                         surface))[0] / h
    assert np.abs(G - G_fd).max() / np.abs(G).max() < 1e-8
    assert np.allclose(G, dipole_surface_potential(continuum, sites.coords))
    auxmol = df.addons.make_auxmol(mol, 'cc-pvdz-jkfit')
    kernel = composite.aux_kernel(auxmol)
    # the closed form is the full coupled linear problem eliminated
    U = continuum.aux_grid_potential(auxmol)
    E = site_field(auxmol, sites.coords).reshape(auxmol.nao_nr(), -1)
    A = np.block([[-np.linalg.inv(continuum.response_matrix()), G],
                  [G.T, np.linalg.inv(sites.B)]])
    S = np.hstack([-U, E])
    block = -S @ np.linalg.solve(A, S.T)
    assert np.abs(kernel - block).max() / np.abs(kernel).max() < 1e-12
    assert np.abs(kernel - kernel.T).max() < 1e-12
    assert np.linalg.eigvalsh(0.5 * (kernel + kernel.T)).max() < 1e-10
    summed = continuum.aux_kernel(auxmol) + sites.aux_kernel(auxmol)
    assert np.abs(kernel - summed).max() > 1e-3 * np.abs(kernel).max()
    # each member alone, exactly
    inert = PolarizableSites.from_response_matrix([SHELL[0][1]],
                                                  np.zeros((3, 3)),
                                                  unit='Angstrom')
    assert np.array_equal(ContinuumWithSites(continuum, inert).aux_kernel(auxmol),
                          continuum.aux_kernel(auxmol))
    vacuum = SolventScreening(mol, eps=1.0, allow_static_eps=True,
                              cavity_atoms=SHELL)
    assert np.array_equal(ContinuumWithSites(vacuum, sites).aux_kernel(auxmol),
                          sites.aux_kernel(auxmol))


def test_through_the_gw_routes():
    """Water/cc-pVDZ: the shell continuum moves the HOMO by +1.568 eV; the
    polarizable shell water adds +18 meV to it (and -9 meV on the LUMO)."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    continuum = _continuum(mol)
    sites = PolarizableSites([SHELL[0][1]], [SITE_ALPHA], mol=mol)
    inert = PolarizableSites.from_response_matrix([SHELL[0][1]],
                                                  np.zeros((3, 3)),
                                                  unit='Angstrom')
    composite = ContinuumWithSites(continuum, sites)
    factory = lambda m: scf.RHF(m).density_fit(auxbasis='cc-pvdz-ri').run(
        conv_tol=1e-11)
    mf = composite.mean_field(mol, factory)
    nocc = mol.nelectron // 2

    def levels(environment, mode='casida'):
        attach_environment(mf, environment)
        try:
            out = calc_qp_energy(mf, state=[nocc - 1, nocc], mode=mode)
        finally:
            attach_environment(mf, None)
        return np.array([out[p]['GW'] for p in (nocc - 1, nocc)]
                        if isinstance(out, dict) else out)

    alone = levels(continuum)
    assert np.array_equal(levels(ContinuumWithSites(continuum, inert)), alone)
    coupled = levels(composite)
    assert coupled[0] - alone[0] > 5e-3 and coupled[1] - alone[1] < -2e-3, \
        (coupled, alone)
    assert np.abs(levels(composite, 'space-time') - coupled).max() < 3e-3


def test_refusals(small):
    mol, continuum, sites = small
    plain = SolventScreening(mol, eps=EPS_INF, eps_static=EPS_S)
    with pytest.raises(ValueError, match='must enclose the explicit shell'):
        ContinuumWithSites(plain, sites)
    stray = PolarizableSites([(0.0, 6.0, 4.0)], [SITE_ALPHA], mol=mol)
    with pytest.raises(ValueError, match='outside every sphere'):
        ContinuumWithSites(continuum, stray)
    composite = ContinuumWithSites(continuum, sites)
    assert np.all(composite.dynamic_factor(np.array([0.1, 1.0])) == 1.0)
    named = ContinuumWithSites(
        SolventScreening(mol, solvent='water', cavity_atoms=SHELL), sites)
    with pytest.raises(NotImplementedError, match='frequency-dependent'):
        named.dynamic_factor(np.array([0.5]))
    for call in (lambda: composite.whitened_transform(mol, None),
                 lambda: composite.static_self_energy(None, mol),
                 lambda: composite.kernel_ao(mol)):
        with pytest.raises(NotImplementedError):
            call()
    # a chain carries the composite's energies and refuses its force (the ISDF
    # grid is tabulated at cc-pVDZ)
    dz = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    chain = ExcitedStateChain(
        dz, lambda m: scf.RHF(m).density_fit(auxbasis='cc-pvdz-ri').run(),
        environment=ContinuumWithSites(_continuum(dz), sites), solver='dense')
    assert np.isfinite(chain.quasiparticle(0))
    with pytest.raises(NotImplementedError, match='no nuclear derivative'):
        chain.quasiparticle_gradient(0)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

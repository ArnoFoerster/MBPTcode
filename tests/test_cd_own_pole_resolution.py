"""A quasiparticle root closer to its own orbital energy than the contour
grid's smallest node: formaldehyde/cc-pVTZ (HF), orbital 6.

The explicit-residue route on the frontier window, and the dense quasi-boson
route on the same molecule as the arbiter.

Orbital 6 (HOMO-1) has a G0W0@HF correction of +9.4 meV with Z = 0.93. The
64-point contour grid's smallest node is 9.7e-5 Ha, below the half-width
3.5e-4 Ha of the Lorentzian its own pole of G puts on the imaginary-frequency
integrand at nu = 0. That Lorentzian's singular part is integrated in closed
form (`cd_integral_weights`), so the self-energy is continuous through eps_6
on the plain grid, and the root and Z are the resolved ones there:

- the guarded Newton converges instead of cycling across eps_6;
- the root is +9.4 meV with Z = 0.931, not a zero read off an unresolved
  slope (+9.2 meV, Z = 0.20 by quadrature alone);
- nothing is demoted.

Against the dense quasi-boson route the four frontier roots differ by 1.06,
0.80, 0.54 and 0.54 meV (orbitals 6 to 9): orbital 6 is as good as the rest.
"""
import os
import sys
import warnings

import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (CD_NFREQ, ISDF_GRID_ACCURACY_TARGET_EV,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.SingleReference.GW.qp_states import is_quasiparticle_root
from src.properties.excitations import (SurfaceSpec, driven_chain,
                                        qp_bookkeeping, surface_of)

FORMALDEHYDE = ('C 0.0 0.0 -0.5296; O 0.0 0.0 0.6742; '
                'H 0.0 0.9429 -1.1123; H 0.0 -0.9429 -1.1123')
GROUND = GroundState('rpa', 'hf')
EXPLICIT = SurfaceSpec(GROUND, chi0='space-time', residues='explicit',
                       factorization='isdf', qp_states=QPStates('frontier'),
                       numerics={'grid_accuracy': 'G3'})
QB = SurfaceSpec(GROUND, chi0='dense-qb', factorization='four-index',
                 qp_states=QPStates('frontier'))
#: The orbital whose root sits inside its own pole's band.
ORBITAL = 6
#: The explicit route against the dense quasi-boson one is ISDF at G3 and the
#: space-time chi0 against exact four-index integrals: held to the level's
#: target. Measured 1.06 meV at orbital 6 and at most 0.80 at the others.
QB_AGREEMENT_EV = ISDF_GRID_ACCURACY_TARGET_EV['G3']
#: The resolved root of orbital 6, +9.4 meV above eps_6 with Z = 0.931.
RESOLVED_SHIFT_MEV, RESOLVED_Z = 9.4, 0.931


def factory(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 100
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom=FORMALDEHYDE, basis='cc-pvtz', verbose=0)


def record(spec, mol):
    """(qp_bookkeeping, chain) of the singlet S1 surface of `spec`, after its
    first forward at `mol`."""
    warnings.simplefilter('ignore')
    surface = surface_of(spec, Excitation('singlet', root=1), mol, factory)
    surface.total_energy(mol)
    return qp_bookkeeping(surface, mol), driven_chain(surface)


@pytest.fixture(scope='module')
def explicit(mol):
    return record(EXPLICIT, mol)


@pytest.fixture(scope='module')
def qb(mol):
    return record(QB, mol)[0]


def test_the_root_is_the_resolved_one_on_the_plain_grid(explicit):
    qp, chain = explicit
    assert ORBITAL in qp['explicit'] and not qp['demoted'], qp
    assert is_quasiparticle_root(qp['z'][ORBITAL]), qp['z']
    assert chain.nfreq_cd == CD_NFREQ, chain.nfreq_cd
    shift = 1e3 * (qp['qp_root_eV'][ORBITAL]
                   - qp['eps_mean_field_eV'][ORBITAL])
    assert abs(shift - RESOLVED_SHIFT_MEV) < 0.1, shift
    assert abs(qp['z'][ORBITAL] - RESOLVED_Z) < 2e-3, qp['z'][ORBITAL]


def test_the_root_agrees_with_the_dense_route(explicit, qb):
    exp_roots = explicit[0]['qp_root_eV']
    qb_roots = qb['qp_root_eV']
    assert set(exp_roots) == set(qb_roots)
    gaps = {p: abs(exp_roots[p] - qb_roots[p]) for p in exp_roots}
    assert max(gaps.values()) < QB_AGREEMENT_EV, gaps


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

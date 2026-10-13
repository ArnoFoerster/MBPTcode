"""The sum-over-poles excitation force is the derivative of the surface it walks.

The pole model's positions Om_m are placed by vector fitting on the reference
geometry's imaginary-axis data and then held with the rest of the frozen
realization (the Newton seeds, the guard offsets, the scissor), so that at
every displaced geometry only the amplitudes A = F^+ wc move, which is what
`sigma_sop_backward` differentiates. Re-fitting the poles at every geometry
puts their motion into the energy and not into the force: on
water/cc-pVDZ/PBE0 that misses a finite difference by 2.6e-5 Ha/Bohr, on
distorted formaldehyde by 1.5e-4.

On the production flag set (space-time ISDF chain, row fit, sliced factors,
grid adjoint, the frontier quasiparticle set with the outside scissor, S1):

  (a) the pole set is fitted once at the reference, reused at every displaced
      geometry, and fitted afresh by `refreeze`;
  (b) Hartree-Fock water, distorted formaldehyde and distorted ethylene: the
      force against a 4-point difference of the chain's own energy;
  (c) PBE0 and LRC-wPBEh water at the default xc grid, the range-separated
      hybrid on a density-fitted and on an ISDF-K reference;
  (d) the pole set re-fitted before every displaced energy fails (c).

On Hartree-Fock water and formaldehyde the force meets the difference to
1.2e-9 and 4.8e-9 Ha/Bohr at h = 1e-3. Ethylene's fitted pole sets carry
coincident and near-coincident poles, so F is rank-deficient or conditioned
at 1e9 to 1e20 and its energy carries ~1e-10 Ha of rounding noise that a
difference divides by h: 1.1e-7 at h = 1e-3, 2.8e-8 to 1.1e-7 at h = 4e-3
(the last bits of the frequency pass move it). On a
Kohn-Sham reference the xc skeleton carries the motion of the Becke grid
(tests/test_xc_grid_response.py), and the force meets the difference to
2.1e-9 on PBE0 and 2.5e-9 on LRC-wPBEh. The translation residual on a
density-fitted reference, ~1e-11, is the tiled fitted exchange skeleton's
explicit inverse of the metric; the ISDF-K force has none (4e-15).
"""
import pathlib
import sys

import numpy as np
import pytest
from pyscf import dft, gto

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.Base.constants import (SCF_DIFFERENTIABLE_CONV_TOL,         # noqa: E402
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates   # noqa: E402
from src.Base.isdf_jk import isdf_jk                                 # noqa: E402
from src.properties.excitations import SurfaceSpec, surface_of       # noqa: E402
import src.gradients.qp_space_time as qp_space_time                  # noqa: E402

#: The test molecules' geometries, Angstrom.
SYSTEMS = {
    'water': 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692',
    'formaldehyde': ('C 0.0 0.0 -0.5296; O 0.0 0.0 0.6742; '
                     'H 0.0 0.9429 -1.1123; H 0.0 -0.9429 -1.1123'),
    'ethylene': ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
                 'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; '
                 'H 0.0 -0.923 -1.238'),
}

#: The seeded distortion of formaldehyde and ethylene, Bohr: no symmetry is
#: left to zero a force component or the force sum.
DISTORTION = 0.05
DISTORTION_SEED = 7
#: (molecule, [(atom, axis)], h in Bohr, bar in Ha/Bohr) on Hartree-Fock.
HF_GATES = (
    ('water', ((0, 2), (1, 1), (2, 2)), 1e-3, 1e-8),
    ('formaldehyde', ((0, 2), (1, 0), (3, 1)), 1e-3, 2e-8),
    ('ethylene', ((0, 2), (2, 1), (3, 0)), 4e-3, 2e-7),
)
#: Kohn-Sham water at the default xc grid, the chain's floor: 2.1e-9
#: (PBE0), 2.5e-9 (LRC-wPBEh, density-fitted), 2.7e-9 (ISDF-K).
KS_COMPONENTS = ((0, 2), (1, 1), (2, 2))
KS_STEP = 1e-3
KS_BAR = 1e-8
#: The range-separated hybrid's references.
RSH_REFERENCES = ('df', 'isdf-k')
#: The force of a translation-invariant surface sums to zero: to the tiled
#: fitted skeletons' 1e-11 (water) to 1.2e-10 (ethylene) on a density-fitted
#: reference, Hartree-Fock or Kohn-Sham alike.
HF_TRANSLATION_BAR = 1e-9
KS_TRANSLATION_BAR = 1e-9


def molecule(name):
    mol = gto.M(atom=SYSTEMS[name], basis='cc-pvdz', verbose=0)
    if name == 'water':
        return mol
    rng = np.random.default_rng(DISTORTION_SEED)
    shift = rng.uniform(-DISTORTION, DISTORTION, (mol.natm, 3))
    return mol.set_geom_(mol.atom_coords() + shift, unit='Bohr', inplace=False)


def factory(xc, reference='df'):
    def scf_factory(mol):
        mf = (isdf_jk(dft.RKS(mol, xc=xc), auxbasis='cc-pvdz-ri')
              if reference == 'isdf-k' else dft.RKS(mol, xc=xc).density_fit())
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        mf.kernel()
        return mf
    return scf_factory


def sop_chain(mol, xc, reference='df'):
    """The SOP-scissor S1 surface on the production flag set, on `mol`."""
    spec = SurfaceSpec(GroundState('dft', xc), environment=None,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf', qp_states=QPStates(kind='frontier'),
                       numerics={'sliced': True, 'fit': 'rows',
                                 'bse_adjoint': 'grid'})
    return surface_of(spec, Excitation('singlet', root=1, kernel='bse'), mol,
                      factory(xc, reference))


def at(mol, atom, axis, step):
    crd = mol.atom_coords().copy()
    crd[atom, axis] += step
    return mol.set_geom_(crd, unit='Bohr', inplace=False)


def fd_residuals(chain, mol, grad, components, h, refit=False):
    """analytic - 4-point difference of the chain's own excitation energy.

    refit: clear the frozen pole set before every displaced energy, which is
    the surface a per-geometry fit walks.
    """
    out = []
    for atom, axis in components:
        om = {}
        for k in (2, 1, -1, -2):
            if refit:
                chain.sop_poles.clear()
            om[k] = chain.excitation(at(mol, atom, axis, k * h))
        fd = (8.0 * (om[1] - om[-1]) - (om[2] - om[-2])) / (12.0 * h)
        out.append(float(grad[atom, axis] - fd))
    return np.array(out)


@pytest.fixture(scope='module')
def water_pbe0():
    mol = molecule('water')
    chain = sop_chain(mol, 'pbe0')
    grad, diags = chain.excitation_gradient()
    return mol, chain, grad, diags


# ------------------------------------------- (a) the pole set is frozen at R0
def test_the_pole_set_is_fitted_once_and_held(water_pbe0, monkeypatch):
    """Every pole-model state has a frozen set after the reference solve, and
    no displaced energy fits another or changes the one on file."""
    mol, chain, _, _ = water_pbe0
    assert sorted(chain.sop_poles) == sorted(int(p) for p in chain.qp_set), \
        'every frontier state is on the pole model at water'
    frozen = {p: v.copy() for p, v in chain.sop_poles.items()}
    fits = []
    real_fit = qp_space_time.sop_from_wc
    monkeypatch.setattr(qp_space_time, 'sop_from_wc',
                        lambda *a, **k: fits.append(1) or real_fit(*a, **k))
    chain.excitation(at(mol, 0, 2, 1e-2))
    assert not fits, 'a displaced geometry re-fitted its poles'
    for p, poles in chain.sop_poles.items():
        assert np.array_equal(poles, frozen[p]), p


def test_refreeze_fits_the_poles_at_the_new_reference(water_pbe0):
    """The pole set is part of the realization `refreeze` rebuilds."""
    mol, chain, _, _ = water_pbe0
    moved = at(mol, 0, 2, 5e-2)
    fresh = chain.refreeze(moved)
    assert fresh.sop_poles == {}, 'the reference pole set was carried over'
    fresh.excitation(moved)
    assert sorted(fresh.sop_poles) == sorted(chain.sop_poles)
    assert any(not np.array_equal(fresh.sop_poles[p], chain.sop_poles[p])
               for p in chain.sop_poles), 'the new reference fitted nothing'


# ---------------------------------- (b) Hartree-Fock: the chain's own floor
@pytest.mark.parametrize('name,components,h,bar', HF_GATES,
                         ids=[g[0] for g in HF_GATES])
def test_the_hartree_fock_sop_force_follows_its_energy(name, components, h, bar):
    mol = molecule(name)
    chain = sop_chain(mol, 'hf')
    grad, _ = chain.excitation_gradient()
    assert chain.sop_poles, 'no state took the pole model'
    translation = np.abs(grad.sum(axis=0)).max()
    assert translation < HF_TRANSLATION_BAR, translation
    diff = fd_residuals(chain, mol, grad, components, h)
    print(f'\n{name}/HF: analytic - FD {np.array2string(diff, precision=2)} '
          f'Ha/Bohr, translation {translation:.1e}')
    assert np.abs(diff).max() < bar, diff


# ------------------------------- (c) PBE0 at the default xc grid, (d) refit
def test_the_pbe0_sop_force_follows_its_energy(water_pbe0):
    mol, chain, grad, _ = water_pbe0
    translation = np.abs(grad.sum(axis=0)).max()
    assert translation < KS_TRANSLATION_BAR, translation
    diff = fd_residuals(chain, mol, grad, KS_COMPONENTS, KS_STEP)
    print(f'\nwater/PBE0: analytic - FD {np.array2string(diff, precision=2)} '
          f'Ha/Bohr, translation {translation:.1e}')
    assert np.abs(diff).max() < KS_BAR, diff


@pytest.mark.parametrize('reference', RSH_REFERENCES)
def test_the_range_separated_sop_force_follows_its_energy(reference):
    """LRC-wPBEh: every exchange channel differentiated, the long-range one
    included, and the xc grid's motion."""
    mol = molecule('water')
    chain = sop_chain(mol, 'lrc-wpbeh', reference)
    grad, _ = chain.excitation_gradient()
    translation = np.abs(grad.sum(axis=0)).max()
    assert translation < KS_TRANSLATION_BAR, translation
    diff = fd_residuals(chain, mol, grad, KS_COMPONENTS, KS_STEP)
    print(f'\nwater/LRC-wPBEh ({reference}): analytic - FD '
          f'{np.array2string(diff, precision=2)} Ha/Bohr, translation '
          f'{translation:.1e}')
    assert np.abs(diff).max() < KS_BAR, diff


def test_poles_refit_per_geometry_fail_the_gate(water_pbe0):
    """The gate can fail: the per-geometry fit is 2.6e-5 off on water."""
    mol, chain, grad, _ = water_pbe0
    frozen = {p: v.copy() for p, v in chain.sop_poles.items()}
    try:
        diff = fd_residuals(chain, mol, grad, KS_COMPONENTS[:2], KS_STEP,
                            refit=True)
    finally:
        chain.sop_poles.clear()
        chain.sop_poles.update(frozen)
    assert np.abs(diff).max() > 10 * KS_BAR, diff


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

"""Gates for src/properties/mode_hessian.py: force constants along chosen modes.

Formaldehyde/cc-pVDZ, RHF with density-fitted exchange, relaxed. The excited
surface is BSE@G0W0 S1 (n -> pi*) on the ISDF/space-time chain.

WHAT EACH GATE IS FOR:

- On the ground surface the modes are its own eigenvectors, so every row must
  come back diagonal, carrying that mode's frequency. This tests the
  projection, the per-mode step and the mass weighting together, against
  pyscf's analytic Hessian, which shares none of that code.
- On S1 the curvature from the GRADIENT difference must equal the curvature
  from the ENERGY second difference along the same mode. The second route uses
  no gradient at all, so a gradient that is not the derivative of its own
  energy at displaced geometries (the h^1 defect `hessian.py` found on the
  ISDF mean-field force) fails it.
- There is no step-ladder gate. The ratio of the frequency changes between
  steps 4h, 2h and h reads 4 for an h^2 error and 2 for an h^1 error, but on
  the Hessian-grade grid (GH) the whole ladder moves the C=O and scissor
  frequencies by 0.011 and 0.037 cm^-1 (1.4e-6 and 2.5e-5 of the frequency,
  asymmetry 5e-10 and below): the changes sit at the ISDF force's noise floor
  (HESSIAN_FD_NOISE_REL), ratios 1.85 and 2.36 measure that floor, not an
  order; a ratio window recorded on a grid whose noise lay under the ladder
  would pass for the wrong reason. The h^1 defect the ratio would catch is
  caught by the gradient-versus-energy curvature gate above, which uses no
  ladder.
- The leakage estimate must predict the mode it was told to leave out. S1
  mixes the C=O stretch with the CH2 scissor and the C-H stretch, all a1. The
  two-mode subspace plus its leakage must land on the three-mode result, and
  the shift it predicts must be large enough that the agreement means
  something.
- Noise must be refused rather than diagonalized into a frequency.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.numerics.constants import (HARTREE_TO_CM, ISDF_HESSIAN_MIN_GRID,
                                         SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.numerics.numerics import Numerics  # noqa: E402
from src.gradients.gwbse.excited_state import ExcitedStateChain  # noqa: E402
from src.Base.geometry.geometry import displaced  # noqa: E402
from src.properties.mode_hessian import (cartesian_mode,  # noqa: E402
                                         frequency_ladder, mode_hessian,
                                         mode_step, stretch_modes)
from src.properties.optimize import relax_ground_state
from src.gradients.mean_field_surface import MeanFieldSurface  # noqa: E402
from src.properties.vibronic import normal_modes  # noqa: E402
from tests.isdf_test_grid import counts_at

CH2O = 'C 0 0 0.0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588'
CARBONYL = [(0, 1)]
CH_BONDS = [(0, 2), (0, 3)]


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def ground():
    mol, mf = relax_ground_state(gto.M(atom=CH2O, basis='cc-pvdz', verbose=0),
                                 factory)
    omega, modes, masses, _ = normal_modes(mf, mol)
    return mol, mf, omega, modes, masses


@pytest.fixture(scope='module')
def s1(ground):
    mol = ground[0]
    return ExcitedStateChain(
        mol, factory, state=0, numerics=Numerics(isdf={'auxbasis': 'cc-pvdz-ri', 'counts': counts_at(mol, ISDF_HESSIAN_MIN_GRID)},
                                                 bse={'nroots': 4}))


@pytest.fixture(scope='module')
def carbonyl(ground):
    mol, _, _, modes, masses = ground
    idx, _ = stretch_modes(mol, modes, masses, CARBONYL, top=1)
    return int(idx[0])


def test_the_ground_surface_returns_its_own_normal_modes(ground):
    mol, mf, omega, modes, masses = ground
    out = mode_hessian(MeanFieldSurface(mol, factory, mf=mf, own=True), mol,
                       omega, modes, masses, range(modes.shape[1]))
    # measured 1.35 cm^-1 against pyscf's analytic Hessian
    assert np.abs(np.sort(out['omega_cm'])
                  - np.sort(omega * HARTREE_TO_CM)).max() < 2.0
    block = out['block']
    off = block - np.diag(np.diag(block))
    # measured 5.5e-04
    assert np.abs(off).max() < 2e-3 * np.abs(np.diag(block)).max()


def test_the_carbonyl_and_ch_stretches_are_found_by_bond(ground):
    mol, _, omega, modes, masses = ground
    idx, w = stretch_modes(mol, modes, masses, CARBONYL, top=2)
    # the C=O stretch is the highest-frequency a1 mode below the C-H stretches
    assert omega[idx[0]] * HARTREE_TO_CM == pytest.approx(2013.4, abs=1.0)
    assert w[1] < 0.02 * w[0]
    idx_ch, w_ch = stretch_modes(mol, modes, masses, CH_BONDS, top=3)
    assert set(idx_ch[:2]) == {4, 5}
    assert w_ch[2] < 0.02 * w_ch[1]


def test_the_gradient_curvature_is_the_energy_curvature(ground, s1, carbonyl):
    """R_kk against (E(+h) + E(-h) - 2E(0)) / h^2, Richardson-extrapolated."""
    mol, _, omega, modes, masses = ground
    e0 = s1.total_energy(mol)
    for k in (carbonyl, 2):
        r = mode_hessian(s1, mol, omega, modes, masses, [k])
        second = []
        for cart in (0.02, 0.01):
            h = mode_step(modes[:, k], masses, cart)
            u = cartesian_mode(modes[:, k], masses)
            ep = s1.total_energy(displaced(mol, h * u))
            em = s1.total_energy(displaced(mol, -h * u))
            second.append((ep + em - 2.0 * e0) / h ** 2)
        reference = (4.0 * second[1] - second[0]) / 3.0
        # measured 5.5e-04 (C=O) and 7.8e-04 (scissor)
        assert abs(r['block'][0, 0] - reference) < 2e-3 * abs(reference)


def test_the_leakage_predicts_the_mode_left_out(ground, s1, carbonyl):
    mol, _, omega, modes, masses = ground
    ch = stretch_modes(mol, modes, masses, CH_BONDS, top=1)[0][0]
    two = mode_hessian(s1, mol, omega, modes, masses, [carbonyl, 2])
    three = mode_hessian(s1, mol, omega, modes, masses, [carbonyl, 2, ch])
    predicted = np.sort(two['omega_cm'] + two['leakage_cm'])
    exact = np.sort(three['omega_cm'])[:2]
    # measured on GH: the two roots move 0.080 and 0.198 cm^-1 (0.43 on the
    # 148-point grid); the other CH stretch and the modes below the
    # C=O do not couple to it. The shift must stand above twice the 0.05 cm^-1
    # slack of the prediction bar below, or agreeing within it says nothing.
    moved = np.abs(np.sort(two['omega_cm']) - exact)
    assert moved.max() > 0.1
    assert np.all(np.abs(predicted - exact) < 0.2 * moved + 0.05)


class _NoisySurface:
    """A ground surface whose force carries 1e-4 Ha/Bohr of random noise."""

    def __init__(self, surface, seed=3):
        self.inner = surface
        self.mol0 = surface.mol0
        self.rng = np.random.default_rng(seed)

    def total_gradient(self, mol=None, mf=None):
        g, e, d = self.inner.total_gradient(mol)
        return np.asarray(g) + 1e-4 * self.rng.normal(size=np.shape(g)), e, d


def test_noisy_gradients_are_refused(ground):
    mol, mf, omega, modes, masses = ground
    noisy = _NoisySurface(MeanFieldSurface(mol, factory, mf=mf, own=True))
    with pytest.raises(RuntimeError, match='asymmetric'):
        mode_hessian(noisy, mol, omega, modes, masses, [2, 3])


def test_the_ladder_refuses_steps_it_cannot_read_an_order_from(ground):
    mol, mf, omega, modes, masses = ground
    with pytest.raises(ValueError, match='halved'):
        frequency_ladder(MeanFieldSurface(mol, factory, mf=mf, own=True), mol,
                         omega, modes, masses, [3], steps=(3e-3, 2e-3, 1e-3))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

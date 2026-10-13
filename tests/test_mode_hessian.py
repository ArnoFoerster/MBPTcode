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
- The step ladder must show the h^2 order. A ratio of 2 is that same defect
  seen from the step side.
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

from src.Base.constants import HARTREE_TO_CM  # noqa: E402
from src.gradients.excited_state import ExcitedStateChain  # noqa: E402
from src.properties.mode_hessian import (displaced_along, mode_hessian,  # noqa: E402
                                         mode_step, step_ladder,
                                         stretch_modes)
from src.properties.optimize import MeanFieldSurface, relax_ground_state  # noqa: E402
from src.properties.vibronic import normal_modes  # noqa: E402

CH2O = 'C 0 0 0.0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588'
CARBONYL = [(0, 1)]
CH_BONDS = [(0, 2), (0, 3)]


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
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
    return ExcitedStateChain(mol, factory, state=0, nroots=4,
                             auxbasis='cc-pvdz-ri')


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
            ep = s1.total_energy(displaced_along(mol, modes[:, k], masses, h))
            em = s1.total_energy(displaced_along(mol, modes[:, k], masses, -h))
            second.append((ep + em - 2.0 * e0) / h ** 2)
        reference = (4.0 * second[1] - second[0]) / 3.0
        # measured 5.5e-04 (C=O) and 7.8e-04 (scissor)
        assert abs(r['block'][0, 0] - reference) < 2e-3 * abs(reference)


def test_the_step_ladder_converges_as_the_square_of_the_step(ground, s1,
                                                             carbonyl):
    mol, _, omega, modes, masses = ground
    results, ratios = step_ladder(s1, mol, omega, modes, masses,
                                  [carbonyl, 2])
    # measured 3.84 and 4.03; an h^1 error gives 2
    assert np.all((ratios[0] > 3.3) & (ratios[0] < 4.7))
    assert all(r['asymmetry'] < 1e-6 for r in results)


def test_the_leakage_predicts_the_mode_left_out(ground, s1, carbonyl):
    mol, _, omega, modes, masses = ground
    ch = stretch_modes(mol, modes, masses, CH_BONDS, top=1)[0][0]
    two = mode_hessian(s1, mol, omega, modes, masses, [carbonyl, 2])
    three = mode_hessian(s1, mol, omega, modes, masses, [carbonyl, 2, ch])
    predicted = np.sort(two['omega_cm'] + two['leakage_cm'])
    exact = np.sort(three['omega_cm'])[:2]
    # measured: the scissor-like root moves 16.6 cm^-1, predicted to 1.2
    moved = np.abs(np.sort(two['omega_cm']) - exact)
    assert moved.max() > 5.0
    assert np.all(np.abs(predicted - exact) < 0.2 * moved + 0.5)


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
        step_ladder(MeanFieldSurface(mol, factory, mf=mf, own=True), mol,
                    omega, modes, masses, [3], steps=(3e-3, 2e-3, 1e-3))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

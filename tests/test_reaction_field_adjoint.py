"""Eq. (18)'s adjoints on (eps, X, D), against finite differences.

The forward pass is checked against `GW.reaction_field`'s congruence form,
which reaches the bare screening from the dressed one instead of building it:
the two are the same function of the same chi0 and must agree, or one of the
gauges is wrong. The reverse pass is checked term by term, because the four
adjoints reach the nuclei through four different chains and a single lumped
check would let one hide inside another.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import df, dft, gto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Base.environment import attach_environment
from src.Base.solvent_screening import SolventScreening, detach_solvent_screening
from src.SingleReference.GW.reaction_field import (bare_gauge_transform,
                                                   separable_quasiparticle_shift)
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.reaction_field_adjoint import (reaction_field_backward,
                                                  reaction_field_shift,
                                                  static_grid)

BASIS, AUXBASIS = 'sto-3g', 'cc-pvdz-ri'
STEP = 1e-4


@pytest.fixture(scope='module')
def setup():
    """The two factorizations of one molecule: X is shared, D is not."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=BASIS, verbose=0)
    nocc = mol.nelectron // 2
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.kernel()
    env = SolventScreening(mol, eps=1.78)

    detach_solvent_screening(mf)
    x_mo, d_bare, _, _ = separable_factors(mf, mol, auxbasis=AUXBASIS)
    attach_environment(mf, env)
    x_dr, d_dressed, _, _ = separable_factors(mf, mol, auxbasis=AUXBASIS)
    detach_solvent_screening(mf)
    assert np.abs(x_dr - x_mo).max() < 1e-12, 'the fit must not see the cavity'

    eps = np.asarray(mf.mo_energy, float)
    grid = static_grid(eps, nocc)
    return dict(mol=mol, mf=mf, env=env, nocc=nocc, eps=eps, x_mo=x_mo,
                d_dressed=d_dressed, d_bare=d_bare, grid=grid)


def test_the_two_factorizations_reproduce_the_congruence(setup):
    """One chi0 screened twice, or two factorizations screened once each: the
    same Delta W, since chi0 is a property of the solute and not of its gauge."""
    s = setup
    two = reaction_field_shift(s['x_mo'], s['d_dressed'], s['d_bare'], s['eps'],
                               s['nocc'], grid=s['grid'])

    auxmol = df.addons.make_auxmol(s['mol'], auxbasis=AUXBASIS)
    t = bare_gauge_transform(auxmol, s['env'])
    from src.gradients.space_time_adjoint import chi0_frequency
    chi0 = chi0_frequency(s['x_mo'], s['d_dressed'], s['eps'], s['nocc'],
                          s['grid'])[0]
    w_dressed = np.linalg.inv(np.eye(chi0.shape[0]) - chi0)
    one = separable_quasiparticle_shift(s['x_mo'], s['d_dressed'], w_dressed,
                                        t, s['nocc'])
    rel = np.abs(two - one).max() / np.abs(one).max()
    assert rel < 1e-7, f'the two Delta W constructions differ by {rel:.2e}'


def test_the_gas_phase_is_a_no_op(setup):
    """Both factors bare: Delta W is identically zero, not small."""
    s = setup
    shift = reaction_field_shift(s['x_mo'], s['d_bare'], s['d_bare'], s['eps'],
                                 s['nocc'], grid=s['grid'])
    assert np.abs(shift).max() < 1e-12


def _weights(nmo, seed=0):
    return np.asarray(np.random.default_rng(seed).normal(size=nmo), float)


def _central(f, arr, idx, step=STEP):
    original = arr[idx]
    arr[idx] = original + step
    plus = f()
    arr[idx] = original - step
    minus = f()
    arr[idx] = original
    return (plus - minus) / (2.0 * step)


@pytest.mark.parametrize('target', ['eps', 'x_mo', 'd_dressed', 'd_bare'])
def test_each_adjoint_matches_a_finite_difference(setup, target):
    """Every input the shift depends on, differentiated on its own.

    The grid is FROZEN across the displacement: it is a numerical axis chosen
    from the reference spectrum, not a function of it, and letting it move with
    eps would compare two different quadratures rather than two values of one.
    """
    s = setup
    nocc, grid = s['nocc'], s['grid']
    arrays = {k: np.array(s[k], dtype=float, copy=True)
              for k in ('eps', 'x_mo', 'd_dressed', 'd_bare')}
    w = _weights(len(s['eps']))

    def energy():
        return float(w @ reaction_field_shift(
            arrays['x_mo'], arrays['d_dressed'], arrays['d_bare'],
            arrays['eps'], nocc, grid=grid))

    analytic = dict(zip(('eps', 'x_mo', 'd_dressed', 'd_bare'),
                        reaction_field_backward(
                            w, arrays['x_mo'], arrays['d_dressed'],
                            arrays['d_bare'], arrays['eps'], nocc,
                            grid=grid)))[target]

    arr = arrays[target]
    rng = np.random.default_rng(7)
    flat = rng.choice(arr.size, size=min(6, arr.size), replace=False)
    numeric, exact = [], []
    for k in flat:
        idx = np.unravel_index(k, arr.shape)
        numeric.append(_central(energy, arr, idx))
        exact.append(analytic[idx])
    numeric, exact = np.asarray(numeric), np.asarray(exact)
    scale = max(np.abs(numeric).max(), 1e-8)
    err = np.abs(numeric - exact).max() / scale
    assert err < 1e-6, (f'{target}: analytic {np.round(exact, 9)} vs finite '
                        f'difference {np.round(numeric, 9)}, rel {err:.2e}')

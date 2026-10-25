"""Vertex-corrected self-energies in a continuum, on the Casida route.

GW Gamma_inf and the PSD vertices share every static term with GW: the
Eq. (18) shift is one per orbital whatever Sigma_c is, and Sigma_c itself
screens with the BARE interaction, so a vertex-corrected Sigma_c in a
continuum is its gas-phase self. The shift is added to the root after the
G0W0 solve, whole (`qp_solve.add_reaction_field`), so on one mean field the
continuum moves every method's level by exactly Eq. (18). Inside the
equation it would move it by Z times Eq. (18) only, a spurious blue shift,
which this test fails on.

  1. Restricted water: GW, GW Gamma_inf and PSD1, HOMO and LUMO, levels
     pinned.
  2. An unrestricted doublet: GW and GW Gamma_inf on both spin channels'
     HOMO, levels pinned.

Run as a script (`python tests/test_vertex_solvent.py`) or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.environments.solvent_screening import (attach_solvent_screening,
                                                detach_solvent_screening)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
AUX = 'cc-pvdz-jkfit'
METHODS = ['GW', 'GWGammaInf', 'PSD1']
#: moved - Eq. (18), eV: round-off; Z x Eq. (18) misses by 50-100 meV here
WHOLE_SHIFT_TOL_EV = 1e-8
#: (gas, water) levels in eV, HOMO then LUMO, water/6-31G RHF; the solvent
#: shift is Eq. (18) itself, the same for the three methods (with the shift
#: inside the equation, Z x Eq. (18), the water levels would be 50-70 meV
#: lower on the HOMO)
PINNED = {'GW': ([-12.053613, 5.355737], [-10.378436, 3.747798]),
          'GWGammaInf': ([-12.306862, 5.362849], [-10.631685, 3.754910]),
          'PSD1': ([-12.369361, 5.314714], [-10.694184, 3.706775])}
#: (gas, water) HOMO levels in eV per spin channel, OH/6-31G UHF
PINNED_UHF = {'alpha': {'GW': (-13.572742, -11.869613),
                        'GWGammaInf': (-13.836510, -12.133381)},
              'beta': {'GW': (-12.339572, -10.635695),
                       'GWGammaInf': (-12.575402, -10.871525)}}
PIN_TOL_EV = 1e-4


def _levels(mf, states, methods, **kw):
    out = calc_qp_energy(mf, selfenergy=methods, state=list(states), **kw)
    return {m: np.array([out[p][m] for p in states]) for m in methods}


def _shift_is_eq18(mf, states, methods, channel=None, row=None):
    kw = {} if channel is None else {'spin_channel': channel}
    gas = _levels(mf, states, methods, **kw)
    attach_solvent_screening(mf, solvent='water')
    try:
        solv = _levels(mf, states, methods, **kw)
        eq18 = environment_quasiparticle_shift(mf)
    finally:
        detach_solvent_screening(mf)
    eq18 = (eq18 if row is None else eq18[row])[list(states)] * HARTREE_TO_EV
    for m in methods:
        moved = solv[m] - gas[m]
        assert np.abs(moved - eq18).max() < WHOLE_SHIFT_TOL_EV, (m, moved,
                                                                 eq18)
    return gas, solv


def test_restricted_vertices_in_water():
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-12)
    nocc = mol.nelectron // 2
    gas, solv = _shift_is_eq18(mf, (nocc - 1, nocc), METHODS)
    for m, (want_gas, want_solv) in PINNED.items():
        assert np.abs(gas[m] - want_gas).max() < PIN_TOL_EV, (m, gas[m])
        assert np.abs(solv[m] - want_solv).max() < PIN_TOL_EV, (m, solv[m])


@pytest.mark.parametrize('channel,row', [('alpha', 0), ('beta', 1)])
def test_unrestricted_vertex_in_water(channel, row):
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-12)
    homo = uhf.nelec[row] - 1
    gas, solv = _shift_is_eq18(uhf, (homo,), ['GW', 'GWGammaInf'], channel,
                               row)
    for m, (want_gas, want_solv) in PINNED_UHF[channel].items():
        assert abs(gas[m][0] - want_gas) < PIN_TOL_EV, (m, gas[m])
        assert abs(solv[m][0] - want_solv) < PIN_TOL_EV, (m, solv[m])


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

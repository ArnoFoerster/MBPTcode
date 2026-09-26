"""Vertex-corrected self-energies in a continuum, on the Casida route.

GW Gamma_inf and the PSD vertices share every static term with GW: the
Eq. (18) shift is one per orbital whatever Sigma_c is, and Sigma_c itself
screens with the BARE interaction, so a vertex-corrected Sigma_c in a
continuum is its gas-phase self. The continuum therefore moves a level by
the shift the quasiparticle equation w = eps + <Sigma_x - v_xc> + shift +
Sigma_c(w) makes of it: Z times Eq. (18), with each method's own Z -- which
this route does not report (`return_z` is refused on it), so the check is the
window a frontier pole strength lives in, 0.8 < moved / Eq. (18) <= 1.

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

from src.Base.constants import HARTREE_TO_EV
from src.Base.solvent_screening import (attach_solvent_screening,
                                        detach_solvent_screening)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
AUX = 'cc-pvdz-jkfit'
METHODS = ['GW', 'GWGammaInf', 'PSD1']
#: the window of a frontier quasiparticle's pole strength
Z_WINDOW = (0.8, 1.0)
#: (gas, water) levels in eV, HOMO then LUMO, water/6-31G RHF; the solvent
#: shifts of the three methods agree to 9 meV, which is Z's spread
PINNED = {'GW': ([-12.053613, 5.355737], [-10.448862, 3.765980]),
          'GWGammaInf': ([-12.306862, 5.362849], [-10.693286, 3.771472]),
          'PSD1': ([-12.369361, 5.314714], [-10.755661, 3.724036])}
#: (gas, water) HOMO levels in eV per spin channel, OH/6-31G UHF
PINNED_UHF = {'alpha': {'GW': (-13.572742, -11.937436),
                        'GWGammaInf': (-13.836510, -12.192164)},
              'beta': {'GW': (-12.339572, -10.699835),
                       'GWGammaInf': (-12.575402, -10.926249)}}
PIN_TOL_EV = 1e-4


def _levels(mf, states, methods, **kw):
    out = calc_qp_energy(mf, selfenergy=methods, state=list(states), **kw)
    return {m: np.array([out[p][m] for p in states]) for m in methods}


def _shift_is_z_times_eq18(mf, states, methods, channel=None, row=None):
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
        ratio = (solv[m] - gas[m]) / eq18
        assert np.all((ratio > Z_WINDOW[0]) & (ratio <= Z_WINDOW[1])), (m, ratio)
    return gas, solv


def test_restricted_vertices_in_water():
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-12)
    nocc = mol.nelectron // 2
    gas, solv = _shift_is_z_times_eq18(mf, (nocc - 1, nocc), METHODS)
    for m, (want_gas, want_solv) in PINNED.items():
        assert np.abs(gas[m] - want_gas).max() < PIN_TOL_EV, (m, gas[m])
        assert np.abs(solv[m] - want_solv).max() < PIN_TOL_EV, (m, solv[m])


@pytest.mark.parametrize('channel,row', [('alpha', 0), ('beta', 1)])
def test_unrestricted_vertex_in_water(channel, row):
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    uhf = scf.UHF(mol).density_fit(auxbasis=AUX).run(conv_tol=1e-12)
    homo = uhf.nelec[row] - 1
    gas, solv = _shift_is_z_times_eq18(uhf, (homo,), ['GW', 'GWGammaInf'],
                                       channel, row)
    for m, (want_gas, want_solv) in PINNED_UHF[channel].items():
        assert abs(gas[m][0] - want_gas) < PIN_TOL_EV, (m, gas[m])
        assert abs(solv[m][0] - want_solv) < PIN_TOL_EV, (m, solv[m])


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

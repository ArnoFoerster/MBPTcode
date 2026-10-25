"""The TADF rate table reads every geometry at the labels its own record names.

At the S1 minimum the record names the singlet the walk followed and the
configured triplet; at the T1 minimum the configured singlet and the triplet
the walk followed. When a walk follows its root off the starting index the
two minima name the state differently, and the table indexed BOTH minima's
spin-orbit tables with the S1 walk's singlet and the T1 walk's triplet, and
took the vertical gap at R0 with those labels too.

Here the S1 walk ended on S2 and the T1 walk on T2. Every rate routine is a
stub that records what it was handed; each spin-orbit element must be the
cell of its own minimum's table at that minimum's labels.
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import HARTREE_TO_EV
from src.photophysics import rate_models
from src.photophysics.minimum import from_gw_bse_record
from src.photophysics import rates
from tests.photophysics_cases import gw_bse_props, unit_modes

WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'


def table(offset):
    """A spin-orbit table whose every cell says where it is: offset + 10 row
    + column."""
    return (offset + 10 * np.arange(4)[:, None] + np.arange(4)[None, :]).tolist()


def props(mol, emitter, reservoir, socme):
    names = ('S1', 'S2', 'S3', 'T1', 'T2', 'T3')
    return gw_bse_props(
        mol, {n: -0.9 + 0.01 * k for k, n in enumerate(names)},
        {n: 0.1 + 0.01 * k for k, n in enumerate(names)}, socme,
        emitter, reservoir)


def test_each_minimum_is_read_at_its_own_labels(monkeypatch):
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    n = 3
    modes = unit_modes(mol, np.array([0.01, 0.02, 0.03]))
    minima = [from_gw_bse_record({'props': props(mol, *rec)}, where)
              for where, rec in (('R0', ('S1', 'T1', table(1000))),
                                 ('S1', ('S2', 'T1', table(2000))),
                                 ('T1', ('S1', 'T2', table(3000))))]
    couplings, dipoles = [], []

    def spin_vibronic_rate(de, v, dv, omega, rho, temperature):
        couplings.append(v)
        return 1.0, 1.0, 0.0

    zeros = np.zeros(n)
    for name, fn in {
            'align_to': lambda mol0, m, return_rotation=False: (m, np.eye(3)),
            'huang_rhys_curvilinear': lambda *a: (zeros, None),
            'mode_derivatives': lambda *a, **k: (None, np.zeros((n, 3))),
            'project_coupling': lambda *a: zeros}.items():
        monkeypatch.setattr(rate_models, name, fn)
    # the table itself is the library's (`rates.rate_table`)
    for name, fn in {
            'spin_vibronic_rate': spin_vibronic_rate,
            'marcus_rate': lambda *a, **k: 0.0,
            'marcus_levich_jortner_rate': lambda *a, **k: 0.0,
            'radiative_rate': lambda w, d: dipoles.append(list(d)) or 0.0,
            'internal_conversion_rate': lambda *a: (0.0, zeros),
            'photoluminescence': lambda *a, **k: {},
            # the stub displacement's zero sum rule would refuse every
            # Franck-Condon row, and the T2 channels with them; the
            # manifold's own crossings are not what this reads
            'fc_lambda_guard': lambda lam_rule, lam: {'status': 'computed'},
            'state_manifold': lambda *a: {'status': 'not declared',
                                          'k_nr_t_eff': None}}.items():
        monkeypatch.setattr(rates, name, fn)
    out = rate_models.tadf_rate_table(modes, *minima, mol, 300.0, 0.05,
                                      'stacked')
    isc, risc, t1_s0, t2_t, t2_s = couplings
    # a crossing out of a triplet is handed |V| / sqrt(3), its sublevel average
    risc, t1_s0, t2_t = (c * np.sqrt(rates.TRIPLET_SUBLEVELS)
                         for c in (risc, t1_s0, t2_t))
    assert isc == 2000 + 10 * 2 + 0, 'ISC: S2 x T1 of the S1 minimum'
    assert risc == pytest.approx(3000 + 10 * 1 + 1, rel=1e-15), (
        'RISC: S1 x T2 of the T1 minimum')
    assert t1_s0 == pytest.approx(3000 + 0 + 1, rel=1e-15), (
        'T1 -> S0: S0 x T2 of the T1 minimum')
    assert t2_t == pytest.approx(3000 + 10 * 1 + 2, rel=1e-15), (
        'T3 above the T1 minimum\'s T2')
    assert t2_s == 2000 + 10 * 2 + 1
    assert dipoles == [[0, 2, 0]], 'the emission is the S1 minimum\'s S2'
    assert out['dE_ST_vertical_eV'] == pytest.approx(
        (0.1 - 0.13) * HARTREE_TO_EV)

"""Numerical events of the quasiparticle equation and of the self-consistent
GW loops reach the record of `calc_qp_energy` as named flags.

- every entry point of the one root finder (`find_qp_root`) warns
  `QPWeightOutside` and raises 'z_outside' for a root kept with Z outside
  (0, 1], keeps the root, and raises 'no_root' for a NaN root;
- the Casida and Pade routes of `calc_qp_energy` put 'z_outside' into
  out['numerical_flags'] (a planted pole strength of the wrong sign);
- an evGW or qsGW loop out of cycles puts 'evgw_not_converged' /
  'qsgw_not_converged' there, also for a scalar `state`.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import Numerics
from src.SingleReference.GW.evGW import EvGWNotConverged
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qsGW import QsGWNotConverged
from src.Solvers import qp_equation
from src.Solvers.qp_equation import (QPNoRoot, QPWeightOutside,
                                     largest_weight_root,
                                     solve_qp_equation_pole_strength,
                                     solve_qp_equation_poles,
                                     solve_qp_equation_scan)

# f(w) = w + 0.1/(w - 1): one negative-weight pole, root 0.1127 Ha, Z 1.1455
NUM, POL = [-0.1], [1.0]


def f(w):
    return w - NUM[0] / (w - POL[0])


def fprime(w):
    return 1.0 + NUM[0] / (w - POL[0]) ** 2


ENTRIES = {
    'poles': lambda out: solve_qp_equation_poles(NUM, POL, 0.0, out=out),
    'largest_weight': lambda out: largest_weight_root(NUM, POL, 0.0, out=out),
    'scan': lambda out: solve_qp_equation_scan(f, fprime, 0.0, out=out),
    'pole_strength': lambda out: (solve_qp_equation_pole_strength(
        np.vectorize(f), 0.0, out=out), None),
}


@pytest.mark.parametrize('entry', sorted(ENTRIES))
def test_z_outside_is_kept_warned_and_flagged(entry):
    flags, out = [], {}
    with collecting(flags), pytest.warns(QPWeightOutside, match='outside'):
        w, _ = ENTRIES[entry](out)
    assert abs(w - 0.1127016654) < 1e-6           # the root, kept as solved
    assert out['z_outside'] and not out['no_root']
    assert [e['flag'] for e in flags] == ['z_outside']


def test_a_weight_inside_raises_no_flag():
    flags, out = [], {}
    with collecting(flags):
        w, z = solve_qp_equation_poles([0.1, 0.2], [-1.0, 1.0], 0.0, out=out)
    assert 0.0 < z <= 1.0 and not out['z_outside'] and flags == []


def test_no_root_is_a_named_flag():
    flags, out = [], {}
    with collecting(flags), pytest.warns(QPNoRoot, match='no root'):
        w, z = solve_qp_equation_poles([-0.5], [0.0], 0.0, out=out)
    assert np.isnan(w) and out['no_root']
    assert [e['flag'] for e in flags] == ['no_root']


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


@pytest.mark.parametrize('mode', ['casida', 'imagfrequency'])
def test_z_outside_reaches_calc_qp_energy(water, mode, monkeypatch):
    # every pole strength the sampled equation weighs is made negative: no
    # root is valid, the one nearest the level is kept and flagged
    weight = qp_equation.SampledEquation._weight
    monkeypatch.setattr(qp_equation.SampledEquation, '_weight',
                        lambda self, w: -abs(weight(self, w)))
    out = {}
    with pytest.warns(QPWeightOutside):
        e = calc_qp_energy(water, mode=mode, state='homo', out=out)
    assert np.isfinite(e)
    assert 'z_outside' in [x['flag'] for x in out['numerical_flags']]


@pytest.mark.parametrize('loop, warning, name', [
    ('evGW', EvGWNotConverged, 'evgw_not_converged'),
    ('qsGW', QsGWNotConverged, 'qsgw_not_converged')])
def test_loop_out_of_cycles_reaches_calc_qp_energy(water, loop, warning, name):
    out = {}
    numerics = Numerics(**{loop.lower(): {'max_cycle': 1}})
    with pytest.warns(warning):
        e = calc_qp_energy(water, mode='casida', self_consistency=loop,
                           numerics=numerics, out=out)
    assert isinstance(e, float)                    # a scalar state
    assert name in [x['flag'] for x in out['numerical_flags']]

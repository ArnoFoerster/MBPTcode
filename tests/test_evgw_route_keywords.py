"""`calc_qp_energy(self_consistency='evGW')` hands its route keywords to the loop.

The evGW dispatcher must forward eta and the route's own keywords to
`evgw_eigenvalues`; dropping df, dm_correction, tda and solver would
run a density-fitted evGW under the name of the four-index one, with the
default root selector in every cycle. Each cycle is a `calc_qp_energy` call, so
the keywords reach it and the route applies or refuses them exactly as a
single-shot call would. An eps_anchor, which the loop sets itself on every
cycle, is refused.

H2/cc-pVDZ on PBE0, the molecule of tests/test_evgw.py, density-fitted on
cc-pvdz-ri: without a fit on the mean field the df=True factors decompose the
exact integrals and the two settings cannot be told apart.
"""
import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.GW import evGW
from src.SingleReference.GW.qp_energy import calc_qp_energy
from tests.isdf_test_grid import g1
from tests.pins import assert_same


@pytest.fixture(scope='module')
def mf():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    m = dft.RKS(mol).density_fit(auxbasis='cc-pvdz-ri')
    m.xc = 'pbe0'
    m.conv_tol = 1e-12
    m.kernel()
    return m


def test_every_keyword_reaches_the_loop(mf, monkeypatch):
    """What the dispatcher hands `evgw_eigenvalues`, captured."""
    seen = {}

    def loop(mf_, mol, **kw):
        seen.update(kw)
        return np.asarray(mf_.mo_energy, float), {}

    monkeypatch.setattr(evGW, 'evgw_eigenvalues', loop)
    dm = mf.make_rdm1()
    calc_qp_energy(mf, mode='casida', self_consistency='evGW', df=False,
                   dm_correction=dm, tda=True, solver='dense')
    assert seen['df'] is False
    assert seen['dm_correction'] is dm
    assert seen['tda'] is True
    assert seen['solver'] == 'dense'


def test_casida_evgw_without_density_fitting_is_the_four_index_loop(mf):
    """df=False through the front end is the loop run with df=False, and not
    the density-fitted one."""
    nocc = mf.mol.nelectron // 2
    four_index = calc_qp_energy(mf, mode='casida', self_consistency='evGW',
                                df=False)
    loop, _ = evGW.evgw_eigenvalues(mf, mf.mol, mode='casida', df=False)
    fitted, _ = evGW.evgw_eigenvalues(mf, mf.mol, mode='casida')
    # two runs of the loop: the same root to a few ulp, not bitwise
    assert_same(four_index, float(loop[nocc - 1]) * HARTREE_TO_EV, unit='eV',
                quantity='energy', label='HOMO')
    assert four_index != float(fitted[nocc - 1]) * HARTREE_TO_EV


def test_the_imaginary_axis_loop_refuses_what_its_route_refuses(mf):
    """space-time evGW refuses df=False and tda as single-shot space-time does."""
    with pytest.raises(ValueError, match=r'requires density fitting'):
        calc_qp_energy(mf, mode='space-time', self_consistency='evGW',
                       df=False, counts=g1(mf))
    with pytest.raises(ValueError, match=r'tda has no meaning'):
        calc_qp_energy(mf, mode='space-time', self_consistency='evGW',
                       tda=True, counts=g1(mf))


def test_an_anchor_the_loop_would_overwrite_is_refused(mf):
    eps = np.asarray(mf.mo_energy, float)
    with pytest.raises(TypeError, match='eps_anchor'):
        calc_qp_energy(mf, mode='casida', self_consistency='evGW',
                       eps_anchor=eps)

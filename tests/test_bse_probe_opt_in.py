"""The BSE (A - B) instability probe is opt-in on every route.

On large DF runs the probe costs more than the BSE solve it guards, so
`solve_bse` (dense and Davidson), `solve_bse_isdf` and `solve_bse_df` run it
only when asked:

  * by default min_eig_amb is None and no 'probe' stage is timed;
  * `probe=True` still measures it, positive on stable water/cc-pVDZ, and the
    roots do not move;
  * without the probe a Davidson whose projected (A - B) breaks down on an
    unstable reference still refuses it by name, from the minimum the
    breakdown measured.
"""
import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.numerics import Numerics
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.bse import solve_bse
from src.SingleReference.LinearResponse.davidson import (_CasidaBreakdown,
                                                         solve_bse_df,
                                                         solve_bse_isdf)
from tests.isdf_test_grid import g1, g1_kw

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS, XC = 'cc-pvdz', 'cc-pvdz-ri', 'pbe0'
NROOTS = 3
#: Ha. The probe reads the roots' action only, so it cannot move them.
ROOT_TOL = 1e-12
#: Ha. The negative min eig(A - B) a stubbed breakdown reports.
UNSTABLE_AMB = -0.05


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    mf = dft.RKS(mol).density_fit(auxbasis=AUXBASIS)
    mf.xc = XC
    mf.conv_tol = 1e-11
    mf.kernel()
    return mol, mf, mol.nelectron // 2


@pytest.mark.parametrize('route', ['df', 'isdf', 'dense'])
def test_the_probe_runs_only_when_asked(water, route):
    """Default: no probe, min_eig_amb None. probe=True: a positive minimum on
    a stable reference, the same roots."""
    mol, mf, nocc = water

    def run(**kw):
        if route == 'df':
            return solve_bse_df(mf, mol, nocc, numerics=Numerics(bse={'nroots': NROOTS}), qp=False,
                                progress=False, **kw)
        if route == 'isdf':
            return solve_bse_isdf(mf, mol, nocc, numerics=Numerics(bse={'nroots': NROOTS}), qp=False,
                                  auxbasis=AUXBASIS, progress=False, **g1_kw(mol, kw))
        return solve_bse(mf, mol, nocc, numerics=Numerics(bse={'nroots': NROOTS}), solver='dense',
                         integrals='df', qp=False, **kw)

    om_off, _, _, info_off = run()
    assert info_off['min_eig_amb'] is None
    assert 'probe' not in info_off['timings']
    om_on, _, _, info_on = run(probe=True)
    assert info_on['min_eig_amb'] is not None and info_on['min_eig_amb'] > 0.0
    assert np.abs(np.sort(om_on) - np.sort(om_off)).max() < ROOT_TOL


@pytest.mark.parametrize('route', ['df', 'isdf'])
def test_a_breakdown_names_the_reference_without_the_probe(water, monkeypatch,
                                                           route):
    """A projected (A - B) breakdown that measured min eig(A - B) <= 0 is
    refused as an unstable reference with probe off, not passed through as a
    bare breakdown."""
    mol, mf, nocc = water

    def broken(*args, **kwargs):
        raise _CasidaBreakdown('projected (A-B) not positive definite',
                               UNSTABLE_AMB)

    if route == 'df':
        monkeypatch.setattr(davidson, 'solve_casida_davidson', broken)
        call = lambda: solve_bse_df(mf, mol, nocc, numerics=Numerics(bse={'nroots': NROOTS}), qp=False,
                                    progress=False)
    else:
        monkeypatch.setattr(davidson, '_run_davidson', broken)
        call = lambda: solve_bse_isdf(mf, mol, nocc, numerics=Numerics(bse={'nroots': NROOTS}), qp=False,
                                      auxbasis=AUXBASIS, progress=False, counts=g1(mol))
    with pytest.raises(RuntimeError,
                       match=r'min eig\(A-B\) = -0\.050000 Ha <= 0') as exc:
        call()
    assert not isinstance(exc.value, _CasidaBreakdown)

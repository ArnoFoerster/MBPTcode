"""`calc_qp_energy(..., diagnostics=)` hands the contour driver's decisions out.

The audit tool records the CD grid size, the residues swept, the pole guard,
the Newton seed and the pole model's admission per route; without this hook it
would re-derive them with production's own helpers, which cannot see a guard
the Newton relaxed. With `'diagnostics'` dropped from `CONTOUR_KEYWORDS` the
first test fails with the TypeError the second expects for 'pade'.
"""
import pytest
from pyscf import gto, scf

from src.SingleReference.GW.qp_energy import calc_qp_energy
from tests.isdf_test_grid import g1

def _water():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    return mf


def test_the_contour_route_fills_the_diagnostics():
    mf = _water()
    diagnostics = {}
    e, z = calc_qp_energy(mf, mode='space-time', continuation='cd',
                          return_z=True, state='homo', diagnostics=diagnostics, counts=g1(mf))
    assert diagnostics['continuation'] == 'cd'
    assert isinstance(diagnostics['ntau'], int)
    assert isinstance(diagnostics['nfreq_cd'], int)
    assert 'w0_cd' in diagnostics
    (record,) = diagnostics['states']          # one state asked for
    assert record['state'] == mf.mol.nelectron // 2 - 1
    for key in ('qp_start', 'residues', 'sop_admits',
                'sop_reach'):
        assert key in record, key
    assert 0.0 < z < 1.0 and e < 0.0


def test_the_pade_route_refuses_the_hook():
    mf = _water()
    with pytest.raises(TypeError, match='diagnostics'):
        calc_qp_energy(mf, mode='space-time', continuation='pade',
                       state='homo', diagnostics={}, counts=g1(mf))
    # the same call without the hook is the allowed case
    assert calc_qp_energy(mf, mode='space-time', continuation='pade',
                          state='homo', counts=g1(mf)) < 0.0

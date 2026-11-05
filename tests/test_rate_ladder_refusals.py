"""Gate for `rate_chain.ablation_ladder` on a model whose switches do not all
make a `RateModel`: a gas-phase ladder with an inhomogeneous Gaussian width
reaches the Marcus-Levich-Jortner rung, which has no width. That rung, and the
assumed-lambda rungs that stand on it, are recorded as refused with the
reason, and every other rung runs.

Synthetic formaldehyde, the fixture of test_rate_chain.py.
"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import pytest  # noqa: F401
from pyscf import gto

from src.photophysics.rate_chain import ABLATION_LADDER, ablation_ladder
from src.properties import duschinsky as dm
import test_rate_chain as trc


def test_a_gas_ladder_with_a_width_refuses_the_mlj_rungs_and_goes_on():
    mol = gto.M(atom=trc.GEO, basis='sto-3g', verbose=0)
    inp = trc.build(mol, solvent=False)
    full = replace(trc.FULL, gaussian_sigma_ev=0.02)
    ladder = ablation_ladder(inp, full, trc.T, trc.STACKED, shared=inp)
    assert [r['step'] for r in ladder] == [s for s, _ in ABLATION_LADDER]
    refused = {r['step']: r['refused'] for r in ladder if 'refused' in r}
    mlj = {s for s, _ in ABLATION_LADDER
           if s == 'mlj' or s.startswith('assumed_lambda')}
    assert set(refused) == mlj
    for step in mlj:
        assert refused[step].startswith('not a rate model:')
        assert 'no inhomogeneous width' in refused[step]
    for r in ladder:
        if 'table' in r:
            assert r['table']['ISC']['k_eff'] > 0.0
            assert r['table']['model'] == r['model']
    assert ladder[-1]['model']['density'] == 'mlj'


def test_a_curvilinear_j_outside_its_bar_refuses_its_rungs(monkeypatch):
    """Hessians whose curvilinear J leaves [bar, 1/bar]: every rung on the
    curvilinear J is refused with the reason (RateRefused, not ValueError),
    the rectilinear rung still runs."""
    mol = gto.M(atom=trc.GEO, basis='sto-3g', verbose=0)
    inp = trc.build(mol)
    monkeypatch.setattr(dm, 'DUSCHINSKY_CURVILINEAR_SV_MIN', 0.999)
    ladder = ablation_ladder(inp, trc.FULL, trc.T, trc.STACKED, shared=inp)
    by_step = {r['step']: r for r in ladder}
    assert 'table' in by_step['rectilinear_duschinsky']
    for step, r in by_step.items():
        if step != 'rectilinear_duschinsky':
            assert 'no curvilinear Duschinsky model' in r['refused']


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))

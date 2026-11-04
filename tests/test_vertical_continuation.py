"""The vertical stage on the analytic continuation (--residues pade).

Gates:
  * the continuation's vertical energies stand beside the pole model's on
    the same declaration: formaldehyde, LRC-wPBEh/aug-cc-pVDZ, gas,
    J and K density-fitted (one mean field for both), the admitted set with
    the outside curve; S1, S2, T1 and T2 within CONTINUATION_VS_SOP_MEV (the
    two differ by how Sigma reaches the real axis on the explicit set and by
    the outside curve's anchors; measured below 1 meV);
  * the vertical record of the continuation carries the energies, the
    oscillator strengths and the spin-orbit table and None for every
    derivative coupling;
  * every reverse pass of a continuation chain is refused by name, and so is
    a run asking a stage beyond s0, hess and vert, before any SCF: the walks
    (with their saddle checks), the promoting modes and the rates;
  * scheme B, the quasiparticle Hamiltonian and the adaptive set are refused
    with the continuation in the declaration;
  * water with a per-element basis (O aug-cc-pVDZ, H cc-pVDZ) runs the
    vertical stage end to end on an ISDF-K mean field fitted on 400 pivoted
    AO-product points, the outside curve and the admitted set.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import HARTREE_TO_MEV
from src.photophysics.declaration import DeclarationRefused
from src.photophysics.records import StageRefused
from src.photophysics.stages import run_tadf, vertical
from tests.photophysics_cases import declaration, run_at_r0, run_in
from tests.molecules import WATER_AT_ORIGIN_COARSE

#: meV: the continuation against the pole model on one declaration.
CONTINUATION_VS_SOP_MEV = 3.0
#: The continuation on a set declared by kind (the command line's default set
#: is adaptive, which the continuation refuses).
PADE = ('--residues', 'pade', '--qp-states', 'admitted')
#: The comparison's declaration: one density-fitted mean field for both.
SHARED = ('--basis', 'aug-cc-pvdz', '--scf', 'df', '--qp-states', 'admitted',
          '--outside-scissor', 'curve')
#: Water on a per-element basis, the factors and the ISDF-K mean field each on
#: 400 pivoted AO-product points.
WATER = WATER_AT_ORIGIN_COARSE
HYBRID = ('--system', 'water', '--atom', WATER, '--basis', 'cc-pvdz',
          '--basis-for', 'O=aug-cc-pvdz', '--scf', 'isdf', '--scf-grid', 'G3',
          '--isdf-test-set', 'ao_products', '--pivot-points', '400',
          '--scf-pivot-points', '400',
          '--qp-states', 'admitted', '--outside-scissor', 'curve')


def vert(*extra):
    """The vertical record of the tests' formaldehyde run with `extra`, R0
    the input geometry."""
    return vertical(run_at_r0(tempfile.mkdtemp(), declaration(*extra)))


@pytest.fixture(scope='module')
def records():
    return {'pade': vert(*SHARED, *PADE), 'sop': vert(*SHARED)}


def test_the_continuation_stands_beside_the_pole_model(records):
    pade, sop = (records[k]['props']['omega'] for k in ('pade', 'sop'))
    for name in ('S1', 'S2', 'T1', 'T2'):
        diff = (pade[name] - sop[name]) * HARTREE_TO_MEV
        assert abs(diff) < CONTINUATION_VS_SOP_MEV, (name, diff)
    assert records['pade']['residues'] == 'pade'
    assert records['sop']['residues'] == 'sop'


def test_the_continuation_records_no_derivative(records):
    props = records['pade']['props']
    for name in ('d_ground', 'nac', 'soc_dv_cart', 'soc_dv_paths'):
        assert props[name] is None, name
    assert np.asarray(props['f_osc']).shape == (3,)
    assert props['soc_v0'] == pytest.approx(
        records['sop']['props']['soc_v0'], abs=1e-6)
    routes = records['pade']['qp_bookkeeping']['route']
    assert set(routes.values()) == {'pade'}


def test_every_reverse_pass_is_refused(tmp_path):
    run = run_at_r0(tmp_path, declaration(*PADE))
    man, ev = run.reference(relaxed=False)
    with pytest.raises(NotImplementedError, match='no adjoint'):
        man.driven.reverse
    with pytest.raises(NotImplementedError, match='no adjoint'):
        man.evaluate(run.mol0, ev.mf, gradients=(run.s_key,))


@pytest.mark.parametrize('stages', [('vert', 's1'), ('t1',), ('rate',),
                                    None])
def test_a_walk_stage_is_refused_before_any_scf(tmp_path, stages):
    d = declaration(*PADE)
    with pytest.raises(StageRefused, match='no adjoint'):
        run_tadf(d, run_in(tmp_path, d).store, max_memory=4000,
                 stages=stages)


def test_the_declaration_refuses_what_the_continuation_does_not_carry():
    with pytest.raises(DeclarationRefused, match='scheme-b-eq'):
        declaration(*PADE, '--environment', 'toluene', '--solvent-scheme',
                    'scheme-b-eq', '--scf', 'df', '--qp-states', 'frontier',
                    '--no-qp-hamiltonian')
    with pytest.raises(DeclarationRefused, match='hamiltonian'):
        declaration(*PADE, '--qp-hamiltonian')
    with pytest.raises(DeclarationRefused, match='adaptive'):
        declaration('--residues', 'pade')


def test_a_hybrid_basis_vertical_on_the_pivoted_mean_field(tmp_path):
    run = run_at_r0(tmp_path, declaration(*HYBRID, *PADE))
    rec = vertical(run)
    assert run.frozen.mf0.with_df.test_set == 'ao_products'
    omega = rec['props']['omega']
    assert 0 < omega['T1'] < omega['S1']
    assert rec['props']['nac'] is None

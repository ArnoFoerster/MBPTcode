"""A stored finite-difference Hessian is reused only if it froze what this one freezes.

Every displaced ISDF-K mean field of a numerical Hessian fits on the reference
geometry's pair layout and places its points on the reference's atomic
frames. A Hessian made before either screened its own pairs and turned its
own frames at each displacement: another Hessian, rough by an O(1/h) column
error or by percent asymmetries, under the same mean-field and grid stamp.
The hess record's stamp therefore carries `fd_hessian` (pair layout and
frames) for the numerical route, and a record without it is refused; an
analytic Hessian stamps nothing of the kind and is reused as before. The
UISC rate model reads the same hess stage, so one rule covers both models.
"""
import numpy as np
import pytest

from src.photophysics import stages
from src.photophysics.store import StoreRefused
from src.properties.hessian import REFERENCE_FRAMES
from tests.photophysics_cases import declaration, run_at_r0, run_of

FROZEN = {'pair_layout': 'reference', 'frames': REFERENCE_FRAMES}


def hess_run(tmp_path, hessian, *extra):
    return run_of(tmp_path, '--hessian', hessian, *extra,
                  qp_states='admitted')


def write_hess(run, stamp):
    run.store.write('hess', {'hess': np.eye(3)}, stamp)


def test_a_numerical_hessian_that_froze_nothing_is_refused(tmp_path):
    run = hess_run(tmp_path, 'numerical')
    want = run.stamp('hess')
    assert want['fd_hessian'] == FROZEN
    assert want['hessian_numerics'] == run.declaration.hessian_numerics.identity()
    before = {k: v for k, v in want.items() if k != 'fd_hessian'}
    write_hess(run, before)
    with pytest.raises(StoreRefused, match='fd_hessian'):
        run.reusable('hess')
    write_hess(run, want)
    assert run.reusable('hess')


def test_an_analytic_hessian_is_reused_as_before(tmp_path):
    run = hess_run(tmp_path, 'analytic-df')
    want = run.stamp('hess')
    assert 'fd_hessian' not in want
    write_hess(run, want)
    assert run.reusable('hess')


def test_the_uisc_model_reads_the_same_hess_stamp(tmp_path):
    tadf = hess_run(tmp_path / 'tadf', 'numerical')
    uisc = hess_run(tmp_path / 'uisc', 'numerical', '--rate-model', 'uisc')
    assert uisc.stamp('hess') == tadf.stamp('hess')
    assert uisc.stamp('hess')['fd_hessian'] == FROZEN


def test_the_numerical_route_fits_on_the_reference_layout(tmp_path,
                                                          monkeypatch):
    """The numerical ground_modes stage leaves `numerical_hessian` its
    reference pair layout and frames (the defaults the stamp names), on the
    derived Hessian numerics' mean field; a sharded run divides the
    displaced forces over the ranks of the one launch, not over separate
    processes, so no displaced force exists outside that call."""
    decl = declaration('--hessian', 'numerical', '--hessian-grid', 'G3',
                       qp_states='admitted')
    run = run_at_r0(tmp_path, decl)
    seen = {}

    def hessian(mol, factory, **kw):
        seen.update(kw, mol=mol, factory=factory)
        n = mol.natm
        return np.eye(3 * n).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    monkeypatch.setattr(stages, 'numerical_hessian', hessian)
    record = stages.ground_modes(run)
    assert 'pair_layout' not in seen and 'frames' not in seen
    assert seen['map_fn'](abs, [-1, 2]) == [1, 2], \
        'one process: the map is the builtin one'
    assert np.array_equal(seen['mol'].atom_coords(), run.mol0.atom_coords())
    assert decl.hessian_numerics.scf.grid_level == 'G3'
    assert run.stamp('hess')['fd_hessian'] == FROZEN
    assert run.read('hess')['route'] == 'numerical' == record['route']

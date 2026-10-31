"""The photophysics ISDF-K mean field on its own declared fit.

Gates:
  * --scf-isdf-test-set ao_products with --scf-pivot-points, and
    --scf-pivot-points alone, reach the mean field: its ISDF-K handle is
    fitted on the AO-product test set with that pivot count, and the record's
    mean-field name and s0 stamp say so; --scf-isdf-test-set duchemin_blase
    and no declaration keep the tabulated test set; beside --scf df the two
    are refused;
  * the factors' fit (--isdf-test-set, --pivot-points) never reaches the
    mean field: the factory makes the call it makes with nothing declared
    (no Numerics), and the s0 stamp is the undeclared one;
  * a handle on another fit than the declared one is refused
    (`require_declared_fit`), in the factory and on its own;
  * the s0 walk runs (water, LRC-wPBEh/cc-pVDZ, ISDF-K GH) with the factors
    pivoted and the mean field tabulated, and with the mean field pivoted,
    whose displaced geometries place the reference's pivoted points on the
    frozen frames (`MeanFieldSurface.frozen_frames`), its force the
    derivative of that energy within FORCE_TOL;
  * the vertical stage stands on a pivoted mean field.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.geometry.geometry import molecule_at
from src.gradients.mean_field_surface import MeanFieldSurface
from src.photophysics import chain
from src.photophysics.cli import declaration_of, parser
from src.photophysics.declaration import scf_fit
from src.photophysics.stages import run_tadf, vertical
from tests.photophysics_cases import argv, declaration, run_at_r0, run_in

SCF_FIT = ('--scf-isdf-test-set', 'ao_products', '--scf-pivot-points', '200')
FACTOR_FIT = ('--isdf-test-set', 'ao_products', '--pivot-points', '200')
#: Water a little off its minimum, Angstrom: a short s0 walk.
WATER = ('--system', 'water', '--atom',
         'O 0 0 0.02; H 0 0.78 0.58; H 0 -0.75 0.60')
#: Ha/Bohr: the pivoted surface's force against its central difference.
FORCE_TOL = 2e-6


def handle_of(*extra):
    """The ISDF-K handle of the tests' run with `extra`, built, not run."""
    d = declaration(*extra)
    run = run_in(tempfile.mkdtemp(), d)
    return d, run, run.factory(run.mol_in).with_df


@pytest.mark.parametrize('extra, want', [
    (SCF_FIT, ('ao_products', 200)),
    (('--scf-pivot-points', '200'), ('ao_products', 200)),
    (('--scf-isdf-test-set', 'ao_products'), ('ao_products', None)),
    (('--scf-isdf-test-set', 'duchemin_blase'), ('duchemin_blase', None)),
    (FACTOR_FIT, ('duchemin_blase', None)),
    ((), ('duchemin_blase', None))])
def test_the_declared_fit_reaches_the_mean_field(extra, want):
    d, _, handle = handle_of(*extra)
    pivots = handle.pivot_points if handle.test_set == 'ao_products' else None
    assert (handle.test_set, pivots) == want
    declared = bool(extra) and extra[0].startswith('--scf')
    assert (scf_fit(d.numerics) is not None) == declared


def test_the_scf_fit_beside_df_is_refused():
    with pytest.raises(SystemExit, match='--scf df'):
        declaration_of(parser().parse_args(argv('--scf', 'df',
                                                '--scf-pivot-points', '200')))


def test_the_record_names_the_fit():
    d = declaration(*SCF_FIT)
    assert d.mean_field_name == 'ISDF-K GH, ao_products fit on 200 pivoted points'
    plain, factors = declaration(), declaration(*FACTOR_FIT)
    assert plain.mean_field_name == factors.mean_field_name == 'ISDF-K GH'
    s0 = {k: run_in(tempfile.mkdtemp(), x).stamp.stamps['s0']
          for k, x in (('scf', d), ('plain', plain), ('factors', factors))}
    assert s0['scf'] != s0['plain'] == s0['factors']
    assert 'test_set' not in s0['plain']['scf']
    assert s0['scf']['mean_field']['name'] == d.mean_field_name


@pytest.mark.parametrize('extra', [(), FACTOR_FIT])
def test_no_declared_fit_is_the_undeclared_build(monkeypatch, extra):
    """With no mean-field fit declared the factory calls
    `isdf_jk_with_force` exactly as before (no Numerics), whatever the
    factors are fitted on, so the mean field is the old one bit for bit."""
    d = declaration(*extra)
    assert chain.scf_fit_keyword(d.numerics) == {}
    seen = []
    real = chain.isdf_jk_with_force

    def spy(mf, **kw):
        seen.append(sorted(kw))
        return real(mf, **kw)
    monkeypatch.setattr(chain, 'isdf_jk_with_force', spy)
    run = run_in(tempfile.mkdtemp(), d)
    run.factory(run.mol_in)
    assert seen == [['auxbasis', 'counts', 'n_start']]


def test_another_fit_is_refused(monkeypatch):
    d = declaration(*SCF_FIT)
    with pytest.raises(chain.UndeclaredFit, match="'ao_products' with 200"):
        chain.require_declared_fit(
            types.SimpleNamespace(test_set='duchemin_blase', pivot_points=None),
            d.numerics)
    with pytest.raises(chain.UndeclaredFit):
        chain.require_declared_fit(
            types.SimpleNamespace(test_set='ao_products', pivot_points=150),
            d.numerics)
    chain.require_declared_fit(
        types.SimpleNamespace(test_set='ao_products', pivot_points=200),
        d.numerics)
    real = chain.isdf_jk_with_force
    monkeypatch.setattr(chain, 'isdf_jk_with_force',
                        lambda mf, numerics=None, **kw: real(mf, **kw))
    run = run_in(tempfile.mkdtemp(), d)
    with pytest.raises(chain.UndeclaredFit):
        run.factory(run.mol_in)


@pytest.mark.parametrize('extra', [FACTOR_FIT, SCF_FIT])
def test_the_s0_walk_runs(tmp_path, extra):
    d = declaration(*extra, *WATER)
    record = run_tadf(d, run_in(tmp_path, d).store, max_memory=4000,
                      stages=('s0',))['s0']
    flags = record.get('numerical_flags') or {}
    assert 'walk_not_converged' not in flags, flags


def test_the_pivoted_ground_surface_keeps_its_selection():
    d = declaration(*SCF_FIT, *WATER)
    run = run_in(tempfile.mkdtemp(), d)
    surface = MeanFieldSurface(run.mol_in, run.factory)
    frames = surface.frozen_frames()
    assert frames is not None and frames.selection is not None
    x0 = run.mol_in.atom_coords()
    step = np.zeros_like(x0)
    step[1, 1] = 1e-3
    energies = []
    for sign in (1, -1):
        mol = molecule_at(run.mol_in, x0 + sign * step)
        _, mf = surface.mean_field(mol)
        # the reference's frames and its pivoted points, placed, not pivoted
        assert mf.with_df.frozen_frames is frames
        assert mf.with_df.selection is frames.selection
        energies.append(surface.total_energy(mol))
    g = np.asarray(surface.total_gradient(run.mol_in)[0])
    e = [x[0] if isinstance(x, tuple) else x for x in energies]
    fd = (e[0] - e[1]) / 2e-3
    assert abs(g[1, 1] - fd) < FORCE_TOL, (g[1, 1], fd)


def test_the_vertical_stands_on_the_declared_fit(tmp_path):
    run = run_at_r0(tmp_path, declaration(*SCF_FIT, '--qp-states', 'admitted'))
    rec = vertical(run)
    handle = run.frozen.mf0.with_df
    assert (handle.test_set, handle.pivot_points) == ('ao_products', 200)
    omega = rec['props']['omega']
    assert 0 < omega['T1'] < omega['S1']

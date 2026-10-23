"""A Davidson root left unconverged outside the excited-state chain is kept
as solved, warned about and named in what its caller returns
(every Davidson caller warns and flags instead of refusing):

  * `solve_bse` (and the two matrix-free drivers under it): info's
    'davidson_flags' (`davidson.davidson_flags`), empty on the dense route;
  * `isdf_manifolds`: each manifold's `davidson_flags`, which
    `spin_orbit_couplings` copies by spin;
  * the photophysics ISDF-vs-DF acceptance check: by spin and route in its
    result, which the stage record keeps under `isdf_df_check`.

A water/cc-pVDZ DF BSE stopped after one cycle; stand-ins for the GW, the
factors and the check's mean field.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics, Stamped
from src.SingleReference.BSE.solve import solve_bse
from src.photophysics import df_check
from src.properties import spin_orbit
from tests.photophysics_cases import run_of

WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
STUCK = {'davidson_not_converged': {'roots': [1], 'residuals': [1e-3],
                                    'verdict': 'unconverged'}}


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


def test_solve_bse_keeps_and_names_an_unconverged_root():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    numerics = Numerics(bse={'nroots': 3, 'conv_tol': 1e-10, 'max_cycle': 1})
    with pytest.warns(RuntimeWarning, match='unconverged'):
        om, _, _, info = solve_bse(mf, mol, 5, numerics=numerics,
                                   integrals='df', qp=False)
    flags = info['davidson_flags']['davidson_not_converged']
    assert flags['roots'] and len(om) == 3
    assert all(r > 1e-10 for r in flags['residuals'])
    _, _, _, dense = solve_bse(mf, mol, 5, numerics=numerics,
                               solver='dense', integrals='df',
                               qp=False)
    assert dense['davidson_flags'] == {}


def test_spin_orbit_manifolds_carry_the_flags(water, monkeypatch):
    mol, mf = water
    nov = 5 * 2
    x = np.eye(nov)[:, :2]
    info = {'eps': mf.mo_energy, 'W_aux': None, 'factors': None,
            'numerics_identity': 'id', 'davidson_flags': {}}
    monkeypatch.setattr(spin_orbit, 'solve_bse_isdf',
                        lambda *a, **k: (np.array([0.3, 0.4]), x, 0 * x,
                                         info))

    def triplet(lr, nocc, timings=None, **kw):
        timings.update(davidson_unconverged=[False, True],
                       davidson_unconfirmed=[False, False],
                       davidson_residuals=[1e-9, 1e-3],
                       davidson_verdict='unconverged')
        return np.array([0.2, 0.35]), x, 0 * x

    monkeypatch.setattr(spin_orbit, 'solve_casida_davidson', triplet)
    s, t = spin_orbit.isdf_manifolds(mf, mol, 5)
    assert s.davidson_flags == {}
    assert t.davidson_flags == STUCK
    out = spin_orbit.spin_orbit_couplings(mf, mol, 5, s, t)
    assert out['davidson_flags'] == {'triplet': STUCK}
    plain = spin_orbit.spin_orbit_couplings(
        mf, mol, 5, Stamped(tuple(s), 'id'), Stamped(tuple(t), 'id'))
    assert plain['davidson_flags'] == {}


def test_the_df_check_names_an_unconverged_root(tmp_path, monkeypatch):
    run = run_of(tmp_path, '--df-check', '1')
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    record = {'props': {'coords_bohr': mol.atom_coords(),
                        'root': {'S1': 0, 'T1': 0},
                        'omega': {'S1': 0.3, 'T1': 0.2}},
              'qp_bookkeeping': {}, 'span': {}}
    monkeypatch.setattr(run, 'read', lambda name: record)
    monkeypatch.setattr(df_check, 'molecule_at', lambda *a: mol)
    monkeypatch.setattr(df_check, 'df_mean_field',
                        lambda *a: SimpleNamespace(e_tot=-75.0))
    monkeypatch.setattr(df_check, 'occupied_count', lambda mf: 5)
    monkeypatch.setattr(df_check, 'chain_diagonal', lambda mf, b: (None, 0.0))
    monkeypatch.setattr(df_check, 'separable_factors', lambda *a, **k: None)
    monkeypatch.setattr(df_check, 'label',
                        lambda key: 'S1' if key[0] == 'singlet' else 'T1')
    roots = np.array([0.3, 0.4])
    monkeypatch.setattr(df_check, 'solve_bse_isdf', lambda *a, **k: (
        roots, None, None, {'davidson_flags': {}}))
    monkeypatch.setattr(df_check, 'solve_bse_df', lambda *a, **k: (
        roots, None, None, {'davidson_flags': STUCK}))
    out = df_check.check_stage(run, 'vert')
    assert out['davidson_flags']
    for spin, flags in out['davidson_flags'].items():
        assert flags == {'df': STUCK}
        assert out['spins'][spin]['davidson_flags'] == {'df': STUCK}

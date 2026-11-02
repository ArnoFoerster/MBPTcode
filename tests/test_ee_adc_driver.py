"""End-to-end gates: every EE-ADC route the driver exposes must give the same
excitation energies -- spin-free dense, spin-free Davidson, density-fitted,
and the spin-orbital arbiter -- and the singlet channel must match pyscf.
Every route hands the Davidson's convergence flags to the caller
(.converged), and a dense solve reports every root converged."""
import numpy as np
import pytest

from pyscf import gto, scf, adc

from src.SingleReference.ADC.eeADC import ee_driver
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from tests.adc_common import ee_numerics

ATOMS = ['H 0 0 0; F 0 0 0.917',
         'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
         'Be 0 0 0; H 0 0 1.34; H 0 0 -1.34']


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level,method', [('adc2', 'adc(2)'),
                                          ('adc2x', 'adc(2)-x'),
                                          ('adc3', 'adc(3)')])
def test_routes_agree_and_match_pyscf(atom, level, method):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    a = adc.ADC(mf); a.method = method; a.method_type = 'ee'; a.verbose = 0
    ref = np.sort(np.array(a.kernel(nroots=3)[0]))

    runs = {
        'dense': solve_ee_adc(mf, level=level, nroots=3, spin='singlet',
                              matrix_free=False)[0],
        'davidson': solve_ee_adc(mf, level=level, nroots=3, spin='singlet')[0],
        # auxbasis='exact' is the plumbing check: the DF kernels must
        # reproduce the dense route bit for bit. Real RI accuracy is a
        # separate, tolerance-based test below.
        'df': solve_ee_adc(mf, level=level, nroots=3, spin='singlet',
                           df=True, auxbasis='exact')[0],
        'spinorbital': solve_ee_adc(mf, level=level, nroots=3,
                                    route='spinorbital', spin='singlet')[0],
    }
    for name, e in runs.items():
        assert np.abs(np.sort(e) - ref).max() < 1e-6, name


@pytest.mark.parametrize('atom', ATOMS[:2])
def test_triplet_channel_agrees_across_routes(atom):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    a = solve_ee_adc(mf, level='adc3', nroots=3, spin='triplet')[0]
    b = solve_ee_adc(mf, level='adc3', nroots=3, spin='triplet',
                     route='spinorbital')[0]
    assert np.abs(np.sort(a) - np.sort(b)).max() < 1e-6


@pytest.mark.parametrize('atom', ATOMS[:2])
def test_default_df_is_within_fitting_error(atom):
    """df=True with no auxbasis uses pyscf's DEFAULT fitting basis, i.e. a
    real RI approximation -- close, not exact. Keeping this separate from the
    'exact' plumbing check is what stopped a silent switch between the two
    from going unnoticed."""
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    ref = solve_ee_adc(mf, level='adc3', nroots=3, spin='singlet')[0]
    got = solve_ee_adc(mf, level='adc3', nroots=3, spin='singlet', df=True)[0]
    d = np.abs(np.sort(got) - np.sort(ref)).max()
    assert d < 0.05, d
    assert d > 0, 'default DF should not be bit-identical to dense integrals'


def test_bad_options_raise():
    mol = gto.M(atom=ATOMS[0], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    with pytest.raises(ValueError):
        solve_ee_adc(mf, level='adc3', route='spinorbital', df=True)
    with pytest.raises(ValueError):
        solve_ee_adc(mf, level='adc3', spin='doublet')
    with pytest.raises(ValueError):
        solve_ee_adc(mf, level='adc5')


def test_every_route_returns_the_convergence_flags(monkeypatch):
    """.converged on every non-fold route: True per root from a dense solve,
    and the Davidson's own flags, here forced False, from an iterative one,
    with (e, Z) still unpacking as before."""
    mol = gto.M(atom=ATOMS[1], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    uhf = scf.UHF(mol).density_fit().run()
    routes = {'spin-free': dict(spin='singlet'),
              'spin-free, both channels': dict(return_parity=True),
              'spin-orbital': dict(route='spinorbital'),
              'spin-orbital, Ms sector': dict(route='spinorbital', ms_sector=0),
              'unrestricted': dict(route='unrestricted', df=True, mf=uhf)}
    for name, kw in routes.items():
        kw = dict(kw)
        m = kw.pop('mf', mf)
        r = solve_ee_adc(m, level='adc2', nroots=3, matrix_free=False, **kw)
        assert r.converged.shape == (3,) and r.converged.all(), name
    real = ee_driver.solve_symmetric

    def unconverged(*a, **k):
        e, X, conv = real(*a, **k)
        return e, X, np.zeros_like(conv)
    monkeypatch.setattr(ee_driver, 'solve_symmetric', unconverged)
    for name, kw in routes.items():
        kw = dict(kw)
        m = kw.pop('mf', mf)
        e, Z, *_ = r = solve_ee_adc(m, level='adc2', nroots=3, numerics=ee_numerics(dense_limit=0),
                                    **kw)
        assert len(e) == 3 and Z.shape[1] == 3, name
        assert r.converged.shape == (3,) and not r.converged.any(), name


def test_the_fold_reads_its_settings_from_numerics(monkeypatch):
    """conv_tol, residual_tol and dense_limit of numerics.adc reach the fold's
    solver, and layout.tile_gb its operator: Numerics is their one source."""
    mf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74',
                       basis='6-31g', verbose=0)).density_fit().run()
    seen = {}

    class Stop(Exception):
        pass

    def solve_folded(pieces, nroots, **kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(ee_driver.ee_fold, 'solve_folded', solve_folded)
    real = ee_driver.ee_ri_fold.RIFoldADC2

    def operator(*a, **kw):
        seen['tile_gb'] = kw['tile_gb']
        return real(*a, **kw)
    monkeypatch.setattr(ee_driver.ee_ri_fold, 'RIFoldADC2', operator)
    with pytest.raises(Stop):
        solve_ee_adc(mf, level='adc2', nroots=1, spin='singlet', fold=True,
                     auxbasis='cc-pvdz-ri',
                     numerics=ee_numerics(conv_tol=1e-7, residual_tol=1e-9,
                                          dense_limit=3, tile_gb=0.5))
    assert (seen['tol_omega'], seen['tol_residual'], seen['dense_limit'],
            seen['tile_gb']) == (1e-7, 1e-9, 3, 0.5), seen

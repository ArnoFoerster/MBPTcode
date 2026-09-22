"""End-to-end gates: every EE-ADC route the driver exposes must give the same
excitation energies -- spin-free dense, spin-free Davidson, density-fitted,
and the spin-orbital arbiter -- and the singlet channel must match pyscf."""
import numpy as np
import pytest

from pyscf import gto, scf, adc

from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc

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

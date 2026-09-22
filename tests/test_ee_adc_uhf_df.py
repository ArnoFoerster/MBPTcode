"""Unrestricted density-fitted EE-ADC (ee_u_r_sigma_df) against its oracle.

The dense spin-orbital route is the arbiter here exactly as it was for the
closed-shell DF path: it forms the full antisymmetrized <pq||rs> over spin
orbitals and makes no approximation beyond the reference, so with an EXACT
(eigendecomposed) B the two must agree to machine precision. Anything that
does not is a bug in the spin bookkeeping, not an RI error.

The molecules are genuine open shells with nocc_a != nocc_b (OH is the 2-Pi
radical whose degenerate singly-occupied shell broke the MP denominators
once, so it is deliberately kept in), plus a closed shell driven through the
unrestricted path as a degenerate case.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals
from src.SingleReference.ADC.eeADC import (ee_driver, ee_utils, ee_u_dense_full,
                                     ee_spin_blocks as sb)
from src.SingleReference.ADC.eeADC import ee_u_r_sigma_df as u
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from src.Base.constants import HARTREE_TO_EV

LEVELS = ('adc1', 'adc2', 'adc2x', 'adc3')

CASES = {
    'oh': ('O 0 0 0; H 0 0 0.97', 1, '6-31g'),
    'nh2': ('N 0 0 0; H 0 1.0 0.3; H 0 -1.0 0.3', 1, 'sto-3g'),
    'ch3': ('C 0 0 0; H 0 1.08 0; H 0.935 -0.54 0; H -0.935 -0.54 0',
            1, 'sto-3g'),
    'h2o': ('O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', 0, 'sto-3g'),
}


def _reference(name):
    atom, spin, basis = CASES[name]
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0, symmetry=False)
    mf = scf.UHF(mol).run()
    assert mf.converged
    return mol, mf


def _arrays(mol, mf):
    eps, g, nocc = ee_driver.spin_orbital_arrays(mf, mol)
    dfi = DFIntegrals.from_scf(mol, mf, exact=True)
    no_a, no_b = mf.nelec
    return (eps, g, nocc, dfi, no_a, no_b,
            np.asarray(mf.mo_energy[0]), np.asarray(mf.mo_energy[1]))


def _dense_spectrum(eps, g, nocc, mf, level, nroots, en_dress=None):
    norb = len(eps)
    H = ee_u_dense_full.build_supermatrix(eps, g, nocc, level=level,
                                          en_dress=en_dress)
    sz = ee_utils.spin_labels(mf, nocc, norb)
    mask = ee_utils.ms_sector_mask(sz, nocc, norb, 0)
    if level == 'adc1':
        mask = mask[:ee_utils.dimensions(nocc, norb)['n_s']]
    return np.linalg.eigvalsh(H[np.ix_(mask, mask)])[:nroots], int(mask.sum())


def _dense_operator(aop, n):
    return np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])


# ----------------------------------------------------------------------
# integral blocks
# ----------------------------------------------------------------------

@pytest.mark.parametrize('name', ['oh', 'nh2'])
def test_integral_blocks_match_spin_orbital(name):
    """g_blocks_df_uhf == the dense spin-orbital blocks, sliced by spin.

    This is where the `ba` blocks earn their own construction: the identity
    <p_b q_a||r_b s_a> = <q_a p_b||s_a r_b> permutes the ORBITAL FAMILIES
    too, so deriving `baba` from `abab` by transposition is only valid for
    families symmetric under (1,0,3,2) and produced shape errors for ooov,
    ovov and ovvv.
    """
    mol, mf = _reference(name)
    eps, g, nocc, dfi, no_a, no_b, _, _ = _arrays(mol, mf)
    norb_a = norb_b = mol.nao
    nv_a, nv_b = norb_a - no_a, norb_b - no_b
    got = u.g_blocks_df_uhf(dfi.B_aa, dfi.B_bb, no_a, no_b, norb_a, norb_b)
    ref = ee_utils.g_blocks(g, nocc, len(eps))
    for fam in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvv'):
        oracle = sb.from_spin_orbital_uhf(ref[fam], fam, no_a, no_b,
                                          nv_a, nv_b)
        for key in set(oracle.keys()) | set(got[fam].keys()):
            A, Bk = oracle.get(key), got[fam].get(key)
            assert A is not None and Bk is not None, f'{fam}.{key} missing'
            assert A.shape == Bk.shape, f'{fam}.{key} {A.shape} vs {Bk.shape}'
            assert np.abs(A - Bk).max() < 1e-10, f'{fam}.{key}'


# ----------------------------------------------------------------------
# operator
# ----------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(CASES))
@pytest.mark.parametrize('level', LEVELS)
def test_spectrum_matches_dense_spin_orbital(name, level):
    mol, mf = _reference(name)
    eps, g, nocc, dfi, no_a, no_b, eps_a, eps_b = _arrays(mol, mf)
    e_ref, n_ref = _dense_spectrum(eps, g, nocc, mf, level, 6)
    aop, diag, d = u.build_operator(eps_a, eps_b, dfi.B_aa, dfi.B_bb,
                                    no_a, no_b, level=level)
    assert d['nH'] == n_ref, 'Delta-Ms = 0 sector dimension'
    Hd = _dense_operator(aop, d['nH'])
    e = np.linalg.eigvalsh(0.5 * (Hd + Hd.T))[:6]
    assert np.abs(e - e_ref).max() < 1e-9


@pytest.mark.parametrize('name', ['oh', 'nh2'])
@pytest.mark.parametrize('level', LEVELS)
def test_operator_symmetric_and_diagonal_exact(name, level):
    """The analytic diagonal feeds the Davidson preconditioner; if it drifts
    from the operator the solver still converges, just slowly and silently."""
    mol, mf = _reference(name)
    _, _, _, dfi, no_a, no_b, eps_a, eps_b = _arrays(mol, mf)
    aop, diag, d = u.build_operator(eps_a, eps_b, dfi.B_aa, dfi.B_bb,
                                    no_a, no_b, level=level)
    Hd = _dense_operator(aop, d['nH'])
    assert np.abs(Hd - Hd.T).max() < 1e-10
    assert np.abs(np.diag(Hd) - diag).max() < 1e-10


@pytest.mark.parametrize('level', ['adc2', 'adc2x', 'adc3'])
@pytest.mark.parametrize('dress', [
    {'hh': True, 'pp': True, 'hp': False},
    {'hh': True, 'pp': False, 'hp': False},
    {'hh': False, 'pp': True, 'hp': False},
    {'hh': True, 'pp': True, 'hp': True},
])
def test_en_matches_dense_spin_orbital(level, dress):
    mol, mf = _reference('nh2')
    eps, g, nocc, dfi, no_a, no_b, eps_a, eps_b = _arrays(mol, mf)
    e_ref, _ = _dense_spectrum(eps, g, nocc, mf, level, 4, en_dress=dress)
    aop, _, d = u.build_operator(eps_a, eps_b, dfi.B_aa, dfi.B_bb,
                                 no_a, no_b, level=level, en_dress=dress)
    Hd = _dense_operator(aop, d['nH'])
    e = np.linalg.eigvalsh(0.5 * (Hd + Hd.T))[:4]
    assert np.abs(e - e_ref).max() < 1e-9


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

@pytest.mark.parametrize('level', LEVELS)
def test_driver_route_matches_arbiter(level):
    mol, mf = _reference('oh')
    e_ref, _ = solve_ee_adc(mf, mol, level=level, nroots=4,
                            route='spinorbital', df=False)
    e_df, _ = solve_ee_adc(mf, mol, level=level, nroots=4,
                           route='unrestricted', df=True, auxbasis='exact')
    err = np.abs(np.asarray(e_df) - np.asarray(e_ref)).max() * HARTREE_TO_EV
    assert err < 1e-8, f'{err:.2e} eV'


def test_closed_shell_through_the_unrestricted_route():
    """An RHF molecule driven as UHF must reproduce the spin-free spectrum.

    The tolerance is 1e-4 eV, not machine precision, because the RHF and UHF
    solutions of the same molecule differ at the SCF convergence threshold --
    a real bug in the unrestricted path shows up orders of magnitude above
    this."""
    atom, _, basis = CASES['h2o']
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    rhf, uhf = scf.RHF(mol).run(), scf.UHF(mol).run()
    for level in ('adc2', 'adc3'):
        er, _ = solve_ee_adc(rhf, mol, level=level, nroots=8, df=True,
                             auxbasis='exact')
        eu, _ = solve_ee_adc(uhf, mol, level=level, nroots=8,
                             route='unrestricted', df=True, auxbasis='exact')
        d = np.abs(np.sort(np.asarray(er))[:6]
                   - np.sort(np.asarray(eu))[:6]).max() * HARTREE_TO_EV
        assert d < 1e-4, f'{level}: {d:.2e} eV'


def test_unrestricted_route_rejects_a_closed_shell_reference():
    atom, _, basis = CASES['h2o']
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    with pytest.raises(ValueError, match='UHF reference'):
        solve_ee_adc(mf, mol, level='adc2', route='unrestricted', df=True)


def test_spinorbital_route_still_refuses_df():
    """The arbiter must stay dense: density fitting an oracle defeats it."""
    mol, mf = _reference('nh2')
    with pytest.raises(ValueError, match='unrestricted'):
        solve_ee_adc(mf, mol, level='adc2', route='spinorbital', df=True)

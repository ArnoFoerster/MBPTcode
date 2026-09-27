"""`calc_qp_energy`'s calling conventions on Ne/cc-pVDZ Hartree-Fock (DF):

  1. the defaults -- GW on an RPA polarizability, density fitting, the HOMO --
     are the HOMO entry of an explicit state list;
  2. a list of self-energies and a list of states return {state: {method:
     energy}}, each entry the energy a single call gives: PSD1 on a BSE
     polarizability, and plain GW, which screens with RPA whatever
     `polarizability` names;
  3. the three 2p orbitals are degenerate, so states 3 and 4 agree for every
     self-energy;
  4. printSpectralFunction=True prints A(omega) and returns the same energy.

Run as a script (`python tests/test_qp_routine.py`) or under pytest.
"""
import functools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import df, gto, scf

from src.SingleReference.GW.qp_energy import calc_qp_energy

#: eV. The same Casida solve reached by different calls, and two orbitals
#: degenerate by symmetry.
SAME_EV = 1e-8
HOMO = 4


@functools.lru_cache(maxsize=None)
def neon():
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    return mf.run()


@functools.lru_cache(maxsize=None)
def listed():
    return calc_qp_energy(neon(), selfenergy=['GW', 'PSD1', 'PSD2'],
                          polarizability='BSE', state=[3, HOMO])


def test_defaults_are_the_homo():
    homo = calc_qp_energy(neon())
    print(f'GW@RPA HOMO: {homo:.6f} eV')
    assert abs(homo - listed()[HOMO]['GW']) < SAME_EV


def test_lists_match_single_calls():
    table = listed()
    print(f'{table}')
    assert sorted(table) == [3, HOMO]
    assert all(sorted(row) == ['GW', 'PSD1', 'PSD2'] for row in table.values())
    psd1 = calc_qp_energy(neon(), selfenergy='PSD1', polarizability='BSE')
    assert abs(psd1 - table[HOMO]['PSD1']) < SAME_EV


def test_degenerate_2p():
    table = listed()
    for method in ('GW', 'PSD1', 'PSD2'):
        assert abs(table[3][method] - table[HOMO][method]) < SAME_EV, method


def test_spectral_function_keeps_the_energy():
    printed = calc_qp_energy(neon(), selfenergy='GW', state=HOMO,
                             printSpectralFunction=True)
    assert abs(printed - listed()[HOMO]['GW']) < SAME_EV


if __name__ == '__main__':
    for test in (test_defaults_are_the_homo, test_lists_match_single_calls,
                 test_degenerate_2p, test_spectral_function_keeps_the_energy):
        test()
        print(f'[OK  ] {test.__name__}')
    print('\nALL PASSED')

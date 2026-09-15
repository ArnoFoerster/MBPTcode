import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf, df

from src.Base.constants import get_method_info
from src.Base.pyscf_interface import (get_orbital_energies,
                                      get_density_fitting_coefficients)
from src.SingleReference.GW.qp_energy import (_casida_spectrum, _self_energy_amplitudes,
                                              _two_electron_integrals)
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver


def check(ok, label, detail=''):
    msg = f"  [{'ok' if ok else 'FAIL'}] {label}"
    print(msg + (f'   ({detail})' if detail else ''))
    return bool(ok)


if __name__ == '__main__':
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    eps = get_orbital_energies(mf, representation='spatial')
    nocc = mol.nelectron // 2
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
    se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')
    w_aux = lr.static_screening_aux(nocc)
    methods = ['GW', 'GWGammaInf', 'PSD1', 'PSD2', 'PSD4', 'PSD5', 'PSD6', 'PSD7',
              'PSD8', 'PSD9']
    infos = {m: get_method_info(m) for m in methods}
    spectrum = _casida_spectrum(lr, nocc, 'BSE', w_aux, False, infos, methods,
                                False, True)

    all_ok = True
    grid = np.linspace(-1.5, 0.5, 37)
    for p in (nocc - 1, nocc, 0):
        amps = _self_energy_amplitudes(se, nocc, spectrum, infos, methods, 'alpha', p,
                                       w_aux, w_aux, False, True)
        for m in methods:
            om, chi_a, chi_b, om_t, chi_b_t = amps[m]
            mode = infos[m]['vertex_mode']
            for calc_imag in (False, True):
                ref = se.calculate_self_energy(p, grid, nocc, om, chi_a, chi_b,
                                               eigenvalues_casida_t=om_t,
                                               chiXYb_t=chi_b_t,
                                               spin_channel='alpha', vertex_mode=mode,
                                               calc_imag=calc_imag)
                sigma = se.self_energy_evaluator(p, nocc, om, chi_a, chi_b,
                                                 eigenvalues_casida_t=om_t,
                                                 chiXYb_t=chi_b_t,
                                                 spin_channel='alpha', vertex_mode=mode,
                                                 calc_imag=calc_imag)
                d_grid = np.max(np.abs(sigma(grid) - ref))
                pairs = zip(grid[::9], ref[::9])
                d_scalar = max(abs(sigma(float(w)) - r) for w, r in pairs)
                tag = f'p={p} {m} {"Im" if calc_imag else "Re"}'
                all_ok &= check(d_grid < 1e-12, f'{tag}: grid parity',
                                f'{d_grid:.1e}')
                all_ok &= check(d_scalar < 1e-12, f'{tag}: scalar parity',
                                f'{d_scalar:.1e}')
                is_float = isinstance(sigma(0.1), float)
                all_ok &= check(is_float, f'{tag}: scalar returns float')

    # Unrestricted section: OH radical (doublet), UHF with density fitting,
    # exercising self_energy_evaluator's unrestricted-only mask, prefactor and
    # two-triplet-channel branches against calculate_self_energy.
    mol_u = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    mf_u = scf.UHF(mol_u).density_fit()
    mf_u.with_df.auxbasis = df.make_auxbasis(mol_u)
    mf_u.run()
    nocc_u = mf_u.nelec
    eps_u = get_orbital_energies(mf_u, representation='spatial')
    df_u, eri_u = _two_electron_integrals(mol_u, mf_u, True, True)
    lr_u = LinearResponseSolver(eps_u, coeff_df=df_u, eri_chemist=eri_u,
                                spin_mode='unrestricted')
    se_u = SelfEnergySolver(eps_u, df_coeff=df_u, eri_chemist=eri_u,
                            spin_mode='unrestricted')
    w_aux_u = lr_u.static_screening_aux(nocc_u)
    spectrum_u = _casida_spectrum(lr_u, nocc_u, 'BSE', w_aux_u, False, infos,
                                  methods, True, True)

    # Near-pole grids of other beta states reach |Sigma| ~ 1e4 Ha on this grid,
    # where a 1e-12 Ha absolute gate is below double-precision resolution.
    states_u = {'alpha': (nocc_u[0] - 1, nocc_u[0], 0), 'beta': (0, nocc_u[1] - 1)}
    for spin in ('alpha', 'beta'):
        for p in states_u[spin]:
            amps_u = _self_energy_amplitudes(se_u, nocc_u, spectrum_u, infos,
                                             methods, spin, p, w_aux_u, w_aux_u,
                                             True, True)
            for m in methods:
                om, chi_a, chi_b, om_t, chi_b_t = amps_u[m]
                mode = infos[m]['vertex_mode']
                for calc_imag in (False, True):
                    ref = se_u.calculate_self_energy(p, grid, nocc_u, om, chi_a,
                                                     chi_b,
                                                     eigenvalues_casida_t=om_t,
                                                     chiXYb_t=chi_b_t,
                                                     spin_channel=spin,
                                                     vertex_mode=mode,
                                                     calc_imag=calc_imag)
                    sigma = se_u.self_energy_evaluator(p, nocc_u, om, chi_a,
                                                       chi_b,
                                                       eigenvalues_casida_t=om_t,
                                                       chiXYb_t=chi_b_t,
                                                       spin_channel=spin,
                                                       vertex_mode=mode,
                                                       calc_imag=calc_imag)
                    d_grid = np.max(np.abs(sigma(grid) - ref))
                    pairs = zip(grid[::9], ref[::9])
                    d_scalar = max(abs(sigma(float(w)) - r) for w, r in pairs)
                    tag = f'UHF {spin} p={p} {m} {"Im" if calc_imag else "Re"}'
                    all_ok &= check(d_grid < 1e-12, f'{tag}: grid parity',
                                    f'{d_grid:.1e}')
                    all_ok &= check(d_scalar < 1e-12, f'{tag}: scalar parity',
                                    f'{d_scalar:.1e}')
                    is_float = isinstance(sigma(0.1), float)
                    all_ok &= check(is_float, f'{tag}: scalar returns float')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

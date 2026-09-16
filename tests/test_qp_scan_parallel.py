import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Solvers.qp_equation import solve_qp_equation


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" +
          (f'   ({detail})' if detail else ''))
    return bool(ok)


class Counted:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, w):
        self.calls += 1
        return self.f(w)


if __name__ == '__main__':
    all_ok = True
    sigma = lambda w: 0.05 / (w + 0.8) + 0.01 / (w - 0.9)      # array-safe
    f = lambda w: w + 0.2 - sigma(w)
    for method in ('pole_strength', 'graphical'):
        s = Counted(f)
        v = Counted(f)
        r_s = solve_qp_equation(s, -0.2, method=method)
        r_v = solve_qp_equation(v, -0.2, method=method, vectorized=True)
        all_ok &= check(r_s == r_v, f'{method}: same root, scalar vs vectorized grid',
                         f'{r_v:.12f}')
        all_ok &= check(v.calls < s.calls - 100, f'{method}: grid took one call',
                         f'{v.calls} vs {s.calls}')

    bad = lambda w: np.atleast_2d(f(w))
    for method in ('pole_strength', 'graphical'):
        try:
            solve_qp_equation(bad, -0.2, method=method, vectorized=True)
            raised = False
        except ValueError:
            raised = True
        all_ok &= check(raised, f'{method}: bad grid shape raises ValueError')

    from pyscf import gto, scf, dft, df
    from src.Base.constants import HARTREE_TO_EV, get_method_info
    from src.Base.pyscf_interface import (get_orbital_energies,
                                          get_density_fitting_coefficients)
    from src.SingleReference.GW.qp_energy import (
        calc_qp_energy, _casida_spectrum, _self_energy_amplitudes,
        qp_energies_from_spectrum, _static_correction)
    from src.SingleReference.GW.self_energy import SelfEnergySolver
    from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

    def head_algorithm(mf, states, xc):
        """Reference per-state loop: scalar Sigma per frequency, built from
        the unchanged private helpers."""
        mol = mf.mol
        eps = get_orbital_energies(mf, representation='spatial')
        nocc = mol.nelectron // 2
        coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted')
        se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted')
        w_aux = lr.static_screening_aux(nocc)
        infos = {'GW': get_method_info('GW')}
        spectrum = _casida_spectrum(lr, nocc, 'RPA', w_aux, False, infos,
                                    ['GW'], False, True)
        out = {}
        for i, p in enumerate(states):
            om, chi_a, _, _, _ = _self_energy_amplitudes(
                se, nocc, spectrum, infos, ['GW'], 'alpha', p, w_aux, w_aux,
                False, True)['GW']
            func = lambda w, p=p, om=om, chi_a=chi_a, x=xc[i]: (
                w - eps[p] - x - se.calculate_self_energy(
                    p, w, nocc, om, chi_a, None, spin_channel='alpha',
                    vertex_mode='GW'))
            out[p] = solve_qp_equation(func, eps[p],
                                       method='pole_strength') * HARTREE_TO_EV
        return out

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g', verbose=0)
    refs = (('RHF', scf.RHF(mol).density_fit()),
           ('PBE', dft.RKS(mol, xc='PBE').density_fit()))
    for label, mf in refs:
        mf.with_df.auxbasis = df.make_auxbasis(mol)
        mf.run()
        norb = mf.mo_coeff.shape[1]
        states = list(range(norb))
        serial = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                                state=states, n_workers=1)
        pooled = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                                state=states, n_workers=4)
        d_pool = max(abs(serial[p]['GW'] - pooled[p]['GW']) for p in states)
        all_ok &= check(d_pool < 1e-10,
                        f'{label}: pool (4) vs serial, all {norb} states',
                        f'{d_pool:.1e} eV')
        df_coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
        se_ = SelfEnergySolver(
            get_orbital_energies(mf, representation='spatial'),
            df_coeff=df_coeff, spin_mode='restricted')
        xc = _static_correction(mf, mol, se_, None, None, 'alpha', False)[states]
        ref = head_algorithm(mf, states, xc)
        d_head = max(abs(serial[p]['GW'] - ref[p]) for p in states)
        all_ok &= check(d_head < 1e-6,
                        f'{label}: helper vs the scalar per-frequency loop',
                        f'{d_head:.1e} eV')
        if label == 'PBE':
            # the per-state static correction formula, inline
            dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
            v_hxc_mo = mf.mo_coeff.T @ mf.get_veff(mol, dm) @ mf.mo_coeff
            v_hx_mo = se_.calculate_sigma_hx(mol, scf.RHF(mol), dm, mf.mo_coeff)
            per_state = np.array([v_hx_mo[p, p] - v_hxc_mo[p, p] for p in states])
            d_xc = np.max(np.abs(xc - per_state))
            all_ok &= check(d_xc < 1e-12,
                            'PBE: _static_correction equals the per-state formula',
                            f'{d_xc:.1e} Ha')
            # the correction is one array over the orbitals, read at p, so a
            # reordered subset of states gets each state's own value
            eps_pbe = get_orbital_energies(mf, representation='spatial')
            nocc = mol.nelectron // 2
            lr_ = LinearResponseSolver(eps_pbe, coeff_df=df_coeff,
                                       spin_mode='restricted')
            w_aux_ = lr_.static_screening_aux(nocc)
            infos = {'GW': get_method_info('GW')}
            spectrum_ = _casida_spectrum(lr_, nocc, 'RPA', w_aux_, False, infos,
                                         ['GW'], False, True)
            xc_orb = _static_correction(mf, mol, se_, None, None, 'alpha', False)
            subset = [nocc, 2, nocc - 1]
            try:
                sub = qp_energies_from_spectrum(
                    se_, nocc, spectrum_, infos, ['GW'], 'alpha', subset,
                    w_aux_, w_aux_, False, True, eps_pbe, xc_orb, n_workers=1)
                d_sub = max(abs(sub[p]['GW'] * HARTREE_TO_EV - serial[p]['GW'])
                            for p in subset)
                detail = f'{d_sub:.1e} eV'
            except Exception as exc:
                d_sub, detail = np.inf, f'{type(exc).__name__}: {exc}'
            all_ok &= check(d_sub < 1e-10,
                            f'PBE: states {subset} read xc at their orbital index',
                            detail)
            try:
                qp_energies_from_spectrum(
                    se_, nocc, spectrum_, infos, ['GW'], 'alpha', subset,
                    w_aux_, w_aux_, False, True, eps_pbe, xc_orb[subset],
                    n_workers=1)
                raised = False
            except ValueError:
                raised = True
            all_ok &= check(raised, 'PBE: xc shorter than the orbitals raises '
                                    'ValueError')
    # a single state and a scalar return keep the old shape
    e_homo = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', state='homo')
    all_ok &= check(isinstance(e_homo, float), "state='homo' returns a float")
    # an empty state window: no states to scan, no result to return
    e_empty = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', state=[])
    all_ok &= check(e_empty == {}, "state=[] returns {}")

    # --- threadpoolctl missing: default n_workers warns and runs serially;
    # an explicit n_workers > 1 still raises ImportError ---
    saved_threadpoolctl = sys.modules.get('threadpoolctl')
    sys.modules['threadpoolctl'] = None
    try:
        mol_tp = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g', verbose=0)
        mf_tp = scf.RHF(mol_tp).density_fit()
        mf_tp.with_df.auxbasis = df.make_auxbasis(mol_tp)
        mf_tp.run()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            e_default = calc_qp_energy(mf_tp, selfenergy='GW',
                                       polarizability='RPA', state=[0, 1, 2])
        e_serial = calc_qp_energy(mf_tp, selfenergy='GW', polarizability='RPA',
                                  state=[0, 1, 2], n_workers=1)
        all_ok &= check(e_default == e_serial,
                        'no threadpoolctl: default n_workers matches n_workers=1')
        all_ok &= check(any('threadpoolctl' in str(w.message) for w in caught),
                        'no threadpoolctl: default n_workers warns')
        try:
            calc_qp_energy(mf_tp, selfenergy='GW', polarizability='RPA',
                           state=[0, 1, 2], n_workers=4)
            raised = False
        except ImportError:
            raised = True
        all_ok &= check(raised,
                        'no threadpoolctl: explicit n_workers=4 raises ImportError')
    finally:
        if saved_threadpoolctl is not None:
            sys.modules['threadpoolctl'] = saved_threadpoolctl
        else:
            sys.modules.pop('threadpoolctl', None)

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

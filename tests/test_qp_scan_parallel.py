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

    # --- worker count: the keyword, else OMP_NUM_THREADS, else the affinity
    # mask; never above the mask (with a warning) or the state count; Slurm's
    # variables are not read ---
    from src.SingleReference.GW.qp_energy import _resolve_workers
    saved_env = {k: os.environ.get(k) for k in ('SLURM_CPUS_PER_TASK',
                                                 'OMP_NUM_THREADS')}
    saved_mask = getattr(os, 'sched_getaffinity', None)
    os.sched_getaffinity = lambda pid: set(range(12))
    cases = [  # (keyword, SLURM_CPUS_PER_TASK, OMP_NUM_THREADS, n_states,
               #  expected, warns)
        (3, None, '8', 100, 3, False),
        (None, None, None, 100, 12, False),
        (None, None, '4', 100, 4, False),
        (None, None, '8,4', 100, 12, False),
        (None, '6', None, 100, 12, False),
        (None, None, '6', 3, 3, False),
        (None, None, '16', 100, 12, True),
        (20, None, None, 100, 12, True),
    ]
    try:
        for keyword, slurm, omp, n_states, expected, warns in cases:
            for name, value in (('SLURM_CPUS_PER_TASK', slurm),
                                ('OMP_NUM_THREADS', omp)):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                got = _resolve_workers(keyword, n_states)
            warned = any('OMP_PROC_BIND' in str(w.message) for w in caught)
            all_ok &= check(
                got == expected and warned == warns,
                f'workers: keyword={keyword} SLURM_CPUS_PER_TASK={slurm} '
                f'OMP_NUM_THREADS={omp} mask=12 states={n_states} -> {expected}'
                f'{", warns" if warns else ""}',
                f'got {got}, warned={warned}')
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if saved_mask is None:
            del os.sched_getaffinity
        else:
            os.sched_getaffinity = saved_mask

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
    # evGW through the front door: every cycle's Casida scan gets n_workers
    import src.SingleReference.GW.qp_energy as qp_module
    seen = []
    real_scan = qp_module.qp_energies_from_spectrum

    def recording_scan(*args, **kwargs):
        seen.append(kwargs.get('n_workers'))
        return real_scan(*args, **kwargs)
    qp_module.qp_energies_from_spectrum = recording_scan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            calc_qp_energy(mf, mode='casida', self_consistency='evGW',
                           n_workers=3, max_cycle=2)
    finally:
        qp_module.qp_energies_from_spectrum = real_scan
    all_ok &= check(len(seen) > 0 and set(seen) == {3},
                    "evGW: n_workers=3 reaches every cycle's scan",
                    f'{len(seen)} scans, n_workers seen {sorted(map(str, set(seen)))}')

    # --- threadpoolctl is a hard dependency, bound when the module loads ---
    all_ok &= check(getattr(qp_module, 'threadpool_limits', None) is not None,
                    'threadpoolctl: imported at module load, no serial fallback')

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

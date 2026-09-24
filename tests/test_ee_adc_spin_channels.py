"""EE-ADC spin channels: the singlet and triplet solves converge and agree.

solve_ee_adc(..., spin='singlet' | 'triplet') solves one eigenspace of the
alpha<->beta flip F on the spin-free route. On water / cc-pVDZ at ADC(2),
ADC(2)-x and ADC(3), for each channel:

  * every returned root is converged, ||A z - e z|| < 1e-6 with A the route's
    own operator, and the solve raises no RuntimeWarning;
  * its energies are those of the unprojected solve (spin=None) split by the
    parity <z|F z>, and the singlets those of pyscf's RADC, to 1e-6 eV, with
    pyscf's own convergence flag read from its log;
  * return_parity gives the spin=None roots that same parity;
  * a spin=None solve from randomised seeds finds the same roots, so the
    unit-vector guess the references share skipped no symmetry block.

The channel checks repeat at ADC(1), with the 1s frozen, density fitted and EN
dressed; a channel smaller than nroots must warn, and return_parity off the
spin-free route must raise. The two channel bases must split the vector space
orthonormally at every level, checked without a molecule.

And davidson itself must warn when it stops with a root above tol. The parity
reference is computed here from spin_flip_vector, not through return_parity.

Run: python tests/test_ee_adc_spin_channels.py
"""
import io
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import adc, gto, scf
from pyscf.lib import logger

from src.Base.pyscf_interface import (DFIntegrals, get_orbital_energies,
                                      get_two_electron_integrals_chemist)
from src.SingleReference.ADC.eeADC import ee_r_sigma, ee_r_sigma_df
from src.SingleReference.ADC.eeADC.ee_driver import solve_ee_adc
from src.Solvers.davidson import davidson

HARTREE_TO_EV = 27.211386245988
LEVELS = [('adc2', 'adc(2)'), ('adc2x', 'adc(2)-x'), ('adc3', 'adc(3)')]
NROOTS = 3
NREF = 8        # roots of the spin=None and pyscf references, above NROOTS so
                # neither channel runs short and pyscf's guess skips nothing


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def build():
    """Water / cc-pVDZ and its RHF, converged to 1e-10."""
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mol, mf


def operator(mol, mf, level, frozen=0, df=False, en_dress=None):
    """The spin-free route's operator, as _solve_spin_free builds it."""
    eps = get_orbital_energies(mf, representation='spatial')[frozen:]
    no = mol.nelectron // 2 - frozen
    act = slice(frozen, None)
    if df:
        B = DFIntegrals.from_scf(mol, mf.density_fit()).B_aa[:, act, act]
        aop, diag, _ = ee_r_sigma_df.build_operator(eps, B, no, level=level,
                                                    en_dress=en_dress)
    else:
        V = get_two_electron_integrals_chemist(
            mol, mf, representation='spatial')[act, act, act, act]
        aop, diag, _ = ee_r_sigma.build_operator(eps, V.transpose(0, 2, 1, 3), no,
                                                 level=level, en_dress=en_dress)
    return aop, diag, no, len(eps) - no


def solve(mf, **kw):
    """solve_ee_adc with its RuntimeWarnings recorded: (result, warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = solve_ee_adc(mf, **kw)
    return out, [w for w in caught if issubclass(w.category, RuntimeWarning)]


def residuals(aop, e, Z):
    """||A z_k - e_k z_k|| for every column z_k of Z."""
    return np.array([np.linalg.norm(np.asarray(aop(Z[:, k])).ravel() - e[k] * Z[:, k])
                     for k in range(Z.shape[1])])


def channels(mol, mf, level, **kw):
    """The spin=None parity split and both channel solves at one setting.

    Returns (ok, singlet energies, spin=None parities)."""
    aop, diag, no, nv = operator(mol, mf, level, kw.get('frozen', 0),
                                 kw.get('df', False), kw.get('en_dress'))
    ok = check(np.abs(diag - ee_r_sigma.spin_flip_vector(diag, no, nv, level)).max()
               < 1e-12, 'the diagonal is flip-symmetric, so one entry per pair '
               'preconditions the channel')

    (e_all, Z_all), warned = solve(mf, level=level, nroots=NREF, **kw)
    parity = np.array([Z_all[:, k] @ ee_r_sigma.spin_flip_vector(Z_all[:, k], no,
                                                                  nv, level)
                       for k in range(NREF)])
    r = residuals(aop, e_all, Z_all)
    ok &= check(not warned and r.max() < 1e-6,
                'spin=None reference converged', f'max residual {r.max():.1e}')
    split = {'singlet': np.sort(e_all[parity > 0.99]),
             'triplet': np.sort(e_all[parity < -0.99])}

    singlets = None
    for spin in ('singlet', 'triplet'):
        (e, Z), warned = solve(mf, level=level, nroots=NROOTS, spin=spin, **kw)
        r = residuals(aop, e, Z)
        ok &= check(r.max() < 1e-6, f'{spin}: every root converged',
                    f'residuals {", ".join(f"{x:.1e}" for x in r)}')
        ok &= check(not warned, f'{spin}: no RuntimeWarning',
                    f'{len(warned)} raised' if warned else '')
        ref = split[spin][:NROOTS]
        d = (np.abs(np.sort(e) - ref).max() * HARTREE_TO_EV
             if len(ref) == NROOTS else np.inf)
        ok &= check(d < 1e-6, f'{spin}: the spin=None roots of that parity',
                    f'max |dE| {d:.1e} eV')
        if spin == 'singlet':
            singlets = np.sort(e)
    return ok, singlets, parity, np.sort(e_all)


def random_guess(mol, mf, level):
    """The spin=None roots from seeds with a random admixture: every symmetry
    block carries weight from the start, which unit-vector seeds do not."""
    aop, diag, _, _ = operator(mol, mf, level)
    want = 2 * NREF + 4
    V = np.zeros((len(diag), want))
    V[np.argsort(diag)[:want], np.arange(want)] = 1.0
    V += 1e-2 * np.random.default_rng(3).standard_normal(V.shape)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        e, _ = davidson(aop, diag, k=NREF, v0=V)
    return np.sort(e), [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_level(mol, mf, level, method):
    """Both channels at one ADC level against spin=None and pyscf."""
    ok, singlets, parity, e_all = channels(mol, mf, level)
    e_rand, warned = random_guess(mol, mf, level)
    d = np.abs(e_rand - e_all).max() * HARTREE_TO_EV
    ok &= check(not warned and d < 1e-6, 'spin=None: a randomised guess finds '
                'the same roots, so no symmetry block was skipped',
                f'max |dE| {d:.1e} eV')
    a = adc.RADC(mf)
    a.method, a.method_type = method, 'ee'
    # pyscf's defaults leave ~1e-6 eV at ADC(2)-x; this setting its own flag
    # calls converged (1e-12 / 1e-8 it does not), and it says so only in its log
    a.conv_tol, a.tol_residual, a.max_cycle = 1e-10, 1e-7, 500
    log = io.StringIO()
    a.verbose, a.stdout = logger.WARN, log
    ref = np.sort(np.asarray(a.kernel(nroots=NREF)[0]))[:NROOTS]
    ok &= check('did not converge' not in log.getvalue(),
                "pyscf's RADC reference converged by its own flag")
    d = np.abs(singlets - ref).max() * HARTREE_TO_EV
    ok &= check(d < 1e-6, "singlet: pyscf's RADC", f'max |dE| {d:.1e} eV')
    try:
        (_, _, got), _ = solve(mf, level=level, nroots=NREF, return_parity=True)
        d = np.abs(np.asarray(got) - parity).max()
        ok &= check(d < 1e-8, 'return_parity labels the spin=None roots',
                    f'max |dp| {d:.1e}')
    except TypeError as exc:
        ok &= check(False, 'return_parity labels the spin=None roots', str(exc))
    return ok


def test_variants(mol, mf):
    """The channel basis at ADC(1), with a frozen core, DF and EN dressing."""
    ok = True
    for label, level, kw in (('adc1, singles only', 'adc1', {}),
                             ('adc2, 1s frozen', 'adc2', {'frozen': 1}),
                             ('adc2, density fitted', 'adc2', {'df': True}),
                             ('adc2, EN dressed', 'adc2', {'en_dress': True})):
        print(f'  --- {label} ---')
        ok &= channels(mol, mf, level, **kw)[0]
    h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    (e, _), warned = solve(scf.RHF(h2).run(), level='adc1', nroots=3,
                           spin='singlet')
    ok &= check(len(e) == 1 and any('fewer than nroots' in str(w.message)
                                    for w in warned),
                'a channel smaller than nroots warns and returns what it holds',
                f'{len(e)} root(s), {len(warned)} warning(s)')
    try:
        solve_ee_adc(mf, level='adc2', nroots=1, route='spinorbital',
                     return_parity=True)
        raised = 'no error'
    except ValueError:
        raised = ''
    except TypeError as exc:
        raised = str(exc)
    return ok & check(not raised, 'return_parity off the spin-free route raises '
                      'ValueError', raised)


def test_channel_basis():
    """Both channel bases, molecule-free: each orthonormal and inside its flip
    eigenspace, restrict the adjoint of embed, together a basis of the space."""
    try:
        from src.SingleReference.ADC.eeADC.ee_driver import _channel_basis
    except ImportError as exc:
        return check(False, 'the channel bases exist', str(exc))
    ok = True
    rng = np.random.default_rng(5)
    for level in ('adc1', 'adc2', 'adc3'):
        err, count = 0.0, True
        for no, nv in ((1, 1), (2, 3), (3, 4)):
            n = ee_r_sigma.dimensions(no, nv, level)['nH']
            cols = []
            for sgn in (+1.0, -1.0):
                embed, restrict, reps = _channel_basis(n, no, nv, level, sgn)
                U = np.column_stack([embed(c) for c in np.eye(len(reps))])
                FU = np.column_stack([ee_r_sigma.spin_flip_vector(u, no, nv, level)
                                      for u in U.T])
                v = rng.standard_normal(n)
                err = max(err, np.abs(FU - sgn * U).max(),
                          np.abs(restrict(v) - U.T @ v).max())
                cols.append(U)
            Q = np.column_stack(cols)
            count &= Q.shape == (n, n)
            err = max(err, np.abs(Q.T @ Q - np.eye(n)).max() if count else np.inf)
        ok &= check(count and err < 1e-14, f'{level}: singlet and triplet bases '
                    'split the space orthonormally', f'max error {err:.1e}')
    return ok


def test_davidson_warns():
    """A root left above tol is named in a RuntimeWarning; a converged run is silent."""
    # a Hartree-scale diagonal, as an excitation spectrum
    rng = np.random.default_rng(7)
    M = 1e-3 * rng.standard_normal((300, 300))
    A = np.diag(np.linspace(0.2, 2.0, 300)) + M + M.T
    runs = {}
    for max_iter in (2, 200):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            davidson(lambda v: A @ v, np.diag(A).copy(), k=2, max_iter=max_iter)
        runs[max_iter] = [str(w.message) for w in caught
                          if issubclass(w.category, RuntimeWarning)]
    ok = check(len(runs[2]) == 1 and 'still above tol' in runs[2][0],
               'max_iter=2 warns', runs[2][0] if runs[2] else 'no warning')
    return ok & check(not runs[200], 'max_iter=200 converges without one')


if __name__ == '__main__':
    mol, mf = build()
    all_ok = True
    for level, method in LEVELS:
        print(f'\n=== water / cc-pVDZ, {level} ===')
        all_ok &= test_level(mol, mf, level, method)
    print('\n=== water / cc-pVDZ, variants ===')
    all_ok &= test_variants(mol, mf)
    print('\n=== channel bases ===')
    all_ok &= test_channel_basis()
    print('\n=== davidson ===')
    all_ok &= test_davidson_warns()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)

"""Exciton descriptors from a full one-particle transition density matrix
(LinearResponse.exciton_descriptors.exciton_descriptors_tdm).

Water, 6-31G. Checks:

  1. an (X, Y) RPA vector written as a 1TDM (ov = X, vo = Y^T, oo = vv = 0)
     gives every descriptor of `exciton_descriptors` to 1e-12; with the sign
     of Y flipped it does not.
  2. the NTO quantities: sum_i lambda_i = Omega, Z_HE >= PR_NTO, one NTO pair
     gives S_H|E = 0 and Z_HE = PR_NTO = 1 at Omega = 0.9, two equal pairs
     S_H|E = 1 and Z_HE = 2 (a Bell pair), the raw entropy is
     Omega S - Omega log2 Omega, and gamma -> c gamma changes only Omega and
     the raw entropy; the raw entropy of one pair at Omega = 0.9 is not 0.
  3. the ISR-ADC(2) ground-to-excited 1TDM of ee_transition_density at
     sqrt(2) times its alpha block equals pyscf's: radc_ee's spin-free
     moments for singlets (up to the root's sign) and uadc_ee's alpha block
     for Ms = 0 triplets, every block to 1e-7; the alpha + beta charge or spin
     density itself does not.
  4. on the RI fold (ee_ri_transition) at sqrt(2) times the alpha block, two
     singlets and two triplets: the singles block alone has Omega = T1, every
     root holds 0.95 <= (1 - T1) + Omega <= 1 and Z_HE >= PR_NTO; the density
     without the sqrt(2) / 2 scaling breaks the upper bound.
  5. a 1TDM on the active columns of mo_coeff gives the descriptors of the
     same 1TDM padded with zeros on all columns; the wrong columns do not; a
     zero 1TDM raises ValueError.

Run: python tests/test_exciton_descriptors_tdm.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import adc, gto, scf

from src.Base.pyscf.pyscf_interface import get_two_electron_integrals_chemist
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
from src.SingleReference.ADC.eeADC import ee_ri_transition as ert
from src.SingleReference.ADC.eeADC import ee_transition_density as etd
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from src.SingleReference.LinearResponse.exciton_descriptors import (
    exciton_descriptors, exciton_descriptors_tdm)
from tests.adc_common import AUX, NF, WATER_C2V, WATER_ORIGIN, check, rhf_df

PARITY = {'singlet': +1, 'triplet': -1}
NTO_KEYS = ('nto_weights', 'pr_nto', 's_he', 's_he_raw', 'z_he')


def rhf(atom, basis):
    """Converged RHF of `atom` in `basis`, to 1e-13 Ha and a gradient of 1e-9."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def xy_as_tdm(X, Y, nocc, nmo):
    """(nroots, nmo, nmo) 1TDM with ov = X, vo = Y^T of (nocc*nvirt, nroots)."""
    nroots = X.shape[1]
    g = np.zeros((nroots, nmo, nmo))
    g[:, :nocc, nocc:] = X.T.reshape(nroots, nocc, -1)
    g[:, nocc:, :nocc] = Y.T.reshape(nroots, nocc, -1).transpose(0, 2, 1)
    return g


def max_diff(a, b, keys):
    """Largest |a[k] - b[k]| over the entries named in `keys` of two result dicts."""
    return max(float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).max()) for k in keys)


def check_xy_reduction():
    """Check 1."""
    print('\n=== 1. an (X, Y) vector as a 1TDM gives the (X, Y) descriptors ===')
    mf = rhf(WATER_ORIGIN, '6-31g')
    mol = mf.mol
    nocc, nmo = mol.nelectron // 2, mf.mo_coeff.shape[1]
    td = mf.TDHF().run(nstates=4)
    X = np.stack([np.sqrt(2.0) * x.ravel() for x, _ in td.xy], axis=1)
    Y = np.stack([np.sqrt(2.0) * y.ravel() for _, y in td.xy], axis=1)
    want = exciton_descriptors(mf, mol, nocc, X, Y)
    got = exciton_descriptors_tdm(mol, mf.mo_coeff, xy_as_tdm(X, Y, nocc, nmo))
    err = max_diff(want, got, want.keys())
    ok = check(err < 1e-12, f'every one of the {len(want)} descriptors equal',
               f'max |d| {err:.1e}')
    flip = exciton_descriptors_tdm(mol, mf.mo_coeff, xy_as_tdm(X, -Y, nocc, nmo))
    off = float(np.abs(flip['R_eh'] - want['R_eh']).max())
    ok &= check(off > 1e-6, 'with Y of the other sign R_eh differs',
                f'max |d| {off:.1e}')
    return ok


def check_nto():
    """Check 2."""
    print('\n=== 2. NTO weights, PR_NTO and S_H|E ===')
    ok = True
    mol = gto.M(atom=WATER_ORIGIN, basis='sto-3g', verbose=0)
    C = np.eye(mol.nao)                     # moments are unused below
    rng = np.random.default_rng(7)
    g = rng.standard_normal((5, mol.nao, mol.nao))
    d = exciton_descriptors_tdm(mol, C, g)
    lam_sum = np.abs(d['nto_weights'].sum(axis=1) - d['c_n']).max()
    ok &= check(lam_sum < 1e-10 * d['c_n'].max(), 'sum_i lambda_i = Omega',
                f'max |d| {lam_sum:.1e}')
    ok &= check(np.all(d['z_he'] >= d['pr_nto'] - 1e-12), 'Z_HE >= PR_NTO',
                f'min Z - PR {np.min(d["z_he"] - d["pr_nto"]):.2e}')
    raw = d['c_n'] * d['s_he'] - d['c_n'] * np.log2(d['c_n'])
    err = float(np.abs(raw - d['s_he_raw']).max())
    ok &= check(err < 1e-10, 'S_raw = Omega S - Omega log2 Omega', f'{err:.1e}')
    c = 0.3
    dc = exciton_descriptors_tdm(mol, C, c * g)
    keys = [k for k in d if k not in ('c_n', 'nto_weights', 's_he_raw')]
    err = max_diff(d, dc, keys)
    ok &= check(err < 1e-10, f'gamma -> {c} gamma leaves {len(keys)} descriptors',
                f'max |d| {err:.1e}')
    one = np.zeros((1, mol.nao, mol.nao))
    one[0, 2, 5] = np.sqrt(0.9)
    d1 = exciton_descriptors_tdm(mol, C, one)
    ok &= check(abs(d1['s_he'][0]) < 1e-14 and abs(d1['z_he'][0] - 1.0) < 1e-14
                and abs(d1['pr_nto'][0] - 1.0) < 1e-14,
                'one NTO pair at Omega 0.9: S_H|E = 0, Z_HE = PR_NTO = 1')
    ok &= check(d1['s_he_raw'][0] > 0.1, 'its raw entropy is not 0',
                f'{d1["s_he_raw"][0]:.3f} bit')
    bell = np.zeros((1, mol.nao, mol.nao))
    bell[0, 2, 5] = bell[0, 3, 6] = np.sqrt(0.5)
    db = exciton_descriptors_tdm(mol, C, bell)
    ok &= check(abs(db['s_he'][0] - 1.0) < 1e-14 and abs(db['z_he'][0] - 2.0) < 1e-13,
                'two equal pairs: S_H|E = 1, Z_HE = 2',
                f'S {db["s_he"][0]:.15f}, Z {db["z_he"][0]:.15f}')
    return ok


def check_pyscf():
    """Check 3."""
    print('\n=== 3. sqrt(2) gamma^alpha of ISR-ADC(2) equals pyscf ===')
    ok = True
    mf = rhf(WATER_ORIGIN, '6-31g')
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    nmo = len(eps)
    V = get_two_electron_integrals_chemist(mol, mf, representation='spatial'
                                           ).transpose(0, 2, 1, 3)
    a = adc.ADC(mf)
    a.method, a.method_type, a.conv_tol, a.tol_residual = 'adc(2)', 'ee', 1e-12, 1e-10
    a.approx_trans_moments = False
    ep, _, _, _ = a.kernel(nroots=3)
    ty = np.asarray(a._adc_es.get_trans_moments()).reshape(-1, nmo, nmo)
    ua = adc.ADC(scf.addons.convert_to_uhf(mf))
    ua.method, ua.method_type, ua.conv_tol, ua.tol_residual = ('adc(2)', 'ee', 1e-12,
                                                               1e-10)
    ua.approx_trans_moments = False
    eu, _, _, _ = ua.kernel(nroots=6)
    ty_a, ty_b = (np.asarray(t).reshape(-1, nmo, nmo)
                  for t in ua._adc_es.get_trans_moments())
    for spin in PARITY:
        e, Z = solve_ee_adc(mf, level='adc2', spin=spin, nroots=2, conv_tol=1e-10)
        err, unscaled = 0.0, np.inf
        for k in range(2):
            c, s = etd.spin_free_ground_transition_density(eps, V, no, Z[:, k], 'adc2',
                                                           PARITY[spin])
            dens = c if spin == 'singlet' else s
            g = dens / np.sqrt(2.0)                  # sqrt(2) gamma^alpha
            if spin == 'singlet':
                ref = ty[int(np.argmin(np.abs(ep - e[k])))]
            else:
                j = int(np.argmin(np.abs(eu - e[k])))
                ref = np.sqrt(2.0) * ty_a[j]
                err = max(err, float(np.abs(ty_a[j] + ty_b[j]).max()))
            sign = np.sign(np.vdot(ref, g))
            err = max(err, float(np.abs(ref - sign * g).max()))
            unscaled = min(unscaled, float(np.abs(ref - sign * dens).max()))
        ok &= check(err < 1e-7, f'{spin}s: every block equals pyscf'
                    + (' radc_ee' if spin == 'singlet' else ' uadc_ee, beta = -alpha'),
                    f'max |d| {err:.1e}')
        ok &= check(unscaled > 1e-3, f'{spin}s: the alpha + beta density does not',
                    f'min max |d| {unscaled:.1e}')
    return ok


def check_fold():
    """Check 4."""
    print('\n=== 4. RI fold: Omega against the singles weight T1 ===')
    ok = True
    mf = rhf_df(WATER_C2V, '6-31g')
    mol = mf.mol
    f = ee_ri_fold.bare_factors(mol, mf, frozen=NF, auxbasis=AUX)
    roots = {}
    for spin in ee_ri_fold.SPINS:
        op = ee_ri_fold.RIFoldADC2(f, spin)
        r = ee_fold.solve_folded(op.pieces(), 2, spin=spin, tol_omega=1e-11,
                                 tol_residual=1e-9)
        for k in range(2):
            roots[spin, k] = (op, np.asarray(r.y)[:, k], float(r.omega[k]),
                              float(r.t1[k]))
    keys = list(roots)
    full = ert.transition_densities(roots, grounds=keys)
    parts = ert.transition_densities(roots, grounds=keys, terms=True)
    C = np.asarray(mf.mo_coeff)[:, NF:]
    no, nv = f.no, f.nv
    singles, bounds, zpr, unscaled = 0.0, [], True, []
    for key in keys:
        which = 0 if key[0] == 'singlet' else 1
        t1 = roots[key][3]
        g = full[ert.GROUND, key][which] / np.sqrt(2.0)
        s = etd.assemble({'ov_s': parts[ert.GROUND, key][which]['ov_s']}, no, nv)
        singles = max(singles, abs(float((s**2).sum()) / 2.0 - t1))
        d = exciton_descriptors_tdm(mol, C, g[None])
        bounds.append((1.0 - t1) + float(d['c_n'][0]))
        zpr &= bool(d['z_he'][0] >= d['pr_nto'][0])
        unscaled.append((1.0 - t1) + 2.0 * float(d['c_n'][0]))
        print(f'    {key[0]} {key[1]}: T1 {t1:.6f}  Omega {d["c_n"][0]:.6f}  '
              f'R_eh {d["R_eh"][0]:.4f}  d_exc {d["d_exc"][0]:.4f}  '
              f'PR_NTO {d["pr_nto"][0]:.4f}  S_H|E {d["s_he"][0]:.4f}  '
              f'Z_HE {d["z_he"][0]:.4f}')
    ok &= check(singles < 1e-12, 'the singles block alone has Omega = T1',
                f'max |d| {singles:.1e}')
    ok &= check(all(0.95 <= b <= 1.0 for b in bounds),
                '0.95 <= (1 - T1) + Omega <= 1 for every root',
                ', '.join(f'{b:.4f}' for b in bounds))
    ok &= check(zpr, 'Z_HE >= PR_NTO for every root')
    ok &= check(all(b > 1.0 for b in unscaled),
                'the unscaled density breaks the upper bound',
                ', '.join(f'{b:.4f}' for b in unscaled))
    return ok


def check_active_and_guard():
    """Check 5."""
    print('\n=== 5. active orbitals and the zero-norm guard ===')
    ok = True
    mf = rhf(WATER_ORIGIN, '6-31g')
    mol, C = mf.mol, mf.mo_coeff
    nmo = C.shape[1]
    g = np.random.default_rng(3).standard_normal((2, nmo - NF, nmo - NF))
    pad = np.zeros((2, nmo, nmo))
    pad[:, NF:, NF:] = g
    act = exciton_descriptors_tdm(mol, C[:, NF:], g)
    full = exciton_descriptors_tdm(mol, C, pad)
    err = max_diff(act, full, [k for k in act if k != 'nto_weights'])
    n = nmo - NF                         # the padded 1TDM adds NF zero weights
    err = max(err, float(np.abs(act['nto_weights'] - full['nto_weights'][:, :n]).max()),
              float(np.abs(full['nto_weights'][:, n:]).max()))
    ok &= check(err < 1e-12, 'active columns equal the zero-padded full 1TDM',
                f'max |d| {err:.1e}')
    wrong = exciton_descriptors_tdm(mol, C[:, :-NF], g)
    off = float(np.abs(wrong['d_exc'] - act['d_exc']).max())
    ok &= check(off > 1e-6, 'the wrong columns move d_exc', f'max |d| {off:.1e}')
    try:
        exciton_descriptors_tdm(mol, C, np.concatenate([pad, np.zeros((1, nmo, nmo))]))
        ok &= check(False, 'a zero 1TDM raises ValueError')
    except ValueError as e:
        ok &= check('zero 1TDM' in str(e), 'a zero 1TDM raises ValueError', str(e))
    return ok


def run():
    """Every check; True when all passed."""
    ok = check_xy_reduction()
    ok &= check_nto()
    ok &= check_pyscf()
    ok &= check_fold()
    ok &= check_active_and_guard()
    return ok


def test_exciton_descriptors_tdm():
    """Every check of run(), under pytest."""
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)

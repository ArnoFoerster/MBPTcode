"""ISR-ADC(2) one-particle transition densities (eeADC/ee_transition_density).

Checks:

  1. the exact intermediate-state representation of the determinant space,
     truncated at the ADC(2) orders (tests/exact_isr_pp_oracle), has the ADC(2)
     secular matrix of the spin-orbital dense route (ee_u_dense_full) as its
     Hamiltonian, element by element to 1e-9 (water and LiH, sto-3g); the
     oracle's doubles basis carries the opposite sign, which maps its
     vectors onto the route's.
  2. on the same eigenvectors, the module's state-to-state densities
     gamma^lr (every ordered pair of the 14 lowest roots: singlets, triplet
     sublevels, l = r) and ground-to-excited densities gamma^0r equal the
     oracle's to 1e-9, the antisymmetric parts included; leaving out any ONE
     term misses the oracle by more than 1e-6 on some pair (each term
     matters).
  3. at l = r the state-to-state density is ee_density's difference density
     to 1e-14.
  4. the spin-blocked route of the dense spin-free solver equals the
     spin-orbital route (the same vectors written out in spin orbitals) to
     1e-13 for singlet-singlet, singlet-triplet, triplet-singlet and
     triplet-triplet pairs and the ground rows (water, 6-31G); between equal
     spins the transition density carries charge only, between a singlet
     and a triplet spin only.
  5. the singlets' oscillator strengths 2/3 omega |<0|r|n>|^2 equal pyscf's
     ISR-ADC(2) ones (radc_ee with its second-order amplitudes, an
     independent code) to 1e-7 (water and formaldehyde, 6-31G); leaving out
     the second-order doubles term vo_t2 misses them by more than 1e-6.
  6. the folded route (ee_ri_fold roots, their doubles rebuilt from the
     singles, ee_transition_density.fold_state) on the DF factors equals
     the dense DF spin-free route on the same factors: the reference
     amplitudes (the DF ladder against the full integrals) to 1e-14, each
     rebuilt vector is a unit vector equal to the dense eigenvector to 1e-8,
     and every transition density (all spin pairs of three roots each, the
     ground rows) to 1e-8, the dense solver's own convergence (water and
     formaldehyde, cc-pVDZ).

Run: python tests/test_ee_adc_transition_density.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import adc, ao2mo, gto, scf

from src.Base.pyscf.pyscf_interface import (DFIntegrals,
                                            get_antisymmetrized_spin_integrals,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.SingleReference.ADC.eeADC import (ee_density, ee_equations, ee_fold,
                                           ee_r_sigma, ee_ri_fold, ee_spin_blocks,
                                           ee_transition_density, ee_u_dense_full,
                                           ee_utils)
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from tests.adc_common import H2CO, WATER_C2V, WATER_ORIGIN, check, ee_numerics
from tests.exact_isr_pp_oracle import build

LIH = 'Li 0 0 0; H 0 0 1.6'
PARITY = {'singlet': +1, 'triplet': -1}
NUMPY = ee_equations.NUMPY
etd = ee_transition_density


def rhf(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def spin_free_transition_density(eps, V, no, vl, vr, level='adc2',
                                 parity_l=None, parity_r=None):
    """`transition_density` of two full eigenvectors of the dense spin-free
    route on the full integrals V (eps, V, level, parity as
    ee_density.spin_free_difference_density)."""
    nv = len(eps) - no
    ref = etd.reference_from_integrals(eps, V, no, level, ground=False)
    return etd.transition_density(ref, etd.flat_state(vl, no, nv, level, parity_l),
                                  etd.flat_state(vr, no, nv, level, parity_r))


def spin_free_ground_transition_density(eps, V, no, v, level='adc2', parity=None,
                                        terms=False):
    """`ground_transition_density` of one full eigenvector of the dense
    spin-free route on the full integrals V."""
    nv = len(eps) - no
    ref = etd.reference_from_integrals(eps, V, no, level, ground=True)
    return etd.ground_transition_density(ref, etd.flat_state(v, no, nv, level,
                                                             parity), terms)


def worst(err):
    """(label, value) of the largest entry of {label: value}."""
    k = max(err, key=err.get)
    return k, err[k]


def check_oracle(atom, nroots=14, basis='sto-3g'):
    """Checks 1-3 on one molecule."""
    ok = True
    o = build(atom, basis)
    no, n, ns = o.nocc, o.norb, o.n_s
    nv = n - no
    M = o.truncated_secular()
    R = ee_u_dense_full.build_supermatrix(o.eps, o.g, no, level='adc2')
    sign = np.ones(len(R))
    sign[ns:] = -1.0
    err = float(np.abs(M - sign[:, None] * R * sign[None, :]).max())
    ok &= check(err < 1e-9, f'{atom.split()[0]}: the truncated exact ISR is the '
                'ADC(2) secular matrix', f'{err:.1e}')
    _, V = np.linalg.eigh(R)
    V = V[:, :nroots]
    amps = ee_utils.mp_amplitudes(o.eps, o.g, no, n, order=3)
    t2, p0 = amps['t2_1'], ee_utils.mp_density2(amps)

    def blocks(v):
        return v[:ns].reshape(no, nv), ee_utils.unfold_doubles(v[ns:], no, n) / 2

    pair_err, ground_err, diag_err = 0.0, 0.0, 0.0
    miss = {name: 0.0 for name in etd.ALL_TERMS}
    gmiss = {name: 0.0 for name in etd.GROUND_TERMS}
    for i in range(nroots):
        l1, l2 = blocks(V[:, i])
        ref = o.ground_transition_density(sign * V[:, i])
        terms = etd.ground_transition_terms(NUMPY, l1, l2, t2, amps['t2_2'],
                                            amps['t1_2'], p0)
        ground_err = max(ground_err, float(np.abs(etd.assemble(terms, no, nv)
                                                  - ref).max()))
        for name in gmiss:
            gmiss[name] = max(gmiss[name], float(np.abs(
                etd.assemble(terms, no, nv, drop=(name,)) - ref).max()))
        diag = ee_density.assemble(ee_density.difference_density_terms(
            NUMPY, l1, l2, t2, p0), no, nv)
        own = etd.assemble(etd.transition_density_terms(NUMPY, l1, l1, l2, l2,
                                                        t2, p0), no, nv)
        diag_err = max(diag_err, float(np.abs(diag - own).max()))
        for j in range(nroots):
            r1, r2 = blocks(V[:, j])
            ref = o.transition_density(sign * V[:, i], sign * V[:, j])
            terms = etd.transition_density_terms(NUMPY, l1, r1, l2, r2, t2, p0)
            pair_err = max(pair_err, float(np.abs(etd.assemble(terms, no, nv)
                                                  - ref).max()))
            for name in miss:
                miss[name] = max(miss[name], float(np.abs(
                    etd.assemble(terms, no, nv, drop=(name,)) - ref).max()))
    label = f'{atom.split()[0]} {basis}'
    ok &= check(pair_err < 1e-9, f'{label}: gamma^lr equals the exact ISR '
                f'({nroots}x{nroots} pairs)', f'{pair_err:.1e}')
    ok &= check(ground_err < 1e-9, f'{label}: gamma^0r equals the exact ISR',
                f'{ground_err:.1e}')
    ok &= check(diag_err < 1e-14, f'{label}: l = r is the ee_density '
                'difference density', f'{diag_err:.1e}')
    for what, m in (('state-to-state', miss), ('ground', gmiss)):
        name, value = min(m.items(), key=lambda kv: kv[1])
        ok &= check(value > 1e-6, f'{label}: every {what} term is needed',
                    f'weakest {name}: {value:.1e}')
    return ok


def spin_orbital_vector(v, parity, no, nv):
    """(u1, u2) of a spin-free flat vector, in interleaved spin orbitals."""
    u1, u2 = ee_r_sigma.to_blocks(v, no, nv, 'adc2', parity=parity)
    NO, N = 2 * no, 2 * (no + nv)
    return (ee_spin_blocks.to_spin_orbital(u1, 'ov', NO, N),
            ee_spin_blocks.to_spin_orbital(u2, 'oovv', NO, N))


def spatial(G, no, nv):
    """(alpha + beta, alpha - beta, largest spin-mixing element) of a
    spin-orbital (occupied first, interleaved) matrix."""
    def idx(s):
        return np.concatenate([np.arange(no) * 2 + s, 2 * no + np.arange(nv) * 2 + s])
    a = G[np.ix_(idx(0), idx(0))]
    b = G[np.ix_(idx(1), idx(1))]
    return a + b, a - b, float(np.abs(G[np.ix_(idx(0), idx(1))]).max())


def check_spin_free():
    """Check 4."""
    ok = True
    mf = rhf(WATER_ORIGIN, '6-31g')
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    nmo = len(eps)
    nv = nmo - no
    V = get_two_electron_integrals_chemist(mol, mf, representation='spatial'
                                           ).transpose(0, 2, 1, 3)
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape((nmo,) * 4)
    _, g = get_antisymmetrized_spin_integrals(
        mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, eri)
    amps = ee_utils.mp_amplitudes(np.repeat(eps, 2), g, 2 * no, 2 * nmo, order=3)
    t2, p0 = amps['t2_1'], ee_utils.mp_density2(amps)
    roots = {s: solve_ee_adc(mf, level='adc2', spin=s, nroots=3, numerics=ee_numerics(conv_tol=1e-10))[1]
             for s in PARITY}
    for sl in PARITY:
        for sr in PARITY:
            err, other = 0.0, 0.0
            for i in range(3):
                for j in range(3):
                    vl, vr = roots[sl][:, i], roots[sr][:, j]
                    c, s = spin_free_transition_density(
                        eps, V, no, vl, vr, 'adc2', PARITY[sl], PARITY[sr])
                    l1, l2 = spin_orbital_vector(vl, PARITY[sl], no, nv)
                    r1, r2 = spin_orbital_vector(vr, PARITY[sr], no, nv)
                    G = etd.assemble(etd.transition_density_terms(
                        NUMPY, l1, r1, l2, r2, t2, p0), 2 * no, 2 * nv)
                    cc, ss, mix = spatial(G, no, nv)
                    err = max(err, float(np.abs(cc - c).max()),
                              float(np.abs(ss - s).max()), mix)
                    other = max(other, float(np.abs(s if sl == sr else c).max()))
            ok &= check(err < 1e-13 and other < 1e-13,
                        f'{sl}-{sr}: spin blocks equal spin orbitals, '
                        + ('charge only' if sl == sr else 'spin only'),
                        f'{err:.1e}, other part {other:.1e}')
    for s in PARITY:
        err, other = 0.0, 0.0
        for k in range(3):
            c, sp = spin_free_ground_transition_density(
                eps, V, no, roots[s][:, k], 'adc2', PARITY[s])
            u1, u2 = spin_orbital_vector(roots[s][:, k], PARITY[s], no, nv)
            G = etd.assemble(etd.ground_transition_terms(
                NUMPY, u1, u2, t2, amps['t2_2'], amps['t1_2'], p0), 2 * no, 2 * nv)
            cc, ss, mix = spatial(G, no, nv)
            err = max(err, float(np.abs(cc - c).max()), float(np.abs(ss - sp).max()),
                      mix)
            other = max(other, float(np.abs(sp if s == 'singlet' else c).max()))
        ok &= check(err < 1e-13 and other < 1e-13,
                    f'ground-{s}: spin blocks equal spin orbitals',
                    f'{err:.1e}, other part {other:.1e}')
    return ok


def check_pyscf(atom, nroots=3):
    """Check 5 on one molecule."""
    ok = True
    mf = rhf(atom, '6-31g')
    mol = mf.mol
    no = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    nv = len(eps) - no
    V = get_two_electron_integrals_chemist(mol, mf, representation='spatial'
                                           ).transpose(0, 2, 1, 3)
    e, Z = solve_ee_adc(mf, level='adc2', spin='singlet', nroots=nroots,
                        numerics=ee_numerics(conv_tol=1e-10))
    a = adc.ADC(mf)
    a.method, a.method_type, a.conv_tol, a.tol_residual = 'adc(2)', 'ee', 1e-12, 1e-10
    a.approx_trans_moments = False
    ep, _, fp, _ = a.kernel(nroots=nroots + 1)
    C = mf.mo_coeff
    r = np.einsum('xmn,mp,nq->xpq', mol.intor('int1e_r'), C, C)
    err, miss = 0.0, 0.0
    for k in range(nroots):
        j = int(np.argmin(np.abs(ep - e[k])))
        charge, _ = spin_free_ground_transition_density(
            eps, V, no, Z[:, k], 'adc2', +1, terms=True)
        for drop, store in (((), 'err'), (('vo_t2',), 'miss')):
            mu = np.einsum('xpq,pq->x', r, etd.assemble(charge, no, nv, drop=drop))
            d = abs(2.0 / 3.0 * e[k] * mu @ mu - fp[j])
            if store == 'err':
                err = max(err, d, abs(ep[j] - e[k]))
            else:
                miss = max(miss, d)
    label = atom.split()[0] + ''.join(a.split()[0] for a in atom.split(';')[1:])
    ok &= check(err < 1e-7, f'{label}: oscillator strengths equal pyscf ISR-ADC(2)',
                f'{err:.1e}')
    ok &= check(miss > 1e-6, f'{label}: without vo_t2 they do not',
                f'{miss:.1e}')
    return ok


def _sb_dot(a, b):
    return float(sum((blk * b.get(k)).sum() for k, blk in a.blocks.items()))


def check_fold(atom, nroots=3):
    """Check 6 on one molecule."""
    ok = True
    mol = gto.M(atom=atom, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    no = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf), float)
    nv = len(eps) - no
    B = DFIntegrals.from_scf(mol, mf).B_aa
    F = ee_ri_fold.RIFactors.from_full(eps, B, no)
    ref_b = etd.reference_from_factors(eps, B, no)
    V = np.einsum('Ppq,Prs->pqrs', B, B, optimize=True).transpose(0, 2, 1, 3)
    ref_v = etd.reference_from_integrals(eps, V, no)
    err = max(float(np.abs(getattr(ref_b, n).get(k) - getattr(ref_v, n).get(k)).max())
              for n in ('t2', 't1_2', 't2_2') for k in getattr(ref_v, n).keys())
    err = max([err] + [float(np.abs(ref_b.p0[b].get(k) - ref_v.p0[b].get(k)).max())
                       for b in ref_v.p0 for k in ref_v.p0[b].keys()])
    label = atom.split()[0] + ''.join(a.split()[0] for a in atom.split(';')[1:])
    ok &= check(err < 1e-14, f'{label}: DF reference amplitudes equal the full '
                'integrals\' ones', f'{err:.1e}')
    states = {}
    for spin in PARITY:
        _, Z = solve_ee_adc(mf, level='adc2', df=True, spin=spin, nroots=nroots,
                            numerics=ee_numerics(conv_tol=1e-10))
        op = ee_ri_fold.RIFoldADC2(F, spin)
        r = ee_fold.solve_folded(op.pieces(), nroots, spin=spin, tol_omega=1e-11,
                                 tol_residual=1e-10, dense_limit=0, solver='newton')
        dense = [etd.flat_state(Z[:, k], no, nv, 'adc2', PARITY[spin])
                 for k in range(nroots)]
        fold = [etd.fold_state(op, r.y[:, k], r.omega[k], r.t1[k])
                for k in range(nroots)]
        # the eigenvectors' arbitrary signs, from the singles
        sign = [np.sign(_sb_dot(f[0], d[0])) for f, d in zip(fold, dense)]
        verr, nerr = 0.0, 0.0
        for f, d, sg in zip(fold, dense, sign):
            nerr = max(nerr, abs(_sb_dot(f[0], f[0]) + _sb_dot(f[1], f[1]) - 1.0))
            for x, y in zip(f, d):
                verr = max(verr, max(float(np.abs(x.get(k) - sg * y.get(k)).max())
                                     for k in y.keys()))
        ok &= check(nerr < 1e-12 and verr < 1e-8, f'{label} {spin}: rebuilt fold '
                    'vectors are the unit dense eigenvectors',
                    f'norm {nerr:.1e}, vectors {verr:.1e}')
        states[spin] = (dense, fold, sign)
    for sl in PARITY:
        for sr in PARITY:
            err = 0.0
            for i in range(nroots):
                for j in range(nroots):
                    dl, fl, gl = (x[i] for x in states[sl])
                    dr, fr, gr = (x[j] for x in states[sr])
                    c1, s1 = etd.transition_density(ref_v, dl, dr)
                    c2, s2 = etd.transition_density(ref_b, fl, fr)
                    err = max(err, float(np.abs(gl * gr * c1 - c2).max()),
                              float(np.abs(gl * gr * s1 - s2).max()))
            ok &= check(err < 1e-8, f'{label} {sl}-{sr}: fold transition densities '
                        'equal the dense DF ones', f'{err:.1e}')
    for spin in PARITY:
        err = 0.0
        for d, f, g in zip(*states[spin]):
            c1, s1 = etd.ground_transition_density(ref_v, d)
            c2, s2 = etd.ground_transition_density(ref_b, f)
            err = max(err, float(np.abs(g * c1 - c2).max()),
                      float(np.abs(g * s1 - s2).max()))
        ok &= check(err < 1e-8, f'{label} ground-{spin}: fold equals dense DF',
                    f'{err:.1e}')
    return ok


def run():
    """Every check; True when all passed."""
    all_ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        all_ok &= check_oracle(WATER_ORIGIN)
        all_ok &= check_oracle(LIH)
        all_ok &= check_spin_free()
        all_ok &= check_pyscf(WATER_ORIGIN)
        all_ok &= check_pyscf(H2CO)
        all_ok &= check_fold(WATER_C2V)
        all_ok &= check_fold(H2CO)
    return all_ok


def test_ee_adc_transition_density():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)

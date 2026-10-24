"""Gates for the non-Dyson IP/EA-ADC(n) construction (nondyson_isr.NonDysonISR
over the generated adc{N}_nd{m}_k{i}k{j} block modules).

  1. External reference: pyscf's non-Dyson IP- and EA-ADC(2), ADC(2)-x and
     ADC(3) (Banerjee and Sokolov, J. Chem. Phys. 151, 224112 (2019)) on
     LiH, H2O and BeH2 in sto-3g: every pyscf root (principal and satellite)
     has a partner in the spin-orbital spectrum to the Davidson tolerance.
  2. The union-space oracle's SectorISR (tests/union_space_oracle.py, the same
     construction in the determinant space) on
     linear H4 and LiH, n = 2..5: every effective block (class-GS + Loewdin
     orthonormalized, per pair and order) equals the oracle's after the
     configuration map (IP class k: (-1)^(k(k+1)/2) from the descending hole
     annihilators of the precursor string, EA class k: (-1)^(k(k-1)/2)); the
     full spectra at lambda = 1 agree; the Rayleigh-Schroedinger series of the
     Koopmans-connected eigenvalue equals the exact E(N-+1) - E0(N) series
     through order n (the Taylor pin; the fit-free proof of order-by-order
     exactness).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf, adc

from src.SingleReference.ADC.nondyson_isr import NonDysonISR
from src.SingleReference.ADC.precursor_classes import class_configs
from src.Base.pyscf.pyscf_interface import (get_orbital_energies,
                                            get_antisymmetrized_spin_integrals)
from tests.generated_runtime import requires_runtime
from tests import union_space_oracle as CB

pytestmark = requires_runtime(5)

PYSCF_SYSTEMS = [('LiH', 'Li 0 0 0; H 0 0 1.6'),
                 ('H2O', 'O 0 0 0; H 0 0 0.96; H 0.96 0 0'),
                 ('BeH2', 'Be 0 0 0; H 0 0 1.3; H 0 0 -1.3')]
PYSCF_METHODS = [('adc(2)', 2, None), ('adc(2)-x', 2, {(1, 1): 1}), ('adc(3)', 3, None)]
NROOTS = 6
TOL_PYSCF = 1e-8          # pyscf's Davidson (conv_tol 1e-14 on the residual) sets this floor
ORACLE_SYSTEMS = ['H4lin_r1.0', 'LiH']
NMAX = 4     # n = 5 needs the MP5 amplitude numerators (refused, nondyson_isr)
TOL_BLOCK = 1e-11
TOL_SPECTRUM = 1e-10
TOL_PIN = 1e-9
_CACHE = {}


def _spinorb(atom):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.run()
    n = mf.mo_coeff.shape[1]
    from pyscf import ao2mo
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n, n, n, n)
    h1_mo = mf.mo_coeff.T @ (mol.intor('int1e_kin') + mol.intor('int1e_nuc')) @ mf.mo_coeff
    _, g = get_antisymmetrized_spin_integrals(h1_mo, eri)
    eps = get_orbital_energies(mf, representation='spin')
    return mf, eps, g, mol.nelectron


@pytest.mark.parametrize('label,atom', PYSCF_SYSTEMS)
@pytest.mark.parametrize('sector', ['IP', 'EA'])
def test_matches_pyscf(label, atom, sector):
    mf, eps, g, nocc = _spinorb(atom)
    nd = NonDysonISR(eps, g, nocc, sector)
    worst = 0.0
    for method, n, extra in PYSCF_METHODS:
        w, _, _ = nd.eigenpairs(n, extra=extra)
        a = adc.ADC(mf)
        a.method, a.method_type, a.verbose = method, sector.lower(), 0
        a.conv_tol, a.max_cycle, a.max_space = 1e-14, 300, 60
        e = np.sort(np.atleast_1d(a.kernel(nroots=NROOTS)[0]))
        # pyscf's doublet-adapted roots are a subset of the spin-orbital
        # spectrum (which also carries the high-spin 2h1p/2p1h roots that
        # cannot couple to the 1h/1p class), so every pyscf root must have a
        # production partner; the roots are matched, not indexed
        d = float(max(np.min(np.abs(w - ei)) for ei in e))
        worst = max(worst, d)
        print(f"{label} {sector} {method}: pyscf {np.round(e, 9)}; nearest production root "
              f"within {d:.1e}")
    assert worst <= TOL_PYSCF, worst


def _distinct(w, tol=1e-6):
    """Sorted distinct values of a spectrum (degenerate roots merged)."""
    out = []
    for x in np.sort(np.asarray(w, dtype=float)):
        if not out or abs(x - out[-1]) > tol:
            out.append(float(x))
    return np.array(out)


def _config_map(us, nd_oracle, nocc, norb, sector, k):
    """Oracle positions of the production configs of class k, and the sign
    relating the production precursor string to the oracle's operator."""
    pos = {tuple(nd_oracle.labels[p]): p for p in np.where(nd_oracle.kcls == k)[0]}
    cols = [np.asarray(c) for c in class_configs(nocc, norb, k, sector)]
    nh = k + 1 if sector == 'IP' else k
    holes, parts = cols[:nh], cols[nh:]
    if k == 0:
        labs = [(int(c),) for c in cols[0]]
    elif sector == 'IP':
        labs = [tuple(int(p[c]) for p in parts) + tuple(int(h[c]) for h in holes)
                for c in range(len(cols[0]))]
    else:
        labs = [tuple(int(h[c]) for h in holes) + tuple(int(p[c]) for p in parts)
                for c in range(len(cols[0]))]
    sign = (-1.0) ** (k * (k + 1) // 2) if sector == 'IP' else (-1.0) ** (k * (k - 1) // 2)
    return np.array([pos[l] for l in labs]), sign


def _oracle_data(name):
    if name in _CACHE:
        return _CACHE[name]
    sysd = CB.build_system(name)
    us = CB.UnionSpace(sysd)
    nord = NMAX + 1
    psi, E, nser = us.rspt_series(nord)
    d = dict(sysd=sysd, us=us, psi=psi, E=E, nser=nser, nord=nord, oracle={}, prod={})
    for sector in ('IP', 'EA'):
        d['oracle'][sector] = CB.SectorISR(us, psi, E, nser, nord, sector)
        d['prod'][sector] = NonDysonISR(sysd['eps'], sysd['g'], sysd['nocc'], sector)
    _CACHE[name] = d
    return d


@pytest.mark.parametrize('name', ORACLE_SYSTEMS)
@pytest.mark.parametrize('sector', ['IP', 'EA'])
def test_effective_blocks_match_oracle(name, sector):
    d = _oracle_data(name)
    orc, prod, sysd = d['oracle'][sector], d['prod'][sector], d['sysd']
    nocc, norb = sysd['nocc'], sysd['norb']
    blocks, dims = prod.effective_blocks(NMAX)
    maps = {k: _config_map(d['us'], orc, nocc, norb, sector, k) for k in dims}
    worst, scale = 0.0, 0.0
    for (k_i, k_j), ser in blocks.items():
        rows, s_i = maps[k_i]
        cols, s_j = maps[k_j]
        for m, A in ser.items():
            if m > NMAX - k_i - k_j:
                continue
            ref = s_i * s_j * orc.M[m][np.ix_(rows, cols)]
            diff = float(np.max(np.abs(A - ref))) if A.size else 0.0
            worst = max(worst, diff)
            scale = max(scale, float(np.max(np.abs(ref))) if ref.size else 0.0)
    print(f"{name} {sector}: effective blocks vs oracle SectorISR, orders through rule at n={NMAX}: "
          f"max|diff| {worst:.2e} (|blocks| up to {scale:.2e}, f_ov {sysd['f_ov']:.1e})")
    assert worst <= TOL_BLOCK, worst


@pytest.mark.parametrize('name', ORACLE_SYSTEMS)
@pytest.mark.parametrize('sector', ['IP', 'EA'])
def test_spectra_and_taylor_pin_match_oracle(name, sector):
    d = _oracle_data(name)
    orc, prod, us = d['oracle'][sector], d['prod'][sector], d['us']
    nocc = us.nocc
    no_sp = nocc // 2
    probes = [no_sp - 1, no_sp - 2] if sector == 'IP' else [nocc]
    probes = [p for p in probes if p >= 0]
    worst_w, worst_pin = 0.0, 0.0
    for n in range(2, NMAX + 1):
        Mser, keep = orc.truncated(n)
        Msum = sum(Mser)
        w_ref = np.linalg.eigvalsh(0.5 * (Msum + Msum.T))
        M_prod, dims, offs = prod.matrix_series(n)
        Ms = sum(M_prod)
        w = np.linalg.eigvalsh(0.5 * (Ms + Ms.T))
        assert w.shape == w_ref.shape, (n, w.shape, w_ref.shape)
        dw = float(np.max(np.abs(w - w_ref)))
        worst_w = max(worst_w, dw)
        cfg0 = prod.configs(0)[0]
        for p in probes:
            exact = CB.sector_state_series(us, sector, p, d['E'], d['nord'])
            idx = int(np.where(cfg0 == (p if sector == 'IP' else p - nocc))[0][0]) + offs[0]
            series = CB.eigenvalue_series(M_prod, idx, n)
            dpin = float(np.max(np.abs(series[:n + 1] - exact[:n + 1])))
            worst_pin = max(worst_pin, dpin)
        print(f"{name} {sector} ADC({n}): spectrum vs oracle {dw:.2e}; Taylor pin through "
              f"order {n} (Koopmans-connected eigenvalue vs exact E(N-+1)-E0) {worst_pin:.2e}; "
              f"lowest root {w[0]:.9f}")
    assert worst_w <= TOL_SPECTRUM, worst_w
    assert worst_pin <= TOL_PIN, worst_pin


def main():
    for label, atom in PYSCF_SYSTEMS:
        for sector in ('IP', 'EA'):
            test_matches_pyscf(label, atom, sector)
    for name in ORACLE_SYSTEMS:
        for sector in ('IP', 'EA'):
            test_effective_blocks_match_oracle(name, sector)
            test_spectra_and_taylor_pin_match_oracle(name, sector)
    print('\nALL OK')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)

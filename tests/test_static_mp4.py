"""The order-consistent static self-energy of ADC(5): G[Delta_gamma] from the
MP4 1-RDM (static_correction.build_mp4_static_correction[_restricted]).

The orbital block of ADC(n) carries Sigma(infinity) through order n, and
G[Delta_gamma^(m)] is of order m+1, so ADC(3) needs the MP2 density, ADC(4)
the MP3 one and ADC(5) the MP4 one.

Oracle: the exact ground-state 1-RDM gamma(lambda) of H(lambda) = F +
lambda (H - F) in the full determinant space (exact_oracle.ExactSelfEnergy),
its lambda-Taylor coefficients gamma_n extracted on a complex-lambda circle
(FFT, radius-independence as the self-gate). Delta_gamma^(n) is gamma_n, so
Sigma(inf) through order 5 is <pr||qs> (gamma_2 + gamma_3 + gamma_4)_rs.

Gates, on linear H4 and LiH (sto-3g): the MP4 correction equals that series
and the MP3 one equals the series through order 4; the MP3 correction misses
the order-5 piece by far more than the tolerance (the reason the MP4 builder
exists); the restricted and spin-orbital builders agree; and
build_static_correction(kind='mp4_unrelaxed') dispatches to the builder.
The spin-orbital driver's own Delta_gamma^(4) (the only consumer of the rank-4
overlap and m4_*_22 pieces of generated_mpn/mpn_density_pieces.py) is gated
against gamma_4 directly.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.pyscf_interface import (get_antisymmetrized_spin_integrals,
                                      get_orbital_energies)
from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.SingleReference.DensityMatrix.mpn_density_driver import MPnDensityDriver
from src.SingleReference.ADC import (build_mp3_static_correction,
                                     build_mp4_static_correction,
                                     build_mp4_static_correction_restricted,
                                     build_static_correction,
                                     static_kind_for_level)

SYSTEMS = {'H4lin_r1.0': '; '.join(f'H 0 0 {i * 1.0}' for i in range(4)),
           'LiH': 'Li 0 0 0; H 0 0 1.6'}
TOL = 1e-11
RADII = (0.3, 0.2)       # the second radius is the contour self-gate
N_POINTS = 32


def _system(atom):
    """Canonical RHF converged to a residual f[o,v] well below TOL: the
    MP-density identity holds for canonical HF only."""
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.run()
    return mol, mf


def _density_series(mol, mf, orders):
    """{n: gamma_n} (spin-orbital, occupied-first [occ_a, occ_b, vir_a,
    vir_b]) and the antisymmetrized g of the same layout, from the exact
    N-electron ground state on a complex-lambda circle; returns the largest
    deviation between the two radii as well."""
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    es = EO.ExactSelfEnergy(eps, h1, g, nocc)
    h_full = EO._sector_h(es.dets_n, h1, g)
    f_diag = np.diag([sum(eps[p] for p in d) for d in es.dets_n])

    def coefficients(radius):
        lams = radius * np.exp(2j * np.pi * np.arange(N_POINTS) / N_POINTS)
        vals = []
        for lam in lams:
            # gamma() reads only the N-electron sector; fill the oracle's
            # per-lambda cache with it instead of building N-1 and N+1 too
            es._cache[lam] = ((1 - lam) * f_diag + lam * h_full, None, None)
            vals.append(es.gamma(lam))
        coefs = np.fft.fft(np.array(vals), axis=0) / N_POINTS
        return {n: coefs[n] / radius ** n for n in orders}

    first, second = (coefficients(r) for r in RADII)
    drift = max(float(np.max(np.abs(first[n] - second[n]))) for n in orders)
    return {n: first[n].real for n in orders}, g, nocc, drift


def _to_occupied_first(M, nocc, norb):
    """A spin-orbital matrix from the builders' interleaved convention
    (spatial p -> 2p alpha, 2p+1 beta) into the oracle's occupied-first
    blocks."""
    no_sp, nv_sp = nocc // 2, (norb - nocc) // 2
    perm = ([2 * p for p in range(no_sp)] + [2 * p + 1 for p in range(no_sp)]
            + [2 * (no_sp + a) for a in range(nv_sp)]
            + [2 * (no_sp + a) + 1 for a in range(nv_sp)])
    return M[np.ix_(perm, perm)]


@pytest.mark.parametrize('name', list(SYSTEMS))
def test_mp4_static_matches_the_exact_series_through_order_five(name):
    mol, mf = _system(SYSTEMS[name])
    gam, g, nocc, drift = _density_series(mol, mf, (1, 2, 3, 4))
    norb = g.shape[0]
    print(f"{name}: contour radius drift {drift:.2e}, |gamma_1| "
          f"{float(np.max(np.abs(gam[1]))):.1e} (zero for canonical HF)")
    assert drift <= TOL, drift

    ref4 = np.einsum('prqs,rs->pq', g, gam[2] + gam[3])
    ref5 = ref4 + np.einsum('prqs,rs->pq', g, gam[4])
    mp4 = _to_occupied_first(build_mp4_static_correction(mf, mol), nocc, norb)
    mp3 = _to_occupied_first(build_mp3_static_correction(mf, mol), nocc, norb)
    d5 = float(np.max(np.abs(mp4 - ref5)))
    d4 = float(np.max(np.abs(mp3 - ref4)))
    short = float(np.max(np.abs(mp3 - ref5)))
    print(f"{name}: MP4 static vs exact Sigma_inf through order 5: {d5:.2e}; "
          f"MP3 static vs order 4: {d4:.2e}; the order-5 piece MP3 misses: {short:.2e} "
          f"(|Sigma_inf| {float(np.max(np.abs(ref5))):.3e})")
    assert d4 <= TOL, d4
    assert d5 <= TOL, d5
    assert short > 1e3 * TOL, short

    r = build_mp4_static_correction_restricted(mf, mol)
    mp4_il = build_mp4_static_correction(mf, mol)
    d_r = float(np.max(np.abs(r - mp4_il[0::2, 0::2])))
    print(f"{name}: restricted MP4 static vs spin-orbital alpha block: {d_r:.2e}")
    assert d_r <= TOL, d_r

    kind = build_static_correction(mf, mol, kind='mp4_unrelaxed', spin='restricted')
    d_k = float(np.max(np.abs(kind - r)))
    print(f"{name}: build_static_correction(kind='mp4_unrelaxed') vs the builder: {d_k:.2e}")
    assert d_k <= TOL, d_k


@pytest.mark.parametrize('name', list(SYSTEMS))
def test_spin_orbital_mp4_density_matches_the_exact_series(name):
    mol, mf = _system(SYSTEMS[name])
    gam, _, nocc, _ = _density_series(mol, mf, (4,))
    from pyscf import ao2mo
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n, n, n, n)
    _, g_il = get_antisymmetrized_spin_integrals(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, eri)
    oo, ov, vv = MPnDensityDriver(get_orbital_energies(mf, representation='spin'),
                                  g_il, nocc).compute_delta_gamma4()
    # the driver's interleaved spin orbitals in the oracle's occupied-first layout
    no_sp, nv_sp = nocc // 2, n - nocc // 2
    perm = [(k % 2) * no_sp + k // 2 if k < nocc
            else nocc + (k % 2) * nv_sp + (k // 2 - no_sp) for k in range(2 * n)]
    ref = gam[4][np.ix_(perm, perm)]
    d = max(float(np.max(np.abs(oo - ref[:nocc, :nocc]))),
            float(np.max(np.abs(ov - ref[:nocc, nocc:]))),
            float(np.max(np.abs(vv - ref[nocc:, nocc:]))))
    print(f"{name}: spin-orbital Delta_gamma^(4) vs exact gamma_4: {d:.2e} "
          f"(|gamma_4| {float(np.max(np.abs(ref))):.1e})")
    assert d <= TOL, d


def test_mp4_kind_refusals():
    """The MP4 kind has no DF route, no EN hook and no UHF density: each is
    refused instead of computing something else."""
    mol, mf = _system(SYSTEMS['H4lin_r1.0'])
    with pytest.raises(NotImplementedError, match='no DF route'):
        build_static_correction(mf, mol, kind='mp4_unrelaxed',
                                B_aa=np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match='en_dress'):
        build_static_correction(mf, mol, kind='mp4_unrelaxed',
                                en_dress={'hh': True})
    with pytest.raises(NotImplementedError, match='RHF-only'):
        build_static_correction(scf.addons.convert_to_uhf(mf), mol,
                                kind='mp4_unrelaxed')


def test_static_kind_for_level():
    assert static_kind_for_level('adc2x') == 'mp2_unrelaxed'
    assert static_kind_for_level('adc3') == 'mp2_unrelaxed'
    assert static_kind_for_level('adc4') == 'mp3_unrelaxed'
    assert static_kind_for_level('adc5') == 'mp4_unrelaxed'

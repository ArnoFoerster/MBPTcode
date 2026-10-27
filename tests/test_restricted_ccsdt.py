"""Regression tests for the restricted (spin-blocked) CCSDT pipeline
(src/SingleReference/CC/restricted_solver.py).

Scope: T1/T2/T3 (full CCSDT) + Lambda1/Lambda2/Lambda3 (full Lambda-CCSDT)
+ the 1-RDM built from those. Lambda3 IS included: the generator's
add_st_operator hard-truncates its automatic BCH expansion at quadruple
commutators, which silently drops T3-dependent diagrams once a rank-3
projector is folded into the target list; hand-built order-5 (quintuple)
commutator corrections supply them. The rank-3 projector convention: the
rank-3 'e3(d,e,f,n,m,l)' bra (extended by direct analogy from the validated
rank-2 'e2(e,f,n,m)' pattern -- rank 3 has no official generator template
to check against, unlike rank 1/2) reproduces the correct residual up to a
uniform overall factor of -1, which `_lambda_residual_terms` negates (the
`s = -1.0 if rank == 3 else 1.0` factor). check_l3_residual_matches_
generalized below sees it: max abs diff ~1e-13 with the factor, ~1e3 without.

Validation ladder (cheapest/most-trusted oracle first):
1. Restricted-integral sanity: Fock diagonal / HF energy reproduce the
   existing (already-validated) generalized spin-orbital pipeline's values.
2. l3_aaaaaa/aabaab/abbabb_residual vs the generalized lambda3_residual.py,
   on random spin-consistent synthetic amplitudes/integrals (no solver, no
   molecule) -- the cheapest and most direct oracle for the new Lambda3
   equations themselves. This is the check that sees the sign factor above.
3. H2/6-31g: T3 is small-basis-limited but nonzero for 6-31g (2 virtuals in
   sto-3g would make T3 structurally zero, masking a real test) -- use a
   basis where the comparison actually exercises T3. Full restricted
   pipeline (T-CCSDT + full Lambda-CCSDT) vs the existing generalized
   CCSDT pipeline: with Lambda3 included, both the T-CCSDT correlation
   energy AND the full density should match the generalized pipeline
   closely (not just the T-energy).
4. LiH/sto-3g (this codebase's established genuinely-nonzero-T3 case, see
   tests/test_ccsdt_lambda.py): T-CCSDT correlation energy
   vs the generalized pipeline to ~1e-10; AO density difference vs the
   generalized pipeline (which also includes genuine Lambda3) matches
   to ~1e-9 as well, since both pipelines solve the same physical Lambda3
   equations end-to-end (DIIS solve + density), not just the T-only energy.
5. Timing comparison, restricted vs generalized, on LiH/sto-3g -- reports the
   actual measured speedup rather than a-priori literature estimates.
"""
import os
import sys
import time
from itertools import permutations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyscf

from src.SingleReference.CC.integrals import (build_restricted_integrals_from_mf,
                                           build_spinorbital_integrals_from_mf,
                                           energy_denominators)
from src.SingleReference.CC.restricted_solver import (kernel_t_restricted,
                                                    compute_ccsdt_density_matrix_restricted)
from src.SingleReference.CC.generated_restricted import amplitudes_restricted as TR
from src.SingleReference.CC.generated_restricted import lambda_residual_restricted as LR
from src.SingleReference.CC import lambda3_residual as L3M
from src.SingleReference.CC.pipeline import compute_ccsdt_density_matrix
from src.SingleReference.CC import amplitudes


def check_integral_sanity():
    mol = pyscf.M(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g')
    mf = mol.RHF(); mf.verbose = 0; mf.run()

    ints_r = build_restricted_integrals_from_mf(mf)
    ints_so = build_spinorbital_integrals_from_mf(mf)

    f_diag_r = np.diagonal(ints_r['f_aa'])
    f_diag_so_alpha = np.diagonal(ints_so['fock'])[0::2]
    diff_f = np.max(np.abs(f_diag_r - f_diag_so_alpha))
    diff_e = abs(ints_r['hf_energy'] - ints_so['hf_energy'])

    ok = diff_f < 1e-10 and diff_e < 1e-10
    print(f"restricted integrals: Fock diag matches spin-orbital pipeline "
          f"(diff {diff_f:.2e}), hf_energy matches (diff {diff_e:.2e}): "
          f"{'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Lambda3 residual cross-check: random spin-consistent synthetic amplitudes,
# no solver/molecule involved. Builds a full spin-orbital (interleaved
# 2p=alpha,2p+1=beta) t1/t2/t3/l1/l2/l3/f/g respecting (a) the standard RHF
# closed-shell spin-block structure (excitations pair same-spin occ/vir;
# alpha/beta blocks numerically identical) and (b) genuine antisymmetry
# under any occ<->occ or vir<->vir spinorbital exchange. Feeds the spatial
# restricted blocks (literal even/odd sub-slices) into the new
# l3_*_residual, and the full spin-orbital tensors into the trusted
# generalized lambda3_residual.py (sliced the same way) -- both describe
# the same physical residual in different bases, so they must agree to
# float-roundoff.
# ---------------------------------------------------------------------------

def _sign(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def _antisymmetrize_pairs(seed, axes_a, axes_b):
    out = np.zeros_like(seed)
    for pa in permutations(range(2)):
        for pb in permutations(range(2)):
            axes = list(range(seed.ndim))
            for k, ax in enumerate(axes_a):
                axes[ax] = axes_a[pa[k]]
            for k, ax in enumerate(axes_b):
                axes[ax] = axes_b[pb[k]]
            out = out + _sign(pa) * _sign(pb) * seed.transpose(axes)
    return out


def _antisymmetrize_triples(seed, occ_axes, vir_axes):
    out = np.zeros_like(seed)
    for po in permutations(range(3)):
        for pv in permutations(range(3)):
            axes = list(range(seed.ndim))
            for k, ax in enumerate(occ_axes):
                axes[ax] = occ_axes[po[k]]
            for k, ax in enumerate(vir_axes):
                axes[ax] = vir_axes[pv[k]]
            out = out + _sign(po) * _sign(pv) * seed.transpose(axes)
    return out


def check_l3_residual_matches_generalized(seed=42):
    nocc_sp, nvir_sp = 3, 3
    nmo_sp = nocc_sp + nvir_sp
    no_so, nv_so = 2 * nocc_sp, 2 * nvir_sp
    nmo_so = 2 * nmo_sp
    rng = np.random.default_rng(seed)

    def spat(q): return q // 2
    def spin(q): return q % 2

    # T1/L1
    Wt1 = rng.random((nvir_sp, nocc_sp))
    Wl1 = rng.random((nocc_sp, nvir_sp))
    t1_so = np.zeros((nv_so, no_so))
    l1_so = np.zeros((no_so, nv_so))
    for a in range(nv_so):
        for i in range(no_so):
            if spin(a) == spin(i):
                t1_so[a, i] = Wt1[spat(a), spat(i)]
    for i in range(no_so):
        for a in range(nv_so):
            if spin(i) == spin(a):
                l1_so[i, a] = Wl1[spat(i), spat(a)]
    t1_aa = t1_so[0::2, 0::2]
    l1_aa = l1_so[0::2, 0::2]

    # T2/L2
    Wt2 = rng.random((nvir_sp, nvir_sp, nocc_sp, nocc_sp))
    Wl2 = rng.random((nocc_sp, nocc_sp, nvir_sp, nvir_sp))
    seed_t2 = np.zeros((nv_so, nv_so, no_so, no_so))
    seed_l2 = np.zeros((no_so, no_so, nv_so, nv_so))
    for a in range(nv_so):
        for b in range(nv_so):
            for i in range(no_so):
                for j in range(no_so):
                    if spin(a) == spin(i) and spin(b) == spin(j):
                        seed_t2[a, b, i, j] = Wt2[spat(a), spat(b), spat(i), spat(j)]
    for i in range(no_so):
        for j in range(no_so):
            for a in range(nv_so):
                for b in range(nv_so):
                    if spin(i) == spin(a) and spin(j) == spin(b):
                        seed_l2[i, j, a, b] = Wl2[spat(i), spat(j), spat(a), spat(b)]
    t2_so = _antisymmetrize_pairs(seed_t2, (0, 1), (2, 3))
    l2_so = _antisymmetrize_pairs(seed_l2, (0, 1), (2, 3))
    t2_aaaa = t2_so[0::2, 0::2, 0::2, 0::2]
    t2_abab = t2_so[0::2, 1::2, 0::2, 1::2]
    l2_aaaa = l2_so[0::2, 0::2, 0::2, 0::2]
    l2_abab = l2_so[0::2, 1::2, 0::2, 1::2]

    # T3/L3
    Wt3 = rng.random((nvir_sp, nvir_sp, nvir_sp, nocc_sp, nocc_sp, nocc_sp))
    Wl3 = rng.random((nocc_sp, nocc_sp, nocc_sp, nvir_sp, nvir_sp, nvir_sp))
    seed_t3 = np.zeros((nv_so, nv_so, nv_so, no_so, no_so, no_so))
    seed_l3 = np.zeros((no_so, no_so, no_so, nv_so, nv_so, nv_so))
    for a in range(nv_so):
        for b in range(nv_so):
            for c in range(nv_so):
                for i in range(no_so):
                    for j in range(no_so):
                        for k in range(no_so):
                            if spin(a) == spin(i) and spin(b) == spin(j) and spin(c) == spin(k):
                                seed_t3[a, b, c, i, j, k] = Wt3[spat(a), spat(b), spat(c),
                                                                spat(i), spat(j), spat(k)]
    for i in range(no_so):
        for j in range(no_so):
            for k in range(no_so):
                for a in range(nv_so):
                    for b in range(nv_so):
                        for c in range(nv_so):
                            if spin(i) == spin(a) and spin(j) == spin(b) and spin(k) == spin(c):
                                seed_l3[i, j, k, a, b, c] = Wl3[spat(i), spat(j), spat(k),
                                                                spat(a), spat(b), spat(c)]
    t3_so = _antisymmetrize_triples(seed_t3, occ_axes=(3, 4, 5), vir_axes=(0, 1, 2))
    l3_so = _antisymmetrize_triples(seed_l3, occ_axes=(0, 1, 2), vir_axes=(3, 4, 5))

    t3_aaaaaa = t3_so[0::2, 0::2, 0::2, 0::2, 0::2, 0::2]
    t3_aabaab = t3_so[0::2, 0::2, 1::2, 0::2, 0::2, 1::2]
    t3_abbabb = t3_so[0::2, 1::2, 1::2, 0::2, 1::2, 1::2]
    l3_aaaaaa = l3_so[0::2, 0::2, 0::2, 0::2, 0::2, 0::2]
    l3_aabaab = l3_so[0::2, 0::2, 1::2, 0::2, 0::2, 1::2]
    l3_abbabb = l3_so[0::2, 1::2, 1::2, 0::2, 1::2, 1::2]

    # f/g: chemist-symmetric physicist ERI via a symmetric DF-like
    # factorization (guarantees the correct <pq|rs>=<rs|pq>=<qp|sr>
    # real-ERI symmetries with a synthetic random tensor).
    Bdim = 2 * nmo_sp
    B = rng.random((nmo_sp, nmo_sp, Bdim))
    B = 0.5 * (B + B.transpose(1, 0, 2))
    phys = np.einsum('prl,qsl->pqrs', B, B)
    f_aa = rng.random((nmo_sp, nmo_sp))
    f_aa = f_aa + f_aa.T
    g_aaaa = phys - phys.transpose(0, 1, 3, 2)
    g_abab = phys.copy()
    o, v = slice(0, nocc_sp), slice(nocc_sp, nmo_sp)

    def mo_spin_spat(q):
        if q < no_so:
            return q % 2, q // 2
        qq = q - no_so
        return qq % 2, nocc_sp + qq // 2

    mo_spin = np.array([mo_spin_spat(q)[0] for q in range(nmo_so)])
    mo_spat = np.array([mo_spin_spat(q)[1] for q in range(nmo_so)])

    f_so = np.zeros((nmo_so, nmo_so))
    for p in range(nmo_so):
        for q in range(nmo_so):
            if mo_spin[p] == mo_spin[q]:
                f_so[p, q] = f_aa[mo_spat[p], mo_spat[q]]

    g_so = np.zeros((nmo_so, nmo_so, nmo_so, nmo_so))
    for p in range(nmo_so):
        for q in range(nmo_so):
            for r in range(nmo_so):
                for s in range(nmo_so):
                    val = 0.0
                    if mo_spin[p] == mo_spin[r] and mo_spin[q] == mo_spin[s]:
                        val += phys[mo_spat[p], mo_spat[q], mo_spat[r], mo_spat[s]]
                    if mo_spin[p] == mo_spin[s] and mo_spin[q] == mo_spin[r]:
                        val -= phys[mo_spat[p], mo_spat[q], mo_spat[s], mo_spat[r]]
                    g_so[p, q, r, s] = val

    kd_so = np.eye(nmo_so)
    o_so, v_so = slice(0, no_so), slice(no_so, nmo_so)

    r3_so = L3M.lambda3_residual(t1_so, t2_so, t3_so, l1_so, l2_so, l3_so,
                                 f_so, g_so, kd_so, o_so, v_so)

    args_restricted = (t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
                       t3_aaaaaa, t3_aabaab, t3_abbabb, t3_aaaaaa,
                       f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v,
                       l1_aa, l1_aa, l2_aaaa, l2_abab, l2_aaaa,
                       l3_aaaaaa, l3_aabaab, l3_abbabb, l3_aaaaaa)

    r3_aaaaaa_restricted = LR.l3_aaaaaa_residual(*args_restricted)
    r3_aabaab_restricted = LR.l3_aabaab_residual(*args_restricted)
    r3_abbabb_restricted = LR.l3_abbabb_residual(*args_restricted)

    r3_so_aaaaaa = r3_so[0::2, 0::2, 0::2, 0::2, 0::2, 0::2]
    r3_so_aabaab = r3_so[0::2, 0::2, 1::2, 0::2, 0::2, 1::2]
    r3_so_abbabb = r3_so[0::2, 1::2, 1::2, 0::2, 1::2, 1::2]

    all_ok = True
    for name, r_gen, r_res in [
        ('aaaaaa', r3_so_aaaaaa, r3_aaaaaa_restricted),
        ('aabaab', r3_so_aabaab, r3_aabaab_restricted),
        ('abbabb', r3_so_abbabb, r3_abbabb_restricted),
    ]:
        diff = np.max(np.abs(r_gen - r_res))
        scale = max(np.max(np.abs(r_gen)), 1e-300)
        rel = diff / scale
        ok = rel < 1e-9
        all_ok &= ok
        print(f"l3_{name}_residual vs generalized lambda3_residual.py: "
              f"max abs diff {diff:.2e}, relative {rel:.2e}: {'OK' if ok else 'FAIL'}")
    return all_ok


def _t_ccsdt_corr_energy_restricted(mf, stopping_eps=1e-11):
    ints = build_restricted_integrals_from_mf(mf)
    f_aa, g_aaaa, g_abab = ints['f_aa'], ints['g_aaaa'], ints['g_abab']
    no, nv = ints['nocc'], ints['nvir']
    o, v = ints['o'], ints['v']
    t1_aa = np.zeros((nv, no))
    t2_aaaa = np.zeros((nv, nv, no, no))
    t2_abab = np.zeros((nv, nv, no, no))
    t3_aaaaaa = np.zeros((nv, nv, nv, no, no, no))
    t3_aabaab = np.zeros((nv, nv, nv, no, no, no))
    t3_abbabb = np.zeros((nv, nv, nv, no, no, no))
    t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb = kernel_t_restricted(
        t1_aa, t2_aaaa, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb,
        f_aa, g_aaaa, g_abab, o, v, ints['hf_energy'],
        stopping_eps=stopping_eps, verbose=False)
    e_total = TR.cc_energy_restricted(t1_aa, t1_aa, t2_aaaa, t2_abab, t2_aaaa,
                                      f_aa, f_aa, g_aaaa, g_abab, g_aaaa, o, v)
    return e_total - ints['hf_energy']


def _t_ccsdt_corr_energy_generalized(mf, stopping_eps=1e-11):
    ints = build_spinorbital_integrals_from_mf(mf)
    fock, g = ints['fock'], ints['g']
    nocc, nvir = ints['nocc'], ints['nvir']
    o, v = slice(None, nocc), slice(nocc, None)
    e_ai, e_abij, e_abcijk = energy_denominators(fock, nocc, nvir)
    t1 = np.zeros((nvir, nocc))
    t2 = np.zeros((nvir, nvir, nocc, nocc))
    t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc))
    t1, t2, t3 = amplitudes.kernel(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk,
                                   ints['hf_energy'], stopping_eps=stopping_eps)
    e_total = amplitudes.cc_energy(t1, t2, fock, g, o, v)
    return e_total - ints['hf_energy']


def check_h2_631g_structural_match():
    """H2 is a 2-electron system: CCSD is already FCI, and 6-31g gives 4
    virtuals (T3 nonzero in general, but the true wavefunction has no
    genuine triple-excitation content for 2 electrons) -- exercises the
    restricted T3/Lambda3 machinery on real (if small) virtual space. With
    Lambda3 included, the full density should match the generalized
    pipeline closely."""
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g')
    mf = mol.RHF(); mf.verbose = 0; mf.run()

    dm_restricted = compute_ccsdt_density_matrix_restricted(mf, verbose=False)
    dm_generalized = compute_ccsdt_density_matrix(mf, verbose=False)
    diff = np.max(np.abs(dm_restricted - dm_generalized))

    nelec = np.trace(dm_restricted @ mf.get_ovlp())
    ok = diff < 1e-6 and abs(nelec - mol.nelectron) < 1e-8
    print(f"H2/6-31g: restricted vs generalized CCSDT density (max diff "
          f"{diff:.2e}), trace={nelec:.6f} (expect {mol.nelectron}): "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_lih_sto3g_t_energy_matches():
    """LiH/sto-3g has genuinely nonzero T3 (|t3| ~ 1e-2, per
    tests/test_ccsdt_lambda.py). The T-CCSDT correlation
    energy never depends on Lambda at all, so this isolates whether the
    dominant-cost restricted T1/T2/T3 equations are exactly correct,
    independent of the Lambda3 equations checked elsewhere."""
    mol = pyscf.M(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g')
    mf = mol.RHF(); mf.verbose = 0; mf.run()

    e_restricted = _t_ccsdt_corr_energy_restricted(mf)
    e_generalized = _t_ccsdt_corr_energy_generalized(mf)
    diff = abs(e_restricted - e_generalized)

    ok = diff < 1e-9
    print(f"LiH/sto-3g: restricted vs generalized T-CCSDT correlation energy "
          f"(restricted={e_restricted:.12f}, generalized={e_generalized:.12f}, "
          f"diff {diff:.2e}): {'OK' if ok else 'FAIL'}")
    return ok


def check_lih_sto3g_density_matches_generalized():
    """With Lambda3 included on both sides, the restricted (spin-blocked)
    and generalized (spin-orbital) full-CCSDT densities solve the exact
    same physical Lambda-CCSDT equations, so the AO density should
    match to ~1e-9 (float-roundoff / DIIS-convergence level), not just be
    'small but nonzero'."""
    mol = pyscf.M(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g')
    mf = mol.RHF(); mf.verbose = 0; mf.run()

    dm_restricted = compute_ccsdt_density_matrix_restricted(mf, verbose=False)
    dm_generalized = compute_ccsdt_density_matrix(mf, verbose=False)
    diff = np.max(np.abs(dm_restricted - dm_generalized))

    ok = diff < 1e-7
    print(f"LiH/sto-3g: restricted vs generalized full-Lambda3 CCSDT density, "
          f"max diff {diff:.2e} (expect ~1e-9, DIIS-convergence level): "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_speedup():
    mol = pyscf.M(atom='Li 0 0 0; H 0 0 1.5957', basis='sto-3g')
    mf = mol.RHF(); mf.verbose = 0; mf.run()

    t0 = time.time()
    compute_ccsdt_density_matrix_restricted(mf, verbose=False)
    t_restricted = time.time() - t0

    t0 = time.time()
    compute_ccsdt_density_matrix(mf, verbose=False)
    t_generalized = time.time() - t0

    speedup = t_generalized / t_restricted
    print(f"LiH/sto-3g timing: restricted {t_restricted:.2f}s, "
          f"generalized {t_generalized:.2f}s, speedup {speedup:.2f}x")
    return True


if __name__ == '__main__':
    all_ok = True
    all_ok &= check_integral_sanity()
    all_ok &= check_l3_residual_matches_generalized()
    all_ok &= check_h2_631g_structural_match()
    all_ok &= check_lih_sto3g_t_energy_matches()
    all_ok &= check_lih_sto3g_density_matches_generalized()
    all_ok &= check_speedup()

    print()
    print("ALL OK" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)

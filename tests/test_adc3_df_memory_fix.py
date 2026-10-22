"""
Regression test for the ADC(3) DF matrix-free memory reduction
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC import ADCSolverRestricted
from src.SingleReference.ADC.adc_utils import (
    _g_slice_df, _b_factor, _b_chunk_einsum, _b_q_chunk)
from src.SingleReference.ADC.adc_r_utils import (
    _build_g_blocks_df, _u_pv_vp_contract_df, _u_Ip_pv_vp_contract_df,
    _u_I_ring_contract_df, _u2_spin_amplitudes,
    _u2_2h1p_amplitude_chunks, _u2_2p1h_amplitude_chunks)
from src.SingleReference.ADC.adc_r_sigma_df import (
    _u_I_ring_contract_streamed_bare, _u_Ip_pv_vp_contract_streamed_bare)
import src.SingleReference.ADC.adc_utils as _adc_utils_mod
import src.SingleReference.ADC.adc_r_utils as _adc_r_utils_mod
import src.SingleReference.ADC.adc_r_sigma_df as _adc_r_sigma_df_mod
from tests.pins import RECORDING_BLAS, matches


class _SpyTargets:
    """_g_slice_df is bound in adc_utils (definer), adc_r_utils and
    adc_r_sigma_df (from-imports). Setting an attribute here sets it on all
    three, so a single-module monkeypatch covers every call site (_B_SLAB_BYTES
    is read from adc_utils'
    own global, also covered)."""
    _mods = (_adc_utils_mod, _adc_r_utils_mod, _adc_r_sigma_df_mod)

    def __getattr__(self, name):
        return getattr(self._mods[0], name)

    def __setattr__(self, name, value):
        for m in self._mods:
            setattr(m, name, value)
from src.SingleReference.EpsteinNesbet import (
    EpsteinNesbetDenominators, restricted_channel_shifts)
from src.Base.pyscf_interface import get_orbital_energies, get_two_electron_integrals_chemist, DFIntegrals


if __name__ == '__main__':
    all_ok = True

    # --- (a) the skip flags actually skip building the blocks ---
    rng = np.random.default_rng(0)
    O, V, norb = 3, 5, 8
    B = rng.standard_normal((7, norb, norb))

    blk_full = _build_g_blocks_df(B, O, V, norb)
    ok = all(k in blk_full for k in ('g_vv_pv', 'g_vv_vp', 'g_vv_pp', 'g_vvvv'))
    all_ok &= ok
    print(f"_build_g_blocks_df default (all True): includes g_vv_pv/vp/pp/vvvv: {'OK' if ok else 'FAIL'}")

    blk_skip = _build_g_blocks_df(B, O, V, norb, need_vvvv=False, need_vv_pv_vp=False, need_vv_pp=False)
    missing = [k for k in ('g_vv_pv', 'g_vv_vp', 'g_vv_pp', 'g_vvvv') if k not in blk_skip]
    ok = sorted(missing) == ['g_vv_pp', 'g_vv_pv', 'g_vv_vp', 'g_vvvv']
    all_ok &= ok
    print(f"_build_g_blocks_df(need_vvvv/need_vv_pv_vp/need_vv_pp=False) omits all four blocks: "
          f"{'OK' if ok else 'FAIL'} (still present: {[k for k in ('g_vv_pv','g_vv_vp','g_vv_pp','g_vvvv') if k in blk_skip]})")

    # every OTHER block must still be present and byte-identical between the
    # two calls -- the skip flags must not have side effects on unrelated blocks
    common_keys = set(blk_full) & set(blk_skip)
    # two builds: each block to a few ulp
    ok = all(matches(blk_full[k], blk_skip[k], RECORDING_BLAS, quantity='root')
             for k in common_keys) and len(common_keys) == 16
    all_ok &= ok
    print(f"remaining 16 blocks unaffected by skip flags: {'OK' if ok else 'FAIL'} (common_keys={len(common_keys)})")

    # --- (b) _u_pv_vp_contract_df reproduces the dense g_vv_pv/g_vv_vp einsum exactly ---
    vir = np.arange(O, norb)
    allo = np.arange(norb)
    g_vv_pv = _g_slice_df(B, vir, vir, allo, vir)   # dense reference block, (V,V,norb,V)
    g_vv_vp = _g_slice_df(B, vir, vir, vir, allo)   # (V,V,V,norb)
    Bv_full = B[:, vir, :]
    Bv_vv = B[:, vir][:, :, vir]

    X_icd = rng.standard_normal((O, V, V))
    dense_pv = np.einsum('icd,cdpa->ipa', X_icd, g_vv_pv, optimize=True)
    dense_vp = np.einsum('icd,cdap->ipa', X_icd, g_vv_vp, optimize=True)
    df_pv, df_vp = _u_pv_vp_contract_df(Bv_full, Bv_vv, X_icd, 'i', 'i')
    diff = max(np.max(np.abs(dense_pv - df_pv)), np.max(np.abs(dense_vp - df_vp)))
    ok = diff < 1e-10
    all_ok &= ok
    print(f"_u_pv_vp_contract_df (rank-1 free index 'i') matches dense g_vv_pv/vp einsum "
          f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    X_ijcd = rng.standard_normal((O, O, V, V))
    dense_pv2 = np.einsum('ijcd,cdpa->ijpa', X_ijcd, g_vv_pv, optimize=True)
    dense_vp2 = np.einsum('ijcd,cdap->ijpa', X_ijcd, g_vv_vp, optimize=True)
    df_pv2, df_vp2 = _u_pv_vp_contract_df(Bv_full, Bv_vv, X_ijcd, 'ij', 'ij')
    diff = max(np.max(np.abs(dense_pv2 - df_pv2)), np.max(np.abs(dense_vp2 - df_vp2)))
    ok = diff < 1e-10
    all_ok &= ok
    print(f"_u_pv_vp_contract_df (rank-2 free index 'ij') matches dense g_vv_pv/vp einsum "
          f"(max diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    # --- (c) end-to-end: DF matrix-free operator (which goes through
    # the skip flags) matches dense build_supermatrix's H@z exactly,
    # on a molecule with nontrivial virtual mixing and level='adc3' (so the
    # U_I/U_II/U_III second-order pieces the flags touch are active --
    # gated off entirely at level='adc2x') ---
    mol = gto.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(mf, representation='spatial')
    eri_chemist = get_two_electron_integrals_chemist(mol, mf)

    dense = ADCSolverRestricted.from_arrays(eps, eri_chemist, level='adc3')
    H_dense = dense.build_supermatrix(nocc)

    dfi = DFIntegrals.from_scf(mol, mf, exact=True)
    drv = ADCSolverRestricted.from_arrays(eps, eri_chemist, level='adc3', B_aa=dfi.B_aa)
    aop, diag, dims = drv.build_matrix_free_operator(nocc)

    rng2 = np.random.default_rng(1)
    nH = H_dense.shape[0]
    max_diff = 0.0
    for _ in range(3):
        z = rng2.standard_normal(nH)
        max_diff = max(max_diff, np.max(np.abs(aop(z) - H_dense @ z)))
    ok = max_diff < 1e-8
    all_ok &= ok
    print(f"post-fix DF matrix-free operator == dense H@z, LiH/sto-3g level=adc3 "
          f"(max diff={max_diff:.2e}): {'OK' if ok else 'FAIL'}")

    # ================== S1: g_ovpv/g_ovvp ==================
    # Same three-part shape as the fix above, for the two blocks it left
    # behind: g_ovpv (O,V,norb,V) and g_ovvp (O,V,V,norb). These are the
    # O*V^2*norb size class -- the same class as the U_IIp/U_IIIp block that
    # apply_U_2p1h_fwd/_adj made matrix-free -- and at hexacene/cc-pVQZ they
    # are 2.8 TB APIECE, 5.6 of that system's 8.3 TB total. Their only
    # consumer on the unscreened DF path is U_Ip's pair of ring
    # contractions, routed through _u_Ip_pv_vp_contract_df.
    print()

    # --- (a) the skip flag actually skips building the blocks ---
    blk_ov_skip = _build_g_blocks_df(B, O, V, norb, need_ovpv_ovvp=False)
    missing = [k for k in ('g_ovpv', 'g_ovvp') if k not in blk_ov_skip]
    ok = sorted(missing) == ['g_ovpv', 'g_ovvp']
    all_ok &= ok
    print(f"_build_g_blocks_df(need_ovpv_ovvp=False) omits g_ovpv/g_ovvp: "
          f"{'OK' if ok else 'FAIL'} (still present: "
          f"{[k for k in ('g_ovpv', 'g_ovvp') if k in blk_ov_skip]})")

    # and, as above, no side effects on any other block (two builds: each
    # block to a few ulp)
    common_keys = set(blk_full) & set(blk_ov_skip)
    ok = (all(matches(blk_full[k], blk_ov_skip[k], RECORDING_BLAS,
                      quantity='root') for k in common_keys)
          and len(common_keys) == 18)
    all_ok &= ok
    print(f"remaining 18 blocks unaffected by need_ovpv_ovvp: "
          f"{'OK' if ok else 'FAIL'} (common_keys={len(common_keys)})")

    # --- (b) _u_Ip_pv_vp_contract_df reproduces the dense einsums exactly,
    # at every chunk boundary (q_chunk sweeps naux from "one Q per chunk" to
    # "all of Q in one chunk", since Q is a summed index carried by both DF
    # factors and per-chunk accumulation has to be exact) ---
    occ = np.arange(O)
    g_ovpv = _g_slice_df(B, occ, vir, allo, vir)    # dense reference, (O,V,norb,V)
    g_ovvp = _g_slice_df(B, occ, vir, vir, allo)    # (O,V,V,norb)
    W1p = rng.standard_normal((V, V, O, O))         # 'ackl'-layout amplitudes
    W2p = rng.standard_normal((V, V, O, O))
    dense_1 = np.einsum('acki,kapc->api', W1p, g_ovpv, optimize=True)
    dense_2 = np.einsum('acki,kacp->api', W2p, g_ovvp, optimize=True)
    worst = 0.0
    for q_chunk in (1, 2, 3, B.shape[0] - 1, B.shape[0], 2 * B.shape[0]):
        df_1, df_2 = _u_Ip_pv_vp_contract_df(B[:, occ, :], B[:, vir][:, :, vir],
                                             B[:, occ][:, :, vir], B[:, vir, :],
                                             W1p, W2p, q_chunk=q_chunk)
        worst = max(worst, np.max(np.abs(dense_1 - df_1)), np.max(np.abs(dense_2 - df_2)))
    ok = worst < 1e-10
    all_ok &= ok
    print(f"_u_Ip_pv_vp_contract_df matches dense g_ovpv/g_ovvp einsums, q_chunk "
          f"1..2*naux (max diff={worst:.2e}): {'OK' if ok else 'FAIL'}")

    # --- (c) the PRODUCTION path really takes the lean route: spy on
    # _g_slice_df and assert build_matrix_free_operator never asks it for an
    # O*V^2*norb-shaped block. Numerical agreement alone cannot show this --
    # the materialized branch would agree too, which is the whole point of
    # the "must actually exercise it" rule at the top of this file. ---
    _restricted_mod = _SpyTargets()

    built_shapes = []
    _real_g_slice_df = _restricted_mod._g_slice_df

    def _spy(B_, p, q, r, s):
        out = _real_g_slice_df(B_, p, q, r, s)
        built_shapes.append(out.shape)
        return out

    _restricted_mod._g_slice_df = _spy
    try:
        drv_spy = ADCSolverRestricted.from_arrays(eps, eri_chemist, level='adc3', B_aa=dfi.B_aa)
        aop_spy, _, _ = drv_spy.build_matrix_free_operator(nocc)
        z = rng2.standard_normal(nH)
        spy_diff = np.max(np.abs(aop_spy(z) - H_dense @ z))
    finally:
        _restricted_mod._g_slice_df = _real_g_slice_df

    O_, V_, norb_ = nocc, len(eps) - nocc, len(eps)
    banned = {(O_, V_, norb_, V_), (O_, V_, V_, norb_)}   # g_ovpv, g_ovvp exactly
    offenders = [s for s in built_shapes if s in banned]
    ok = not offenders and spy_diff < 1e-8
    all_ok &= ok
    print(f"unscreened DF build_matrix_free_operator builds NO g_ovpv/g_ovvp-shaped "
          f"block ({len(built_shapes)} slices built, none in {sorted(banned)}; H@z "
          f"still exact at {spy_diff:.2e}): {'OK' if ok else 'FAIL'}"
          + (f" offenders={offenders}" if offenders else ""))

    # ================== S3: the 2h1p U side ==================
    # u_II/u_III/Y_mix_1/Y_mix_2 + the g_ijpa-g_ijap_T temporary are five
    # (O,O,norb,V) arrays (92 GB apiece at hexacene/QZ), the explicit
    # U_II/U_III another 45 GB each, and g_oo_pp/g_vopo/g_voho (95/92/92 GB)
    # exist only to feed them. U_II/U_III are applied matrix-free and
    # U_I's rings go through _u_I_ring_contract_df.
    print()

    for flag, keys in (('need_oo_pp', ('g_oo_pp',)),
                       ('need_vopo_voho', ('g_vopo', 'g_voho'))):
        blk_s = _build_g_blocks_df(B, O, V, norb, **{flag: False})
        ok = all(k not in blk_s for k in keys)
        all_ok &= ok
        print(f"_build_g_blocks_df({flag}=False) omits {'/'.join(keys)}: "
              f"{'OK' if ok else 'FAIL'}")

    # _u_I_ring_contract_df vs the dense einsums it replaces, at every chunk
    # boundary (it streams over i AND chunks over Q)
    g_vopo = _g_slice_df(B, vir, occ, allo, occ)
    g_voho = _g_slice_df(B, vir, occ, occ, allo)
    W1r = rng.standard_normal((O, O, V, V))
    W2r = rng.standard_normal((O, O, V, V))
    d1 = np.einsum('ikca,cipk->ipa', W1r, g_vopo, optimize=True)
    d2 = np.einsum('ikca,cikp->ipa', W2r, g_voho, optimize=True)
    worst = 0.0
    for q_chunk in (1, 2, B.shape[0] - 1, B.shape[0], 2 * B.shape[0]):
        r1, r2 = _u_I_ring_contract_df(B[:, occ, :], B[:, vir, :],
                                       B[:, occ][:, :, occ], B[:, vir][:, :, occ],
                                       W1r, W2r, q_chunk=q_chunk)
        worst = max(worst, np.max(np.abs(d1 - r1)), np.max(np.abs(d2 - r2)))
    ok = worst < 1e-10
    all_ok &= ok
    print(f"_u_I_ring_contract_df matches dense g_vopo/g_voho einsums, q_chunk "
          f"1..2*naux (max diff={worst:.2e}): {'OK' if ok else 'FAIL'}")

    # the production path builds none of the three (spy reused from above)
    built_shapes.clear()
    _restricted_mod._g_slice_df = _spy
    try:
        drv3 = ADCSolverRestricted.from_arrays(eps, eri_chemist, level='adc3', B_aa=dfi.B_aa)
        aop3, _, _ = drv3.build_matrix_free_operator(nocc)
        z = rng2.standard_normal(nH)
        d3 = np.max(np.abs(aop3(z) - H_dense @ z))
    finally:
        _restricted_mod._g_slice_df = _real_g_slice_df
    banned3 = {(O_, O_, norb_, norb_), (V_, O_, norb_, O_), (O_, V_, O_, norb_),
               (V_, O_, O_, norb_)}
    off3 = [s for s in built_shapes if s in banned3]
    ok = not off3 and d3 < 1e-8
    all_ok &= ok
    print(f"unscreened DF build_matrix_free_operator builds NO "
          f"g_oo_pp/g_vopo/g_voho-shaped block ({len(built_shapes)} slices built; "
          f"H@z still exact at {d3:.2e}): {'OK' if ok else 'FAIL'}"
          + (f" offenders={off3}" if off3 else ""))

    # ================== S2: view-vs-copy DF factors ==================
    # The six shared factors (Bv_pp/Bv_full/Bvo_full/Bo_full/Bov_full/Boo_full)
    # were built by DOUBLE FANCY INDEXING (e.g. B[:, vidx_abs][:, :, vidx_abs]),
    # each a persistent copy -- 217 GB of pure duplication at hexacene/cc-pVQZ.
    # _b_factor(B, row_idx, col_idx, view) returns a copy below _B_SLAB_BYTES,
    # else the bare view. THE COPY MUST MATCH FANCY INDEXING'S OWN MEMORY
    # LAYOUT, not just its values: forcing C-contiguous via np.ascontiguousarray
    # on a basic-slice view measured 1.5x SLOWER end-to-end on H2O/cc-pVQZ,
    # because these factors feed dozens of OTHER contractions elsewhere in
    # build_matrix_free_operator (U_I's rings, U_Ip's, U_IIp/U_IIIp's), not just
    # the calls any one fix touches -- so a naive "make it contiguous" fix looks
    # right in isolation and regresses the whole matvec.
    print()

    rng3 = np.random.default_rng(2)
    naux3, norb3, O3 = 12, 9, 4
    B3 = rng3.standard_normal((naux3, norb3, norb3))
    vidx3, oidx3, allo3 = np.arange(O3, norb3), np.arange(O3), np.arange(norb3)

    small = _b_factor(B3, vidx3, vidx3, B3[:, O3:, O3:])
    ok = np.array_equal(small, B3[:, vidx3][:, :, vidx3])
    all_ok &= ok
    print(f"_b_factor (small) matches double fancy indexing exactly, values AND "
          f"layout ({small.flags}=={B3[:, vidx3][:, :, vidx3].flags}): "
          f"{'OK' if ok else 'FAIL'}")

    orig_slab = _restricted_mod._B_SLAB_BYTES
    try:
        _restricted_mod._B_SLAB_BYTES = 1  # force every factor onto the view branch
        big = _b_factor(B3, oidx3, allo3, B3[:, :O3, :])
        ok = not big.flags.owndata and big.base is B3
        all_ok &= ok
        print(f"_b_factor (forced large) returns a bare VIEW of B, no copy: "
              f"{'OK' if ok else 'FAIL'}")

        Y3 = rng3.standard_normal((naux3, O3))
        ref3 = np.einsum('Qip,Qi->p', big, Y3, optimize=True)
        worst3 = 0.0
        for slab in (50, 500, 5000, 50000):
            _restricted_mod._B_SLAB_BYTES = slab
            got3 = _b_chunk_einsum('Qip,Qi->p', (big, Y3), (True, True), big)
            worst3 = max(worst3, np.max(np.abs(got3 - ref3)))
        ok = worst3 < 1e-10
        all_ok &= ok
        print(f"_b_chunk_einsum on a forced view matches unchunked einsum, every "
              f"slab size (max diff={worst3:.2e}): {'OK' if ok else 'FAIL'}")
    finally:
        _restricted_mod._B_SLAB_BYTES = orig_slab

    # end-to-end timing parity: S2 must not slow the matvec down, on the same
    # LiH/sto-3g system and EN dressing already exercised above
    import time
    drv_s2 = ADCSolverRestricted.from_arrays(eps, None, level='adc3', B_aa=dfi.B_aa,
                                 en_dress={'hh': True, 'pp': True})
    aop_s2, _, dims_s2 = drv_s2.build_matrix_free_operator(nocc)
    zt = rng2.standard_normal(dims_s2['nH'])
    aop_s2(zt)
    t0 = time.time()
    for _ in range(20):
        aop_s2(zt)
    per_call = (time.time() - t0) / 20
    ok = per_call < 0.05   # generous ceiling for LiH/sto-3g; catches gross regressions
    all_ok &= ok
    print(f"S2 matvec sanity timing (LiH/sto-3g, EN hh+pp): {per_call*1000:.2f} ms/call: "
          f"{'OK' if ok else 'FAIL'}")

    # ================== S4: occupied-index streaming, bare EN only ==================
    # t_same/t_opp/W1/W2/A_mix/X_ijcd (2h1p) and tp_same/tp_opp/W1p/W2p/X_abkl
    # (2p1h) are each O^2V^2 -- 89 GB apiece at hexacene/cc-pVQZ, 17 alive
    # simultaneously (11 amplitude-derived + 6 "common" integrals). For BARE
    # (undressed) EN, D is a trivial orbital-energy broadcast -- separable,
    # not a genuine 4-index object -- so this streams the amplitude
    # construction over an occupied index, never materializing the full
    # array. Gated to u2_denom_dress falsy + B_aa is not None + level!=adc2x
    # + not screened; EN-dressed/adc2x/dense/screened fall back unchanged
    # (their pair shifts are NOT orbital-separable).
    print()

    rng4 = np.random.default_rng(14)
    O4, V4, naux4, norb4 = 5, 8, 17, 13
    eps_o4 = np.sort(rng4.uniform(-2, -0.3, O4))
    eps_v4 = np.sort(rng4.uniform(0.2, 3, V4))
    B4 = rng4.standard_normal((naux4, norb4, norb4))
    B4 = 0.5 * (B4 + B4.transpose(0, 2, 1))
    occ4, vir4 = np.arange(O4), np.arange(O4, norb4)

    # --- (a) need_oovv_vvoo=False actually skips the blocks, no side effects ---
    blk_full4 = _build_g_blocks_df(B4, O4, V4, norb4)
    ok = 'g_oovv' in blk_full4 and 'g_vvoo' in blk_full4
    all_ok &= ok
    print(f"_build_g_blocks_df default (all True): includes g_oovv/g_vvoo: {'OK' if ok else 'FAIL'}")

    blk_skip4 = _build_g_blocks_df(B4, O4, V4, norb4, need_oovv_vvoo=False)
    ok = 'g_oovv' not in blk_skip4 and 'g_vvoo' not in blk_skip4
    all_ok &= ok
    print(f"_build_g_blocks_df(need_oovv_vvoo=False) omits g_oovv/g_vvoo: {'OK' if ok else 'FAIL'}")

    common4 = set(blk_full4) & set(blk_skip4)
    ok = all(matches(blk_full4[k], blk_skip4[k], RECORDING_BLAS, quantity='root')
             for k in common4) and len(common4) == 18
    all_ok &= ok
    print(f"remaining 18 blocks unaffected by need_oovv_vvoo: {'OK' if ok else 'FAIL'} "
          f"(common_keys={len(common4)})")

    # --- (b) the chunk generators reproduce the full-array construction ---
    g_oovv4 = _g_slice_df(B4, occ4, occ4, vir4, vir4)
    g_oovv4_T = g_oovv4.transpose(0, 1, 3, 2)
    dens4 = EpsteinNesbetDenominators(eps_o4, eps_v4)
    t_same4, t_opp4 = _u2_spin_amplitudes(g_oovv4, g_oovv4_T, dens4, 'ikca')
    W1_ref4 = t_same4 - 2 * t_opp4
    X_ijcd_ref4 = g_oovv4 / dens4.denom('ijcd', 'opp')

    g_vvoo4 = _g_slice_df(B4, vir4, vir4, occ4, occ4)
    g_vvoo4_T = g_vvoo4.transpose(0, 1, 3, 2)
    tp_same4, tp_opp4 = _u2_spin_amplitudes(g_vvoo4, g_vvoo4_T, dens4, 'ackl')
    W1p_ref4 = 2 * tp_opp4 - tp_same4
    X_abkl_ref4 = g_vvoo4 / dens4.denom('abkl', 'opp')

    worst4 = 0.0
    for chunk in (1, 2, 3, O4):
        W1_c = np.zeros_like(W1_ref4); X_ijcd_c = np.zeros_like(X_ijcd_ref4)
        for lo, hi, ts, to, w1, w2, am, xi in _u2_2h1p_amplitude_chunks(
                B4, O4, V4, norb4, eps_o4, eps_v4, chunk):
            W1_c[lo:hi] = w1; X_ijcd_c[lo:hi] = xi
        worst4 = max(worst4, np.abs(W1_c - W1_ref4).max(), np.abs(X_ijcd_c - X_ijcd_ref4).max())
        W1p_c = np.zeros_like(W1p_ref4); X_abkl_c = np.zeros_like(X_abkl_ref4)
        for lo, hi, tps, tpo, w1p, w2p, xa in _u2_2p1h_amplitude_chunks(
                B4, O4, V4, norb4, eps_o4, eps_v4, chunk):
            W1p_c[:, :, :, lo:hi] = w1p; X_abkl_c[:, :, :, lo:hi] = xa
        worst4 = max(worst4, np.abs(W1p_c - W1p_ref4).max(), np.abs(X_abkl_c - X_abkl_ref4).max())
    ok = worst4 < 1e-10
    all_ok &= ok
    print(f"_u2_2h1p_amplitude_chunks/_u2_2p1h_amplitude_chunks match the "
          f"full-array construction, chunk sizes 1,2,3,O (max diff={worst4:.2e}): "
          f"{'OK' if ok else 'FAIL'}")

    # --- (c) the streamed U_I/U_Ip ring builders match the full-array ones ---
    Bo4 = B4[:, occ4, :]; Bv4 = B4[:, vir4, :]
    Boo4 = B4[:, occ4][:, :, occ4]; Bvo4 = B4[:, vir4][:, :, occ4]
    Bv_pp4 = B4[:, vir4][:, :, vir4]; Bov4 = B4[:, occ4][:, :, vir4]
    W2_ref4 = t_opp4 - 2 * t_same4
    term1_ref, term2_ref = _u_I_ring_contract_df(Bo4, Bv4, Boo4, Bvo4, W1_ref4, W2_ref4)
    term1, term2 = _u_I_ring_contract_streamed_bare(B4, O4, V4, norb4, eps_o4, eps_v4)
    d_ui = max(np.abs(term1 - term1_ref).max(), np.abs(term2 - term2_ref).max())

    W2p_ref4 = 2 * tp_same4 - tp_opp4
    t1_ref, t2_ref = _u_Ip_pv_vp_contract_df(Bo4, Bv_pp4, Bov4, Bv4, W1p_ref4, W2p_ref4)
    t1, t2 = _u_Ip_pv_vp_contract_streamed_bare(B4, O4, V4, norb4, eps_o4, eps_v4)
    d_uip = max(np.abs(t1 - t1_ref).max(), np.abs(t2 - t2_ref).max())

    ok = max(d_ui, d_uip) < 1e-10
    all_ok &= ok
    print(f"_u_I_ring_contract_streamed_bare/_u_Ip_pv_vp_contract_streamed_bare "
          f"match the full-array builders (U_I diff={d_ui:.2e}, U_Ip diff={d_uip:.2e}): "
          f"{'OK' if ok else 'FAIL'}")

    # --- (d) end-to-end: DF matrix-free operator == dense H@z for bare EN,
    # and the spy confirms the production path builds NO (O,O,V,V)/(V,V,O,O)
    # slice (the actual point of S4) ---
    mol4 = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf4 = scf.RHF(mol4).run()
    nocc4 = mol4.nelectron // 2
    eps4 = get_orbital_energies(mf4, representation='spatial')
    eri4 = get_two_electron_integrals_chemist(mol4, mf4)
    dfi4 = DFIntegrals.from_scf(mol4, mf4, exact=True)
    dense4 = ADCSolverRestricted.from_arrays(eps4, eri4, level='adc3')
    H_dense4 = dense4.build_supermatrix(nocc4)

    built_shapes4 = []
    real_slice = _restricted_mod._g_slice_df
    def _spy4(B, p, q, r, s):
        out = real_slice(B, p, q, r, s)
        built_shapes4.append(out.shape)
        return out
    _restricted_mod._g_slice_df = _spy4
    try:
        drv4 = ADCSolverRestricted.from_arrays(eps4, eri4, level='adc3', B_aa=dfi4.B_aa)
        aop4, _, dims4 = drv4.build_matrix_free_operator(nocc4)
        rng5 = np.random.default_rng(15)
        max_diff4 = 0.0
        for _ in range(3):
            z = rng5.standard_normal(H_dense4.shape[0])
            max_diff4 = max(max_diff4, np.max(np.abs(aop4(z) - H_dense4 @ z)))
    finally:
        _restricted_mod._g_slice_df = real_slice

    O4_, norb4_ = nocc4, len(eps4); V4_ = norb4_ - O4_
    banned4 = {(O4_, O4_, V4_, V4_), (V4_, V4_, O4_, O4_)}
    offenders4 = [s for s in built_shapes4 if s in banned4]
    ok = max_diff4 < 1e-9 and not offenders4
    all_ok &= ok
    print(f"unscreened bare-EN DF build_matrix_free_operator: H@z exact "
          f"(max diff={max_diff4:.2e}), builds NO g_oovv/g_vvoo-shaped slice "
          f"({len(built_shapes4)} slices, 0 of shape {sorted(banned4)}): "
          f"{'OK' if ok else 'FAIL'}" + (f" offenders={offenders4}" if offenders4 else ""))

    # --- S4 followup: EN-dressed (hh/pp, spin_adapted) streaming --------
    # _u2_2h1p_amplitude_chunks/_u2_2p1h_amplitude_chunks gained optional
    # dh/dp shift matrices so the same occupied-index streaming covers
    # u2_denom_dress={'hh':.., 'pp':..} (spin_adapted=True, the production
    # DRESS_MODES), not just bare EN -- see the banner comment above
    # _u2_2h1p_amplitude_chunks and _dress_is_streamable. Checks: (a) the
    # chunked dh/dp construction matches EpsteinNesbetDenominators' full
    # -array build exactly, (b) the end-to-end dressed matrix-free operator
    # matches dense H@z and builds no O^2V^2-shaped slice, and (c) 'hp'/
    # spin-resolved dressing correctly fall back to the materialized path
    # (still exact, just not memory-lean) rather than silently mis-dressing.
    mol5 = gto.M(atom='C 0 0 0; O 0 0 1.13', basis='sto-3g', verbose=0)
    mf5 = scf.RHF(mol5).run()
    nocc5 = mol5.nelectron // 2
    eps5 = get_orbital_energies(mf5, representation='spatial')
    eri5 = get_two_electron_integrals_chemist(mol5, mf5)
    dfi5 = DFIntegrals.from_scf(mol5, mf5, exact=True)
    norb5 = len(eps5)
    O5, V5 = nocc5, norb5 - nocc5
    eps_o5, eps_v5 = eps5[:O5], eps5[O5:]
    occ5, vir5 = np.arange(O5), np.arange(O5, norb5)
    vidx_abs5 = np.arange(O5, norb5)

    dress5 = {'hh': True, 'pp': True}
    g_oooo5 = _g_slice_df(dfi5.B_aa, occ5, occ5, occ5, occ5)
    d_h5, d_p5, d_hp5 = restricted_channel_shifts(dress5, dfi5.B_aa, g_oooo5, None, O5, vidx_abs5)
    dh5, dp5 = d_h5[0], d_p5[0]
    dens5 = EpsteinNesbetDenominators(eps_o5, eps_v5, d_h5, d_p5, d_hp5)
    ok = dens5.spin_adapted and d_hp5 is None
    all_ok &= ok
    print(f"restricted_channel_shifts(hh+pp) is spin_adapted with no hp channel: {'OK' if ok else 'FAIL'}")

    D_ikca_ref, _ = dens5.build('ikca')

    # Direct D-construction check (mirrors scratchpad/en_dressed_chunks.py exactly)
    def _chunked_D_ikca(lo, hi):
        eo_c = eps_o5[lo:hi]
        D_bare = (eo_c[:, None, None, None] + eps_o5[None, :, None, None]
                 - eps_v5[None, None, :, None] - eps_v5[None, None, None, :])
        return D_bare - (dh5[lo:hi, :][:, :, None, None] + dp5[None, None, :, :])
    worst = 0.0
    for chunk in (1, 2, 3, O5):
        got = np.zeros_like(D_ikca_ref)
        for lo in range(0, O5, chunk):
            hi = min(lo + chunk, O5)
            got[lo:hi] = _chunked_D_ikca(lo, hi)
        worst = max(worst, np.abs(got - D_ikca_ref).max())
    ok = worst < 1e-10
    all_ok &= ok
    print(f"chunked EN-dressed D_ikca matches EpsteinNesbetDenominators.build(): "
          f"worst diff={worst:.2e}  {'OK' if ok else 'FAIL'}")

    # The FULL rank-4 ban list (adc_spinorbital rank rule): the O^2V^2
    # class, plus g_oooo (O^4) and g_oo_po/g_oo_op ((O,O,norb,O)) -- the
    # last two eliminated via _oooo_contract_df /
    # _corr_kl_Ip_streamed_df / the DF-direct g_klpi chains.
    banned5 = {(O5, O5, V5, V5), (V5, V5, O5, O5),
              (V5, O5, O5, V5), (V5, O5, V5, O5), (O5, V5, O5, V5), (O5, V5, V5, O5),
              (O5, O5, O5, O5), (O5, O5, norb5, O5), (O5, O5, O5, norb5)}

    def _run_dressed(dress, expect_streamed):
        # dense reference built with the SAME dress dict -- each dress
        # variant (hp on/off, spin_adapted on/off) changes the physics, so
        # comparing against a reference built with a DIFFERENT dress would
        # be a false failure, not a real one.
        H_dense = ADCSolverRestricted.from_arrays(eps5, eri5, level='adc3',
                                      en_dress=dress).build_supermatrix(nocc5)
        built = []
        real5 = _restricted_mod._g_slice_df
        def _spy5(B, p, q, r, s):
            out = real5(B, p, q, r, s)
            built.append(out.shape)
            return out
        _restricted_mod._g_slice_df = _spy5
        try:
            drv5 = ADCSolverRestricted.from_arrays(eps5, eri5, level='adc3', B_aa=dfi5.B_aa, en_dress=dress)
            aop5, _, dims5 = drv5.build_matrix_free_operator(nocc5)
            rng6 = np.random.default_rng(16)
            z = rng6.standard_normal(H_dense.shape[0])
            diff = np.max(np.abs(aop5(z) - H_dense @ z))
        finally:
            _restricted_mod._g_slice_df = real5
        offenders = [s for s in built if s in banned5]
        streamed_ok = (not offenders) if expect_streamed else True
        ok = diff < 1e-8 and streamed_ok
        return ok, diff, offenders

    ok, diff, offenders = _run_dressed(dress5, expect_streamed=True)
    all_ok &= ok
    print(f"EN-dressed (hh+pp) matrix-free operator: H@z exact (diff={diff:.2e}), "
          f"no O^2V^2-shaped slice built: {'OK' if ok else 'FAIL'}"
          + (f" offenders={offenders}" if offenders else ""))

    ok, diff, offenders = _run_dressed({'hh': True, 'pp': True, 'hp': True}, expect_streamed=False)
    all_ok &= ok
    print(f"EN-dressed (hh+pp+hp) falls back to materialized, still exact "
          f"(diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    ok, diff, offenders = _run_dressed({'hh': True, 'pp': True, 'spin_adapted': False}, expect_streamed=False)
    all_ok &= ok
    print(f"EN-dressed (hh+pp, spin_adapted=False) falls back to materialized, "
          f"still exact (diff={diff:.2e}): {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)

"""Restricted matrix-free (sigma-vector) operator from the DF factor
B_aa: the production route. Streams every O^2V^2 amplitude/integral
(no rank-4 array on the fully-streamed path); W_aux screening enters
via the screened DF factor M_aa. W_chemist (dense W) is dense-route
only -- use W_aux here."""
import math
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum
from src.SingleReference.EpsteinNesbet import (EpsteinNesbetDenominators,
                                               restricted_channel_shifts)
from src.SingleReference.ADC.adc_utils import (
    _g_slice_df, _g_diag_df, _g_diag_mid_df, _g_diag_outer_df,
    _b_factor, _b_q_chunk, _b_chunk_einsum)
from src.SingleReference.ADC.adc_r_utils import (
    _build_g_blocks_df, _dress_is_streamable, _u2_spin_amplitudes,
    _u_2h1p_unfold, _u_2p1h_unfold, _u_2p1h_zparts_df,
    _u_I_ring_contract_df, _u_pv_vp_contract_df, _u_Ip_pv_vp_contract_df,
    _u2_2h1p_amplitude_chunks, _u2_2p1h_amplitude_chunks,
    _build_u_blocks_unstreamed, _no_u_blocks)


# ===================== S4 followup: C-block occupied-index streaming =====================
# g_voov/g_vovo (2h1p) and g_ovov/g_ovvo (2p1h) are the last O^2V^2 arrays
# alive on the bare-DF path after S4/S5 -- they feed the C sub-block
# (coupling) appliers directly (not via an EN-dressed amplitude quotient, so
# _u2_*_amplitude_chunks don't apply here). Same idea as those generators:
# stream over one occupied axis, rebuilt from B every matvec, never
# materialize the full array.
def _c2h1p_df_chunks(B, O, V, norb, chunk_size=4, B_w=None):
    """Yields (lo, hi, m1a_c, m1b_c), each shape (O,V,hi-lo,V) -- streaming
    over g_vovo's own axis-3 / g_voov's own axis-2 (both labeled 'j' in the
    C_II_II/C_III_III/... derivations), the axis every consumer treats as
    either free/output (fill) or summed (accumulate) -- never the same axis
    m1a_c/m1b_c's own axis-0 ('i'), which stays full in every chunk.

    m1a_c = g_vovo[c,i,a,j_chunk] transposed to (i,a,j_chunk,c) == M1a's
            functional form; m1b_c = g_voov[c,i,j_chunk,a] transposed to
            (i,a,j_chunk,c) == M1b's (and G_I_II's) functional form.
    B_w: the DF factor for the SCREENED-eligible g_vovo chunk (m1a, the
    ring-exchange piece) -- M_aa under W_aux screening, defaults to B
    (bare). g_voov (m1b, the direct RPA ring) is always built from B."""
    if B_w is None:
        B_w = B
    occ = np.arange(O)
    vir = np.arange(O, norb)
    for lo in range(0, O, chunk_size):
        hi = min(lo + chunk_size, O)
        occ_chunk = occ[lo:hi]
        gv_chunk = _g_slice_df(B_w, vir, occ, vir, occ_chunk)  # (V,O,V,ci) = g[c,i,a,j_chunk]
        go_chunk = _g_slice_df(B, vir, occ, occ_chunk, vir)   # (V,O,ci,V) = g[c,i,j_chunk,a]
        m1a_c = gv_chunk.transpose(1, 2, 3, 0)   # (i,a,j_chunk,c)
        m1b_c = go_chunk.transpose(1, 3, 2, 0)   # (i,a,j_chunk,c)
        yield lo, hi, m1a_c, m1b_c


def _c2p1h_df_chunks(B, O, V, norb, chunk_size=4, B_w=None):
    """Yields (lo, hi, g_ovov_c, g_ovvo_c), each shape (hi-lo,V,O,V)/
    (hi-lo,V,V,O) -- streaming over dim0 ('k', the outer occ index shared by
    both integrals). Every consumer (compute_Apps and the Ip-IIp/Ip-IIIp
    cross appliers) either sums over k (accumulate) or holds it free/output
    (fill); the OTHER occ axis (g_ovov's dim2 / g_ovvo's dim3, the 2p1h
    manifold's own 'i') stays full in every chunk.
    B_w: the DF factor for the SCREENED-eligible g_ovov chunk (the
    ring-exchange piece) -- M_aa under W_aux screening, defaults to B
    (bare). g_ovvo (the direct RPA ring) is always built from B."""
    if B_w is None:
        B_w = B
    occ = np.arange(O)
    vir = np.arange(O, norb)
    for lo in range(0, O, chunk_size):
        hi = min(lo + chunk_size, O)
        occ_chunk = occ[lo:hi]
        g_ovov_c = _g_slice_df(B_w, occ_chunk, vir, occ, vir)  # (kc,V,O,V) = g[k_chunk,a,i,c]
        g_ovvo_c = _g_slice_df(B, occ_chunk, vir, vir, occ)   # (kc,V,V,O) = g[k_chunk,a,c,i]
        yield lo, hi, g_ovov_c, g_ovvo_c


def _oooo_contract_df(Foo, Vt):
    """DF-direct replacement for the C-block hh-ladder pair

        direct[i,j,a] = einsum('ijkl,kla->ija', g_oooo, Vt)
        exch[i,j,a]   = einsum('ijkl,kla->ija', g_oooo.transpose(0,1,3,2), Vt)

    with g_oooo[i,j,k,l] = sum_Q Foo[Q,i,k]Foo[Q,j,l] -- so the O^4 array
    (the last C-side rank-4 integral) need not exist. Foo is the oo block
    of the DF factor: bare Boo, or the SCREENED M_aa oo block under W_aux
    (the substitution g_oooo -> W_oooo is just a factor swap here).
    Per-i loop over O iterations, every step a rank-3 GEMM. Consumers
    combine the pair: C_II_II's T0 = direct - exch (the g_oooo_anti
    contraction), C_III_III's term_sym1/2 = -direct/-exch."""
    O = Foo.shape[1]
    Va = Vt.shape[2]
    direct = np.zeros((O, O, Va))
    exch = np.zeros((O, O, Va))
    for i in range(O):
        M = np.einsum('Qk,kla->Qla', Foo[:, i, :], Vt, optimize=True)
        direct[i] = np.einsum('Qjl,Qla->ja', Foo, M, optimize=True)
        P = np.einsum('Ql,kla->Qka', Foo[:, i, :], Vt, optimize=True)
        exch[i] = np.einsum('Qjk,Qka->ja', Foo, P, optimize=True)
    return direct, exch


def _u_I_ring_contract_streamed_bare(B, O, V, norb, eps_o, eps_v,
                                     occ_chunk=4, q_chunk=8, dh=None, dp=None):
    """Streamed twin of _u_I_ring_contract_df: identical (term1, term2)
    output, (O,norb,V) each, but never materializes W1/W2 (or g_oovv) as
    full (O,O,V,V) arrays -- built occupied-chunk by occupied-chunk from
    B_aa. dh/dp: optional EN hh/pp shift matrices, see
    _u2_2h1p_amplitude_chunks's docstring for scope."""
    occ = np.arange(O)
    vir = np.arange(O, norb)
    Bo_full = B[:, occ, :]
    Bv_full = B[:, vir, :]
    Boo = B[:, occ][:, :, occ]
    Bvo = B[:, vir][:, :, occ]
    term1 = np.zeros((O, norb, V))
    term2 = np.zeros((O, norb, V))
    for lo, hi, t_same, t_opp, W1c, W2c, A_mix, X_ijcd in _u2_2h1p_amplitude_chunks(
            B, O, V, norb, eps_o, eps_v, occ_chunk, dh=dh, dp=dp):
        for i_local in range(hi - lo):
            i = lo + i_local
            for start in range(0, Boo.shape[0], q_chunk):
                sl = slice(start, start + q_chunk)
                F1 = _cached_einsum('kca,qk->qca', W1c[i_local], Boo[sl, i], optimize=True)
                term1[i] += _cached_einsum('qcp,qca->pa', Bv_full[sl], F1, optimize=True)
                G2 = _cached_einsum('kca,qck->qa', W2c[i_local], Bvo[sl], optimize=True)
                term2[i] += _cached_einsum('qp,qa->pa', Bo_full[sl, i], G2, optimize=True)
    return term1, term2


def _corr_kl_Ip_streamed_df(Bo_full, Boo, X_akl, q_chunk=32):
    """DF-direct replacement for U_Ip's one-time hole-ladder correction

        0.5 * einsum('akl,klpi->api', X_akl, g_klpi_sym)

    with g_klpi_sym[k,l,p,i] = sum_Q B[Q,k,p]B[Q,l,i] + B[Q,k,i]B[Q,l,p]
    -- so g_oo_po/g_oo_op ((O,O,norb,O), the last U-side rank-4 integrals)
    need not exist on the streamed path. Q-slab tensordots; the (a,k,q,i)
    transient is bounded by q_chunk and freed per slab. Returns (V,norb,O)
    in the (a,p,i) order the caller consumes."""
    naux = Bo_full.shape[0]
    V, O = X_akl.shape[0], X_akl.shape[1]
    norb = Bo_full.shape[2]
    t1 = np.zeros((V, O, norb))
    t2 = np.zeros((V, O, norb))
    for st in range(0, naux, q_chunk):
        sl = slice(st, st + q_chunk)
        H1 = np.tensordot(X_akl, Boo[sl], axes=([2], [1]))          # (a,k,q,i)
        t1 += np.tensordot(H1, Bo_full[sl], axes=([1, 2], [1, 0]))  # (a,i,p)
        H2 = np.tensordot(X_akl, Boo[sl], axes=([1], [1]))          # (a,l,q,i)
        t2 += np.tensordot(H2, Bo_full[sl], axes=([1, 2], [1, 0]))  # (a,i,p)
    return 0.5 * (t1 + t2).transpose(0, 2, 1)


def _u_Ip_pv_vp_contract_streamed_bare(B, O, V, norb, eps_o, eps_v,
                                       occ_chunk=4, q_chunk=32, dh=None, dp=None):
    """Streamed twin of _u_Ip_pv_vp_contract_df: identical (term1, term2)
    output, (V,norb,O) each, never materializing W1p/W2p (or g_vvoo) as full
    (V,V,O,O) arrays. dh/dp: optional EN hh/pp shift matrices, see
    _u2_2h1p_amplitude_chunks's docstring for scope. Verified against
    _u_Ip_pv_vp_contract_df fed the full-array W1p/W2p to 1.1e-13."""
    occ = np.arange(O)
    vir = np.arange(O, norb)
    Bo_full = B[:, occ, :]
    Bv_full = B[:, vir, :]
    Bv_pp = B[:, vir][:, :, vir]
    Bov_full = B[:, occ][:, :, vir]
    term1 = np.zeros((V, norb, O))
    term2 = np.zeros((V, norb, O))
    for lo, hi, tp_same, tp_opp, W1p_c, W2p_c, X_abkl in _u2_2p1h_amplitude_chunks(
            B, O, V, norb, eps_o, eps_v, occ_chunk, dh=dh, dp=dp):
        for start in range(0, Bo_full.shape[0], q_chunk):
            sl = slice(start, start + q_chunk)
            term1[:, :, lo:hi] += _cached_einsum('ackI,Qkp,Qac->apI', W1p_c, Bo_full[sl], Bv_pp[sl],
                                                 optimize=True)
            term2[:, :, lo:hi] += _cached_einsum('ackI,Qkc,Qap->apI', W2p_c, Bov_full[sl], Bv_full[sl],
                                                 optimize=True)
    return term1, term2


# ============== U-block builders: streamed vs unstreamed ==============
# build_matrix_free_operator dispatches to exactly ONE of the two functions
# below (on its _u_streamable flag) to construct the explicit U_I/U_Ip
# matrices and the four apply_U_* closures. They used to live inline in that
# method as three sequential def-sites relying on Python's "last def wins"
# (base DF defs, then a streamed override, then a materialized override) --
# split out so each path's dependency surface is explicit in its parameter
# list, at the accepted price of duplicating the U_I/U_Ip skeletons and the
# _2h1p_specs/_2p1h_specs tables between the two.
def _build_u_blocks_streamed(B, O, V, norb, eps_o, eps_v, dh, dp, dens,
                             Bo_full, Bv_full, Bov_full, Boo_full, Bvo_full, Bv_pp,
                             iu_o, ju_o, iu_v, ju_v, ints):
    """U blocks + appliers for the STREAMED path (_u_streamable): DF factors
    present, no screening, no adc2x, dress bare or hh/pp spin-adapted.
    Never materializes any O^2V^2 amplitude or integral -- everything is
    rebuilt occupied-chunk by occupied-chunk from B every call, via
    _u2_2h1p_amplitude_chunks/_u2_2p1h_amplitude_chunks (dh/dp: the small
    EN shift matrices, None for bare).

    ints keys read: g_ii_U, g_ii, g_aa_U, g_aa. (g_oo_po/g_oo_op are NOT
    read: every g_klpi consumer is DF-direct here, so no rank-4 integral
    survives on this path.)
    Returns (U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj,
             apply_U_2p1h_fwd, apply_U_2p1h_adj).
    """
    s32 = math.sqrt(1.5)
    s12 = math.sqrt(0.5)
    nP_o = O * (O - 1) // 2
    nP_v = V * (V - 1) // 2
    nI = O * V
    nIp = O * V
    g_ii_U, g_ii = ints['g_ii_U'], ints['g_ii']
    g_aa_U, g_aa = ints['g_aa_U'], ints['g_aa']

    # ===================== U_I =====================
    # (adc2x is excluded from the streamed path, so the U^(2) pieces are
    # built unconditionally here.)
    u_I = g_ii_U[:, :, O:].copy()  # 1st-order piece
    # i=j slice: purely opposite-spin (the aaaa amplitude is
    # antisymmetric in its occupied pair, hence zero here), and its hh
    # shift is the <ii|ii> diagonal.
    g_iicd = g_ii[:, O:, O:]
    X_icd = g_iicd / dens.build_icd()
    term_pv, term_vp = _u_pv_vp_contract_df(Bv_full, Bv_pp, X_icd, 'i', 'i')
    u_I -= 0.5 * (term_pv + term_vp)
    # aaaa/abab amplitudes on 'ikca'; W1/W2/A_anti/A_mix are their
    # unique linear combinations (see _u2_spin_amplitudes).
    # never materializes t_same/t_opp/W1/W2 (or g_oovv) as full
    # (O,O,V,V) -- built occupied-chunk by occupied-chunk from
    # B_aa. See this section's banner comment above
    # _u2_2h1p_amplitude_chunks.
    r1_I, r2_I = _u_I_ring_contract_streamed_bare(
        B, O, V, norb, eps_o, eps_v, dh=dh, dp=dp)
    u_I += r1_I
    u_I += r2_I
    U_I = u_I.transpose(1, 0, 2).reshape(norb, nI)

    # ============ U_II / U_III -- MATRIX-FREE, STREAMED ============
    # See the unstreamed builder's banner for the six-terms-to-four
    # collapse and the _2h1p_specs tabulation; identical algebra here,
    # with the amplitudes rebuilt per chunk instead of held in closures.
    # (scale, ring amplitude, ring sign, Y_mix sign, antisymmetric?)
    _2h1p_specs = ((s32, 'A_anti', -1.0, +1.0, True),
                   (s12, 'W1', +1.0, -1.0, False))
    q_chunk_2h1p = 32

    # Never materializes t_same/W1/A_mix/X_ijcd (or g_oovv) as full
    # (O,O,V,V) arrays -- rebuilt occupied-chunk by occupied-chunk
    # from B_aa, every matvec. See the banner comment above
    # _u2_2h1p_amplitude_chunks.
    def apply_U_2h1p_fwd(z_II, z_III):
        y = np.zeros(norb)
        Zs = [_u_2h1p_unfold(z, iu_o, ju_o, O, V, anti)
             for (_, _, _, _, anti), z in zip(_2h1p_specs, (z_II, z_III))]
        for (sc, _, _, _, _), Zfull in zip(_2h1p_specs, Zs):
            M = _cached_einsum('Qja,ija->Qi', Bov_full, Zfull, optimize=True)
            y += sc * _b_chunk_einsum('Qip,Qi->p', (Bo_full, M), (True, True), Bo_full)
        for lo, hi, t_same_c, t_opp_c, W1c, W2c, A_mix_c, X_ijcd_c in _u2_2h1p_amplitude_chunks(
                B, O, V, norb, eps_o, eps_v, 4, dh=dh, dp=dp):
            _amps_c = {'A_anti': t_same_c, 'W1': W1c}
            for (sc, akey, qsign, ysign, anti), Zfull in zip(_2h1p_specs, Zs):
                Z = Zfull[lo:hi]
                for st in range(0, Bv_pp.shape[0], q_chunk_2h1p):
                    sl = slice(st, st + q_chunk_2h1p)
                    T = _cached_einsum('Qda,ija->Qijd', Bv_pp[sl], Z, optimize=True)
                    P = _cached_einsum('ijcd,Qijd->Qc', X_ijcd_c, T, optimize=True)
                    y -= sc * _cached_einsum('Qcp,Qc->p', Bv_full[sl], P, optimize=True)
                S = _cached_einsum('ikca,ija->kcj', _amps_c[akey], Z, optimize=True)
                R = _cached_einsum('Qjk,kcj->Qc', Boo_full, S, optimize=True)
                y += qsign * sc * _b_chunk_einsum('Qcp,Qc->p', (Bv_full, R), (True, True), Bv_full)
                S4 = _cached_einsum('ikca,ija->kcj', A_mix_c, Z, optimize=True)
                R4 = _cached_einsum('Qck,kcj->Qj', Bvo_full, S4, optimize=True)
                y += ysign * sc * _b_chunk_einsum('Qjp,Qj->p', (Bo_full, R4), (True, True), Bo_full)
        return y

    def apply_U_2h1p_adj(z_p):
        bzo = _b_chunk_einsum('Qip,p->Qi', (Bo_full, z_p), (True, False),
                              Bo_full, (Bo_full.shape[0], O))
        bzv = _b_chunk_einsum('Qcp,p->Qc', (Bv_full, z_p), (True, False),
                              Bv_full, (Bv_full.shape[0], V))
        gz1 = _cached_einsum('Qi,Qja->ija', bzo, Bov_full, optimize=True)
        gvo = _cached_einsum('Qc,Qjk->cjk', bzv, Boo_full, optimize=True)
        gvh = _cached_einsum('Qck,Qj->cjk', Bvo_full, bzo, optimize=True)
        tp = np.zeros((O, O, V))
        y1 = np.zeros((O, O, V))
        q3_by_key = {'A_anti': np.zeros((O, O, V)), 'W1': np.zeros((O, O, V))}
        for lo, hi, t_same_c, t_opp_c, W1c, W2c, A_mix_c, X_ijcd_c in _u2_2h1p_amplitude_chunks(
                B, O, V, norb, eps_o, eps_v, 4, dh=dh, dp=dp):
            _amps_c = {'A_anti': t_same_c, 'W1': W1c}
            for st in range(0, Bv_pp.shape[0], q_chunk_2h1p):
                sl = slice(st, st + q_chunk_2h1p)
                Xb = _cached_einsum('ijcd,Qc->Qijd', X_ijcd_c, bzv[sl], optimize=True)
                tp[lo:hi] += _cached_einsum('Qijd,Qda->ija', Xb, Bv_pp[sl], optimize=True)
            y1[lo:hi] = _cached_einsum('ikca,cjk->ija', A_mix_c, gvh, optimize=True)
            for akey in ('A_anti', 'W1'):
                q3_by_key[akey][lo:hi] = _cached_einsum(
                    'ikca,cjk->ija', _amps_c[akey], gvo, optimize=True)
        out = []
        for sc, akey, qsign, ysign, anti in _2h1p_specs:
            sgn = -1.0 if anti else 1.0
            tot = gz1 - tp + qsign * q3_by_key[akey] + ysign * y1
            tot = sc * (tot + sgn * tot.transpose(1, 0, 2))
            out.append(tot[iu_o, ju_o])
        return out

    # ===================== U_Ip =====================
    u_Ip = g_aa_U[:, :, :O].transpose(2, 1, 0).copy()  # 1st-order piece
    g_aakl = g_aa[:, :O, :O]
    # a=b slice: mirror of X_icd -- purely opposite-spin, pp shift is
    # the <aa|aa> diagonal.
    X_akl = g_aakl / dens.build_akl()
    # hole-ladder correction DF-direct (g_klpi_sym never materialized;
    # see _corr_kl_Ip_streamed_df's docstring).
    corr_kl_Ip = _corr_kl_Ip_streamed_df(Bo_full, Boo_full, X_akl)
    u_Ip += corr_kl_Ip.transpose(2, 1, 0)
    # never materializes tp_same/tp_opp/W1p/W2p (or g_vvoo) as
    # full (V,V,O,O) -- built occupied-chunk by occupied-chunk
    # from B_aa. See the banner comment above
    # _u2_2h1p_amplitude_chunks.
    t1_Ip, t2_Ip = _u_Ip_pv_vp_contract_streamed_bare(
        B, O, V, norb, eps_o, eps_v, dh=dh, dp=dp)
    u_Ip += t1_Ip.transpose(2, 1, 0)
    u_Ip += t2_Ip.transpose(2, 1, 0)
    U_Ip = u_Ip.transpose(1, 0, 2).reshape(norb, nIp)

    # ============ U_IIp / U_IIIp -- MATRIX-FREE, STREAMED ============
    # See the unstreamed builder's banner for the antisym/sym collapse and
    # _2p1h_specs tabulation. The g_klpi_anti/g_klpi_sym terms are applied
    # DF-direct below (rank-3 chains through Bo_full/Boo_full), with the
    # +/- of the exchange piece carried by _gk_sign -- so g_oo_po/g_oo_op
    # ((O,O,norb,O)) never exist on this path.
    # (scale, g_klpi, ring amplitude, mix sign, antisym?)
    _2p1h_specs = ((s32, 'anti', 'A_anti_p', -1.0, True),
                   (s12, 'sym', 'W1p', +1.0, False))
    _gk_sign = {'anti': -1.0, 'sym': +1.0}

    # Never materializes tp_same/W1p/W2p/X_abkl (or g_vvoo) as full
    # (V,V,O,O) arrays -- rebuilt occupied-chunk by occupied-chunk
    # from B_aa, every matvec, streaming over the axis every
    # consumer here treats as the amplitude's own batch/output axis
    # (axis3='l', renamed 'i' at consumption). TWO DF builds per
    # chunk (direct+exchange) -- unlike the 2h1p twin's one -- since
    # g_vvoo's transpose swaps the SAME two occupied axes being
    # chunked here.
    def apply_U_2p1h_fwd(z_IIp, z_IIIp):
        y = np.zeros(norb)
        Zs = [_u_2p1h_unfold(z, iu_v, ju_v, O, V, anti)
             for (_, _, _, _, anti), z in zip(_2p1h_specs, (z_IIp, z_IIIp))]
        for (sc, _, _, _, _), Zfull in zip(_2p1h_specs, Zs):
            M = _cached_einsum('Qbi,iab->Qa', Bvo_full, Zfull, optimize=True)
            y += sc * _b_chunk_einsum('Qap,Qa->p', (Bv_full, M), (True, True), Bv_full)
        for lo, hi, tp_same_c, tp_opp_c, W1p_c, W2p_c, X_abkl_c in _u2_2p1h_amplitude_chunks(
                B, O, V, norb, eps_o, eps_v, 4, dh=dh, dp=dp):
            _amps_c = {'A_anti_p': tp_same_c, 'W1p': W1p_c}
            A_mix_p_c = W2p_c
            for (sc, gkey, akey, hsign, anti), Zfull in zip(_2p1h_specs, Zs):
                # X_abkl's l (chunked) and Z's own 'i' axis are
                # UNRELATED indices sharing the letter 'i' in the
                # source (X_abkl has no 'i' of its own here), so Z
                # stays FULL for this term -- the l-restriction rides on
                # P's own l-axis. The former einsum against a materialized
                # g_klpi[:, lo:hi] is replaced by its two DF halves
                # (g_klpi[k,l,p,i] = sum_Q B[Q,k,p]B[Q,l,i] +/- (k<->l)),
                # each a rank-3 GEMM chain.
                P = _cached_einsum('abkl,iab->kli', X_abkl_c, Zfull, optimize=True)
                R1 = _cached_einsum('Qli,kli->Qk', Boo_full[:, lo:hi, :], P, optimize=True)
                yA = _b_chunk_einsum('Qkp,Qk->p', (Bo_full, R1), (True, True), Bo_full)
                R2 = _cached_einsum('Qki,kli->Ql', Boo_full, P, optimize=True)
                yB = _b_chunk_einsum('Qlp,Ql->p', (Bo_full[:, lo:hi], R2), (True, True), Bo_full)
                y += 0.5 * sc * (yA + _gk_sign[gkey] * yB)
                # Here 'i' genuinely IS the amplitude's l-axis
                # (shared with Z's first axis in this contraction). 'i' is
                # then fully summed downstream, so contract it EARLY
                # together with 'a': '->ckb' is a single clean GEMM,
                # whereas carrying i as a batch index ('->ckib') would hit
                # numpy's single-threaded scalar-kernel path.
                Z = Zfull[lo:hi]
                Y = _cached_einsum('acki,iab->ckb', _amps_c[akey], Z, optimize=True)
                R = _b_chunk_einsum('Qbc,ckb->Qk', (Bv_pp, Y), (True, False),
                                    Bv_pp, (Bv_pp.shape[0], O))
                y += sc * _b_chunk_einsum('Qk,Qkp->p', (R, Bo_full), (True, True), Bo_full)
                Y2 = _cached_einsum('acki,iab->ckb', A_mix_p_c, Z, optimize=True)
                R2 = _b_chunk_einsum('Qkc,ckb->Qb', (Bov_full, Y2), (True, False),
                                     Bov_full, (Bov_full.shape[0], V))
                y += hsign * sc * _b_chunk_einsum('Qb,Qbp->p', (R2, Bv_full), (True, True), Bv_full)
        return y

    def apply_U_2p1h_adj(z_p):
        vidx_abs = np.arange(O, norb)
        oidx = np.arange(O)
        gz, gzr, gzm = _u_2p1h_zparts_df(B, O, norb, vidx_abs, oidx, z_p)
        # gkz[k,l,i] = sum_p g_klpi[k,l,p,i] z_p, DF-direct: contract z_p
        # into the p-carrying factor FIRST (boz), then close with Boo_full
        # -- rank-3 throughout, no g_klpi.
        boz = _b_chunk_einsum('Qkp,p->Qk', (Bo_full, z_p), (True, False),
                              Bo_full, (Bo_full.shape[0], O))
        _gkz_direct = _cached_einsum('Qk,Qli->kli', boz, Boo_full, optimize=True)
        _gkz_exch = _cached_einsum('Qki,Ql->kli', Boo_full, boz, optimize=True)
        gkz = {gkey: _gkz_direct + _gk_sign[gkey] * _gkz_exch
              for gkey in ('anti', 'sym')}
        abkl_term = {akey: np.zeros((V, V, O)) for akey in ('A_anti_p', 'W1p')}
        Gz = {akey: np.zeros((V, V, O)) for akey in ('A_anti_p', 'W1p')}
        Hz = {akey: np.zeros((V, V, O)) for akey in ('A_anti_p', 'W1p')}
        for lo, hi, tp_same_c, tp_opp_c, W1p_c, W2p_c, X_abkl_c in _u2_2p1h_amplitude_chunks(
                B, O, V, norb, eps_o, eps_v, 4, dh=dh, dp=dp):
            _amps_c = {'A_anti_p': tp_same_c, 'W1p': W1p_c}
            A_mix_p_c = W2p_c
            for sc, gkey, akey, hsign, anti in _2p1h_specs:
                # term2 (abkl,kli->abi): k,l BOTH summed -> accumulate
                # over l-chunks, slicing gkz on ITS OWN l-axis (axis
                # 1, 'kli') to match.
                abkl_term[akey] += _cached_einsum(
                    'abkl,kli->abi', X_abkl_c, gkz[gkey][:, lo:hi, :], optimize=True)
                # term3/4 (acki,kbc->abi): c,k summed, a & i(=l)
                # OUTPUT -> fill this l-chunk's output slice directly.
                Gz[akey][:, :, lo:hi] = _cached_einsum('acki,kbc->abi', _amps_c[akey], gzr, optimize=True)
                Hz[akey][:, :, lo:hi] = _cached_einsum('acki,kbc->abi', A_mix_p_c, gzm, optimize=True)
        out = []
        for sc, gkey, akey, hsign, anti in _2p1h_specs:
            sgn = -1.0 if anti else 1.0
            tot = sc * (gz + sgn * gz.transpose(1, 0, 2))
            tot = tot + sc * abkl_term[akey]
            tot = tot + sc * (Gz[akey] + sgn * Gz[akey].transpose(1, 0, 2))
            tot = tot + hsign * sc * (Hz[akey] + sgn * Hz[akey].transpose(1, 0, 2))
            out.append(tot.transpose(2, 0, 1)[:, iu_v, ju_v])
        return out

    return U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj, apply_U_2p1h_fwd, apply_U_2p1h_adj


def _pp_ladder_matvec_df(Bv, Vmat, iu_v, ju_v, V, antisymmetric, q_chunk=32):
    """
    Matrix-free DF/RI application of the pp-ladder ("Term0")
    piece of the C_IIp_IIp ('anti', antisymmetric=True) / C_IIIp_IIIp
    ('sym1+sym2', antisymmetric=False) 2p1h-2p1h coupling blocks

    Bv: (naux, V, V), B restricted to the virtual-virtual block
    (B_aa[:, vir, vir]). Vmat: (O, nP_v) trial-vector segment, same layout
    apply_C_IIp_IIp/apply_C_IIIp_IIIp already use. Returns (O, nP_v).
    """
    O_ = Vmat.shape[0]
    M = np.zeros((O_, V, V))
    M[:, iu_v, ju_v] = Vmat
    M[:, ju_v, iu_v] = -Vmat if antisymmetric else Vmat
    naux = Bv.shape[0]
    X = np.zeros((O_, V, V))
    for start in range(0, naux, q_chunk):
        BQ = Bv[start:start + q_chunk]
        Y = np.einsum('qac,icd->qiad', BQ, M, optimize=True)
        X += np.einsum('qiad,qbd->iab', Y, BQ, optimize=True)
    return X[:, iu_v, ju_v]


def build_operator(s, nocc, static_correction=None, couplings=True):
    """(aop, diag, dims) for the DF route. couplings=False: U = 0, i.e. F
    and the satellite block K + C only, nothing of U built or applied."""
    if s.W_chemist is not None:
        raise ValueError(
            'dense W_chemist screening is a dense-integral route; with DF '
            'integrals use W_aux (screened DF factor), or drop B_aa')
    eps, eri_chemist = s.eps, s.eri
    norb = s.norb
    O, V = nocc, norb - nocc
    eps_o, eps_v = eps[:O], eps[O:]

    s32 = math.sqrt(1.5)
    s12 = math.sqrt(0.5)
    s2 = math.sqrt(2.0)
    s3 = math.sqrt(3.0)

    d = s.dimensions(nocc)
    nI, nII, nIII, n2h1p = d['nI'], d['nII'], d['nIII'], d['n2h1p']
    nIp, nIIp, nIIIp = d['nIp'], d['nIIp'], d['nIIIp']
    nH = d['nH']
    nP_o = O * (O - 1) // 2
    nP_v = V * (V - 1) // 2

    oI = norb
    oII = oI + nI
    oIII = oII + nII
    oIp = norb + n2h1p
    oIIp = oIp + nIp
    oIIIp = oIIp + nIIp

    iu_o, ju_o = np.triu_indices(O, k=1)
    iu_v, ju_v = np.triu_indices(V, k=1)
    oidx = np.arange(O)
    vidx_local = np.arange(V)
    vidx_abs = np.arange(O, norb)

    # ===================== F block =====================
    if static_correction is None:
        F = np.diag(eps)
    else:
        F = np.diag(eps) + static_correction

    _u_streamable = not s._is_adc2x and _dress_is_streamable(s.u2_denom_dress)

    # Common integral slices, DF-gathered; the streamed paths skip every
    # slice they rebuild chunkwise from B themselves.
    blk = _build_g_blocks_df(s.B_aa, O, V, norb,
                             need_vvvv=False, need_vv_pv_vp=False, need_vv_pp=False,
                             need_ovpv_ovvp=False,
                             need_oo_pp=False, need_vopo_voho=False,
                             need_oovv_vvoo=(not _u_streamable
                                             and not s._is_adc2x),
                             need_voov_vovo=False,
                             need_ovov_ovvo=False,
                             need_oo_po_op=(not _u_streamable
                                            and not s._is_adc2x),
                             need_oooo=False)
    g_oovv, g_vvoo = blk.get('g_oovv'), blk.get('g_vvoo')
    g_oo_po, g_oo_op = blk.get('g_oo_po'), blk.get('g_oo_op')
    g_ii, g_aa = blk['g_ii'], blk['g_aa']

    allo_full = np.arange(norb)
    Bv_pp = _b_factor(s.B_aa, vidx_abs, vidx_abs, s.B_aa[:, O:, O:])
    Bv_full = _b_factor(s.B_aa, vidx_abs, allo_full, s.B_aa[:, O:, :])
    Bvo_full = _b_factor(s.B_aa, vidx_abs, oidx, s.B_aa[:, O:, :O])
    Bo_full = _b_factor(s.B_aa, oidx, allo_full, s.B_aa[:, :O, :])
    Bov_full = _b_factor(s.B_aa, oidx, vidx_abs, s.B_aa[:, :O, O:])
    Boo_full = _b_factor(s.B_aa, oidx, oidx, s.B_aa[:, :O, :O])

    g_oovv_T = g_oovv.transpose(0, 1, 3, 2) if g_oovv is not None else None
    g_vvoo_T = g_vvoo.transpose(0, 1, 3, 2) if g_vvoo is not None else None

    # EN shifts are built DF-natively from B_aa (bare by construction)
    g_ii_U, g_aa_U = g_ii, g_aa

    # ===== W_aux screening: screened DF factor M_aa =====
    M_vv_screened = None
    M_aa = None
    if s.W_aux is not None:
        waux = np.asarray(s.W_aux)
        # Screened DF factor via eigendecomposition of waux = (1-chi0)^-1
        # (symmetric; mathematically PD with eigenvalues in (0,1] at
        # w=0). W_pqrs = sum_R M_aa[R,p,r] M_aa[R,q,s]
        lam_w, Q_w = np.linalg.eigh(waux)
        if lam_w.min() < -1e-10:
            raise ValueError(
                f"W_aux is not positive semidefinite (min eigenvalue "
                f"{lam_w.min():.3e}): RPA instability -- screened MCDE "
                f"is not defined here")
        M_aa = np.einsum('PR,Ppq->Rpq', Q_w * np.sqrt(np.clip(lam_w, 0.0, None)),
                         s.B_aa, optimize=True)
        M_vv_screened = M_aa[:, O:, O:]
        g_ii_C = None
        g_aa_C = None
    else:
        g_ii_C = g_ii
        g_aa_C = g_aa

    # The four small C^(1) ladder slices -- from M_aa under W_aux, else bare.
    if M_aa is not None:
        # W[i,i,k,l] = sum_R M[R,i,k] M[R,i,l]; (O,O,O) is tiny.
        _g_ii3 = _g_diag_df(M_aa, oidx, oidx, oidx)

        # (O,O)  = W[i,i,k,k]
        g_ii_kk = _g_ii3[:, oidx, oidx]

        # (O,nP_o) = W[i,i,k_r,l_r]
        g_ii_pair = _g_ii3[:, iu_o, ju_o]
        Mvv_w = M_vv_screened
        g_aabb = np.einsum('Rab,Rab->ab', Mvv_w, Mvv_w, optimize=True)  # (V,V) = W[a,a,b,b]
        # W[a,a,c_r,d_r], a-chunked so the transient stays (chunk,V,V)
        # instead of a V^3 _g_diag_df over the whole virtual space.
        g_aa_cd = np.empty((V, nP_v))
        for _alo in range(0, V, 32):
            _ahi = min(_alo + 32, V)
            _t = np.einsum('Rac,Rad->acd', Mvv_w[:, _alo:_ahi, :],
                           Mvv_w[:, _alo:_ahi, :], optimize=True)
            g_aa_cd[_alo:_ahi] = _t[:, iu_v, ju_v]
    else:
        g_ii_kk = g_ii_C[:, oidx, oidx]
        g_ii_pair = g_ii_C[:, iu_o, ju_o]
        g_aabb = g_aa_C[:, vidx_abs, vidx_abs]
        g_aa_cd = g_aa_C[:, vidx_abs[iu_v], vidx_abs[ju_v]]

    # EN bare-integral carriers: never gathered on the DF path -- the shifts
    # are built DF-natively from B_aa inside restricted_channel_shifts.
    g_oooo_bare = g_vvvv_bare = None
    g_vovo_bare = g_voov_bare = None
    # See build_supermatrix's identical block for the layout key and the
    # spin-resolved EN rationale (kept in sync deliberately).
    _deltas = (None, None, None)
    if s.u2_denom_dress and not s._is_adc2x:
        _deltas = restricted_channel_shifts(
            s.u2_denom_dress, s.B_aa,
            g_oooo_bare, g_vvvv_bare, O, vidx_abs,
            g_vovo=g_vovo_bare, g_voov=g_voov_bare)
    dens = EpsteinNesbetDenominators(eps_o, eps_v, *_deltas)

    # Small (O,O)/(V,V) hh/pp shift matrices for the streamed U-block
    _dh_c = _deltas[0][0] if _deltas[0] is not None else None
    _dp_c = _deltas[1][0] if _deltas[1] is not None else None

    # ============ U blocks: streamed, or DF-applier fallback ============
    _u_ints = dict(
        g_ii_U=g_ii_U, g_ii=g_ii, g_aa_U=g_aa_U, g_aa=g_aa)
    if not couplings:
        (U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj,
         apply_U_2p1h_fwd, apply_U_2p1h_adj) = _no_u_blocks(norb, O, V, nP_o, nP_v)
    elif _u_streamable:
        (U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj,
         apply_U_2p1h_fwd, apply_U_2p1h_adj) = _build_u_blocks_streamed(
            s.B_aa, O, V, norb, eps_o, eps_v, _dh_c, _dp_c, dens,
            Bo_full, Bv_full, Bov_full, Boo_full, Bvo_full, Bv_pp,
            iu_o, ju_o, iu_v, ju_v, _u_ints)
    else:
        _u_ints.update(
            g_oo_po=g_oo_po, g_oo_op=g_oo_op,
            g_oovv=g_oovv, g_oovv_T=g_oovv_T, g_vvoo=g_vvoo, g_vvoo_T=g_vvoo_T,
            g_vopo=None, g_voho=None, g_ovpv=None, g_ovvp=None,
            g_vv_pv=None, g_vv_vp=None,
            g_oo_pp_U=None, g_vv_pp_U=None)
        (U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj,
         apply_U_2p1h_fwd, apply_U_2p1h_adj) = _build_u_blocks_unstreamed(
            s.B_aa, O, V, norb, eps_o, eps_v, dens,
            Bo_full, Bv_full, Bov_full, Boo_full, Bvo_full, Bv_pp,
            iu_o, ju_o, iu_v, ju_v, _u_ints,
            is_adc2x=s._is_adc2x, use_materialized=False, w_chemist=None)
    # ===================== K diagonals =====================
    e_I = (2 * eps_o[:, None] - eps_v[None, :]).ravel()
    e_II = (eps_o[iu_o, None] + eps_o[ju_o, None] - eps_v[None, :]).ravel()
    e_III = e_II.copy()
    e_Ip = (2 * eps_v[None, :] - eps_o[:, None]).ravel()
    e_IIp_pair = eps_v[iu_v] + eps_v[ju_v]
    e_IIp = (e_IIp_pair[None, :] - eps_o[:, None]).ravel()
    e_IIIp = e_IIp.copy()

    # ==================== C-block sigma-vector pieces (streamed) ====================
    _B_scr = M_aa if M_aa is not None else s.B_aa

    g_ciia = _g_diag_mid_df(s.B_aa, vidx_abs, oidx, vidx_abs)
    g_ciai = _g_diag_outer_df(_B_scr, vidx_abs, oidx, vidx_abs)
    inner_I = -(g_ciia - 2 * g_ciai).transpose(1, 2, 0)

    def apply_C_I_I(Vmat):
        term1 = -g_ii_kk @ Vmat
        term2 = np.einsum('iac,ic->ia', inner_I, Vmat, optimize=True)
        return term1 + term2

    _Foo_scr = _B_scr[:, :O, :O]
    def unfold_pair(Vmat, iu, ju, O_, Vv):
        Vfull = np.zeros((O_, O_, Vv))
        Vfull[iu, ju, :] = Vmat
        Vfull[ju, iu, :] = -Vmat
        return Vfull

    def upper_pair(Vmat, iu, ju, O_, Vv):
        Vup = np.zeros((O_, O_, Vv))
        Vup[iu, ju, :] = Vmat
        return Vup
    # Streamed replacement for the eight M1a/M1b-derived appliers above:
    def _apply_C_2h1p_coupled(z_I, z_II, z_III):
        Vfull_II = unfold_pair(z_II, iu_o, ju_o, O, V)
        Vupper_II = upper_pair(z_II, iu_o, ju_o, O, V)
        Vupper_III = upper_pair(z_III, iu_o, ju_o, O, V)

        T1 = np.zeros((O, V, O)); T2 = np.zeros((O, V, O))
        BV1 = np.zeros((O, O, V)); BV2 = np.zeros((O, O, V))
        term1_I_II = np.zeros((O, V)); term2_I_II = np.zeros((O, V))
        W_II_I = np.zeros((O, O, V))
        term1_I_III = np.zeros((O, V)); term2_I_III = np.zeros((O, V))
        Wb_III_I = np.zeros((O, O, V))
        Tu_III = np.zeros((O, O, V)); Tl_III = np.zeros((O, O, V))
        Tu_II = np.zeros((O, O, V)); Tl_II = np.zeros((O, O, V))

        for lo, hi, m1a_c, m1b_c in _c2h1p_df_chunks(s.B_aa, O, V, norb, B_w=_B_scr):
            bracket_c = m1a_c - 0.5 * m1b_c
            M1_c = -m1a_c + 1.5 * m1b_c
            M2_c = -1.5 * m1b_c + m1a_c

            Vc_II_full = Vfull_II[:, lo:hi, :]
            T1 += np.einsum('iamb,kmb->iak', M1_c, Vc_II_full, optimize=True)
            T2 += np.einsum('jamb,kmb->jak', M2_c, Vc_II_full, optimize=True)

            Vc_III_a = Vupper_III[:, lo:hi, :]
            BV1 += np.einsum('xamb,Kmb->Kxa', bracket_c, Vc_III_a, optimize=True)
            BV2 += np.einsum('xamb,mKb->Kxa', bracket_c, Vupper_III[lo:hi], optimize=True)
            term1_I_III += np.einsum('ialc,ilc->ia', bracket_c, Vc_III_a, optimize=True)
            term2_I_III += np.einsum('iaxc,xic->ia', bracket_c, Vupper_III[lo:hi], optimize=True)
            Wb_III_I[:, lo:hi, :] = np.einsum('iaxc,ia->ixc', bracket_c, z_I, optimize=True)

            Vc_II_a = Vupper_II[:, lo:hi, :]
            term1_I_II += np.einsum('ialc,ilc->ia', m1b_c, Vc_II_a, optimize=True)
            term2_I_II += np.einsum('iaxc,xic->ia', m1b_c, Vupper_II[lo:hi], optimize=True)
            W_II_I[:, lo:hi, :] = np.einsum('iaxc,ia->ixc', m1b_c, z_I, optimize=True)
            Tu_III += np.einsum('xanc,Knc->Kxa', m1b_c, Vc_III_a, optimize=True)
            Tl_III += np.einsum('xamc,mKc->Kxa', m1b_c, Vupper_III[lo:hi], optimize=True)
            Tu_II += np.einsum('xanc,Knc->Kxa', m1b_c, Vc_II_a, optimize=True)
            Tl_II += np.einsum('xamc,mKc->Kxa', m1b_c, Vupper_II[lo:hi], optimize=True)

        # hh-ladder terms DF-direct (no g_oooo/g_oooo_anti on this
        # path): direct-exch IS the g_oooo_anti contraction, and
        # C_III_III's two sym terms are the pair itself -- see
        # _oooo_contract_df.
        d_II, e_II_ = _oooo_contract_df(_Foo_scr, Vfull_II)
        T0 = d_II - e_II_
        c_II_II = -0.5 * T0[iu_o, ju_o, :] + T1[iu_o, :, ju_o] + T2[ju_o, :, iu_o]

        d_III, e_III_ = _oooo_contract_df(_Foo_scr, Vupper_III)
        term_sym1 = -d_III
        term_sym2 = -e_III_
        groupA = BV1[iu_o, ju_o, :] + BV2[iu_o, ju_o, :]
        groupB = BV1[ju_o, iu_o, :] + BV2[ju_o, iu_o, :]
        c_III_III = term_sym1[iu_o, ju_o, :] + term_sym2[iu_o, ju_o, :] + groupA + groupB

        c_I_II = -s32 * (term1_I_II - term2_I_II)
        c_II_I = -s32 * (W_II_I[iu_o, ju_o, :] - W_II_I[ju_o, iu_o, :])

        c_I_III = -s2 * (g_ii_pair @ z_III) + s2 * term1_I_III + s2 * term2_I_III
        c_III_I = (-s2 * (g_ii_pair.T @ z_I)
                   + s2 * Wb_III_I[iu_o, ju_o, :] + s2 * Wb_III_I[ju_o, iu_o, :])

        c_II_III = 0.5 * s3 * (Tu_III[ju_o, iu_o, :] - Tu_III[iu_o, ju_o, :]
                               + Tl_III[ju_o, iu_o, :] - Tl_III[iu_o, ju_o, :])
        c_III_II = (-0.5 * s3 * (Tu_II[iu_o, ju_o, :] + Tu_II[ju_o, iu_o, :])
                    + 0.5 * s3 * (Tl_II[ju_o, iu_o, :] + Tl_II[iu_o, ju_o, :]))

        contrib_I = c_I_II + c_I_III
        contrib_II = c_II_II + c_II_I + c_II_III
        contrib_III = c_III_III + c_III_I + c_III_II
        return contrib_I, contrib_II, contrib_III

    g_kaai = _g_diag_mid_df(s.B_aa, oidx, vidx_abs, oidx)
    g_kaia = _g_diag_outer_df(_B_scr, oidx, vidx_abs, oidx)
    inner_Ip = (g_kaai - 2 * g_kaia).transpose(2, 1, 0)

    def apply_C_Ip_Ip(Vmat):
        term1 = np.einsum('ic,ac->ia', Vmat, g_aabb, optimize=True)
        term2 = np.einsum('iak,ka->ia', inner_Ip, Vmat, optimize=True)
        return term1 + term2
    # --- App1..4 building blocks for the virt-pair (2p1h) C-blocks ---
    def make_upper_k(Vmat, iu, ju, O_, Vv):
        Vup = np.zeros((O_, Vv, Vv))
        Vup[:, iu, ju] = Vmat
        return Vup
    # Streamed replacement for compute_Apps and the four Ip-IIp/Ip-IIIp
    def _apply_C_2p1h_coupled(z_Ip, z_IIp, z_IIIp):
        Vk_IIp = make_upper_k(z_IIp, iu_v, ju_v, O, V)
        Vk_IIIp = make_upper_k(z_IIIp, iu_v, ju_v, O, V)

        App1_IIp = np.zeros((O, V, V)); App2_IIp = np.zeros((O, V, V))
        App3_IIp = np.zeros((O, V, V)); App4_IIp = np.zeros((O, V, V))
        App1_IIIp = np.zeros((O, V, V)); App2_IIIp = np.zeros((O, V, V))
        App3_IIIp = np.zeros((O, V, V)); App4_IIIp = np.zeros((O, V, V))
        term1_IpIIp = np.zeros((O, V)); term2_IpIIp = np.zeros((O, V))
        Wc = np.zeros((O, V, V))
        term1_IpIIIp = np.zeros((O, V)); term2_IpIIIp = np.zeros((O, V))
        Wd1 = np.zeros((O, V, V))

        for lo, hi, g_ovov_c, g_ovvo_c in _c2p1h_df_chunks(s.B_aa, O, V, norb, B_w=_B_scr):
            Vk_IIp_c = Vk_IIp[lo:hi]
            Vk_IIIp_c = Vk_IIIp[lo:hi]

            App1_IIp += np.einsum('kXin,kYn->iXY', g_ovov_c, Vk_IIp_c, optimize=True)
            App2_IIp += np.einsum('kXni,kYn->iXY', g_ovvo_c, Vk_IIp_c, optimize=True)
            App3_IIp += np.einsum('kXim,kmY->iXY', g_ovov_c, Vk_IIp_c, optimize=True)
            App4_IIp += np.einsum('kXmi,kmY->iXY', g_ovvo_c, Vk_IIp_c, optimize=True)

            App1_IIIp += np.einsum('kXin,kYn->iXY', g_ovov_c, Vk_IIIp_c, optimize=True)
            App2_IIIp += np.einsum('kXni,kYn->iXY', g_ovvo_c, Vk_IIIp_c, optimize=True)
            App3_IIIp += np.einsum('kXim,kmY->iXY', g_ovov_c, Vk_IIIp_c, optimize=True)
            App4_IIIp += np.einsum('kXmi,kmY->iXY', g_ovvo_c, Vk_IIIp_c, optimize=True)

            term1_IpIIp += s32 * np.einsum('kaDi,kaD->ia', g_ovvo_c, Vk_IIp_c, optimize=True)
            term2_IpIIp += s32 * np.einsum('kaCi,kCa->ia', g_ovvo_c, Vk_IIp_c, optimize=True)
            Wc[lo:hi] = np.einsum('kXYi,iX->kXY', g_ovvo_c, z_Ip, optimize=True)

            term1_IpIIIp += s2 * (-np.einsum('kaiD,kaD->ia', g_ovov_c, Vk_IIIp_c, optimize=True)
                                  + 0.5 * np.einsum('kaDi,kaD->ia', g_ovvo_c, Vk_IIIp_c, optimize=True))
            term2_IpIIIp += s2 * (-np.einsum('kaiC,kCa->ia', g_ovov_c, Vk_IIIp_c, optimize=True)
                                  + 0.5 * np.einsum('kaCi,kCa->ia', g_ovvo_c, Vk_IIIp_c, optimize=True))
            Wd1[lo:hi] = (-np.einsum('kXiY,iX->kXY', g_ovov_c, z_Ip, optimize=True)
                          + 0.5 * np.einsum('kXYi,iX->kXY', g_ovvo_c, z_Ip, optimize=True))

        Apps_IIp = (App1_IIp, App2_IIp, App3_IIp, App4_IIp)
        Apps_IIIp = (App1_IIIp, App2_IIIp, App3_IIIp, App4_IIIp)

        C_r, D_r = iu_v, ju_v
        c_Ip = ((term1_IpIIp - term2_IpIIp)
                + s2 * (z_IIIp @ g_aa_cd.T) + term1_IpIIIp + term2_IpIIIp)
        c_IIp = s32 * Wc[:, C_r, D_r] - s32 * Wc[:, D_r, C_r]
        c_IIIp = s2 * (z_Ip @ g_aa_cd) + s2 * Wd1[:, C_r, D_r] + s2 * Wd1[:, D_r, C_r]

        return c_Ip, c_IIp, c_IIIp, Apps_IIp, Apps_IIIp

    # pp-ladder Term0: screened DF factor under W_aux, bare Bv_pp otherwise
    if M_vv_screened is not None:
        def _term0_IIp(Vmat):
            return _pp_ladder_matvec_df(M_vv_screened, Vmat, iu_v, ju_v, V, antisymmetric=True)

        def _term0_IIIp(Vmat):
            return _pp_ladder_matvec_df(M_vv_screened, Vmat, iu_v, ju_v, V, antisymmetric=False)
    else:
        def _term0_IIp(Vmat):
            return _pp_ladder_matvec_df(Bv_pp, Vmat, iu_v, ju_v, V, antisymmetric=True)

        def _term0_IIIp(Vmat):
            return _pp_ladder_matvec_df(Bv_pp, Vmat, iu_v, ju_v, V, antisymmetric=False)
    def apply_C_IIp_IIp(Vmat, Apps):
        App1, App2, App3, App4 = Apps
        Term0 = _term0_IIp(Vmat)
        A, B = iu_v, ju_v
        return (Term0 - App3[:, A, B] + App1[:, A, B] + 1.5 * App2[:, B, A]
                - App1[:, B, A] - 1.5 * App4[:, B, A] + App3[:, B, A]
                - 1.5 * App2[:, A, B] + 1.5 * App4[:, A, B])

    def apply_C_IIIp_IIIp(Vmat, Apps):
        App1, App2, App3, App4 = Apps
        Term0 = _term0_IIIp(Vmat)
        A, B = iu_v, ju_v
        return (Term0 - (App1[:, A, B] + App1[:, B, A]) - (App3[:, A, B] + App3[:, B, A])
                + 0.5 * (App2[:, A, B] + App2[:, B, A]) + 0.5 * (App4[:, A, B] + App4[:, B, A]))

    def apply_C_IIp_IIIp(Apps_IIIp):
        App1, App2, App3, App4 = Apps_IIIp
        A, B = iu_v, ju_v
        return 0.5 * s3 * (App2[:, B, A] - App2[:, A, B]) + 0.5 * s3 * (App4[:, B, A] - App4[:, A, B])

    def apply_C_IIIp_IIp(Apps_IIp):
        App1, App2, App3, App4 = Apps_IIp
        A, B = iu_v, ju_v
        return 0.5 * s3 * (App2[:, B, A] + App2[:, A, B]) - 0.5 * s3 * (App4[:, A, B] + App4[:, B, A])

    def aop(z):
        z_p = z[:norb]
        z_I = z[oI:oII].reshape(O, V)
        z_II = z[oII:oIII].reshape(nP_o, V)
        z_III = z[oIII:oIp].reshape(nP_o, V)
        z_Ip = z[oIp:oIIp].reshape(O, V)
        z_IIp = z[oIIp:oIIIp].reshape(O, nP_v)
        z_IIIp = z[oIIIp:nH].reshape(O, nP_v)

        y_p = (F @ z_p + U_I @ z_I.reshape(-1) + apply_U_2h1p_fwd(z_II, z_III)
               + U_Ip @ z_Ip.reshape(-1) + apply_U_2p1h_fwd(z_IIp, z_IIIp))

        c2h1p_I, c2h1p_II, c2h1p_III = _apply_C_2h1p_coupled(z_I, z_II, z_III)

        y_I = (U_I.T @ z_p) + e_I * z_I.reshape(-1) + (apply_C_I_I(z_I) + c2h1p_I).reshape(-1)
        uT_II, uT_III = apply_U_2h1p_adj(z_p)
        y_II = uT_II.reshape(-1) + e_II * z_II.reshape(-1) + c2h1p_II.reshape(-1)
        y_III = uT_III.reshape(-1) + e_III * z_III.reshape(-1) + c2h1p_III.reshape(-1)

        c2p1h_Ip, c2p1h_IIp, c2p1h_IIIp, Apps_IIp, Apps_IIIp = _apply_C_2p1h_coupled(
            z_Ip, z_IIp, z_IIIp)

        y_Ip = (U_Ip.T @ z_p) + e_Ip * z_Ip.reshape(-1) + (apply_C_Ip_Ip(z_Ip) + c2p1h_Ip).reshape(-1)
        uT_IIp, uT_IIIp = apply_U_2p1h_adj(z_p)
        y_IIp = uT_IIp.reshape(-1) + e_IIp * z_IIp.reshape(-1) + (
            c2p1h_IIp + apply_C_IIp_IIp(z_IIp, Apps_IIp) + apply_C_IIp_IIIp(Apps_IIIp)).reshape(-1)
        y_IIIp = uT_IIIp.reshape(-1) + e_IIIp * z_IIIp.reshape(-1) + (
            c2p1h_IIIp + apply_C_IIIp_IIp(Apps_IIp) + apply_C_IIIp_IIIp(z_IIIp, Apps_IIIp)).reshape(-1)

        return np.concatenate([y_p, y_I, y_II, y_III, y_Ip, y_IIp, y_IIIp])

    diag = np.concatenate([np.diag(F), e_I, e_II, e_III, e_Ip, e_IIp, e_IIIp])
    return aop, diag, d

"""Restricted (CSF Type I/II/III) helpers: dimensions, amplitude
combinations, pair unfolds, DF block gathers, streaming amplitude-chunk
generators, and the unstreamed U-block fallback builder shared by both
sigma route modules."""
import math
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum
from src.SingleReference.ADC.adc_utils import (
    _g_slice_df, _g_diag_df, _g_diag_mid_df, _g_diag_outer_df,
    _b_factor, _b_q_chunk, _b_chunk_einsum)


def dimensions(norb, nocc):
    """Segment sizes of the restricted CSF supermatrix."""
    nvirt = norb - nocc
    nP_o = nocc * (nocc - 1) // 2
    nP_v = nvirt * (nvirt - 1) // 2
    nI, nII, nIII = nocc * nvirt, nP_o * nvirt, nP_o * nvirt
    n2h1p = nI + nII + nIII
    nIp, nIIp, nIIIp = nocc * nvirt, nocc * nP_v, nocc * nP_v
    n2p1h = nIp + nIIp + nIIIp
    return {'norb': norb, 'nocc': nocc, 'nvirt': nvirt,
            'nI': nI, 'nII': nII, 'nIII': nIII, 'n2h1p': n2h1p,
            'nIp': nIp, 'nIIp': nIIp, 'nIIIp': nIIIp, 'n2p1h': n2p1h,
            'nH': norb + n2h1p + n2p1h}


def _no_u_blocks(norb, O, V, nP_o, nP_v):
    """The six U-block pieces of _build_u_blocks_* identically zero: the
    operator then carries F and the satellite block K + C alone, with no
    coupling work per matvec (Faddeev-ADC(3) applies its own U)."""
    zero_p = np.zeros(norb)
    return (np.zeros((norb, O * V)), np.zeros((norb, O * V)),
            lambda z_II, z_III: zero_p,
            lambda z_p: (np.zeros((nP_o, V)), np.zeros((nP_o, V))),
            lambda z_IIp, z_IIIp: zero_p,
            lambda z_p: (np.zeros((O, nP_v)), np.zeros((O, nP_v))))


def _u2_spin_amplitudes(g_num, g_num_T, dens, layout):
    """(t_same, t_opp): the same-spin (aaaa) and opposite-spin (abab) T2^(1)
    amplitudes for one denominator layout, each with its OWN EN-dressed
    denominator:

        v = g_num                 -> t_opp  = v / D_opp     (t2_1_abab)
        u = g_num - g_num_T       -> t_same = u / D_same    (t2_1_aaaa)

    Undressed, D_same is D_opp, so this reduces to the single-denominator
    expression.

    UNRESOLVED: callers decompose each bare (cA*g + cB*g^T)/D object into
    same-/opposite-spin pieces by guessing alpha=1 where the numerator is
    antisymmetrized and alpha=0 otherwise; undressed this is exact, dressed
    it is unverified, and the restricted EN result is not yet identical to
    the spin-orbital one as a result. Fix by reading the term-by-term
    assignment off the generated restricted blocks (l2_aaaa/l2_abab as
    separate arguments) -- do not tune against an eigenvalue residual."""
    D_same, D_opp = dens.build(layout)
    t_opp = g_num / D_opp
    if D_same is D_opp:                      # undressed: one division suffices
        return g_num / D_same - g_num_T / D_same, t_opp
    return (g_num - g_num_T) / D_same, t_opp


def _u_2p1h_zparts_df(B, O, norb, vidx_abs, oidx, z_p):
    """The three z_p-contracted DF intermediates the U_2p1h adjoint needs.

    Every term of u_IIp_full/u_IIIp_full carries the orbital index p on exactly
    ONE factor, so contracting z_p into that factor FIRST collapses each term
    from (V,V,norb,O) to (V,V,O) / (O,V,V) and the norb axis never materializes:

        gz [a,b,i] = sum_p g_abpi[a,b,p,i] z_p          (g_abpi = B[Q,a,p]B[Q,b,i])
        gzr[k,b,c] = sum_p g_ovpv[k,b,p,c] z_p          (g_ovpv = B[Q,k,p]B[Q,b,c])
        gzm[k,b,c] = sum_p g_ovvp[k,b,c,p] z_p          (g_ovvp = B[Q,k,c]B[Q,b,p])

    All three are AMPLITUDE-INDEPENDENT, so one build serves both the II' and
    III' blocks. Note g_abip_T[a,b,p,i] = g_abpi[b,a,p,i] (both are (ai|bp) in
    chemist form), so gz's own transpose covers that term too -- no second
    build. Peak transient here is O(V^2 O) plus O(naux*V), never O(V^2*norb*O)."""
    Bvp = B[:, vidx_abs, :]                     # (naux, V, norb)
    Bop = B[:, oidx, :]                         # (naux, O, norb)
    bz = _cached_einsum('Qap,p->Qa', Bvp, z_p, optimize=True)
    boz = _cached_einsum('Qkp,p->Qk', Bop, z_p, optimize=True)
    Bvo = B[:, vidx_abs][:, :, oidx]
    Bvv = B[:, vidx_abs][:, :, vidx_abs]
    Bov = B[:, oidx][:, :, vidx_abs]
    gz = _cached_einsum('Qa,Qbi->abi', bz, Bvo, optimize=True)
    gzr = _cached_einsum('Qk,Qbc->kbc', boz, Bvv, optimize=True)
    gzm = _cached_einsum('Qkc,Qb->kbc', Bov, bz, optimize=True)
    return gz, gzr, gzm


def _u_2p1h_unfold(z, iu_v, ju_v, O, V, anti):
    """(O,nP_v) pair-packed trial vector -> full (O,V,V), antisymmetric or
    symmetric in (a,b). Every term of u_IIp_full is exactly antisymmetric in
    (a,b) and every term of u_IIIp_full exactly symmetric (unaffected by EN
    dressing since the dressed denom_ab_kl stays symmetric under the
    SIMULTANEOUS (a,b)+(k,l) swap), so

        sum_{a<b} u[a,b,p,i] z[i,(ab)] == 1/2 sum_{a,b} u[a,b,p,i] Z[i,a,b]

    term by term -- which is what lets each term be contracted separately
    without ever assembling u."""
    Z = np.zeros((O, V, V))
    Z[:, iu_v, ju_v] = z
    Z[:, ju_v, iu_v] = -z if anti else z
    return Z


def _u_2h1p_unfold(z, iu_o, ju_o, O, V, anti):
    """(nP_o,V) pair-packed 2h1p trial vector -> full (O,O,V), antisymmetric or
    symmetric in the OCCUPIED pair -- the 2h1p mirror of _u_2p1h_unfold.

    Every term of u_II is exactly antisymmetric in (i,j) and every term of
    u_III exactly symmetric. The load-bearing identities:

        g_ijap_T[i,j,p,a] == g_ijpa[j,i,p,a]     (both are (ip|ja) in chemist)
        term_vp[i,j,p,a]  == term_pv[j,i,p,a]    (X_ijcd is symmetric under the
                                                  SIMULTANEOUS (i,j)+(c,d) swap,
                                                  which survives EN dressing)
        Y_mix_2           == Y_mix_1^T(i,j)      (by construction)
        the two A_anti/W1 ring terms are each other's (i,j) transpose

    so each block's six terms collapse to FOUR representative shapes, and

        sum_{i<j} u[i,j,p,a] z[(ij),a] == 1/2 sum_{i,j} u[i,j,p,a] Z[i,j,a]

    term by term -- which is what lets each term be contracted separately
    without ever assembling u. Since the paired terms contribute equally
    against a Z of matching symmetry, the 1/2 and the pairing cancel and one
    representative of each shape carries the whole block."""
    Z = np.zeros((O, O, V))
    Z[iu_o, ju_o] = z
    Z[ju_o, iu_o] = -z if anti else z
    return Z


def _u_I_ring_contract_df(Bo_full, Bv_full, Boo, Bvo, W1, W2, q_chunk=8):
    """U_I's two U^(2) ring terms straight from the DF factors, so that the
    (V,O,norb,O)-shaped g_vopo/g_voho need not exist:

        einsum('ikca,cipk->ipa', W1, g_vopo),  g_vopo[c,i,p,k] = sum_Q B[Q,c,p]B[Q,i,k]
        einsum('ikca,cikp->ipa', W2, g_voho),  g_voho[c,i,k,p] = sum_Q B[Q,c,k]B[Q,i,p]

    U_I itself stays materialized -- only (norb, O*V), cheap -- so this is a
    one-time build, not a per-matvec cost. Streamed over the occupied index i
    (carried by every factor) and chunked over the auxiliary index Q, so the
    peak transient is O(q_chunk*V^2) rather than O(naux*O*V^2).

    Returns (term1, term2), each (O,norb,V), matching the einsums' 'ipa'."""
    naux, O, _ = Boo.shape
    V = W1.shape[2]
    norb = Bo_full.shape[2]
    term1 = np.zeros((O, norb, V))
    term2 = np.zeros((O, norb, V))
    for i in range(O):
        for start in range(0, naux, q_chunk):
            sl = slice(start, start + q_chunk)
            # sum_k W1[i,k,c,a] B[Q,i,k]  ->  (nq,V,V), then close with B[Q,c,p]
            F1 = _cached_einsum('kca,qk->qca', W1[i], Boo[sl, i], optimize=True)
            term1[i] += _cached_einsum('qcp,qca->pa', Bv_full[sl], F1, optimize=True)
            # sum_{k,c} W2[i,k,c,a] B[Q,c,k]  ->  (nq,V), then close with B[Q,i,p]
            G2 = _cached_einsum('kca,qck->qa', W2[i], Bvo[sl], optimize=True)
            term2[i] += _cached_einsum('qp,qa->pa', Bo_full[sl, i], G2, optimize=True)
    return term1, term2


def _dress_is_streamable(dress):
    """Whether a u2_denom_dress dict is one _u2_2h1p_amplitude_chunks/
    _u2_2p1h_amplitude_chunks (and hence the whole U-block streaming path)
    can represent: None/falsy (bare) or hh/pp-only with spin_adapted=True
    (the default). 'hp' and spin-resolved (spin_adapted=False) dressing are
    excluded -- see the banner comment above _u2_2h1p_amplitude_chunks."""
    if not dress:
        return True
    if dress.get('hp', False):
        return False
    return dress.get('spin_adapted', True)


def _build_g_blocks_df(B, O, V, norb, need_vvvv=True, need_vv_pv_vp=True, need_vv_pp=True,
                       need_ovpv_ovvp=True, need_oo_pp=True, need_vopo_voho=True,
                       need_oovv_vvoo=True, need_voov_vovo=True, need_ovov_ovvo=True,
                       need_oo_po_op=True, need_oooo=True):
    """
    All twenty named integral-slice blocks build_supermatrix/
    build_matrix_free_operator's 'Common integral slices' section needs,
    """
    occ = np.arange(O)
    vir = np.arange(O, norb)
    allo = np.arange(norb)
    blocks = dict(
        g_ii=_g_diag_df(B, occ, allo, allo),
        g_aa=_g_diag_df(B, vir, allo, allo),
    )
    # g_oo_po/g_oo_op ((O,O,norb,O)) and g_oooo (O^4) are the last rank-4
    # integrals on the streamed DF path: the U-side g_klpi terms go
    # DF-direct (_corr_kl_Ip_streamed_df + the rank-3 chains in
    # _build_u_blocks_streamed) and the C-side g_oooo terms through
    # _oooo_contract_df.
    if need_oo_po_op:
        blocks['g_oo_po'] = _g_slice_df(B, occ, occ, allo, occ)
        blocks['g_oo_op'] = _g_slice_df(B, occ, occ, occ, allo)
    if need_oooo:
        blocks['g_oooo'] = _g_slice_df(B, occ, occ, occ, occ)
    # g_voov/g_vovo (2h1p) and g_ovov/g_ovvo (2p1h) are (V,O,O,V)/(O,V,V,O),
    # the last O^2V^2 arrays standing on the streamed C-block path: nothing
    # reads the full arrays there -- the C-block appliers rebuild
    # occupied-chunk slices directly from B via _c2h1p_df_chunks/
    # _c2p1h_df_chunks, and the small diagonal pieces (C_I_I/C_Ip_Ip) go
    # through _g_diag_mid_df/_g_diag_outer_df instead.
    if need_voov_vovo:
        blocks['g_voov'] = _g_slice_df(B, vir, occ, occ, vir)
        blocks['g_vovo'] = _g_slice_df(B, vir, occ, vir, occ)
    if need_ovov_ovvo:
        blocks['g_ovov'] = _g_slice_df(B, occ, vir, occ, vir)
        blocks['g_ovvo'] = _g_slice_df(B, occ, vir, vir, occ)
    # g_oovv/g_vvoo (O,O,V,V)/(V,V,O,O): the streamed bare-DF path never
    # reads either -- U_I/U_Ip's ring construction and the apply_U_2h1p/
    # apply_U_2p1h closures rebuild occupied-chunk slices of them directly
    # from B via _u2_2h1p_amplitude_chunks/_u2_2p1h_amplitude_chunks instead.
    if need_oovv_vvoo:
        blocks['g_oovv'] = _g_slice_df(B, occ, occ, vir, vir)
        blocks['g_vvoo'] = _g_slice_df(B, vir, vir, occ, occ)
    # g_oo_pp feeds only U_II/U_III's bare g_ijpa/g_ijap_T; g_vopo/g_voho
    # only the U^(2) ring terms of U_I/U_II/U_III. All three vanish once
    # U_II/U_III are matrix-free and U_I goes through _u_I_ring_contract_df.
    if need_oo_pp:
        blocks['g_oo_pp'] = _g_slice_df(B, occ, occ, allo, allo)
    if need_vopo_voho:
        blocks['g_vopo'] = _g_slice_df(B, vir, occ, allo, occ)
        blocks['g_voho'] = _g_slice_df(B, vir, occ, occ, allo)
    if need_ovpv_ovvp:
        blocks['g_ovpv'] = _g_slice_df(B, occ, vir, allo, vir)
        blocks['g_ovvp'] = _g_slice_df(B, occ, vir, vir, allo)
    if need_vv_pv_vp:
        blocks['g_vv_pv'] = _g_slice_df(B, vir, vir, allo, vir)
        blocks['g_vv_vp'] = _g_slice_df(B, vir, vir, vir, allo)
    if need_vv_pp:
        blocks['g_vv_pp'] = _g_slice_df(B, vir, vir, allo, allo)
    if need_vvvv:
        blocks['g_vvvv'] = _g_slice_df(B, vir, vir, vir, vir)
    return blocks


def _u_pv_vp_contract_df(Bv_full, Bv_vv, X, x_letters, out_letters, q_chunk=32):
    """Direct DF replacement for the pair
    einsum(f'{x_letters}cd,cdpa->{out_letters}pa', X, g_vv_pv),
    einsum(f'{x_letters}cd,cdap->{out_letters}pa', X, g_vv_vp)

    Chunked over the auxiliary index Q
    """
    x_sub = f'{x_letters}cd'
    out_sub = f'{out_letters}pa'
    naux = Bv_full.shape[0]
    out_shape = X.shape[:-2] + (Bv_full.shape[2], Bv_vv.shape[2])
    term_pv = np.zeros(out_shape)
    term_vp = np.zeros(out_shape)
    for start in range(0, naux, q_chunk):

        # (nq, V, norb) -- B[Q,c,p]
        B1c = Bv_full[start:start + q_chunk]

        # (nq, V, V)    -- B[Q,d,a]
        B2c = Bv_vv[start:start + q_chunk]
        term_pv += _cached_einsum(f'{x_sub},Qcp,Qda->{out_sub}', X, B1c, B2c, optimize=True)
        term_vp += _cached_einsum(f'{x_sub},Qca,Qdp->{out_sub}', X, B2c, B1c, optimize=True)
    return term_pv, term_vp


def _u_Ip_pv_vp_contract_df(Bo_full, Bv_pp, Bov_full, Bv_full, W1p, W2p, q_chunk=32):
    """
    Direct DF replacement for U_Ip's pair of ring contractions

        einsum('acki,kapc->api', W1p, g_ovpv),
        einsum('acki,kacp->api', W2p, g_ovvp)

    Substituting the module's usual DF identity g[p,q,r,s] =
    sum_Q B[Q,p,r]*B[Q,q,s] (B_aa's own convention) gives

        g_ovpv[k,a,p,c] = sum_Q B[Q,k,p]*B[Q,a,c]
        g_ovvp[k,a,c,p] = sum_Q B[Q,k,c]*B[Q,a,p]

    so each einsum collapses into a direct 3-operand contraction:

        term1[a,p,i] = sum_{c,k,Q} W1p[a,c,k,i] * B[Q,k,p] * B[Q,a,c]
        term2[a,p,i] = sum_{c,k,Q} W2p[a,c,k,i] * B[Q,k,c] * B[Q,a,p]

    (note 'a' is a free index carried by BOTH factors of term2 and by the
    amplitude in both terms -- it is a batch index here, not summed.)
    """
    naux = Bo_full.shape[0]
    V, O, norb = W1p.shape[0], W1p.shape[2], Bo_full.shape[2]
    term1 = np.zeros((V, norb, O))
    term2 = np.zeros((V, norb, O))
    for start in range(0, naux, q_chunk):
        sl = slice(start, start + q_chunk)
        term1 += _cached_einsum('acki,Qkp,Qac->api', W1p, Bo_full[sl], Bv_pp[sl],
                                optimize=True)
        term2 += _cached_einsum('acki,Qkc,Qap->api', W2p, Bov_full[sl], Bv_full[sl],
                                optimize=True)
    return term1, term2


# ===================== S4: occupied-index streaming (bare + hh/pp-dressed EN) =====================
# The amplitudes below (each O^2V^2, held live for the whole aop lifetime)
# are built chunk by chunk, streaming over one occupied axis, never as full
# arrays. Bare EN: D is a trivial broadcast, so streaming is free. Dressed
# hh/pp (spin_adapted=True, production): the shift is a sum of broadcasts of
# small (O,O)/(V,V) matrices, equally separable, so chunking is just slicing
# their matching axis. 'hp' and determinant-wise (spin_adapted=False) EN are
# EXCLUDED here -- their shifts don't reduce to a small-matrix broadcast per
# axis, so callers must not pass them through dh/dp.
def _u2_2h1p_amplitude_chunks(B, O, V, norb, eps_o, eps_v, chunk_size, dh=None, dp=None):
    """Yields (lo, hi, t_same, t_opp, W1, W2, A_mix, X_ijcd) chunks, each
    shape (hi-lo, O, V, V), streaming over the FIRST occupied axis of the
    'ikca'/'ijcd' amplitudes (== g_oovv's own first axis).

    dh/dp: optional (O,O)/(V,V) EN hh/pp shift matrices (spin_adapted,
    restricted_channel_shifts' d_h[0]/d_p[0]); None reproduces the bare
    denominator exactly (dh/dp=None is the previous, bare-only behavior).

    Only ONE DF build per chunk: the transpose _u2_spin_amplitudes needs
    (g_oovv.transpose(0,1,3,2), swapping the two V-sized axes) never touches
    axis 0, so chunking axis 0 commutes with it -- unlike the 2p1h generator
    below, whose transpose swaps the SAME two axes being chunked."""
    occ = np.arange(O)
    vir = np.arange(O, norb)
    for lo in range(0, O, chunk_size):
        hi = min(lo + chunk_size, O)
        occ_chunk = occ[lo:hi]
        g_chunk = _g_slice_df(B, occ_chunk, occ, vir, vir)       # (ci,O,V,V)
        g_chunk_T = g_chunk.transpose(0, 1, 3, 2)
        eo_chunk = eps_o[lo:hi]
        D_ikca = (eo_chunk[:, None, None, None] + eps_o[None, :, None, None]
                 - eps_v[None, None, :, None] - eps_v[None, None, None, :])
        D_ijcd = (eps_v[None, None, :, None] + eps_v[None, None, None, :]
                 - eo_chunk[:, None, None, None] - eps_o[None, :, None, None])
        if dh is not None or dp is not None:
            Delta = np.zeros(D_ikca.shape)
            if dh is not None:
                Delta = Delta + dh[lo:hi, :][:, :, None, None]
            if dp is not None:
                Delta = Delta + dp[None, None, :, :]
            D_ikca = D_ikca - Delta      # 'ikca' bare_sign=+1 -> D = Dbare-Delta
            D_ijcd = D_ijcd + Delta      # 'ijcd' bare_sign=-1 -> D = Dbare+Delta
        t_opp = g_chunk / D_ikca
        t_same = (g_chunk - g_chunk_T) / D_ikca
        W1 = t_same - 2 * t_opp
        W2 = t_opp - 2 * t_same
        A_mix = 2 * t_same - t_opp
        X_ijcd = g_chunk / D_ijcd
        yield lo, hi, t_same, t_opp, W1, W2, A_mix, X_ijcd


def _u2_2p1h_amplitude_chunks(B, O, V, norb, eps_o, eps_v, chunk_size, dh=None, dp=None):
    """Yields (lo, hi, tp_same, tp_opp, W1p, W2p, X_abkl) chunks, each shape
    (V, V, O, hi-lo), streaming over the LAST occupied axis (l) of the
    'ackl'/'abkl' amplitudes -- the axis every consumer (U_Ip's ring build,
    apply_U_2p1h_fwd/adj) treats as the batch/output index (its OWN axis3,
    renamed 'i' at consumption).

    dh/dp: same EN hh/pp shift matrices as _u2_2h1p_amplitude_chunks (see
    its docstring for scope/exclusions); None reproduces the bare
    denominator.

    TWO DF builds per chunk, unlike the 2h1p generator's one: g_vvoo's
    transpose swaps the SAME two occupied axes being chunked here (both
    O-sized), so the exchange term needs its own build (g[a,c,l_chunk,k]
    transposed back), never the full (V,V,O,O)."""
    occ = np.arange(O)
    vir = np.arange(O, norb)
    for lo in range(0, O, chunk_size):
        hi = min(lo + chunk_size, O)
        occ_chunk = occ[lo:hi]
        g_direct = _g_slice_df(B, vir, vir, occ, occ_chunk)                      # g[a,c,k,l_chunk]
        g_exch = _g_slice_df(B, vir, vir, occ_chunk, occ).transpose(0, 1, 3, 2)  # g[a,c,l_chunk,k]->[a,c,k,l_chunk]
        el_chunk = eps_o[lo:hi]
        D_ackl = (eps_v[:, None, None, None] + eps_v[None, :, None, None]
                 - eps_o[None, None, :, None] - el_chunk[None, None, None, :])
        D_abkl = (eps_o[None, None, :, None] + el_chunk[None, None, None, :]
                 - eps_v[:, None, None, None] - eps_v[None, :, None, None])
        if dh is not None or dp is not None:
            Delta = np.zeros(D_ackl.shape)
            if dh is not None:
                Delta = Delta + dh[:, lo:hi][None, None, :, :]
            if dp is not None:
                Delta = Delta + dp[:, :, None, None]
            D_ackl = D_ackl + Delta      # 'ackl' bare_sign=-1 -> D = Dbare+Delta
            D_abkl = D_abkl - Delta      # 'abkl' bare_sign=+1 -> D = Dbare-Delta
        tp_opp = g_direct / D_ackl
        tp_same = (g_direct - g_exch) / D_ackl
        W1p = 2 * tp_opp - tp_same
        W2p = 2 * tp_same - tp_opp
        X_abkl = g_direct / D_abkl
        yield lo, hi, tp_same, tp_opp, W1p, W2p, X_abkl


def _build_u_blocks_unstreamed(B, O, V, norb, eps_o, eps_v, dens,
                               Bo_full, Bv_full, Bov_full, Boo_full, Bvo_full, Bv_pp,
                               iu_o, ju_o, iu_v, ju_v, ints,
                               is_adc2x, use_materialized, w_chemist):
    """U blocks + appliers for every path _build_u_blocks_streamed cannot
    take: adc2x, 'hp'/spin-resolved EN dressing, screened MCDE, or no DF
    factors at all. Amplitudes (t_same/t_opp/W1/W2/X_ijcd and 2p1h mirrors)
    are materialized as full O^2V^2 arrays and either consumed by
    DF-contraction appliers per matvec (use_materialized=False) or folded
    into explicit dense U matrices once (use_materialized=True: the dense
    no-DF path and the screened-MCDE diagnostic).

    ints keys read: g_ii_U, g_ii, g_aa_U, g_aa, g_oo_po, g_oo_op, g_oovv,
    g_oovv_T, g_vvoo, g_vvoo_T, g_vopo, g_voho, g_ovpv, g_ovvp, g_vv_pv,
    g_vv_vp, g_oo_pp_U, g_vv_pp_U (unneeded ones may be None).
    Returns the same 6-tuple as _build_u_blocks_streamed.
    """
    s32 = math.sqrt(1.5)
    s12 = math.sqrt(0.5)
    nP_o = O * (O - 1) // 2
    nP_v = V * (V - 1) // 2
    nI = O * V
    nIp = O * V
    nII = nIII = nP_o * V
    nIIp = nIIIp = O * nP_v
    g_ii_U, g_ii = ints['g_ii_U'], ints['g_ii']
    g_aa_U, g_aa = ints['g_aa_U'], ints['g_aa']
    g_oo_po, g_oo_op = ints['g_oo_po'], ints['g_oo_op']
    g_oovv, g_oovv_T = ints['g_oovv'], ints['g_oovv_T']
    g_vvoo, g_vvoo_T = ints['g_vvoo'], ints['g_vvoo_T']
    g_vopo, g_voho = ints['g_vopo'], ints['g_voho']
    g_ovpv, g_ovvp = ints['g_ovpv'], ints['g_ovvp']
    g_vv_pv, g_vv_vp = ints['g_vv_pv'], ints['g_vv_vp']
    g_oo_pp_U, g_vv_pp_U = ints['g_oo_pp_U'], ints['g_vv_pp_U']

    # ===================== U_I =====================
    u_I = g_ii_U[:, :, O:].copy()  # 1st-order piece
    if not is_adc2x:
        # i=j slice: purely opposite-spin (the aaaa amplitude is
        # antisymmetric in its occupied pair, hence zero here), and its hh
        # shift is the <ii|ii> diagonal.
        g_iicd = g_ii[:, O:, O:]
        X_icd = g_iicd / dens.build_icd()
        if Bv_pp is not None and w_chemist is None:
            term_pv, term_vp = _u_pv_vp_contract_df(Bv_full, Bv_pp, X_icd, 'i', 'i')
        else:
            term_pv = np.einsum('icd,cdpa->ipa', X_icd, g_vv_pv, optimize=True)
            term_vp = np.einsum('icd,cdap->ipa', X_icd, g_vv_vp, optimize=True)
        u_I -= 0.5 * (term_pv + term_vp)
        # aaaa/abab amplitudes on 'ikca'; W1/W2/A_anti/A_mix are their
        # unique linear combinations (see _u2_spin_amplitudes).
        t_same, t_opp = _u2_spin_amplitudes(g_oovv, g_oovv_T, dens, 'ikca')
        W1 = t_same - 2 * t_opp                     # -(g + g^T)/D
        W2 = t_opp - 2 * t_same                     # (-g + 2g^T)/D
        # g_vopo/g_voho are skipped on the unscreened DF path; same
        # algebra either way -- see _u_I_ring_contract_df.
        if g_vopo is None:
            r1_I, r2_I = _u_I_ring_contract_df(Bo_full, Bv_full, Boo_full,
                                               Bvo_full, W1, W2)
        else:
            r1_I = np.einsum('ikca,cipk->ipa', W1, g_vopo, optimize=True)
            r2_I = np.einsum('ikca,cikp->ipa', W2, g_voho, optimize=True)
        u_I += r1_I
        u_I += r2_I
    U_I = u_I.transpose(1, 0, 2).reshape(norb, nI)

    # ============ U_II / U_III -- MATRIX-FREE (2h1p mirror) ============
    # Each block's SIX terms collapse to FOUR representative shapes,
    # because u_II is exactly antisymmetric in (i,j) and u_III exactly
    # symmetric, and the terms pair up under that transpose
    # (g_ijap_T[i,j] = g_ijpa[j,i]; term_vp = term_pv^T; Y_mix_2 =
    # Y_mix_1^T; the two ring terms are each other's transpose). Against
    # an unfolded trial vector Z of matching symmetry the paired terms
    # contribute equally, so the 1/2 from _u_2h1p_unfold cancels the
    # pairing and ONE representative of each shape carries the block --
    # see _u_2h1p_unfold. The blocks then differ only by (scale, which
    # amplitude the ring term uses, two signs), tabulated in _2h1p_specs.
    # As on the 2p1h side, each term carries p on exactly ONE factor, so
    # the adjoint contracts z_p in first and the norb axis never
    # materializes. Peak transient is O(q_chunk*O^2*V).
    X_ijcd = A_anti = A_mix = None
    if not is_adc2x:
        X_ijcd = g_oovv / dens.denom('ijcd', 'opp')   # pure opposite-spin
        A_anti = t_same                             # (g - g^T)/D
        A_mix = 2 * t_same - t_opp                  # (g - 2g^T)/D

    # (scale, ring amplitude, ring sign, Y_mix sign, antisymmetric?)
    _2h1p_specs = ((s32, 'A_anti', -1.0, +1.0, True),
                   (s12, 'W1', +1.0, -1.0, False))
    _amps_2h1p = ({} if is_adc2x else {'A_anti': A_anti, 'W1': W1})
    q_chunk_2h1p = 32

    if use_materialized:
        # Materialized build: the DF-contraction appliers below rebuild
        # g_oo_pp/g_vopo/g_voho from the BARE DF factors, so they are valid
        # only for a bare vertex and only when there ARE DF factors.
        # Screened MCDE replaces g_oo_pp_U (screen_coupling) with a
        # W-substituted tensor the bare reconstruction cannot reproduce;
        # screening is a small-system diagnostic, so the memory cost there
        # is irrelevant. build_supermatrix's own copy of these blocks stays
        # the reference this path is checked against.
        g_ijpa = g_oo_pp_U[:, :, :, O:]
        g_ijap_T = g_oo_pp_U[:, :, O:, :].transpose(0, 1, 3, 2)
        u_II = s32 * (g_ijpa - g_ijap_T)
        u_III = s12 * (g_ijpa + g_ijap_T)
        if not is_adc2x:
            if Bv_pp is not None and w_chemist is None:
                term_pv, term_vp = _u_pv_vp_contract_df(Bv_full, Bv_pp, X_ijcd, 'ij', 'ij')
            else:
                term_pv = np.einsum('ijcd,cdpa->ijpa', X_ijcd, g_vv_pv, optimize=True)
                term_vp = np.einsum('ijcd,cdap->ijpa', X_ijcd, g_vv_vp, optimize=True)
            Y_mix_1 = np.einsum('ikca,cjkp->ijpa', A_mix, g_voho, optimize=True)
            Y_mix_2 = np.einsum('jkca,cikp->jipa', A_mix, g_voho, optimize=True).transpose(1, 0, 2, 3)
            u_II -= s32 * (term_pv - term_vp)
            u_II -= s32 * np.einsum('ikca,cjpk->ijpa', A_anti, g_vopo, optimize=True)
            u_II += s32 * Y_mix_1
            u_II += s32 * np.einsum('jkca,cipk->jipa', A_anti, g_vopo, optimize=True).transpose(1, 0, 2, 3)
            u_II -= s32 * Y_mix_2
            u_III -= s12 * (term_pv + term_vp)
            u_III += s12 * np.einsum('jkca,cipk->jipa', W1, g_vopo, optimize=True).transpose(1, 0, 2, 3)
            u_III -= s12 * Y_mix_2
            u_III += s12 * np.einsum('ikca,cjpk->ijpa', W1, g_vopo, optimize=True)
            u_III -= s12 * Y_mix_1
        _U_II = u_II[iu_o, ju_o].transpose(1, 0, 2).reshape(norb, nII)
        _U_III = u_III[iu_o, ju_o].transpose(1, 0, 2).reshape(norb, nIII)
        del u_II, u_III

        def apply_U_2h1p_fwd(z_II, z_III):
            return _U_II @ z_II.reshape(-1) + _U_III @ z_III.reshape(-1)

        def apply_U_2h1p_adj(z_p):
            return ((_U_II.T @ z_p).reshape(nP_o, V),
                    (_U_III.T @ z_p).reshape(nP_o, V))
    else:
        def apply_U_2h1p_fwd(z_II, z_III):
            """y_p += U_II @ z_II + U_III @ z_III."""
            y = np.zeros(norb)
            for (sc, akey, qsign, ysign, anti), z in zip(_2h1p_specs, (z_II, z_III)):
                Z = _u_2h1p_unfold(z, iu_o, ju_o, O, V, anti)
                # bare 1st-order piece: g_ijpa = sum_Q B[Q,i,p] B[Q,j,a]
                M = _cached_einsum('Qja,ija->Qi', Bov_full, Z, optimize=True)
                y += sc * _b_chunk_einsum('Qip,Qi->p', (Bo_full, M), (True, True), Bo_full)
                if is_adc2x:
                    continue
                # -term_pv: g_vv_pv[c,d,p,a] = sum_Q B[Q,c,p] B[Q,d,a]
                for st in range(0, Bv_pp.shape[0], q_chunk_2h1p):
                    sl = slice(st, st + q_chunk_2h1p)
                    T = _cached_einsum('Qda,ija->Qijd', Bv_pp[sl], Z, optimize=True)
                    P = _cached_einsum('ijcd,Qijd->Qc', X_ijcd, T, optimize=True)
                    y -= sc * _cached_einsum('Qcp,Qc->p', Bv_full[sl], P, optimize=True)
                # ring term: g_vopo[c,j,p,k] = sum_Q B[Q,c,p] B[Q,j,k]
                S = _cached_einsum('ikca,ija->kcj', _amps_2h1p[akey], Z, optimize=True)
                R = _cached_einsum('Qjk,kcj->Qc', Boo_full, S, optimize=True)
                y += qsign * sc * _b_chunk_einsum('Qcp,Qc->p', (Bv_full, R), (True, True), Bv_full)
                # Y_mix term: g_voho[c,j,k,p] = sum_Q B[Q,c,k] B[Q,j,p]
                S4 = _cached_einsum('ikca,ija->kcj', A_mix, Z, optimize=True)
                R4 = _cached_einsum('Qck,kcj->Qj', Bvo_full, S4, optimize=True)
                y += ysign * sc * _b_chunk_einsum('Qjp,Qj->p', (Bo_full, R4), (True, True), Bo_full)
            return y

        def apply_U_2h1p_adj(z_p):
            """(U_II.T @ z_p, U_III.T @ z_p), each returned as (nP_o, V)."""
            bzo = _b_chunk_einsum('Qip,p->Qi', (Bo_full, z_p), (True, False),
                                  Bo_full, (Bo_full.shape[0], O))
            bzv = _b_chunk_einsum('Qcp,p->Qc', (Bv_full, z_p), (True, False),
                                  Bv_full, (Bv_full.shape[0], V))
            gz1 = _cached_einsum('Qi,Qja->ija', bzo, Bov_full, optimize=True)
            if not is_adc2x:
                gvo = _cached_einsum('Qc,Qjk->cjk', bzv, Boo_full, optimize=True)
                gvh = _cached_einsum('Qck,Qj->cjk', Bvo_full, bzo, optimize=True)
                tp = np.zeros((O, O, V))
                for st in range(0, Bv_pp.shape[0], q_chunk_2h1p):
                    sl = slice(st, st + q_chunk_2h1p)
                    Xb = _cached_einsum('ijcd,Qc->Qijd', X_ijcd, bzv[sl], optimize=True)
                    tp += _cached_einsum('Qijd,Qda->ija', Xb, Bv_pp[sl], optimize=True)
                y1 = _cached_einsum('ikca,cjk->ija', A_mix, gvh, optimize=True)
            out = []
            for sc, akey, qsign, ysign, anti in _2h1p_specs:
                sgn = -1.0 if anti else 1.0
                tot = gz1
                if not is_adc2x:
                    q3 = _cached_einsum('ikca,cjk->ija', _amps_2h1p[akey], gvo,
                                        optimize=True)
                    tot = tot - tp + qsign * q3 + ysign * y1
                tot = sc * (tot + sgn * tot.transpose(1, 0, 2))
                out.append(tot[iu_o, ju_o])
            return out

    # ===================== U_Ip =====================
    u_Ip = g_aa_U[:, :, :O].transpose(2, 1, 0).copy()  # 1st-order piece
    g_klpi_sym = None
    if not is_adc2x:
        g_aakl = g_aa[:, :O, :O]
        # a=b slice: mirror of X_icd -- purely opposite-spin, pp shift is
        # the <aa|aa> diagonal.
        X_akl = g_aakl / dens.build_akl()
        g_klpi_sym = g_oo_po + g_oo_op.transpose(0, 1, 3, 2)
        corr_kl_Ip = 0.5 * np.einsum('akl,klpi->api', X_akl, g_klpi_sym, optimize=True)
        u_Ip += corr_kl_Ip.transpose(2, 1, 0)
        tp_same, tp_opp = _u2_spin_amplitudes(g_vvoo, g_vvoo_T, dens, 'ackl')
        W1p = 2 * tp_opp - tp_same                   # (g + g^T)/D
        W2p = 2 * tp_same - tp_opp                   # (g - 2g^T)/D
        # g_ovpv/g_ovvp are skipped on the unscreened DF path (O*V^2*norb
        # sized), routed through the DF contraction instead; the two
        # branches are the same algebra, see _u_Ip_pv_vp_contract_df.
        if g_ovpv is None:
            t1_Ip, t2_Ip = _u_Ip_pv_vp_contract_df(Bo_full, Bv_pp, Bov_full,
                                                   Bv_full, W1p, W2p)
        else:
            t1_Ip = np.einsum('acki,kapc->api', W1p, g_ovpv, optimize=True)
            t2_Ip = np.einsum('acki,kacp->api', W2p, g_ovvp, optimize=True)
        u_Ip += t1_Ip.transpose(2, 1, 0)
        u_Ip += t2_Ip.transpose(2, 1, 0)
    U_Ip = u_Ip.transpose(1, 0, 2).reshape(norb, nIp)

    # ================= U_IIp / U_IIIp -- MATRIX-FREE ==================
    # Two structural facts make the DF-contraction appliers exact (both
    # verified numerically, and both survive EN dressing -- see
    # _u_2p1h_unfold and _u_2p1h_zparts_df):
    #   * every term of u_IIp_full is antisymmetric in (a,b) and every term
    #     of u_IIIp_full symmetric, so the pair sum becomes 1/2 * the full
    #     (a,b) sum against an unfolded trial vector, term by term;
    #   * each term carries p on exactly one factor, so the adjoint's
    #     z_p contraction collapses it before any big object forms.
    # g_klpi_anti/g_klpi_sym_l are small (O,O,norb,O) -- not O^2V^2.
    X_abkl = g_klpi_anti = g_klpi_sym_l = A_anti_p = A_mix_p = None
    if not is_adc2x:
        g_klpi_anti = g_oo_po - g_oo_op.transpose(0, 1, 3, 2)
        g_klpi_sym_l = g_klpi_sym
        X_abkl = g_vvoo / dens.denom('abkl', 'opp')   # pure opposite-spin
        A_anti_p = tp_same                            # (g - g^T)/D
        A_mix_p = W2p                                 # == U_Ip's W2p

    # (scale, g_klpi, ring amplitude, mix amplitude, mix sign, antisym?)
    _2p1h_specs = ((s32, 'anti', 'A_anti_p', -1.0, True),
                   (s12, 'sym', 'W1p', +1.0, False))
    # W1p/A_anti_p only exist above the adc2x truncation (U^(2) is dropped
    # there, leaving only the bare 1st-order piece), and the closures only
    # index _amps under the same guard.
    _amps = ({} if is_adc2x else {'A_anti_p': A_anti_p, 'W1p': W1p})
    _gk = {'anti': g_klpi_anti, 'sym': g_klpi_sym_l}

    if use_materialized:
        # Materialized build. build_supermatrix's own copy of these
        # blocks is the reference the matrix-free path is checked against,
        # so this branch stays deliberately literal.
        if Bv_pp is not None:
            allo = np.arange(norb)
            oidx = np.arange(O)
            vidx_abs = np.arange(O, norb)
            g_abpi = _g_slice_df(B, vidx_abs, vidx_abs, allo, oidx)
            g_abip_T = _g_slice_df(B, vidx_abs, vidx_abs, oidx, allo).transpose(0, 1, 3, 2)
        else:
            g_abpi = g_vv_pp_U[:, :, :, :O]
            g_abip_T = g_vv_pp_U[:, :, :O, :].transpose(0, 1, 3, 2)
        u_IIp_full = s32 * (g_abpi - g_abip_T)
        u_IIIp_full = s12 * (g_abpi + g_abip_T)
        if not is_adc2x:
            u_IIp_full += s32 * np.einsum('abkl,klpi->abpi', X_abkl, g_klpi_anti, optimize=True)
            Yp1 = np.einsum('acki,kbcp->abpi', A_mix_p, g_ovvp, optimize=True)
            Yp2 = np.einsum('bcki,kacp->bapi', A_mix_p, g_ovvp, optimize=True).transpose(1, 0, 2, 3)
            u_IIp_full += s32 * np.einsum('acki,kbpc->abpi', A_anti_p, g_ovpv, optimize=True)
            u_IIp_full -= s32 * Yp1
            u_IIp_full -= s32 * np.einsum('bcki,kapc->bapi', A_anti_p, g_ovpv, optimize=True).transpose(1, 0, 2, 3)
            u_IIp_full += s32 * Yp2
            u_IIIp_full += s12 * np.einsum('abkl,klpi->abpi', X_abkl, g_klpi_sym_l, optimize=True)
            u_IIIp_full += s12 * np.einsum('bcki,kapc->bapi', W1p, g_ovpv, optimize=True).transpose(1, 0, 2, 3)
            u_IIIp_full += s12 * Yp2
            u_IIIp_full += s12 * np.einsum('acki,kbpc->abpi', W1p, g_ovpv, optimize=True)
            u_IIIp_full += s12 * Yp1
        _U_IIp = u_IIp_full[iu_v, ju_v].transpose(2, 0, 1).transpose(2, 0, 1).reshape(norb, nIIp)
        _U_IIIp = u_IIIp_full[iu_v, ju_v].transpose(2, 0, 1).transpose(2, 0, 1).reshape(norb, nIIIp)
        del u_IIp_full, u_IIIp_full

        def apply_U_2p1h_fwd(z_IIp, z_IIIp):
            return _U_IIp @ z_IIp.reshape(-1) + _U_IIIp @ z_IIIp.reshape(-1)

        def apply_U_2p1h_adj(z_p):
            return (_U_IIp.T @ z_p).reshape(O, nP_v), (_U_IIIp.T @ z_p).reshape(O, nP_v)
    else:
        def apply_U_2p1h_fwd(z_IIp, z_IIIp):
            """y_p += U_IIp @ z_IIp + U_IIIp @ z_IIIp."""
            y = np.zeros(norb)
            for (sc, gkey, akey, hsign, anti), z in zip(_2p1h_specs, (z_IIp, z_IIIp)):
                Z = _u_2p1h_unfold(z, iu_v, ju_v, O, V, anti)
                # bare 1st-order piece: g_abpi = sum_Q B[Q,a,p] B[Q,b,i]
                M = _cached_einsum('Qbi,iab->Qa', Bvo_full, Z, optimize=True)
                y += sc * _b_chunk_einsum('Qap,Qa->p', (Bv_full, M), (True, True), Bv_full)
                if is_adc2x:
                    continue
                P = _cached_einsum('abkl,iab->kli', X_abkl, Z, optimize=True)
                y += 0.5 * sc * _cached_einsum('klpi,kli->p', _gk[gkey], P, optimize=True)
                Y = _cached_einsum('acki,iab->ckib', _amps[akey], Z, optimize=True)
                R = _b_chunk_einsum('ckib,Qbc->Qk', (Y, Bv_pp), (False, True),
                                    Bv_pp, (Bv_pp.shape[0], O))
                y += sc * _b_chunk_einsum('Qk,Qkp->p', (R, Bo_full), (True, True), Bo_full)
                Y2 = _cached_einsum('acki,iab->ckib', A_mix_p, Z, optimize=True)
                R2 = _b_chunk_einsum('ckib,Qkc->Qb', (Y2, Bov_full), (False, True),
                                     Bov_full, (Bov_full.shape[0], V))
                y += hsign * sc * _b_chunk_einsum('Qb,Qbp->p', (R2, Bv_full), (True, True), Bv_full)
            return y

        def apply_U_2p1h_adj(z_p):
            """(U_IIp.T @ z_p, U_IIIp.T @ z_p), each returned as (O, nP_v)."""
            vidx_abs = np.arange(O, norb)
            oidx = np.arange(O)
            gz, gzr, gzm = _u_2p1h_zparts_df(B, O, norb, vidx_abs, oidx, z_p)
            out = []
            for sc, gkey, akey, hsign, anti in _2p1h_specs:
                sgn = -1.0 if anti else 1.0
                tot = sc * (gz + sgn * gz.transpose(1, 0, 2))
                if not is_adc2x:
                    gkz = _cached_einsum('klpi,p->kli', _gk[gkey], z_p, optimize=True)
                    tot = tot + sc * _cached_einsum('abkl,kli->abi', X_abkl, gkz, optimize=True)
                    Gz = _cached_einsum('acki,kbc->abi', _amps[akey], gzr, optimize=True)
                    tot = tot + sc * (Gz + sgn * Gz.transpose(1, 0, 2))
                    Hz = _cached_einsum('acki,kbc->abi', A_mix_p, gzm, optimize=True)
                    tot = tot + hsign * sc * (Hz + sgn * Hz.transpose(1, 0, 2))
                out.append(tot.transpose(2, 0, 1)[:, iu_v, ju_v])
            return out

    return U_I, U_Ip, apply_U_2h1p_fwd, apply_U_2h1p_adj, apply_U_2p1h_fwd, apply_U_2p1h_adj

"""The MP1 amplitude pieces every correlated package reads, in one low module
that imports nothing from ADC, DensityMatrix or EpsteinNesbet (they import
from it, never from each other's private names):

- orbital_denominator: sum(eps_occ) - sum(eps_vir) of any rank, in the
  [vir..., occ...] storage of t{rank}; orbital_denominator_axes is the same
  sum with one energy array per axis (spin-resolved blocks);
- to_l: the t{rank} -> l{rank} axis convention of the generated equations;
- u2_2h1p/2p1h_amplitude_chunks: the restricted DF U^(2) amplitudes,
  streamed over one occupied axis.
"""
import numpy as np

from src.SingleReference.common.eri_blocks import g_slice_df


def orbital_denominator_axes(eps_o_axes, eps_v_axes):
    """D[a,b,...,i,j,...] = sum_r eps_o_axes[r][i_r] - sum_r eps_v_axes[r][a_r]:
    len(eps_v_axes) virtual axes (leading) then len(eps_o_axes) occupied axes
    (trailing), the occupied energies added first, axis by axis, then the
    virtual ones subtracted."""
    rank = len(eps_o_axes)
    d = np.zeros([len(e) for e in eps_v_axes] + [len(e) for e in eps_o_axes])
    for r, e in enumerate(eps_o_axes):
        shape = [1] * (2 * rank)
        shape[rank + r] = len(e)
        d = d + e.reshape(shape)
    for r, e in enumerate(eps_v_axes):
        shape = [1] * (2 * rank)
        shape[r] = len(e)
        d = d - e.reshape(shape)
    return d


def orbital_denominator(eps_o, eps_v, rank):
    """D[a,b,...,i,j,...] = sum(eps_occ) - sum(eps_vir), rank virtual axes
    (leading) then rank occupied axes (trailing) -- t{rank}'s own
    [vir..., occ...] storage convention."""
    return orbital_denominator_axes([eps_o] * rank, [eps_v] * rank)


def to_l(t_arr, rank, block=None):
    """A t{rank}-convention array (axes [vir_asc(rank), occ_asc(rank)],
    antisymmetric under any same-type pair swap) in the l{rank} convention
    the generator's 'l{rank}' tag expects.

    Same-spin (block None, spin-orbital, or one spin letter throughout, e.g.
    'aaaa'): axes [occ_asc(rank), vir_DESC(rank)], with the sign
    (-1)**(rank//2) of reversing the virtual block (n//2 transpositions
    reverse n elements). Mixed-spin blocks (e.g. 'abab', 'aabaab'; occ spins
    then vir spins, each rank letters): the dummy letters do not carry the
    same alpha/beta axis assignment between l and t, so the virtual block is
    NOT reversed and the sign is +1 (confirmed against the spin-orbital
    pipeline folded to alpha on LiH/sto-3g, where the reversed transpose is
    subtly wrong, not just sign-flipped)."""
    if block is None or len(set(block)) == 1:
        axes = list(range(rank, 2 * rank)) + list(range(rank - 1, -1, -1))
        sign = (-1) ** (rank // 2)
    else:
        axes = list(range(rank, 2 * rank)) + list(range(rank))
        sign = 1
    return sign * t_arr.transpose(*axes)


# ===================== S4: occupied-index streaming (bare + hh/pp-dressed EN) =====================
# The amplitudes below (each O^2V^2, held live for the whole aop lifetime)
# are built chunk by chunk, streaming over one occupied axis, never as full
# arrays. Bare EN: D is a trivial broadcast, so streaming is free. Dressed
# hh/pp (spin_adapted=True, production): the shift is a sum of broadcasts of
# small (O,O)/(V,V) matrices, equally separable, so chunking is just slicing
# their matching axis. 'hp' and determinant-wise (spin_adapted=False) EN are
# EXCLUDED here -- their shifts don't reduce to a small-matrix broadcast per
# axis, so callers must not pass them through dh/dp.
def u2_2h1p_amplitude_chunks(B, O, V, norb, eps_o, eps_v, chunk_size, dh=None, dp=None):
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
        g_chunk = g_slice_df(B, occ_chunk, occ, vir, vir)       # (ci,O,V,V)
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


def u2_2p1h_amplitude_chunks(B, O, V, norb, eps_o, eps_v, chunk_size, dh=None, dp=None):
    """Yields (lo, hi, tp_same, tp_opp, W1p, W2p, X_abkl) chunks, each shape
    (V, V, O, hi-lo), streaming over the LAST occupied axis (l) of the
    'ackl'/'abkl' amplitudes -- the axis every consumer (U_Ip's ring build,
    apply_U_2p1h_fwd/adj) treats as the batch/output index (its OWN axis3,
    renamed 'i' at consumption).

    dh/dp: same EN hh/pp shift matrices as u2_2h1p_amplitude_chunks (see
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
        g_direct = g_slice_df(B, vir, vir, occ, occ_chunk)                      # g[a,c,k,l_chunk]
        g_exch = g_slice_df(B, vir, vir, occ_chunk, occ).transpose(0, 1, 3, 2)  # g[a,c,l_chunk,k]->[a,c,k,l_chunk]
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

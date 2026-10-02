"""Shared spin-agnostic helpers: DF integral gathers and chunked
B-factor einsum plumbing. Used by both the restricted (adc_r_*) and
spin-orbital (adc_u_*) route modules."""
import numpy as np

from src.Base.eri_blocks import g_slice_df as _g_slice_df
from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum


def _g_diag_df(B, diag_idx, r_idx, s_idx):
    """
    DF/RI build of g[diag_idx[i], diag_idx[i], r_idx, s_idx] (both of
    g[p,q,...]'s leading axes pinned to the SAME index i, e.g. g_ii/g_aa's
    'i,i,:,:' diagonal gather)
    """
    Bd_r = B[:, diag_idx][:, :, r_idx]   # (naux, |diag|, |r|)
    Bd_s = B[:, diag_idx][:, :, s_idx]   # (naux, |diag|, |s|)
    return np.einsum('Qir,Qis->irs', Bd_r, Bd_s, optimize=True)


def _g_diag_mid_df(B, p_idx, diag_idx, s_idx):
    """
    DF/RI build of g[p, i, i, s] = sum_Q B[Q,p,i]*B[Q,i,s] (both of g's
    MIDDLE two axes pinned to the same index i, e.g. g_voov's j=i diagonal
    (C_I_I's g_ciia) or g_ovvo's c=a diagonal (C_Ip_Ip's g_kaai)). Small
    (O(naux*|p|*|diag|*|s|)) -- never materializes the full (|p|,|diag|,
    |diag|,|s|) array.
    """
    Bp_d = B[:, p_idx][:, :, diag_idx]   # (naux, |p|, |diag|)
    Bd_s = B[:, diag_idx][:, :, s_idx]   # (naux, |diag|, |s|)
    return np.einsum('Qpi,Qis->pis', Bp_d, Bd_s, optimize=True)


def _g_diag_outer_df(B, p_idx, diag_idx, r_idx):
    """
    DF/RI build of g[p, i, r, i] = sum_Q B[Q,p,r]*B[Q,i,i] (both of g's
    OUTER two axes pinned to the same index i, e.g. g_vovo's j=i diagonal
    (C_I_I's g_ciai) or g_ovov's c=a diagonal (C_Ip_Ip's g_kaia)). Small,
    same rationale as _g_diag_mid_df.
    """
    Bpr = B[:, p_idx][:, :, r_idx]                              # (naux, |p|, |r|)
    Bdd = np.einsum('Qii->Qi', B[:, diag_idx][:, :, diag_idx])   # (naux, |diag|)
    return np.einsum('Qpr,Qi->pir', Bpr, Bdd, optimize=True)


def _g_block_df(B, p_idx, q_idx, r_idx, s_idx):
    """DF/RI antisymmetrized block <p_idx,q_idx||r_idx,s_idx> from a
    blockstacked spin-orbital DF factor B (naux, nso, nso). Rank-4 output --
    only used for the one-off bare-T2^(1) amplitude built once per nocc;
    every per-matvec term instead uses rank<=3 B-factor slices."""
    Bpr = B[:, p_idx][:, :, r_idx]
    Bqs = B[:, q_idx][:, :, s_idx]
    Bps = B[:, p_idx][:, :, s_idx]
    Bqr = B[:, q_idx][:, :, r_idx]
    direct = _cached_einsum('Qpr,Qqs->pqrs', Bpr, Bqs, optimize=True)
    exchange = _cached_einsum('Qps,Qqr->pqrs', Bps, Bqr, optimize=True)
    return direct - exchange


#: Target size of one Q-slab when a DF-factor VIEW has to be chunked. Below
#: this the whole view is used directly (chunking a small factor is pure
#: loop overhead); above it, an unchunked call would materialize the full
#: contiguous copy the view exists to avoid.
_B_SLAB_BYTES = 1 << 30


def _b_factor(B, row_idx, col_idx, view):
    """A DF-factor slice B[:, row_idx, col_idx]: a copy via fancy indexing
    when small, the bare strided view when large (a copy at that size would
    duplicate a large fraction of the whole factor). The small branch
    reproduces fancy indexing's own memory layout rather than a generic
    contiguous copy -- the two are not equivalent for BLAS throughput on the
    dozens of contractions these factors feed."""
    if view.nbytes > _B_SLAB_BYTES:
        return view
    return B[:, row_idx][:, :, col_idx]


def _b_q_chunk(Bx):
    """Q-chunk size for a DF-factor view: no chunking when the whole factor is
    small, else the largest chunk whose contiguous copy fits _B_SLAB_BYTES."""
    if Bx.nbytes <= _B_SLAB_BYTES:
        return Bx.shape[0]
    per_q = max(1, Bx.nbytes // Bx.shape[0])
    return max(1, int(_B_SLAB_BYTES // per_q))


def _b_chunk_einsum(subs, ops, has_q, Bref, out_shape=None):
    """einsum over operands where one or more carry the auxiliary index Q,
    chunked over Q when the DF factor Bref is large. Preserves the caller's
    operand order verbatim -- reordering changes which operand BLAS has to
    transpose and costs real time.

    has_q marks which operands carry Q (those get sliced; the rest are passed
    whole). out_shape=None means Q is summed away, so chunks accumulate;
    otherwise the output still carries Q on axis 0 and each chunk fills its
    own slice."""
    q_chunk = _b_q_chunk(Bref)
    if q_chunk >= Bref.shape[0]:
        return _cached_einsum(subs, *ops, optimize=True)
    acc = None if out_shape is None else np.empty(out_shape)
    for st in range(0, Bref.shape[0], q_chunk):
        sl = slice(st, st + q_chunk)
        chunk = [o[sl] if hq else o for o, hq in zip(ops, has_q)]
        part = _cached_einsum(subs, *chunk, optimize=True)
        if out_shape is None:
            acc = part if acc is None else acc + part
        else:
            acc[sl] = part
    return acc

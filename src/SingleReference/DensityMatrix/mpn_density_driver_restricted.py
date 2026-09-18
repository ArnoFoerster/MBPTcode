"""Restricted (spin-blocked, closed-shell RHF) analogue of
mpn_density_driver.py -- same recursion (see that module's docstring and
the generator), wiring together
src/SingleReference/DensityMatrix/generated_mpn_restricted/mpn_density_pieces_restricted.py
instead of the spin-orbital generated_mpn/mpn_density_pieces.py, and reusing
this repo's existing restricted CC infrastructure for denominators
(src/SingleReference/CC/integrals.py::energy_denominators, the same
1/(eps_occ-eps_vir)-type helper restricted_solver.py itself uses) rather than
reimplementing it.

l-tag axis-convention note: the l{rank} tag conversion turned out to
depend on whether a spin block is "pure" (all letters equal, e.g. 'aaaa',
'aaaaaa') or "mixed" (e.g. 'abab', 'aabaab', 'abbabb') -- these need
genuinely DIFFERENT transposes, not just different signs:

- Pure blocks: same as the spin-orbital case (mpn_density_driver.py::_to_l)
  -- axis layout (occ ascending, vir REVERSED) with sign (-1)**(rank//2), the
  parity of reversing `rank` genuinely-antisymmetric (same-spin) indices.
- Mixed blocks: NO vir reversal at all, sign always +1 -- l2_abab[i,j,a,b] =
  t2_abab[a,b,i,j], a plain occ<->vir side transpose with no internal
  reordering. Reasoning: alpha and beta virtuals are NOT interchangeable
  (different spin spaces), so there is no antisymmetry operation to encode
  by reversing their order -- the generator internal axis convention for a
  mixed tag apparently keeps the (alpha, beta) order fixed regardless of l/t
  role, unlike a pure tag's genuinely reversed printing convention.

This was NOT obvious from reading the raw generated term output alone --
overlap2_restricted's abab term happens to print with the SAME textual
subscript pattern as its aaaa term ('ijba,baij->'), which looks like it
implies the same reversed convention, but is misleading: confirmed by direct
numeric test that this pattern's dummy letters do not carry the same
alpha/beta axis assignment between l2 and t2 for a mixed block, so applying
the "reversed" transpose to abab silently produces a matrix that is subtly
WRONG (not just sign-flipped) -- caught only via a matrix-level (not scalar
overlap-level) comparison against MPnDensityDriver's independently-validated
spin-orbital pipeline folded to alpha, on a molecule (LiH/sto-3g) with
nontrivial virtual-orbital mixing (HF/sto-3g's simpler virtual structure
didn't expose it: max diff was exactly 0, not just small). The "unreversed,
sign=+1" convention below was confirmed to reproduce that same oracle's N^(2)
value bit-for-bit (0.008944961578635474 both), not just approximately.
"""
import inspect

import numpy as np

from src.SingleReference.DensityMatrix.generated_mpn_restricted import mpn_density_pieces_restricted as gen
from src.SingleReference.DensityMatrix.generated_mpn_restricted import mpn_density_pieces_restricted_df as gen_df
from src.SingleReference.DensityMatrix.generated_mpn_restricted import mp4_laplace_restricted as gen_lap4
from src.SingleReference.DensityMatrix.generated_mpn_restricted import mp4_t2_3_laplace_restricted as gen_lap_t23
from src.SingleReference.DensityMatrix.generated_mpn_restricted import mp4_t1_4_laplace_restricted as gen_lap_t14
from src.SingleReference.DensityMatrix.generated_mpn_restricted import mp3_t3_laplace_restricted as gen_lap3
from src.SingleReference.DensityMatrix.generated_mpn_restricted import mp3_t3_laplace_restricted_df as gen_lap3_df
from src.Base.utils.grids import minimax_time_grid


def _denom_restricted(eps_a, rank, no, nv):
    """D[a,b,...,i,j,...] = sum(eps_occ) - sum(eps_vir), rank virtual axes
    (leading) then rank occupied axes (trailing) -- restricted analogue of
    mpn_density_driver.py's _denom, generalized beyond integrals.py's
    energy_denominators (which only goes up to rank 3) for T4^(2)/T4^(3)."""
    eps_o, eps_v = eps_a[:no], eps_a[no:no + nv]
    d = np.zeros((nv,) * rank + (no,) * rank)
    for r in range(rank):
        shape = [1] * (2 * rank)
        shape[rank + r] = no
        d = d + eps_o.reshape(shape)
    for r in range(rank):
        shape = [1] * (2 * rank)
        shape[r] = nv
        d = d - eps_v.reshape(shape)
    return d


def _to_l_restricted(t_arr, rank, block):
    """See module docstring. `block` is e.g. 'aaaa', 'abab', 'aabaab' --
    T_BRA[rank][1]'s own key naming (occ spins then vir spins, concatenated,
    each `rank` characters long)."""
    if len(set(block)) == 1:
        axes = list(range(rank, 2 * rank)) + list(range(rank - 1, -1, -1))
        sign = (-1) ** (rank // 2)
    else:
        axes = list(range(rank, 2 * rank)) + list(range(rank))
        sign = 1
    return sign * t_arr.transpose(*axes)


def _laplace_aaaaaa_contribution(ei_a, ea_a, g_aaaa, o, v, t2_1_aaaa, ntau):
    """Laplace-quadrature replacement for T3^(2)_aaaaaa's contribution to
    (t1_3_aa's numerator, the ov M3 block), without ever forming the six-index
    t3_aaaaaa tensor -- direct port of
    density_matrix.py::MP3DensityMatrixSolverUnrestricted._laplace_aaa_contribution
    to this module's generic generated pipeline, "same trick" per
    that method's docstring (see it for the full derivation and validation
    history: 1/D3 = -sum_tau sigma_tau*exp(D3*tau) via GreenX minimax
    quadrature, D3's additive separability across the six external indices
    lets exp(D3*tau) factorize into per-orbital dressings applied to the
    numerator's own inputs before contracting).

    Confirmed applicable unchanged: the generator
    generated t3_2_aaaaaa_numerator (this module's
    generated_mpn_restricted/mpn_density_pieces_restricted.py) has the exact
    same 8 raw terms, same letters, same signs as the hand-derived
    compute_t3_second_order this method's original was built against --
    verified via the generator own generation log and
    this module's own n=3 validation (tests/test_mpn_density_restricted.py).

    Returns (t1_3_aa_numerator_contrib, ov_a_contrib) -- t1_3_aa_numerator_contrib
    is added to t1_3_aa's numerator (pre-division by e_ai, matching
    t1_3_aa_numerator's own _tmp6 term it replaces); ov_a_contrib is added
    directly to the final ov block (matching m3_ov_12_restricted's own
    aaaaaa-only nonzero term it replaces, already at the correct final scale,
    no extra division).

    Only the pure-same-spin (aaaaaa) sector is accelerated here, exactly
    mirroring the original: the aabaab/abbabb cross-spin sector is NOT yet
    Laplace-accelerated and t1_3_aa_numerator still forms t3_2_aabaab/
    t3_2_abbabb explicitly for those terms.
    """
    no_a, nv_a = ei_a.shape[0], ea_a.shape[0]
    g_aaaa_oooo = g_aaaa[o, o, v, v]
    g_vvvo = g_aaaa[v, v, v, o]
    g_ovoo = g_aaaa[o, v, o, o]

    gap_min = max(ea_a.min() - ei_a.max(), 1e-3)
    gap_max = ea_a.max() - ei_a.min()
    e_min, e_max = 3.0 * gap_min, 3.0 * gap_max
    tau, sigma = minimax_time_grid(ntau, e_min, e_max)

    outer1 = g_aaaa_oooo   # (k,j,b,c) -- feeds t1_3_aa
    outer2 = t2_1_aaaa     # (b,c,j,k) -- feeds the ov term
    t1_3_aa_contrib = np.zeros((no_a, nv_a))
    ov_a_contrib = np.zeros((no_a, nv_a))

    for tk in range(ntau):
        t = tau[tk]
        w = -sigma[tk]  # 1/D = -sum_tau sigma_tau * exp(D*tau)
        Oe = np.exp(ei_a * t)
        Ve = np.exp(-ea_a * t)
        N_outer1 = np.zeros((no_a, nv_a))
        N_outer2 = np.zeros((no_a, nv_a))

        G1d = g_ovoo * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
        T1d = t2_1_aaaa * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, :, None]
        p1 = -np.einsum('kjbc,lajk,bcil->ia', outer1, G1d, T1d, optimize=True)
        p2 = -np.einsum('kjbc,laik,bcjl->ia', outer1, G1d, T1d, optimize=True)
        p3 = -np.einsum('kjbc,lbjk,acil->ia', outer1, G1d, T1d, optimize=True)
        p4 = -np.einsum('kjbc,lbik,acjl->ia', outer1, G1d, T1d, optimize=True)
        N_outer1 += p1 - p2 - p3 + p4
        q1 = -np.einsum('bcjk,lajk,bcil->ia', outer2, G1d, T1d, optimize=True)
        q2 = -np.einsum('bcjk,laik,bcjl->ia', outer2, G1d, T1d, optimize=True)
        q3 = -np.einsum('bcjk,lbjk,acil->ia', outer2, G1d, T1d, optimize=True)
        q4 = -np.einsum('bcjk,lbik,acjl->ia', outer2, G1d, T1d, optimize=True)
        N_outer2 += q1 - q2 - q3 + q4

        r1 = -np.einsum('kjbc,laij,bckl->ia', outer1, G1d, T1d, optimize=True)
        r2 = -np.einsum('kjbc,lbij,ackl->ia', outer1, G1d, T1d, optimize=True)
        N_outer1 += r1 - r2
        s1 = -np.einsum('bcjk,laij,bckl->ia', outer2, G1d, T1d, optimize=True)
        s2 = -np.einsum('bcjk,lbij,ackl->ia', outer2, G1d, T1d, optimize=True)
        N_outer2 += s1 - s2

        u1 = -np.einsum('kjbc,lcjk,abil->ia', outer1, G1d, T1d, optimize=True)
        u2 = -np.einsum('kjbc,lcik,abjl->ia', outer1, G1d, T1d, optimize=True)
        N_outer1 += u1 - u2
        v1 = -np.einsum('bcjk,lcjk,abil->ia', outer2, G1d, T1d, optimize=True)
        v2 = -np.einsum('bcjk,lcik,abjl->ia', outer2, G1d, T1d, optimize=True)
        N_outer2 += v1 - v2

        N_outer1 += -np.einsum('kjbc,lcij,abkl->ia', outer1, G1d, T1d, optimize=True)
        N_outer2 += -np.einsum('bcjk,lcij,abkl->ia', outer2, G1d, T1d, optimize=True)

        G5d = g_vvvo * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, None, :]
        T5d = t2_1_aaaa * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
        N_outer1 += 4.0 * (-np.einsum('kjbc,abdk,dcij->ia', outer1, G5d, T5d, optimize=True))
        N_outer2 += 4.0 * (-np.einsum('bcjk,abdk,dcij->ia', outer2, G5d, T5d, optimize=True))

        N_outer1 += 2.0 * (-np.einsum('kjbc,abdi,dcjk->ia', outer1, G5d, T5d, optimize=True))
        N_outer2 += 2.0 * (-np.einsum('bcjk,abdi,dcjk->ia', outer2, G5d, T5d, optimize=True))

        N_outer1 += 2.0 * (-np.einsum('kjbc,bcdk,daij->ia', outer1, G5d, T5d, optimize=True))
        N_outer2 += 2.0 * (-np.einsum('bcjk,bcdk,daij->ia', outer2, G5d, T5d, optimize=True))

        N_outer1 += -np.einsum('kjbc,bcdi,dajk->ia', outer1, G5d, T5d, optimize=True)
        N_outer2 += -np.einsum('bcjk,bcdi,dajk->ia', outer2, G5d, T5d, optimize=True)

        t1_3_aa_contrib += w * (-0.25) * N_outer1
        ov_a_contrib += w * (0.25) * N_outer2

    return t1_3_aa_contrib, ov_a_contrib


def _g_df(B, p_idx, q_idx, r_idx, s_idx):
    """self.g_aaaa[p,q,r,s] = sum_Q B[Q,p,r]*B[Q,q,s] - sum_Q B[Q,p,s]*B[Q,q,r]
    unlike src.SingleReference.ADC.adc_restricted._g_slice_df (which reconstructs
    the BARE, non-antisymmetrized Coulomb integral, matching that module's
    own eri_chemist.transpose(...) convention), this driver's g_aaaa is
    ALREADY antisymmetrized-within-spin (get_antisymmetrized_spin_block_eri's
    own convention -- <pq||rs>, exchange term included), so this helper must
    include the exchange subtraction too, or it silently reproduces only the
    Coulomb half of g_aaaa (caught by test_mp3_density_df.py's exact-
    plumbing check: the bare-Coulomb version was ~0.1-0.4 absolute off on
    every one of the three slices this function builds, non-negligible
    error that only showed up once LiH/H2O exercised a nontrivial aaaaaa
    T3^(2) sector -- HF/sto-3g's single occupied orbital happened to mask
    it). p_idx/q_idx/r_idx/s_idx may be plain Python slice objects (this
    module's own o/v convention) or integer index arrays."""
    Bpr = B[:, p_idx][:, :, r_idx]
    Bqs = B[:, q_idx][:, :, s_idx]
    Bps = B[:, p_idx][:, :, s_idx]
    Bqr = B[:, q_idx][:, :, r_idx]
    return (np.einsum('Qpr,Qqs->pqrs', Bpr, Bqs, optimize=True)
            - np.einsum('Qps,Qqr->pqrs', Bps, Bqr, optimize=True))


def _laplace_aaaaaa_contribution_df(ei_a, ea_a, B_aa, o, v, t2_1_aaaa, ntau):
    """DF/RI variant of _laplace_aaaaaa_contribution
    identical physics/derivation -- only the
    three g_aaaa slices this function's own preamble builds are replaced by
    DF contractions from B_aa via _g_df; every line after that (the whole
    tau-quadrature loop) is byte-identical to the dense version, since it
    only ever operates on the already-materialized slice arrays
    (g_aaaa_oooo/g_vvvo/g_ovoo), never on g_aaaa itself.
    """
    no_a, nv_a = ei_a.shape[0], ea_a.shape[0]
    g_aaaa_oooo = _g_df(B_aa, o, o, v, v)
    g_vvvo = _g_df(B_aa, v, v, v, o)
    g_ovoo = _g_df(B_aa, o, v, o, o)

    gap_min = max(ea_a.min() - ei_a.max(), 1e-3)
    gap_max = ea_a.max() - ei_a.min()
    e_min, e_max = 3.0 * gap_min, 3.0 * gap_max
    tau, sigma = minimax_time_grid(ntau, e_min, e_max)

    outer1 = g_aaaa_oooo   # (k,j,b,c) -- feeds t1_3_aa
    outer2 = t2_1_aaaa     # (b,c,j,k) -- feeds the ov term
    t1_3_aa_contrib = np.zeros((no_a, nv_a))
    ov_a_contrib = np.zeros((no_a, nv_a))

    for tk in range(ntau):
        t = tau[tk]
        w = -sigma[tk]  # 1/D = -sum_tau sigma_tau * exp(D*tau)
        Oe = np.exp(ei_a * t)
        Ve = np.exp(-ea_a * t)
        N_outer1 = np.zeros((no_a, nv_a))
        N_outer2 = np.zeros((no_a, nv_a))

        G1d = g_ovoo * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
        T1d = t2_1_aaaa * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, :, None]
        p1 = -np.einsum('kjbc,lajk,bcil->ia', outer1, G1d, T1d, optimize=True)
        p2 = -np.einsum('kjbc,laik,bcjl->ia', outer1, G1d, T1d, optimize=True)
        p3 = -np.einsum('kjbc,lbjk,acil->ia', outer1, G1d, T1d, optimize=True)
        p4 = -np.einsum('kjbc,lbik,acjl->ia', outer1, G1d, T1d, optimize=True)
        N_outer1 += p1 - p2 - p3 + p4
        q1 = -np.einsum('bcjk,lajk,bcil->ia', outer2, G1d, T1d, optimize=True)
        q2 = -np.einsum('bcjk,laik,bcjl->ia', outer2, G1d, T1d, optimize=True)
        q3 = -np.einsum('bcjk,lbjk,acil->ia', outer2, G1d, T1d, optimize=True)
        q4 = -np.einsum('bcjk,lbik,acjl->ia', outer2, G1d, T1d, optimize=True)
        N_outer2 += q1 - q2 - q3 + q4

        r1 = -np.einsum('kjbc,laij,bckl->ia', outer1, G1d, T1d, optimize=True)
        r2 = -np.einsum('kjbc,lbij,ackl->ia', outer1, G1d, T1d, optimize=True)
        N_outer1 += r1 - r2
        s1 = -np.einsum('bcjk,laij,bckl->ia', outer2, G1d, T1d, optimize=True)
        s2 = -np.einsum('bcjk,lbij,ackl->ia', outer2, G1d, T1d, optimize=True)
        N_outer2 += s1 - s2

        u1 = -np.einsum('kjbc,lcjk,abil->ia', outer1, G1d, T1d, optimize=True)
        u2 = -np.einsum('kjbc,lcik,abjl->ia', outer1, G1d, T1d, optimize=True)
        N_outer1 += u1 - u2
        v1 = -np.einsum('bcjk,lcjk,abil->ia', outer2, G1d, T1d, optimize=True)
        v2 = -np.einsum('bcjk,lcik,abjl->ia', outer2, G1d, T1d, optimize=True)
        N_outer2 += v1 - v2

        N_outer1 += -np.einsum('kjbc,lcij,abkl->ia', outer1, G1d, T1d, optimize=True)
        N_outer2 += -np.einsum('bcjk,lcij,abkl->ia', outer2, G1d, T1d, optimize=True)

        G5d = g_vvvo * Ve[:, None, None, None] * Ve[None, :, None, None] * Oe[None, None, None, :]
        T5d = t2_1_aaaa * Ve[None, :, None, None] * Oe[None, None, :, None] * Oe[None, None, None, :]
        N_outer1 += 4.0 * (-np.einsum('kjbc,abdk,dcij->ia', outer1, G5d, T5d, optimize=True))
        N_outer2 += 4.0 * (-np.einsum('bcjk,abdk,dcij->ia', outer2, G5d, T5d, optimize=True))

        N_outer1 += 2.0 * (-np.einsum('kjbc,abdi,dcjk->ia', outer1, G5d, T5d, optimize=True))
        N_outer2 += 2.0 * (-np.einsum('bcjk,abdi,dcjk->ia', outer2, G5d, T5d, optimize=True))

        N_outer1 += 2.0 * (-np.einsum('kjbc,bcdk,daij->ia', outer1, G5d, T5d, optimize=True))
        N_outer2 += 2.0 * (-np.einsum('bcjk,bcdk,daij->ia', outer2, G5d, T5d, optimize=True))

        N_outer1 += -np.einsum('kjbc,bcdi,dajk->ia', outer1, G5d, T5d, optimize=True)
        N_outer2 += -np.einsum('bcjk,bcdi,dajk->ia', outer2, G5d, T5d, optimize=True)

        t1_3_aa_contrib += w * (-0.25) * N_outer1
        ov_a_contrib += w * (0.25) * N_outer2

    return t1_3_aa_contrib, ov_a_contrib


def compute_delta_gamma2_df_streamed(B_aa, eps, nocc, dh=None, dp=None, e_ai=None,
                                     chunk_size=4):
    """(oo, ov, vv): MPnDensityDriverRestricted.compute_delta_gamma2_df's
    result without EVER materializing an O^2V^2 (rank-4) array -- amplitudes
    included, not just integrals (which is also why this is a module
    function, not a driver method: constructing the driver itself eagerly
    builds the (V,V,O,O) bare e_abij). Streams occupied chunks of the
    T2^(1) amplitudes straight from B_aa via the ADC module's
    _u2_2p1h_amplitude_chunks (lazy import; no module-level cycle) and
    accumulates only rank<=2 outputs plus three persistent (naux,V,O)
    transients. Bit-identical to compute_delta_gamma2_df, bare or EN
    hh/pp-dressed.

    eps: (norb,) orbital energies (NOT the diag matrix the driver takes).
    dh/dp: optional spin-adapted EN hh/pp shift matrices ((O,O)/(V,V),
    restricted_channel_shifts' d_h[0]/d_p[0]) -- replaces the
    _build_dressed_e_abij (V,V,O,O) inverse-denominator array outright.
    e_ai: optional dressed (V,O) singles denominator (_build_dressed_e_ai);
    bare when None. 'hp'/spin-resolved dressing is NOT representable here --
    callers must use the materialized compute_delta_gamma2_df for those
    (same policy/exception as the ADC solver's _dress_is_streamable gate).

    Index algebra (l2 = sign*transpose of t2 absorbed into the subscripts;
    terms sharing t2's FIRST occupied axis converted to the chunked LAST
    axis via t2_aaaa's occupied antisymmetry; all verified numerically):

        N2       = 0.5*<t2_aaaa,t2_aaaa> + <t2_abab,t2_abab>
        m2_oo_11 = N2*I - 0.5*einsum('abni,abmi->mn', t2_aaaa, t2_aaaa)
                        - 1.0*einsum('bani,bami->mn', t2_abab, t2_abab)
        m2_vv_11 = 0.5*einsum('eaij,faij->ef', t2_aaaa, t2_aaaa)
                 + 1.0*einsum('eaij,faij->ef', t2_abab, t2_abab)

    and m2_oo_11's +N2*I trace cancels c2*I = -N2*I exactly in the oo
    block, so neither is formed. Sign convention: the chunk generator's
    tp_same/tp_opp carry the particle-minus-hole denominator, so t2 = -tp
    (cancels in the bilinears, explicit in the linear t1_2 numerator
    terms, which reproduce gen_df.t1_2_aa_numerator_df's six terms
    chunk-by-chunk)."""
    # call-time import: ADC/__init__ imports base, whose chain reaches this
    # module through EpsteinNesbet before its names exist
    from src.SingleReference.ADC.adc_r_utils import _u2_2p1h_amplitude_chunks
    norb = B_aa.shape[1]
    no, nv = nocc, norb - nocc
    eps_o, eps_v = eps[:no], eps[no:]
    o, v = slice(0, no), slice(no, norb)
    naux = B_aa.shape[0]
    B_ov = B_aa[:, o, v]
    B_oo = B_aa[:, o, o]
    B_vv = B_aa[:, v, v]

    tmp3_oo = np.zeros((no, no))
    tmp4_oo = np.zeros((no, no))
    vv_acc = np.zeros((nv, nv))
    num_t1 = np.zeros((nv, no))
    T3_acc = np.zeros((naux, nv, no))
    T4_acc = np.zeros((naux, nv, no))
    T5_acc = np.zeros((naux, nv, no))

    for lo, hi, tp_same, tp_opp, W1p, W2p, X_abkl in _u2_2p1h_amplitude_chunks(
            B_aa, no, nv, norb, eps_o, eps_v, chunk_size, dh=dh, dp=dp):
        t2a = -tp_same            # t2_aaaa[:, :, :, lo:hi]
        t2b = -tp_opp             # t2_abab[:, :, :, lo:hi]

        tmp3_oo += np.einsum('abni,abmi->mn', t2a, t2a, optimize=True)
        tmp4_oo += np.einsum('bani,bami->mn', t2b, t2b, optimize=True)
        vv_acc += 0.5 * np.einsum('eaij,faij->ef', t2a, t2a, optimize=True)
        vv_acc += 1.0 * np.einsum('eaij,faij->ef', t2b, t2b, optimize=True)

        Bov_j = B_ov[:, lo:hi, :]
        Boo_j = B_oo[:, lo:hi, :]
        num_t1 -= 0.5 * np.einsum('Qkb,Qji,bakj->ai', B_ov, Boo_j, t2a, optimize=True)
        num_t1 += 0.5 * np.einsum('Qki,Qjb,bakj->ai', B_oo, Bov_j, t2a, optimize=True)
        num_t1 -= 1.0 * np.einsum('Qki,Qjb,abkj->ai', B_oo, Bov_j, t2b, optimize=True)
        T3_acc += np.einsum('Qjb,bcij->Qci', Bov_j, t2a, optimize=True)
        T4_acc += np.einsum('Qjc,bcij->Qbi', Bov_j, t2a, optimize=True)
        T5_acc += np.einsum('Qjc,bcij->Qbi', Bov_j, t2b, optimize=True)

    num_t1 -= 0.5 * np.einsum('Qac,Qci->ai', B_vv, T3_acc, optimize=True)
    num_t1 += 0.5 * np.einsum('Qab,Qbi->ai', B_vv, T4_acc, optimize=True)
    num_t1 += 1.0 * np.einsum('Qab,Qbi->ai', B_vv, T5_acc, optimize=True)

    if e_ai is None:
        e_ai = 1.0 / (eps_o[None, :] - eps_v[:, None])
    t1_2 = num_t1 * e_ai

    oo = -0.5 * tmp3_oo - tmp4_oo
    return oo, t1_2.T, vv_acc


class MPnDensityDriverRestricted:
    """Restricted spin-blocked MPn density-matrix correction.

    f_aa/g_aaaa/g_abab/g_bbbb match src.Base.pyscf_interface's restricted
    convention: f_aa the spatial-MO alpha Fock matrix (canonical -> diagonal;
    energy_denominators only reads its diagonal), g_aaaa/g_abab/g_bbbb the
    antisymmetrized-within-spin spatial-MO physicist ERI spin blocks from
    get_antisymmetrized_spin_block_eri (RHF: all three built from the one
    restricted integral tensor -- see that function's docstring).
    """

    def __init__(self, f_aa, g_aaaa, g_abab, g_bbbb, nocc, B_aa=None, B_bb=None, e_abij=None, e_ai=None):
        self.g_aaaa, self.g_abab, self.g_bbbb = g_aaaa, g_abab, g_bbbb
        # DF/RI factors optional -- only needed by the compute_*_df methods below.
        # Pass src.Base.pyscf_interface.DFIntegrals(mol, mf).B_aa/B_bb (or
        # .from_scf(..., exact=True) for the naux=norb^2 plumbing check).
        self.B_aa, self.B_bb = B_aa, B_bb
        self.nocc = nocc
        self.norb = f_aa.shape[0]
        self.o = slice(0, nocc)
        self.v = slice(nocc, self.norb)
        self.no = nocc
        self.nv = self.norb - nocc
        self.d_aa = np.eye(self.norb)
        self._eps_a = np.diagonal(f_aa).copy()
        # e_ai/e_abij only -- NOT energy_denominators(f_aa, nocc, self.nv), which
        # unconditionally also builds e_abcijk, an O(nv^3*no^3) dense array (~3GB
        # at nv=103/no=7, cc-pVQZ CO) that production (laplace_ntau is not None)
        # never touches: t3_2_aaaaaa/aabaab/abbabb (its only consumers, see
        # compute_order2_amplitudes) are skipped entirely in Laplace mode. Measured
        # via staged peak-RSS profiling: this was the single largest contributor to
        # compute_delta_gamma23's memory footprint, dwarfing even the g_aaaa/g_abab/
        # g_bbbb integral tensors. e_abcijk itself is now built lazily, on demand,
        # only inside the laplace_ntau is None branch that actually needs it (see
        # compute_order2_amplitudes below), via the equivalent
        # 1/_denom_restricted(..., rank=3) -- same formula energy_denominators uses
        # internally for its own e_abcijk.
        if e_ai is not None:
            self.e_ai = e_ai
        else:
            self.e_ai = 1.0 / _denom_restricted(self._eps_a, 1, nocc, self.nv)
        # e_abij may be a single array (one denominator for both spin blocks,
        # the ordinary Moller-Plesset case) or an (aaaa, abab) PAIR. The pair
        # is needed for Epstein-Nesbet denominators, where the shift is a
        # diagonal element of a determinant and so differs between the
        # same-spin and opposite-spin amplitudes -- one array applied to both
        # is not any determinant's own denominator. See
        # static_correction._build_dressed_e_abij.
        if e_abij is None:
            e_abij = 1.0 / _denom_restricted(self._eps_a, 2, nocc, self.nv)
        if isinstance(e_abij, tuple):
            self.e_abij_aaaa, self.e_abij_abab = e_abij
        else:
            self.e_abij_aaaa = self.e_abij_abab = e_abij
        self.e_abij = self.e_abij_abab   # back-compat alias (rank-2 consumers)

    def _args(self):
        return dict(g_aaaa=self.g_aaaa, g_abab=self.g_abab, g_bbbb=self.g_bbbb,
                    d_aa=self.d_aa, o=self.o, v=self.v, nv=self.nv, no=self.no)

    def compute_t2_1(self):
        args = self._args()
        t2_1_aaaa = gen.t2_1_aaaa_numerator(**args) * self.e_abij_aaaa
        t2_1_abab = gen.t2_1_abab_numerator(**args) * self.e_abij_abab
        return t2_1_aaaa, t2_1_abab

    def compute_order2_amplitudes(self, t2_1_aaaa, t2_1_abab, laplace_ntau=None, max_rank=4):
        """laplace_ntau: if not None, skip materializing the O(nv^3*no^3)
        t3_2_aaaaaa/aabaab/abbabb tensors entirely -- their only two
        downstream uses (t1_3_aa_numerator's and m3_ov_12_restricted's
        aaaaaa/aabaab/abbabb terms) are instead filled in from
        _laplace_aaaaaa_contribution (aaaaaa) and
        gen_lap3.t1_3_aa_t3crossspin_laplace/m3_ov_a_t3crossspin_laplace
        (aabaab+abbabb) by compute_t1_3/compute_delta_gamma3 below.

        max_rank: highest T^(2) rank actually materialized (returned slots for
        higher ranks are None). Delta_gamma^(2) only needs rank 1 (and cheap
        rank 2); Delta_gamma^(3) needs ranks up to 3 -- rank 4 of Psi^(2)
        cannot couple to <D2| through a one-body operator, so T4^(2)'s
        O(nv^4*no^4) tensors contribute exactly nothing below MP4 and only
        compute_delta_gamma4 asks for max_rank=4 (always with laplace_ntau=None,
        since it needs the full materialized T3^(2)/T4^(2) for its own
        overlap3/overlap4/m4_22 terms -- see compute_delta_gamma4's docstring)."""
        args = self._args()
        t1_2_aa = gen.t1_2_aa_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_ai
        t2_2_aaaa = gen.t2_2_aaaa_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_abij_aaaa
        t2_2_abab = gen.t2_2_abab_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_abij_abab
        t3_2_aaaaaa = t3_2_aabaab = t3_2_abbabb = None
        if max_rank >= 3 and laplace_ntau is None:
            # Built lazily here, NOT cached on self -- see __init__'s comment for
            # why (this is the only consumer, and it's off the production path).
            e_abcijk = 1.0 / _denom_restricted(self._eps_a, 3, self.no, self.nv)
            t3_2_aaaaaa = gen.t3_2_aaaaaa_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * e_abcijk
            t3_2_aabaab = gen.t3_2_aabaab_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * e_abcijk
            t3_2_abbabb = gen.t3_2_abbabb_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * e_abcijk
        t4_2_aaaaaaaa = t4_2_aaabaaab = t4_2_aabbaabb = t4_2_abbbabbb = None
        if max_rank >= 4:
            d4 = _denom_restricted(self._eps_a, 4, self.no, self.nv)
            t4_2_aaaaaaaa = gen.t4_2_aaaaaaaa_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) / d4
            t4_2_aaabaaab = gen.t4_2_aaabaaab_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) / d4
            t4_2_aabbaabb = gen.t4_2_aabbaabb_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) / d4
            t4_2_abbbabbb = gen.t4_2_abbbabbb_numerator(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) / d4
        return (t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
               t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb)

    def compute_E(self, t2_aaaa, t2_abab):
        """E^(k) = 1/2 sum <ij||ab>_aaaa t2_ijab^(aaaa) + sum <ij||ab>_abab
        t2_ijab^(abab) -- restricted spin decomposition of
        mpn_density_driver.py's compute_E (0.25*sum over ALL spin-orbital
        pairs), using aaaa=bbbb (closed shell, so their two 0.25-weighted
        contributions combine to a single 0.5-weighted aaaa term) and abab
        carrying weight 1 (no 0.25, standard for a mixed-spin/Coulomb-only
        sum -- same convention as every other cross-spin piece in this
        pipeline). Confirmed bit-for-bit against mpn_density_driver.py's
        spin-orbital compute_E (folded oracle) on LiH/sto-3g.
        """
        return (0.5 * np.einsum('ijab,abij->', self.g_aaaa[self.o, self.o, self.v, self.v], t2_aaaa, optimize=True)
               + np.einsum('ijab,abij->', self.g_abab[self.o, self.o, self.v, self.v], t2_abab, optimize=True))

    def compute_t1_3(self, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
                     laplace_t1_3_contrib=None, laplace_t1_3_crossspin_contrib=None):
        args = self._args()
        if t3_2_aaaaaa is None:
            # Laplace mode: T3^(2)'s entire contribution comes from
            # laplace_t1_3_contrib/laplace_t1_3_crossspin_contrib below, so
            # this call must go through the '_no_t3' generated variant --
            # NOT gen.t1_3_aa_numerator with a zero placeholder, since the
            # ndim<=4 invariant is enforced per-operand regardless of value
            #.
            num = gen.t1_3_aa_numerator_no_t3(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab)
        else:
            num = gen.t1_3_aa_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                        t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb)
        if laplace_t1_3_contrib is not None:
            # laplace_t1_3_contrib is (no,nv) (density_matrix.py's own
            # convention); this module's t1_3_aa_numerator is (nv,no) (this
            # module's amp_out_indices convention, vir-first) -- transpose.
            num = num + laplace_t1_3_contrib.T
        if laplace_t1_3_crossspin_contrib is not None:
            # gen_lap3.t1_3_aa_t3crossspin_laplace's own out_indices are
            # ['a','i'] -- already (nv,no), no transpose needed.
            num = num + laplace_t1_3_crossspin_contrib
        return num * self.e_ai

    def compute_delta_gamma2(self, t2_1=None, order2=None):
        """t2_1/order2: optional precomputed (t2_1_aaaa, t2_1_abab) /
        compute_order2_amplitudes(..., max_rank>=2) result, to avoid
        recomputing them when the caller also wants Delta_gamma^(3) from the
        same T2^(1)/T1^(2) -- see compute_delta_gamma23."""
        args = self._args()
        t2_1_aaaa, t2_1_abab = t2_1 if t2_1 is not None else self.compute_t2_1()
        if order2 is not None:
            t1_2_aa = order2[0]
        else:
            t1_2_aa, *_ = self.compute_order2_amplitudes(t2_1_aaaa, t2_1_abab, max_rank=2)

        l2_1_aaaa = _to_l_restricted(t2_1_aaaa, 2, 'aaaa')
        l2_1_abab = _to_l_restricted(t2_1_abab, 2, 'abab')
        l1_2_aa = _to_l_restricted(t1_2_aa, 1, 'aa')

        N2 = gen.overlap2_restricted(l_aaaa=l2_1_aaaa, l_abab=l2_1_abab, t_aaaa=t2_1_aaaa, t_abab=t2_1_abab)
        c2 = -N2

        gamma_hf = {'oo': self.d_aa[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            m11 = getattr(gen, f'm2_{block}_11_restricted')(
                **args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab, t_2_1_aaaa=t2_1_aaaa, t_2_1_abab=t2_1_abab)
            m20 = getattr(gen, f'm2_{block}_20_restricted')(**args, l_1_2_aa=l1_2_aa)
            m02 = getattr(gen, f'm2_{block}_02_restricted')(**args, t_1_2_aa=t1_2_aa)
            M2 = m11 + m20 + m02
            blocks[block] = M2 + c2 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

    def _args_df(self):
        return dict(B_aa=self.B_aa, B_bb=self.B_bb,
                    d_aa=self.d_aa, o=self.o, v=self.v, nv=self.nv, no=self.no)

    def compute_t2_1_df(self):
        """DF/RI variant of compute_t2_1: requires B_aa/B_bb (see __init__), never
        forms g_aaaa[v,v,o,o]-scale integral blocks."""
        args = self._args_df()
        t2_1_aaaa = gen_df.t2_1_aaaa_numerator_df(**args) * self.e_abij_aaaa
        t2_1_abab = gen_df.t2_1_abab_numerator_df(**args) * self.e_abij_abab
        return t2_1_aaaa, t2_1_abab

    def compute_t1_2_aa_df(self, t2_1_aaaa, t2_1_abab):
        """DF/RI variant of the T1^(2) piece of compute_order2_amplitudes --
        never forms g_aaaa[o,v,v,v]-scale integral blocks (the term this
        rewrite exists for: MP2 density's own vvvv/ovvv-shaped consumer)."""
        args = self._args_df()
        return gen_df.t1_2_aa_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_ai



    def compute_delta_gamma2_df(self, t2_1=None, t1_2_aa=None):
        """DF/RI variant of compute_delta_gamma2: T2^(1)/T1^(2) come from
        the DF-dressed numerators above; every OTHER n=2 piece (overlap2,
        m2_*_11/20/02) has no bracket integral factor at all (verified by
        inspection of the generated dense code -- see
        the generator
        _amplitude_numerator_funcs_restricted_df docstring), so those are
        called UNCHANGED from `gen`, with g_aaaa/g_abab/g_bbbb passed as
        None (dead params those specific functions never reference)."""
        t2_1_aaaa, t2_1_abab = t2_1 if t2_1 is not None else self.compute_t2_1_df()
        if t1_2_aa is None:
            t1_2_aa = self.compute_t1_2_aa_df(t2_1_aaaa, t2_1_abab)

        l2_1_aaaa = _to_l_restricted(t2_1_aaaa, 2, 'aaaa')
        l2_1_abab = _to_l_restricted(t2_1_abab, 2, 'abab')
        l1_2_aa = _to_l_restricted(t1_2_aa, 1, 'aa')

        N2 = gen.overlap2_restricted(l_aaaa=l2_1_aaaa, l_abab=l2_1_abab, t_aaaa=t2_1_aaaa, t_abab=t2_1_abab)
        c2 = -N2

        gamma_hf = {'oo': self.d_aa[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}
        dead_args = dict(g_aaaa=None, g_abab=None, g_bbbb=None,
                         d_aa=self.d_aa, o=self.o, v=self.v, nv=self.nv, no=self.no)

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            m11 = getattr(gen, f'm2_{block}_11_restricted')(
                **dead_args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab, t_2_1_aaaa=t2_1_aaaa, t_2_1_abab=t2_1_abab)
            m20 = getattr(gen, f'm2_{block}_20_restricted')(**dead_args, l_1_2_aa=l1_2_aa)
            m02 = getattr(gen, f'm2_{block}_02_restricted')(**dead_args, t_1_2_aa=t1_2_aa)
            M2 = m11 + m20 + m02
            blocks[block] = M2 + c2 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

    def compute_order2_amplitudes_df(self, t2_1_aaaa, t2_1_abab, laplace_ntau=None, max_rank=3):
        """DF/RI variant of compute_order2_amplitudes. max_rank>3 (T4^(2)) is not supported --
        MP4 is out of the DF rewrite's scope, no DF-dressed t4_2_* generated."""
        if max_rank > 3:
            raise NotImplementedError(
                "compute_order2_amplitudes_df: max_rank>3 (T4^(2)) has no DF variant (MP4 out of scope)")
        args = self._args_df()
        t1_2_aa = gen_df.t1_2_aa_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_ai
        t2_2_aaaa = gen_df.t2_2_aaaa_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_abij_aaaa
        t2_2_abab = gen_df.t2_2_abab_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * self.e_abij_abab
        t3_2_aaaaaa = t3_2_aabaab = t3_2_abbabb = None
        if max_rank >= 3 and laplace_ntau is None:
            e_abcijk = 1.0 / _denom_restricted(self._eps_a, 3, self.no, self.nv)
            t3_2_aaaaaa = gen_df.t3_2_aaaaaa_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * e_abcijk
            t3_2_aabaab = gen_df.t3_2_aabaab_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * e_abcijk
            t3_2_abbabb = gen_df.t3_2_abbabb_numerator_df(**args, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab) * e_abcijk
        return t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb

    def compute_t1_3_df(self, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
                        laplace_t1_3_contrib=None, laplace_t1_3_crossspin_contrib=None):
        """DF/RI variant of compute_t1_3 -- never forms
        g_aaaa[o,v,v,v]/g_abab[v,o,v,v]-scale integral blocks."""
        args = self._args_df()
        if t3_2_aaaaaa is None:
            num = gen_df.t1_3_aa_numerator_no_t3_df(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab)
        else:
            num = gen_df.t1_3_aa_numerator_df(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                              t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb)
        if laplace_t1_3_contrib is not None:
            num = num + laplace_t1_3_contrib.T
        if laplace_t1_3_crossspin_contrib is not None:
            num = num + laplace_t1_3_crossspin_contrib
        return num * self.e_ai

    def compute_delta_gamma3_df(self, laplace_ntau=None, t2_1=None, order2=None):
        """DF/RI variant of compute_delta_gamma3 -- T2^(1)/T1^(2)/T2^(2)/
        T3^(2)/T1^(3) come from the DF-dressed numerators above (including
        the DF+Laplace-composed gen_lap3_df pieces and
        _laplace_aaaaaa_contribution_df); every m3_* cross-density piece is
        bracket-free (same reasoning as compute_delta_gamma2_df's m2_*
        pieces) and is called UNCHANGED from `gen` with
        g_aaaa=g_abab=g_bbbb=None (dead params)."""
        args_df = self._args_df()
        t2_1_aaaa, t2_1_abab = t2_1 if t2_1 is not None else self.compute_t2_1_df()
        (t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb) = order2 if order2 is not None else \
            self.compute_order2_amplitudes_df(t2_1_aaaa, t2_1_abab, laplace_ntau=laplace_ntau, max_rank=3)

        l2_1_aaaa = _to_l_restricted(t2_1_aaaa, 2, 'aaaa')
        l2_1_abab = _to_l_restricted(t2_1_abab, 2, 'abab')

        laplace_t1_3_contrib = laplace_ov_contrib = None
        laplace_t1_3_cs_contrib = laplace_ov_cs_contrib = None
        if laplace_ntau is not None:
            ei_a, ea_a = self._eps_a[self.o], self._eps_a[self.v]
            laplace_t1_3_contrib, laplace_ov_contrib = _laplace_aaaaaa_contribution_df(
                ei_a, ea_a, self.B_aa, self.o, self.v, t2_1_aaaa, laplace_ntau)
            laplace_t1_3_cs_contrib = gen_lap3_df.t1_3_aa_t3crossspin_laplace_df(
                **args_df, eps_a=self._eps_a, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab, ntau=laplace_ntau)
            laplace_ov_cs_contrib = gen_lap3_df.m3_ov_a_t3crossspin_laplace_df(
                **args_df, eps_a=self._eps_a, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab,
                l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab, ntau=laplace_ntau)

        t1_3_aa = self.compute_t1_3_df(t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
                                       laplace_t1_3_contrib=laplace_t1_3_contrib,
                                       laplace_t1_3_crossspin_contrib=laplace_t1_3_cs_contrib)

        z_aaaa = np.zeros((self.nv, self.nv, self.no, self.no))
        z_aaaaaa = np.zeros((self.nv, self.nv, self.nv, self.no, self.no, self.no))
        t3_aaaaaa_arg = t3_2_aaaaaa if t3_2_aaaaaa is not None else z_aaaaaa

        l1_2_aa = _to_l_restricted(t1_2_aa, 1, 'aa')
        l2_2_aaaa = _to_l_restricted(t2_2_aaaa, 2, 'aaaa')
        l2_2_abab = _to_l_restricted(t2_2_abab, 2, 'abab')
        l3_2_aaaaaa = z_aaaaaa
        l3_2_aabaab = z_aaaaaa
        l3_2_abbabb = z_aaaaaa
        l1_3_aa = _to_l_restricted(t1_3_aa, 1, 'aa')

        N3 = 2.0 * gen.overlap2_restricted(l_aaaa=l2_1_aaaa, l_abab=l2_1_abab, t_aaaa=t2_2_aaaa, t_abab=t2_2_abab)
        c3 = -N3

        gamma_hf = {'oo': self.d_aa[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        t3_aabaab_arg = t3_2_aabaab if t3_2_aabaab is not None else z_aaaaaa
        t3_abbabb_arg = t3_2_abbabb if t3_2_abbabb is not None else z_aaaaaa
        dead_args = dict(g_aaaa=None, g_abab=None, g_bbbb=None,
                         d_aa=self.d_aa, o=self.o, v=self.v, nv=self.nv, no=self.no)

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            if block == 'ov' and t3_2_aabaab is None:
                m3_12 = getattr(gen, 'm3_ov_12_restricted_no_t3')(
                    **dead_args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab,
                    t_1_2_aa=t1_2_aa, t_2_2_aaaa=t2_2_aaaa, t_2_2_abab=t2_2_abab)
            else:
                m3_12 = getattr(gen, f'm3_{block}_12_restricted')(
                    **dead_args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab,
                    t_1_2_aa=t1_2_aa, t_2_2_aaaa=t2_2_aaaa, t_2_2_abab=t2_2_abab,
                    t_3_2_aaaaaa=t3_aaaaaa_arg, t_3_2_aabaab=t3_aabaab_arg, t_3_2_abbabb=t3_abbabb_arg)
            if block == 'ov':
                if laplace_ov_contrib is not None:
                    m3_12 = m3_12 + laplace_ov_contrib
                if laplace_ov_cs_contrib is not None:
                    m3_12 = m3_12 + laplace_ov_cs_contrib
            m3_21 = getattr(gen, f'm3_{block}_21_restricted')(
                **dead_args, l_1_2_aa=l1_2_aa, l_2_2_aaaa=l2_2_aaaa, l_2_2_abab=l2_2_abab,
                l_3_2_aaaaaa=l3_2_aaaaaa, l_3_2_aabaab=l3_2_aabaab, l_3_2_abbabb=l3_2_abbabb,
                t_2_1_aaaa=t2_1_aaaa, t_2_1_abab=t2_1_abab)
            m3_30 = getattr(gen, f'm3_{block}_30_restricted')(
                **dead_args, l_1_3_aa=l1_3_aa, l_2_3_aaaa=z_aaaa, l_2_3_abab=z_aaaa,
                l_3_3_aaaaaa=z_aaaaaa, l_3_3_aabaab=z_aaaaaa, l_3_3_abbabb=z_aaaaaa)
            m3_03 = getattr(gen, f'm3_{block}_03_restricted')(
                **dead_args, t_1_3_aa=t1_3_aa, t_2_3_aaaa=z_aaaa, t_2_3_abab=z_aaaa,
                t_3_3_aaaaaa=z_aaaaaa, t_3_3_aabaab=z_aaaaaa, t_3_3_abbabb=z_aaaaaa)
            M3 = m3_12 + m3_21 + m3_30 + m3_03
            blocks[block] = M3 + c3 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

    def compute_delta_gamma3(self, laplace_ntau=None, t2_1=None, order2=None):
        """laplace_ntau: see compute_order2_amplitudes -- if not None, the
        O(nv^3*no^3) t3_2_aaaaaa/aabaab/abbabb tensors are never
        materialized; their downstream uses (t1_3_aa's and
        m3_ov_12_restricted's aaaaaa/aabaab/abbabb terms) are filled in via
        _laplace_aaaaaa_contribution (aaaaaa) and
        gen_lap3.t1_3_aa_t3crossspin_laplace/m3_ov_a_t3crossspin_laplace
        (aabaab+abbabb) instead.

        T4^(2) is never built here (max_rank=3): a one-body operator cannot
        couple <D2| and |D4>, so the rank-4 sector of Psi^(2) contributes
        exactly nothing to Delta_gamma^(3) (the regenerated m3 bodies carry
        no t4/l4 terms.

        t2_1/order2: optional precomputed compute_t2_1()/
        compute_order2_amplitudes(..., max_rank=3) results, to avoid
        recomputing T2^(1)/T1^(2)/T2^(2) when the caller also wants
        Delta_gamma^(2) from the same amplitudes -- see compute_delta_gamma23.
        """
        args = self._args()
        t2_1_aaaa, t2_1_abab = t2_1 if t2_1 is not None else self.compute_t2_1()
        (t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
         _, _, _, _) = order2 if order2 is not None else \
            self.compute_order2_amplitudes(t2_1_aaaa, t2_1_abab, laplace_ntau=laplace_ntau, max_rank=3)

        l2_1_aaaa = _to_l_restricted(t2_1_aaaa, 2, 'aaaa')
        l2_1_abab = _to_l_restricted(t2_1_abab, 2, 'abab')

        laplace_t1_3_contrib = laplace_ov_contrib = None
        laplace_t1_3_cs_contrib = laplace_ov_cs_contrib = None
        if laplace_ntau is not None:
            ei_a, ea_a = self._eps_a[self.o], self._eps_a[self.v]
            laplace_t1_3_contrib, laplace_ov_contrib = _laplace_aaaaaa_contribution(
                ei_a, ea_a, self.g_aaaa, self.o, self.v, t2_1_aaaa, laplace_ntau)
            laplace_t1_3_cs_contrib = gen_lap3.t1_3_aa_t3crossspin_laplace(
                **args, eps_a=self._eps_a, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab, ntau=laplace_ntau)
            laplace_ov_cs_contrib = gen_lap3.m3_ov_a_t3crossspin_laplace(
                **args, eps_a=self._eps_a, t2_1_aaaa=t2_1_aaaa, t2_1_abab=t2_1_abab,
                l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab, ntau=laplace_ntau)

        t1_3_aa = self.compute_t1_3(t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
                                    laplace_t1_3_contrib=laplace_t1_3_contrib,
                                    laplace_t1_3_crossspin_contrib=laplace_t1_3_cs_contrib)

        # rank 2/3 of Psi^(3) never actually contribute to m3_*_30/m3_*_03
        # (0 terms in the generated bodies,.py's spin-
        # orbital compute_delta_gamma3, just per spin block here.
        z_aaaa = np.zeros((self.nv, self.nv, self.no, self.no))
        z_aaaaaa = np.zeros((self.nv, self.nv, self.nv, self.no, self.no, self.no))
        # t3_2_{aaaaaa,aabaab,abbabb} are all None in laplace mode: l_3_2_*
        # is passed to m3_{oo,vv,ov}_21_restricted only as a
        # declared-but-unused parameter (verified: none of those three
        # generated bodies ever reference l3_aaaaaa/l3_aabaab/l3_abbabb in an
        # einsum -- only l1/l2), so a zero placeholder is always safe here
        # regardless of laplace_ntau, and skips the O(nv^3*no^3) transpose
        # _to_l_restricted would otherwise perform on a real tensor for no
        # reason.
        t3_aaaaaa_arg = t3_2_aaaaaa if t3_2_aaaaaa is not None else z_aaaaaa

        l1_2_aa = _to_l_restricted(t1_2_aa, 1, 'aa')
        l2_2_aaaa = _to_l_restricted(t2_2_aaaa, 2, 'aaaa')
        l2_2_abab = _to_l_restricted(t2_2_abab, 2, 'abab')
        l3_2_aaaaaa = z_aaaaaa
        l3_2_aabaab = z_aaaaaa
        l3_2_abbabb = z_aaaaaa
        l1_3_aa = _to_l_restricted(t1_3_aa, 1, 'aa')

        N3 = 2.0 * gen.overlap2_restricted(l_aaaa=l2_1_aaaa, l_abab=l2_1_abab, t_aaaa=t2_2_aaaa, t_abab=t2_2_abab)
        c3 = -N3

        gamma_hf = {'oo': self.d_aa[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        t3_aabaab_arg = t3_2_aabaab if t3_2_aabaab is not None else z_aaaaaa
        t3_abbabb_arg = t3_2_abbabb if t3_2_abbabb is not None else z_aaaaaa

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            # Delta_gamma^(3) = M^(3) + c^(3)*M^(0) only -- see
            # mpn_density_driver.py's compute_delta_gamma3 for why c^(2)*M^(2)
            # must NOT appear (c^(2) only ever multiplies M^(1)=0).
            if block == 'ov' and t3_2_aabaab is None:
                # Laplace mode: m3_ov_12_restricted's ENTIRE contribution is
                # t3-dependent (0 terms from ranks 1,2 alone -- see
                # the generator printed generation
                # log), so the '_no_t3' variant just returns zeros here;
                # laplace_ov_contrib/laplace_ov_cs_contrib below supply the
                # real (Laplace-fused) value. Using the ordinary
                # m3_ov_12_restricted with a zero t_3_2_* placeholder would
                # pass a rank-6 array to einsum even though its VALUE is
                # zero -- forbidden by the ndim<=4 invariant regardless of
                # value (see compute_t1_3's matching comment).
                m3_12 = getattr(gen, 'm3_ov_12_restricted_no_t3')(
                    **args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab,
                    t_1_2_aa=t1_2_aa, t_2_2_aaaa=t2_2_aaaa, t_2_2_abab=t2_2_abab)
            else:
                m3_12 = getattr(gen, f'm3_{block}_12_restricted')(
                    **args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab,
                    t_1_2_aa=t1_2_aa, t_2_2_aaaa=t2_2_aaaa, t_2_2_abab=t2_2_abab,
                    t_3_2_aaaaaa=t3_aaaaaa_arg, t_3_2_aabaab=t3_aabaab_arg, t_3_2_abbabb=t3_abbabb_arg)
            if block == 'ov':
                if laplace_ov_contrib is not None:
                    m3_12 = m3_12 + laplace_ov_contrib
                if laplace_ov_cs_contrib is not None:
                    m3_12 = m3_12 + laplace_ov_cs_contrib
            m3_21 = getattr(gen, f'm3_{block}_21_restricted')(
                **args, l_1_2_aa=l1_2_aa, l_2_2_aaaa=l2_2_aaaa, l_2_2_abab=l2_2_abab,
                l_3_2_aaaaaa=l3_2_aaaaaa, l_3_2_aabaab=l3_2_aabaab, l_3_2_abbabb=l3_2_abbabb,
                t_2_1_aaaa=t2_1_aaaa, t_2_1_abab=t2_1_abab)
            m3_30 = getattr(gen, f'm3_{block}_30_restricted')(
                **args, l_1_3_aa=l1_3_aa, l_2_3_aaaa=z_aaaa, l_2_3_abab=z_aaaa,
                l_3_3_aaaaaa=z_aaaaaa, l_3_3_aabaab=z_aaaaaa, l_3_3_abbabb=z_aaaaaa)
            m3_03 = getattr(gen, f'm3_{block}_03_restricted')(
                **args, t_1_3_aa=t1_3_aa, t_2_3_aaaa=z_aaaa, t_2_3_abab=z_aaaa,
                t_3_3_aaaaaa=z_aaaaaa, t_3_3_aabaab=z_aaaaaa, t_3_3_abbabb=z_aaaaaa)
            M3 = m3_12 + m3_21 + m3_30 + m3_03

            blocks[block] = M3 + c3 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

    def compute_delta_gamma23(self, laplace_ntau=6):
        """Delta_gamma^(2) and Delta_gamma^(3) together, computing T2^(1)/
        T1^(2)/T2^(2) exactly once instead of once per call (production MP3
        density always needs both -- see density_matrix.py's
        compute_mp2/mp3_density_matrix_ao and static_correction.py's
        build_mp2/mp3_static_correction call sites). Returns
        (gamma2_blocks, gamma3_blocks), each an (oo, ov, vv) tuple."""
        t2_1 = self.compute_t2_1()
        order2 = self.compute_order2_amplitudes(*t2_1, laplace_ntau=laplace_ntau, max_rank=3)
        gamma2 = self.compute_delta_gamma2(t2_1=t2_1, order2=order2)
        gamma3 = self.compute_delta_gamma3(laplace_ntau=laplace_ntau, t2_1=t2_1, order2=order2)
        return gamma2, gamma3

    def compute_delta_gamma23_df(self, laplace_ntau=6):
        """DF/RI variant of compute_delta_gamma23 -- calls the compute_*_df
        methods throughout. never reading self.g_aaaa/g_abab/g_bbbb (safe to construct this driver
        with those as None when only this method is used -- see
        __init__'s B_aa docstring note). Production wiring entry point for
        static_correction.py's _mp2_dgamma_spatial/_mp3_dgamma_spatial."""
        t2_1 = self.compute_t2_1_df()
        order2 = self.compute_order2_amplitudes_df(*t2_1, laplace_ntau=laplace_ntau, max_rank=3)
        gamma2 = self.compute_delta_gamma2_df(t2_1=t2_1, t1_2_aa=order2[0])
        gamma3 = self.compute_delta_gamma3_df(laplace_ntau=laplace_ntau, t2_1=t2_1, order2=order2)
        return gamma2, gamma3

    def compute_order3_amplitudes(self, t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
                                  t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb,
                                  t2_1_aaaa, t2_1_abab, E2, laplace_ntau=None):
        """T2^(3)_aaaa/abab need the -E^(2)*T2^(1) correction (nonzero
        starting here, mirrors mpn_density_driver.py's compute_order3_amplitudes);
        T1^(3)/T3^(3) don't (their own E^(2)*T_rank^(1) vanishes). E2 must be
        E^(2) = compute_E(t2_1_aaaa, t2_1_abab) (order 1, NOT order 2) --
        passed in explicitly since compute_delta_gamma4 needs the same value
        again for c^(4).

        laplace_ntau: if not None, t4_2_aaaaaaaa is not read for t2_3_aaaa's
        or t3_3_aaaaaa's own numerator (pass None for that argument) --
        their t4_2_aaaaaaaa-dependent piece is instead filled in via
        generate_mp4_laplace_restricted.py's Laplace-fused
        t2_3_aaaa_t4_laplace/t3_3_aaaaaa_t4_laplace (see that module's
        docstring for scope: only these two of the five T2^(3)/T3^(3)
        equations reference t4_aaaaaaaa at all). t4_2_aaaaaaaa itself is
        NOT eliminated everywhere by this -- m4_{oo,vv,ov}_22_restricted and
        overlap4_restricted still need it materialized (see that module's
        docstring for why: T4^(2) appears in BOTH bra and ket roles there
        simultaneously, a harder sub-problem not attempted here).
        """
        args = self._args()
        d1 = _denom_restricted(self._eps_a, 1, self.no, self.nv)
        d2 = _denom_restricted(self._eps_a, 2, self.no, self.nv)
        d3 = _denom_restricted(self._eps_a, 3, self.no, self.nv)
        t1_3_aa = gen.t1_3_aa_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                        t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab,
                                        t3_2_abbabb=t3_2_abbabb) / d1

        # t4_2_aaaaaaaa is passed in as a real (materialized) array either
        # way -- compute_delta_gamma4 still needs it for m4_*_22/overlap4
        # regardless of laplace_ntau (see this method's docstring) -- but in
        # laplace mode, t2_3_aaaa_numerator/t3_3_aaaaaa_numerator's OWN
        # reference to it is zeroed out and replaced by the Laplace-fused
        # piece instead, avoiding re-deriving that specific contraction from
        # the full materialized tensor.
        t4_zero = np.zeros_like(t4_2_aaaaaaaa)
        t4_for_t2_3_t3_3 = t4_2_aaaaaaaa if laplace_ntau is None else t4_zero

        t2_3_aaaa_num = gen.t2_3_aaaa_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                                t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb,
                                                t4_2_aaaaaaaa=t4_for_t2_3_t3_3, t4_2_aaabaaab=t4_2_aaabaaab,
                                                t4_2_aabbaabb=t4_2_aabbaabb, t4_2_abbbabbb=t4_2_abbbabbb)
        t3_3_aaaaaa_num = gen.t3_3_aaaaaa_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                                    t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb,
                                                    t4_2_aaaaaaaa=t4_for_t2_3_t3_3, t4_2_aaabaaab=t4_2_aaabaaab,
                                                    t4_2_aabbaabb=t4_2_aabbaabb, t4_2_abbbabbb=t4_2_abbbabbb)
        if laplace_ntau is not None:
            t2_3_aaaa_num = t2_3_aaaa_num + gen_lap4.t2_3_aaaa_t4_laplace(
                **args, eps_a=self._eps_a, t2_1_aaaa=t2_1_aaaa, ntau=laplace_ntau)
            t3_3_aaaaaa_num = t3_3_aaaaaa_num + gen_lap4.t3_3_aaaaaa_t4_laplace(
                **args, eps_a=self._eps_a, t2_1_aaaa=t2_1_aaaa, ntau=laplace_ntau)

        t2_3_aaaa = (t2_3_aaaa_num - E2 * t2_1_aaaa) / d2
        t2_3_abab = (gen.t2_3_abab_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                             t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb,
                                             t4_2_aaaaaaaa=t4_zero, t4_2_aaabaaab=t4_2_aaabaaab,
                                             t4_2_aabbaabb=t4_2_aabbaabb, t4_2_abbbabbb=t4_2_abbbabbb)
                    - E2 * t2_1_abab) / d2
        t3_3_aaaaaa = t3_3_aaaaaa_num / d3
        t3_3_aabaab = gen.t3_3_aabaab_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                                t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb,
                                                t4_2_aaaaaaaa=t4_zero, t4_2_aaabaaab=t4_2_aaabaaab,
                                                t4_2_aabbaabb=t4_2_aabbaabb, t4_2_abbbabbb=t4_2_abbbabbb) / d3
        t3_3_abbabb = gen.t3_3_abbabb_numerator(**args, t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                                                t3_2_aaaaaa=t3_2_aaaaaa, t3_2_aabaab=t3_2_aabaab, t3_2_abbabb=t3_2_abbabb,
                                                t4_2_aaaaaaaa=t4_zero, t4_2_aaabaaab=t4_2_aaabaaab,
                                                t4_2_aabbaabb=t4_2_aabbaabb, t4_2_abbbabbb=t4_2_abbbabbb) / d3
        return t1_3_aa, t2_3_aaaa, t2_3_abab, t3_3_aaaaaa, t3_3_aabaab, t3_3_abbabb

    def compute_t2_3_laplace(self, t1_2_aa, t2_2_aaaa, t2_2_abab,
                             t2_1_aaaa, t2_1_abab, E2, ntau):
        """T2^(3) (order-3 doubles) with NO rank>=3 tensor materialized -- the
        PRODUCTION twin of compute_order3_amplitudes' t2_3.

        t2_3's numerator consumes t3_2 (O(no^3 nv^3)) and t4_2 (O(no^4 nv^4)).
        Here every one of those rank>=3-leaf CONTRIBUTIONS is Laplace-fused
        (mp4_t2_3_laplace_restricted, one function per (block, leaf-tag)), and
        the rank<=2 DIRECT part is the compiled t2_3_{block}_numerator with
        t3_2=t4_2=0 -- which correctly carries the numerator's MIXED-ORDER
        direct leaves (a direct t2_2 order-2 AND a V.V.t2_1 direct t2_1 order-1)
        that a uniform-order fuser would mis-tag. Then
        t2_3_{block} = (direct + sum_tag fused - E2 * t2_1_{block}) / D_2,
        the same closing algebra as compute_order3_amplitudes. == the
        materialized t2_3 to the minimax accuracy (converges with ntau; gate
        tests/test_mp4_t2_3_amplitude.py, 3-10e-9 at ntau=6 on NH3). Returns
        (t2_3_aaaa, t2_3_abab)."""
        args = self._args()
        no, nv = self.no, self.nv
        d2 = _denom_restricted(self._eps_a, 2, no, nv)
        z3 = np.zeros((nv, nv, nv, no, no, no))
        z4 = np.zeros((nv, nv, nv, nv, no, no, no, no))
        zeroed = dict(t1_2_aa=t1_2_aa, t2_2_aaaa=t2_2_aaaa, t2_2_abab=t2_2_abab,
                      t3_2_aaaaaa=z3, t3_2_aabaab=z3, t3_2_abbabb=z3,
                      t4_2_aaaaaaaa=z4, t4_2_aaabaaab=z4, t4_2_aabbaabb=z4,
                      t4_2_abbbabbb=z4)
        pool = {**args, 'eps_a': self._eps_a, 'ntau': ntau,
                't2_1_aaaa': t2_1_aaaa, 't2_1_abab': t2_1_abab}
        out = {}
        for block, t2_1 in (('aaaa', t2_1_aaaa), ('abab', t2_1_abab)):
            num = getattr(gen, f't2_3_{block}_numerator')(**args, **zeroed)
            for fname in sorted(dir(gen_lap_t23)):
                if fname.startswith(f't2_3_{block}_') and fname.endswith('_laplace'):
                    fn = getattr(gen_lap_t23, fname)
                    num = num + fn(**{p: pool[p]
                                      for p in inspect.signature(fn).parameters})
            out[block] = (num - E2 * t2_1) / d2
        return out['aaaa'], out['abab']

    def compute_t1_4_laplace(self, t1_3_aa, t2_3_aaaa, t2_3_abab, t2_2_aaaa,
                             t2_2_abab, t1_2_aa, t2_1_aaaa, t2_1_abab, E2, ntau):
        """T1^(4) (order-4 singles) with NO rank>=3 tensor materialized -- the
        PRODUCTION twin of compute_t1_4, and the NESTED-amplitude piece of the
        order-4 chain.

        compute_t1_4's numerator consumes t3_3 (order 3, O(no^3 nv^3)), whose OWN
        numerator consumes t3_2/t4_2 (order 2) -- a fusion nested two deep. The
        whole thing is Laplace-fused by mp4_t1_4_laplace_restricted.t1_4_aa_
        laplace (fuse_nested_numerator, DEPTH-QUALIFIED taus so the outer t3_3
        and inner t3_2 -- both rank 3 -- integrate independently), consuming only
        the rank<=2 leaves t1_3/t2_3 (order 3, pass this driver's Laplace
        T1^(3)/T2^(3)) + t2_2/t1_2 (order 2) + t2_1 (order 1). Then
        t1_4 = (fused numerator - E2 * t1_2) / D_1, the same closing as
        compute_t1_4. == the materialized t1_4 to the minimax accuracy
        (5.6e-10 @ntau=6 -> 4.5e-13 @ntau=10 on NH3; gate
        tests/test_mp4_t1_4_laplace.py). Returns t1_4_aa."""
        d1 = _denom_restricted(self._eps_a, 1, self.no, self.nv)
        pool = {**self._args(), 'eps_a': self._eps_a, 'ntau': ntau,
                't1_3_aa': t1_3_aa, 't2_3_aaaa': t2_3_aaaa, 't2_3_abab': t2_3_abab,
                't2_2_aaaa': t2_2_aaaa, 't2_2_abab': t2_2_abab, 't1_2_aa': t1_2_aa,
                't2_1_aaaa': t2_1_aaaa, 't2_1_abab': t2_1_abab}
        fn = gen_lap_t14.t1_4_aa_laplace
        num = fn(**{p: pool[p] for p in inspect.signature(fn).parameters})
        return (num - E2 * t1_2_aa) / d1

    def compute_t1_4(self, t1_3_aa, t2_3_aaaa, t2_3_abab, t3_3_aaaaaa, t3_3_aabaab, t3_3_abbabb, t1_2_aa, E2):
        """T1^(4)_aa needs the -E^(2)*T1^(2)_aa correction."""
        args = self._args()
        d1 = _denom_restricted(self._eps_a, 1, self.no, self.nv)
        num = gen.t1_4_aa_numerator(**args, t1_3_aa=t1_3_aa, t2_3_aaaa=t2_3_aaaa, t2_3_abab=t2_3_abab,
                                    t3_3_aaaaaa=t3_3_aaaaaa, t3_3_aabaab=t3_3_aabaab,
                                    t3_3_abbabb=t3_3_abbabb) - E2 * t1_2_aa
        return num / d1

    def compute_delta_gamma4(self, laplace_ntau=None):
        """Delta_gamma^(4) = M^(4) + c^(2)*M^(2) + c^(4)*M^(0), mirroring
        mpn_density_driver.py's compute_delta_gamma4 (see that docstring for
        the general recursion).

        laplace_ntau: if not None, t2_3_aaaa/t3_3_aaaaaa's own
        t4_2_aaaaaaaa-dependent piece is computed via the Laplace-fused
        generate_mp4_laplace_restricted.py path instead of directly from the
        materialized t4_2_aaaaaaaa tensor -- see
        compute_order3_amplitudes's docstring for exactly which two of the
        many T4^(2)-touching equations this affects (only these two;
        m4_*_22/overlap4 are unaffected and still need t4_2_aaaaaaaa
        materialized, which compute_order2_amplitudes always builds
        regardless of this flag).
        """
        args = self._args()
        t2_1_aaaa, t2_1_abab = self.compute_t2_1()
        (t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
         t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb) = \
            self.compute_order2_amplitudes(t2_1_aaaa, t2_1_abab)
        # E^(2) = compute_E(t2^(1)) -- order 1, NOT order 2 (an earlier
        # version of this line wrongly passed t2_2_aaaa/t2_2_abab; caught by
        # comparing against mpn_density_driver.py's spin-orbital compute_E,
        # which correctly uses t2_1 -- see compute_order3_amplitudes'/
        # compute_t1_4's own docstrings for where E2 is consumed).
        E2 = self.compute_E(t2_1_aaaa, t2_1_abab)
        t1_3_aa, t2_3_aaaa, t2_3_abab, t3_3_aaaaaa, t3_3_aabaab, t3_3_abbabb = \
            self.compute_order3_amplitudes(t1_2_aa, t2_2_aaaa, t2_2_abab, t3_2_aaaaaa, t3_2_aabaab, t3_2_abbabb,
                                           t4_2_aaaaaaaa, t4_2_aaabaaab, t4_2_aabbaabb, t4_2_abbbabbb,
                                           t2_1_aaaa, t2_1_abab, E2=E2, laplace_ntau=laplace_ntau)
        t1_4_aa = self.compute_t1_4(t1_3_aa, t2_3_aaaa, t2_3_abab, t3_3_aaaaaa, t3_3_aabaab, t3_3_abbabb, t1_2_aa, E2)

        l2_1_aaaa = _to_l_restricted(t2_1_aaaa, 2, 'aaaa')
        l2_1_abab = _to_l_restricted(t2_1_abab, 2, 'abab')
        l1_2_aa = _to_l_restricted(t1_2_aa, 1, 'aa')
        l2_2_aaaa = _to_l_restricted(t2_2_aaaa, 2, 'aaaa')
        l2_2_abab = _to_l_restricted(t2_2_abab, 2, 'abab')
        l3_2_aaaaaa = _to_l_restricted(t3_2_aaaaaa, 3, 'aaaaaa')
        l3_2_aabaab = _to_l_restricted(t3_2_aabaab, 3, 'aabaab')
        l3_2_abbabb = _to_l_restricted(t3_2_abbabb, 3, 'abbabb')
        l4_2_aaaaaaaa = _to_l_restricted(t4_2_aaaaaaaa, 4, 'aaaaaaaa')
        l4_2_aaabaaab = _to_l_restricted(t4_2_aaabaaab, 4, 'aaabaaab')
        l4_2_aabbaabb = _to_l_restricted(t4_2_aabbaabb, 4, 'aabbaabb')
        l4_2_abbbabbb = _to_l_restricted(t4_2_abbbabbb, 4, 'abbbabbb')
        l1_3_aa = _to_l_restricted(t1_3_aa, 1, 'aa')
        l2_3_aaaa = _to_l_restricted(t2_3_aaaa, 2, 'aaaa')
        l2_3_abab = _to_l_restricted(t2_3_abab, 2, 'abab')
        l3_3_aaaaaa = _to_l_restricted(t3_3_aaaaaa, 3, 'aaaaaa')
        l3_3_aabaab = _to_l_restricted(t3_3_aabaab, 3, 'aabaab')
        l3_3_abbabb = _to_l_restricted(t3_3_abbabb, 3, 'abbabb')
        l1_4_aa = _to_l_restricted(t1_4_aa, 1, 'aa')

        # N^(4) = 2*<Psi1|Psi3> (rank-2-matched) + <Psi2|Psi2> (rank-matched
        # sum over Psi^(2)'s own ranks 1-4).
        N4 = (2.0 * gen.overlap2_restricted(l_aaaa=l2_1_aaaa, l_abab=l2_1_abab, t_aaaa=t2_3_aaaa, t_abab=t2_3_abab)
             + gen.overlap1_restricted(l_aa=l1_2_aa, t_aa=t1_2_aa)
             + gen.overlap2_restricted(l_aaaa=l2_2_aaaa, l_abab=l2_2_abab, t_aaaa=t2_2_aaaa, t_abab=t2_2_abab)
             + gen.overlap3_restricted(l_aaaaaa=l3_2_aaaaaa, l_aabaab=l3_2_aabaab, l_abbabb=l3_2_abbabb,
                                       t_aaaaaa=t3_2_aaaaaa, t_aabaab=t3_2_aabaab, t_abbabb=t3_2_abbabb)
             + gen.overlap4_restricted(l_aaaaaaaa=l4_2_aaaaaaaa, l_aaabaaab=l4_2_aaabaaab,
                                       l_aabbaabb=l4_2_aabbaabb, l_abbbabbb=l4_2_abbbabbb,
                                       t_aaaaaaaa=t4_2_aaaaaaaa, t_aaabaaab=t4_2_aaabaaab,
                                       t_aabbaabb=t4_2_aabbaabb, t_abbbabbb=t4_2_abbbabbb))
        N2 = gen.overlap2_restricted(l_aaaa=l2_1_aaaa, l_abab=l2_1_abab, t_aaaa=t2_1_aaaa, t_abab=t2_1_abab)
        c2 = -N2
        c4 = N2 ** 2 - N4

        gamma_hf = {'oo': self.d_aa[self.o, self.o],
                    'vv': np.zeros((self.nv, self.nv)),
                    'ov': np.zeros((self.no, self.nv))}

        blocks = {}
        for block in ('oo', 'vv', 'ov'):
            m2_11 = getattr(gen, f'm2_{block}_11_restricted')(
                **args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab, t_2_1_aaaa=t2_1_aaaa, t_2_1_abab=t2_1_abab)
            m2_20 = getattr(gen, f'm2_{block}_20_restricted')(**args, l_1_2_aa=l1_2_aa)
            m2_02 = getattr(gen, f'm2_{block}_02_restricted')(**args, t_1_2_aa=t1_2_aa)
            M2 = m2_11 + m2_20 + m2_02

            m4_13 = getattr(gen, f'm4_{block}_13_restricted')(
                **args, l_2_1_aaaa=l2_1_aaaa, l_2_1_abab=l2_1_abab,
                t_1_3_aa=t1_3_aa, t_2_3_aaaa=t2_3_aaaa, t_2_3_abab=t2_3_abab,
                t_3_3_aaaaaa=t3_3_aaaaaa, t_3_3_aabaab=t3_3_aabaab, t_3_3_abbabb=t3_3_abbabb)
            m4_31 = getattr(gen, f'm4_{block}_31_restricted')(
                **args, l_1_3_aa=l1_3_aa, l_2_3_aaaa=l2_3_aaaa, l_2_3_abab=l2_3_abab,
                l_3_3_aaaaaa=l3_3_aaaaaa, l_3_3_aabaab=l3_3_aabaab, l_3_3_abbabb=l3_3_abbabb,
                t_2_1_aaaa=t2_1_aaaa, t_2_1_abab=t2_1_abab)
            m4_22 = getattr(gen, f'm4_{block}_22_restricted')(
                **args, l_1_2_aa=l1_2_aa, l_2_2_aaaa=l2_2_aaaa, l_2_2_abab=l2_2_abab,
                l_3_2_aaaaaa=l3_2_aaaaaa, l_3_2_aabaab=l3_2_aabaab, l_3_2_abbabb=l3_2_abbabb,
                l_4_2_aaaaaaaa=l4_2_aaaaaaaa, l_4_2_aaabaaab=l4_2_aaabaaab,
                l_4_2_aabbaabb=l4_2_aabbaabb, l_4_2_abbbabbb=l4_2_abbbabbb,
                t_1_2_aa=t1_2_aa, t_2_2_aaaa=t2_2_aaaa, t_2_2_abab=t2_2_abab,
                t_3_2_aaaaaa=t3_2_aaaaaa, t_3_2_aabaab=t3_2_aabaab, t_3_2_abbabb=t3_2_abbabb,
                t_4_2_aaaaaaaa=t4_2_aaaaaaaa, t_4_2_aaabaaab=t4_2_aaabaaab,
                t_4_2_aabbaabb=t4_2_aabbaabb, t_4_2_abbbabbb=t4_2_abbbabbb)
            m4_40 = getattr(gen, f'm4_{block}_40_restricted')(**args, l_1_4_aa=l1_4_aa)
            m4_04 = getattr(gen, f'm4_{block}_04_restricted')(**args, t_1_4_aa=t1_4_aa)
            M4 = m4_13 + m4_31 + m4_22 + m4_40 + m4_04

            blocks[block] = M4 + c2 * M2 + c4 * gamma_hf[block]
        return blocks['oo'], blocks['ov'], blocks['vv']

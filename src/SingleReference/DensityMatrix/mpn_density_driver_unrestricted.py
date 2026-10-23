"""Unrestricted (spin-blocked, genuinely separate alpha/beta, UHF) analogue
of mpn_density_driver_restricted.py -- same M^(n)/N^(n)/c^(n) recursion (see
that module's docstring), wiring together
src/SingleReference/DensityMatrix/generated_mpn_unrestricted/
mpn_density_pieces_unrestricted.py. Scope: MP2 + MP3 density
(compute_delta_gamma2/compute_delta_gamma3) -- all the UHF production
branches consume; no MP4. The order-3 density takes the t3^(2) contributions
either materialized (O(nv^3*no^3) per spin sector) or Laplace-fused
(laplace_ntau, the default MPN_DENSITY_LAPLACE_NTAU; generated mp3_t3_laplace
modules).

l-tag axis convention: same pure-vs-mixed rule as the restricted driver
(to_l's docstring has the full derivation/validation history):
pure blocks (all one spin, now including the genuinely distinct all-beta
ones) get (occ, vir-REVERSED) with sign (-1)**(rank//2); mixed blocks get a
plain occ<->vir side swap, sign +1, no internal reordering.

Convention notes (matching the hand-written MP2/MP3DensityMatrixSolver-
Unrestricted oracle this is validated against in
tests/test_mpn_density_unrestricted.py):
- g_aaaa/g_abab/g_bbbb from get_antisymmetrized_spin_block_eri (abab axis
  order = (p_alpha, q_beta, r_alpha, s_beta), Coulomb-only for mixed spin);
- amplitude storage t{rank}_{block}[vir..., occ...] with the block's pair
  spin pattern giving each axis's spin (abab -> (nv_a, nv_b, no_a, no_b)),
  matching the oracle's own t2_1_abab convention;
- returned density blocks are per-spin and active-space sized, in the
  oracle's (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b) order.
"""
import numpy as np

from src.Base.numerics.constants import MPN_DENSITY_LAPLACE_NTAU

from src.SingleReference.common.mp_amplitudes import orbital_denominator_axes, to_l

from src.SingleReference.DensityMatrix.generated_mpn_unrestricted import mpn_density_pieces_unrestricted as gen
from src.SingleReference.DensityMatrix.generated_mpn_unrestricted import mpn_density_pieces_unrestricted_df as gen_df
from src.SingleReference.DensityMatrix.generated_mpn_unrestricted import mp3_t3_laplace_unrestricted as gen_lap3
from src.SingleReference.DensityMatrix.generated_mpn_unrestricted import mp3_t3_laplace_unrestricted_df as gen_lap3_df

T2_BLOCKS = ('aaaa', 'abab', 'bbbb')
T1_BLOCKS = ('aa', 'bb')
T3_BLOCKS = ('aaaaaa', 'aabaab', 'abbabb', 'bbbbbb')


class MPnDensityDriverUnrestricted:
    """UHF spin-blocked MPn (n=2,3) density-matrix correction.

    f_aa/f_bb are the per-spin spatial-MO Fock matrices (canonical ->
    diagonal; only the diagonals are read), g_* the antisymmetrized-within-
    spin spatial-MO physicist ERI spin blocks from
    get_antisymmetrized_spin_block_eri.
    """

    def __init__(self, f_aa, f_bb, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b, B_aa=None, B_bb=None, custom_denom=None):
        self.g_aaaa, self.g_abab, self.g_bbbb = g_aaaa, g_abab, g_bbbb
        # DF/RI factors: optional --
        # only needed by the compute_*_df methods below. See
        # MPnDensityDriverRestricted's own __init__ docstring note.
        self.B_aa, self.B_bb = B_aa, B_bb
        self.no_a, self.no_b = nocc_a, nocc_b
        self.norb_a, self.norb_b = f_aa.shape[0], f_bb.shape[0]
        self.nv_a, self.nv_b = self.norb_a - nocc_a, self.norb_b - nocc_b
        self.o_a, self.v_a = slice(0, nocc_a), slice(nocc_a, self.norb_a)
        self.o_b, self.v_b = slice(0, nocc_b), slice(nocc_b, self.norb_b)
        self.d_aa, self.d_bb = np.eye(self.norb_a), np.eye(self.norb_b)
        self._eps = {'a': np.diagonal(f_aa).copy(), 'b': np.diagonal(f_bb).copy()}
        self._no = {'a': nocc_a, 'b': nocc_b}
        self.custom_denom = custom_denom or {}

    def _args(self):
        return dict(g_aaaa=self.g_aaaa, g_abab=self.g_abab, g_bbbb=self.g_bbbb,
            d_aa=self.d_aa, d_bb=self.d_bb,
            o_a=self.o_a, v_a=self.v_a, o_b=self.o_b, v_b=self.v_b,
            nv_a=self.nv_a, no_a=self.no_a, nv_b=self.nv_b, no_b=self.no_b)

    def _denom(self, block):
        """D[vir(rank)..., occ(rank)...] = sum(eps_occ) - sum(eps_vir), each
        axis's spin taken from the block's pair pattern (block[:rank])."""
        if self.custom_denom and block in self.custom_denom:
            return self.custom_denom[block]
        spins = block[:len(block) // 2]
        return orbital_denominator_axes([self._eps[s][:self._no[s]] for s in spins],
                                        [self._eps[s][self._no[s]:] for s in spins])

    def _t3_shape(self, block):
        """t3_2_{block}'s own storage shape (vir(3)..., occ(3)...), per-spin
        dims from block[:3] -- used only for laplace-mode zero placeholders
        (see compute_t1_3/compute_delta_gamma3), never for a real
        materialized array."""
        spins = block[:3]
        nv = {'a': self.nv_a, 'b': self.nv_b}
        no = {'a': self.no_a, 'b': self.no_b}
        return tuple(nv[s] for s in spins) + tuple(no[s] for s in spins)

    def compute_t2_1(self):
        args = self._args()
        return {b: getattr(gen, f't2_1_{b}_numerator')(**args) / self._denom(b) for b in T2_BLOCKS}

    def compute_order2_amplitudes(self, t2_1, max_rank=3, laplace_ntau=None):
        """Returns (t1_2, t2_2, t3_2) block dicts; entries above max_rank are
        None (Delta_gamma^(2) only needs rank 1). laplace_ntau: if not None,
        skip materializing all four O(nv^3*no^3) t3_2_* tensors entirely --
        their only two downstream uses (t1_3_{aa,bb}'s and
        m3_ov_{a,b}_12_unrestricted's t3-dependent terms) are instead filled
        in via gen_lap3's Laplace-fused pieces by compute_t1_3/
        compute_delta_gamma3 below (see
        generate_mp3_t3_laplace_unrestricted.py)."""
        args = self._args()
        kw = {f't2_1_{b}': t2_1[b] for b in T2_BLOCKS}
        t1_2 = {b: getattr(gen, f't1_2_{b}_numerator')(**args, **kw) / self._denom(b) for b in T1_BLOCKS}
        t2_2 = t3_2 = None
        if max_rank >= 2:
            t2_2 = {b: getattr(gen, f't2_2_{b}_numerator')(**args, **kw) / self._denom(b) for b in T2_BLOCKS}
        if max_rank >= 3 and laplace_ntau is None:
            t3_2 = {b: getattr(gen, f't3_2_{b}_numerator')(**args, **kw) / self._denom(b) for b in T3_BLOCKS}
        return t1_2, t2_2, t3_2

    def compute_t1_3(self, t1_2, t2_2, t3_2, laplace_t1_3_contrib=None):
        """laplace_t1_3_contrib: optional {'aa': ..., 'bb': ...} dict of the
        Laplace-fused T3^(2)-dependent contribution to t1_3_{aa,bb}'s own
        numerator (gen_lap3.t1_3_aa_t3_laplace/t1_3_bb_t3_laplace's own
        out_indices are ['a','i'], matching t1_3_{aa,bb}_numerator's native
        (nv,no) layout directly -- no transpose needed).

        t3_2 is None in laplace mode: goes through the '_no_t3' generated
        variant (never references any t3_2_* block, not even as a param) --
        NOT the ordinary t1_3_{aa,bb}_numerator with a zero placeholder,
        since the ndim<=4 invariant is enforced per-operand regardless of
        value."""
        args = self._args()
        kw = {}
        kw.update({f't1_2_{b}': t1_2[b] for b in T1_BLOCKS})
        kw.update({f't2_2_{b}': t2_2[b] for b in T2_BLOCKS})
        if t3_2 is None:
            num = {b: getattr(gen, f't1_3_{b}_numerator_no_t3')(**args, **kw) for b in T1_BLOCKS}
        else:
            kw.update({f't3_2_{b}': t3_2[b] for b in T3_BLOCKS})
            num = {b: getattr(gen, f't1_3_{b}_numerator')(**args, **kw) for b in T1_BLOCKS}
        if laplace_t1_3_contrib is not None:
            for b in T1_BLOCKS:
                num[b] = num[b] + laplace_t1_3_contrib[b]
        return {b: num[b] / self._denom(b) for b in T1_BLOCKS}

    def _gamma_hf(self, block, s):
        no = self._no[s]
        nv = self.nv_a if s == 'a' else self.nv_b
        if block == 'oo':
            return np.eye(no)
        return np.zeros((no, no) if block == 'oo' else ((nv, nv) if block == 'vv' else (no, nv)))

    def compute_delta_gamma2(self, t2_1=None, order2=None):
        """Delta_gamma^(2) = M^(2) + c^(2)*M^(0), per spin channel. Returns
        (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b), the hand-written oracle's own
        block order.

        t2_1/order2: optional precomputed compute_t2_1()/
        compute_order2_amplitudes(...) results, to avoid recomputing
        T2^(1)/T1^(2) when the caller also wants Delta_gamma^(3) from the
        same amplitudes -- see compute_delta_gamma23."""
        t2_1 = t2_1 if t2_1 is not None else self.compute_t2_1()
        t1_2 = order2[0] if order2 is not None else self.compute_order2_amplitudes(t2_1, max_rank=1)[0]
        return self._gamma2_blocks(self._args(), t2_1, t1_2)

    def _dead_args(self):
        """The m2_*/m3_* cross-density pieces carry no bracket integral, so the DF
        route calls them with g_aaaa = g_abab = g_bbbb = None (dead parameters)."""
        return dict(g_aaaa=None, g_abab=None, g_bbbb=None, d_aa=self.d_aa, d_bb=self.d_bb,
                    o_a=self.o_a, v_a=self.v_a, o_b=self.o_b, v_b=self.v_b,
                    nv_a=self.nv_a, no_a=self.no_a, nv_b=self.nv_b, no_b=self.no_b)

    def _gamma2_blocks(self, margs, t2_1, t1_2):
        """Delta_gamma^(2) per spin channel from the amplitudes; margs the
        generated pieces' integral arguments (self._args() or self._dead_args())."""
        l2_1 = {b: to_l(t2_1[b], len(b) // 2, b) for b in T2_BLOCKS}
        l1_2 = {b: to_l(t1_2[b], len(b) // 2, b) for b in T1_BLOCKS}

        N2 = gen.overlap2_unrestricted(**{f'l_{b}': l2_1[b] for b in T2_BLOCKS},
                                       **{f't_{b}': t2_1[b] for b in T2_BLOCKS})
        c2 = -N2

        out = {}
        for s in ('a', 'b'):
            for block in ('oo', 'vv', 'ov'):
                m11 = getattr(gen, f'm2_{block}_{s}_11_unrestricted')(
                    **margs, **{f'l_2_1_{b}': l2_1[b] for b in T2_BLOCKS},
                    **{f't_2_1_{b}': t2_1[b] for b in T2_BLOCKS})
                m20 = getattr(gen, f'm2_{block}_{s}_20_unrestricted')(
                    **margs, **{f'l_1_2_{b}': l1_2[b] for b in T1_BLOCKS})
                m02 = getattr(gen, f'm2_{block}_{s}_02_unrestricted')(
                    **margs, **{f't_1_2_{b}': t1_2[b] for b in T1_BLOCKS})
                out[(block, s)] = m11 + m20 + m02 + c2 * self._gamma_hf(block, s)
        return (out[('oo', 'a')], out[('oo', 'b')], out[('ov', 'a')],
                out[('ov', 'b')], out[('vv', 'a')], out[('vv', 'b')])

    def _args_df(self):
        return dict(B_aa=self.B_aa, B_bb=self.B_bb, d_aa=self.d_aa, d_bb=self.d_bb,
                    o_a=self.o_a, v_a=self.v_a, o_b=self.o_b, v_b=self.v_b,
                    nv_a=self.nv_a, no_a=self.no_a, nv_b=self.nv_b, no_b=self.no_b)

    def compute_t2_1_df(self):
        """DF/RI variant of compute_t2_1: requires B_aa/B_bb (see __init__)."""
        args = self._args_df()
        return {b: getattr(gen_df, f't2_1_{b}_numerator_df')(**args) / self._denom(b) for b in T2_BLOCKS}

    def compute_t1_2_df(self, t2_1):
        """DF/RI variant of the T1^(2) piece of compute_order2_amplitudes."""
        args = self._args_df()
        kw = {f't2_1_{b}': t2_1[b] for b in T2_BLOCKS}
        return {b: getattr(gen_df, f't1_2_{b}_numerator_df')(**args, **kw) / self._denom(b) for b in T1_BLOCKS}

    def compute_delta_gamma2_df(self, t2_1=None, t1_2=None):
        """DF/RI variant of compute_delta_gamma2: T2^(1)/T1^(2) from the
        DF-dressed numerators, the bracket-free m2_* pieces with dead integral
        arguments (_dead_args)."""
        t2_1 = t2_1 if t2_1 is not None else self.compute_t2_1_df()
        t1_2 = t1_2 if t1_2 is not None else self.compute_t1_2_df(t2_1)
        return self._gamma2_blocks(self._dead_args(), t2_1, t1_2)

    def compute_order2_amplitudes_df(self, t2_1, laplace_ntau=None, max_rank=3):
        """DF/RI variant of compute_order2_amplitudes. max_rank>3 not supported (MP4 out of scope,
        no DF-dressed t4_2_* -- this driver has no T4^(2) at all anyway)."""
        if max_rank > 3:
            raise NotImplementedError("compute_order2_amplitudes_df: max_rank>3 not supported")
        args = self._args_df()
        kw = {f't2_1_{b}': t2_1[b] for b in T2_BLOCKS}
        t1_2 = {b: getattr(gen_df, f't1_2_{b}_numerator_df')(**args, **kw) / self._denom(b) for b in T1_BLOCKS}
        t2_2 = t3_2 = None
        if max_rank >= 2:
            t2_2 = {b: getattr(gen_df, f't2_2_{b}_numerator_df')(**args, **kw) / self._denom(b) for b in T2_BLOCKS}
        if max_rank >= 3 and laplace_ntau is None:
            t3_2 = {b: getattr(gen_df, f't3_2_{b}_numerator_df')(**args, **kw) / self._denom(b) for b in T3_BLOCKS}
        return t1_2, t2_2, t3_2

    def compute_t1_3_df(self, t1_2, t2_2, t3_2, laplace_t1_3_contrib=None):
        """DF/RI variant of compute_t1_3 -- never forms
        g_aaaa[o,v,v,v]/g_abab[v,o,v,v]-scale (or bb/baba analogues)
        integral blocks."""
        args = self._args_df()
        kw = {}
        kw.update({f't1_2_{b}': t1_2[b] for b in T1_BLOCKS})
        kw.update({f't2_2_{b}': t2_2[b] for b in T2_BLOCKS})
        if t3_2 is None:
            num = {b: getattr(gen_df, f't1_3_{b}_numerator_no_t3_df')(**args, **kw) for b in T1_BLOCKS}
        else:
            kw.update({f't3_2_{b}': t3_2[b] for b in T3_BLOCKS})
            num = {b: getattr(gen_df, f't1_3_{b}_numerator_df')(**args, **kw) for b in T1_BLOCKS}
        if laplace_t1_3_contrib is not None:
            for b in T1_BLOCKS:
                num[b] = num[b] + laplace_t1_3_contrib[b]
        return {b: num[b] / self._denom(b) for b in T1_BLOCKS}

    def compute_delta_gamma3_df(self, laplace_ntau=MPN_DENSITY_LAPLACE_NTAU, t2_1=None, order2=None):
        """DF/RI variant of compute_delta_gamma3 -- T2^(1)/T1^(2)/T2^(2)/
        T3^(2)/T1^(3) come from the DF-dressed numerators above (including
        the DF+Laplace-composed gen_lap3_df pieces); every m3_* cross-
        density piece is bracket-free (same reasoning as
        compute_delta_gamma2_df's m2_* pieces) and is called UNCHANGED from
        `gen` with g_aaaa=g_abab=g_bbbb=None (dead params)."""
        t2_1 = t2_1 if t2_1 is not None else self.compute_t2_1_df()
        order2 = order2 if order2 is not None else \
            self.compute_order2_amplitudes_df(t2_1, max_rank=3, laplace_ntau=laplace_ntau)
        return self._gamma3_blocks(True, laplace_ntau, t2_1, order2)

    def compute_delta_gamma23(self, laplace_ntau=MPN_DENSITY_LAPLACE_NTAU):
        """Delta_gamma^(2) and Delta_gamma^(3) together, computing T2^(1)/
        T1^(2)/T2^(2) exactly once instead of once per call (production MP3
        density always needs both -- see the restricted driver's own
        compute_delta_gamma23). Returns (gamma2_blocks, gamma3_blocks), each
        an (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b) tuple."""
        t2_1 = self.compute_t2_1()
        order2 = self.compute_order2_amplitudes(t2_1, max_rank=3, laplace_ntau=laplace_ntau)
        gamma2 = self.compute_delta_gamma2(t2_1=t2_1, order2=order2)
        gamma3 = self.compute_delta_gamma3(laplace_ntau=laplace_ntau, t2_1=t2_1, order2=order2)
        return gamma2, gamma3

    def compute_delta_gamma3(self, laplace_ntau=MPN_DENSITY_LAPLACE_NTAU, t2_1=None, order2=None):
        """Delta_gamma^(3) = M^(3) + c^(3)*M^(0), per spin channel (c^(2)
        only ever multiplies M^(1) = 0 -- see the restricted driver). Returns
        (oo_a, oo_b, ov_a, ov_b, vv_a, vv_b).

        laplace_ntau: if not None (default: production laplace_ntau=MPN_DENSITY_LAPLACE_NTAU,
        matching the restricted driver's own production default), none of
        the four O(nv^3*no^3) t3_2_* tensors are ever materialized -- their
        only two live consumers (t1_3_{aa,bb}'s and
        m3_ov_{a,b}_12_unrestricted's t3-dependent terms) are filled in via
        gen_lap3 instead (see generate_mp3_t3_laplace_unrestricted.py). Every
        other m3_{oo,vv}_{a,b}_12/m3_{oo,vv,ov}_{a,b}_21 body declares
        t_3_2_*/l_3_2_* as a parameter but never references it in an einsum
        (confirmed dead, same pattern as the restricted driver), so a zero
        placeholder there is always exact, laplace mode or not -- pass
        laplace_ntau=None to get the old fully-materialized behavior (kept
        for tests/oracle cross-checks only)."""
        t2_1 = t2_1 if t2_1 is not None else self.compute_t2_1()
        order2 = order2 if order2 is not None else \
            self.compute_order2_amplitudes(t2_1, max_rank=3, laplace_ntau=laplace_ntau)
        return self._gamma3_blocks(False, laplace_ntau, t2_1, order2)

    def _gamma3_blocks(self, df, laplace_ntau, t2_1, order2):
        """Delta_gamma^(3) per spin channel from T2^(1) and the order-2
        amplitudes (t1_2, t2_2, t3_2), dense (df False) or DF (df True: DF
        Laplace pieces and t1_3, dead integral arguments for the m3_* pieces)."""
        args = self._args_df() if df else self._args()
        margs = self._dead_args() if df else args
        lap = gen_lap3_df if df else gen_lap3
        sfx = '_df' if df else ''
        t1_2, t2_2, t3_2 = order2

        l2_1 = {b: to_l(t2_1[b], len(b) // 2, b) for b in T2_BLOCKS}
        l1_2 = {b: to_l(t1_2[b], len(b) // 2, b) for b in T1_BLOCKS}
        l2_2 = {b: to_l(t2_2[b], len(b) // 2, b) for b in T2_BLOCKS}

        laplace_t1_3_contrib = laplace_ov_contrib = None
        if laplace_ntau is not None:
            laplace_t1_3_contrib = {
                'aa': getattr(lap, 't1_3_aa_t3_laplace' + sfx)(**args, eps_a=self._eps['a'], eps_b=self._eps['b'],
                                                  **{f't2_1_{b}': t2_1[b] for b in T2_BLOCKS}, ntau=laplace_ntau),
                'bb': getattr(lap, 't1_3_bb_t3_laplace' + sfx)(**args, eps_a=self._eps['a'], eps_b=self._eps['b'],
                                                  **{f't2_1_{b}': t2_1[b] for b in T2_BLOCKS}, ntau=laplace_ntau),
            }
            laplace_ov_contrib = {
                'a': getattr(lap, 'm3_ov_a_t3_laplace' + sfx)(**args, eps_a=self._eps['a'], eps_b=self._eps['b'],
                                                 **{f't2_1_{b}': t2_1[b] for b in T2_BLOCKS},
                                                 **{f'l_2_1_{b}': l2_1[b] for b in T2_BLOCKS}, ntau=laplace_ntau),
                'b': getattr(lap, 'm3_ov_b_t3_laplace' + sfx)(**args, eps_a=self._eps['a'], eps_b=self._eps['b'],
                                                 **{f't2_1_{b}': t2_1[b] for b in T2_BLOCKS},
                                                 **{f'l_2_1_{b}': l2_1[b] for b in T2_BLOCKS}, ntau=laplace_ntau),
            }

        t1_3 = (self.compute_t1_3_df if df else self.compute_t1_3)(t1_2, t2_2, t3_2, laplace_t1_3_contrib=laplace_t1_3_contrib)

        z3 = {b: np.zeros(self._t3_shape(b)) for b in T3_BLOCKS}
        t3_arg = t3_2 if t3_2 is not None else z3
        l3_2 = z3  # declared-but-unused in m3_{oo,vv,ov}_{a,b}_21 (see docstring)
        l1_3 = {b: to_l(t1_3[b], len(b) // 2, b) for b in T1_BLOCKS}

        N3 = 2.0 * gen.overlap2_unrestricted(**{f'l_{b}': l2_1[b] for b in T2_BLOCKS},
                                             **{f't_{b}': t2_2[b] for b in T2_BLOCKS})
        c3 = -N3

        out = {}
        for s in ('a', 'b'):
            for block in ('oo', 'vv', 'ov'):
                if block == 'ov' and t3_2 is None:
                    # Laplace mode: m3_ov_{a,b}_12_unrestricted's ENTIRE
                    # contribution is t3-dependent (0 terms from ranks 1,2
                    # alone. See compute_t1_3's
                    # matching comment for why a zero t_3_2_* placeholder
                    # isn't an option (ndim<=4 is enforced per-operand).
                    m12 = getattr(gen, f'm3_ov_{s}_12_unrestricted_no_t3')(
                        **margs, **{f'l_2_1_{b}': l2_1[b] for b in T2_BLOCKS},
                        **{f't_1_2_{b}': t1_2[b] for b in T1_BLOCKS},
                        **{f't_2_2_{b}': t2_2[b] for b in T2_BLOCKS})
                else:
                    m12 = getattr(gen, f'm3_{block}_{s}_12_unrestricted')(
                        **margs, **{f'l_2_1_{b}': l2_1[b] for b in T2_BLOCKS},
                        **{f't_1_2_{b}': t1_2[b] for b in T1_BLOCKS},
                        **{f't_2_2_{b}': t2_2[b] for b in T2_BLOCKS},
                        **{f't_3_2_{b}': t3_arg[b] for b in T3_BLOCKS})
                if block == 'ov' and laplace_ov_contrib is not None:
                    m12 = m12 + laplace_ov_contrib[s]
                m21 = getattr(gen, f'm3_{block}_{s}_21_unrestricted')(
                    **margs, **{f'l_1_2_{b}': l1_2[b] for b in T1_BLOCKS},
                    **{f'l_2_2_{b}': l2_2[b] for b in T2_BLOCKS},
                    **{f'l_3_2_{b}': l3_2[b] for b in T3_BLOCKS},
                    **{f't_2_1_{b}': t2_1[b] for b in T2_BLOCKS})
                m30 = getattr(gen, f'm3_{block}_{s}_30_unrestricted')(
                    **margs, **{f'l_1_3_{b}': l1_3[b] for b in T1_BLOCKS})
                m03 = getattr(gen, f'm3_{block}_{s}_03_unrestricted')(
                    **margs, **{f't_1_3_{b}': t1_3[b] for b in T1_BLOCKS})
                out[(block, s)] = m12 + m21 + m30 + m03 + c3 * self._gamma_hf(block, s)
        return (out[('oo', 'a')], out[('oo', 'b')], out[('ov', 'a')],
                out[('ov', 'b')], out[('vv', 'a')], out[('vv', 'b')])

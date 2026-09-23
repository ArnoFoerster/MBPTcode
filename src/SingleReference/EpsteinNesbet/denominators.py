"""Epstein-Nesbet-dressed MP1 denominators for a restricted (RHF, closed-shell)
reference: the per-channel shift matrices (restricted_channel_shifts) and the on-demand
4-index denominator builder (EpsteinNesbetDenominators) that turns them into the
denominators a T2^(1) doubles amplitude needs.

See shifts.py for the underlying Epstein-Nesbet partitioning and the spin-adapted vs
determinant-wise (Jiang & Engel) distinction.
"""
import numpy as np

from src.Base.pyscf_interface import get_two_electron_integrals_chemist
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import _denom_restricted
from src.SingleReference.EpsteinNesbet.shifts import (
    epstein_nesbet_shift_restricted_spinadapted,
    epstein_nesbet_shift_restricted_spinresolved)
from src.SingleReference.LinearResponse.linear_response import (
    static_screened_coulomb_aux, static_screened_coulomb_chemist,
    static_screened_coulomb_chemist_uhf)


def restricted_channel_shifts(dress, B_aa, g_oooo, g_vvvv, O, vidx_abs,
                              g_vovo=None, g_voov=None):
    """Hole-hole (d_h, (O,O)), particle-particle (d_p, (V,V)) and hole-particle cross
    (d_hp, (V,O)) Epstein-Nesbet shifts requested by the `dress` dict, from whichever
    integral SOURCE is available: dense physicist g_oooo/g_vvvv/g_vovo/g_voov, or
    DF-native from B_aa. g_vvvv may be None (the memory-lean DF path never builds it) as
    long as B_aa is given whenever pp or hp is requested.

    dress keys: 'hh'/'pp'/'hp' (channel toggles, default False), 'spin_adapted' (default
    True) and 'shift' (the EN_SPIN_WEIGHTINGS choice, default 'mean').

    Returns (d_h, d_p, d_hp), each either None (channel off) or a (same_spin,
    opposite_spin) PAIR of matrices:

        same:  <pq||pq> = J - K, zero on the diagonal
        opp:   <pq|pq>  = J,     diagonal KEPT

    In spin-adapted mode the pair is the SAME object twice, which is how
    EpsteinNesbetDenominators.spin_adapted detects the mode. Each channel enters Jiang &
    Engel Eq. 20 with a MINUS sign per (particle,hole) pair:
        Delta^EN = <ij||ij> + <ab||ab> - <ai||ai> - <aj||aj> - <bi||bi> - <bj||bj>
    with the antisymmetrized or plain integral chosen per pair by its spins
    (EpsteinNesbetDenominators._delta does that bookkeeping)."""
    # spin_adapted=True is the only mode a CSF solver can represent: it averages the two
    # spin cases into ONE shift per channel, so D_same == D_opp and the dressed
    # amplitudes stay spin-pure. False gives the determinant-wise (Jiang & Engel) shifts,
    # correct only for a spin-BLOCKED solver.
    spin_adapted = dress.get('spin_adapted', True)
    weighting = dress.get('shift', 'mean')
    if spin_adapted:
        def _shift(g_same=None, B_block=None):
            d = epstein_nesbet_shift_restricted_spinadapted(
                g_same=g_same, B_block=B_block, weighting=weighting)
            return (d, d)          # same object -> one denominator downstream
    else:
        _shift = epstein_nesbet_shift_restricted_spinresolved
    hh = dress.get('hh', False)
    pp = dress.get('pp', False)
    hp = dress.get('hp', False)
    d_h = d_p = d_hp = None
    if hh:
        if g_oooo is not None:
            d_h = _shift(g_same=g_oooo)
        else:
            d_h = _shift(B_block=B_aa[:, :O, :O])
    if pp:
        if g_vvvv is not None:
            d_p = _shift(g_same=g_vvvv)
        else:
            d_p = _shift(B_block=B_aa[:, vidx_abs][:, :, vidx_abs])
    if hp:
        if g_vovo is not None and g_voov is not None:
            # g_vovo[a,i,c,k] = <ai|ck> (physicist); the diagonal needed here is
            # direct   <ai|ai> = einsum('aiai->ai', g_vovo)
            # exchange <ai|ia> = einsum('aiia->ai', g_voov)
            direct_hp = np.einsum('aiai->ai', g_vovo, optimize=True)    # (V,O)
            exchange_hp = np.einsum('aiia->ai', g_voov, optimize=True)  # (V,O)
        elif B_aa is not None:
            # DF-native: <ai|ai> = sum_Q B[Q,a,a]*B[Q,i,i], <ai|ia> = sum_Q B[Q,a,i]^2
            Bv = B_aa[:, vidx_abs][:, :, vidx_abs]   # (naux, V, V)
            Bo = B_aa[:, :O, :O]                      # (naux, O, O)
            Bvo = B_aa[:, vidx_abs][:, :, :O]         # (naux, V, O)
            diag_v = np.einsum('Qaa->Qa', Bv, optimize=True)
            diag_o = np.einsum('Qii->Qi', Bo, optimize=True)
            direct_hp = diag_v.T @ diag_o
            exchange_hp = np.einsum('Qai,Qai->ai', Bvo, Bvo, optimize=True)
        else:
            raise ValueError("hp=True requires either dense g_vovo/g_voov or B_aa "
                             "(DF integrals)")
        # A particle and a hole are never the same spatial orbital, so there is no
        # diagonal to special-case here (unlike the hh/pp channels).
        if spin_adapted:
            d = {'mean': direct_hp - 0.5 * exchange_hp,
                 'opposite': direct_hp,
                 'sum': 2.0 * direct_hp - exchange_hp}[weighting]
            d_hp = (d, d)
        else:
            d_hp = (direct_hp - exchange_hp, direct_hp)
    return d_h, d_p, d_hp


class EpsteinNesbetDenominators:
    """EN-dressed MP1 denominators for the T2^(1) doubles amplitude, built on demand for
    BOTH spin cases from the small (O,O)/(V,V)/(V,O) shift matrices.

    Why two: the Epstein-Nesbet shift is a diagonal Hamiltonian element of one
    DETERMINANT, so it depends on the spin case and does not collapse to a single
    spin-adapted number (see shifts.epstein_nesbet_shift_restricted_spinresolved). Every
    quotient a restricted U-block forms is a unique linear combination of the same-spin
    (aaaa) and opposite-spin (abab) T2^(1) amplitudes, so each half must carry its OWN
    denominator:

        with  v = g_oovv            (== t2_1_abab numerator, verified 5.6e-17)
              u = g_oovv - g_oovv_T (== t2_1_aaaa numerator, verified 8.8e-17)
        u/D -> u/D_same   and   v/D -> v/D_opp

    The decomposition is unique because u and v are linearly independent, so this
    substitution is exact, not a modelling choice.

    Spin bookkeeping for the opposite-spin case, in the abab convention
    [h0=i(alpha), h1=k(beta), p0=c(alpha), p1=a(beta)] (matches
    static_correction._build_dressed_denoms_uhf's abab branch exactly): the SAME-spin hp
    pairs are (p0,h0) and (p1,h1); the two cross pairs (p0,h1) and (p1,h0) are
    opposite-spin and carry no exchange.

    Holds only O(O^2 + V^2 + O*V) state -- the 4-index denominators are built per layout
    on request and can be dropped as soon as the amplitude that needs them has been
    divided, so nothing here forces an O^2V^2 array to stay resident. That matters at
    cc-pVQZ scale, where one O^2V^2 array is ~89 GB for hexacene.
    """

    # layout -> (hole axes, particle axes, bare_sign); bare_sign=+1 when D_bare is
    # (sum of hole eps) - (sum of particle eps), so that D = D_bare - bare_sign*Delta
    # always increases |D_bare| for the hh/pp ladders (shifts.epstein_nesbet_denominator's
    # convention).
    _LAYOUTS = {
        'ikca': ((0, 1), (2, 3), +1.0),   # [i,k,c,a]  eps_i+eps_k-eps_c-eps_a
        'ijcd': ((0, 1), (2, 3), -1.0),   # [i,j,c,d]  eps_c+eps_d-eps_i-eps_j
        'ackl': ((2, 3), (0, 1), -1.0),   # [a,c,k,l]  eps_a+eps_c-eps_k-eps_l
        'abkl': ((2, 3), (0, 1), +1.0),   # [a,b,k,l]  eps_k+eps_l-eps_a-eps_b
    }

    def __init__(self, eps_o, eps_v, dh=None, dp=None, dhp=None):
        """dh/dp/dhp: (same, opp) tuples of shift matrices (restricted_channel_shifts),
        or None for a channel that is switched off. dh is (O,O), dp is (V,V), dhp is
        (V,O)."""
        self.eps_o, self.eps_v = eps_o, eps_v
        self.O, self.V = len(eps_o), len(eps_v)
        self.dh, self.dp, self.dhp = dh, dp, dhp
        self.dressed = not (dh is None and dp is None and dhp is None)

    @staticmethod
    def _place(mat, ax_a, ax_b, ndim=4):
        b = [1] * ndim
        b[ax_a], b[ax_b] = mat.shape
        return mat.reshape(b)

    def _bare(self, layout):
        h_ax, p_ax, sign = self._LAYOUTS[layout]
        eo, ev = self.eps_o, self.eps_v
        out = np.zeros(self._shape(layout))
        for ax in h_ax:
            b = [1, 1, 1, 1]; b[ax] = self.O
            out = out + sign * eo.reshape(b)
        for ax in p_ax:
            b = [1, 1, 1, 1]; b[ax] = self.V
            out = out - sign * ev.reshape(b)
        return out

    def _shape(self, layout):
        h_ax, p_ax, _ = self._LAYOUTS[layout]
        s = [0, 0, 0, 0]
        for ax in h_ax:
            s[ax] = self.O
        for ax in p_ax:
            s[ax] = self.V
        return tuple(s)

    def _delta(self, layout, spin):
        """Jiang & Engel's Delta for one spin case:
            same: Delta = <h0 h1||h0 h1> + <p0 p1||p0 p1> - sum_{4 pairs} <p h||p h>
            opp : hh/pp use the no-exchange <..|..>; of the 4 hp pairs only
                  (p0,h0) and (p1,h1) are same-spin (exchange survives)."""
        h_ax, p_ax, _ = self._LAYOUTS[layout]
        h0, h1 = h_ax
        p0, p1 = p_ax
        k = 0 if spin == 'same' else 1
        t = np.zeros(self._shape(layout))
        if self.dh is not None:
            t = t + self._place(self.dh[k], h0, h1)
        if self.dp is not None:
            t = t + self._place(self.dp[k], p0, p1)
        if self.dhp is not None:
            s_hp = self.dhp[0]                       # same-spin (V,O)
            o_hp = self.dhp[k]                       # cross pairs: same->same, opp->opp
            t = t - self._place(s_hp, p0, h0) - self._place(s_hp, p1, h1)
            t = t - self._place(o_hp, p0, h1) - self._place(o_hp, p1, h0)
        return t

    def denom(self, layout, spin):
        """One dressed denominator ('same' or 'opp') for `layout`. Use this when only one
        spin case is consumed (e.g. the pure opposite-spin X_ijcd/X_abkl quotients) so
        the other is never built."""
        D = self._bare(layout)
        if not self.dressed:
            return D
        return D - self._LAYOUTS[layout][2] * self._delta(layout, spin)

    @property
    def spin_adapted(self):
        """True when every channel carries ONE shift for both spin cases, so
        D_same == D_opp identically and the dressed amplitudes stay spin-pure (see
        shifts.epstein_nesbet_shift_restricted_spinadapted). Detected by object identity,
        which restricted_channel_shifts guarantees in spin-adapted mode."""
        return all(c is None or c[0] is c[1] for c in (self.dh, self.dp, self.dhp))

    def build(self, layout):
        """(D_same, D_opp) for `layout`. Undressed OR spin-adapted -> the SAME array
        object twice, so every consumer reduces exactly to its former single-denominator
        form (and callers can detect it).

        That collapse is the whole point of the spin-adapted mode: with one denominator
        the merged quotients the U-blocks are written in -- e.g. (g - g^T)/D -- are
        exactly valid again, because the identity that justifies the merge
        (t_aaaa == t_abab - t_abab^T) is restored."""
        if not self.dressed:
            D = self._bare(layout)
            return D, D
        if self.spin_adapted:
            D = self.denom(layout, 'same')
            return D, D
        return self.denom(layout, 'same'), self.denom(layout, 'opp')

    def build_icd(self):
        """D_opp for the doubled-hole slice [i,c,d] (both holes = i).

        Only the OPPOSITE-spin denominator exists here: the same-spin T2^(1) amplitude is
        antisymmetric in its occupied pair and so vanishes identically at i=j.
        Correspondingly the hh shift is <ii|ii> = J_ii, the diagonal element the old
        spin-summed code zeroed out."""
        eo, ev = self.eps_o, self.eps_v
        D = (ev[None, :, None] + ev[None, None, :] - 2 * eo[:, None, None])
        if not self.dressed:
            return D
        t = np.zeros(D.shape)
        if self.dh is not None:
            t = t + np.diag(self.dh[1]).reshape(self.O, 1, 1)     # <ii|ii>
        if self.dp is not None:
            t = t + self.dp[1].reshape(1, self.V, self.V)         # <cd|cd>
        if self.dhp is not None:
            s_hp, o_hp = self.dhp[0], self.dhp[1]                 # (V,O)
            for mat in (s_hp, o_hp):
                t = t - mat.T.reshape(self.O, self.V, 1)          # (c,i)
                t = t - mat.T.reshape(self.O, 1, self.V)          # (d,i)
        return D + t                                              # bare_sign = -1

    def build_akl(self):
        """D_opp for the doubled-particle slice [a,k,l] (both particles = a). Mirror of
        build_icd; the pp shift here is <aa|aa> = J_aa."""
        eo, ev = self.eps_o, self.eps_v
        D = (eo[None, :, None] + eo[None, None, :] - 2 * ev[:, None, None])
        if not self.dressed:
            return D
        t = np.zeros(D.shape)
        if self.dh is not None:
            t = t + self.dh[1].reshape(1, self.O, self.O)         # <kl|kl>
        if self.dp is not None:
            t = t + np.diag(self.dp[1]).reshape(self.V, 1, 1)     # <aa|aa>
        if self.dhp is not None:
            s_hp, o_hp = self.dhp[0], self.dhp[1]                 # (V,O)
            for mat in (s_hp, o_hp):
                t = t - mat.reshape(self.V, self.O, 1)            # (a,k)
                t = t - mat.reshape(self.V, 1, self.O)            # (a,l)
        return D - t                                              # bare_sign = +1


def _build_dressed_e_ai(u2_denom_dress, eps, O, V, B_aa=None, eri_chemist=None):
    """
    Epstein-Nesbet-dressed inverse denominator 1/D for the t1^(2) singles
    amplitude (Jiang & Engel Eq. 21):

        D_ai = eps_i - eps_a + <ia||ia>,   <ia||ia> = (ii|aa) - (ia|ai) = J - K

    singles='screened' replaces the e-h direct integral J by its static
    (omega=0) full-RPA-screened counterpart,

        D_ai = eps_i - eps_a + (ii|W|aa) - (ia|ai) = J_W - K

    with the BSE-kernel channel structure: the term screened in the CIS/BSE
    diagonal is the e-h DIRECT attraction (-J -> -W); the exchange integral
    (ia|ia) is the RPA ring/bubble term W itself resums, so it must stay
    bare -- screening it instead would double-count rings (same channel
    logic as screened_coulomb's gmix e-h slice). With ncore>0 the RPA solve
    runs in the active window, i.e. core excitations are excluded from the
    screening.
    """
    singles = u2_denom_dress.get('singles', True)
    if singles is False:
        return None
    D_bare = _denom_restricted(eps, 1, O, V)

    if B_aa is not None:
        vidx_abs = np.arange(O, O + V)
        Bv = B_aa[:, vidx_abs][:, :, vidx_abs]   # (naux, V, V)
        Bo = B_aa[:, :O, :O]                      # (naux, O, O)
        Bvo = B_aa[:, vidx_abs][:, :, :O]          # (naux, V, O)
        diag_v = np.einsum('Qaa->Qa', Bv, optimize=True)   # (naux, V)
        diag_o = np.einsum('Qii->Qi', Bo, optimize=True)   # (naux, O)
        if singles == 'screened':
            # (aa|W|ii) = sum_PQ B[P,a,a] W_aux[P,Q] B[Q,i,i], with W_aux the
            # (naux,naux) static inverse-dielectric metric (bare part included).
            W_aux = static_screened_coulomb_aux(eps, B_aa, O)
            direct_hp = diag_v.T @ (W_aux @ diag_o)          # (V,O)
        else:
            direct_hp = diag_v.T @ diag_o          # (V,O): sum_Q B[Q,a,a]*B[Q,i,i]
        exchange_hp = np.einsum('Qai,Qai->ai', Bvo, Bvo, optimize=True)  # (V,O)
    else:
        # (aa|ii) and (ai|ai) gathered without the O(V*O) python loop this
        # used to run (~110k iterations at hexacene/cc-pVQZ scale).
        vsl = slice(O, O + V)
        if singles == 'screened':
            W = static_screened_coulomb_chemist(eps, eri_chemist, O)
            direct_hp = np.einsum('aaii->ai', W[vsl, vsl, :O, :O], optimize=True)
        else:
            direct_hp = np.einsum('aaii->ai', eri_chemist[vsl, vsl, :O, :O], optimize=True)
        exchange_hp = np.einsum('aiai->ai', eri_chemist[vsl, :O, vsl, :O], optimize=True)

    d_hp = direct_hp - exchange_hp               # <ia||ia> = J - K  (J_W - K screened)
    D_dressed = D_bare + d_hp
    return 1.0 / D_dressed


def _build_dressed_e_abij(u2_denom_dress, eps, O, V, B_aa=None, eri_chemist=None):
    """(e_abij_aaaa, e_abij_abab): the EN-dressed inverse T2^(1) denominators
    for the SAME-spin and OPPOSITE-spin amplitudes, each (V,V,O,O).

    Two, not one: the EN shift is a diagonal element of a determinant, so it
    differs between t2_1_aaaa and t2_1_abab (same-spin keeps the exchange and
    has no p==q configuration; opposite-spin has no exchange and DOES have a
    legitimate p==q term). A single array built from the spin-summed 2J-K
    shift with its diagonal zeroed, applied to both blocks, is NOT any
    determinant's denominator -- the two must stay separate, matching the
    UHF path (_build_dressed_denoms_uhf).

    The layout here is [a,b,i,j] (particles first), so relative to
    EpsteinNesbetDenominators' 'ackl' key the hole/particle axes coincide and the SAME
    spin pairing applies: (p0,h0) = (a,i) and (p1,h1) = (b,j) are the
    same-spin hp pairs, the two cross pairs are opposite-spin."""
    D_bare = _denom_restricted(eps, 2, O, V)
    vidx_abs = np.arange(O, O + V)
    g_oooo = g_vvvv = g_vovo = g_voov = None
    if B_aa is None and eri_chemist is not None:
        g = eri_chemist.transpose(0, 2, 1, 3)
        g_oooo = g[:O, :O, :O, :O]
        g_vvvv = g[O:, O:, O:, O:]
        g_vovo = g[O:, :O, O:, :O]
        g_voov = g[O:, :O, :O, O:]
    d_h, d_p, d_hp = restricted_channel_shifts(u2_denom_dress, B_aa, g_oooo, g_vvvv,
                                              O, vidx_abs, g_vovo=g_vovo, g_voov=g_voov)
    dens = EpsteinNesbetDenominators(eps[:O], eps[O:O + V], d_h, d_p, d_hp)

    def _shift(spin):
        # D_bare here is (holes - particles) in [a,b,i,j] order, i.e. exactly
        # EpsteinNesbetDenominators' 'abkl' layout (particles @ (0,1), holes @ (2,3),
        # bare_sign = +1 -> D = D_bare - Delta).
        return dens._delta('abkl', spin)

    if not dens.dressed:
        return 1.0 / D_bare, 1.0 / D_bare
    if dens.spin_adapted:
        # ONE denominator for both spin blocks -- this is what keeps the
        # dressed T2^(1) spin-pure (t_aaaa == t_abab - t_abab^T). Returning the
        # same object also lets MPnDensityDriverRestricted skip a division.
        e = 1.0 / (D_bare - _shift('same'))
        return e, e
    return 1.0 / (D_bare - _shift('same')), 1.0 / (D_bare - _shift('opp'))


def _extract_singles_shift_uhf(eri, no, W=None):
    """<ia||ia> = J-K (or J_W-K if W is given) for ONE spin channel, from the
    RAW (non-antisymmetrized) same-spin chemist tensor eri[p,q,r,s]=(pq|rs)
    -- unlike _extract_d_hp (which reads the diagonal of an already-
    antisymmetrized g and so cannot isolate J from K), this needs J and K
    separately because only J gets screened. Same layout convention as
    _build_dressed_e_ai (direct 'aaii->ai', exchange 'aiai->ai'). Returns
    (V, O)."""
    v = slice(no, eri.shape[0])
    direct = np.einsum('aaii->ai', (eri if W is None else W)[v, v, :no, :no],
                       optimize=True)
    exchange = np.einsum('aiai->ai', eri[v, :no, v, :no], optimize=True)
    return direct - exchange


def _build_dressed_denoms_uhf(u2_denom_dress, eps_a, eps_b, g_aaaa, g_abab, g_bbbb, nocc_a, nocc_b, ncore,
                              mol=None, mf_eval=None):
    """mol/mf_eval: only needed (and only used) when u2_denom_dress['singles']
    == 'screened' -- builds the raw per-spin chemist ERI and the static-RPA-
    screened same-spin tensors from them (an extra O(norb^4) AO->MO
    transform, paid only in that case, not on the bare/doubles-only/hh-hp
    paths)."""
    w = slice(ncore, None)
    ea = eps_a[w]
    eb = eps_b[w]
    Oa = nocc_a - ncore
    Ob = nocc_b - ncore
    Va = len(ea) - Oa
    Vb = len(eb) - Ob

    g_aaaa_w = g_aaaa[w, w, w, w]
    g_bbbb_w = g_bbbb[w, w, w, w]
    g_abab_w = g_abab[w, w, w, w]

    hh = u2_denom_dress.get('hh', False)
    pp = u2_denom_dress.get('pp', False)
    hp = u2_denom_dress.get('hp', False)

    def _extract_d_h(g, no):
        return np.einsum('ijij->ij', g[:no, :no, :no, :no], optimize=True)
    def _extract_d_p(g, no):
        return np.einsum('abab->ab', g[no:, no:, no:, no:], optimize=True)
    def _extract_d_hp(g, no):
        return np.einsum('iaia->ai', g[:no, no:, :no, no:], optimize=True)

    custom_denom = {}

    # 1. aaaa sector
    D_bare_aaaa = (ea[:Oa].reshape(1, 1, Oa, 1) + ea[:Oa].reshape(1, 1, 1, Oa)
                   - ea[Oa:].reshape(Va, 1, 1, 1) - ea[Oa:].reshape(1, Va, 1, 1))
    shift_aaaa = np.zeros_like(D_bare_aaaa)
    if hh:
        d_h_a = _extract_d_h(g_aaaa_w, Oa)
        shift_aaaa += d_h_a.reshape(1, 1, Oa, Oa)
    if pp:
        d_p_a = _extract_d_p(g_aaaa_w, Oa)
        shift_aaaa += d_p_a.reshape(Va, Va, 1, 1)
    if hp:
        d_hp_a = _extract_d_hp(g_aaaa_w, Oa)
        shift_aaaa -= d_hp_a.reshape(Va, 1, Oa, 1)
        shift_aaaa -= d_hp_a.reshape(Va, 1, 1, Oa)
        shift_aaaa -= d_hp_a.reshape(1, Va, Oa, 1)
        shift_aaaa -= d_hp_a.reshape(1, Va, 1, Oa)
    custom_denom['aaaa'] = D_bare_aaaa - shift_aaaa

    # 2. bbbb sector
    D_bare_bbbb = (eb[:Ob].reshape(1, 1, Ob, 1) + eb[:Ob].reshape(1, 1, 1, Ob)
                   - eb[Ob:].reshape(Vb, 1, 1, 1) - eb[Ob:].reshape(1, Vb, 1, 1))
    shift_bbbb = np.zeros_like(D_bare_bbbb)
    if hh:
        d_h_b = _extract_d_h(g_bbbb_w, Ob)
        shift_bbbb += d_h_b.reshape(1, 1, Ob, Ob)
    if pp:
        d_p_b = _extract_d_p(g_bbbb_w, Ob)
        shift_bbbb += d_p_b.reshape(Vb, Vb, 1, 1)
    if hp:
        d_hp_b = _extract_d_hp(g_bbbb_w, Ob)
        shift_bbbb -= d_hp_b.reshape(Vb, 1, Ob, 1)
        shift_bbbb -= d_hp_b.reshape(Vb, 1, 1, Ob)
        shift_bbbb -= d_hp_b.reshape(1, Vb, Ob, 1)
        shift_bbbb -= d_hp_b.reshape(1, Vb, 1, Ob)
    custom_denom['bbbb'] = D_bare_bbbb - shift_bbbb

    # 3. abab sector
    D_bare_abab = (ea[:Oa].reshape(1, 1, Oa, 1) + eb[:Ob].reshape(1, 1, 1, Ob)
                   - ea[Oa:].reshape(Va, 1, 1, 1) - eb[Ob:].reshape(1, Vb, 1, 1))
    shift_abab = np.zeros_like(D_bare_abab)
    if hh:
        d_h_ab = np.einsum('ijij->ij', g_abab_w[:Oa, :Ob, :Oa, :Ob], optimize=True)
        shift_abab += d_h_ab.reshape(1, 1, Oa, Ob)
    if pp:
        d_p_ab = np.einsum('abab->ab', g_abab_w[Oa:, Ob:, Oa:, Ob:], optimize=True)
        shift_abab += d_p_ab.reshape(Va, Vb, 1, 1)
    if hp:
        d_hp_a = _extract_d_hp(g_aaaa_w, Oa)
        d_hp_b = _extract_d_hp(g_bbbb_w, Ob)
        d_hp_ab_ib = np.einsum('ibib->bi', g_abab_w[:Oa, Ob:, :Oa, Ob:], optimize=True)
        d_hp_ab_aj = np.einsum('ajaj->aj', g_abab_w[Oa:, :Ob, Oa:, :Ob], optimize=True)
        shift_abab -= d_hp_a.reshape(Va, 1, Oa, 1)
        shift_abab -= d_hp_b.reshape(1, Vb, 1, Ob)
        shift_abab -= d_hp_ab_ib.reshape(1, Vb, Oa, 1)
        shift_abab -= d_hp_ab_aj.reshape(Va, 1, 1, Ob)
    custom_denom['abab'] = D_bare_abab - shift_abab

    # 4/5. aa/bb sectors (singles): 'singles' was previously read nowhere in
    # this function -- the <ia||ia> shift was added UNCONDITIONALLY whenever
    # u2_denom_dress was not None, regardless of the hh/pp/hp/singles keys
    # actually passed. Now gated like every other channel (default True,
    # matching _build_dressed_e_ai's restricted-branch convention) so
    # singles=False genuinely turns it off instead of silently no-op'ing.
    singles = u2_denom_dress.get('singles', True)
    D_bare_aa = (ea[:Oa].reshape(1, Oa) - ea[Oa:].reshape(Va, 1))
    D_bare_bb = (eb[:Ob].reshape(1, Ob) - eb[Ob:].reshape(Vb, 1))
    if singles == 'screened':
        if mol is None or mf_eval is None:
            raise ValueError(
                "u2_denom_dress={'singles': 'screened', ...} needs mol= and "
                "mf_eval= (the raw per-spin chemist ERI and RPA screening are "
                "built from them) -- not passed to _build_dressed_denoms_uhf.")
        eri_aa, eri_ab, eri_bb = get_two_electron_integrals_chemist(
            mol, mf_eval, representation='spatial')
        eri_aa_w, eri_bb_w, eri_ab_w = eri_aa[w, w, w, w], eri_bb[w, w, w, w], eri_ab[w, w, w, w]
        W_a, W_b = static_screened_coulomb_chemist_uhf(
            ea, eb, eri_aa_w, eri_bb_w, eri_ab_w, Oa, Ob)
        d_hp_a = _extract_singles_shift_uhf(eri_aa_w, Oa, W=W_a)
        d_hp_b = _extract_singles_shift_uhf(eri_bb_w, Ob, W=W_b)
        custom_denom['aa'] = D_bare_aa + d_hp_a
        custom_denom['bb'] = D_bare_bb + d_hp_b
    elif singles:
        d_hp_a = _extract_d_hp(g_aaaa_w, Oa)
        d_hp_b = _extract_d_hp(g_bbbb_w, Ob)
        custom_denom['aa'] = D_bare_aa + d_hp_a
        custom_denom['bb'] = D_bare_bb + d_hp_b
    else:
        custom_denom['aa'] = D_bare_aa
        custom_denom['bb'] = D_bare_bb

    return custom_denom

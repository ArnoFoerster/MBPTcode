"""Density-fitted spin-free EE-ADC sigma operator -- the production path.

Same working equations as ee_r_sigma (they live once, in ee_equations); the
difference is that every integral is built from the DF factor

    (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s]     ->     <pq|rs> = sum_Q B[Q,p,r] B[Q,q,s]

and the FOUR-VIRTUAL block <ab||cd> is never formed. Only three contractions
in the ADC(3) hierarchy touch it (ee_equations.VvvvKernels): the particle
ladder, the virtual/virtual density term of M^(3), and the Z^(10) term of
M^(3). Each is Q-factorized here into two matrix multiplications with at most
four-index intermediates, so the peak memory is O(no^2 nv^2) and the ladder
costs O(naux no^2 nv^3) instead of O(no^2 nv^4).

The spin bookkeeping for the kernels follows one rule, read straight off
<p_s q_t || r_u s_v> = d(s,u) d(t,v) V[pqrs] - d(s,v) d(t,u) V[pqsr]:

    ladder:      R[ij,ab] = Ldir(X[ij,ab]) - Ldir(X[ij,ba] transposed in cd)
    vv_density:  R[aa] = sum_s Ddir(rho[ss]) - Dexch(rho[aa]),  spin-diagonal
    Z^(10):      R[ia,jb] = d(sa,sb) sum_s Zdir(Z[i s j s]) - Zexch(Z[ia,jb])

where the direct/exchange halves differ only in which B index pair is
contracted. tests/test_ee_adc_spinfree.py checks the whole operator against
ee_r_sigma with an EXACT (eigendecomposed) B, so any spin slip in these rules
shows up at machine precision rather than as a plausible small error.
"""
import numpy as np

from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC import ee_en as _en
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB, anti4

LEVELS = _r.LEVELS


# ----------------------------------------------------------------------
# integral blocks from B (no vvvv)
# ----------------------------------------------------------------------

def _phys(Bl, Br):
    """<pq|rs> = sum_Q B[Q,p,r] B[Q,q,s] for one (family) block pair."""
    return np.einsum('Qpr,Qqs->pqrs', Bl, Br, optimize=True)


def g_blocks_df(B, no, norb):
    """The five blocks with at most two virtual indices, spin-blocked.
    <ab||cd> is deliberately absent -- it goes through the kernels below."""
    o, v = slice(0, no), slice(no, norb)
    Boo, Bov, Bvo, Bvv = B[:, o, o], B[:, o, v], B[:, v, o], B[:, v, v]
    return {
        'oooo': anti4(_phys(Boo, Boo), _phys(Boo, Boo).transpose(0, 1, 3, 2)),
        'ooov': anti4(_phys(Boo, Bov), _phys(Bov, Boo).transpose(0, 1, 3, 2)),
        'oovv': anti4(_phys(Bov, Bov)),
        'ovov': anti4(_phys(Boo, Bvv), _phys(Bov, Bvo).transpose(0, 1, 3, 2)),
        'ovvv': anti4(_phys(Bov, Bvv)),
    }


# ----------------------------------------------------------------------
# vvvv kernels
# ----------------------------------------------------------------------

class DFVvvvKernels(_eq.VvvvKernels):
    """ee_equations.VvvvKernels evaluated from B, never forming <ab||cd>.

    The auxiliary sum is taken in CHUNKS rather than one Q at a time: each
    chunk is two large einsums instead of `naux` tiny ones, which is worth
    roughly an order of magnitude in wall time (a per-Q loop leaves numpy
    call overhead dominating whenever no^2 nv^3 is small). The chunk carries a
    bounded auxiliary batch axis on top of the four orbital indices; its size
    is chosen from `mem_gb`, so it is a memory knob, not a scaling axis.
    """

    def __init__(self, B, no, norb, mem_gb=2.0):
        self.B = B
        self.Bvv = np.ascontiguousarray(B[:, no:norb, no:norb])
        self.no, self.nv = no, norb - no
        self.mem_gb = mem_gb

    def _chunks(self, per_aux_elems):
        """Aux-function slices sized so one batch stays inside mem_gb."""
        naux = self.Bvv.shape[0]
        n = max(1, min(naux, int(self.mem_gb * 1.25e8 / max(per_aux_elems, 1))))
        return [slice(k, min(k + n, naux)) for k in range(0, naux, n)]

    # ---- particle ladder -------------------------------------------------
    def _ldir(self, Y):
        """sum_cd (sum_Q B[Q,a,c] B[Q,b,d]) Y[i,j,c,d]."""
        no2 = Y.shape[0] * Y.shape[1]
        out = np.zeros(Y.shape[:2] + (self.nv,) * 2)
        for sl in self._chunks(no2 * self.nv ** 2):
            Bc = self.Bvv[sl]
            U = np.einsum('Qac,ijcd->Qijad', Bc, Y, optimize=True)
            out += np.einsum('Qijad,Qbd->ijab', U, Bc, optimize=True)
        return out

    def ladder(self, X):
        """R[key] = Ldir(X[key]) - Ldir(X[key with a,b swapped] transposed).

        Ldir touches only the (c,d) -> (a,b) pair, so the two occupied indices
        are spectators and

            Ldir(Y transposed in its last two axes)
                == Ldir(Y) transposed in its last two axes.

        That lets the second term reuse the FIRST term's result for the
        swapped key instead of running the kernel again -- one Ldir per spin
        block rather than two. The ladder is ~70% of a DF sigma, so this is
        the single biggest lever in the operator."""
        # memoize on exact array identity (buffer, shape, strides): several
        # spin blocks are literally the SAME object or a view of one -- anti4
        # stores three distinct arrays under six keys -- so this collapses
        # six kernel runs to three or four with no assumption about how the
        # SB was built
        # Ldir maps (c,d) -> (a,b) with the occupied indices as spectators,
        # so it commutes with any permutation within {i,j} and within {a,b}:
        #     Ldir(sign * Y^P) == sign * Ldir(Y)^P.
        # When the caller declares which blocks are signed transposes of
        # which (ee_r_sigma.to_blocks does), the kernel runs on the base
        # blocks only -- three instead of six for a doubles vector.
        derived = X.derived or {}
        L = {}
        for key, blk in X.items():
            if key in derived and derived[key][0] in X.keys():
                continue
            L[key] = self._ldir(blk)
        for key, (base, perm, sign) in derived.items():
            if base in L:
                L[key] = sign * L[base].transpose(perm)
        out = {}
        for key in X.keys():
            val = L[key]
            sw = L.get(key[0] + key[1] + key[3] + key[2])
            if sw is not None:
                val = val - sw.transpose(0, 1, 3, 2)
            out[key] = val
        return SB(out)

    # ---- <ac||bd> rho_cd -------------------------------------------------
    def vv_density(self, rho_vv):
        direct = 0.0
        for s in ('aa', 'bb'):
            r = rho_vv.get(s)
            if r is None:
                continue
            direct = direct + np.einsum(
                'Qab,Q->ab', self.Bvv, np.einsum('Qcd,cd->Q', self.Bvv, r,
                                                 optimize=True), optimize=True)
        out = {}
        for s in ('aa', 'bb'):
            r = rho_vv.get(s)
            exch = 0.0
            if r is not None:
                # sum_Q B[Q,a,d] (sum_c B[Q,c,b] rho[c,d])
                W = np.einsum('Qcb,cd->Qbd', self.Bvv, r, optimize=True)
                exch = np.einsum('Qad,Qbd->ab', self.Bvv, W, optimize=True)
            out[s] = direct - exch
        return SB(out)

    # ---- <ad||bc> Z10[i,c,j,d] ------------------------------------------
    def z10(self, Z10):
        out = {}
        # direct half: <a_sa d_sd || b_sb c_sc> survives when sa == sb and
        # sd == sc, so the closed spin s is summed over for each (si, sj)
        for si in ('a', 'b'):
            for sj in ('a', 'b'):
                acc = None
                for s in ('a', 'b'):
                    blk = Z10.get(f'{si}{s}{sj}{s}')
                    if blk is None:
                        continue
                    term = np.einsum('Qdc,icjd->Qij', self.Bvv, blk,
                                     optimize=True)
                    acc = term if acc is None else acc + term
                if acc is None:
                    continue
                direct = np.einsum('Qab,Qij->iajb', self.Bvv, acc,
                                   optimize=True)
                for sa in ('a', 'b'):
                    key = f'{si}{sa}{sj}{sa}'
                    out[key] = out.get(key, 0.0) + direct
        # exchange half: sa == sc and sd == sb, i.e. the Z10 block that
        # already carries the output spins
        no2 = self.no ** 2
        for key, blk in Z10.items():
            exch = np.zeros(blk.shape)
            for sl in self._chunks(no2 * self.nv ** 2):
                Bc = self.Bvv[sl]
                T = np.einsum('Qac,icjd->Qaijd', Bc, blk, optimize=True)
                exch += np.einsum('Qaijd,Qdb->iajb', T, Bc, optimize=True)
            out[key] = out.get(key, 0.0) - exch
        return SB(out)


# ----------------------------------------------------------------------
# operator
# ----------------------------------------------------------------------

def build_operator(eps, B, nocc_spatial, level='adc3', en_dress=None,
                   cache=None):
    """(aop, diag, dims) from spatial orbital energies and the DF factor
    B (naux, norb, norb) with (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s].

    Same vector layout and spin-channel handling as ee_r_sigma."""
    if level not in LEVELS:
        raise ValueError(f'level={level!r}; expected one of {LEVELS}')
    no, norb = nocc_spatial, len(eps)
    nv = norb - no
    o_ss, o_sd, o_dd = _r._BLOCK_ORDERS[level]
    d = _r.dimensions(no, nv, level)

    gb, vk, cache = _r._ingredients(cache, lambda: g_blocks_df(B, no, norb),
                                    lambda: DFVvvvKernels(B, no, norb))
    eo, ev = eps[:no], eps[no:]
    d_ijab = (ev[None, None, :, None] + ev[None, None, None, :]
              - eo[:, None, None, None] - eo[None, :, None, None])
    d_ia = 2.0 * (eo[:, None] - ev[None, :])
    d_ph = np.broadcast_to((ev[None, :] - eo[:, None])[:, :, None, None],
                           (no, nv, no, nv))

    be = _eq.SPIN_BLOCKED
    order = {'adc1': 1, 'adc2': 2, 'adc2x': 2, 'adc3': 3}[level]
    if en_dress is None:
        d_amp, d_ia_amp, en_shift = d_ijab, d_ia, None
    else:
        d_amp, d_ia_amp = _en.en_denominators_spin_free(
            eps, *_en.jk_from_B(B), no, nv, en_dress)
        # what the EN denominator has resummed, so amplitudes() can
        # take it back out of the t2^(2) numerator and not count the
        # diagonal ladder twice (see ee_equations.amplitudes)
        en_shift = _en.shift_from_denominators(d_amp, d_ijab)
    amps, zint, rho = _r._amplitudes_cached(cache, be, gb, d_amp, d_ia_amp,
                                            order, vk, en_shift)
    M = _eq.m_ss(be, gb, o_ss, amps, zint, rho, d_ph, no, nv, vk=vk)
    diag = _diagonal_df(eps, B, no, nv, M, o_dd)

    def aop(vec):
        vec = np.asarray(vec).ravel()
        y1, Y = _r.to_blocks(vec, no, nv, level)
        w1 = be.ein('iajb,jb->ia', M, y1)
        if o_sd is None:
            return _r.from_blocks(w1, SB(), no, nv, level)
        w1 = w1 + _eq.sigma_s_from_d(be, gb, amps, zint, Y, o_sd)
        W = (_eq.sigma_d_from_s(be, gb, amps, zint, y1, o_sd)
             + _eq.sigma_d_from_d(be, gb, Y, d_ijab, o_dd, vk=vk))
        return _r.from_blocks(w1, W, no, nv, level)

    return aop, diag, d


def _diagonal_df(eps, B, no, nv, M, o_dd):
    """ee_r_sigma's diagonal with the Coulomb/exchange arrays built from B:
    J[p,q] = <pq|pq> = (pp|qq),  K[p,q] = <pq|qp> = (pq|qp)."""
    J = np.einsum('Qpp,Qqq->pq', B, B, optimize=True)
    K = np.einsum('Qpq,Qpq->pq', B, B, optimize=True)
    return _r.diagonal_from_JK(eps, J, K, no, nv, M, o_dd)

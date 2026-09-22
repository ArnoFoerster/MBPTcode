"""Unrestricted density-fitted EE-ADC sigma -- the open-shell production path.

The spin-free route (ee_r_sigma_df) and this one run the SAME working
equations: ee_equations is spin-orbital algebra, and ee_spin_blocks.sb_einsum
evaluates it by enumerating the spin assignment of every index. Nothing in
either module assumes a closed shell. What a closed shell buys is that the
alpha and beta spatial orbitals coincide, which collapses

    <pq||rs>_aaaa == <pq||rs>_bbbb        <pq||rs>_abab == <pq||rs>_baba

and lets one B serve both spins. An unrestricted reference loses exactly
that, and nothing else -- so this module rebuilds the integral blocks, the
vvvv kernels and the denominators spin-resolved, and reuses the equations
untouched.

Three things carry over unchanged, which is why this is a small module and
not a second implementation:

* **The vector layout is already unrestricted-shaped.** ee_r_sigma keeps
  y_aa and y_bb apart and stores doubles as aaaa | bbbb | abab; only the
  segment SIZES differ here (no_a != no_b, nv_a != nv_b).
* **The derived-block identities survive.** `baba`/`abba`/`baab` are signed
  transposes of `abab` by ANTISYMMETRY of the configuration coefficient, not
  by any orbital-basis coincidence, so the ladder's provenance shortcut still
  applies.
* **The Delta-Ms = 0 sector is implicit.** Spin-flip singles (i_a -> a_b)
  have no slot in the vector, so every block sb_einsum would build for them
  is dropped for want of an operand -- the sector is enforced by the layout
  rather than by a mask.

Vector layout, flat:

    y_a (no_a*nv_a) | y_b (no_b*nv_b)
    | D_aaaa (i<j, a<b in alpha) | D_bbbb (i<j, a<b in beta)
    | D_abab (no_a*no_b*nv_a*nv_b)

Oracle: the dense spin-orbital route (ee_u_dense_full / ee_u_sigma_full) fed
the same UHF reference, which is what tests/test_ee_adc_uhf_df.py checks --
the same arrangement that validated the closed-shell DF path.
"""
import numpy as np

from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC import ee_en as _en
from src.SingleReference.ADC.eeADC.ee_equations import _Backend
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB, anti4_uhf

LEVELS = _r.LEVELS
SCALE = _r.SCALE

_SPINS = ('a', 'b')


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------

def dimensions(no_a, no_b, nv_a, nv_b, level='adc3'):
    """Segment sizes. ADC(1) has no 2p2h manifold."""
    n_sa, n_sb = no_a * nv_a, no_b * nv_b
    doubles = level != 'adc1'
    n_da = (no_a * (no_a - 1) // 2) * (nv_a * (nv_a - 1) // 2) if doubles else 0
    n_db = (no_b * (no_b - 1) // 2) * (nv_b * (nv_b - 1) // 2) if doubles else 0
    n_mx = no_a * no_b * nv_a * nv_b if doubles else 0
    return {'no_a': no_a, 'no_b': no_b, 'nv_a': nv_a, 'nv_b': nv_b,
            'n_sa': n_sa, 'n_sb': n_sb, 'n_da': n_da, 'n_db': n_db,
            'n_mixed': n_mx, 'doubles': doubles,
            'nH': n_sa + n_sb + n_da + n_db + n_mx}


def _slices(d):
    sizes = [d['n_sa'], d['n_sb'], d['n_da'], d['n_db'], d['n_mixed']]
    out, off = [], 0
    for n in sizes:
        out.append(slice(off, off + n))
        off += n
    return out


def to_blocks(v, d, level='adc3'):
    """flat vector -> (singles SB, doubles SB) in the paper's tensor scaling."""
    no_a, no_b, nv_a, nv_b = d['no_a'], d['no_b'], d['nv_a'], d['nv_b']
    sa, sb, da, db, dm = _slices(d)
    y1 = SB({'aa': v[sa].reshape(no_a, nv_a),
             'bb': v[sb].reshape(no_b, nv_b)})
    if not d['doubles']:
        return y1, SB()
    Yaa = _r._expand_same(v[da], no_a, nv_a) / SCALE
    Ybb = _r._expand_same(v[db], no_b, nv_b) / SCALE
    Yab = v[dm].reshape(no_a, no_b, nv_a, nv_b) / SCALE
    Yneg = -Yab                     # one array, three views (see ee_r_sigma)
    Y = SB({'aaaa': Yaa, 'bbbb': Ybb, 'abab': Yab,
            'baba': Yab.transpose(1, 0, 3, 2),
            'abba': Yneg.transpose(0, 1, 3, 2),
            'baab': Yneg.transpose(1, 0, 2, 3)},
           derived={'baba': ('abab', (1, 0, 3, 2), +1.0),
                    'abba': ('abab', (0, 1, 3, 2), -1.0),
                    'baab': ('abab', (1, 0, 2, 3), -1.0)})
    return y1, Y


def from_blocks(w1, W, d, level='adc3'):
    no_a, no_b, nv_a, nv_b = d['no_a'], d['no_b'], d['nv_a'], d['nv_b']
    za, zb = np.zeros((no_a, nv_a)), np.zeros((no_b, nv_b))
    out = [(w1.get('aa') if w1.get('aa') is not None else za).ravel(),
           (w1.get('bb') if w1.get('bb') is not None else zb).ravel()]
    if not d['doubles']:
        return np.concatenate(out)
    for key, no, nv in (('aaaa', no_a, nv_a), ('bbbb', no_b, nv_b)):
        T = W.get(key)
        T = T if T is not None else np.zeros((no, no, nv, nv))
        out.append(SCALE * _r._restrict_same(T, no, nv))
    T = W.get('abab')
    T = T if T is not None else np.zeros((no_a, no_b, nv_a, nv_b))
    out.append(SCALE * T.ravel())
    return np.concatenate(out)


# ----------------------------------------------------------------------
# integral blocks from B_a, B_b (no vvvv)
# ----------------------------------------------------------------------

def _phys(Bl, Br):
    """<pq|rs> = sum_Q B[Q,p,r] B[Q,q,s]."""
    return np.einsum('Qpr,Qqs->pqrs', Bl, Br, optimize=True)


def g_blocks_df_uhf(Ba, Bb, no_a, no_b, norb_a, norb_b):
    """The five blocks with at most two virtual indices, spin-resolved.
    <ab||cd> is deliberately absent -- it goes through the kernels below."""
    sl = {'a': {'o': slice(0, no_a), 'v': slice(no_a, norb_a)},
          'b': {'o': slice(0, no_b), 'v': slice(no_b, norb_b)}}
    Bs = {'a': Ba, 'b': Bb}

    def blk(spin, f1, f2):
        return Bs[spin][:, sl[spin][f1], sl[spin][f2]]

    out = {}
    for fam in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvv'):
        f0, f1, f2, f3 = fam
        # <pq|rs>: p,r on one electron, q,s on the other.
        # <pq|sr>: the exchange partner, transposed back to the pqrs order.
        args = []
        for s, t in (('a', 'a'), ('b', 'b'), ('a', 'b'), ('b', 'a')):
            direct = _phys(blk(s, f0, f2), blk(t, f1, f3))
            exch = _phys(blk(s, f0, f3),
                         blk(t, f1, f2)).transpose(0, 1, 3, 2)
            args += [direct, exch]
        out[fam] = anti4_uhf(*args)
    return out


# ----------------------------------------------------------------------
# vvvv kernels
# ----------------------------------------------------------------------

class UDFVvvvKernels(_eq.VvvvKernels):
    """ee_equations.VvvvKernels from (B_a, B_b), never forming <ab||cd>.

    Identical Q-factorization to DFVvvvKernels; the only change is that every
    B index now carries a spin, chosen from the spin string of the block being
    contracted. The rule is read straight off

        <p_s q_t || r_u s_v> = d(s,u) d(t,v) (pr|qs) - d(s,v) d(t,u) (ps|qr)

    with (pr|qs) = sum_Q B_s[Q,p,r] B_t[Q,q,s]."""

    def __init__(self, Ba, Bb, no_a, no_b, norb_a, norb_b, mem_gb=2.0):
        self.Bvv = {'a': np.ascontiguousarray(Ba[:, no_a:norb_a, no_a:norb_a]),
                    'b': np.ascontiguousarray(Bb[:, no_b:norb_b, no_b:norb_b])}
        self.no = {'a': no_a, 'b': no_b}
        self.nv = {'a': norb_a - no_a, 'b': norb_b - no_b}
        self.mem_gb = mem_gb

    def _chunks(self, per_aux_elems):
        naux = self.Bvv['a'].shape[0]
        n = max(1, min(naux, int(self.mem_gb * 1.25e8 / max(per_aux_elems, 1))))
        return [slice(k, min(k + n, naux)) for k in range(0, naux, n)]

    # ---- particle ladder -------------------------------------------------
    def _ldir(self, Y, sa, sb):
        """sum_cd (sum_Q B_sa[Q,a,c] B_sb[Q,b,d]) Y[i,j,c,d], c of spin sa
        and d of spin sb."""
        no2 = Y.shape[0] * Y.shape[1]
        out = np.zeros(Y.shape[:2] + (self.nv[sa], self.nv[sb]))
        Ba, Bb = self.Bvv[sa], self.Bvv[sb]
        for sl in self._chunks(no2 * self.nv[sa] * self.nv[sb]):
            Bc, Bd = Ba[sl], Bb[sl]
            U = np.einsum('Qac,ijcd->Qijad', Bc, Y, optimize=True)
            out += np.einsum('Qijad,Qbd->ijab', U, Bd, optimize=True)
        return out

    def ladder(self, X):
        """R[key] = Ldir(X[key]) - Ldir(X[key, a<->b]) transposed in (a,b).

        Ldir maps (c,d) -> (a,b) with the occupied indices as spectators, so
        it commutes with any permutation within {i,j} and within {a,b} -- and
        for an unrestricted reference that permutation carries the SPIN
        labels along, which is consistent because the spins are taken from
        the key each time:

            Ldir(Y^(1,0,3,2), s_b, s_a) == Ldir(Y, s_a, s_b)^(1,0,3,2)

        So the caller's provenance map still lets the kernel run on the base
        blocks only -- three Ldir calls for a doubles vector instead of six.
        The ladder is the dominant cost of a sigma, so this matters as much
        here as it does on the closed-shell path."""
        derived = X.derived or {}
        L = {}
        for key, blk in X.items():
            if key in derived and derived[key][0] in X.keys():
                continue
            L[key] = self._ldir(blk, key[2], key[3])
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
        """R[a_s,b_s] = sum_t Ddir_t(rho[tt]) - Dexch(rho[ss]).

        The direct half sums the closed pair over BOTH spins, but unlike the
        closed-shell case its B for the (a,b) pair depends on the output
        spin, so it is rebuilt per output block rather than shared."""
        aux = {}
        for t in _SPINS:
            r = rho_vv.get(t + t)
            if r is not None:
                aux[t] = np.einsum('Qcd,cd->Q', self.Bvv[t], r, optimize=True)
        total = sum(aux.values()) if aux else None
        out = {}
        for s in _SPINS:
            B = self.Bvv[s]
            val = 0.0
            if total is not None:
                val = np.einsum('Qab,Q->ab', B, total, optimize=True)
            r = rho_vv.get(s + s)
            if r is not None:
                # sum_Q B[Q,a,d] (sum_c B[Q,c,b] rho[c,d])
                W = np.einsum('Qcb,cd->Qbd', B, r, optimize=True)
                val = val - np.einsum('Qad,Qbd->ab', B, W, optimize=True)
            out[s + s] = val
        return SB(out)

    # ---- <ad||bc> Z10[i,c,j,d] ------------------------------------------
    def z10(self, Z10):
        out = {}
        # direct half: <a_sa d_sd || b_sb c_sc> survives when sa == sb and
        # sd == sc, so the closed spin is summed for each (si, sj)
        for si in _SPINS:
            for sj in _SPINS:
                acc = {}
                for s in _SPINS:
                    blk = Z10.get(f'{si}{s}{sj}{s}')
                    if blk is None:
                        continue
                    acc[s] = np.einsum('Qdc,icjd->Qij', self.Bvv[s], blk,
                                       optimize=True)
                if not acc:
                    continue
                total = sum(acc.values())
                for sa in _SPINS:
                    key = f'{si}{sa}{sj}{sa}'
                    direct = np.einsum('Qab,Qij->iajb', self.Bvv[sa], total,
                                       optimize=True)
                    out[key] = out.get(key, 0.0) + direct
        # exchange half: sa == sc and sd == sb, i.e. the Z10 block that
        # already carries the output spins
        for key, blk in Z10.items():
            si, sa, sj, sb = key
            exch = np.zeros(blk.shape)
            n = blk.shape[0] * blk.shape[2]
            for sl in self._chunks(n * self.nv[sa] * self.nv[sb]):
                Bc, Bd = self.Bvv[sa][sl], self.Bvv[sb][sl]
                T = np.einsum('Qac,icjd->Qaijd', Bc, blk, optimize=True)
                exch = exch + np.einsum('Qaijd,Qdb->iajb', T, Bd,
                                        optimize=True)
            out[key] = out.get(key, 0.0) - exch
        return SB(out)


# ----------------------------------------------------------------------
# backend and denominators
# ----------------------------------------------------------------------

def _backend():
    """SPIN_BLOCKED with a spin-resolved identity.

    ee_equations touches `nocc`/`nvirt` in exactly one place, `be.eye`, so
    passing the sizes as an (n_alpha, n_beta) pair is all it takes to make
    m_ss unrestricted -- the equations themselves need no edit."""
    base = _eq.SPIN_BLOCKED
    return _Backend(ein=base.ein, scale=base.scale, divide=base.divide,
                    eye=lambda n: SB({'aa': np.eye(n[0]), 'bb': np.eye(n[1])}))


def _denominators(eps_a, eps_b, no_a, no_b):
    """(d_ijab, d_ia, d_ph) as spin-resolved SBs.

    d_ijab carries ALL SIX keys a doubles quantity can have: SB.divide keeps
    only keys the weight also has, so a missing key would silently drop that
    block of the amplitudes rather than fail."""
    eo = {'a': eps_a[:no_a], 'b': eps_b[:no_b]}
    ev = {'a': eps_a[no_a:], 'b': eps_b[no_b:]}
    d_ijab = {}
    for key in ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab'):
        si, sj, sa, sb = key
        d_ijab[key] = (ev[sa][None, None, :, None]
                       + ev[sb][None, None, None, :]
                       - eo[si][:, None, None, None]
                       - eo[sj][None, :, None, None])
    d_ia = SB({s + s: 2.0 * (eo[s][:, None] - ev[s][None, :]) for s in _SPINS})
    d_ph = SB({s + s + s + s: np.broadcast_to(
        (ev[s][None, :] - eo[s][:, None])[:, :, None, None],
        (len(eo[s]), len(ev[s]), len(eo[s]), len(ev[s]))) for s in _SPINS})
    return SB(d_ijab), d_ia, d_ph


def _jk_uhf(Ba, Bb):
    """J[s][t][p,q] = (p_s p_s | q_t q_t) and K[s][p,q] = (p_s q_s | q_s p_s).

    Exchange only exists within a spin, which is why K has one index and J
    has two."""
    B = {'a': Ba, 'b': Bb}
    diag = {s: np.einsum('Qpp->Qp', B[s]) for s in _SPINS}
    J = {s: {t: np.einsum('Qp,Qq->pq', diag[s], diag[t], optimize=True)
             for t in _SPINS} for s in _SPINS}
    K = {s: np.einsum('Qpq,Qpq->pq', B[s], B[s], optimize=True)
         for s in _SPINS}
    return J, K


def _diagonal(eps_a, eps_b, d, M, o_dd, Ba, Bb):
    """Exact operator diagonal per spin block, for the Davidson
    preconditioner -- ee_r_sigma.diagonal_from_JK with the spin pairs made
    explicit instead of implied."""
    parts = [np.einsum('iaia->ia', M.get('aaaa'), optimize=True).ravel(),
             np.einsum('iaia->ia', M.get('bbbb'), optimize=True).ravel()]
    if o_dd is None:
        return np.concatenate(parts)
    J, K = _jk_uhf(Ba, Bb)
    no = {'a': d['no_a'], 'b': d['no_b']}
    eps = {'a': eps_a, 'b': eps_b}
    eo = {s: eps[s][:no[s]] for s in _SPINS}
    ev = {s: eps[s][no[s]:] for s in _SPINS}

    def part(s, t, f1, f2):
        """J/K block for orbital families f1 (spin s) and f2 (spin t), with
        the exchange kept only when the two spins agree."""
        s1 = slice(0, no[s]) if f1 == 'o' else slice(no[s], len(eps[s]))
        s2 = slice(0, no[t]) if f2 == 'o' else slice(no[t], len(eps[t]))
        out = J[s][t][s1, s2]
        if s == t:
            out = out - K[s][s1, s2]
        return out

    def block(si, sj, sa, sb):
        D = (ev[sa][None, None, :, None] + ev[sb][None, None, None, :]
             - eo[si][:, None, None, None] - eo[sj][None, :, None, None])
        if o_dd >= 1:
            D = D + part(sa, sb, 'v', 'v')[None, None, :, :]
            D = D + part(si, sj, 'o', 'o')[:, :, None, None]
            D = D - part(si, sa, 'o', 'v')[:, None, :, None]
            D = D - part(sj, sb, 'o', 'v')[None, :, None, :]
            D = D - part(sj, sa, 'o', 'v')[None, :, :, None]
            D = D - part(si, sb, 'o', 'v')[:, None, None, :]
        return D

    parts.append(_r._restrict_same(block('a', 'a', 'a', 'a'),
                                   d['no_a'], d['nv_a']))
    parts.append(_r._restrict_same(block('b', 'b', 'b', 'b'),
                                   d['no_b'], d['nv_b']))
    parts.append(block('a', 'b', 'a', 'b').ravel())
    return np.concatenate(parts)


# ----------------------------------------------------------------------
# operator
# ----------------------------------------------------------------------

def build_operator(eps_a, eps_b, Ba, Bb, no_a, no_b, level='adc3',
                   en_dress=None, cache=None):
    """(aop, diag, dims) from UHF spatial orbital energies and the two DF
    factors, with (p_s q_s|r_t s_t) = sum_Q B_s[Q,p,q] B_t[Q,r,s]."""
    if level not in LEVELS:
        raise ValueError(f'level={level!r}; expected one of {LEVELS}')
    norb_a, norb_b = len(eps_a), len(eps_b)
    nv_a, nv_b = norb_a - no_a, norb_b - no_b
    o_ss, o_sd, o_dd = _r._BLOCK_ORDERS[level]
    d = dimensions(no_a, no_b, nv_a, nv_b, level)

    gb, vk, cache = _r._ingredients(
        cache,
        lambda: g_blocks_df_uhf(Ba, Bb, no_a, no_b, norb_a, norb_b),
        lambda: UDFVvvvKernels(Ba, Bb, no_a, no_b, norb_a, norb_b))
    d_ijab, d_ia, d_ph = _denominators(eps_a, eps_b, no_a, no_b)
    _eq._check_denominator(d_ijab)

    if en_dress is None:
        d_amp, d_ia_amp, en_shift = d_ijab, d_ia, None
    else:
        J, K = _en.jk_from_B_uhf(Ba, Bb)
        d_amp, d_ia_amp = _en.en_denominators_unrestricted(
            eps_a, eps_b, J, K, no_a, no_b, en_dress)
        # what the EN denominator has resummed, so amplitudes() can take it
        # back out of the t2^(2) numerator instead of double-counting the
        # diagonal ladder
        en_shift = _en.shift_from_denominators(d_amp, d_ijab)

    be = _backend()
    order = {'adc1': 1, 'adc2': 2, 'adc2x': 2, 'adc3': 3}[level]
    amps, zint, rho = _r._amplitudes_cached(cache, be, gb, d_amp, d_ia_amp,
                                            order, vk, en_shift)
    M = _eq.m_ss(be, gb, o_ss, amps, zint, rho, d_ph,
                 (no_a, no_b), (nv_a, nv_b), vk=vk)
    diag = _diagonal(eps_a, eps_b, d, M, o_dd, Ba, Bb)

    def aop(vec):
        vec = np.asarray(vec).ravel()
        y1, Y = to_blocks(vec, d, level)
        w1 = be.ein('iajb,jb->ia', M, y1)
        if o_sd is None:
            return from_blocks(w1, SB(), d, level)
        w1 = w1 + _eq.sigma_s_from_d(be, gb, amps, zint, Y, o_sd)
        W = (_eq.sigma_d_from_s(be, gb, amps, zint, y1, o_sd)
             + _eq.sigma_d_from_d(be, gb, Y, d_ijab, o_dd, vk=vk))
        return from_blocks(w1, W, d, level)

    return aop, diag, d

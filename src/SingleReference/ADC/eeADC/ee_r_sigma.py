"""Spin-free (spatial-orbital) matrix-free EE-ADC sigma operator.

Every tensor here is SPATIAL: the working equations are the ones in
ee_equations, evaluated through the SPIN_BLOCKED backend, so a quantity that
costs (2n)^4 in the spin-orbital route costs a few n^4 blocks here. No
spin-summed equation is written by hand -- see ee_spin_blocks.

Vector layout (Ms = 0, the sector electronic excitations live in), flat:

    y_aa (no*nv) | y_bb (no*nv)
    | D_aaaa (i<j, a<b) | D_bbbb (i<j, a<b) | D_abab (no*no*nv*nv)

The doubles entries are ORTHONORMAL configuration coefficients, i.e. the
paper's tensor times ee_utils.PAPER_DOUBLES_SCALE, so the operator is
symmetric and the Davidson metric is the plain Euclidean one. (For the abab
block that identification is up to a per-configuration sign from ordering
the mixed-spin index pair -- a diagonal orthogonal transformation, which
changes neither symmetry nor spectrum.)

The alpha/beta blocks are kept separately rather than folded with a
closed-shell relation, so both spin channels come out of one operator; the
spin-flip involution `spin_flip_vector` splits them (singlet = +1, triplet =
-1) and can be used to halve the Davidson space.
"""
import numpy as np

from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC import ee_en as _en
from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

LEVELS = ('adc1', 'adc2', 'adc2x', 'adc3')
_BLOCK_ORDERS = {'adc1': (1, None, None), 'adc2': (2, 1, 0),
                 'adc2x': (2, 1, 1), 'adc3': (3, 2, 1)}
SCALE = 2.0                                    # ee_utils.PAPER_DOUBLES_SCALE


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------

def dimensions(no, nv, level='adc3'):
    """Segment sizes. ADC(1) has no 2p2h manifold, so its vector is the two
    singles blocks alone."""
    npo, npv = no * (no - 1) // 2, nv * (nv - 1) // 2
    n_s = no * nv
    doubles = level != 'adc1'
    n_same = npo * npv if doubles else 0
    n_mixed = no * no * nv * nv if doubles else 0
    return {'no': no, 'nv': nv, 'n_s': n_s, 'n_same': n_same,
            'n_mixed': n_mixed, 'doubles': doubles,
            'nH': 2 * n_s + 2 * n_same + n_mixed}


def _slices(d):
    n_s, n_sm, n_mx = d['n_s'], d['n_same'], d['n_mixed']
    o = [0, n_s, 2 * n_s, 2 * n_s + n_sm, 2 * n_s + 2 * n_sm,
         2 * n_s + 2 * n_sm + n_mx]
    return [slice(o[k], o[k + 1]) for k in range(5)]


def _expand_same(v, no, nv):
    """unique (i<j, a<b) coefficients -> full antisymmetric spatial tensor."""
    iu, ju = np.triu_indices(no, k=1)
    au, bu = np.triu_indices(nv, k=1)
    T = np.zeros((no, no, nv, nv))
    blk = v.reshape(len(iu), len(au))
    T[iu[:, None], ju[:, None], au[None, :], bu[None, :]] = blk
    T[ju[:, None], iu[:, None], au[None, :], bu[None, :]] = -blk
    T[iu[:, None], ju[:, None], bu[None, :], au[None, :]] = -blk
    T[ju[:, None], iu[:, None], bu[None, :], au[None, :]] = blk
    return T


def _restrict_same(T, no, nv):
    iu, ju = np.triu_indices(no, k=1)
    au, bu = np.triu_indices(nv, k=1)
    return T[iu[:, None], ju[:, None], au[None, :], bu[None, :]].ravel()


def to_blocks(v, no, nv, level='adc3'):
    """flat vector -> (singles SB, doubles SB) in the paper's tensor scaling."""
    d = dimensions(no, nv, level)
    sa, sb_, da, db, dm = _slices(d)
    y1 = SB({'aa': v[sa].reshape(no, nv), 'bb': v[sb_].reshape(no, nv)})
    if not d['doubles']:
        return y1, SB()
    Yaa = _expand_same(v[da], no, nv) / SCALE
    Ybb = _expand_same(v[db], no, nv) / SCALE
    Yab = v[dm].reshape(no, no, nv, nv) / SCALE
    # one negated array, then VIEWS of it: writing -Yab.transpose(...) twice
    # allocates two copies and, more importantly, hides from the DF ladder
    # that the two blocks share data
    Yneg = -Yab
    Y = SB({'aaaa': Yaa, 'bbbb': Ybb, 'abab': Yab,
            'baba': Yab.transpose(1, 0, 3, 2),
            'abba': Yneg.transpose(0, 1, 3, 2),
            'baab': Yneg.transpose(1, 0, 2, 3)},
           derived={'baba': ('abab', (1, 0, 3, 2), +1.0),
                    'abba': ('abab', (0, 1, 3, 2), -1.0),
                    'baab': ('abab', (1, 0, 2, 3), -1.0)})
    return y1, Y


def from_blocks(w1, W, no, nv, level='adc3'):
    d = dimensions(no, nv, level)
    z = np.zeros((no, nv))
    zz = np.zeros((no, no, nv, nv))
    parts = [w1.get('aa') if w1.get('aa') is not None else z,
             w1.get('bb') if w1.get('bb') is not None else z]
    out = [p.ravel() for p in parts]
    if not d['doubles']:
        return np.concatenate(out)
    for key in ('aaaa', 'bbbb'):
        T = W.get(key)
        out.append(SCALE * _restrict_same(T if T is not None else zz, no, nv))
    T = W.get('abab')
    out.append(SCALE * (T if T is not None else zz).ravel())
    return np.concatenate(out)


def spin_flip_vector(v, no, nv, level='adc3'):
    """The closed-shell involution alpha <-> beta acting on a flat vector.
    Its +1 / -1 eigenspaces are the singlet and (Ms=0) triplet channels."""
    d = dimensions(no, nv, level)
    sa, sb_, da, db, dm = _slices(d)
    out = np.empty_like(v)
    out[sa], out[sb_] = v[sb_], v[sa]
    if not d['doubles']:
        return out
    out[da], out[db] = v[db], v[da]
    Yab = v[dm].reshape(no, no, nv, nv)
    out[dm] = Yab.transpose(1, 0, 3, 2).ravel()
    return out


# ----------------------------------------------------------------------
# operator
# ----------------------------------------------------------------------

def new_cache():
    """Per-(molecule, basis, en_dress) scratch shared across levels and spin
    channels.

    Everything in it is level-independent: the integral blocks, the vvvv
    kernels, and the MP/EN amplitudes with their Z intermediates and density.
    Building them once instead of once per (level, spin) is worth a factor of
    ~6 on a benchmark sweep, and the third-order pieces are the expensive
    part. Pass the SAME dict to every build_operator call for one molecule;
    pass a fresh one when the integrals or the dressing change."""
    return {}


def _ingredients(cache, make_gb, make_vk):
    cache = new_cache() if cache is None else cache
    if 'gb' not in cache:
        cache['gb'] = make_gb()
        cache['vk'] = make_vk()
    return cache['gb'], cache['vk'], cache

def build_operator(eps, V, nocc_spatial, level='adc3', en_dress=None,
                   cache=None):
    """(aop, diag, dims) from SPATIAL orbital energies `eps` and the spatial
    physicist tensor V[p,q,r,s] = <pq|rs> (= eri_chemist.transpose(0,2,1,3)).

    en_dress: optional Epstein-Nesbet channel dict (ee_en) dressing the
    AMPLITUDE denominators only; the supermatrix keeps its MP zeroth order."""
    if level not in LEVELS:
        raise ValueError(f'level={level!r}; expected one of {LEVELS}')
    no, norb = nocc_spatial, len(eps)
    nv = norb - no
    o_ss, o_sd, o_dd = _BLOCK_ORDERS[level]
    d = dimensions(no, nv, level)

    gb, vk, cache = _ingredients(cache, lambda: _sb.g_blocks_sb(V, no, norb),
                                 lambda: None)
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
            eps, *_en.jk_from_V(V), no, nv, en_dress)
        # what the EN denominator has resummed, so amplitudes() can
        # take it back out of the t2^(2) numerator and not count the
        # diagonal ladder twice (see ee_equations.amplitudes)
        en_shift = _en.shift_from_denominators(d_amp, d_ijab)
    amps, zint, rho = _amplitudes_cached(cache, be, gb, d_amp, d_ia_amp,
                                         order, vk, en_shift)
    M = _eq.m_ss(be, gb, o_ss, amps, zint, rho, d_ph, no, nv)
    # NOT spin-diagonal: the aabb block of M_ia,jb is the exchange term that
    # splits singlet from triplet (M_singlet = M_aaaa + M_aabb, M_triplet =
    # M_aaaa - M_aabb, already at ADC(1)/CIS). Contract through sb_einsum so
    # every spin block is picked up.
    diag = _diagonal(eps, V, no, nv, M, o_dd)

    def aop(vec):
        vec = np.asarray(vec).ravel()
        y1, Y = to_blocks(vec, no, nv, level)
        w1 = be.ein('iajb,jb->ia', M, y1)
        if o_sd is None:
            return from_blocks(w1, SB(), no, nv, level)
        w1 = w1 + _eq.sigma_s_from_d(be, gb, amps, zint, Y, o_sd)
        W = (_eq.sigma_d_from_s(be, gb, amps, zint, y1, o_sd)
             + _eq.sigma_d_from_d(be, gb, Y, d_ijab, o_dd))
        return from_blocks(w1, W, no, nv, level)

    return aop, diag, d


def _amplitudes_cached(cache, be, gb, d_amp, d_ia_amp, order, vk,
                       en_shift=None):
    """Amplitudes/Z/density at the REQUESTED order, cached and upgraded.

    Order 3 is a superset of order 2, so a cell running ADC(2), ADC(2)-x and
    ADC(3) still pays for the third-order ingredients once. But it must not
    pay for them when nothing asks: ADC(2) uses t^(1) alone, and computing at
    order 3 regardless pulls in t^(2) -- whose numerator contains the
    particle ladder -- plus the whole Z set, three of which (Z1, Z6, Z7) are
    ovvv-sized. That is the dominant cost of the run, incurred for terms
    ADC(2) never contracts. An ADC(2)-only sweep of the database is a factor
    of several cheaper for this one line.

    The cache records the order it holds and recomputes only to go UP."""
    if order < 2:
        return None, None, None
    if cache.get('order', 0) < order:
        cache['amps'] = _eq.amplitudes(be, gb, d_amp, d_ia_amp, order, vk=vk,
                                       en_shift=en_shift)
        cache['rho'] = _eq.density2(be, cache['amps'])
        # Z is third-order content only, and the expensive part of it
        cache['zint'] = (_eq.z_intermediates(be, gb, cache['amps'], vk=vk)
                         if order >= 3 else None)
        cache['order'] = order
    return cache['amps'], cache['zint'], cache['rho']


def _diagonal(eps, V, no, nv, M, o_dd):
    """Exact operator diagonal, per spin block, for the preconditioner.

    Doubles: eps_a + eps_b - eps_i - eps_j + <ab||ab> + <ij||ij>
             - <ia||ia> - <jb||jb> - <ja||ja> - <ib||ib>,
    with <pq||pq> = J[p,q] - delta(spin_p, spin_q) K[p,q]."""
    J = np.einsum('pqpq->pq', V, optimize=True)
    K = np.einsum('pqqp->pq', V, optimize=True)
    return diagonal_from_JK(eps, J, K, no, nv, M, o_dd)


def diagonal_from_JK(eps, J, K, no, nv, M, o_dd):
    """The diagonal from the Coulomb/exchange arrays J[p,q] = <pq|pq> and
    K[p,q] = <pq|qp>, so the DF route can supply them without forming V."""
    parts = [np.einsum('iaia->ia', M.get(k), optimize=True).ravel()
             for k in ('aaaa', 'bbbb')]
    if o_dd is None:
        return np.concatenate(parts)
    eo, ev = eps[:no], eps[no:]
    o, v = slice(0, no), slice(no, no + nv)
    base = (ev[None, None, :, None] + ev[None, None, None, :]
            - eo[:, None, None, None] - eo[None, :, None, None])

    def block(si, sj, sa, sb):
        D = base.copy()
        if o_dd >= 1:
            J_vv, K_vv = J[v, v], K[v, v]
            J_oo, K_oo = J[o, o], K[o, o]
            J_ov, K_ov = J[o, v], K[o, v]
            D = D + (J_vv - (sa == sb) * K_vv)[None, None, :, :]
            D = D + (J_oo - (si == sj) * K_oo)[:, :, None, None]
            D = D - (J_ov - (si == sa) * K_ov)[:, None, :, None]
            D = D - (J_ov - (sj == sb) * K_ov)[None, :, None, :]
            D = D - (J_ov - (sj == sa) * K_ov)[None, :, :, None]
            D = D - (J_ov - (si == sb) * K_ov)[:, None, None, :]
        return D

    parts.append(_restrict_same(block(0, 0, 0, 0), no, nv))
    parts.append(_restrict_same(block(1, 1, 1, 1), no, nv))
    parts.append(block(0, 1, 0, 1).ravel())
    return np.concatenate(parts)

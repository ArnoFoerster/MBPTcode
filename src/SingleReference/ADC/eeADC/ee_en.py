"""Epstein-Nesbet-dressed amplitude denominators for the EE-ADC route.

Same discipline as the IP/EA side (src/SingleReference/EpsteinNesbet): EN
dresses the AMPLITUDES, not the supermatrix. The ADC secular matrix keeps its
Moller-Plesset zeroth order -- its 2p2h diagonal is still the Fock difference,
because that diagonal defines the perturbation order the ADC(n) truncation is
built on -- while t2^(1), t1^(2) and t2^(2) are evaluated with the
determinant-diagonal denominator

    D^EN_ijab = eps_a + eps_b - eps_i - eps_j
                + <ij||ij> + <ab||ab>
                - <ia||ia> - <ib||ib> - <ja||ja> - <jb||jb>
              = <Phi_ij^ab| H |Phi_ij^ab> - E_HF

(Jiang & Engel Eq. 20), and correspondingly for the singles amplitude,
D^EN_ia = eps_i - eps_a + <ia||ia>.

DOUBLES ONLY: the singles amplitude t1^(2) keeps its Moller-Plesset
denominator. Dressing it as well would shift the ground-state density blocks
rho^(2) that enter M^(3) without any corresponding change in the secular
matrix, which is not what the EN partitioning is for.

`dress` is a dict of channel toggles matching
EpsteinNesbet.denominators.restricted_channel_shifts:

    {'hh': True, 'pp': True}       <- the standard choice, and what True means

hh switches on the <ij||ij> hole-hole term and pp the <ab||ab>
particle-particle term. The hole-particle cross channel 'hp' is available but
OFF by default: it is the channel that mixes the hole and particle
denominators and is not part of the usual EN doubles dressing.

Unlike the CSF route on the IP/EA side, nothing here has to average the two
spin cases: the spin-free backend carries a separate denominator per spin
block, so the dressing is the exact determinant-wise one in both routes --
which is also why the spin-orbital and spin-free EN amplitudes agree to
machine precision (tests/test_ee_adc_en.py).
"""
import numpy as np

from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB

CHANNELS = ('hh', 'pp', 'hp')


def validate_dress(dress):
    """Normalize and check the channel dict; None means no dressing."""
    if dress is None:
        return None
    if dress is True:
        return {'hh': True, 'pp': True, 'hp': False}
    bad = set(dress) - set(CHANNELS)
    if bad:
        raise ValueError(f'unknown EN channel(s) {sorted(bad)}; '
                         f'expected a subset of {CHANNELS}')
    out = {c: bool(dress.get(c, False)) for c in CHANNELS}
    if not any(out.values()):
        raise ValueError('en_dress was given with every channel off; pass '
                         'None for plain Moller-Plesset denominators')
    return out


# ----------------------------------------------------------------------
# spin-orbital route
# ----------------------------------------------------------------------

def en_denominators_spin_orbital(eps, g, nocc, dress):
    """(d_ijab, d_ia) as full spin-orbital arrays. The spin bookkeeping is
    already inside g, so <pq||pq> is just its diagonal."""
    dress = validate_dress(dress)
    eo, ev = eps[:nocc], eps[nocc:]
    d = (ev[None, None, :, None] + ev[None, None, None, :]
         - eo[:, None, None, None] - eo[None, :, None, None])
    d_ia = 2.0 * (eo[:, None] - ev[None, :])
    if dress is None:
        return d, d_ia
    a = np.einsum('pqpq->pq', g, optimize=True)          # <pq||pq>
    o, v = slice(0, nocc), slice(nocc, len(eps))
    if dress['hh']:
        d = d + a[o, o][:, :, None, None]
    if dress['pp']:
        d = d + a[v, v][None, None, :, :]
    if dress['hp']:
        ov = a[o, v]
        d = d - ov[:, None, :, None] - ov[None, :, None, :]
        d = d - ov[None, :, :, None] - ov[:, None, None, :]
    return d, d_ia          # singles denominator stays MP -- doubles only


# ----------------------------------------------------------------------
# spin-free route
# ----------------------------------------------------------------------

def en_denominators_spin_free(eps, J, K, no, nv, dress):
    """(d_ijab, d_ia) as SBs, one denominator per spin block, from the
    Coulomb/exchange arrays J[p,q] = <pq|pq> and K[p,q] = <pq|qp>.

    <pq||pq> = J[p,q] - delta(spin_p, spin_q) K[p,q] -- the single rule the
    whole spin resolution rests on."""
    dress = validate_dress(dress)
    eo, ev = eps[:no], eps[no:]
    base = (ev[None, None, :, None] + ev[None, None, None, :]
            - eo[:, None, None, None] - eo[None, :, None, None])
    d_ia_mp = 2.0 * (eo[:, None] - ev[None, :])
    keys4 = ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab')
    if dress is None:
        return (SB({k: base for k in keys4}),
                SB({'aa': d_ia_mp, 'bb': d_ia_mp}))

    o, v = slice(0, no), slice(no, no + nv)
    J_vv, K_vv = J[v, v], K[v, v]
    J_oo, K_oo = J[o, o], K[o, o]
    J_ov, K_ov = J[o, v], K[o, v]

    def block(si, sj, sa, sb):
        D = base.copy()
        if dress['pp']:
            D = D + (J_vv - (sa == sb) * K_vv)[None, None, :, :]
        if dress['hh']:
            D = D + (J_oo - (si == sj) * K_oo)[:, :, None, None]
        if dress['hp']:
            D = D - (J_ov - (si == sa) * K_ov)[:, None, :, None]
            D = D - (J_ov - (sj == sb) * K_ov)[None, :, None, :]
            D = D - (J_ov - (sj == sa) * K_ov)[None, :, :, None]
            D = D - (J_ov - (si == sb) * K_ov)[:, None, None, :]
        return D

    d4 = SB({k: block(*[c == 'a' for c in k]) for k in keys4})
    # singles denominator stays MP -- doubles only
    return d4, SB({'aa': d_ia_mp, 'bb': d_ia_mp})


def shift_from_denominators(d_en, d_mp):
    """Delta = d^EN - d^MP, the content the EN denominator has resummed.

    ee_equations.amplitudes needs it to take that content back out of the
    t2^(2) numerator; without it an EN-dressed ADC(3) double-counts the
    diagonal hole-hole and particle-particle ladder."""
    if isinstance(d_en, SB):
        # d_mp is a single array on the closed-shell route (one MP denominator
        # serves every spin block) but an SB on the unrestricted one, where
        # eps_alpha != eps_beta makes it block dependent
        if isinstance(d_mp, SB):
            return SB({k: v - d_mp.get(k) for k, v in d_en.items()
                       if d_mp.get(k) is not None})
        return SB({k: v - d_mp for k, v in d_en.items()})
    return d_en - d_mp


def jk_from_V(V):
    """J[p,q] = <pq|pq>, K[p,q] = <pq|qp> from the spatial physicist tensor."""
    return (np.einsum('pqpq->pq', V, optimize=True),
            np.einsum('pqqp->pq', V, optimize=True))


def jk_from_B(B):
    """The same two arrays from a DF factor, without forming V."""
    return (np.einsum('Qpp,Qqq->pq', B, B, optimize=True),
            np.einsum('Qpq,Qpq->pq', B, B, optimize=True))


def jk_from_B_uhf(B_a, B_b):
    """Spin-resolved Coulomb/exchange arrays from the two UHF DF factors:

        J[s][t][p,q] = <p_s q_t | p_s q_t> = (p_s p_s | q_t q_t)
        K[s][p,q]    = <p_s q_s | q_s p_s> = (p_s q_s | q_s p_s)

    K has one spin index because exchange only exists within a spin -- which
    is the whole content of the `delta(spin_p, spin_q)` in <pq||pq>."""
    B = {'a': B_a, 'b': B_b}
    diag = {s: np.einsum('Qpp->Qp', B[s]) for s in ('a', 'b')}
    J = {s: {t: np.einsum('Qp,Qq->pq', diag[s], diag[t], optimize=True)
             for t in ('a', 'b')} for s in ('a', 'b')}
    K = {s: np.einsum('Qpq,Qpq->pq', B[s], B[s], optimize=True)
         for s in ('a', 'b')}
    return J, K


def en_denominators_unrestricted(eps_a, eps_b, J, K, no_a, no_b, dress):
    """(d_ijab, d_ia) as SBs for an UNRESTRICTED reference.

    Same rule as en_denominators_spin_free -- <pq||pq> = J - delta(spin) K --
    with the spin pair made explicit instead of implied, because alpha and
    beta no longer share an orbital set (nor even a count). The singles
    denominator stays MP: the dressing is doubles only."""
    dress = validate_dress(dress)
    eps = {'a': np.asarray(eps_a), 'b': np.asarray(eps_b)}
    no = {'a': no_a, 'b': no_b}
    eo = {s: eps[s][:no[s]] for s in ('a', 'b')}
    ev = {s: eps[s][no[s]:] for s in ('a', 'b')}
    keys4 = ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab')

    def base(si, sj, sa, sb):
        return (ev[sa][None, None, :, None] + ev[sb][None, None, None, :]
                - eo[si][:, None, None, None] - eo[sj][None, :, None, None])

    d_ia = SB({s + s: 2.0 * (eo[s][:, None] - ev[s][None, :])
               for s in ('a', 'b')})
    if dress is None:
        return SB({k: base(*k) for k in keys4}), d_ia

    def part(s, t, f1, f2):
        s1 = slice(0, no[s]) if f1 == 'o' else slice(no[s], len(eps[s]))
        s2 = slice(0, no[t]) if f2 == 'o' else slice(no[t], len(eps[t]))
        out = J[s][t][s1, s2]
        return out - K[s][s1, s2] if s == t else out

    def block(si, sj, sa, sb):
        D = base(si, sj, sa, sb).copy()
        if dress['pp']:
            D = D + part(sa, sb, 'v', 'v')[None, None, :, :]
        if dress['hh']:
            D = D + part(si, sj, 'o', 'o')[:, :, None, None]
        if dress['hp']:
            D = D - part(si, sa, 'o', 'v')[:, None, :, None]
            D = D - part(sj, sb, 'o', 'v')[None, :, None, :]
            D = D - part(sj, sa, 'o', 'v')[None, :, :, None]
            D = D - part(si, sb, 'o', 'v')[:, None, None, :]
        return D

    return SB({k: block(*k) for k in keys4}), d_ia

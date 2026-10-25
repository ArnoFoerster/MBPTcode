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

`dress` is the en_dress dict validated by
EpsteinNesbet.validate_en_dress, restricted to the channel toggles:

    {'hh': True, 'pp': True}       <- the standard choice, and what True means

hh switches on the <ij||ij> hole-hole term and pp the <ab||ab>
particle-particle term. The hole-particle cross channel 'hp' is available but
OFF by default: it is the channel that mixes the hole and particle
denominators and is not part of the usual EN doubles dressing. The IP/EA keys
'singles', 'shift' and 'spin_adapted' are refused by name: the singles stay MP
and the dressing is determinant-wise.

The shifts themselves are the EpsteinNesbet ones (epstein_nesbet_denominator
on the spin-orbital route, dressed_doubles_block per spin block on the
spin-free and unrestricted routes). Unlike the CSF route on the IP/EA side,
nothing here has to average the two spin cases: the spin-free backend carries
a separate denominator per spin block, so the dressing is the exact
determinant-wise one in both routes -- which is also why the spin-orbital and
spin-free EN amplitudes agree to machine precision (tests/test_ee_adc_en.py).
"""
import numpy as np

from src.SingleReference.ADC.eeADC.ee_spin_blocks import SB
from src.SingleReference.ADC.eeADC.ee_utils import (excitation_denominator,
                                                   orbital_denominators)
from src.SingleReference.EpsteinNesbet import (dressed_doubles_block,
                                               epstein_nesbet_denominator,
                                               validate_en_dress)

#: the en_dress keys EE-ADC implements: the doubles channels only
DRESS_KEYS = ('hh', 'pp', 'hp')
KEYS4 = ('aaaa', 'bbbb', 'abab', 'baba', 'abba', 'baab')


# ----------------------------------------------------------------------
# spin-orbital route
# ----------------------------------------------------------------------

def en_denominators_spin_orbital(eps, g, nocc, dress):
    """(d_ijab, d_ia) as full spin-orbital arrays. The spin bookkeeping is
    already inside g, so <pq||pq> is just its diagonal."""
    dress = validate_en_dress(dress, keys=DRESS_KEYS)
    d, d_ia, _ = orbital_denominators(eps, nocc)
    if dress is None:
        return d, d_ia
    d = epstein_nesbet_denominator(d, g, nocc, layout='hhpp', bare_sign=-1.0,
                                   **{c: bool(dress.get(c)) for c in DRESS_KEYS})
    return d, d_ia          # singles denominator stays MP -- doubles only


# ----------------------------------------------------------------------
# spin-free route
# ----------------------------------------------------------------------

def en_denominators_spin_free(eps, J, K, no, nv, dress):
    """(d_ijab, d_ia) as SBs, one denominator per spin block, from the
    Coulomb/exchange arrays J[p,q] = <pq|pq> and K[p,q] = <pq|qp>
    (pyscf_interface.get_coulomb_exchange_diagonals[_df]); both spins share
    them."""
    dress = validate_en_dress(dress, keys=DRESS_KEYS)
    base, d_ia_mp, _ = orbital_denominators(eps, no)
    if dress is None:
        return (SB({k: base for k in KEYS4}),
                SB({'aa': d_ia_mp, 'bb': d_ia_mp}))
    Js = {s: {t: J for t in 'ab'} for s in 'ab'}
    Ks = {s: K for s in 'ab'}
    d4 = SB({k: dressed_doubles_block(base, Js, Ks, {'a': no, 'b': no}, k, dress)
             for k in KEYS4})
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


def en_denominators_unrestricted(eps_a, eps_b, J, K, no_a, no_b, dress):
    """(d_ijab, d_ia) as SBs for an UNRESTRICTED reference.

    Same rule as en_denominators_spin_free -- <pq||pq> = J - delta(spin) K --
    with the spin pair made explicit instead of implied, because alpha and
    beta no longer share an orbital set (nor even a count). J, K are
    EpsteinNesbet.coulomb_exchange_diagonals_uhf_df. The singles denominator
    stays MP: the dressing is doubles only."""
    dress = validate_en_dress(dress, keys=DRESS_KEYS)
    eps = {'a': np.asarray(eps_a), 'b': np.asarray(eps_b)}
    no = {'a': no_a, 'b': no_b}
    eo = {s: eps[s][:no[s]] for s in ('a', 'b')}
    ev = {s: eps[s][no[s]:] for s in ('a', 'b')}

    def base(si, sj, sa, sb):
        return excitation_denominator((eo[si], eo[sj]), (ev[sa], ev[sb]))

    d_ia = SB({s + s: -2.0 * excitation_denominator((eo[s],), (ev[s],))
               for s in ('a', 'b')})
    if dress is None:
        return SB({k: base(*k) for k in KEYS4}), d_ia
    return SB({k: dressed_doubles_block(base(*k), J, K, no, k, dress)
               for k in KEYS4}), d_ia

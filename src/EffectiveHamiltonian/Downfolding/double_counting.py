"""The double counting of the downfolded Hamiltonian: everything `dc` decides.

v_eff = v + vtilde, so the active electrons of a solver feel the mean field
of vtilde too, which the canonical reference does not contain. The one-body
term subtracts it, t^DC = hartree - exchange (`double_counting_potential`),
and E_0 adds its energy back, E_HF,ee[vtilde] (`double_counting_energy`), so
that the reference determinant of H_eff keeps E_HF + E_c[v] - E_c^A.

With a static self-energy Sigma_bar, which carries the exchange, the
convention is dc='hartree': h_eff subtracts the Hartree part only, E_0 gains
the exchange energy term (`exchange_energy_term`) for the exchange half that
stays, and subtracts Tr[Sigma_bar gamma^A,0] (`sigma_trace`) for the
Sigma_bar added to h_eff. `double_counting` makes all of it from a method's
`DownfoldingTerms`, as the `DoubleCounting` record. Every quantity is per
spin channel (spin.py).
"""
from dataclasses import dataclass

import numpy as np

from .options import choice
from .spin import spin_weight

DC_MODES = ('full', 'hartree')


@dataclass(frozen=True)
class DoubleCounting:
    """The double counting of one Hamiltonian.

    mode: 'full' or 'hartree'. hartree, exchange: the two parts of t^DC, one
    (n_act, n_act) array per spin channel; h_eff subtracts hartree and, with
    'full', adds exchange back (`effective_one_body`), kept apart so that
    h_eff is formed in that order. e_hartree_dc, e_exchange_dc: the two
    halves of E_HF,ee[vtilde]. e_exchange_term, e_sigma_trace: the two E_0
    terms of 'hartree', zero with 'full' (the trace also without a
    Sigma_bar).
    """
    mode: str
    hartree: tuple
    exchange: tuple
    e_hartree_dc: float
    e_exchange_dc: float
    e_exchange_term: float
    e_sigma_trace: float


def double_counting(terms, n_act_occ):
    """The `DoubleCounting` of a method's `DownfoldingTerms`, n_act_occ the
    active occupied count of each spin channel, (n,) or (n_a, n_b).

    A Sigma_bar with dc='full' is refused: Sigma_bar carries the exchange,
    so it comes with dc='hartree', which also sets the double counting and
    E_0 to match.
    """
    mode = choice(terms.dc, DC_MODES, 'dc')
    if terms.sigma_bar is not None and mode == 'full':
        raise ValueError(
            "the low-level method returned a static self-energy with "
            "dc='full'; Sigma_bar carries the exchange, so it comes with "
            "dc='hartree', which also sets the double counting and E_0 to "
            "match.")
    hartree, exchange = double_counting_potential(terms.vtilde, n_act_occ)
    e_hartree_dc, e_exchange_dc = double_counting_energy(terms.vtilde,
                                                         n_act_occ)
    e_exchange_term = e_sigma_trace = 0.0
    if mode == 'hartree':
        e_exchange_term = exchange_energy_term(terms.vtilde, n_act_occ)
        if terms.sigma_bar is not None:
            e_sigma_trace = sigma_trace(terms.sigma_bar, n_act_occ)
    return DoubleCounting(mode, hartree, exchange, e_hartree_dc,
                          e_exchange_dc, e_exchange_term, e_sigma_trace)


def double_counting_potential(vtilde, n_act_occ):
    """(hartree, exchange), each one (n_act, n_act) array per spin channel:
    the Hartree and exchange parts of the double counting of vtilde with the
    active reference density.

    hartree^s_tu  = sum_{k in A_o^s} vt^{ss}_tu,kk + sum_{k in A_o^s'} vt^{ss'}_tu,kk
    exchange^s_tu = sum_{k in A_o^s} vt^{ss}_tk,ku
    so t^DC = hartree - exchange. Two channels: vtilde = (aa, ab, bb) with
    ab[t,u,v,w] = (t_a u_a | v_b w_b), the beta channel reading the
    opposite-spin block as ab.transpose(2, 3, 0, 1), and n_act_occ =
    (n_a, n_b). One restricted channel: vtilde = (v,), n_act_occ = (n,), and
    the opposite spin sees the same block, hartree_tu = 2 sum_k vt_tu,kk.
    """
    if spin_weight(n_act_occ) == 2:
        (v,), (n,) = vtilde, n_act_occ
        return ((2 * np.einsum('tukk->tu', v[:, :, :n, :n]),),
                (np.einsum('tkku->tu', v[:, :n, :n, :]),))
    aa, ab, bb = vtilde
    n_a, n_b = n_act_occ
    channels = ((aa, ab, n_a, n_b), (bb, ab.transpose(2, 3, 0, 1), n_b, n_a))
    hartree, exchange = [], []
    for same, other, n_own, n_other in channels:
        hartree.append(np.einsum('tukk->tu', same[:, :, :n_own, :n_own])
                       + np.einsum('tukk->tu', other[:, :, :n_other, :n_other]))
        exchange.append(np.einsum('tkku->tu', same[:, :n_own, :n_own, :]))
    return tuple(hartree), tuple(exchange)


def double_counting_energy(vtilde, n_act_occ):
    """(hartree, exchange): the two parts of E_HF,ee[vtilde], the HF energy of
    vtilde in the active reference determinant.

    hartree  = 1/2 sum_{tu} vt_tt,uu rho_tt rho_uu over the occupied active
               spin orbitals, both spin blocks and the opposite-spin pair twice;
    exchange = -1/2 sum_{tu} vt_tu,ut rho_tt rho_uu, same spin only (negative
               for a positive-definite vtilde).
    One restricted channel, vtilde = (v,) and n_act_occ = (n,):
    hartree = 2 sum_kl vt_kk,ll, exchange = -sum_kl vt_kl,lk. Two channels:
    vtilde = (aa, ab, bb), n_act_occ = (n_a, n_b).
    """
    if spin_weight(n_act_occ) == 2:
        (v,), (n,) = vtilde, n_act_occ
        occ = v[:n, :n, :n, :n]
        return (float(2 * np.einsum('kkll->', occ)),
                float(-np.einsum('kllk->', occ)))
    aa, ab, bb = vtilde
    n_a, n_b = n_act_occ
    hartree = 0.5 * (np.einsum('kkll->', aa[:n_a, :n_a, :n_a, :n_a])
                     + np.einsum('kkll->', bb[:n_b, :n_b, :n_b, :n_b])
                     + 2 * np.einsum('kkll->', ab[:n_a, :n_a, :n_b, :n_b]))
    exchange = -0.5 * (np.einsum('kllk->', aa[:n_a, :n_a, :n_a, :n_a])
                       + np.einsum('kllk->', bb[:n_b, :n_b, :n_b, :n_b]))
    return float(hartree), float(exchange)


def exchange_energy_term(vtilde, n_act_occ):
    """The exchange energy term that comes with Sigma^E, sum_{t,v} vt_tv,vt
    over the occupied active spin orbitals (restricted: 2 K~ with
    K~ = sum_{t,v in A_o} vt_tv,vt), i.e. -2 times the exchange part of
    `double_counting_energy`.

    With Sigma^E the one-body term keeps only the Hartree double counting; this
    term gives E_0 back what the dropped exchange half took from the reference
    energy.
    """
    return -2.0 * double_counting_energy(vtilde, n_act_occ)[1]


def sigma_trace(sigma_bar, n_act_occ):
    """Tr[Sigma_bar gamma^A,0] = sum_s sum_{k in A_o^s} Sigma_bar^s_kk
    (restricted: 2 sum_k Sigma_bar_kk), the energy the static self-energy adds
    to the reference determinant; E_0 subtracts it. sigma_bar holds one array
    per spin channel, (s,) restricted or (a, b)."""
    return float(spin_weight(sigma_bar)
                 * sum(np.trace(s[:n, :n]) for s, n in zip(sigma_bar, n_act_occ)))

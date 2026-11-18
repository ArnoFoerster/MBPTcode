"""The constant E_0 of the downfolded Hamiltonian, term by term.

E_0 = E_nuc + E_HF,core + E_HF,ee + E_HF,ee[vtilde] + E_c[v] - E_c^A
(+ the exchange energy term - Tr[Sigma_bar gamma^A,0] with Sigma^E), spec §3.7.
The HF terms of the environment are those of the inactive core alone (the
active-environment cross terms are in the core Fock operator), and
E_HF,ee[vtilde] enters with a plus sign so that the reference determinant of
H_eff keeps the energy E_HF + E_c[v] - E_c^A. Every quantity is per spin
channel, as in one_body.py: channel tuples (alpha, beta), vtilde = (aa, ab,
bb) with ab[t,u,v,w] = (t_a u_a | v_b w_b), active indices occupied first.
"""
from dataclasses import dataclass, field

import numpy as np

from .one_body import DC_MODES, inactive_jk
from .terms import choice


def environment_hf_energy(mol, hcore_ao, mo, parts):
    """(e_core, e_ee): the one- and two-electron HF energy of the inactive core.

    e_core = sum_s Tr[h D_I^s], e_ee = 1/2 sum_s Tr[(J[D_I] - K[D_I^s]) D_I^s]
    with D_I = D_I^a + D_I^b, on the exact J and K of `inactive_jk`; with every
    occupied orbital inactive, E_nuc + e_core + e_ee is the HF energy.
    """
    dms, j, vk = inactive_jk(mol, mo, parts)
    e_core = sum(np.einsum('ij,ji->', hcore_ao, d) for d in dms)
    e_ee = 0.5 * sum(np.einsum('ij,ji->', j - k, d) for d, k in zip(dms, vk))
    return float(e_core), float(e_ee)


def active_hf_energy(vtilde, n_act_occ):
    """(hartree, exchange): the two parts of E_HF,ee[vtilde], the HF energy of
    vtilde in the active reference determinant.

    hartree  = 1/2 sum_{tu} vt_tt,uu rho_tt rho_uu over the occupied active
               spin orbitals, both spin blocks and the opposite-spin pair twice;
    exchange = -1/2 sum_{tu} vt_tu,ut rho_tt rho_uu, same spin only (negative
               for a positive-definite vtilde).
    Restricted: hartree = 2 sum_kl vt_kk,ll, exchange = -sum_kl vt_kl,lk.
    n_act_occ = (n_a, n_b).
    """
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
    `active_hf_energy`.

    With Sigma^E the one-body term keeps only the Hartree double counting; this
    term gives E_0 back what the dropped exchange half took from the reference
    energy.
    """
    return -2.0 * active_hf_energy(vtilde, n_act_occ)[1]


def sigma_trace(sigma_bar, n_act_occ):
    """Tr[Sigma_bar gamma^A,0] = sum_s sum_{k in A_o^s} Sigma_bar^s_kk
    (restricted: 2 sum_k Sigma_bar_kk), the energy the static self-energy adds
    to the reference determinant; E_0 subtracts it. sigma_bar = (a, b)."""
    return float(sum(np.trace(s[:n, :n]) for s, n in zip(sigma_bar, n_act_occ)))


@dataclass(frozen=True)
class Energies:
    """Every term of E_0 separately (the paper's Fig. 4 breakdown).

    e_nuc: the nuclear repulsion.
    e_hf_core, e_hf_ee: the one- and two-electron HF energy of the inactive core.
    e_hf_ee_vtilde: E_HF,ee[vtilde], the HF energy of vtilde in the active
        reference determinant.
    e_c_full, e_c_active: the correlation energy of the full space and of the
        active space alone.
    e_exchange_term, e_sigma_trace: the exchange energy term and
        Tr[Sigma_bar gamma^A,0]; nonzero only with dc='hartree'.
    diagnostics: the low-level method's diagnostics, keyed by name.
    """
    e_nuc: float
    e_hf_core: float
    e_hf_ee: float
    e_hf_ee_vtilde: float
    e_c_full: float
    e_c_active: float
    e_exchange_term: float
    e_sigma_trace: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def e0(self):
        """E_0, the constant of H_eff (g_eff[0])."""
        return (self.e_nuc + self.e_hf_core + self.e_hf_ee
                + self.e_hf_ee_vtilde + self.e_c_full - self.e_c_active
                + self.e_exchange_term - self.e_sigma_trace)


def assemble_energies(e_nuc, e_core, e_ee, terms, n_act_occ):
    """The `Energies` of a `DownfoldingTerms`, given the environment HF terms.

    The exchange energy term and Tr[Sigma_bar gamma^A,0] belong to the
    Hartree-only double counting (dc='hartree') and are zero with dc='full';
    the trace is zero too when no sigma_bar is given. n_act_occ = (n_a, n_b).
    """
    dc = choice(terms.dc, DC_MODES, 'dc')
    hartree, exchange = active_hf_energy(terms.vtilde, n_act_occ)
    e_exchange_term = e_sigma_trace = 0.0
    if dc == 'hartree':
        e_exchange_term = exchange_energy_term(terms.vtilde, n_act_occ)
        if terms.sigma_bar is not None:
            e_sigma_trace = sigma_trace(terms.sigma_bar, n_act_occ)
    return Energies(e_nuc=e_nuc, e_hf_core=e_core, e_hf_ee=e_ee,
                    e_hf_ee_vtilde=hartree + exchange,
                    e_c_full=terms.e_c_full, e_c_active=terms.e_c_active,
                    e_exchange_term=e_exchange_term,
                    e_sigma_trace=e_sigma_trace,
                    diagnostics=dict(terms.diagnostics))

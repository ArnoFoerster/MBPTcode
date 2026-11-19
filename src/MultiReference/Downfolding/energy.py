"""The constant E_0 of the downfolded Hamiltonian, term by term.

E_0 = E_nuc + E_core + E_Hartree + E_exchange        (the environment)
      + E_Hartree,dc + E_exchange,dc                  (E_HF,ee[vtilde])
      + E_c[v] - E_c^A
      (+ the exchange energy term - Tr[Sigma_bar gamma^A,0] with Sigma^E).
The `Energies` record keeps every one of these on its own; `eh`, `ex` and
`ec` are the Hartree, exchange and correlation sums of each line pair.
The HF terms of the environment are those of the inactive core alone (the
active-environment cross terms are in the core Fock operator), and
E_HF,ee[vtilde] enters with a plus sign so that the reference determinant of
H_eff keeps the energy E_HF + E_c[v] - E_c^A. Every quantity is per spin
channel, as in one_body.py: one channel for a restricted reference, with
vtilde = (v,), or the two (alpha, beta), with vtilde = (aa, ab, bb) and
ab[t,u,v,w] = (t_a u_a | v_b w_b); active indices occupied first.
"""
from dataclasses import dataclass, field

import numpy as np

from .one_body import DC_MODES, inactive_jk, spin_weight
from .terms import choice


def environment_hf_energy(mol, hcore_ao, mo, parts, jk=None):
    """(e_core, e_hartree, e_exchange): the one-electron, Hartree and exchange
    energies of the inactive core.

    e_core     = sum_s Tr[h D_I^s]
    e_hartree  = 1/2 sum_s Tr[J[D_I] D_I^s]
    e_exchange = -1/2 sum_s Tr[K[D_I^s] D_I^s]
    with D_I the whole core's density, the sums over both spins (twice the
    one restricted channel), on the exact J and K of `inactive_jk`, built
    here unless handed in as `jk`. Restricted: e_core = 2 sum_i h_ii,
    e_hartree = 2 sum_ij (ii|jj), e_exchange = -sum_ij (ij|ji). With every
    occupied orbital inactive, E_nuc plus the three is the HF energy.
    """
    weight = spin_weight(mo)
    dms, j, vk = inactive_jk(mol, mo, parts) if jk is None else jk
    e_core = weight * sum(np.einsum('ij,ji->', hcore_ao, d) for d in dms)
    e_hartree = 0.5 * weight * sum(np.einsum('ij,ji->', j, d) for d in dms)
    e_exchange = -0.5 * weight * sum(np.einsum('ij,ji->', k, d)
                                     for d, k in zip(dms, vk))
    return float(e_core), float(e_hartree), float(e_exchange)


def active_hf_energy(vtilde, n_act_occ):
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
    `active_hf_energy`.

    With Sigma^E the one-body term keeps only the Hartree double counting; this
    term gives E_0 back what the dropped exchange half took from the reference
    energy.
    """
    return -2.0 * active_hf_energy(vtilde, n_act_occ)[1]


def sigma_trace(sigma_bar, n_act_occ):
    """Tr[Sigma_bar gamma^A,0] = sum_s sum_{k in A_o^s} Sigma_bar^s_kk
    (restricted: 2 sum_k Sigma_bar_kk), the energy the static self-energy adds
    to the reference determinant; E_0 subtracts it. sigma_bar holds one array
    per spin channel, (s,) restricted or (a, b)."""
    return float(spin_weight(sigma_bar)
                 * sum(np.trace(s[:n, :n]) for s, n in zip(sigma_bar, n_act_occ)))


@dataclass(frozen=True)
class Energies:
    """Every term of E_0 on its own (the paper's Fig. 4 breakdown).

    Stored, each with the sign it has before entering E_0:
    e_nuc: the nuclear repulsion.
    e_core: the one-electron energy of the inactive core.
    e_hartree, e_exchange: the Hartree and exchange energies of the inactive
        core (exchange is negative).
    e_hartree_dc, e_exchange_dc: the Hartree and exchange halves of
        E_HF,ee[vtilde], the HF energy of the screening correction in the
        active reference determinant: the double-counting correction of the
        mean-field energy. Both zero when vtilde has no occupied-occupied
        element ('bare', 'crpa_ph', 'mrpa', method 'hf').
    e_c_full: the correlation energy of the full space, E_c[v].
    e_c_active: the correlation energy of the active space alone,
        E_c^A[v_eff], the double-counting correction of the correlation
        energy; E_0 subtracts it.
    e_exchange_term: the exchange energy term that comes with Sigma^E,
        -2 e_exchange_dc; nonzero only with dc='hartree'.
    e_sigma_trace: Tr[Sigma_bar gamma^A,0]; E_0 subtracts it; nonzero only
        with a Sigma_bar.
    diagnostics: the low-level method's diagnostics, keyed by name.

    Sums (properties): `eh`, `ex`, `ec` are the Hartree, exchange and
    correlation energies of E_0 with their corrections, so that
    e0 = e_nuc + e_core + eh + ex + ec - e_sigma_trace. `e_hf_core`,
    `e_hf_ee` and `e_hf_ee_vtilde` are the earlier names of e_core and of the
    two Hartree-plus-exchange sums.
    """
    e_nuc: float
    e_core: float
    e_hartree: float
    e_exchange: float
    e_hartree_dc: float
    e_exchange_dc: float
    e_c_full: float
    e_c_active: float
    e_exchange_term: float
    e_sigma_trace: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def eh(self):
        """The Hartree energy in E_0: environment plus double counting."""
        return self.e_hartree + self.e_hartree_dc

    @property
    def ex(self):
        """The exchange energy in E_0: environment, double counting and,
        with Sigma^E, the exchange energy term."""
        return self.e_exchange + self.e_exchange_dc + self.e_exchange_term

    @property
    def ec(self):
        """The correlation energy in E_0, E_c[v] - E_c^A[v_eff]."""
        return self.e_c_full - self.e_c_active

    @property
    def e_hf_core(self):
        """e_core under its earlier name."""
        return self.e_core

    @property
    def e_hf_ee(self):
        """The two-electron HF energy of the inactive core,
        e_hartree + e_exchange."""
        return self.e_hartree + self.e_exchange

    @property
    def e_hf_ee_vtilde(self):
        """E_HF,ee[vtilde], e_hartree_dc + e_exchange_dc."""
        return self.e_hartree_dc + self.e_exchange_dc

    @property
    def e0(self):
        """E_0, the constant of H_eff (g_eff[0])."""
        return (self.e_nuc + self.e_core + self.eh + self.ex + self.ec
                - self.e_sigma_trace)


def assemble_energies(e_nuc, e_core, e_hartree, e_exchange, terms, n_act_occ):
    """The `Energies` of a `DownfoldingTerms`, given the nuclear repulsion and
    the three environment terms of `environment_hf_energy`.

    The exchange energy term and Tr[Sigma_bar gamma^A,0] belong to the
    Hartree-only double counting (dc='hartree') and are zero with dc='full';
    the trace is zero too when no sigma_bar is given. n_act_occ holds the
    active occupied count of each spin channel, (n,) or (n_a, n_b).
    """
    dc = choice(terms.dc, DC_MODES, 'dc')
    hartree, exchange = active_hf_energy(terms.vtilde, n_act_occ)
    e_exchange_term = e_sigma_trace = 0.0
    if dc == 'hartree':
        e_exchange_term = exchange_energy_term(terms.vtilde, n_act_occ)
        if terms.sigma_bar is not None:
            e_sigma_trace = sigma_trace(terms.sigma_bar, n_act_occ)
    return Energies(e_nuc=e_nuc, e_core=e_core, e_hartree=e_hartree,
                    e_exchange=e_exchange, e_hartree_dc=hartree,
                    e_exchange_dc=exchange,
                    e_c_full=terms.e_c_full, e_c_active=terms.e_c_active,
                    e_exchange_term=e_exchange_term,
                    e_sigma_trace=e_sigma_trace,
                    diagnostics=dict(terms.diagnostics))

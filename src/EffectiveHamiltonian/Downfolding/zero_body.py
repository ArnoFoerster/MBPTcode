"""The constant E_0 of the downfolded Hamiltonian, term by term.

E_0 = E_nuc + E_core + E_Hartree + E_exchange        (the environment)
      + E_Hartree,dc + E_exchange,dc                  (E_HF,ee[vtilde])
      + E_c[v] - E_c^A
      (+ the exchange energy term - Tr[Sigma_bar gamma^A,0] with Sigma^E).
The `Energies` record keeps every one of these on its own; `eh`, `ex` and
`ec` are the Hartree, exchange and correlation sums of each line pair.
The HF terms of the environment are those of the inactive core alone (the
active-environment cross terms are in the core Fock operator), from core.py;
the double-counting terms come from double_counting.py, which says why
E_HF,ee[vtilde] enters with a plus sign. This file only assembles them.
"""
from dataclasses import dataclass, field


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
    numerical_flags: the named flags behind this Hamiltonian
        (src.Base.numerical_flags), each {'flag': name, 'message': text};
        filled by `build_effective_hamiltonian`, empty for a clean result.

    Sums (properties): `eh`, `ex`, `ec` are the Hartree, exchange and
    correlation energies of E_0 with their corrections, so that
    e0 = e_nuc + e_core + eh + ex + ec - e_sigma_trace.
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
    numerical_flags: list = field(default_factory=list)

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
    def e0(self):
        """E_0, the constant of H_eff (g_eff[0])."""
        return (self.e_nuc + self.e_core + self.eh + self.ex + self.ec
                - self.e_sigma_trace)


def assemble_energies(e_nuc, core, dc, terms):
    """The `Energies` of one Hamiltonian: the nuclear repulsion, the frozen
    core's three energies (`Core`), the four energies of the double counting
    (`DoubleCounting`) and the method's correlation-energy pair and
    diagnostics (`DownfoldingTerms`)."""
    return Energies(e_nuc=e_nuc, e_core=core.e_core,
                    e_hartree=core.e_hartree, e_exchange=core.e_exchange,
                    e_hartree_dc=dc.e_hartree_dc,
                    e_exchange_dc=dc.e_exchange_dc,
                    e_c_full=terms.e_c_full, e_c_active=terms.e_c_active,
                    e_exchange_term=dc.e_exchange_term,
                    e_sigma_trace=dc.e_sigma_trace,
                    diagnostics=dict(terms.diagnostics))

"""Excitation-transfer rates between molecules in the golden rule: triplet
(Dexter) energy transfer T1(D) S0(A) -> S0(D) T1(A), and the spin channels of
triplet-triplet annihilation T1 T1 -> S1 S0.

    k = (2 pi / hbar) |V|^2 rho_FC(dE)          (`rates.golden_rule`)

V is an argument: the effective coupling of the two diabats
(`fragments.transfer.dexter_coupling` for triplet transfer,
`fragments.tta_coupling` for annihilation). Each molecule of the process
changes its own electronic state (`Transition`: its adiabatic energy change
and the Huang-Rhys spectrum between its two states), and the molecules' modes
are independent, so the generating function of the whole process is the
product of theirs: rho_FC is the Franck-Condon weighted density of states of
the joined spectrum (`franck_condon.fc_weighted_dos`, every mode at its own
frequency and Bose factor), at dE = the sum of the molecules' energy changes,
E_final - E_initial, negative downhill. `lambda_env` is the classical
reorganization of what the mode lists do not hold (solvent, protein).

THE DRIVING FORCE is adiabatic: each molecule's energy change is E_final at
its own minimum less E_initial at its own (`vibronic.adiabatic_gap` of two
`relax_state` records), with the ground state's own minimum for S0. The
Huang-Rhys spectrum is the linear coupling model's at the ground-state minimum
(`transition_from_gradients`): the two states' gradients there, on the ground
state's mass-weighted normal modes, S_k = (g_f,k - g_i,k)^2 / (2 omega_k^3).
Adiabatic energies with vertical-gradient displacements are the usual
compromise of the transfer literature; the reorganization the spectrum
implies, sum S_k omega_k, is in the record next to the energies, so the two
can be compared.

SPIN STATISTICS. Two triplets meet in nine spin states: one singlet pair
(total spin 0), three triplet and five quintet ones. A channel open to pairs
of total spin S proceeds from (2S + 1) / 9 of the encounters
(`TTA_SPIN_WEIGHTS`), so the rate of T1 + T1 -> S1 + S0 from an encounter
complex is 1/9 of the golden-rule rate of the singlet pair (`spin_weight`
of `transfer_rate`). Diffusion and encounter kinetics in a solution or a
film are outside this module.

Every energy in and out is in Hartree, temperatures in Kelvin, rates in s^-1.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.numerics.constants import BOLTZMANN_HARTREE_PER_KELVIN
from src.photophysics.rates import golden_rule, marcus_rate
from src.properties.franck_condon import fc_weighted_dos
from src.properties.vibronic import huang_rhys_from_gradient

#: Fraction of the nine spin states of a triplet pair with total spin S:
#: (2S + 1) / 9.
TTA_SPIN_WEIGHTS = {'singlet': 1.0 / 9.0, 'triplet': 3.0 / 9.0,
                    'quintet': 5.0 / 9.0}


@dataclass(frozen=True)
class Transition:
    """One molecule's part of a transfer: its state changes from `initial`
    to `final` (labels, e.g. 'T1' -> 'S0'). delta_e: E_final(R_final) -
    E_initial(R_initial), Hartree; s_k, omega_k: the Huang-Rhys factors
    between the two states' minima and the frequencies of the modes they
    are measured on (Hartree, all positive); method: what produced them."""
    name: str
    initial: str
    final: str
    delta_e: float
    s_k: np.ndarray
    omega_k: np.ndarray
    method: str = None

    def __post_init__(self):
        s, w = np.atleast_1d(self.s_k), np.atleast_1d(self.omega_k)
        if s.shape != w.shape:
            raise ValueError(f'{self.name}: {s.size} Huang-Rhys factors for '
                             f'{w.size} modes')
        if np.any(w <= 0.0) or np.any(s < 0.0):
            raise ValueError(f'{self.name}: the modes need omega > 0 and '
                             f'S >= 0 (drop the rigid-body and imaginary ones)')

    @property
    def reorganization(self):
        """sum_k S_k omega_k, Hartree."""
        return float(np.dot(self.s_k, self.omega_k))

    def reverse(self):
        """The same molecule going back: -delta_e, the same spectrum."""
        return Transition(self.name, self.final, self.initial, -self.delta_e,
                          self.s_k, self.omega_k, self.method)


def transition_from_gradients(name, initial, final, delta_e, omega, modes,
                              masses, grad_initial=None, grad_final=None,
                              method=None):
    """The `Transition` of the linear coupling model: the gradients (natm, 3)
    of the initial and the final state at the ground-state minimum where
    `omega`, `modes`, `masses` were taken (`vibronic.normal_modes`); None is
    the ground state there (zero gradient). S_k = d_k^2 / (2 omega_k^3) with
    d_k the projected gradient difference (`vibronic.huang_rhys_from_gradient`,
    which refuses a mode that is not a vibration)."""
    if grad_initial is None and grad_final is None:
        raise ValueError(f'{name}: no gradient of either state given')
    natm = len(masses)
    g_i = np.zeros((natm, 3)) if grad_initial is None else grad_initial
    g_f = np.zeros((natm, 3)) if grad_final is None else grad_final
    omega = np.asarray(omega, float)
    s_k, _ = huang_rhys_from_gradient(np.asarray(g_f) - np.asarray(g_i), omega,
                                      modes, masses)
    return Transition(name, initial, final, float(delta_e), s_k, omega, method)


def transfer_rate(coupling, transitions, temperature, lambda_env=0.0,
                  broadening=0.0, spin_weight=1.0):
    """The golden-rule rate of the process in which every molecule makes
    its `Transition`, with coupling V (Hartree). Returns a dict:

    k: spin_weight * (2 pi / hbar) |V|^2 rho_FC(dE), s^-1;
    k_reverse: the reverse process's (every transition reversed) with the
        same weight; k / k_reverse = exp(-dE / k_B T), detailed balance;
    k_marcus: the classical limit (every mode classical, lambda the joined
        spectrum's plus lambda_env), for comparison;
    delta_e, rho_fc, lambda_total and each molecule's lambda (Hartree).
    """
    transitions = list(transitions)
    if not transitions:
        raise ValueError('a transfer needs at least one molecule')
    s_k = np.concatenate([np.atleast_1d(t.s_k) for t in transitions])
    omega_k = np.concatenate([np.atleast_1d(t.omega_k) for t in transitions])
    delta_e = float(sum(t.delta_e for t in transitions))
    rho = fc_weighted_dos(np.array([delta_e, -delta_e]), s_k, omega_k,
                          temperature, lambda_classical=lambda_env,
                          broadening=broadening)
    lam = float(np.dot(s_k, omega_k)) + float(lambda_env)
    weight = float(spin_weight)
    return dict(
        k=weight * golden_rule(coupling, rho[0]),
        k_reverse=weight * golden_rule(coupling, rho[1]),
        k_marcus=weight * marcus_rate(coupling, delta_e, lam, temperature),
        delta_e=delta_e, rho_fc=float(rho[0]), lambda_total=lam,
        lambda_by_molecule={t.name: t.reorganization for t in transitions},
        coupling=float(abs(coupling)), spin_weight=weight,
        temperature=float(temperature),
        detailed_balance=float(np.exp(-delta_e / (
            BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)))))


def dexter_rate(coupling, donor, acceptor, temperature, lambda_env=0.0,
                broadening=0.0):
    """Triplet energy transfer T1(D) S0(A) -> S0(D) T1(A): `transfer_rate`
    with the donor's T1 -> S0 and the acceptor's S0 -> T1 `Transition`s."""
    for t, want in ((donor, ('T1', 'S0')), (acceptor, ('S0', 'T1'))):
        if (t.initial, t.final) != want:
            raise ValueError(f'{t.name}: {t.initial} -> {t.final}; a triplet '
                             f'transfer takes the donor T1 -> S0 and the '
                             f'acceptor S0 -> T1')
    return transfer_rate(coupling, (donor, acceptor), temperature,
                         lambda_env=lambda_env, broadening=broadening)
